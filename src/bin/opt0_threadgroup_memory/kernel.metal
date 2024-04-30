// Max threadgroup memory (32KB) / Bytes per entry (6 * 4B) = 1365.33
#define L_HASHMAP_LEN 1365
#define L_BUCKET_LEN 6
#define G_BUCKET_LEN 5
#define MIN_FIELD 1
#define MAX_FIELD 2
#define SUM_FIELD 3
#define COUNT_FIELD 4
// Caches the global hashmap bucket index a name would occupy in the local
// hashmap to avoid rehashing (requiring scanning the string again) when
// inserting to the global one.
#define G_BUCKET_FIELD 5
// No names start at index 1 so treat it as an empty value
#define EMPTY_BUCKET_KEY 1

#include <metal_stdlib>

constant uint G_HASHMAP_LEN [[ function_constant(0) ]];
// Whether to reinterpret atomics as non-atomics when race condition
// is guaranteed not to occur.
constant bool REINTERPRET_ATOMICS [[ function_constant(1) ]];

using namespace metal;

// Compare string until `;`
bool name_eq(
    const device uchar* buf,
    uint a_idx,
    uint b_idx
) {
    const device uchar* a_ptr = &buf[a_idx];
    const device uchar* b_ptr = &buf[b_idx];
    uchar a, b;
    while (true) {
        a = *a_ptr++;
        b = *b_ptr++;
        if (a == b) {
            if (a == ';') return true;
        } else {
            return false;
        }
    }
}

void init_local_hashmap(threadgroup atomic_int* buckets, uint lid, uint threadgroup_size) {
    for (uint i = lid; i < L_HASHMAP_LEN; i += threadgroup_size) {
        uint offset = i * L_BUCKET_LEN;

        if (REINTERPRET_ATOMICS) {
            // Each thread initialize a disjoint set of slots so no need for atomic stores.
            threadgroup int* buckets_ = reinterpret_cast<threadgroup int*>(buckets);

            // Global bucket field is overwrite only so no need to initialize
            buckets_[offset] = EMPTY_BUCKET_KEY;
            buckets_[offset + MIN_FIELD] = INT_MAX;
            buckets_[offset + MAX_FIELD] = INT_MIN;
            buckets_[offset + SUM_FIELD] = 0;
            buckets_[offset + COUNT_FIELD] = 0;
        } else {
            atomic_store_explicit(&buckets[offset], EMPTY_BUCKET_KEY, memory_order_relaxed);
            atomic_store_explicit(&buckets[offset + MIN_FIELD], INT_MAX, memory_order_relaxed);
            atomic_store_explicit(&buckets[offset + MAX_FIELD], INT_MIN, memory_order_relaxed);
            atomic_store_explicit(&buckets[offset + SUM_FIELD], 0, memory_order_relaxed);
            atomic_store_explicit(&buckets[offset + COUNT_FIELD], 0, memory_order_relaxed);
        }
    }
}

void update_local_hashmap(
    const device uchar* buf,
    threadgroup atomic_int* buckets,
    uint name_idx,
    uint64_t hash,
    int temp
) {
    uint bucket_idx = hash % L_HASHMAP_LEN;
    uint bucket_offset;
    bool is_empty;

    // Open addressing with linear probing
    while (true) {
        bucket_offset = bucket_idx * L_BUCKET_LEN;
        int existing_idx = EMPTY_BUCKET_KEY;
        // Insert name idx if bucket is empty
        is_empty = atomic_compare_exchange_weak_explicit(
            &buckets[bucket_offset], // object
            &existing_idx, // expected
            as_type<int>(name_idx), // desired
            memory_order_relaxed, 
            memory_order_relaxed
        );
        if (is_empty || name_eq(buf, as_type<uint>(existing_idx), name_idx)) {
            break;
        } else {
            bucket_idx = (bucket_idx + 1) % L_HASHMAP_LEN;
        }
    }

    if (is_empty) {
        // Cache the global hashmap bucket idx since we are dropping the hash value now
        uint g_bucket_idx = hash % G_HASHMAP_LEN;
        atomic_store_explicit(
            &buckets[bucket_offset + G_BUCKET_FIELD], 
            as_type<int>(g_bucket_idx), 
            memory_order_relaxed
        );
    }

    atomic_fetch_min_explicit(&buckets[bucket_offset + MIN_FIELD], temp, memory_order_relaxed);
    atomic_fetch_max_explicit(&buckets[bucket_offset + MAX_FIELD], temp, memory_order_relaxed);
    atomic_fetch_add_explicit(&buckets[bucket_offset + SUM_FIELD], temp, memory_order_relaxed);
    atomic_fetch_add_explicit(&buckets[bucket_offset + COUNT_FIELD], 1, memory_order_relaxed);
}

void merge_global_hashmap(
    const device uchar* buf,
    device atomic_int* g_buckets,
    threadgroup atomic_int* l_buckets,
    uint lid,
    uint threadgroup_size
) {
    for (uint l_bucket_idx = lid; l_bucket_idx < L_HASHMAP_LEN; l_bucket_idx += threadgroup_size) {
        uint l_bucket_offset = l_bucket_idx * L_BUCKET_LEN;
        int name_idx, min, max, sum, count, g_bucket_idx;
        uint g_bucket_offset;
        if (REINTERPRET_ATOMICS) {
            threadgroup int* l_buckets_ = reinterpret_cast<threadgroup int*>(l_buckets);

            name_idx = l_buckets_[l_bucket_offset];
            min = l_buckets_[l_bucket_offset + MIN_FIELD];
            max = l_buckets_[l_bucket_offset + MAX_FIELD];
            sum = l_buckets_[l_bucket_offset + SUM_FIELD];
            count = l_buckets_[l_bucket_offset + COUNT_FIELD];
            g_bucket_idx = as_type<uint>(l_buckets_[l_bucket_offset + G_BUCKET_FIELD]);
        } else {
            name_idx = atomic_load_explicit(&l_buckets[l_bucket_offset], memory_order_relaxed);
            min = atomic_load_explicit(&l_buckets[l_bucket_offset + MIN_FIELD], memory_order_relaxed);
            max = atomic_load_explicit(&l_buckets[l_bucket_offset + MAX_FIELD], memory_order_relaxed);
            sum = atomic_load_explicit(&l_buckets[l_bucket_offset + SUM_FIELD], memory_order_relaxed);
            count = atomic_load_explicit(&l_buckets[l_bucket_offset + COUNT_FIELD], memory_order_relaxed);
            g_bucket_idx = as_type<uint>(
                atomic_load_explicit(&l_buckets[l_bucket_offset + G_BUCKET_FIELD], memory_order_relaxed));
        }

        // Open addressing with linear probing
        while (true) {
            g_bucket_offset = g_bucket_idx * G_BUCKET_LEN;
            // Insert name idx if bucket is empty
            int existing_idx = EMPTY_BUCKET_KEY;
            bool is_empty = atomic_compare_exchange_weak_explicit(
                &g_buckets[g_bucket_offset], // object
                &existing_idx, // expected
                name_idx, // desired
                memory_order_relaxed, 
                memory_order_relaxed
            );
            if (is_empty || name_eq(buf, as_type<uint>(existing_idx), as_type<uint>(name_idx))) {
                break;
            } else {
                g_bucket_idx = (g_bucket_idx + 1) % G_HASHMAP_LEN;
            }
        }

        atomic_fetch_min_explicit(&g_buckets[g_bucket_offset + MIN_FIELD], min, memory_order_relaxed);
        atomic_fetch_max_explicit(&g_buckets[g_bucket_offset + MAX_FIELD], max, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_buckets[g_bucket_offset + SUM_FIELD], sum, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_buckets[g_bucket_offset + COUNT_FIELD], count, memory_order_relaxed);
    }
}

kernel void histogram(
    const device uchar* buf,
    device atomic_int* g_buckets,
    const device uint& buf_len,
    const device uint& chunk_len,
    uint gid              [[ thread_position_in_grid ]],
    uint lid              [[ thread_position_in_threadgroup ]],
    uint grid_size        [[ threads_per_grid ]],
    uint threadgroup_size [[ threads_per_threadgroup ]]
) {
    threadgroup atomic_int l_buckets[L_HASHMAP_LEN * L_BUCKET_LEN];
    init_local_hashmap(l_buckets, lid, threadgroup_size);
    threadgroup_barrier(mem_flags::mem_none);

    const uint unaligned_start_idx = (uint)gid * chunk_len;

    // Align start and end
    uint start_idx = unaligned_start_idx;
    if (gid > 0) {
        while (buf[start_idx++] != '\n');
    }
    uint end_idx = unaligned_start_idx + chunk_len;
    if (gid < grid_size - 1) {
        while (buf[end_idx++] != '\n');
    } else {
        end_idx = buf_len;
    }

    uint i = start_idx;
    while (i < end_idx) {
        // read name
        uint name_idx = i;
        uint64_t hash = 17;
        uchar c;
        while ((c = buf[i++]) != ';') {
            hash = 31 * hash + c;
        }

        // read temp
        int sign = 1;
        if (buf[i] == '-') {
            sign = -1;
            i = i + 1;
        }
        int temp = 0;
        while ((c = buf[i++]) != '\n') {
            if (c != '.') {
                temp = temp * 10 + (c - '0');
            }
        }

        update_local_hashmap(buf, l_buckets, name_idx, hash, sign * temp);
    }

    threadgroup_barrier(mem_flags::mem_none);
    merge_global_hashmap(buf, g_buckets, l_buckets, lid, threadgroup_size);
}
