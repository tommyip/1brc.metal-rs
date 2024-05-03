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

using namespace metal;

// SIMD-accelerated DJBX33A hash function
// Advance `i` to one position after the semicolon
uint64_t hash_name(const device uchar* buf, thread uint* i) {
    ulong4 hash4 = ulong4(5381);
    ulong4 semi4 = ulong4((ulong)';');
    ulong4 char4;
    bool4 is_semi4;
    for (;; *i += 4) {
        char4 = ulong4(buf[*i], buf[*i + 1], buf[*i + 2], buf[*i + 3]);
        is_semi4 = char4 == semi4;
        if (any(is_semi4)) break;
        hash4 = 33 * hash4 + char4;
    }
    uint j;
    for (j = 0; j < 4; ++j) {
        if (is_semi4[j]) break;
        hash4[j] = 33 * hash4[j] + char4[j];
    }
    *i += j + 1;

    // XORing the lanes together is an arbitrary choice
    return hash4[0] ^ hash4[1] ^ hash4[2] ^ hash4[3];
}

// Compare string until `;`
bool name_eq(
    const device uchar* buf,
    uint a_idx,
    uint b_idx
) {
    packed_uchar4 a, b;
    packed_uchar4 semi(';');
    bool4 eq;
    for (;; a_idx += 4, b_idx += 4) {
        a = packed_uchar4(buf[a_idx], buf[a_idx + 1], buf[a_idx + 2], buf[a_idx + 3]);
        b = packed_uchar4(buf[b_idx], buf[b_idx + 1], buf[b_idx + 2], buf[b_idx + 3]);
        eq = a == b;
        if (any(a == semi | b == semi)) {
            if (!eq[0]) break;
            if (a[0] == ';') return true;
            if (!eq[1]) break;
            if (a[1] == ';') return true;
            if (!eq[2]) break;
            if (a[2] == ';') return true;
            if (!eq[3]) break;
            return true;
        } else {
            if (!all(eq)) break;
        }
    }
    return false;
}

int read_temp(const device uchar* buf, thread uint* j) {
    uint i = *j;
    int sign = 1;
    if (buf[i] == '-') {
        sign = -1;
        ++i;
    }
    int temp = buf[i++] - '0';
    char c = buf[i];
    if (c != '.') {
        temp = temp * 10 + (c - '0');
        ++i;
    }
    temp = temp * 10 + (buf[i + 1] - '0');
    *j = i + 3;
    return sign * temp;
}

void init_local_hashmap(threadgroup atomic_int* buckets, uint lid, uint threadgroup_size) {
    for (uint i = lid; i < L_HASHMAP_LEN; i += threadgroup_size) {
        uint offset = i * L_BUCKET_LEN;

        // Each thread initialize a disjoint set of slots so no need for atomic stores.
        threadgroup int* buckets_ = reinterpret_cast<threadgroup int*>(buckets);

        // Global bucket field is overwrite only so no need to initialize
        buckets_[offset] = EMPTY_BUCKET_KEY;
        buckets_[offset + MIN_FIELD] = INT_MAX;
        buckets_[offset + MAX_FIELD] = INT_MIN;
        buckets_[offset + SUM_FIELD] = 0;
        buckets_[offset + COUNT_FIELD] = 0;
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
    threadgroup atomic_int* l_buckets_,
    uint lid,
    uint threadgroup_size
) {
    // No more updates to local hashmap at this point
    threadgroup int* l_buckets = reinterpret_cast<threadgroup int*>(l_buckets_);

    for (uint l_bucket_idx = lid; l_bucket_idx < L_HASHMAP_LEN; l_bucket_idx += threadgroup_size) {
        uint l_bucket_offset = l_bucket_idx * L_BUCKET_LEN;
        int name_idx, min, max, sum, count, g_bucket_idx;
        uint g_bucket_offset;

        name_idx = l_buckets[l_bucket_offset];
        min = l_buckets[l_bucket_offset + MIN_FIELD];
        max = l_buckets[l_bucket_offset + MAX_FIELD];
        sum = l_buckets[l_bucket_offset + SUM_FIELD];
        count = l_buckets[l_bucket_offset + COUNT_FIELD];
        g_bucket_idx = as_type<uint>(l_buckets[l_bucket_offset + G_BUCKET_FIELD]);

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

    // Align start
    uint start_idx = unaligned_start_idx;
    while (buf[start_idx++] != '\n');
    // End bound does not require aligning as our main loop fully reads each
    // line.
    uint end_idx = unaligned_start_idx + chunk_len;

    uint i = start_idx;
    while (i <= end_idx) {
        uint name_idx = i;
        uint64_t hash = hash_name(buf, &i);
        int temp = read_temp(buf, &i);
        
        update_local_hashmap(buf, l_buckets, name_idx, hash, temp);
    }

    threadgroup_barrier(mem_flags::mem_none);
    merge_global_hashmap(buf, g_buckets, l_buckets, lid, threadgroup_size);
}
