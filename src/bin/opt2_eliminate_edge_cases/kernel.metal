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

uint64_t hash_name(const device uchar* buf, uint start, uint end) {
    uint64_t h = 5381;
    for (uint i = start; i < end; ++i) {
        h = 33 * h + buf[i];
    }
    return h;
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

int read_temp(const device uchar* buf, uint start, uint end) {
    uint temp_len = end - start;
    int sign = 1;
    if (buf[start] == '-') {
        sign = -1;
        temp_len -= 1;
        ++start;
    }
    int temp;
    if (temp_len == 3) {
        temp = (buf[start] - '0') * 10 + (buf[start+2] - '0');
    } else {
        temp = (buf[start] - '0') * 100 + (buf[start+1] - '0') * 10 + (buf[start+3] - '0');
    }
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
    for (uint i = 0; i < L_HASHMAP_LEN; ++i) {
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
        for (uint i = 0; i < G_HASHMAP_LEN; ++i) {
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

    const uint chunk_offset = gid * chunk_len;
    const device uchar* chunk_buf = &buf[chunk_offset];
    const device packed_uchar4* simd_buf = reinterpret_cast<const device packed_uchar4*>(chunk_buf);
    const packed_uchar4 semi = packed_uchar4(';');
    const packed_uchar4 newline = packed_uchar4('\n');

    bool4 eq_semi = simd_buf[0] == semi;
    bool4 eq_newline = simd_buf[0] == newline; 
    uint i = 0;

    // Align start
    while (!any(eq_newline)) {
        ++i;
        eq_semi = simd_buf[i] == semi;
        eq_newline = simd_buf[i] == newline;
    }
    uint lane_id = ctz(*reinterpret_cast<thread uint*>(&eq_newline)) >> 3;
    uint line_offset = 4 * i + lane_id + 1;
    eq_newline[lane_id] = false;
    
    while (i <= chunk_len / 4) {
        // Find ;
        while (!any(eq_semi)) {
            ++i;
            eq_semi = simd_buf[i] == semi;
            eq_newline = simd_buf[i] == newline;
        }
        lane_id = ctz(*reinterpret_cast<thread uint*>(&eq_semi)) >> 3;
        eq_semi[lane_id] = false;
        uint semi_pos = 4 * i + lane_id;

        // Find \n
        while (!any(eq_newline)) {
            ++i;
            eq_semi = simd_buf[i] == semi;
            eq_newline = simd_buf[i] == newline;
        }
        lane_id = ctz(*reinterpret_cast<thread uint*>(&eq_newline)) >> 3;
        eq_newline[lane_id] = false;
        uint newline_pos = 4 * i + lane_id;
        
        uint64_t hash = hash_name(chunk_buf, line_offset, semi_pos);
        int temp = read_temp(chunk_buf, semi_pos + 1, newline_pos);
        
        update_local_hashmap(buf, l_buckets, chunk_offset + line_offset, hash, temp);

        line_offset = newline_pos + 1;
    }

    threadgroup_barrier(mem_flags::mem_none);
    merge_global_hashmap(buf, g_buckets, l_buckets, lid, threadgroup_size);
}
