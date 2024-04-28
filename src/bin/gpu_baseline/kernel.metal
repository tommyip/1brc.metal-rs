#define HASHMAP_LEN 20000
#define MIN_FIELD 1
#define MAX_FIELD 2
#define SUM_FIELD 3
#define COUNT_FIELD 4

#include <metal_stdlib>

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

void hashmap_update(
    const device uchar* buf,
    device atomic_int* buckets,
    uint name_idx,
    uint64_t hash,
    int temp
) {
    uint bucket_idx = hash % HASHMAP_LEN;
    int name_idx_int = as_type<int>(name_idx);
    // Open addressing with linear probing
    while (true) {
        // No names starts at index 1 so treat it as an empty bucket
        int existing_idx = 1;
        // Insert name idx if bucket is empty
        bool is_empty = atomic_compare_exchange_weak_explicit(
            &buckets[bucket_idx * 5], // object
            &existing_idx, // expected
            name_idx_int, // desired
            memory_order_relaxed, 
            memory_order_relaxed
        );
        if (is_empty || name_eq(buf, as_type<uint>(existing_idx), name_idx)) {
            break;
        } else {
            bucket_idx = (bucket_idx + 1) % HASHMAP_LEN;
        }
    }
    bucket_idx = bucket_idx * 5;
    atomic_fetch_min_explicit(&buckets[bucket_idx + MIN_FIELD], temp, memory_order_relaxed);
    atomic_fetch_max_explicit(&buckets[bucket_idx + MAX_FIELD], temp, memory_order_relaxed);
    atomic_fetch_add_explicit(&buckets[bucket_idx + SUM_FIELD], temp, memory_order_relaxed);
    atomic_fetch_add_explicit(&buckets[bucket_idx + COUNT_FIELD], 1, memory_order_relaxed);
}

kernel void histogram(
    const device uchar* buf,
    device atomic_int* buckets,
    const device uint& buf_len,
    const device uint& chunk_len,
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
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

        hashmap_update(buf, buckets, name_idx, hash, sign * temp);
    }
}
