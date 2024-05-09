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

#define SEMICOLON 0x3B3B3B3B3B3B3B3B
#define NEWLINE 0x0A0A0A0A0A0A0A0A
#define DOT_BITS 0x10101000
#define MAGIC_MULTIPLIER (100 * 0x1000000 + 10 * 0x10000 + 1)

#include <metal_stdlib>

constant uint G_HASHMAP_LEN [[ function_constant(0) ]];

using namespace metal;

template<uint64_t BROADCAST_CHAR>
uint64_t swar_find_char(uint64_t input) {
    uint64_t diff = input ^ BROADCAST_CHAR;
    return (diff - 0x0101010101010101) & (~diff & 0x8080808080808080);
}

template uint64_t swar_find_char<NEWLINE>(uint64_t);
template uint64_t swar_find_char<SEMICOLON>(uint64_t);

uint64_t mask_name(uint64_t name_chunk, uint64_t semi_bits) {
    uint64_t mask = semi_bits ^ (semi_bits - 1);
    return name_chunk & mask;
}

void read_simd(
    const device ulong2* simd_buf,
    thread ulong4& out, // `out.x` is the output, `out.yzw` is internal lookahead
    thread uint& simd_offset, // Next `simd_buf` index
    thread uint& offset, // Offset of `out` to read (internal)
    uint incr // How much to increment buf by (range: [0, 8])
) {
    offset += incr;
    uint shift_left;
    uint take_right = offset & 0b111;
    if (offset < 8) {
        shift_left = incr;
    } else if (offset < 16) {
        out.x = out.y;
        shift_left = take_right;
    } else {
        offset = take_right;
        shift_left = take_right;
        out.xy = out.zw;
        out.zw = simd_buf[simd_offset++];
    }
    take_right <<= 3;
    out.x >>= shift_left << 3;
    ulong right = offset < 8 ? out.y : out.z;
    out.x = insert_bits(out.x, right, (64 - take_right) & 0b111111, take_right);
}

int swar_parse_temp(int64_t temp, uint dot_pos) {
    int shift = 28 - dot_pos;
    int64_t sign = (~temp << 59) >> 63;
    int64_t minus_filter = ~(sign & 0xFF);
    int64_t digits = ((temp & minus_filter) << shift) & 0x0F000F0F00;
    uint64_t abs_value = (as_type<uint64_t>(digits * MAGIC_MULTIPLIER) >> 32) & 0x3FF;
    return (abs_value ^ sign) - sign;
}

// Compare string until `;`
bool name_eq(
    const device uchar* buf,
    uint a_idx,
    uint b_idx
) {
    const uchar4 semi(';');
    const device packed_uchar4* a_buf = reinterpret_cast<const device packed_uchar4*>(&buf[a_idx]);
    const device packed_uchar4* b_buf = reinterpret_cast<const device packed_uchar4*>(&buf[b_idx]);
    uchar4 a, b;
    bool4 eq;
    for (;; ++a_buf, ++b_buf) {
        a = *a_buf;
        b = *b_buf;
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
    const device uchar* buf, // Must be 16 bytes aligned
    device atomic_int* g_buckets,
    const device uint& buf_len,
    const device uint& chunk_len,  // Must be a multiple of 16 bytes
    uint gid              [[ thread_position_in_grid ]],
    uint lid              [[ thread_position_in_threadgroup ]],
    uint grid_size        [[ threads_per_grid ]],
    uint threadgroup_size [[ threads_per_threadgroup ]]
) {
    threadgroup atomic_int l_buckets[L_HASHMAP_LEN * L_BUCKET_LEN];
    init_local_hashmap(l_buckets, lid, threadgroup_size);
    threadgroup_barrier(mem_flags::mem_none);

    uint i = gid * chunk_len;
    const device ulong2* simd_buf = reinterpret_cast<const device ulong2*>(&buf[i]);
    ulong4 chunk(simd_buf[0], simd_buf[1]);
    uint simd_offset = 2;
    uint offset = 0;

    // Align start
    uint64_t newline_bits;
    while ((newline_bits = swar_find_char<NEWLINE>(chunk.x)) == 0) {
        read_simd(simd_buf, chunk, simd_offset, offset, 8);
        i += 8;
    }
    uint incr = (ctz(newline_bits) >> 3) + 1;
    i += incr;

    while (i <= gid * chunk_len + chunk_len) {
        uint name_idx = i;
        uint64_t hash = 5381;
        uint64_t semi_bits;
        for (uint j = 0; j <= 100 / 8; ++j) {
            read_simd(simd_buf, chunk, simd_offset, offset, incr);
            semi_bits = swar_find_char<SEMICOLON>(chunk.x);
            if (semi_bits != 0) break;
            hash = 33 * hash + chunk.x;
            incr = 8;
            i += 8;
        }
        hash = 33 * hash + mask_name(chunk.x, semi_bits);
        uint temp_pos = (ctz(semi_bits) >> 3) + 1;
        read_simd(simd_buf, chunk, simd_offset, offset, temp_pos);
        uint dot_bit_pos = ctz(~chunk.x & DOT_BITS);
        incr = (dot_bit_pos >> 3) + 3;
        int temp = swar_parse_temp(as_type<int64_t>(chunk.x), dot_bit_pos);
        i += temp_pos + incr;

        update_local_hashmap(buf, l_buckets, name_idx, hash, temp);
    }

    threadgroup_barrier(mem_flags::mem_none);
    merge_global_hashmap(buf, g_buckets, l_buckets, lid, threadgroup_size);
}
