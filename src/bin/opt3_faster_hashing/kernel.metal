#define HASHMAP_LEN 512
#define BUCKET_LEN 4
#define MIN_FIELD 0
#define MAX_FIELD 1
#define SUM_FIELD 2
#define COUNT_FIELD 3

#define SEMICOLON 0x3B3B3B3B3B3B3B3B
#define NEWLINE 0x0A0A0A0A0A0A0A0A
#define DOT_BITS 0x10101000
#define MAGIC_MULTIPLIER (100 * 0x1000000 + 10 * 0x10000 + 1)

#define FX_HASH_K 0x517cc1b727220a95
#define PTR_REM_C1 38
#define PTR_REM_C2 132
#define PTR_C3 -55
#define PTR_P1 11068046444225730560
#define PTR_MUL_REDUCE_MASK ((1 << 9) - 1)

#include <metal_stdlib>

using namespace metal;

constant uint8_t PILOTS[] = {
    50, 110, 71, 13, 18, 27, 21, 14, 10, 16, 1, 14, 6, 11, 0, 2, 17, 2, 1, 4, 68, 79, 21, 0,
    22, 20, 60, 12, 30, 53, 62, 78, 27, 17, 2, 17, 13, 43, 21, 108, 19, 12, 25, 1, 55, 36, 1,
    0, 4, 184, 0, 21, 69, 25, 13, 177, 11, 97, 3, 29, 14, 104, 30, 4, 50, 23, 6, 102, 137, 10,
    227, 32, 29, 21, 7, 4, 244, 0,
};

uint mph_index(uint64_t hash) {
    bool is_large = hash >= PTR_P1;
    uint64_t rem = is_large ? PTR_REM_C2 : PTR_REM_C1;
    uint bucket = is_large * PTR_C3 + mulhi(hash, rem);
    uint64_t pilot = PILOTS[bucket];
    return (uint)mulhi((uint64_t)FX_HASH_K, hash ^ (FX_HASH_K * pilot)) & PTR_MUL_REDUCE_MASK;
}

template<uint64_t BROADCAST_CHAR>
uint64_t swar_find_char(uint64_t input) {
    uint64_t diff = input ^ BROADCAST_CHAR;
    return (diff - 0x0101010101010101) & (~diff & 0x8080808080808080);
}

template uint64_t swar_find_char<NEWLINE>(uint64_t);
template uint64_t swar_find_char<SEMICOLON>(uint64_t);

uint64_t mask_name(uint64_t name_chunk, uint64_t semi_bits) {
    uint64_t mask = semi_bits ^ (semi_bits - 1);
    return name_chunk & mask >> 8;
}

void read_simd(
    const device ulong2* simd_buf,
    thread ulong4& out, // `out.x` is the output, `out.yzw` is internal lookahead
    thread uint& simd_offset, // Next `simd_buf` index
    thread uint& offset, // Offset of `out` to read (internal) in number of bits from LSB
    uint incr // How much to increment buf by (range: [0, 64]) in bits (multiple of 8)
) {
    incr <<= 3;
    offset += incr;
    uint offset_mod64 = offset & 0b111111;
    uint shift_left = offset_mod64;
    if (offset < 64) {
        shift_left = incr;
    } else if (offset < 128) {
        out.x = out.y;
    } else {
        offset = shift_left;
        out = ulong4(out.zw, simd_buf[simd_offset++]);
    }
    out.x >>= shift_left;
    if (shift_left > 0) {
        ulong right = offset < 64 ? out.y : out.z;
        out.x |= right << (64 - offset_mod64);
    }
}

int swar_parse_temp(int64_t temp, uint dot_pos) {
    int shift = 28 - dot_pos;
    int64_t sign = (~temp << 59) >> 63;
    int64_t minus_filter = ~(sign & 0xFF);
    int64_t digits = ((temp & minus_filter) << shift) & 0x0F000F0F00;
    uint64_t abs_value = (as_type<uint64_t>(digits * MAGIC_MULTIPLIER) >> 32) & 0x3FF;
    return (abs_value ^ sign) - sign;
}

void init_threadgroup_mem(
    threadgroup atomic_int* buckets_, 
    uint lid, 
    uint threadgroup_size
) {
    // Each thread initialize a disjoint set of slots so no need for atomic stores.
    threadgroup int* buckets = reinterpret_cast<threadgroup int*>(buckets_);

    for (uint i = lid; i < HASHMAP_LEN; i += threadgroup_size) {
        uint offset = i * BUCKET_LEN;
        buckets[offset + MIN_FIELD] = INT_MAX;
        buckets[offset + MAX_FIELD] = INT_MIN;
        buckets[offset + SUM_FIELD] = 0;
        buckets[offset + COUNT_FIELD] = 0;
    }
}

// Return whether the key can be used in the fast path
bool update_local_hashmap(
    threadgroup atomic_int* buckets,
    const device ulong4* mph_keys,
    thread ulong4& key_prefix,
    uint64_t hash,
    int temp
) {
    uint bucket = mph_index(hash);
    // Fail if key is not in the MPH set
    if (any(key_prefix != mph_keys[bucket])) return false;
    
    uint bucket_offset = bucket * BUCKET_LEN;
    atomic_fetch_min_explicit(&buckets[bucket_offset + MIN_FIELD], temp, memory_order_relaxed);
    atomic_fetch_max_explicit(&buckets[bucket_offset + MAX_FIELD], temp, memory_order_relaxed);
    atomic_fetch_add_explicit(&buckets[bucket_offset + SUM_FIELD], temp, memory_order_relaxed);
    atomic_fetch_add_explicit(&buckets[bucket_offset + COUNT_FIELD], 1, memory_order_relaxed);
    return true;
}

void merge_global_hashmap(
    device atomic_int* g_buckets,
    const threadgroup atomic_int* buckets_,
    uint lid,
    uint threadgroup_size
) {
    const threadgroup int* buckets = reinterpret_cast<const threadgroup int*>(buckets_);

    for (uint i = lid; i < HASHMAP_LEN; i += threadgroup_size) {
        uint offset = i * BUCKET_LEN;
        int min = buckets[offset + MIN_FIELD];
        int max = buckets[offset + MAX_FIELD];
        int sum = buckets[offset + SUM_FIELD];
        int count = buckets[offset + COUNT_FIELD];
        atomic_fetch_min_explicit(&g_buckets[offset + MIN_FIELD], min, memory_order_relaxed);
        atomic_fetch_max_explicit(&g_buckets[offset + MAX_FIELD], max, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_buckets[offset + SUM_FIELD], sum, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_buckets[offset + COUNT_FIELD], count, memory_order_relaxed);
    }
}

kernel void histogram(
    const device uchar* buf, // Must be 16 bytes aligned
    const device ulong4* mph_keys,
    device atomic_int* g_buckets,
    const device uint& chunk_len,  // Must be a multiple of 16 bytes
    device bool& fallback,
    uint gid              [[ thread_position_in_grid ]],
    uint lid              [[ thread_position_in_threadgroup ]],
    uint grid_size        [[ threads_per_grid ]],
    uint threadgroup_size [[ threads_per_threadgroup ]]
) {
    threadgroup atomic_int buckets[HASHMAP_LEN * BUCKET_LEN];
    init_threadgroup_mem(buckets, lid, threadgroup_size);
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
        uint64_t hash = 0;
        uint64_t semi_bits;
        uint key_lane;
        ulong4 key_prefix = 0;
        for (key_lane = 0; key_lane <= 100 / 8; ++key_lane) {
            read_simd(simd_buf, chunk, simd_offset, offset, incr);
            semi_bits = swar_find_char<SEMICOLON>(chunk.x);
            if (semi_bits != 0) break;
            if (key_lane < 4) key_prefix[key_lane] = chunk.x;
            hash = (rotate(hash, 5UL) ^ chunk.x) * FX_HASH_K;
            incr = 8;
            i += 8;
        }
        // first character is ;
        if (semi_bits != 0x80) {
            uint tail_lane = min(key_lane, 3U);
            key_prefix[tail_lane] = mask_name(chunk.x, semi_bits);
            hash = (rotate(hash, 5UL) ^ key_prefix[tail_lane]) * FX_HASH_K;
        }
        uint temp_pos = (ctz(semi_bits) >> 3) + 1;
        read_simd(simd_buf, chunk, simd_offset, offset, temp_pos);
        uint dot_bit_pos = ctz(~chunk.x & DOT_BITS);
        int temp = swar_parse_temp(as_type<int64_t>(chunk.x), dot_bit_pos);
        incr = (dot_bit_pos >> 3) + 3;
        i += temp_pos + incr;

        if (key_lane >= 4 ||
            !update_local_hashmap(buckets, mph_keys, key_prefix, hash, temp)
        ) {
            // Fallback to slow path
            fallback = true;
        }
    }

    threadgroup_barrier(mem_flags::mem_none);
    merge_global_hashmap(g_buckets, buckets, lid, threadgroup_size);
}
