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

uint ptr_hash_index(uint64_t hash) {
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
    uint incr // How much to increment buf by (range: [0, 8]) in bytes
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

kernel void debug(
    const device uchar* buf, // 16-byte aligned
    const device uchar* keys, // 32-byte aligned
    device uint64_t* res,
    const device uint& len,
    uint gid [[ thread_position_in_grid ]]
) {
    threadgroup ulong4 lkeys[512];
    if (gid < 512) {
        const device ulong4* simd_keys = reinterpret_cast<const device ulong4*>(keys);
        lkeys[gid] = simd_keys[gid];
    }
    if (gid != 0) {
        return;
    }

    const device ulong2* simd_buf = reinterpret_cast<const device ulong2*>(buf);
    ulong4 chunk(simd_buf[0], simd_buf[1]);
    uint simd_offset = 2;
    uint offset = 0;
    uint i = 0;

    // uint64_t newline_bits;
    // while ((newline_bits = swar_find_char<NEWLINE>(chunk.x)) == 0) {
    //     read_simd(simd_buf, chunk, simd_offset, offset, 8);
    //     i += 8;
    // }
    // uint incr = (ctz(newline_bits) >> 3) + 1;
    // i += incr;
    uint incr = 0;

    for (uint k = 0; k < 12; ++k) {
        uint64_t hash = 0;
        uint64_t semi_bits;
        uint start = i;
        uint j;
        ulong4 name = 0;
        for (j = 0; j <= 100 / 8; ++j) {
            read_simd(simd_buf, chunk, simd_offset, offset, incr);
            semi_bits = swar_find_char<SEMICOLON>(chunk.x);
            if (semi_bits != 0) break;
            if (j < 3) name[j] = chunk.x;
            hash = (rotate(hash, 5UL) ^ chunk.x) * FX_HASH_K;
            incr = 8;
            i += 8;
        }
        // first character is ;
        if (semi_bits != 0x80) {
            uint64_t tail = mask_name(chunk.x, semi_bits);
            hash = (rotate(hash, 5UL) ^ tail) * FX_HASH_K;
            if (j < 4) name[j] = tail;
        }
        if (j < 4) {
            uint slot = ptr_hash_index(hash);
            ulong4 key = lkeys[slot];
            res[k] = all(name == key);
        }
        uint temp_pos = (ctz(semi_bits) >> 3) + 1;
        read_simd(simd_buf, chunk, simd_offset, offset, temp_pos);
        uint dot_pos = ctz(~chunk.x & DOT_BITS);
        incr = (dot_pos >> 3) + 3;
        int temp = swar_parse_temp(as_type<int64_t>(chunk.x), dot_pos);
        i += temp_pos + incr;
    }
}
