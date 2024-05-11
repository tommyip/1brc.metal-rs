#define SEMICOLON 0x3B3B3B3B3B3B3B3B
#define NEWLINE 0x0A0A0A0A0A0A0A0A
#define DOT_BITS 0x10101000
#define MAGIC_MULTIPLIER (100 * 0x1000000 + 10 * 0x10000 + 1)

#include <metal_stdlib>

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
    const device uchar* buf,
    device uint64_t* res,
    const device uint& len
) {
    const device ulong2* simd_buf = reinterpret_cast<const device ulong2*>(buf);
    ulong4 chunk(simd_buf[0], simd_buf[1]);
    uint simd_offset = 2;
    uint offset = 0;
    uint i = 0;

    uint64_t newline_bits;
    while ((newline_bits = swar_find_char<NEWLINE>(chunk.x)) == 0) {
        read_simd(simd_buf, chunk, simd_offset, offset, 8);
        i += 8;
    }
    uint incr = (ctz(newline_bits) >> 3) + 1;
    i += incr;

    for (uint k = 0; k < 3; ++k) {
        uint64_t hash = 5381;
        uint64_t semi_bits;
        uint start = i;
        uint j;
        for (j = 0; j <= 100 / 8; ++j) {
            read_simd(simd_buf, chunk, simd_offset, offset, incr);
            semi_bits = swar_find_char<SEMICOLON>(chunk.x);
            if (j == 0) {
                res[k] = chunk.x;
            }
            if (semi_bits != 0) break;
            hash = 33 * hash + chunk.x;
            incr = 8;
            i += 8;
        }
        hash = 33 * hash + mask_name(chunk.x, semi_bits);
        uint temp_pos = (ctz(semi_bits) >> 3) + 1;
        read_simd(simd_buf, chunk, simd_offset, offset, temp_pos);
        uint dot_pos = ctz(~chunk.x & DOT_BITS);
        incr = (dot_pos >> 3) + 3;
        int temp = swar_parse_temp(as_type<int64_t>(chunk.x), dot_pos);
        i += temp_pos + incr;
    }
}
