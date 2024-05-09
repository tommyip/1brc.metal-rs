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

kernel void debug(
    const device uchar* buf,
    device int* res,
    const device uint& len
) {
    const device ulong2* simd_buf = reinterpret_cast<const device ulong2*>(buf);
    ulong4 chunk(simd_buf[0], simd_buf[1]);
    uint simd_offset = 2;
    uint offset = 0;

    uint64_t newline_bits;
    while ((newline_bits = swar_find_char<NEWLINE>(chunk.x)) == 0) {
        read_simd(simd_buf, chunk, simd_offset, offset, 8);
    }
    uint incr = (ctz(newline_bits) >> 3) + 1;

    for (uint k = 0; k < 2; ++k) {
        uint64_t hash = 5381;
        uint64_t name_chunk;
        uint64_t semi_bits;
        for (uint j = 0; j <= 100 / 8; ++j) {
            read_simd(simd_buf, chunk, simd_offset, offset, incr);
            semi_bits = swar_find_char<SEMICOLON>(chunk.x);
            if (semi_bits != 0) break;
            hash = 33 * hash + chunk.x;
            incr = 8;
        }
        hash = 33 * hash + mask_name(chunk.x, semi_bits);
        read_simd(simd_buf, chunk, simd_offset, offset, (ctz(semi_bits) >> 3) + 1);
        uint dot_pos = ctz(~chunk.x & DOT_BITS);
        incr = (dot_pos >> 3) + 3;
        int temp = swar_parse_temp(as_type<int64_t>(chunk.x), dot_pos);
        res[k] = temp;
    }
}
