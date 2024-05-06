#define BROADCAST_SEMICOLON 0x3B3B3B3B3B3B3B3B
#define BROADCAST_0x01 0x0101010101010101
#define BROADCAST_0x80 0x8080808080808080
#define DOT_BITS 0x10101000
#define MAGIC_MULTIPLIER (100 * 0x1000000 + 10 * 0x10000 + 1)

#include <metal_stdlib>

using namespace metal;

uint64_t find_semi_bits(uint64_t name_chunk) {
    uint64_t diff = name_chunk ^ BROADCAST_SEMICOLON;
    return (diff - BROADCAST_0x01) & (~diff & BROADCAST_0x80);
}

uint64_t mask_name(uint64_t name_chunk, uint64_t semi_bits) {
    uint64_t mask = semi_bits ^ (semi_bits - 1);
    return name_chunk & mask;
}

uint64_t read_simd(const device uchar* buf, uint i) {
    const device packed_uchar4* simd_buf = reinterpret_cast<const device packed_uchar4*>(&buf[i]);
    const uchar4 lo = *simd_buf;
    const uchar4 hi = *(simd_buf + 1);
    uint64_t name_chunk = *reinterpret_cast<const thread uint*>(&hi);
    return name_chunk << 32 | *reinterpret_cast<const thread uint*>(&lo);
}

int parse_temp(int64_t temp, uint dot_pos) {
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
    uint i = 0;
    for (uint k = 0; k < 4; ++k) {
        uint64_t hash = 5381;
        uint64_t name_chunk;
        uint64_t semi_bits;
        for (uint j = 0; j <= 100 / 8; ++j) {
            name_chunk = read_simd(buf, i);
            semi_bits = find_semi_bits(name_chunk);
            if (semi_bits != 0) break;
            hash = 33 * hash + name_chunk;
            i += 8;
        }
        i += (ctz(semi_bits) >> 3) + 1;
        name_chunk = mask_name(name_chunk, semi_bits);
        hash = 33 * hash + name_chunk;
        uint64_t temp_chunk = read_simd(buf, i);
        uint dot_pos = ctz(~temp_chunk & DOT_BITS);
        i += (dot_pos >> 3) + 3;
        int temp = parse_temp(as_type<int64_t>(temp_chunk), dot_pos);
        res[k] = temp;
    }
}
