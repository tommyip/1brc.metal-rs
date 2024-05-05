#include <metal_stdlib>

using namespace metal;

kernel void debug(
    const device uchar* buf,
    device int* res,
    const device uint& len
) {
    const packed_uchar4 semi = packed_uchar4(';');
    const packed_uchar4 newline = packed_uchar4('\n');
    const device packed_uchar4* simd_buf = reinterpret_cast<const device packed_uchar4*>(buf);
    bool4 eq_semi = simd_buf[0] == semi;
    bool4 eq_newline = simd_buf[0] == newline;

    uint i = 0;

    while (!any(eq_newline)) {
        ++i;
        eq_semi = simd_buf[i] == semi;
        eq_newline = simd_buf[i] == newline;
    }
    uint offset = ctz(*reinterpret_cast<thread uint*>(&eq_newline)) >> 3;
    eq_newline[offset] = false;
    
    for (uint j = 0; j < 3; ++j) {
        while (!any(eq_semi)) {
            ++i;
            eq_semi = simd_buf[i] == semi;
            eq_newline = simd_buf[i] == newline;
        }

        offset = ctz(*reinterpret_cast<thread uint*>(&eq_semi)) >> 3;
        eq_semi[offset] = false;
        uint semi_pos = 4 * i + offset;

        while (!any(eq_newline)) {
            ++i;
            eq_semi = simd_buf[i] == semi;
            eq_newline = simd_buf[i] == newline;
        }

        offset = ctz(*reinterpret_cast<thread uint*>(&eq_newline)) >> 3;
        eq_newline[offset] = false;
        
        uint newline_pos = 4 * i + offset;
        uint temp_len = newline_pos - (semi_pos + 1);

        int sign = 1;
        uint i = semi_pos + 1;
        if (buf[i] == '-') {
            sign = -1;
            temp_len -= 1;
            ++i;
        }
        int temp;
        if (temp_len == 3) {
            temp = (buf[i] - '0') * 10 + (buf[i + 2] - '0');
        } else {
            temp = (buf[i] - '0') * 100 + (buf[i+1] - '0') * 10 + (buf[i+3] - '0');
        }
        res[j] = sign * temp;
    } 


}
