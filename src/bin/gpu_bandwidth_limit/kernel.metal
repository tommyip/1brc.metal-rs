#include <metal_stdlib>

using namespace metal;

kernel void blackbox(
    const device uchar* chunk,
    device uint64_t* res,
    const device uint64_t& chunk_len,
    const device uint64_t& thread_chunk_len,
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    const uint64_t start_idx = (uint64_t)gid * thread_chunk_len;
    const uint64_t cur_thread_chunk_len = gid < grid_size - 1 ?
        thread_chunk_len : chunk_len % thread_chunk_len;

    ulong4 simd_sum = ulong4(0);
    const uint64_t simd_len = cur_thread_chunk_len - cur_thread_chunk_len % 16;
    uint64_t i;
    for (i = 0; i < simd_len; i += 16) {
        uint64_t offset = start_idx + i;
        simd_sum += ulong4(chunk[offset], chunk[offset+1], chunk[offset+2], chunk[offset+3]);
        simd_sum += ulong4(chunk[offset+4], chunk[offset+5], chunk[offset+6], chunk[offset+7]);
        simd_sum += ulong4(chunk[offset+8], chunk[offset+9], chunk[offset+10], chunk[offset+11]);
        simd_sum += ulong4(chunk[offset+12], chunk[offset+13], chunk[offset+14], chunk[offset+15]);
    }
    uint64_t sum = simd_sum.w + simd_sum.x + simd_sum.y + simd_sum.z;
    for (; i < cur_thread_chunk_len; ++i) {
        sum += chunk[start_idx + i];
    }
    res[gid] = sum;
}
