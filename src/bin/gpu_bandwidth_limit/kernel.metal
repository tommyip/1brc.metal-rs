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

    uint64_t sum = 0;
    for (uint64_t i = 0; i < cur_thread_chunk_len; ++i) {
        sum += chunk[start_idx + i];
    }
    res[gid] = sum;
}
