#include <metal_stdlib>

using namespace metal;

kernel void histogram(
    const device uchar* cchunk,
    device uint64_t* res,
    const device uint64_t& cchunk_len,
    const device uint64_t& mchunk_len,
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    const uint64_t raw_start_idx = (uint64_t)gid * mchunk_len;
    uint64_t start_idx = raw_start_idx;
    // Shift start to after the first newline
    if (gid > 0) {
        while (cchunk[start_idx++] != '\n');
    }

    uint64_t end_idx = raw_start_idx + mchunk_len;
    // Shift end to after the first newline following the current mchunk
    if (gid < grid_size - 1 || cchunk_len % mchunk_len != 0) {
        while (cchunk[end_idx++] != '\n');
    }

    res[gid] = end_idx - start_idx;
}
