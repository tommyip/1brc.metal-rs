use std::{
    collections::HashMap,
    fs::File,
    iter,
    mem::{self},
    ops::Range,
};

use memmap2::Mmap;
use metal::{objc::rc::autoreleasepool, Device, MTLSize};

use crate::{
    c_void, device_buffer, is_newline, metal_frame_capture, Station, Stations, I32_SIZE, U32_SIZE,
    U64_SIZE,
};

const CHUNK_LEN: u64 = 2 * 1024;
const MAX_NAMES: u64 = 10_000;
/// Target 50% load factor
pub const HASHMAP_LEN: u64 = MAX_NAMES * 2;
const HASHMAP_FIELDS: usize = 5;

fn reconstruct_gpu_hashmap<'a>(
    buf: &'a [u8],
    stations: &mut HashMap<&'a str, Station>,
    buckets: &[i32],
) {
    for bucket in buckets.chunks_exact(HASHMAP_FIELDS) {
        let name_idx = unsafe { mem::transmute::<i32, u32>(bucket[0]) } as usize;
        if name_idx != 1 {
            let name_len = buf[name_idx..].iter().position(|c| *c == b';').unwrap();
            let name =
                unsafe { std::str::from_utf8_unchecked(&buf[name_idx..name_idx + name_len]) };
            let min = bucket[1];
            let max = bucket[2];
            let sum = bucket[3];
            let count = bucket[4];
            if let Some(station) = stations.get_mut(name) {
                station.min = station.min.min(min);
                station.max = station.max.max(max);
                station.sum += sum;
                station.count += count;
            } else {
                stations.insert(
                    name,
                    Station {
                        min,
                        max,
                        sum,
                        count,
                    },
                );
            }
        }
    }
}

fn cpu_impl<'a>(buf: &'a [u8], stations: &mut HashMap<&'a str, Station>) {
    buf[..buf.len() - 1].split(is_newline).for_each(|line| {
        let name_len = line.iter().position(|&c| c == b';').unwrap();
        let name = unsafe { std::str::from_utf8_unchecked(&line[..name_len]) };
        let mut sign = 1;
        let mut temp = 0;
        for &c in &line[name_len + 1..] {
            if c == b'-' {
                sign = -1;
            } else if c.is_ascii_digit() {
                temp = temp * 10 + (c - b'0') as i32;
            }
        }
        temp *= sign;
        if let Some(station) = stations.get_mut(name) {
            station.min = station.min.min(temp);
            station.max = station.max.max(temp);
            station.sum += temp;
            station.count += 1;
        } else {
            stations.insert(
                name,
                Station {
                    min: temp,
                    max: temp,
                    sum: temp,
                    count: 1,
                },
            );
        }
    });
}

/// Our GPU concurrent hashmap uses Metal's
/// `atomic_compare_exchange_weak_explicit` instruction key (index of a station
/// name) lookup/insert, but it only supports up to 32 bits integer and not 64
/// bits integer which is necessary for fully indexing the measurements buffer
/// (~13GB). To work around this issue we split the file into chunks aligned to
/// line boundaries.
fn split_aligned_measurements<const MAX_SIZE: usize>(buf: &[u8]) -> Vec<Range<usize>> {
    let mut ranges = Vec::with_capacity(buf.len().div_ceil(MAX_SIZE));
    let mut start = 0;
    loop {
        let end = start + MAX_SIZE;
        if end < buf.len() {
            let end = buf[..end].iter().rposition(|&c| c == b'\n').unwrap() + 1;
            ranges.push(start..end);
            start = end;
        } else {
            ranges.push(start..buf.len());
            break;
        };
    }
    ranges
}

struct ChopInfo<'a> {
    /// Exactly one line
    head: &'a [u8],
    /// At least one line
    tail: &'a [u8],
    /// How many GPU kernels to spawn to handle the buffer
    n_threads: u64,
}

/// Chop off the head and tail lines of the GPU buffer so each kernel can align
/// itself uniformly by finding the next newline at the boundaries. This also allows
/// the kernel to read a large block of buffer without accounting for out of bound
/// access.
fn chop_head_and_tail<'a, const CHUNK_LEN: usize, const MIN_TAIL_LEN: usize>(
    buf: &'a [u8],
) -> ChopInfo<'a> {
    let head_offset = buf.iter().position(is_newline).unwrap() + 1;
    let tail_offset = buf[..buf.len() - MIN_TAIL_LEN]
        .iter()
        .rposition(is_newline)
        .unwrap()
        + 1;
    let n_threads = (tail_offset / CHUNK_LEN) as u64
        // If the chunk end align exactly at the split boundary then the n-2 th
        // thread would have extended its bound to here.
        - (tail_offset % CHUNK_LEN == 0) as u64;
    ChopInfo {
        head: &buf[..head_offset],
        tail: &buf[tail_offset..],
        n_threads,
    }
}

pub fn process<'a, F>(file: &'a File, kernel_getter: F)
where
    F: FnOnce(&metal::Device) -> metal::Function,
{
    let buf = unsafe { &Mmap::map(file).unwrap() };
    let buf_len = buf.len() as u64;
    let split_ranges = split_aligned_measurements::<{ u32::MAX as usize }>(&buf);
    let chop_infos = split_ranges
        .iter()
        .map(|range| chop_head_and_tail::<{ CHUNK_LEN as usize }, 8>(&buf[range.clone()]))
        .collect::<Vec<_>>();

    // A connected list of hashmaps mapping station name (as offset)
    // to min, max, sum and count of temperatures. The name offset is represented
    // as (rust) i32 and (metal) atomic_int but actually contains a u32.
    let buckets = iter::repeat([1, i32::MAX, i32::MIN, 0, 0])
        .take(HASHMAP_LEN as usize * split_ranges.len())
        .flatten()
        .collect::<Vec<_>>();

    let mut stations = Stations::default();

    autoreleasepool(|| {
        let device = &Device::system_default().expect("No Metal device found");
        assert!(
            buf_len < device.max_buffer_length(),
            "Measurements file does not fit in a single metal buffer"
        );

        let _guard = metal_frame_capture(
            device,
            "/Users/thomas/repos/1brc.metal-rs/framecapture.gputrace",
        );

        let cmd_q = &device.new_command_queue();
        let cmd_buf = cmd_q.new_command_buffer();

        let kernel = kernel_getter(&device);
        let pipeline_state = &device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        let device_buf = device_buffer(device, &buf);
        let device_buckets = device_buffer(device, &buckets);

        for (i, (split_range, &ChopInfo { n_threads, .. })) in
            split_ranges.iter().zip(&chop_infos).enumerate()
        {
            let split_len = split_range.len() as u64;
            let threads_per_grid = MTLSize::new(n_threads, 1, 1);
            let threads_per_threadgroup =
                MTLSize::new(pipeline_state.max_total_threads_per_threadgroup(), 1, 1);
            let split_range_offset = split_range.start as u64 * U64_SIZE;
            let buckets_offset = HASHMAP_LEN * i as u64 * HASHMAP_FIELDS as u64 * I32_SIZE;

            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&device_buf), split_range_offset);
            encoder.set_buffer(1, Some(&device_buckets), buckets_offset);
            encoder.set_bytes(2, U32_SIZE, c_void(&(split_len as u32)));
            encoder.set_bytes(3, U32_SIZE, c_void(&(CHUNK_LEN as u32)));

            encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            encoder.end_encoding();
        }

        cmd_buf.commit();

        for ChopInfo { head, tail, .. } in chop_infos {
            cpu_impl(head, &mut stations.inner);
            cpu_impl(tail, &mut stations.inner);
        }

        cmd_buf.wait_until_completed();
    });

    for (i, split_range) in split_ranges.into_iter().enumerate() {
        let split_buf = &buf[split_range];
        let bucket_start = HASHMAP_LEN as usize * HASHMAP_FIELDS * i;
        let bucket_end = bucket_start + HASHMAP_LEN as usize * HASHMAP_FIELDS;
        let split_buckets = &buckets[bucket_start..bucket_end];
        reconstruct_gpu_hashmap(split_buf, &mut stations.inner, split_buckets);
    }

    println!("{}", stations);
}
