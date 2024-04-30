use std::{
    collections::HashMap,
    ffi,
    fs::File,
    iter,
    mem::{self},
    ops::Range,
    path::Path,
};

use memmap2::Mmap;
use metal::{objc::rc::autoreleasepool, Device, MTLSize};

use crate::{device_buffer, metal_frame_capture, Station, Stations, I32_SIZE, U32_SIZE, U64_SIZE};

const CHUNK_LEN: u64 = 2 * 1024;
const MAX_NAMES: u64 = 10_000;
/// Target 50% load factor
const HASHMAP_LEN: u64 = MAX_NAMES * 2;
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
            stations.insert(
                name,
                Station {
                    min: bucket[1],
                    max: bucket[2],
                    sum: bucket[3],
                    count: bucket[4],
                },
            );
        }
    }
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

pub fn baseline<'a>(file: &'a File, metallib: &Path) {
    let buf = unsafe { &Mmap::map(file).unwrap() };
    let buf_len = buf.len() as u64;
    let split_ranges = split_aligned_measurements::<{ u32::MAX as usize }>(&buf);

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

        let library = device.new_library_with_file(metallib).unwrap();
        let kernel = library.get_function("histogram", None).unwrap();
        let pipeline_state = &device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        let device_buf = device_buffer(device, &buf);
        let device_buckets = device_buffer(device, &buckets);
        let chunk_len_ptr = (&(CHUNK_LEN as u32) as *const u32) as *const ffi::c_void;

        for (i, split_range) in split_ranges.iter().enumerate() {
            let split_len = split_range.len() as u64;
            let split_len_ptr = (&(split_len as u32) as *const u32) as *const ffi::c_void;
            let n_threads = split_len / CHUNK_LEN - (split_len % CHUNK_LEN == 0) as u64;
            let threads_per_grid = MTLSize::new(n_threads, 1, 1);
            let threads_per_threadgroup =
                MTLSize::new(pipeline_state.max_total_threads_per_threadgroup(), 1, 1);
            let split_range_offset = split_range.start as u64 * U64_SIZE;
            let buckets_offset = HASHMAP_LEN * i as u64 * HASHMAP_FIELDS as u64 * I32_SIZE;

            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&device_buf), split_range_offset);
            encoder.set_buffer(1, Some(&device_buckets), buckets_offset);
            encoder.set_bytes(2, U32_SIZE, split_len_ptr);
            encoder.set_bytes(3, U32_SIZE, chunk_len_ptr);

            encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            encoder.end_encoding();
        }

        cmd_buf.commit();
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
