//! # Optimization 0: Local histogram accumulation with threadgroup memory
//!
//! Process entire file on the GPU. Each Metal kernel handles a small chunk of
//! the measurements file, parsing it line by line and updating its parent
//! threadgroup local hashmap. After a threadgroup finishes processing its
//! mega-chunk the local hashmap is merged with the global hashmap. This
//! reduces expensive global memory access.
//!
//! Still no micro-optimization is applied.

use core::fmt;
use std::{
    collections::HashMap,
    env, ffi,
    fs::File,
    iter,
    mem::{self, size_of},
    ops::Range,
    path::PathBuf,
    u32,
};

use memmap2::Mmap;
use metal::{self, objc::rc::autoreleasepool, Device, MTLResourceOptions, MTLSize};

use one_billion_row::round_to_positive;

const CHUNK_LEN: u64 = 2 * 1024;
const MAX_NAMES: u64 = 10_000;
/// Target 50% load factor
const HASHMAP_LEN: u64 = MAX_NAMES * 2;
const HASHMAP_FIELDS: usize = 5;
const U64_SIZE: u64 = size_of::<u64>() as u64;
const I32_SIZE: u64 = size_of::<i32>() as u64;
const U32_SIZE: u64 = size_of::<u32>() as u64;

struct Station {
    min: i32,
    max: i32,
    sum: i32,
    count: i32,
}

#[derive(Default)]
struct Stations<'a> {
    inner: HashMap<&'a str, Station>,
}

impl fmt::Display for Station {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let min = self.min as f32 / 10.;
        let max = self.max as f32 / 10.;
        let mean = round_to_positive(((self.sum as f32 / 10.) / self.count as f32) * 10.) / 10.;
        f.write_fmt(format_args!("{:.1}/{:.1}/{:.1}", min, mean, max))
    }
}

impl<'a> fmt::Display for Stations<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut names = self.inner.keys().collect::<Vec<_>>();
        names.sort_unstable();
        for (i, name) in names.into_iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            let station = &self.inner[name];
            f.write_fmt(format_args!("{}={}", name, station))?;
        }
        f.write_str("}")
    }
}

fn device_buffer<T>(device: &metal::Device, buf: &[T]) -> metal::Buffer {
    device.new_buffer_with_bytes_no_copy(
        buf.as_ptr() as *const ffi::c_void,
        (buf.len() * size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    )
}

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

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
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
        let cmd_q = &device.new_command_queue();
        let cmd_buf = cmd_q.new_command_buffer();

        let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/opt0_threadgroup_memory/kernel.metallib");
        let library = device.new_library_with_file(library_path).unwrap();
        let kernel = library.get_function("histogram", None).unwrap();
        let pipeline_state = &device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        let device_buf = device_buffer(&device, &buf);
        let device_buckets = device_buffer(&device, &buckets);
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
