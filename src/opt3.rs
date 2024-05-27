use core::slice;
use std::{
    collections::HashMap,
    fs::File,
    ops::{BitXor, Range},
};

use memmap2::Mmap;
use metal::{objc::rc::autoreleasepool, Device, MTLSize};

use crate::{
    c_void, device_buffer, is_newline, metal_frame_capture, Station, Stations, STATION_NAMES,
    U32_SIZE,
};

const CHUNK_LEN: u64 = {
    let len = 2 * 1024;
    assert!(len % 16 == 0);
    len
};
const HASHMAP_LEN: u64 = 512;
const HASHMAP_RAW_LEN: usize = HASHMAP_LEN as usize * 4;

#[repr(align(32))]
struct MPHStationNames {
    /// 32-byte aligned station names sorted using the perfect minimal hash index
    keys: [u8; 512 * 32],
    /// Map position in `keys` to index in the alphabetically sorted station name array
    indices: [usize; 512],
}

impl MPHStationNames {
    fn init() -> Self {
        let mut this = MPHStationNames {
            keys: [0; 512 * 32],
            indices: [0; 512],
        };
        for (i, name) in STATION_NAMES.iter().enumerate() {
            let idx = Self::mph_index(name);
            this.indices[idx] = i;
            this.keys[idx * 32..idx * 32 + name.len()].copy_from_slice(name.as_bytes());
        }
        this
    }

    /// Slightly modified FxHash (from Rustc & Firefox)
    ///
    /// Does not break down trailing bytes into 4, 2, 1 bytes
    /// chunks unlike the original implementation.
    fn fxhash(s: &str) -> u64 {
        const K: u64 = 0x517cc1b727220a95;

        let mut h: u64 = 0;
        for chunk in s.as_bytes().chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            let i = u64::from_le_bytes(buf);
            h = h.rotate_left(5).bitxor(i).wrapping_mul(K);
        }
        h
    }

    /// Minimal perfect hash index of station names
    ///
    /// Constructed with PTRHash (https://github.com/RagnarGrootKoerkamp/PTRHash)
    /// Map all 314 station names uniquely to 512 buckets
    fn mph_index(name: &str) -> usize {
        const C: u64 = 0x517cc1b727220a95;
        const PILOTS: [u8; 78] = [
            50, 110, 71, 13, 18, 27, 21, 14, 10, 16, 1, 14, 6, 11, 0, 2, 17, 2, 1, 4, 68, 79, 21,
            0, 22, 20, 60, 12, 30, 53, 62, 78, 27, 17, 2, 17, 13, 43, 21, 108, 19, 12, 25, 1, 55,
            36, 1, 0, 4, 184, 0, 21, 69, 25, 13, 177, 11, 97, 3, 29, 14, 104, 30, 4, 50, 23, 6,
            102, 137, 10, 227, 32, 29, 21, 7, 4, 244, 0,
        ];
        const REM_C1: u64 = 38;
        const REM_C2: u64 = 132;
        const C3: isize = -55;
        const P1: u64 = 11068046444225730560;

        let hash = Self::fxhash(name);
        let is_large = hash >= P1;
        let rem = if is_large { REM_C2 } else { REM_C1 };
        let bucket = (is_large as isize * C3) + ((hash as u128 * rem as u128) >> 64) as isize;
        let pilot = PILOTS[bucket as usize];
        return ((C as u128 * (hash ^ C.wrapping_mul(pilot as u64)) as u128) >> 64) as usize
            & ((1 << 9) - 1) as usize;
    }
}

fn cpu_impl<'a>(buf: &'a [u8], stations: &mut HashMap<&'a [u8], Station>) {
    buf[..buf.len() - 1].split(is_newline).for_each(|line| {
        let name_len = line.iter().position(|&c| c == b';').unwrap();
        let name = &line[..name_len];
        let mut sign = 1;
        let mut temp = 0;
        for &c in &line[name_len + 1..] {
            if c == b'-' {
                sign = -1;
            } else if c.is_ascii_digit() {
                temp = temp * 10 + (c - b'0') as i16;
            }
        }
        temp *= sign;
        if let Some(station) = stations.get_mut(name) {
            station.min = station.min.min(temp);
            station.max = station.max.max(temp);
            station.sum += temp as i32;
            station.count += 1;
        } else {
            stations.insert(
                name,
                Station {
                    min: temp,
                    max: temp,
                    sum: temp as i32,
                    count: 1,
                },
            );
        }
    });
}

fn insert_gpu_hashmap(
    buckets: &[i32; HASHMAP_RAW_LEN],
    mph: &MPHStationNames,
    stations: &mut HashMap<&[u8], Station>,
) {
    for (&sorted_idx, bucket) in mph.indices.iter().zip(buckets.chunks_exact(4)) {
        let name = STATION_NAMES[sorted_idx].as_bytes();
        let (min, max, sum, count) = (
            bucket[0] as i16,
            bucket[1] as i16,
            bucket[2],
            bucket[3] as u32,
        );
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
    /// 16-byte aligned offset of GPU chunk
    body_offset: u64,
    /// Contain complete lines
    head: &'a [u8],
    /// Contain complete lines
    tail: &'a [u8],
    /// How many GPU kernels to spawn to handle the buffer
    n_threads: u64,
}

/// Chop off the head and tail lines of the GPU buffer. Chopping off the head is needed
/// for aligning our buffer to 16 bytes multiplies. Chopping off the tail allow us
/// to read 16 bytes at a time without going out of bounds at the end.
fn chop_head_and_tail<'a, const CHUNK_LEN: usize>(buf: &'a [u8]) -> ChopInfo<'a> {
    let (prefix, middle) = unsafe {
        let (prefix, middle, _) = buf.align_to::<u128>();
        let middle = slice::from_raw_parts(
            (middle.as_ptr() as *const u128) as *const u8,
            middle.len() * 16,
        );
        (prefix, middle)
    };
    let body_offset = prefix.len();
    let head_offset = middle.iter().position(is_newline).unwrap() + 1;
    let n_threads = (buf.len() - 16 - body_offset) / CHUNK_LEN;
    let unaligned_tail_offset = body_offset + n_threads * CHUNK_LEN;
    let tail_offset = unaligned_tail_offset
        + buf[unaligned_tail_offset..]
            .iter()
            .position(is_newline)
            .unwrap()
        + 1;

    ChopInfo {
        head: &buf[..head_offset],
        body_offset: body_offset as u64,
        tail: &buf[tail_offset..],
        n_threads: n_threads as u64,
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
        .map(|range| chop_head_and_tail::<{ CHUNK_LEN as usize }>(&buf[range.clone()]))
        .collect::<Vec<_>>();

    let buckets: [i32; HASHMAP_RAW_LEN] = std::array::from_fn(|i| match i % 4 {
        0 => i32::MAX,
        1 => i32::MIN,
        _ => 0,
    });
    let mut stations = Stations::default();
    let mph = MPHStationNames::init();

    let fallback = [false];

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

        let device_buckets = device_buffer(device, &buckets);
        let device_mph_keys = device_buffer(device, &mph.keys);
        let device_fallback = device_buffer(device, &fallback);

        for (
            split_range,
            &ChopInfo {
                n_threads,
                body_offset,
                ..
            },
        ) in split_ranges.iter().zip(&chop_infos)
        {
            let threads_per_grid = MTLSize::new(n_threads, 1, 1);
            let threads_per_threadgroup =
                MTLSize::new(pipeline_state.max_total_threads_per_threadgroup(), 1, 1);
            let device_split_buf = device_buffer(device, &buf[split_range.clone()]);

            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&device_split_buf), body_offset);
            encoder.set_buffer(1, Some(&device_mph_keys), 0);
            encoder.set_buffer(2, Some(&device_buckets), 0);
            encoder.set_bytes(3, U32_SIZE, c_void(&(CHUNK_LEN as u32)));
            encoder.set_buffer(4, Some(&device_fallback), 0);

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

    assert!(!fallback[0], "Need fallback");

    insert_gpu_hashmap(&buckets, &mph, &mut stations.inner);

    println!("{}", stations);
}
