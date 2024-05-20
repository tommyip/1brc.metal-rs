use core::slice;
use std::{
    collections::HashMap,
    fs::File,
    hash::{BuildHasher, Hasher},
    ops::BitXor,
    sync::{atomic::AtomicUsize, Mutex},
    thread,
    time::Instant,
};

use memmap2::Mmap;
use metal::{objc::rc::autoreleasepool, Device, MTLSize};

use crate::{
    c_void, device_buffer, is_newline, metal_frame_capture, Station, Stations, STATION_NAMES,
    U32_SIZE,
};

type StationsHashMap<'a> = HashMap<&'a str, Station, BuildFxHash>;

const GPU_OFFLOAD_RATIO: f32 = 0.5;
const CHUNK_LEN: u64 = {
    let len = 2 * 1024;
    assert!(len % 16 == 0);
    len
};
const HASHMAP_LEN: usize = 512;
const HASHMAP_RAW_LEN: usize = HASHMAP_LEN * 4;

/// Slightly modified FxHash (from Rustc & Firefox)
///
/// Does not break down trailing bytes into 4, 2, 1 bytes
/// chunks unlike the original implementation.
struct FxHash {
    hash: u64,
}

impl FxHash {
    const K: u64 = 0x517cc1b727220a95;

    fn hash64(&mut self, x: u64) {
        self.hash = self.hash.rotate_left(5).bitxor(x).wrapping_mul(Self::K);
    }

    fn hash(s: &str) -> u64 {
        let mut h = FxHash { hash: 0 };
        h.write(s.as_bytes());
        h.finish()
    }
}

impl Hasher for FxHash {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let chunk = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            self.hash64(chunk);
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut buf = [0u8; 8];
            buf[..bytes.len()].copy_from_slice(bytes);
            self.hash64(u64::from_le_bytes(buf));
        }
    }
}

struct BuildFxHash;

impl BuildHasher for BuildFxHash {
    type Hasher = FxHash;

    fn build_hasher(&self) -> Self::Hasher {
        FxHash { hash: 0 }
    }
}

struct MPHStations<'a> {
    buckets: [i32; HASHMAP_RAW_LEN],
    fallback: StationsHashMap<'a>,
}

impl<'a> MPHStations<'a> {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|i| match i % 4 {
                0 => i32::MAX,
                1 => i32::MIN,
                _ => 0,
            }),
            fallback: StationsHashMap::with_hasher(BuildFxHash),
        }
    }

    fn merge(&mut self, other: MPHStations<'a>) {
        for (bucket, other_bucket) in self
            .buckets
            .chunks_exact_mut(4)
            .zip(other.buckets.chunks_exact(4))
        {
            if other_bucket[0] < bucket[0] {
                bucket[0] = other_bucket[0];
            }
            if other_bucket[1] > bucket[1] {
                bucket[1] = other_bucket[1];
            }
            bucket[2] += other_bucket[2];
            bucket[3] += other_bucket[3];
        }
        for (name, other_station) in other.fallback.into_iter() {
            if let Some(station) = self.fallback.get_mut(name) {
                station.merge(&other_station);
            } else {
                self.fallback.insert(name, other_station);
            }
        }
    }

    fn to_stations(mut self, params: &MPHParams) -> Stations<'a> {
        for (i, name) in STATION_NAMES.into_iter().enumerate() {
            let offset = params.indices[i] * 4;
            let min = self.buckets[offset];
            let max = self.buckets[offset + 1];
            let sum = self.buckets[offset + 2];
            let count = self.buckets[offset + 3];
            if let Some(station) = self.fallback.get_mut(name) {
                station.min = station.min.min(min);
                station.max = station.max.max(max);
                station.sum += sum;
                station.count += count;
            } else {
                self.fallback.insert(
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
        Stations {
            inner: self.fallback.into_iter().collect(),
        }
    }
}

#[repr(align(32))]
struct MPHParams {
    /// 32-byte aligned station names sorted using the perfect minimal hash index
    keys: [u8; HASHMAP_LEN * 32],
    /// Map station names to keys index
    indices: [usize; STATION_NAMES.len()],
}

impl MPHParams {
    fn init() -> Self {
        let mut this = MPHParams {
            keys: [0; HASHMAP_LEN * 32],
            indices: [0; STATION_NAMES.len()],
        };
        for (i, name) in STATION_NAMES.iter().enumerate() {
            let idx = Self::index(name);
            this.indices[i] = idx;
            this.keys[idx * 32..idx * 32 + name.len()].copy_from_slice(name.as_bytes());
        }
        this
    }

    /// Minimal perfect hash index of station names
    ///
    /// Constructed with PTRHash (https://github.com/RagnarGrootKoerkamp/PTRHash)
    /// Map all 314 station names uniquely to 512 buckets
    fn index(name: &str) -> usize {
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

        let hash = FxHash::hash(name);
        let is_large = hash >= P1;
        let rem = if is_large { REM_C2 } else { REM_C1 };
        let bucket = (is_large as isize * C3) + ((hash as u128 * rem as u128) >> 64) as isize;
        let pilot = PILOTS[bucket as usize];
        return ((FxHash::K as u128 * (hash ^ FxHash::K.wrapping_mul(pilot as u64)) as u128) >> 64)
            as usize
            & ((1 << 9) - 1) as usize;
    }
}

fn cpu_impl<'a>(buf: &'a [u8], stations: &mut StationsHashMap<'a>) {
    buf[..buf.len().saturating_sub(1)]
        .split(is_newline)
        .filter(|line| !line.is_empty())
        .for_each(|line| {
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

fn split_buf_once(buf: &[u8], max_len: usize) -> (&[u8], &[u8]) {
    if buf.len() > max_len {
        let mid = buf[..max_len].iter().rposition(|&c| c == b'\n').unwrap() + 1;
        (&buf[..mid], &buf[mid..])
    } else {
        (buf, &[])
    }
}

/// Our GPU concurrent hashmap uses Metal's
/// `atomic_compare_exchange_weak_explicit` instruction key (index of a station
/// name) lookup/insert, but it only supports up to 32 bits integer and not 64
/// bits integer which is necessary for fully indexing the measurements buffer
/// (~13GB). To work around this issue we split the file into chunks aligned to
/// line boundaries.
fn split_buf(mut buf: &[u8], max_len: usize) -> Vec<&[u8]> {
    let mut bufs = vec![];
    while !buf.is_empty() {
        let (head, tail) = split_buf_once(buf, max_len);
        buf = tail;
        bufs.push(head);
    }
    bufs
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

fn process_gpu<'a, F>(buf: &'a [u8], mph_params: &MPHParams, kernel_getter: F) -> MPHStations<'a>
where
    F: FnOnce(&metal::Device) -> metal::Function,
{
    let start = Instant::now();
    let split_bufs = split_buf(&buf, u32::MAX as usize);
    let chop_infos = split_bufs
        .iter()
        .map(|buf| chop_head_and_tail::<{ CHUNK_LEN as usize }>(buf))
        .collect::<Vec<_>>();

    let mut stations = MPHStations::new();

    let fallback = [false];

    autoreleasepool(|| {
        let device = &Device::system_default().expect("No Metal device found");

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

        let device_buckets = device_buffer(device, &stations.buckets);
        let device_mph_keys = device_buffer(device, &mph_params.keys);
        let device_fallback = device_buffer(device, &fallback);

        for (
            split_buf,
            &ChopInfo {
                n_threads,
                body_offset,
                ..
            },
        ) in split_bufs.iter().zip(&chop_infos)
        {
            let threads_per_grid = MTLSize::new(n_threads, 1, 1);
            let threads_per_threadgroup =
                MTLSize::new(pipeline_state.max_total_threads_per_threadgroup(), 1, 1);
            let device_split_buf = device_buffer(device, split_buf);

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
            cpu_impl(head, &mut stations.fallback);
            cpu_impl(tail, &mut stations.fallback);
        }

        cmd_buf.wait_until_completed();
    });

    assert!(!fallback[0], "Need fallback");

    eprintln!(
        "process_gpu elapsed={:.2}ms buf={}MB",
        start.elapsed().as_secs_f64() * 1000.,
        buf.len() / (1024 * 1024)
    );

    stations
}

fn process_cpu<'a>(buf: &'a [u8], mph_params: &MPHParams, offset: &AtomicUsize) -> MPHStations<'a> {
    let mut stations = MPHStations::new();
    let start = Instant::now();
    cpu_impl(buf, &mut stations.fallback);
    eprintln!(
        "process_cpu elapsed={:.2}ms buf={}",
        start.elapsed().as_secs_f64() * 1000.,
        buf.len()
    );
    stations
}

pub fn process<'a, F: Send>(file: &'a File, kernel_getter: F)
where
    F: FnOnce(&metal::Device) -> metal::Function,
{
    let buf = unsafe { &Mmap::map(file).unwrap() };
    let (gpu_buf, cpu_buf) = split_buf_once(buf, (buf.len() as f32 * GPU_OFFLOAD_RATIO) as usize);

    let n_cpus = thread::available_parallelism().unwrap().get() - 1;
    let cpu_chunk_size = cpu_buf.len().div_ceil(n_cpus);
    let cpu_chunks = split_buf(cpu_buf, cpu_chunk_size);
    let cpu_buf_offset = AtomicUsize::new(0);

    let stations = Mutex::new(MPHStations::new());
    let mph_params = MPHParams::init();

    thread::scope(|s| {
        s.spawn(|| {
            let local = process_gpu(gpu_buf, &mph_params, kernel_getter);
            stations.lock().unwrap().merge(local);
        });
        for chunk in cpu_chunks {
            s.spawn(|| {
                let local = process_cpu(chunk, &mph_params, &cpu_buf_offset);
                stations.lock().unwrap().merge(local);
            });
        }
    });

    println!(
        "{}",
        stations.into_inner().unwrap().to_stations(&mph_params)
    );
}
