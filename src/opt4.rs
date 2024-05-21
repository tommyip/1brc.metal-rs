use core::slice;
use std::{
    collections::HashMap,
    fs::File,
    hash::{BuildHasher, Hasher},
    ops::BitXor,
    simd::{cmp::SimdPartialEq, mask8x32, u64x4, u8x16, u8x32},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    thread,
    time::Instant,
};

use memmap2::Mmap;
use metal::{objc::rc::autoreleasepool, Device, MTLSize};

use crate::{
    c_void, device_buffer, is_newline, metal_frame_capture, Station, Stations, STATION_NAMES,
    U32_SIZE,
};

type StationsHashMap<'a> = HashMap<&'a [u8], Station, BuildFxHash>;

const GPU_OFFLOAD_RATIO: f32 = 0.5;
const GPU_CHUNK_LEN: u64 = {
    let len = 2 * 1024;
    assert!(len % 16 == 0);
    len
};
const CPU_CHUNK_LEN: usize = 16 * 1024;
const HASHMAP_LEN: usize = 512;

fn swar_parse_temp(temp: i64, dot_pos: usize) -> i32 {
    const MAGIC_MUL: i64 = 100 * 0x1000000 + 10 * 0x10000 + 1;
    let shift = 28 - dot_pos;
    let sign = (!temp << 59) >> 63;
    let minus_filter = !(sign & 0xFF);
    let digits = ((temp & minus_filter) << shift) & 0x0F000F0F00;
    let abs_value = (digits.wrapping_mul(MAGIC_MUL) as u64 >> 32) & 0x3FF;
    ((abs_value as i64 ^ sign) - sign) as i32
}

/// Slightly modified FxHash (from Rustc & Firefox)
///
/// Does not break down trailing bytes into 4, 2, 1 bytes
/// chunks unlike the original implementation.
#[derive(Default)]
struct FxHash {
    hash: u64,
}

impl FxHash {
    const K: u64 = 0x517cc1b727220a95;

    fn hash64(&mut self, x: u64) {
        self.hash = self.hash.rotate_left(5).bitxor(x).wrapping_mul(Self::K);
    }

    #[inline]
    fn hash(s: u8x32, len: usize) -> u64 {
        let mut h = FxHash::default();
        let chunks: u64x4 = unsafe { std::mem::transmute(s) };
        for bytes in chunks.to_array().into_iter().take(len.div_ceil(8)) {
            h.hash64(bytes);
        }
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
    buckets: [Station; HASHMAP_LEN],
    fallback: StationsHashMap<'a>,
    params: &'a MPHParams,
}

impl<'a> MPHStations<'a> {
    fn new(params: &'a MPHParams) -> Self {
        Self {
            buckets: [Station::default(); HASHMAP_LEN],
            fallback: StationsHashMap::with_hasher(BuildFxHash),
            params,
        }
    }

    fn merge(&mut self, other: MPHStations<'a>) {
        for (bucket, other_bucket) in self.buckets.iter_mut().zip(other.buckets) {
            bucket.merge(&other_bucket);
        }
        for (name, other_station) in other.fallback {
            self.fallback
                .entry(name)
                .and_modify(|station| station.merge(&other_station))
                .or_insert(other_station);
        }
    }

    fn to_stations(mut self) -> Stations<'a> {
        for (i, name) in STATION_NAMES.into_iter().enumerate() {
            let bucket_station = self.buckets[self.params.indices[i]];
            self.fallback
                .entry(name.as_bytes())
                .and_modify(|station| station.merge(&bucket_station))
                .or_insert(bucket_station);
        }
        Stations {
            inner: self
                .fallback
                .into_iter()
                .map(|(k, v)| (unsafe { std::str::from_utf8_unchecked(k) }, v))
                .collect(),
        }
    }
}

struct MPHParams {
    /// 32-byte aligned station names sorted using the perfect minimal hash index
    keys: [u8x32; HASHMAP_LEN],
    /// Map station names to keys index
    indices: [usize; STATION_NAMES.len()],
}

impl MPHParams {
    fn init() -> Self {
        let mut this = MPHParams {
            keys: [u8x32::splat(0); HASHMAP_LEN],
            indices: [0; STATION_NAMES.len()],
        };
        for (i, name) in STATION_NAMES.iter().enumerate() {
            let name_simd = u8x32::load_or_default(name.as_bytes());
            let idx = Self::index(name_simd, name.len());
            this.indices[i] = idx;
            this.keys[idx] = name_simd;
        }
        this
    }

    /// Minimal perfect hash index of station names
    ///
    /// Constructed with PTRHash (https://github.com/RagnarGrootKoerkamp/PTRHash)
    /// Map all 314 station names uniquely to 512 buckets
    #[inline]
    fn index(name: u8x32, len: usize) -> usize {
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

        let hash = FxHash::hash(name, len);
        let is_large = hash >= P1;
        let rem = if is_large { REM_C2 } else { REM_C1 };
        let bucket = (is_large as isize * C3) + ((hash as u128 * rem as u128) >> 64) as isize;
        let pilot = PILOTS[bucket as usize];
        return ((FxHash::K as u128 * (hash ^ FxHash::K.wrapping_mul(pilot as u64)) as u128) >> 64)
            as usize
            & ((1 << 9) - 1) as usize;
    }
}

/// Minimal CPU implementation without any bells or whistle
fn process_cpu_naive<'a>(buf: &'a [u8], stations: &mut StationsHashMap<'a>) {
    buf[..buf.len().saturating_sub(1)]
        .split(is_newline)
        .filter(|line| !line.is_empty())
        .for_each(|line| {
            let name_len = line.iter().position(|&c| c == b';').unwrap();
            let (name, rest) = line.split_at(name_len);
            let (sign, temp) = if rest[1] == b'-' {
                (-1, &rest[2..])
            } else {
                (1, &rest[1..])
            };
            let temp = sign
                * match temp {
                    [b, b'.', c] => 10 * (b - b'0') as i32 + (c - b'0') as i32,
                    [a, b, b'.', c] => {
                        100 * (a - b'0') as i32 + 10 * (b - b'0') as i32 + (c - b'0') as i32
                    }
                    _ => unreachable!(),
                };
            stations
                .entry(name)
                .and_modify(|station| station.update(temp))
                .or_insert(Station::new(temp));
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
    body_offset: usize,
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
fn chop_head_and_tail<'a, const CHUNK_LEN: usize, const EXCESS: usize>(
    buf: &'a [u8],
) -> ChopInfo<'a> {
    let (prefix, middle) = unsafe {
        let (prefix, middle, _) = buf.align_to::<u128>();
        let middle = slice::from_raw_parts(
            (middle.as_ptr() as *const u128) as *const u8,
            middle.len() * 16,
        );
        (prefix, middle)
    };
    let body_offset = prefix.len();
    let head_offset = body_offset + middle.iter().position(is_newline).unwrap() + 1;
    let n_threads = (buf.len() - EXCESS - body_offset) / CHUNK_LEN;
    let unaligned_tail_offset = body_offset + n_threads * CHUNK_LEN;
    let tail_offset = unaligned_tail_offset
        + buf[unaligned_tail_offset..]
            .iter()
            .position(is_newline)
            .unwrap()
        + 1;

    ChopInfo {
        head: &buf[..head_offset],
        body_offset,
        tail: &buf[tail_offset..],
        n_threads: n_threads as u64,
    }
}

fn process_gpu<'a, F>(buf: &'a [u8], mph_params: &'a MPHParams, kernel_getter: F) -> MPHStations<'a>
where
    F: FnOnce(&metal::Device) -> metal::Function,
{
    let start = Instant::now();
    let split_bufs = split_buf(&buf, u32::MAX as usize);
    let chop_infos = split_bufs
        .iter()
        .map(|buf| chop_head_and_tail::<{ GPU_CHUNK_LEN as usize }, 16>(buf))
        .collect::<Vec<_>>();

    let mut stations = MPHStations::new(mph_params);

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
            encoder.set_buffer(0, Some(&device_split_buf), body_offset as u64);
            encoder.set_buffer(1, Some(&device_mph_keys), 0);
            encoder.set_buffer(2, Some(&device_buckets), 0);
            encoder.set_bytes(3, U32_SIZE, c_void(&(GPU_CHUNK_LEN as u32)));
            encoder.set_buffer(4, Some(&device_fallback), 0);

            encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            encoder.end_encoding();
        }

        cmd_buf.commit();

        for ChopInfo { head, tail, .. } in chop_infos {
            process_cpu_naive(head, &mut stations.fallback);
            process_cpu_naive(tail, &mut stations.fallback);
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

fn process_cpu<'a>(
    buf: &'a [u8],
    mph_params: &'a MPHParams,
    offset: &AtomicUsize,
    end: usize,
) -> MPHStations<'a> {
    let mut stations = MPHStations::new(mph_params);
    let start = Instant::now();

    loop {
        let mut i = offset.fetch_add(CPU_CHUNK_LEN, Ordering::Relaxed);
        if i >= end {
            break;
        }
        let chunk_end = (i + CPU_CHUNK_LEN + 1).min(end);

        let mut newline_bytes = u8x16::from_slice(&buf[i..]);
        loop {
            let newline_mask = newline_bytes.simd_eq(u8x16::splat(b'\n'));
            if newline_mask.any() {
                i += (newline_mask.to_bitmask().trailing_zeros() + 1) as usize;
                break;
            }
            i += u8x16::LEN;
            newline_bytes = u8x16::from_slice(&buf[i..]);
        }

        while i < chunk_end {
            let mut name_prefix = u8x32::from_slice(&buf[i..]);
            let semi_mask = name_prefix.simd_eq(u8x32::splat(b';'));
            let station = if let Some(semi_idx) = semi_mask.first_set() {
                i += semi_idx + 1;
                let name_mask = mask8x32::from_bitmask(!(u64::MAX << semi_idx));
                name_prefix = name_mask.select(name_prefix, u8x32::splat(0));
                let mph_idx = MPHParams::index(name_prefix, semi_idx);
                if name_prefix == mph_params.keys[mph_idx] {
                    &mut stations.buckets[mph_idx]
                } else {
                    unreachable!("fallback path");
                }
            } else {
                unreachable!("fallback path");
            };
            let mut temp_buf = [0u8; 8];
            temp_buf.copy_from_slice(&buf[i..i + 8]);
            let temp_bytes = i64::from_le_bytes(temp_buf);
            let dot_pos = (!temp_bytes & 0x10101000).trailing_zeros() as usize;
            i += (dot_pos >> 3) + 3;
            let temp = swar_parse_temp(temp_bytes, dot_pos);
            station.update(temp);
        }
    }

    eprintln!(
        "process_cpu elapsed={:.2}ms",
        start.elapsed().as_secs_f64() * 1000.,
    );
    assert!(stations.fallback.is_empty());
    stations
}

pub fn process<'a, F: Send>(file: &'a File, kernel_getter: F)
where
    F: FnOnce(&metal::Device) -> metal::Function,
{
    let mph_params = MPHParams::init();
    let mut stations = MPHStations::new(&mph_params);

    let buf = unsafe { &Mmap::map(file).unwrap() };
    let (gpu_buf, cpu_buf) = split_buf_once(buf, (buf.len() as f32 * GPU_OFFLOAD_RATIO) as usize);
    // let cpu_buf = buf;

    let (cpu_buf, cpu_buf_offset, cpu_buf_end) = {
        let ChopInfo {
            body_offset,
            head,
            tail,
            ..
        } = chop_head_and_tail::<1, 32>(cpu_buf);
        process_cpu_naive(head, &mut stations.fallback);
        process_cpu_naive(tail, &mut stations.fallback);
        let cpu_buf_offset = AtomicUsize::new(body_offset);
        let cpu_buf_end = cpu_buf.len() - body_offset - tail.len();
        (&cpu_buf[body_offset..], cpu_buf_offset, cpu_buf_end)
    };
    let n_cpus = thread::available_parallelism().unwrap().get() - 1;
    // let n_cpus = 1;

    let stations = Mutex::new(stations);

    thread::scope(|s| {
        s.spawn(|| {
            let local = process_gpu(gpu_buf, &mph_params, kernel_getter);
            stations.lock().unwrap().merge(local);
        });
        for _ in 0..n_cpus {
            s.spawn(|| {
                let local = process_cpu(cpu_buf, &mph_params, &cpu_buf_offset, cpu_buf_end);
                stations.lock().unwrap().merge(local);
            });
        }
    });

    println!("{}", stations.into_inner().unwrap().to_stations());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_simd() {
        const S: &str = "Hello, world!";

        let expected = {
            let mut ref_hasher = FxHash::default();
            ref_hasher.write(S.as_bytes());
            ref_hasher.finish()
        };

        let actual = {
            let s_simd = u8x32::load_or_default(S.as_bytes());
            FxHash::hash(s_simd, S.len())
        };

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_swar_parse_temp() {
        fn parse(s: &str) -> i32 {
            let mut buf = [0u8; 8];
            buf[..s.len()].copy_from_slice(s.as_bytes());
            let bytes = i64::from_le_bytes(buf);
            let dot_pos = (!bytes & 0x10101000).trailing_zeros();
            swar_parse_temp(bytes, dot_pos as usize)
        }

        assert_eq!(parse("0.0"), 0);
        assert_eq!(parse("1.0"), 10);
        assert_eq!(parse("1.2"), 12);
        assert_eq!(parse("-1.2"), -12);
        assert_eq!(parse("12.3"), 123);
        assert_eq!(parse("-12.3"), -123);
    }
}
