#![allow(dead_code)]
#![feature(portable_simd)]

use std::{collections::HashSet, fmt, mem::transmute, ops::BitXor};

use one_billion_row::STATION_NAMES;
use ptr_hash::{hash::Hasher, DefaultPtrHash, PtrHash, PtrHashParams};

fn name_len_stats() {
    let n_gt_8 = STATION_NAMES.iter().filter(|name| name.len() > 8).count();
    let n_gt_15 = STATION_NAMES.iter().filter(|name| name.len() > 15).count();
    let n_gt_16 = STATION_NAMES.iter().filter(|name| name.len() > 16).count();
    let max_len = STATION_NAMES.iter().map(|name| name.len()).max().unwrap();
    let unicode_prefix = STATION_NAMES
        .iter()
        .filter(|name| name.as_bytes()[0] >> 7 == 1)
        .collect::<Vec<_>>();
    println!(
        "<=8={} <=15={} >15={} <=16={} >16={} max={}",
        STATION_NAMES.len() - n_gt_8,
        STATION_NAMES.len() - n_gt_15,
        n_gt_15,
        STATION_NAMES.len() - n_gt_16,
        n_gt_16,
        max_len,
    );
    println!("names with unicode prefix: {:?}", unicode_prefix);
    for len in 1..32 {
        let uniques = STATION_NAMES
            .map(|name| &name.as_bytes()[..name.len().min(len)])
            .into_iter()
            .collect::<HashSet<_>>();
        if uniques.len() == STATION_NAMES.len() {
            println!("Min unique len {}", len);
            break;
        }
    }
    let hash_u16 = STATION_NAMES
        .map(|name| {
            let mut buf = [0u8; 8];
            let len = name.len().min(8);
            buf[..len].copy_from_slice(&name.as_bytes()[..len]);
            let x0 = u32::from_le_bytes(*unsafe { transmute::<&[u8; 8], &[u8; 4]>(&buf) });
            let x1 = u32::from_le_bytes(*unsafe {
                transmute::<*const u8, &[u8; 4]>(buf.as_ptr().add(4))
            });
            x0 ^ x1 ^ name.len() as u32
        })
        .into_iter()
        .collect::<HashSet<_>>();
    println!("u32 hash {}/{}", hash_u16.len(), STATION_NAMES.len());
}

fn min_prefix() {
    for len in 1.. {
        let n_unique_prefix = STATION_NAMES
            .iter()
            .map(|name| {
                let name = name.as_bytes();
                &name[..name.len().min(len)]
            })
            .collect::<HashSet<_>>()
            .len();
        if n_unique_prefix == STATION_NAMES.len() {
            println!("Minimimum name prefix name: {}", len);
            break;
        }
    }
}

#[derive(Clone)]
pub struct FxHash;

impl FxHash {
    const K: u64 = 0x517cc1b727220a95;
}

impl Hasher<&str> for FxHash {
    type H = u64;

    fn hash(x: &&str, _seed: u64) -> Self::H {
        let mut h: u64 = 0;
        for chunk in x.as_bytes().chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            let i = u64::from_le_bytes(buf);
            h = h.rotate_left(5).bitxor(i).wrapping_mul(Self::K);
        }
        h
    }
}

#[derive(Clone)]
struct AESHasher;

impl Hasher<&str> for AESHasher {
    type H = u64;

    fn hash(x: &&str, _seed: u64) -> Self::H {
        use std::arch::aarch64::*;
        let mut buf = [0u8; 32];
        buf[..x.len()].copy_from_slice(x.as_bytes());
        unsafe {
            let data = vld1q_u8(buf.as_ptr());
            let key = vld1q_u8(buf.as_ptr().add(8));
            let aes = vaeseq_u8(data, key);
            let aes_u64x2 = vreinterpretq_u64_u8(aes);
            let aes_lo = vgetq_lane_u64(aes_u64x2, 0) as u64;
            let aes_hi = vgetq_lane_u64(aes_u64x2, 1) as u64;
            aes_lo.wrapping_mul(aes_hi)
        }
    }
}

#[derive(Clone)]
struct CCHasher;

impl Hasher<&str> for CCHasher {
    type H = u64;

    fn hash(x: &&str, _seed: u64) -> Self::H {
        let mut chunk =
            u64::from_le_bytes(unsafe { *transmute::<*const u8, &[u8; 8]>(x.as_ptr()) });
        if x.len() < 8 {
            let mask = (1 << (x.len() << 3)) - 1;
            chunk &= mask;
        }
        (chunk ^ x.len() as u64).wrapping_mul(FxHash::K)
    }
}

struct Stats {
    min: u32,
    max: u32,
    avg: f32,
    occupied: usize,
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "min: {}, max: {}, avg: {}, occupied: {}",
            self.min, self.max, self.avg, self.occupied
        ))
    }
}

fn statistics<F>(hash_fn: F, hashmap_len: usize) -> Stats
where
    F: Fn(&[u8]) -> u64,
{
    let mut load = vec![0u32; hashmap_len];
    for name in STATION_NAMES {
        let hash = hash_fn(name.as_bytes());
        let idx = hash as usize % hashmap_len;
        load[idx] += 1;
    }
    let occupied_load = load
        .iter()
        .filter(|&&x| x != 0)
        .map(|&x| x)
        .collect::<Vec<_>>();
    let max = *occupied_load.iter().max().unwrap();
    let min = *occupied_load.iter().min().unwrap();
    let avg = occupied_load.iter().sum::<u32>() as f32 / occupied_load.len() as f32;
    Stats {
        max,
        min,
        avg,
        occupied: occupied_load.len(),
    }
}

fn ptrhash(hash: u64) -> usize {
    const C: u64 = 0x517cc1b727220a95;
    let pilots: Vec<u8> = vec![
        0, 21, 32, 6, 13, 20, 13, 4, 2, 0, 47, 6, 3, 2, 4, 28, 1, 17, 24, 31, 23, 15, 37, 0, 42,
        39, 22, 1, 1, 24, 0, 6, 59, 0, 0, 30, 10, 46, 0, 7, 38, 28, 46, 23, 60, 14, 12, 23, 0, 56,
        43, 15, 165, 30, 138, 46, 25, 14, 180, 0, 42, 72, 6, 99, 46, 0, 125, 39, 75, 94, 70, 47,
        245, 7, 55, 41, 237, 0,
    ];
    let rem_c1: u64 = 38;
    let rem_c2: u64 = 132;
    let c3 = -55;
    let p1 = 11068046444225730560;
    let is_large = hash >= p1;
    let rem = if is_large { rem_c2 } else { rem_c1 };
    let bucket = (is_large as isize * c3) + ((hash as u128 * rem as u128) >> 64) as isize;
    let pilot = pilots[bucket as usize];
    return ((C as u128 * (hash ^ C.wrapping_mul(pilot as u64)) as u128) >> 64) as usize
        & ((1 << 9) - 1) as usize;
}

fn main() {
    // let params = PtrHashParams {
    //     alpha: 0.9,
    //     c: 1.5,
    //     slots_per_part: STATION_NAMES.len() * 2,
    //     ..Default::default()
    // };
    // let mphf: PtrHash<&str, ptr_hash::local_ef::LocalEf, CCHasher> =
    //     DefaultPtrHash::new(&STATION_NAMES, params);

    // println!("{}", mphf.index(&"Hong Kong"));
    // ptrhash(FxHash::hash(&"Hong Kong", 0));

    // let mut taken = vec![false; 512];
    // for name in STATION_NAMES {
    //     let hash = CCHasher::hash(&name, 0);
    //     let ref_idx = ptrhash(hash);
    //     // let ref_idx = mphf.index(&name);
    //     // let idx = ptrhash(FxHash::hash(&name, 0));
    //     // assert_eq!(ref_idx, idx);
    //     assert!(!taken[ref_idx]);
    //     taken[ref_idx] = true;
    // }

    name_len_stats();
}
