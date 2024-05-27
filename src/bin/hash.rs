#![allow(dead_code)]
#![feature(portable_simd)]

use std::{collections::HashSet, fmt, ops::BitXor};

use one_billion_row::STATION_NAMES;
use ptr_hash::hash::Hasher;

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

fn djbx33a(s: &[u8]) -> u64 {
    s.iter()
        .fold(5381, |h, c| h.wrapping_mul(33).wrapping_add(*c as u64))
}

fn djbx33a_x4(s: &[u8]) -> u64 {
    let mut chunks = s.chunks_exact(4);
    let mut h = [5381u64, 5381, 5381, 5381];
    while let Some(chunk) = chunks.next() {
        for i in 0..4 {
            h[i] = h[i].wrapping_mul(33).wrapping_add(chunk[i] as u64);
        }
    }
    for (i, &c) in chunks.remainder().iter().enumerate() {
        h[i] = h[i].wrapping_mul(33).wrapping_add(c as u64);
    }
    h[0] ^ h[1] ^ h[2] ^ h[3]
}

fn djbx33a_u64(s: &[u8]) -> u64 {
    let mut h: u64 = 5381;
    for chunk in s.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        h = h.wrapping_mul(33).wrapping_add(u64::from_le_bytes(buf));
    }
    h
}

fn djbx33a_u64_parametric(s: &[u8], seed: u64, mul: u64) -> u64 {
    let mut h = seed;
    for chunk in s.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        h = h.wrapping_mul(mul).wrapping_add(u64::from_le_bytes(buf));
    }
    h
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
        50, 110, 71, 13, 18, 27, 21, 14, 10, 16, 1, 14, 6, 11, 0, 2, 17, 2, 1, 4, 68, 79, 21, 0,
        22, 20, 60, 12, 30, 53, 62, 78, 27, 17, 2, 17, 13, 43, 21, 108, 19, 12, 25, 1, 55, 36, 1,
        0, 4, 184, 0, 21, 69, 25, 13, 177, 11, 97, 3, 29, 14, 104, 30, 4, 50, 23, 6, 102, 137, 10,
        227, 32, 29, 21, 7, 4, 244, 0,
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

#[derive(Clone)]
enum Key<'a> {
    Inline([u8; 16]),
    Str(&'a [u8]),
}

fn main() {
    println!(
        "{} {}",
        std::mem::size_of::<Key<'_>>(),
        std::mem::size_of::<Option<Key<'_>>>()
    );

    // let buf = u8x16::from_slice(b"abcdefghixxxxxxx;hijklidsfjakldsfjkadsfd");
    // let mask = buf.simd_eq(u8x16::splat(b';'));
    // if let Some(name_len) = mask.first_set() {
    //     let word: &u128 = unsafe { std::mem::transmute(&buf) };
    //     let shift = (16 - name_len) * 8;
    //     let word_masked = (word << shift) >> shift;
    //     let masked_buf = word_masked.to_ne_bytes();
    //     println!("{}", std::str::from_utf8(&masked_buf).unwrap());
    // }

    // let params = PtrHashParams {
    //     alpha: 0.9,
    //     c: 1.5,
    //     slots_per_part: STATION_NAMES.len() * 2,
    //     ..Default::default()
    // };
    // let mphf: PtrHash<&str, ptr_hash::local_ef::LocalEf, FxHash> =
    //     DefaultPtrHash::new(&STATION_NAMES, params);

    // println!("{}", mphf.index(&"Hong Kong"));
    // ptrhash(FxHash::hash(&"Hong Kong", 0));

    // let mut taken = vec![false; 512];
    // for name in STATION_NAMES {
    //     let ref_idx = mphf.index(&name);
    //     let idx = ptrhash(FxHash::hash(&name, 0));
    //     assert_eq!(ref_idx, idx);
    //     assert!(!taken[ref_idx]);
    //     taken[ref_idx] = true;
    // }

    // println!(
    //     "{}",
    //     STATION_NAMES.map(|x| x.len()).into_iter().max().unwrap()
    // );

    // let global_len = 10_000;
    // let threadgroup_len = 1_365;
    // println!("Total names: {}", STATION_NAMES.len());
    name_len_stats();
    // min_prefix();

    // println!(
    //     "djbx33a(buckets={}): {}",
    //     global_len,
    //     statistics(djbx33a, global_len)
    // );
    // println!(
    //     "djbx33a(buckets={}): {}",
    //     threadgroup_len,
    //     statistics(djbx33a, threadgroup_len)
    // );
    // println!(
    //     "djbx33a_x4(buckets={}): {}",
    //     global_len,
    //     statistics(djbx33a_x4, global_len)
    // );
    // println!(
    //     "djbx33a_x4(buckets={}): {}",
    //     threadgroup_len,
    //     statistics(djbx33a_x4, threadgroup_len)
    // );
    // println!(
    //     "djbx33a_u64(buckets={}): {}",
    //     1024,
    //     statistics(djbx33a_u64, threadgroup_len)
    // );
    // find_perfect_hash(1024);
}
