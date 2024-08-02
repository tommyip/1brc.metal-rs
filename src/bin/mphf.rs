#![feature(is_none_or, array_chunks, test)]

use std::{
    arch::aarch64::*,
    collections::{BinaryHeap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    mem::transmute,
};

use one_billion_row::STATION_NAMES;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug)]
#[allow(unused)]
struct Fch {
    n: u64,
    m: u64,
    p1: u64,
    p2: u64,
    s2: u64,
    ds: Vec<u64>,
}

#[allow(unused)]
impl Fch {
    const S1: u64 = 0;

    fn construct(keys: &[&str], c: f64) -> Self {
        let n = keys.len() as f64;
        let m = ((c * n) / (n.log2() + 1.)).ceil();
        let p1 = (0.6 * n) as u64;
        let p2 = (0.3 * m) as u64;
        let n = keys.len() as u64;
        let m = m as u64;

        // 1. Mapping
        let mut buckets = (0..m as usize).map(|i| (i, vec![])).collect::<Vec<_>>();
        for &key in keys {
            let h = Self::hash(key, Self::S1);
            let bucket_idx = if (h % n) < p1 {
                h % p2 // S1
            } else {
                p2 + (h % (m - p2)) // S2
            } as usize;
            buckets[bucket_idx].1.push(key);
        }

        // 2. Ordering
        buckets.sort_by_key(|(_, bucket)| -(bucket.len() as i64));

        // 3. Searching
        let mut s2 = 0;
        let ds = 's: loop {
            let mut occupied = vec![false; keys.len()];
            let mut ds = vec![u64::MAX; buckets.len()];
            'bucket: for &(bucket_idx, ref bucket) in &buckets {
                for d in 0.. {
                    let mut pos_arr = bucket
                        .iter()
                        .map(|&key| ((Self::hash(key, s2) + d) % n) as usize)
                        .collect::<Vec<_>>();
                    if has_collision(&mut pos_arr) {
                        s2 += 1;
                        continue 's;
                    }
                    if pos_arr.iter().all(|&pos| !occupied[pos]) {
                        pos_arr.into_iter().for_each(|pos| occupied[pos] = true);
                        ds[bucket_idx] = d;
                        continue 'bucket;
                    }
                }
            }
            break ds;
        };

        Self {
            n,
            m,
            p1,
            p2,
            s2,
            ds,
        }
    }

    fn index(&self, key: &str) -> usize {
        let h = Self::hash(key, Self::S1);
        let bucket_idx = if (h % self.n) < self.p1 {
            h % self.p2
        } else {
            self.p2 + (h % (self.m - self.p2))
        };
        let d = self.ds[bucket_idx as usize];
        ((Self::hash(key, self.s2) + d as u64) % self.n) as usize
    }

    fn hash(key: &str, seed: u64) -> u64 {
        let mut s = DefaultHasher::new();
        seed.hash(&mut s);
        key.hash(&mut s);
        s.finish()
    }
}

#[derive(Debug)]
#[allow(unused)]
struct PTHash {
    n: u64,
    m: u64,
    p1: u64,
    p2: u64,
    s: u64,
    pilots: Vec<u64>,
}

#[allow(unused)]
impl PTHash {
    fn construct(keys: &[&str], c: f64) -> Self {
        let n = keys.len() as f64;
        let m = ((c * n) / (n.log2() + 1.)).ceil();
        let p1 = (0.6 * n) as u64;
        let p2 = (0.3 * m) as u64;
        let (n, m) = (keys.len() as u64, m as u64);

        // 1. Mapping
        let mut buckets = (0..m as usize).map(|i| (i, vec![])).collect::<Vec<_>>();
        for &key in keys {
            let h = Self::hash(key, 0);
            let bucket_idx = if (h % n) < p1 {
                h % p2 // S1
            } else {
                p2 + (h % (m - p2)) // S2
            } as usize;
            buckets[bucket_idx].1.push(key);
        }

        // 2. Ordering
        buckets.sort_by_key(|(_, bucket)| -(bucket.len() as i64));

        // 3. Searching
        // Find s so that hashes within each bucket are distinct
        let mut s = 0;
        's: loop {
            for (_, bucket) in &buckets {
                if has_collision(&mut bucket.clone()) {
                    s += 1;
                    continue 's;
                }
            }
            break;
        }
        let mut taken = vec![false; keys.len()];
        let mut pilots = vec![u64::MAX; m as usize];
        for (bucket_idx, bucket) in buckets {
            for k in 0.. {
                let mut poss = bucket
                    .iter()
                    .map(|key| ((Self::hash(key, s) ^ Self::hash(k, s)) % n) as usize)
                    .collect::<Vec<_>>();
                if !has_collision(&mut poss) && poss.iter().all(|&pos| !taken[pos]) {
                    poss.into_iter().for_each(|pos| taken[pos] = true);
                    pilots[bucket_idx] = k;
                    break;
                }
            }
        }

        Self {
            n,
            m,
            p1,
            p2,
            s,
            pilots,
        }
    }

    fn index(&self, key: &str) -> usize {
        let h = Self::hash(key, 0);
        let bucket_idx = if (h % self.n) < self.p1 {
            h % self.p2
        } else {
            self.p2 + (h % (self.m - self.p2))
        } as usize;
        let pilot = self.pilots[bucket_idx];
        ((Self::hash(key, self.s) ^ Self::hash(pilot, self.s)) % self.n) as usize
    }

    fn hash<T: Hash>(key: T, seed: u64) -> u64 {
        let mut s = DefaultHasher::new();
        seed.hash(&mut s);
        key.hash(&mut s);
        s.finish()
    }
}

/// Simplied PTRHash with u32 keys
/// Adapted from https://github.com/RagnarGrootKoerkamp/ptrhash
#[derive(Debug)]
struct PTRHash {
    p1: u32,
    #[allow(unused)]
    b: u32,
    s: u32,
    c1: u32,
    c2: u32,
    c3: i32,
    pilots: Vec<u8>,
}

impl PTRHash {
    const BETA: f64 = 0.6;
    const GAMMA: f64 = 0.3;
    /// Multiply hash constant
    /// `(u32::MAX as f64 * std::f64::consts::FRAC_1_PI) as u32`
    /// From https://nnethercote.github.io/2021/12/08/a-brutally-effective-hash-function-in-rust.html
    const C: u32 = 0x517cc1b6;
    const N_RECENT: usize = 2;

    fn construct(keys: &[&str], alpha: f64, c: f64) -> Self {
        let b = {
            let s = keys.len() as f64 / alpha;
            c * s / s.log2()
        } as u32;
        let p1 = (Self::BETA * (u32::MAX as f64)) as u32;
        let p2 = (Self::GAMMA * b as f64) as u32;
        let s = 1 << (keys.len() as f64).log2().ceil() as u32;
        let c1 = (Self::GAMMA / Self::BETA * b.saturating_sub(2) as f64).floor() as u32;
        let c2 = ((1. - Self::GAMMA) / (1. - Self::BETA) * b.saturating_sub(2) as f64) as u32;
        let c3 = p2 as i32 - (Self::BETA * c2 as f64) as i32 + 1;

        let mut buckets = vec![vec![]; b as usize];
        let mut hashes = HashSet::new();
        // 1. Mapping
        for key in keys {
            let h = Self::hash(key);
            if !hashes.insert(h) {
                panic!("Hash collision {h}");
            }
            let bucket = Self::bucket(h, p1, c1, c2, c3);
            buckets[bucket].push(h);
        }

        // 2. Sort buckets
        let mut bucket_indices = (0..b as usize).collect::<Vec<_>>();
        bucket_indices.sort_by_key(|&i| -(buckets[i].len() as i64));

        // 3. Find pilots
        let mut rng = StdRng::seed_from_u64(0);
        let mut taken = vec![Option::<usize>::None; s as usize];
        let mut pilots = vec![0u8; b as usize];
        let mut displacements = 0;
        for &bucket_idx in &bucket_indices {
            let mut displaced_q = BinaryHeap::new();
            displaced_q.push((0, bucket_idx));
            let mut recent = [usize::MAX; Self::N_RECENT];
            let mut recent_idx = 0;
            recent[0] = bucket_idx;

            'q: while let Some((_, bucket_idx)) = displaced_q.pop() {
                let bucket = &buckets[bucket_idx];

                // 3.1 Try finding pilot with no collision
                let mut best_score = None;
                let p0 = rng.gen::<u8>() as u32;
                for delta in 0..(u8::MAX as u32) {
                    let pilot = ((p0 + delta) % u8::MAX as u32) as u8;
                    let mut slots = bucket
                        .iter()
                        .map(|&h| Self::slot(h, pilot, s))
                        .collect::<Vec<_>>();
                    if has_collision(&mut slots) {
                        continue;
                    }
                    let mut colliding_buckets = slots
                        .iter()
                        .filter_map(|&slot| taken[slot])
                        .collect::<Vec<_>>();
                    colliding_buckets.sort_unstable();
                    colliding_buckets.dedup();
                    if colliding_buckets.is_empty() {
                        // No collision, accept pilot for current bucket
                        slots
                            .into_iter()
                            .for_each(|slot| taken[slot] = Some(bucket_idx));
                        pilots[bucket_idx] = pilot;
                        continue 'q;
                    }
                    let score = colliding_buckets
                        .iter()
                        .map(|&bucket_idx| buckets[bucket_idx].len().pow(2))
                        .sum::<usize>();
                    if colliding_buckets
                        .iter()
                        .any(|bucket_idx| recent.contains(bucket_idx))
                    {
                        continue;
                    }
                    if best_score
                        .as_ref()
                        .is_none_or(|(best_score, _, _, _)| score < *best_score)
                    {
                        best_score = Some((score, pilot, slots, colliding_buckets));
                    }
                }

                // 3.2 Displace buckets with conflicts
                let Some((_, pilot, slots, colliding_buckets)) = &best_score else {
                    panic!("Cannot find bucket to displace");
                };
                pilots[bucket_idx] = *pilot;
                for slot in &mut taken {
                    if let Some(taken_by) = slot {
                        if colliding_buckets.contains(taken_by) {
                            *slot = None;
                        }
                    }
                }
                for &slot in slots {
                    taken[slot] = Some(bucket_idx);
                }
                for &bucket_idx in colliding_buckets {
                    displaced_q.push((buckets[bucket_idx].len(), bucket_idx));
                }
                displacements += colliding_buckets.len();
                if displacements >= 10 * s as usize {
                    panic!("Too many displacements {displacements}");
                }

                recent_idx += 1;
                recent_idx %= Self::N_RECENT;
                recent[recent_idx] = bucket_idx;
            }
        }

        Self {
            p1,
            b,
            s,
            c1,
            c2,
            c3,
            pilots,
        }
    }

    fn index(&self, key: &str) -> usize {
        let h = Self::hash(key);
        let bucket = Self::bucket(h, self.p1, self.c1, self.c2, self.c3);
        let pilot = self.pilots[bucket];
        Self::slot(h, pilot, self.s)
    }

    fn hash(key: &str) -> u32 {
        let mut buf = [0u8; 8];
        let len = key.len().min(8);
        buf[..len].copy_from_slice(&key.as_bytes()[..len]);
        let x0 = u32::from_le_bytes(*unsafe { transmute::<&[u8; 8], &[u8; 4]>(&buf) });
        let x1 =
            u32::from_le_bytes(*unsafe { transmute::<*const u8, &[u8; 4]>(buf.as_ptr().add(4)) });
        (x0 ^ x1 ^ key.len() as u32).wrapping_mul(Self::C)
    }

    fn bucket(h: u32, p1: u32, c1: u32, c2: u32, c3: i32) -> usize {
        (if h < p1 {
            fast_reduce(h, c1)
        } else {
            (c3 + (fast_reduce(h, c2) as i32)) as u32
        }) as usize
    }

    fn slot(h: u32, p: u8, s: u32) -> usize {
        debug_assert!(h % 2 == 0);
        debug_assert!(Self::C % 2 == 0);
        (fast_reduce(Self::C, h ^ Self::C.wrapping_mul(p as u32)) & (s - 1)) as usize
    }

    unsafe fn hash_neon_x4(key_arr: &[&str; 4]) -> uint32x4_t {
        let len_arr = key_arr.map(|key| key.len() as u32);
        let lens = vld1q_u32(len_arr.as_ptr());
        let mask_arr = len_arr.map(|len| ((1i64.checked_shl(len << 3).unwrap_or(0) - 1) as u64));
        let masks = vld1q_u64_x2(mask_arr.as_ptr());
        let keys_lo = vld1q_lane_u64::<0>(transmute(key_arr[0].as_ptr()), vdupq_n_u64(0));
        let keys_lo = vld1q_lane_u64::<1>(transmute(key_arr[1].as_ptr()), keys_lo);
        let keys_lo = vandq_u64(keys_lo, masks.0);
        let keys_hi = vld1q_lane_u64::<0>(transmute(key_arr[2].as_ptr()), vdupq_n_u64(0));
        let keys_hi = vld1q_lane_u64::<1>(transmute(key_arr[3].as_ptr()), keys_hi);
        let keys_hi = vandq_u64(keys_hi, masks.1);
        let keys = vuzpq_u32(
            vreinterpretq_u32_u64(keys_lo),
            vreinterpretq_u32_u64(keys_hi),
        );
        let xor = veor3q_u32(keys.0, keys.1, lens);
        vmulq_n_u32(xor, PTRHash::C)
    }

    unsafe fn index_neon_x4(&self, keys: &[&str; 4]) -> [usize; 4] {
        let hash = Self::hash_neon_x4(keys);

        // Calculate bucket
        let hash_half = vshrq_n_u32::<1>(hash);
        let is_large = vcgeq_u32(hash, vdupq_n_u32(self.p1));
        let large_buckets = vandq_s32(vreinterpretq_s32_u32(is_large), vdupq_n_s32(self.c3));
        let rem_c = vbslq_u32(is_large, vdupq_n_u32(self.c2), vdupq_n_u32(self.c1));
        let local_bucket = vqdmulhq_s32(
            vreinterpretq_s32_u32(hash_half),
            vreinterpretq_s32_u32(rem_c),
        );
        let bucket = vreinterpretq_u32_s32(vaddq_s32(large_buckets, local_bucket));

        // Lookup pilot
        let pilot_lut = vld1q_u8_x4(self.pilots.as_ptr());
        let pilot = vreinterpretq_u32_u8(vqtbl4q_u8(pilot_lut, vreinterpretq_u8_u32(bucket)));
        let pilot = vandq_u32(pilot, vdupq_n_u32(0xFF));

        // Calculate slot
        let hp = vshrq_n_u32::<1>(vmulq_n_u32(pilot, Self::C));
        let slot_inner = vreinterpretq_u32_s32(vqdmulhq_n_s32(
            vreinterpretq_s32_u32(veorq_u32(hash_half, hp)),
            Self::C as i32,
        ));
        let slot = vandq_u32(slot_inner, vdupq_n_u32(self.s - 1));

        let mut out = [0; 4];
        vst1q_u32(out.as_mut_ptr(), slot);

        out.map(|x| x as usize)
    }
}

/// Like module (%) but fast
/// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
fn fast_reduce(h: u32, d: u32) -> u32 {
    ((h as u64 * d as u64) >> 32) as u32
}

fn has_collision<T: PartialOrd + Ord>(arr: &mut Vec<T>) -> bool {
    let n = arr.len();
    arr.sort_unstable();
    arr.dedup();
    arr.len() < n
}

fn verify<const N_SLOTS: usize, F>(index_fn: F)
where
    F: Fn(&str) -> usize,
{
    let mut occupied = vec![false; N_SLOTS];
    let mut success = true;
    for name in STATION_NAMES {
        let i = index_fn(name);
        if occupied[i] {
            println!("{} {}", name, i);
            success = false;
        }
        occupied[i] = true;
    }
    assert!(success);
}

fn main() {
    // let fch = Fch::construct(&STATION_NAMES, 5.);
    // println!("{:?}", fch);
    // verify(|x| fch.index(x));
    // let pthash = PTHash::construct(&STATION_NAMES, 1.5);
    // println!("{:?}", pthash);
    // verify(|x| pthash.index(x));
    let ptrhash = PTRHash::construct(&STATION_NAMES, 0.98, 1.34);
    println!("{:?}", ptrhash);
    verify::<512, _>(|x| ptrhash.index(x));
}

#[cfg(test)]
mod tests {
    extern crate test;

    use std::arch::aarch64::vst1q_u32;

    use test::{black_box, Bencher};

    use super::*;

    fn default_ptrhash() -> PTRHash {
        PTRHash {
            p1: 2576980377,
            b: 64,
            s: 512,
            c1: 31,
            c2: 108,
            c3: -44,
            pilots: vec![
                95, 219, 20, 180, 128, 176, 28, 128, 105, 245, 169, 122, 225, 13, 124, 163, 38,
                148, 2, 162, 215, 122, 171, 43, 78, 185, 27, 220, 158, 139, 144, 6, 181, 174, 24,
                229, 40, 236, 63, 41, 128, 218, 136, 8, 254, 140, 188, 137, 120, 192, 204, 174,
                121, 17, 64, 198, 123, 108, 245, 249, 212, 78, 113, 221,
            ],
        }
    }

    #[test]
    fn test_ptrhash_hash_neon_x4() {
        for chunk in STATION_NAMES.array_chunks::<4>() {
            let actual = unsafe {
                let actual = PTRHash::hash_neon_x4(chunk);
                let mut out = [0; 4];
                vst1q_u32(out.as_mut_ptr(), actual);
                out
            };
            let expected = chunk.map(PTRHash::hash);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_ptrhash_index_neon_x4() {
        let ptrhash = default_ptrhash();
        for chunk in STATION_NAMES.array_chunks::<4>() {
            let actual = unsafe { ptrhash.index_neon_x4(chunk) };
            let expected = chunk.map(|key| ptrhash.index(key));
            assert_eq!(actual, expected);
        }
    }

    #[bench]
    fn bench_ptrhash_cpu(b: &mut Bencher) {
        let ptrhash = default_ptrhash();
        b.iter(|| {
            for name in STATION_NAMES {
                black_box(ptrhash.index(name));
            }
        });
    }

    #[bench]
    fn bench_ptrhash_neon(b: &mut Bencher) {
        let ptrhash = default_ptrhash();
        b.iter(|| unsafe {
            for chunk in STATION_NAMES.array_chunks::<4>() {
                black_box(ptrhash.index_neon_x4(chunk));
            }
        });
    }
}
