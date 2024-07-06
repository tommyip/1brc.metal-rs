use std::hash::{DefaultHasher, Hash, Hasher};

use one_billion_row::STATION_NAMES;

#[derive(Debug)]
struct Fch {
    n: u64,
    m: u64,
    p1: u64,
    p2: u64,
    s2: u64,
    ds: Vec<u64>,
}

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
struct PTHash {
    n: u64,
    m: u64,
    p1: u64,
    p2: u64,
    s: u64,
    pilots: Vec<u64>,
}

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

fn has_collision<T: PartialOrd + Ord>(arr: &mut Vec<T>) -> bool {
    let n = arr.len();
    arr.sort_unstable();
    arr.dedup();
    arr.len() < n
}

fn verify<F>(index_fn: F)
where
    F: Fn(&str) -> usize,
{
    let mut occupied = vec![false; STATION_NAMES.len()];
    for name in STATION_NAMES {
        let i = index_fn(name);
        if occupied[i] {
            println!("{} {}", name, i);
        }
        occupied[i] = true;
    }
}

fn main() {
    // let fch = Fch::construct(&STATION_NAMES, 5.);
    // println!("{:?}", fch);
    // verify(|x| fch.index(x));
    let pthash = PTHash::construct(&STATION_NAMES, 1.5);
    println!("{:?}", pthash);
    verify(|x| pthash.index(x));
}
