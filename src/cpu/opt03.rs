//! Optimizations
//! 1. Fast hashmap with custom linear probing open addressing FxHash hashmap

use std::{collections::HashMap, ops::BitXor};

use crate::{Station, Stations};

const HASHMAP_SIZE: usize = 1 << 14; // 16,384

struct FxHash;

impl FxHash {
    const K: u64 = 0x517cc1b727220a95;

    #[inline]
    fn update(hash: u64, x: u64) -> u64 {
        hash.rotate_left(5).bitxor(x).wrapping_mul(Self::K)
    }
}

fn swar_find_semi(chars: u64) -> u64 {
    let diff = chars ^ 0x3B3B3B3B3B3B3B3B;
    (diff.wrapping_sub(0x0101010101010101)) & (!diff & 0x8080808080808080)
}

pub fn process<'a>(buf: &'a [u8], len: usize) -> Stations<'a> {
    let mut hashmap: Vec<(&[u8], Station)> = vec![(&[], Station::default()); HASHMAP_SIZE];

    let mut i = 0;
    let mut chars_buf = [0u8; 8];
    while i < len {
        let name_idx = i;
        let mut hash = 0u64;
        let semi_idx = loop {
            chars_buf.copy_from_slice(&buf[i..i + 8]);
            let chars = u64::from_le_bytes(chars_buf);
            let semi_bits = swar_find_semi(chars);
            if semi_bits != 0 {
                let lane_id = semi_bits.trailing_zeros() >> 3;
                if lane_id > 0 {
                    let chars = chars & (u64::MAX >> ((8 - lane_id) << 3));
                    hash = FxHash::update(hash, chars);
                }
                break i + lane_id as usize;
            }
            hash = FxHash::update(hash, chars);
            i += 8;
        };
        let name = &buf[name_idx..semi_idx];
        i = semi_idx + 1;

        let mut sign = 1;
        if buf[i] == b'-' {
            sign = -1;
            i += 1;
        }
        let (abs_temp, temp_len) = match &buf[i..i + 4] {
            [b, b'.', c, _] => (10 * (b - b'0') as i16 + (c - b'0') as i16, 3),
            [a, b, b'.', c] => (
                100 * (a - b'0') as i16 + 10 * (b - b'0') as i16 + (c - b'0') as i16,
                4,
            ),
            _ => unreachable!(),
        };
        let temp = sign * abs_temp;
        i += temp_len + 1;

        const BUCKET_MASK: usize = (u64::MAX >> (64 - 14)) as usize;
        let mut bucket_idx = hash as usize & BUCKET_MASK;
        let station = loop {
            let (bucket_key, station) = &mut hashmap[bucket_idx];
            if bucket_key.is_empty() {
                *bucket_key = name;
                break station;
            } else if *bucket_key == name {
                break station;
            }
            bucket_idx = (bucket_idx + 1) & BUCKET_MASK;
        };
        station.update(temp);
    }

    Stations {
        inner: hashmap
            .into_iter()
            .filter(|(key, _)| !key.is_empty())
            .collect::<HashMap<_, _>>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test;

    #[test]
    fn test_correctness() {
        test::correctness(process);
    }
}
