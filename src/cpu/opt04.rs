//! Optimizations
//! 1. Use inline hash key if station name is less than 16 bytes

use std::{collections::HashMap, ops::BitXor};

use crate::{Station, Stations};

const HASHMAP_SIZE: usize = 1 << 14; // 16,384
const INLINE_KEY_SIZE: usize = 2;

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

#[derive(Clone)]
enum Key<'a> {
    Inline([u64; INLINE_KEY_SIZE], &'a [u8]),
    Str(&'a [u8]),
}

impl<'a> Key<'a> {
    fn as_bytes(&self) -> &'a [u8] {
        match self {
            Key::Inline(_, bytes) => bytes,
            Key::Str(bytes) => bytes,
        }
    }
}

impl PartialEq for Key<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Key::Inline(self_prefix, _), Key::Inline(other_prefix, _))
                if self_prefix == other_prefix =>
            {
                true
            }
            (Key::Str(self_str), Key::Str(other_str)) if self_str == other_str => true,
            _ => false,
        }
    }
}

impl Eq for Key<'_> {}

pub fn process<'a>(buf: &'a [u8], len: usize) -> Stations<'a> {
    let mut hashmap: Vec<Option<(Key<'a>, Station)>> = vec![None; HASHMAP_SIZE];

    let mut i = 0;
    let mut chars_buf = [0u8; 8];
    while i < len {
        let name_idx = i;
        let mut hash = 0u64;
        let mut prefix = [0u64; INLINE_KEY_SIZE];
        let mut semi_idx = 0;
        for block in 0.. {
            chars_buf.copy_from_slice(&buf[i..i + 8]);
            let mut chars = u64::from_le_bytes(chars_buf);
            let semi_bits = swar_find_semi(chars);
            if semi_bits != 0 {
                let lane_id = semi_bits.trailing_zeros() >> 3;
                if lane_id > 0 {
                    chars = chars & (u64::MAX >> ((8 - lane_id) << 3));
                    hash = FxHash::update(hash, chars);
                    if block < prefix.len() {
                        prefix[block] = chars;
                    }
                }
                semi_idx = i + lane_id as usize;
                i = semi_idx + 1;
                break;
            }
            hash = FxHash::update(hash, chars);
            if block < prefix.len() {
                prefix[block] = chars;
            }
            i += 8;
        }

        let mut sign = 1;
        if buf[i] == b'-' {
            sign = -1;
            i += 1;
        }
        let (abs_temp, temp_len) = match &buf[i..i + 4] {
            [b, b'.', c, _] => (10 * (b - b'0') as i32 + (c - b'0') as i32, 3),
            [a, b, b'.', c] => (
                100 * (a - b'0') as i32 + 10 * (b - b'0') as i32 + (c - b'0') as i32,
                4,
            ),
            _ => unreachable!(),
        };
        let temp = sign * abs_temp;
        i += temp_len + 1;

        let name = &buf[name_idx..semi_idx];
        let key = if name.len() <= INLINE_KEY_SIZE * 8 {
            Key::Inline(prefix, name)
        } else {
            Key::Str(name)
        };
        const BUCKET_MASK: usize = (u64::MAX >> (64 - 14)) as usize;
        let mut bucket_idx = hash as usize & BUCKET_MASK;
        loop {
            if let Some((bucket_key, station)) = &mut hashmap[bucket_idx] {
                if key == *bucket_key {
                    station.update(temp);
                    break;
                }
            } else {
                hashmap[bucket_idx] = Some((key, Station::new(temp)));
                break;
            }
            bucket_idx = (bucket_idx + 1) & BUCKET_MASK;
        }
    }

    Stations {
        inner: hashmap
            .into_iter()
            .filter_map(|bucket| bucket.map(|(key, station)| (key.as_bytes(), station)))
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
