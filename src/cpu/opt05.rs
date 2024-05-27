//! Optimizations
//! 1. Branchless temperature parsing with SWAR

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

fn swar_parse_temp(chars: i64, dot_pos: usize) -> i16 {
    const MAGIC_MUL: i64 = 100 * 0x1000000 + 10 * 0x10000 + 1;
    let shift = 28 - dot_pos;
    let sign = (!chars << 59) >> 63;
    let minus_filter = !(sign & 0xFF);
    let digits = ((chars & minus_filter) << shift) & 0x0F000F0F00;
    let abs_value = (digits.wrapping_mul(MAGIC_MUL) as u64 >> 32) & 0x3FF;
    ((abs_value as i64 ^ sign) - sign) as i16
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

        chars_buf.copy_from_slice(&buf[i..i + 8]);
        let chars = i64::from_le_bytes(chars_buf);
        let dot_pos = (!chars & 0x10101000).trailing_zeros() as usize;
        i += (dot_pos >> 3) + 3;
        let temp = swar_parse_temp(chars, dot_pos);

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

    #[test]
    fn test_swar_parse_temp() {
        fn parse(s: &[u8]) -> i16 {
            let mut buf = [0u8; 8];
            buf[..s.len()].copy_from_slice(s);
            let chars = i64::from_le_bytes(buf);
            let dot_pos = (!chars & 0x10101000).trailing_zeros();
            swar_parse_temp(chars, dot_pos as usize)
        }
        assert_eq!(parse(b"0.0"), 0);
        assert_eq!(parse(b"1.0"), 10);
        assert_eq!(parse(b"1.2"), 12);
        assert_eq!(parse(b"-1.2"), -12);
        assert_eq!(parse(b"12.3"), 123);
        assert_eq!(parse(b"-12.3"), -123);
    }
}
