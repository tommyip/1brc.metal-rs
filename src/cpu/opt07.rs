//! Optimizations
//! 1. Parse 16-byte prefix than fallback to 8-byte loop

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    mem::transmute,
    ops::BitXor,
};

use crate::{Station, Stations, STATION_NAMES};

const MPH_SIZE: usize = 512;
const INLINE_KEY_SIZE: usize = 4;

#[derive(Default)]
struct PerfectHash {
    hash: u64,
}

/// FxHash from firefox/rustc
impl Hasher for PerfectHash {
    fn write_u64(&mut self, i: u64) {
        self.hash = self.hash.rotate_left(5).bitxor(i).wrapping_mul(Self::K);
    }

    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            let i = u64::from_le_bytes(buf);
            self.write_u64(i);
        }
    }
}

impl PerfectHash {
    const K: u64 = 0x517cc1b727220a95;

    /// Convert hash to bucket index
    ///
    /// Constructed with https://github.com/RagnarGrootKoerkamp/ptrhash with params:
    /// ```rs
    /// PtrHashParams {
    ///     alpha: 0.9,
    ///     c: 1.5,
    ///     slots_per_part: STATION_NAMES.len() * 2,
    ///     ..Default::default()
    /// };
    /// ```
    fn index(&self) -> usize {
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

        let is_large = self.hash >= P1;
        let rem = if is_large { REM_C2 } else { REM_C1 };
        let bucket = (is_large as isize * C3) + ((self.hash as u128 * rem as u128) >> 64) as isize;
        let pilot = PILOTS[bucket as usize];
        ((Self::K as u128 * (self.hash ^ Self::K.wrapping_mul(pilot as u64)) as u128) >> 64)
            as usize
            & ((1 << 9) - 1) as usize
    }
}

struct PerfectHashData {
    /// Station names matching MPH bucket indices
    keys: [[u64; INLINE_KEY_SIZE]; MPH_SIZE],
    /// Map bucket to STATION_NAMES
    indices: [usize; MPH_SIZE],
}

impl PerfectHashData {
    fn new() -> Self {
        let mut keys = [[0; 4]; MPH_SIZE];
        let mut indices = [0; MPH_SIZE];
        for (i, name) in STATION_NAMES.into_iter().enumerate() {
            let mut hash = PerfectHash::default();
            hash.write(name.as_bytes());
            let idx = hash.index();
            let bucket: &mut [u8; 32] = unsafe { transmute(&mut keys[idx]) };
            bucket[..name.len()].copy_from_slice(name.as_bytes());
            indices[idx] = i;
        }
        Self { keys, indices }
    }
}

fn swar_find_semi(chars: u64) -> u64 {
    let diff = chars ^ 0x3B3B3B3B3B3B3B3B;
    (diff.wrapping_sub(0x0101010101010101)) & (!diff & 0x8080808080808080)
}

fn swar_find_semi_2x(chars: u128) -> u128 {
    let diff = chars ^ 0x3B3B3B3B3B3B3B3B3B3B3B3B3B3B3B3B;
    (diff.wrapping_sub(0x01010101010101010101010101010101))
        & (!diff & 0x80808080808080808080808080808080)
}

struct Name {
    prefix: [u64; INLINE_KEY_SIZE],
    name_len: usize,
    hash: PerfectHash,
}

fn read_name<'a>(buf: &'a [u8]) -> Name {
    let mut hash = PerfectHash::default();
    let mut prefix = [0u64; INLINE_KEY_SIZE];
    let mut name_len = 0usize;

    // Special case for first 16 bytes
    let mut u128_buf = [0u128; 1];
    let u64x2_buf: &[u64; 2] = unsafe { transmute(&u128_buf) };
    unsafe { transmute::<_, &mut [u8; 16]>(&mut u128_buf) }.copy_from_slice(&buf[..16]);

    let semi_bits = swar_find_semi_2x(u128_buf[0]);
    if semi_bits != 0 {
        name_len = (semi_bits.trailing_zeros() >> 3) as usize;
        u128_buf[0] &= u128::MAX >> ((16 - name_len) << 3);
        hash.write_u64(u64x2_buf[0]);
        if name_len > 8 {
            hash.write_u64(u64x2_buf[1]);
        }
        prefix[0] = u64x2_buf[0];
        prefix[1] = u64x2_buf[1];
    } else {
        name_len += 16;
        prefix[0] = u64x2_buf[0];
        prefix[1] = u64x2_buf[1];
        hash.write_u64(u64x2_buf[0]);
        hash.write_u64(u64x2_buf[1]);

        // Fall back to remaining bytes
        let mut chars_buf = [0u8; 8];
        for block in 2.. {
            chars_buf.copy_from_slice(&buf[name_len..name_len + 8]);
            let mut chars = u64::from_le_bytes(chars_buf);
            let semi_bits = swar_find_semi(chars);
            if semi_bits != 0 {
                let lane_id = semi_bits.trailing_zeros() >> 3;
                if lane_id > 0 {
                    chars = chars & (u64::MAX >> ((8 - lane_id) << 3));
                    hash.write_u64(chars);
                    if block < prefix.len() {
                        prefix[block] = chars;
                    }
                }
                name_len += lane_id as usize;
                break;
            }
            hash.write_u64(chars);
            if block < prefix.len() {
                prefix[block] = chars;
            }
            name_len += 8;
        }
    }

    Name {
        prefix,
        name_len,
        hash,
    }
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

fn read_temp(buf: &[u8]) -> (i16, usize) {
    let mut chars_buf = [0u8; 8];
    chars_buf.copy_from_slice(&buf[..8]);
    let chars = i64::from_le_bytes(chars_buf);
    let dot_pos = (!chars & 0x10101000).trailing_zeros() as usize;
    let temp = swar_parse_temp(chars, dot_pos);
    (temp, (dot_pos >> 3) + 2)
}

pub fn process<'a>(buf: &'a [u8], len: usize) -> Stations<'a> {
    let mph_data = PerfectHashData::new();
    let mut mph: Vec<Station> = vec![Station::default(); MPH_SIZE];
    let mut stations = HashMap::<&'a [u8], Station, BuildHasherDefault<PerfectHash>>::default();

    let mut i = 0;
    while i < len {
        let name_idx = i;
        let Name {
            prefix,
            name_len,
            hash,
        } = read_name(&buf[name_idx..]);
        i += name_len + 1;

        let (temp, temp_len) = read_temp(&buf[i..]);
        i += temp_len + 1;

        // Try storing in minimal perfect hash table
        if name_len <= 26 {
            let idx = hash.index();
            if prefix == mph_data.keys[idx] {
                mph[idx].update(temp);
                continue;
            }
        }
        // Fallback to general hashmap
        stations
            .entry(&buf[name_idx..name_idx + name_len])
            .and_modify(|station| station.update(temp))
            .or_insert(Station::new(temp));
    }

    let mph_pairs = mph
        .into_iter()
        .enumerate()
        .filter(|(_, station)| station.count > 0)
        .map(|(i, station)| (STATION_NAMES[mph_data.indices[i]].as_bytes(), station));
    Stations {
        inner: stations.into_iter().chain(mph_pairs).collect(),
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
    fn test_swar_find_semi_2x() {
        fn find(s: &[u8; 16]) -> u32 {
            let bytes = u128::from_le_bytes(unsafe { std::mem::transmute_copy(s) });
            swar_find_semi_2x(bytes).trailing_zeros() >> 3
        }
        assert_eq!(find(b";aaaaaaaaaaaaaaa"), 0);
        assert_eq!(find(b"aaaaaaaaaaaaaaa;"), 15);
        assert_eq!(find(b"ab;1.0\nab;-23.4\n"), 2);
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
