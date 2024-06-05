//! Optimizations

use std::{
    arch::aarch64::*,
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    mem::transmute,
    ops::BitXor,
};

use crate::{Station, Stations, STATION_NAMES};

const MPH_SIZE: usize = 512;

fn neon_load_u8x32(s: &[u8]) -> uint8x16x2_t {
    unsafe { vld1q_u8_x2(s.as_ptr()) }
}

/// 64-byte match mask for `;` and `\n` (aka branchless searching ; and \n 64-bytes at a time)
///
/// Credit to Danila Kutenin (Porting Intel SSE movemasks to ARM Neon for 16-bytes)
/// https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
/// Credit to aqrit (Extend above to 64-bytes)
/// https://stackoverflow.com/a/74748402
fn neon_find_delimiters_u8x64(chunk: uint8x16x4_t) -> u64 {
    unsafe {
        let v_semi0 = vceqq_u8(chunk.0, vdupq_n_u8(b';'));
        let v_semi1 = vceqq_u8(chunk.1, vdupq_n_u8(b';'));
        let v_semi2 = vceqq_u8(chunk.2, vdupq_n_u8(b';'));
        let v_semi3 = vceqq_u8(chunk.3, vdupq_n_u8(b';'));

        let v_newline0 = vceqq_u8(chunk.0, vdupq_n_u8(b'\n'));
        let v_newline1 = vceqq_u8(chunk.1, vdupq_n_u8(b'\n'));
        let v_newline2 = vceqq_u8(chunk.2, vdupq_n_u8(b'\n'));
        let v_newline3 = vceqq_u8(chunk.3, vdupq_n_u8(b'\n'));

        let v0 = vorrq_u8(v_semi0, v_newline0);
        let v1 = vorrq_u8(v_semi1, v_newline1);
        let v2 = vorrq_u8(v_semi2, v_newline2);
        let v3 = vorrq_u8(v_semi3, v_newline3);

        let acc = vsriq_n_u8(vsriq_n_u8(v3, v2, 1), vsriq_n_u8(v1, v0, 1), 2);
        let res = vshrn_n_u16(vreinterpretq_u16_u8(vsriq_n_u8(acc, acc, 4)), 4);
        vget_lane_u64(vreinterpret_u64_u8(res), 0)
    }
}

/// `temp_len` is the number of bytes of between ; and \n (ie |temp| + 1)
fn swar_parse_temp(chars: i64, temp_len: u32) -> i16 {
    const MAGIC_MUL: i64 = 100 * 0x1000000 + 10 * 0x10000 + 1;
    let shift = 48 - (temp_len << 3);
    let sign = (!chars << 59) >> 63;
    let minus_filter = !(sign & 0xFF);
    let digits = ((chars & minus_filter) << shift) & 0x0F000F0F00;
    let abs_value = (digits.wrapping_mul(MAGIC_MUL) as u64 >> 32) & 0x3FF;
    ((abs_value as i64 ^ sign) - sign) as i16
}

fn next_line(buf: &[u8], i: usize) -> usize {
    i + buf[i..].into_iter().position(|&b| b == b'\n').unwrap() + 1
}

struct Line<'a> {
    name: &'a [u8],
    temp: i64,     // a slice of 8 bytes starting at the beginning of the temperature
    temp_len: u32, // number of bytes between semicolon and newline (4..=6)
}

struct LineIter<'a> {
    buf: &'a [u8],
    mask: u64,
    chunk_i: usize,
    i: usize,
    end: usize,
}

impl<'a> LineIter<'a> {
    fn new(buf: &'a [u8], i: usize, end: usize) -> Self {
        assert!(buf.as_ptr().is_aligned_to(16));
        let chunk_i = (i / 64) * 64;
        let chunk = unsafe { vld4q_u8(buf.as_ptr().add(chunk_i)) };
        let mask = {
            let mask = neon_find_delimiters_u8x64(chunk);
            mask & (u64::MAX << (i % 64))
        };
        Self {
            buf,
            mask,
            chunk_i,
            i,
            end,
        }
    }

    fn is_empty(&self) -> bool {
        self.i >= self.end
    }

    /// Return (name, temp index, semicolon offset in 16-byte chunk)
    fn next_name(&mut self) -> (&'a [u8], u32) {
        while self.mask == 0 {
            self.chunk_i += 64;
            let chunk = self.neon_load_interleaved_u8x16x4(self.chunk_i);
            self.mask = neon_find_delimiters_u8x64(chunk);
        }
        let semi_offset = self.mask.trailing_zeros();
        self.mask &= self.mask - 1;
        let semi_pos = self.chunk_i + semi_offset as usize;
        let name = &self.buf[self.i..semi_pos];
        self.i = semi_pos + 1;
        (name, semi_offset)
    }

    /// Find number of bytes between semicolon and newline (4..=6)
    fn next_temp_len(&mut self, semi_offset: u32) -> u32 {
        let temp_len = if self.mask == 0 {
            self.chunk_i += 64;
            let chunk = self.neon_load_interleaved_u8x16x4(self.chunk_i);
            self.mask = neon_find_delimiters_u8x64(chunk);
            let lf_offset = self.mask.trailing_zeros();
            64 - semi_offset + lf_offset
        } else {
            let lf_offset = self.mask.trailing_zeros();
            lf_offset - semi_offset
        };
        self.mask &= self.mask - 1;
        self.i += temp_len as usize;
        temp_len
    }

    /// Read the 8-byte slice containing the temperature string but not parse it
    fn read_temp_str(&self) -> i64 {
        i64::from_le_bytes(unsafe {
            *transmute::<*const u8, &[u8; 8]>(self.buf[self.i..].as_ptr())
        })
    }

    fn next_unchecked(&mut self) -> Line<'a> {
        let (name, semi_offset) = self.next_name();
        let temp = self.read_temp_str();
        let temp_len = self.next_temp_len(semi_offset);
        Line {
            name,
            temp,
            temp_len,
        }
    }

    fn neon_load_interleaved_u8x16x4(&self, i: usize) -> uint8x16x4_t {
        unsafe { vld4q_u8(self.buf.as_ptr().add(i)) }
    }
}

impl<'a> Iterator for LineIter<'a> {
    type Item = Line<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (!self.is_empty()).then(|| self.next_unchecked())
    }
}

struct Name<'a> {
    value: &'a [u8],
    /// 32-bytes prefix of name (trailing bytes not masked)
    nomask_32: uint8x16x2_t,
    /// 8-bytes prefix of name (trailing bytes masked)
    masked_8: u64,
}

impl<'a> Name<'a> {
    fn new(value: &'a [u8]) -> Self {
        Self {
            value,
            nomask_32: neon_load_u8x32(value),
            masked_8: Self::load_masked8(value),
        }
    }

    fn load_masked8(name: &[u8]) -> u64 {
        let mut x = u64::from_le_bytes(unsafe { *transmute::<*const u8, &[u8; 8]>(name.as_ptr()) });
        if name.len() < 8 {
            x &= (1 << (name.len() << 3)) - 1;
        }
        x
    }

    fn mph_idx(&self) -> usize {
        mph_index(cc_hash(self.masked_8, self.value.len()))
    }

    /// Byte equality for up to 32 bytes prefix of station name
    ///
    /// We simply check the number of matching bytes is equal to the expected length. `expected`
    /// have trailing bytes padded with `0xFF` which means that no `actual` bytes should match these
    /// padding bytes. We don't use `0x0` for padding since that is used by the mmap trailer.
    fn neon_equal(&self, expected: &[u8; 32], expected_len: u32) -> bool {
        let matching_bytes = unsafe {
            let expected = vld1q_u8_x2(expected.as_ptr());
            let v_a8b8 = vceqq_u8(self.nomask_32.0, expected.0);
            let v_c8d8 = vceqq_u8(self.nomask_32.1, expected.1);
            let v_a6b2 = vshrn_n_u16(vreinterpretq_u16_u8(v_a8b8), 6);
            let v_c6d2 = vshrn_n_u16(vreinterpretq_u16_u8(v_c8d8), 6);
            let v_aabbccdd = vsli_n_u8(v_a6b2, v_c6d2, 4);
            let aabbccdd = vget_lane_u64(vreinterpret_u64_u8(v_aabbccdd), 0);
            aabbccdd.count_ones() >> 1
        };
        matching_bytes == expected_len
    }
}

#[derive(Default)]
struct FxHash {
    hash: u64,
}

impl FxHash {
    const K: u64 = 0x517cc1b727220a95;
}

impl Hasher for FxHash {
    fn write_u64(&mut self, i: u64) {
        self.hash = self.hash.rotate_left(5).bitxor(i).wrapping_mul(Self::K);
    }

    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, mut bytes: &[u8]) {
        while let Some((chunk, rest)) = bytes.split_first_chunk() {
            let x = u64::from_le_bytes(*chunk);
            self.write_u64(x);
            bytes = rest;
        }
        if !bytes.is_empty() {
            let x =
                u64::from_le_bytes(unsafe { *transmute::<*const u8, &[u8; 8]>(bytes.as_ptr()) });
            let x = x & (1 << (bytes.len() << 3)) - 1;
            self.write_u64(x);
        }
    }
}

/// For the 413 station names dataset, the leading 9 bytes of the name is enough
/// to ensure key uniqueness. It can be reduce to 8 bytes by XORing in the full
/// name length.
/// Credit: Ragnar Groot Koerkamp
/// https://curiouscoding.nl/posts/1brc/#inline-hash-keys-50s
fn cc_hash(prefix: u64, len: usize) -> u64 {
    // While $prefix ⊕ len$ gives us unique hashes, its distribution is poor.
    // To make it work with PTRHash to find a minimal perfect hashmap, we
    // additionally multiply it with a large constant.
    (prefix ^ len as u64).wrapping_mul(FxHash::K)
}

/// Convert CC hash to bucket index
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
fn mph_index(hash: u64) -> usize {
    const PILOTS: [u8; 78] = [
        0, 21, 32, 6, 13, 20, 13, 4, 2, 0, 47, 6, 3, 2, 4, 28, 1, 17, 24, 31, 23, 15, 37, 0, 42,
        39, 22, 1, 1, 24, 0, 6, 59, 0, 0, 30, 10, 46, 0, 7, 38, 28, 46, 23, 60, 14, 12, 23, 0, 56,
        43, 15, 165, 30, 138, 46, 25, 14, 180, 0, 42, 72, 6, 99, 46, 0, 125, 39, 75, 94, 70, 47,
        245, 7, 55, 41, 237, 0,
    ];
    const REM_C1: u64 = 38;
    const REM_C2: u64 = 132;
    const C3: isize = -55;
    const P1: u64 = 11068046444225730560;

    let is_large = hash >= P1;
    let rem = if is_large { REM_C2 } else { REM_C1 };
    let bucket = (is_large as isize * C3) + ((hash as u128 * rem as u128) >> 64) as isize;
    let pilot = PILOTS[bucket as usize];
    ((FxHash::K as u128 * (hash ^ FxHash::K.wrapping_mul(pilot as u64)) as u128) >> 64) as usize
        & ((1 << 9) - 1) as usize
}

struct PerfectHashData {
    /// Station names matching MPH bucket indices
    keys: [([u8; 32], u32); MPH_SIZE],
    /// Map bucket to STATION_NAMES
    indices: [usize; MPH_SIZE],
}

impl PerfectHashData {
    fn new() -> Self {
        let mut keys = [([0xFF; 32], 0); MPH_SIZE];
        let mut indices = [0; MPH_SIZE];
        for (i, name) in STATION_NAMES.into_iter().enumerate() {
            let prefix = Name::load_masked8(name.as_bytes());
            let idx = mph_index(cc_hash(prefix, name.len()));
            let (bucket, len) = &mut keys[idx];
            bucket[..name.len()].copy_from_slice(name.as_bytes());
            *len = name.len() as u32;
            indices[idx] = i;
        }
        Self { keys, indices }
    }
}

struct Records<'a> {
    mph_data: PerfectHashData,
    mph: Vec<Station>,
    fallback: HashMap<&'a [u8], Station, BuildHasherDefault<FxHash>>,
}

impl<'a> Records<'a> {
    fn new() -> Self {
        let mph_data = PerfectHashData::new();
        let mph = vec![Station::default(); MPH_SIZE];
        let fallback = HashMap::default();
        Self {
            mph_data,
            mph,
            fallback,
        }
    }

    fn use_fast_path(&self, name: &Name<'a>, mph_idx: usize) -> bool {
        let &(ref key, len) = &self.mph_data.keys[mph_idx];
        len > 0 && name.neon_equal(key, len as u32)
    }

    fn insert_fast(&mut self, mph_idx: usize, temp: i16) {
        self.mph[mph_idx].update(temp);
    }

    #[inline(never)]
    fn insert_slow(&mut self, name: &'a [u8], temp: i16) {
        self.fallback
            .entry(name)
            .and_modify(|station| station.update(temp))
            .or_insert(Station::new(temp));
    }

    fn finish(self) -> Stations<'a> {
        let mph_pairs = self
            .mph
            .into_iter()
            .enumerate()
            .filter(|(_, station)| station.count > 0)
            .map(|(i, station)| (STATION_NAMES[self.mph_data.indices[i]].as_bytes(), station));
        Stations {
            inner: self.fallback.into_iter().chain(mph_pairs).collect(),
        }
    }
}

pub fn process<'a>(buf: &'a [u8], len: usize) -> Stations<'a> {
    let mut records = Records::new();

    let chunk_len = len / 2;
    let chunk1_offset = next_line(buf, chunk_len);
    let mut lines0 = LineIter::new(buf, 0, chunk1_offset);
    let mut lines1 = LineIter::new(buf, chunk1_offset, len);

    while !(lines0.is_empty() || lines1.is_empty()) {
        let (name_str0, semi_offset0) = lines0.next_name();
        let (name_str1, semi_offset1) = lines1.next_name();

        let temp_str0 = lines0.read_temp_str();
        let temp_str1 = lines1.read_temp_str();

        let temp_len0 = lines0.next_temp_len(semi_offset0);
        let temp_len1 = lines1.next_temp_len(semi_offset1);

        let temp0 = swar_parse_temp(temp_str0, temp_len0);
        let temp1 = swar_parse_temp(temp_str1, temp_len1);

        let name0 = Name::new(name_str0);
        let name1 = Name::new(name_str1);

        let mph_idx0 = name0.mph_idx();
        let mph_idx1 = name1.mph_idx();

        let use_fast_path0 = records.use_fast_path(&name0, mph_idx0);
        let use_fast_path1 = records.use_fast_path(&name1, mph_idx1);

        if use_fast_path0 && use_fast_path1 {
            records.insert_fast(mph_idx0, temp0);
            records.insert_fast(mph_idx1, temp1);
        } else {
            records.insert_slow(name0.value, temp0);
            records.insert_slow(name1.value, temp1);
        }
    }

    for line in lines0 {
        let temp = swar_parse_temp(line.temp, line.temp_len);
        let name = Name::new(line.name);
        let mph_idx = name.mph_idx();
        if records.use_fast_path(&name, mph_idx) {
            records.insert_fast(mph_idx, temp);
        } else {
            records.insert_slow(name.value, temp);
        }
    }

    for line in lines1 {
        let temp = swar_parse_temp(line.temp, line.temp_len);
        let name = Name::new(line.name);
        let mph_idx = name.mph_idx();
        if records.use_fast_path(&name, mph_idx) {
            records.insert_fast(mph_idx, temp);
        } else {
            records.insert_slow(name.value, temp);
        }
    }

    records.finish()
}

#[cfg(test)]
mod tests {
    extern crate test;

    use std::{arch::aarch64, fs::File, path::Path};

    use test::{black_box, Bencher};

    use super::*;
    use crate::{mmap, BUF_EXCESS};

    #[test]
    fn test_correctness() {
        crate::test::correctness(process);
    }

    #[test]
    fn test_line_iter() {
        let (mmap, len) = mmap::<BUF_EXCESS>(
            &File::open(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests")
                    .join("measurements-3.txt"),
            )
            .unwrap(),
        );
        let lines = LineIter::new(&mmap, 0, len)
            .map(|line| (line.name, swar_parse_temp(line.temp, line.temp_len)))
            .collect::<Vec<_>>();
        assert_eq!(
            vec![
                (b"Bosaso" as &[u8], 50i16),
                (b"Bosaso", 200),
                (b"Bosaso", -50),
                (b"Bosaso", -150),
                (b"Petropavlovsk-Kamchatsky", 95),
                (b"Petropavlovsk-Kamchatsky", -95)
            ],
            lines
        );
    }

    #[test]
    fn test_swar_parse_temp() {
        fn parse(s: &[u8]) -> i16 {
            let mut buf = [0u8; 8];
            buf[..s.len()].copy_from_slice(s);
            let chars = i64::from_le_bytes(buf);
            swar_parse_temp(chars, s.len() as u32 + 1)
        }
        assert_eq!(parse(b"0.0"), 0);
        assert_eq!(parse(b"1.0"), 10);
        assert_eq!(parse(b"1.2"), 12);
        assert_eq!(parse(b"-1.2"), -12);
        assert_eq!(parse(b"12.3"), 123);
        assert_eq!(parse(b"-12.3"), -123);
    }

    #[test]
    fn test_neon_find_delimiters() {
        let s = b";bc\ndefgh;\n0123;5678\n;123;;\n\n12;;bc\ndefgh;\n0123;5678\n;123;;\n\n12;";
        let mut matches = unsafe {
            let chunk = aarch64::vld4q_u8(s.as_ptr());
            neon_find_delimiters_u8x64(chunk)
        };
        println!("{:064b}", matches);
        let mut indices = vec![];
        while matches != 0 {
            let i = matches.trailing_zeros();
            indices.push(i);
            matches &= matches - 1;
        }
        assert_eq!(
            vec![
                0, 3, 9, 10, 15, 20, 21, 25, 26, 27, 28, 31, 32, 35, 41, 42, 47, 52, 53, 57, 58,
                59, 60, 63
            ],
            indices
        );
    }

    #[test]
    fn test_neon_name_equal() {
        let name0 = Name::new(b"abcdefghijk                     ");
        let name1 = Name::new(b"abcdefghijz                     ");
        let name2 = Name::new(b"abcdefghi                       ");
        let expected = b"abcdefghijk\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        assert!(name0.neon_equal(expected, 11));
        assert!(!name1.neon_equal(expected, 11));
        assert!(!name2.neon_equal(expected, 11));
    }

    #[bench]
    fn bench_swar_find_delimiters(b: &mut Bencher) {
        const fn broadcast(c: u8) -> u128 {
            0x01010101010101010101010101010101 * c as u128
        }
        fn swar_find_delimiters(chunk: u128) -> u128 {
            let diff_semi = chunk ^ broadcast(b';');
            let diff_newline = chunk ^ broadcast(b'\n');
            let diff = diff_semi & diff_newline;
            diff.wrapping_sub(broadcast(0x01)) & (!diff & broadcast(0x80))
        }

        #[repr(align(64))]
        struct Aligned {
            bytes: [u8; 64],
        }
        let s = Aligned {
            bytes: *b";bc\ndefgh;\n0123;;bc\ndefgh;\n0123;;bc\ndefgh;\n0123;;bc\ndefgh;\n0123;",
        };
        b.iter(|| {
            for _ in 0..black_box(1000) {
                let chunk = unsafe { transmute::<&[u8; 64], &[u128; 4]>(&s.bytes) };
                black_box(swar_find_delimiters(chunk[0]));
                black_box(swar_find_delimiters(chunk[1]));
                black_box(swar_find_delimiters(chunk[2]));
                black_box(swar_find_delimiters(chunk[3]));
            }
        });
    }

    #[bench]
    fn bench_neon_find_delimiters(b: &mut Bencher) {
        #[repr(align(64))]
        struct Aligned {
            bytes: [u8; 64],
        }
        let s = Aligned {
            bytes: *b";bc\ndefgh;\n0123;;bc\ndefgh;\n0123;;bc\ndefgh;\n0123;;bc\ndefgh;\n0123;",
        };
        b.iter(|| unsafe {
            for _ in 0..black_box(1000) {
                let chunk = aarch64::vld4q_u8(s.bytes.as_ptr());
                black_box(neon_find_delimiters_u8x64(chunk));
            }
        });
    }
}
