//! Optimizations
//! 1. 4x Instruction Level Parallelism (ILP)

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    mem::transmute,
    ops::BitXor,
};

use crate::{Station, Stations, STATION_NAMES};

const MPH_SIZE: usize = 512;
const PREFIX_SIZE_U64: usize = 4;

struct Reader<'a> {
    buf: &'a [u8],
    i: usize,
    end: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8], i: usize, end: usize) -> Self {
        Self { buf, i, end }
    }

    fn read16(&mut self) -> u128 {
        let buf: &[u8; 16] = unsafe { transmute(self.buf.as_ptr().add(self.i)) };
        u128::from_le_bytes(*buf)
    }

    fn read8(&mut self) -> u64 {
        let buf: &[u8; 8] = unsafe { transmute(self.buf.as_ptr().add(self.i)) };
        u64::from_le_bytes(*buf)
    }

    fn advance(&mut self, incr: usize) {
        self.i += incr;
    }

    fn skip_to_line(&mut self) {
        loop {
            let word = self.read8();
            let newline_bits = swar_find_char::<NEWLINE>(word);
            if newline_bits != 0 {
                let newline_idx = (newline_bits.trailing_zeros() >> 3) as usize;
                self.advance(newline_idx + 1);
                break;
            }
            self.advance(8);
        }
    }

    fn eof(&self) -> bool {
        self.i >= self.end
    }
}

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
    keys: [[u64; PREFIX_SIZE_U64]; MPH_SIZE],
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

const SEMI: u64 = 0x3B3B3B3B3B3B3B3B;
const NEWLINE: u64 = 0x0A0A0A0A0A0A0A0A;

fn swar_find_char<const C: u64>(chars: u64) -> u64 {
    let diff = chars ^ C;
    (diff.wrapping_sub(0x0101010101010101)) & (!diff & 0x8080808080808080)
}

fn swar_find_semi_2x(chars: u128) -> u128 {
    let diff = chars ^ 0x3B3B3B3B3B3B3B3B3B3B3B3B3B3B3B3B;
    (diff.wrapping_sub(0x01010101010101010101010101010101))
        & (!diff & 0x80808080808080808080808080808080)
}

struct Name<'a> {
    prefix: [u64; PREFIX_SIZE_U64],
    name: &'a [u8],
    hash: PerfectHash,
}

/// Read station name and advance reader to the start of the temperature
fn read_name<'a>(rdr: &mut Reader<'a>) -> Name<'a> {
    let start = rdr.i;
    let mut hash = PerfectHash::default();
    let mut prefix = [0u64; PREFIX_SIZE_U64];

    // Special case for first 16 bytes
    let mut prefix0 = rdr.read16();
    let semi_bits = swar_find_semi_2x(prefix0);
    if semi_bits != 0 {
        let lane_id = (semi_bits.trailing_zeros() >> 3) as usize;
        rdr.advance(lane_id + 1);
        prefix0 &= u128::MAX >> ((16 - lane_id) << 3);
        let prefix0_u64x2: &[u64; 2] = unsafe { transmute(&prefix0) };
        hash.write_u64(prefix0_u64x2[0]);
        if lane_id > 8 {
            hash.write_u64(prefix0_u64x2[1]);
        }
        prefix[0] = prefix0_u64x2[0];
        prefix[1] = prefix0_u64x2[1];
    } else {
        let prefix0_u64x2: &[u64; 2] = unsafe { transmute(&prefix0) };
        prefix[0] = prefix0_u64x2[0];
        prefix[1] = prefix0_u64x2[1];
        hash.write_u64(prefix0_u64x2[0]);
        hash.write_u64(prefix0_u64x2[1]);

        rdr.advance(16);
        // Fall back to remaining bytes
        for block in 2.. {
            let chars = rdr.read8();
            let semi_bits = swar_find_char::<SEMI>(chars);
            if semi_bits != 0 {
                let lane_id = (semi_bits.trailing_zeros() >> 3) as usize;
                if lane_id > 0 {
                    let chars = chars & (u64::MAX >> ((8 - lane_id) << 3));
                    hash.write_u64(chars);
                    if block < prefix.len() {
                        prefix[block] = chars;
                    }
                }
                rdr.advance(lane_id + 1);
                break;
            }
            hash.write_u64(chars);
            if block < prefix.len() {
                prefix[block] = chars;
            }
            rdr.advance(8);
        }
    }

    let name = &rdr.buf[start..rdr.i - 1];
    Name { prefix, name, hash }
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

/// Read temperature and advance reader to the next line
fn read_temp(rdr: &mut Reader<'_>) -> i16 {
    let chars = rdr.read8() as i64;
    let dot_pos = (!chars & 0x10101000).trailing_zeros() as usize;
    let temp = swar_parse_temp(chars, dot_pos);
    rdr.advance((dot_pos >> 3) + 3);
    temp
}

struct Records<'a> {
    mph_data: PerfectHashData,
    mph: Vec<Station>,
    fallback: HashMap<&'a [u8], Station, BuildHasherDefault<PerfectHash>>,
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

    fn insert(&mut self, name: Name<'a>, temp: i16) {
        let Name { prefix, name, hash } = name;
        // Try storing in minimal perfect hash table
        if name.len() <= 26 {
            let idx = hash.index();
            if prefix == self.mph_data.keys[idx] {
                self.mph[idx].update(temp);
                return;
            }
        }
        // Fallback to general hashmap
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

    let chunk_len = len / 4;
    let mut rdr3 = Reader::new(buf, chunk_len * 3, len);
    rdr3.skip_to_line();
    let mut rdr2 = Reader::new(buf, chunk_len * 2, rdr3.i);
    rdr2.skip_to_line();
    let mut rdr1 = Reader::new(buf, chunk_len, rdr2.i);
    rdr1.skip_to_line();
    let mut rdr0 = Reader::new(buf, 0, rdr1.i);

    while !(rdr0.eof() || rdr1.eof() || rdr2.eof() || rdr3.eof()) {
        let name0 = read_name(&mut rdr0);
        let name1 = read_name(&mut rdr1);
        let name2 = read_name(&mut rdr2);
        let name3 = read_name(&mut rdr3);

        let temp0 = read_temp(&mut rdr0);
        let temp1 = read_temp(&mut rdr1);
        let temp2 = read_temp(&mut rdr2);
        let temp3 = read_temp(&mut rdr3);

        records.insert(name0, temp0);
        records.insert(name1, temp1);
        records.insert(name2, temp2);
        records.insert(name3, temp3);
    }

    while !rdr0.eof() {
        let name = read_name(&mut rdr0);
        let temp = read_temp(&mut rdr0);
        records.insert(name, temp);
    }
    while !rdr1.eof() {
        let name = read_name(&mut rdr1);
        let temp = read_temp(&mut rdr1);
        records.insert(name, temp);
    }
    while !rdr2.eof() {
        let name = read_name(&mut rdr2);
        let temp = read_temp(&mut rdr2);
        records.insert(name, temp);
    }
    while !rdr3.eof() {
        let name = read_name(&mut rdr3);
        let temp = read_temp(&mut rdr3);
        records.insert(name, temp);
    }

    records.finish()
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::Path};

    use super::*;
    use crate::{mmap, test, BUF_EXCESS};

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

    #[test]
    fn test_reader() {
        let (mmap, len) = mmap::<BUF_EXCESS>(
            &File::open(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests")
                    .join("measurements-10.txt"),
            )
            .unwrap(),
        );
        let mut rdr = Reader::new(&mmap, 0, len);
        assert_eq!(
            std::str::from_utf8(&rdr.read16().to_le_bytes()).unwrap(),
            "Halifax;12.9\nZag"
        );
        rdr.advance(8);
        assert_eq!(
            std::str::from_utf8(&rdr.read8().to_le_bytes()).unwrap(),
            "12.9\nZag"
        );
        rdr.advance(5);
        assert_eq!(
            std::str::from_utf8(&rdr.read16().to_le_bytes()).unwrap(),
            "Zagreb;12.2\nCabo"
        );
        rdr.advance(7);
        assert_eq!(
            std::str::from_utf8(&rdr.read8().to_le_bytes()).unwrap(),
            "12.2\nCab"
        );
        rdr.advance(5);
        assert_eq!(
            std::str::from_utf8(&rdr.read16().to_le_bytes()).unwrap(),
            "Cabo San Lucas;1"
        );
        rdr.advance(16);
        assert_eq!(
            std::str::from_utf8(&rdr.read16().to_le_bytes()).unwrap(),
            "4.9\nAdelaide;15."
        );
    }
}
