//! Optimizations
//! 1. SWAR parse `;`
//! 2. Mmap pass end of file to read 8 bytes at a time

use crate::{Station, Stations};

fn swar_find_semi(chars: u64) -> u64 {
    let diff = chars ^ 0x3B3B3B3B3B3B3B3B;
    (diff.wrapping_sub(0x0101010101010101)) & (!diff & 0x8080808080808080)
}

pub fn process<'a>(buf: &'a [u8], len: usize) -> Stations<'a> {
    let mut stations = Stations::default();

    let mut i = 0;
    let mut chars_buf = [0u8; 8];
    while i < len {
        let name_idx = i;
        let semi_idx = loop {
            chars_buf.copy_from_slice(&buf[i..i + 8]);
            let semi_bits = swar_find_semi(u64::from_le_bytes(chars_buf));
            if semi_bits != 0 {
                break i + (semi_bits.trailing_zeros() >> 3) as usize;
            }
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
            [b, b'.', c, _] => (10 * (b - b'0') as i32 + (c - b'0') as i32, 3),
            [a, b, b'.', c] => (
                100 * (a - b'0') as i32 + 10 * (b - b'0') as i32 + (c - b'0') as i32,
                4,
            ),
            _ => unreachable!(),
        };
        let temp = sign * abs_temp;
        i += temp_len + 1;

        stations
            .inner
            .entry(name)
            .and_modify(|station| station.update(temp))
            .or_insert(Station::new(temp));
    }

    stations
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
