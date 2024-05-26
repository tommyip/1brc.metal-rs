//! Optimizations
//! 1. Use &[u8] instead of &str
//! 2. Parse temp as i32 with pattern matching

use crate::{Station, Stations};

pub fn process<'a>(buf: &'a [u8], len: usize) -> Stations<'a> {
    let buf = &buf[..len];
    let mut stations = Stations::default();

    for line in buf.strip_suffix(&[b'\n']).unwrap().split(|&c| c == b'\n') {
        let semi_idx = line.iter().position(|&c| c == b';').unwrap();
        let name = &line[..semi_idx];
        let mut temp = &line[semi_idx + 1..];
        let mut sign = 1;
        if temp[0] == b'-' {
            temp = &temp[1..];
            sign = -1;
        }
        let temp = sign
            * match temp {
                [b, b'.', c] => 10 * (b - b'0') as i32 + (c - b'0') as i32,
                [a, b, b'.', c] => {
                    100 * (a - b'0') as i32 + 10 * (b - b'0') as i32 + (c - b'0') as i32
                }
                _ => unreachable!(),
            };
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
