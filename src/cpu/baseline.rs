use crate::{Station, Stations};

pub fn process<'a>(buf: &'a str) -> Stations<'a> {
    let mut stations = Stations::default();

    for line in buf.trim_end_matches('\n').split_terminator('\n') {
        let (name, temp) = line.split_once(';').unwrap();
        let temp = (temp.parse::<f32>().unwrap() * 10.) as i32;
        stations
            .inner
            .entry(name.as_bytes())
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
