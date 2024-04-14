use core::fmt;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

struct Station {
    min: f64,
    max: f64,
    sum: f64,
    n: u32,
}

struct Output {
    agg: HashMap<String, Station>,
}

fn round(x: f64) -> f64 {
    let y = x.floor();
    if x == y {
        x
    } else {
        let z = (2.0 * x - y).floor();
        z.copysign(x)
    }
}

impl fmt::Display for Station {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mean = round(((round(self.sum * 10.0) / 10.0) / self.n as f64) * 10.0) / 10.0;
        f.write_fmt(format_args!("{:.1}/{:.1}/{:.1}", self.min, mean, self.max))
    }
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut names = self.agg.keys().collect::<Vec<_>>();
        names.sort_unstable();
        for (i, name) in names.into_iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            let station = &self.agg[name];
            f.write_fmt(format_args!("{}={}", name, station))?;
        }
        f.write_str("}")
    }
}

fn main() {
    let f = File::open("measurements.txt").unwrap();
    let reader = BufReader::new(f);

    let mut agg = HashMap::<String, Station>::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let (name, temp) = line.split_once(';').unwrap();
        let temp = temp.parse::<f64>().unwrap();
        if let Some(station) = agg.get_mut(name) {
            if temp > station.max {
                station.max = temp;
            } else if temp < station.min {
                station.min = temp;
            }
            station.sum += temp;
            station.n += 1;
        } else {
            agg.insert(
                name.to_owned(),
                Station {
                    min: temp,
                    max: temp,
                    sum: temp,
                    n: 1,
                },
            );
        }
    }

    println!("{}", Output { agg });
}
