use crate::{Stations, MMAP_EXCESS};

struct Sample {
    name: &'static str,
    txt: &'static str,
    out: &'static str,
}

macro_rules! sample {
    ($name:literal) => {
        Sample {
            name: $name,
            txt: include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/",
                concat!($name, ".txt")
            )),
            out: include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/",
                concat!($name, ".out")
            )),
        }
    };
}

const SAMPLES: [Sample; 12] = [
    sample!("measurements-1"),
    sample!("measurements-10"),
    sample!("measurements-10000-unique-keys"),
    sample!("measurements-2"),
    sample!("measurements-20"),
    sample!("measurements-3"),
    sample!("measurements-boundaries"),
    sample!("measurements-complex-utf8"),
    sample!("measurements-dot"),
    sample!("measurements-rounding"),
    sample!("measurements-short"),
    sample!("measurements-shortest"),
];

pub fn correctness<F>(process: F)
where
    F: Fn(&[u8], usize) -> Stations<'_>,
{
    for sample in SAMPLES {
        println!("Sample {}", sample.name);
        let mut buf = vec![0u8; sample.txt.len() + MMAP_EXCESS];
        buf[..sample.txt.len()].copy_from_slice(sample.txt.as_bytes());

        let actual = process(&buf, sample.txt.len());
        assert_eq!(format!("{}\n", actual.to_string()), sample.out);
    }
}
