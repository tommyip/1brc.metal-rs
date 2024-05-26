use crate::Stations;

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

const SAMPLES: [Sample; 16] = [
    sample!("measurements-1"),
    sample!("measurements-1"),
    sample!("measurements-10"),
    sample!("measurements-10000-unique-keys"),
    sample!("measurements-10000-unique-keys"),
    sample!("measurements-2"),
    sample!("measurements-2"),
    sample!("measurements-20"),
    sample!("measurements-3"),
    sample!("measurements-3"),
    sample!("measurements-boundaries"),
    sample!("measurements-complex-utf8"),
    sample!("measurements-dot"),
    sample!("measurements-rounding"),
    sample!("measurements-short"),
    sample!("measurements-shortest"),
];

pub fn correctness<'a, F>(process: F)
where
    F: Fn(&'a str) -> Stations<'a>,
{
    for sample in SAMPLES {
        println!("Sample {}", sample.name);
        let actual = process(sample.txt);
        assert_eq!(format!("{}\n", actual.to_string()), sample.out);
    }
}
