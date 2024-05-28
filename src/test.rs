use std::{
    fs::{self, File},
    path::Path,
};

use crate::{mmap, Stations, BUF_EXCESS};

const SAMPLE_NAMES: [&str; 12] = [
    "measurements-1",
    "measurements-10",
    "measurements-10000-unique-keys",
    "measurements-2",
    "measurements-20",
    "measurements-3",
    "measurements-boundaries",
    "measurements-complex-utf8",
    "measurements-dot",
    "measurements-rounding",
    "measurements-short",
    "measurements-shortest",
];

pub fn correctness<F>(process: F)
where
    F: Fn(&[u8], usize) -> Stations<'_>,
{
    let base_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    for sample in SAMPLE_NAMES {
        println!("Sample {}", sample);
        let txt_path = base_path.join(sample).with_extension("txt");
        let out_path = base_path.join(sample).with_extension("out");

        let (mmap, len) = mmap::<BUF_EXCESS>(&File::open(txt_path).unwrap());
        let out = fs::read_to_string(out_path).unwrap();

        let actual = process(&mmap[..], len);
        assert_eq!(format!("{}\n", actual.to_string()), out);
    }
}
