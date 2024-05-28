use std::{env, fs::File};

use one_billion_row::{cpu, mmap_aligned, BUF_ALIGNMENT};

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = File::open(measurements_path).unwrap();
    let (mmap, len) = mmap_aligned::<BUF_ALIGNMENT>(&file);

    let stations = cpu::opt07::process(&mmap[..], len);

    println!("{}", stations);
}
