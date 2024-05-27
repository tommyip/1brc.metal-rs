use std::{env, fs::File};

use memmap2::MmapOptions;
use one_billion_row::{cpu, MMAP_EXCESS};

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let mmap = unsafe { MmapOptions::new().len(len + MMAP_EXCESS).map(file).unwrap() };

    let stations = cpu::opt07::process(&mmap[..], len);

    println!("{}", stations);
}
