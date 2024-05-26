use std::{env, fs::File};

use memmap2::Mmap;
use one_billion_row::cpu;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
    let mmap = unsafe { &Mmap::map(file).unwrap() };

    let stations = cpu::opt02::process(&mmap[..]);

    println!("{}", stations);
}
