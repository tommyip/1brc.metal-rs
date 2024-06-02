use std::{env, fs::File};

use one_billion_row::{cpu, mmap, BUF_EXCESS};

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = File::open(measurements_path).unwrap();
    let (mmap, len) = mmap::<BUF_EXCESS>(&file);

    let stations = cpu::opt09::process(&mmap[..], len);

    println!("{}", stations);
}
