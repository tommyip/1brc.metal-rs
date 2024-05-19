//! Optimization 4: Split workload between CPU and GPU

use std::{env, fs::File, path::PathBuf};

use one_billion_row::opt4;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");

    let file = File::open(measurements_path).unwrap();
    let metallib_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/opt3_faster_hashing/kernel.metallib");
    opt4::process(&file, |device| {
        device
            .new_library_with_file(&metallib_path)
            .unwrap()
            .get_function("histogram", None)
            .unwrap()
    });
}
