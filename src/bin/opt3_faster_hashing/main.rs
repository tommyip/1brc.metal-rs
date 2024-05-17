//! Optimization 3: Faster hashing

use std::{env, fs::File, path::PathBuf};

use one_billion_row::opt3;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");

    let file = File::open(measurements_path).unwrap();
    let metallib_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/opt3_faster_hashing/kernel.metallib");
    opt3::process(&file, |device| {
        device
            .new_library_with_file(&metallib_path)
            .unwrap()
            .get_function("histogram", None)
            .unwrap()
    });
}
