//! # GPU baseline
//!
//! Process entire file on the GPU. Each Metal kernel handles a small chunk of
//! the measurements file, parsing it line by line and updating a global hashmap.
//! The kernel is implemented naïvely like one would writing CPU code with
//! no (GPU or otherwise) micro-optimization.

use std::{env, fs::File, path::PathBuf};

use one_billion_row::gpu_baseline::baseline;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = File::open(measurements_path).unwrap();
    let metallib =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/gpu_baseline/kernel.metallib");
    baseline(&file, &metallib);
}
