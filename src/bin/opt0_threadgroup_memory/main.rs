//! # Optimization 0: Local histogram accumulation with threadgroup memory
//!
//! Process entire file on the GPU. Each Metal kernel handles a small chunk of
//! the measurements file, parsing it line by line and updating its parent
//! threadgroup local hashmap. After a threadgroup finishes processing its
//! mega-chunk the local hashmap is merged with the global hashmap. This
//! reduces expensive global memory access.
//!
//! Still no micro-optimization is applied.

use std::{env, fs::File, path::PathBuf};

use one_billion_row::gpu_baseline::baseline;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = File::open(measurements_path).unwrap();
    let metallib = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/opt0_threadgroup_memory/kernel.metallib");
    baseline(&file, &metallib);
}
