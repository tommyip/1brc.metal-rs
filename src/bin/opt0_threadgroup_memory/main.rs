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

use metal::{FunctionConstantValues, MTLDataType};

use one_billion_row::{
    c_void,
    gpu_baseline::{process, HASHMAP_LEN},
};

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let reinterpret_atomics = env::var("REINTERPRET_ATOMICS")
        .ok()
        .map(|x| &x == "1")
        .unwrap_or(true);

    let file = File::open(measurements_path).unwrap();
    let metallib_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/opt0_threadgroup_memory/kernel.metallib");
    let kernel_constants = FunctionConstantValues::new();
    kernel_constants.set_constant_value_with_name(
        c_void(&(HASHMAP_LEN as u32)),
        MTLDataType::UInt,
        "G_HASHMAP_LEN",
    );
    kernel_constants.set_constant_value_with_name(
        c_void(&reinterpret_atomics),
        MTLDataType::Bool,
        "REINTERPRET_ATOMICS",
    );
    process(&file, |device| {
        device
            .new_library_with_file(&metallib_path)
            .unwrap()
            .get_function("histogram", Some(kernel_constants))
            .unwrap()
    });
}
