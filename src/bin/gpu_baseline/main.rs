//! # GPU baseline
//!
//! Process entire file on the GPU. Each Metal kernel handles a small chunk of
//! the measurements file, parsing it line by line and updating a global hashmap.
//! The kernel is implemented naïvely like one would writing CPU code with
//! no (GPU or otherwise) micro-optimization.

use std::{env, fs::File, path::PathBuf};

use metal::{FunctionConstantValues, MTLDataType};

use one_billion_row::{
    c_void,
    gpu_baseline::{process, HASHMAP_LEN},
};

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = File::open(measurements_path).unwrap();
    let metallib_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/gpu_baseline/kernel.metallib");
    let kernel_constants = FunctionConstantValues::new();
    kernel_constants.set_constant_value_with_name(
        c_void(&(HASHMAP_LEN as u32)),
        MTLDataType::UInt,
        "HASHMAP_LEN",
    );
    process(&file, |device| {
        device
            .new_library_with_file(&metallib_path)
            .unwrap()
            .get_function("histogram", Some(kernel_constants))
            .unwrap()
    });
}
