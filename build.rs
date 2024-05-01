use std::{
    io::{self, Write},
    process::Command,
};

fn build_metal_kernel(path: &str) {
    println!("cargo::rerun-if-changed={}", path);
    let metallib_output = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-frecord-sources",
            "-o",
            &format!("{}lib", path),
            path,
        ])
        .output()
        .expect("Building metallib failed");
    io::stdout().write_all(&metallib_output.stdout).unwrap();
    io::stderr().write_all(&metallib_output.stderr).unwrap();
    assert!(metallib_output.status.success());
}

fn main() {
    build_metal_kernel("src/bin/gpu_bandwidth_limit/kernel.metal");
    build_metal_kernel("src/bin/gpu_baseline/kernel.metal");
    build_metal_kernel("src/bin/opt0_threadgroup_memory/kernel.metal");
    build_metal_kernel("src/bin/opt1_reduce_buffer_access/kernel.metal");
}
