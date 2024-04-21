use std::{
    io::{self, Write},
    process::Command,
};

fn build_metal_kernel(path: &str) {
    println!("cargo::rerun-if-changed={}", path);
    let metalar_output = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            path,
            "-o",
            &format!("{}ar", path),
        ])
        .output()
        .expect("Building metalar failed");
    io::stdout().write_all(&metalar_output.stdout).unwrap();
    io::stderr().write_all(&metalar_output.stderr).unwrap();
    assert!(metalar_output.status.success());
    let metallib_output = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metallib",
            &format!("{}ar", path),
            "-o",
            &format!("{}lib", path),
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
}
