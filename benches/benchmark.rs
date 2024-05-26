use std::{fs, path::PathBuf};

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use one_billion_row::cpu;

pub fn benchmark(c: &mut Criterion) {
    let measurements_str = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../1brc/data/measurements-1m.txt"),
    )
    .expect("Missing 1M line benchmark measurements file");
    let measurements = measurements_str.as_bytes();

    let mut group = c.benchmark_group("1brc");
    group.throughput(Throughput::Bytes(measurements.len() as u64));

    group.bench_function("cpu_baseline", |b| {
        b.iter(|| black_box(cpu::baseline::process(&measurements_str)))
    });
    group.bench_function("cpu_01", |b| {
        b.iter(|| black_box(cpu::opt01::process(&measurements)))
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
