use std::{fs::File, path::PathBuf};

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use one_billion_row::{cpu, mmap, BUF_EXCESS};

pub fn benchmark(c: &mut Criterion) {
    let file = File::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../1brc/data/measurements-1m.txt"),
    )
    .unwrap();
    let (measurements, len) = mmap::<BUF_EXCESS>(&file);

    let mut group = c.benchmark_group("1brc");
    group.throughput(Throughput::Bytes(measurements.len() as u64));

    group.bench_function("cpu_baseline", |b| {
        let measurements = unsafe { std::str::from_utf8_unchecked(&measurements[..len]) };
        b.iter(|| black_box(cpu::baseline::process(measurements)))
    });
    group.bench_function("cpu_01", |b| {
        b.iter(|| black_box(cpu::opt01::process(&measurements, len)))
    });
    group.bench_function("cpu_02", |b| {
        b.iter(|| black_box(cpu::opt02::process(&measurements, len)))
    });
    group.bench_function("cpu_03", |b| {
        b.iter(|| black_box(cpu::opt03::process(&measurements, len)))
    });
    group.bench_function("cpu_04", |b| {
        b.iter(|| black_box(cpu::opt04::process(&measurements, len)))
    });
    group.bench_function("cpu_05", |b| {
        b.iter(|| black_box(cpu::opt05::process(&measurements, len)))
    });
    group.bench_function("cpu_06", |b| {
        b.iter(|| black_box(cpu::opt06::process(&measurements, len)))
    });
    group.bench_function("cpu_07", |b| {
        b.iter(|| black_box(cpu::opt07::process(&measurements, len)))
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
