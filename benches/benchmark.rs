use std::{fs::File, path::PathBuf};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use one_billion_row::{cpu, mmap, BUF_EXCESS};

pub fn benchmark(c: &mut Criterion) {
    let file = File::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../1brc/data/measurements-1m.txt"),
    )
    .unwrap();
    let (measurements, len) = mmap::<BUF_EXCESS>(&file);

    let file_10k = File::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../1brc/data/measurements-10k-1m.txt"),
    )
    .unwrap();
    let (measurements_10k, len_10k) = mmap::<BUF_EXCESS>(&file_10k);

    let inputs = [
        ("default", measurements, len),
        ("10k", measurements_10k, len_10k),
    ];

    let mut group = c.benchmark_group("1brc");

    for (name, measurements, len) in inputs {
        macro_rules! bench {
            ($module:path) => {{
                use $module as module;
                group.bench_with_input(
                    BenchmarkId::new(stringify!($module), name),
                    &(&measurements, len),
                    |b, &(m, len)| b.iter(|| module::process(m, len)),
                );
            }};
        }

        group.throughput(Throughput::Bytes(len as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu::baseline", name),
            unsafe { std::str::from_utf8_unchecked(&measurements[..len]) },
            |b, measurements| b.iter(|| cpu::baseline::process(measurements)),
        );
        bench!(cpu::opt01);
        bench!(cpu::opt02);
        bench!(cpu::opt03);
        bench!(cpu::opt04);
        bench!(cpu::opt05);
        bench!(cpu::opt06);
        bench!(cpu::opt07);
        bench!(cpu::opt08);
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
