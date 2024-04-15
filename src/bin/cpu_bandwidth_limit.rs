use std::{
    env,
    fs::File,
    hint::black_box,
    os::unix::fs::{FileExt, MetadataExt},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

const CHUNK_SIZE: usize = 8 * 1024 * 1024;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
    let n_threads = thread::available_parallelism().unwrap().get();
    let chunk_idx = &AtomicUsize::new(0);
    let file_size = file.metadata().unwrap().size() as usize;
    let n_chunks = file_size.div_ceil(CHUNK_SIZE);
    thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(move || {
                let mut buf = vec![0; CHUNK_SIZE];
                loop {
                    let chunk_idx = chunk_idx.fetch_add(1, Ordering::Relaxed);
                    let offset = CHUNK_SIZE * chunk_idx;
                    let chunk_size = if chunk_idx >= n_chunks {
                        break;
                    } else if chunk_idx == n_chunks - 1 {
                        file_size - offset
                    } else {
                        CHUNK_SIZE
                    };
                    let chunk = &mut buf[..chunk_size];
                    file.read_exact_at(chunk, offset as u64).unwrap();
                    for c in chunk {
                        black_box(c);
                    }
                }
            });
        }
    });
}
