use std::{
    env,
    fs::File,
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use memmap2::Mmap;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
    let mmap = unsafe { &Mmap::map(file).unwrap() };
    let n_threads = thread::available_parallelism().unwrap().get();
    let chunk_size = mmap.len().div_ceil(n_threads);
    let sum = &AtomicU64::new(0);
    thread::scope(|s| {
        for chunk in mmap.chunks(chunk_size) {
            s.spawn(move || {
                let mut local_sum = 0;
                for c in chunk {
                    local_sum += *c as u64;
                }
                sum.fetch_add(local_sum, Ordering::Relaxed);
            });
        }
    });
    println!("{}", sum.load(Ordering::Relaxed));
}
