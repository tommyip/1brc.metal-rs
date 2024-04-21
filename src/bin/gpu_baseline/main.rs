use std::{
    env, ffi,
    fs::File,
    mem::size_of,
    ops::Range,
    os::unix::fs::FileExt,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use metal::{objc::rc::autoreleasepool, Device, MTLResourceOptions, MTLSize, NSUInteger};

const MAX_LINE_LEN: u64 = 107;
/// (C)PU chunk length (how much to load into memory at once)
const CCHUNK_LEN: u64 = 16 * 1024 * 1024;
/// CPU chunk length taking into account excess space needed to reach the next newline
const CCHUNK_WITH_EXCESS_LEN: u64 = CCHUNK_LEN + MAX_LINE_LEN - 1;
/// (M)etal/GPU chunk length (how much bytes to process by a kernel)
const MCHUNK_LEN: u64 = 1024;
const MCHUNKS_PER_CCHUNK: u64 = {
    assert!(CCHUNK_LEN % MCHUNK_LEN == 0);
    CCHUNK_LEN / MCHUNK_LEN
};

fn is_newline(c: &u8) -> bool {
    *c == b'\n'
}

/// Read `chunk_idx`-th CPU chunk and aligned start and end to the next newline.
/// Return the range within `buf` containing the aligned chunk.
fn get_aligned_cchunk<'a, const CCHUNK_LEN: u64>(
    file: &File,
    cchunk_idx: u64,
    n_cchunks: u64,
    buf: &'a mut [u8],
) -> Range<usize> {
    let offset = CCHUNK_LEN * cchunk_idx;
    let len = file.read_at(buf, offset).unwrap();
    let start = if cchunk_idx == 0 {
        0
    } else {
        buf.iter().position(is_newline).unwrap() + 1
    };
    let end = if cchunk_idx == n_cchunks - 1 {
        len
    } else {
        buf[CCHUNK_LEN as usize..]
            .iter()
            .position(is_newline)
            .unwrap()
            + CCHUNK_LEN as usize
            + 1
    };
    start..end
}

/// After aligning a CPU chunk to newline delimiters its length might not be
/// divisible by the GPU chunk length. Split off the chunk's tail starting after
/// the last complete GPU chunk's ending newline.
fn split_cchunk_excess<'a, const MCHUNK_LEN: u64>(chunk: &'a [u8]) -> (&'a [u8], &'a [u8]) {
    let excess = chunk.len() % MCHUNK_LEN as usize;
    if excess == 0 {
        return (chunk, &[]);
    }
    let last_mchunk_offset = chunk.len() - excess;
    if let Some(excess_start) = chunk[last_mchunk_offset..]
        .iter()
        .position(is_newline)
        .map(|i| last_mchunk_offset + i + 1)
    {
        chunk.split_at(excess_start)
    } else {
        // `last_mchunk_offset` is in the middle of the last line, the second last
        // mchunk will consume the whole of the last line.
        (chunk, &[])
    }
}

fn histogram(buf: &[u8]) -> u64 {
    buf.len() as u64
}

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
    let file_len = file.metadata().unwrap().len();
    let n_cchunks = file_len.div_ceil(CCHUNK_LEN);
    // This is over-allocated, slots at the end of each CPU chunk
    // might not be used.
    let n_mthreads = n_cchunks * MCHUNKS_PER_CCHUNK;
    let res = vec![0u64; n_mthreads as usize];
    let u64_size = size_of::<u64>() as NSUInteger;

    println!("CCHUNK_LEN {}, MCHUNK_LEN {}", CCHUNK_LEN, MCHUNK_LEN);
    println!(
        "file_len {}, n_cchunks {}, n_mthreads {}",
        file_len, n_cchunks, n_mthreads
    );

    let cpu_total = &AtomicU64::new(0);
    let gpu_total = &AtomicU64::new(0);

    autoreleasepool(|| {
        let device = &Device::system_default().unwrap();
        let cmd_q = &device.new_command_queue();
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/gpu_baseline/kernel.metallib");
        let library = device.new_library_with_file(library_path).unwrap();
        let kernel = library.get_function("histogram", None).unwrap();
        let pipeline_state = &device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();
        let threads_per_threadgroup =
            MTLSize::new(pipeline_state.max_total_threads_per_threadgroup(), 1, 1);

        let mres = &device.new_buffer_with_bytes_no_copy(
            res.as_ptr() as *const ffi::c_void,
            res.len() as u64 * u64_size,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let next_cchunk_idx = &AtomicU64::new(0);

        thread::scope(|s| {
            for _ in 0..thread::available_parallelism().unwrap().get() {
                s.spawn(move || {
                    let mut buf = vec![0u8; CCHUNK_WITH_EXCESS_LEN as usize];
                    let mchunk = device.new_buffer_with_bytes_no_copy(
                        buf.as_ptr() as *const ffi::c_void,
                        buf.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    );
                    let mchunk_len_ptr = (&MCHUNK_LEN as *const u64) as *const ffi::c_void;

                    // Process next CPU chunk
                    loop {
                        let cchunk_idx = next_cchunk_idx.fetch_add(1, Ordering::Relaxed);
                        if cchunk_idx >= n_cchunks {
                            break;
                        }

                        let cchunk_range =
                            get_aligned_cchunk::<CCHUNK_LEN>(file, cchunk_idx, n_cchunks, &mut buf);
                        cpu_total.fetch_add(cchunk_range.len() as u64, Ordering::Relaxed);
                        let cchunk_start = cchunk_range.start as u64;
                        let (cchunk, excess) =
                            split_cchunk_excess::<MCHUNK_LEN>(&buf[cchunk_range.clone()]);
                        let cchunk_len = cchunk.len() as u64;

                        if !excess.is_empty() {
                            let excess_res = histogram(excess);
                            gpu_total.fetch_add(excess_res, Ordering::Relaxed);
                        }

                        if cchunk_len == 0 {
                            continue;
                        }

                        let n_threads = cchunk_len / MCHUNK_LEN;
                        let threads_per_grid = MTLSize::new(n_threads, 1, 1);
                        let cchunk_len_ptr = (&cchunk_len as *const u64) as *const ffi::c_void;
                        let res_offset = MCHUNKS_PER_CCHUNK * cchunk_idx;

                        let cmd_buf = cmd_q.new_command_buffer();
                        let enc = cmd_buf.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(&pipeline_state);
                        enc.set_buffer(0, Some(&mchunk), cchunk_start);
                        enc.set_buffer(1, Some(&mres), res_offset * u64_size);
                        enc.set_bytes(2, u64_size, cchunk_len_ptr);
                        enc.set_bytes(3, u64_size, mchunk_len_ptr);

                        enc.dispatch_threads(threads_per_grid, threads_per_threadgroup);
                        enc.end_encoding();
                        cmd_buf.commit();

                        cmd_buf.wait_until_completed();
                    }
                });
            }
        })
    });

    gpu_total.fetch_add(res.iter().sum::<u64>(), Ordering::Relaxed);

    println!("{:?}", &res[..20]);

    let cpu_total = cpu_total.load(Ordering::Relaxed);
    let gpu_total = gpu_total.load(Ordering::Relaxed);
    println!(
        "CPU total: {}, GPU total: {}, match: {}",
        cpu_total,
        gpu_total,
        cpu_total == gpu_total
    );
}
