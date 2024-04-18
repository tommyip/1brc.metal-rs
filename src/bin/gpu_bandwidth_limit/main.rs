use std::{
    env, ffi,
    fs::File,
    mem::size_of,
    os::unix::fs::FileExt,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use metal::{self, objc::rc::autoreleasepool, Device, MTLResourceOptions, MTLSize, NSUInteger};

const LOAD_CHUNK_SIZE: u64 = 16 * 1024 * 1024;
const GPU_CHUNK_SIZE: u64 = 1 * 1024;

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
    let file_len = file.metadata().unwrap().len();

    let n_cpus = thread::available_parallelism().unwrap().get();

    let n_loads = file_len.div_ceil(LOAD_CHUNK_SIZE);
    let n_threads_per_load = LOAD_CHUNK_SIZE.div_ceil(GPU_CHUNK_SIZE);
    let n_total_threads = {
        let mut total = (file_len / LOAD_CHUNK_SIZE) * n_threads_per_load;
        if file_len % LOAD_CHUNK_SIZE != 0 {
            total += (file_len % LOAD_CHUNK_SIZE).div_ceil(GPU_CHUNK_SIZE);
        }
        total
    };
    println!(
        "n loads {}, n gpu threads {}, n threads per load {}",
        n_loads, n_total_threads, n_threads_per_load
    );
    let u64_size = size_of::<u64>() as NSUInteger;

    let res = vec![0u64; n_total_threads as usize];

    autoreleasepool(|| {
        let device = &Device::system_default().expect("No Metal device found");

        let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/gpu_bandwidth_limit/kernel.metallib");
        let library = device.new_library_with_file(library_path).unwrap();
        let kernel = library.get_function("blackbox", None).unwrap();
        let pipeline_state = &device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        let mtl_res = &device.new_buffer_with_bytes_no_copy(
            res.as_ptr() as *const ffi::c_void,
            res.len() as u64 * u64_size,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let threads_per_threadgroup =
            MTLSize::new(pipeline_state.max_total_threads_per_threadgroup(), 1, 1);

        let cmd_q = &device.new_command_queue();

        let load_i = &AtomicU64::new(0);

        thread::scope(|s| {
            for _ in 0..n_cpus {
                s.spawn(move || {
                    let mut chunk = vec![0; LOAD_CHUNK_SIZE as usize];
                    let mtl_chunk = device.new_buffer_with_bytes_no_copy(
                        chunk.as_ptr() as *const ffi::c_void,
                        chunk.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    );
                    let thread_chunk_len_ptr =
                        (&GPU_CHUNK_SIZE as *const u64) as *const ffi::c_void;

                    loop {
                        let i = load_i.fetch_add(1, Ordering::Relaxed);
                        if i >= n_loads {
                            break;
                        }

                        let offset = LOAD_CHUNK_SIZE * i;
                        let chunk_len = file.read_at(&mut chunk, offset).unwrap() as u64;

                        let n_threads = chunk_len.div_ceil(GPU_CHUNK_SIZE);
                        let threads_per_grid = MTLSize::new(n_threads, 1, 1);
                        let chunk_len_ptr = (&chunk_len as *const u64) as *const ffi::c_void;

                        let cmd_buf = cmd_q.new_command_buffer();
                        let encoder = cmd_buf.new_compute_command_encoder();
                        encoder.set_compute_pipeline_state(&pipeline_state);
                        encoder.set_buffer(0, Some(&mtl_chunk), 0);
                        encoder.set_buffer(1, Some(&mtl_res), n_threads_per_load * i * u64_size);
                        encoder.set_bytes(2, u64_size, chunk_len_ptr);
                        encoder.set_bytes(3, u64_size, thread_chunk_len_ptr);

                        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
                        encoder.end_encoding();

                        cmd_buf.commit();
                        cmd_buf.wait_until_completed();
                    }
                });
            }
        });
    });

    println!("{:?} {:?}", &res[..20], &res[res.len() - 20..]);
}
