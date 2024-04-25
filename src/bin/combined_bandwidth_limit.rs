use std::{env, ffi, fs::File, mem::size_of, path::PathBuf, thread};

use memmap2::Mmap;
use metal::{self, objc::rc::autoreleasepool, Device, MTLResourceOptions, MTLSize};

const CPU_GPU_RATIO: f32 = 0.5;
const GPU_CHUNK_SIZE: u64 = 2 * 1024;
const U64_SIZE: u64 = size_of::<u64>() as u64;

fn cpu_impl(chunk: &[u8]) -> u64 {
    chunk.into_iter().map(|c| *c as u64).sum::<u64>()
}

fn gpu_impl(chunk: &[u8]) -> u64 {
    let n_total_threads = (chunk.len() as u64).div_ceil(GPU_CHUNK_SIZE);
    println!("n_total_threads {}", n_total_threads);
    let sums = vec![0u64; n_total_threads as usize];

    autoreleasepool(|| {
        let device = &Device::system_default().expect("No Metal device found");

        let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/gpu_bandwidth_limit/kernel.metallib");
        let library = device.new_library_with_file(library_path).unwrap();
        let kernel = library.get_function("blackbox", None).unwrap();
        let pipeline_state = &device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        let max_buf_len = device.max_buffer_length();
        let complete_chunk_len = max_buf_len.div_ceil(GPU_CHUNK_SIZE);

        let mtl_res = &device.new_buffer_with_bytes_no_copy(
            sums.as_ptr() as *const ffi::c_void,
            sums.len() as u64 * U64_SIZE,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let threads_per_threadgroup =
            MTLSize::new(pipeline_state.max_total_threads_per_threadgroup(), 1, 1);

        let cmd_q = &device.new_command_queue();

        for (i, chunk) in chunk.chunks(max_buf_len as usize).enumerate() {
            let mtl_chunk = device.new_buffer_with_bytes_no_copy(
                chunk.as_ptr() as *const ffi::c_void,
                chunk.len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let thread_chunk_len_ptr = (&GPU_CHUNK_SIZE as *const u64) as *const ffi::c_void;

            let chunk_len = chunk.len() as u64;
            let chunk_len_ptr = (&chunk_len as *const u64) as *const ffi::c_void;
            let n_threads = chunk_len.div_ceil(GPU_CHUNK_SIZE);
            let threads_per_grid = MTLSize::new(n_threads, 1, 1);

            let cmd_buf = cmd_q.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&mtl_chunk), 0);
            encoder.set_buffer(1, Some(&mtl_res), complete_chunk_len * U64_SIZE * i as u64);
            encoder.set_bytes(2, U64_SIZE, chunk_len_ptr);
            encoder.set_bytes(3, U64_SIZE, thread_chunk_len_ptr);

            encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            encoder.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }

        sums.iter().sum::<u64>()
    })
}

fn main() {
    let measurements_path = env::args().skip(1).next().expect("Missing path");
    let file = &File::open(measurements_path).unwrap();
    let mmap = unsafe { &Mmap::map(file).unwrap() };
    let split_offset = (mmap.len() as f32 * CPU_GPU_RATIO) as usize;
    let (cpu_mmap, gpu_mmap) = mmap.split_at(split_offset);

    let sum = thread::scope(|s| {
        let mut sums = vec![];
        sums.push(s.spawn(|| gpu_impl(gpu_mmap)));
        for chunk in cpu_mmap.chunks(
            cpu_mmap
                .len()
                .div_ceil(thread::available_parallelism().unwrap().get()),
        ) {
            sums.push(s.spawn(|| cpu_impl(chunk)));
        }
        sums.into_iter().map(|t| t.join().unwrap()).sum::<u64>()
    });

    println!("{}", sum);
}
