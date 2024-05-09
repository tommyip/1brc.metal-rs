use core::slice;
use std::{ffi, mem::size_of, path::PathBuf};

use metal::{objc::rc::autoreleasepool, Device, MTLSize};
use one_billion_row::device_buffer;

fn swar(buf: &[u8]) {
    const BROADCAST_SEMICOLON: u64 = 0x3B3B3B3B3B3B3B3B;
    const BROADCAST_0X01: u64 = 0x0101010101010101;
    const BROADCAST_0X80: u64 = 0x8080808080808080;
    println!("swar");

    let mut chunk = [0u8; 8];
    chunk.copy_from_slice(&buf[..8]);
    let word = u64::from_le_bytes(chunk);

    let diff = word ^ BROADCAST_SEMICOLON;
    let semi_bits = (diff - BROADCAST_0X01) & (!diff & BROADCAST_0X80);
    let len = semi_bits.trailing_zeros() >> 3;
    println!("{:064b}", word);
    println!("{:064b}", semi_bits);
    println!("len: {}", len);
}

fn u64_to_str(x: u64) -> Option<String> {
    std::str::from_utf8(&x.to_le_bytes())
        .ok()
        .map(|x| x.to_owned())
}

fn main() {
    let metallib_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/mtl_debug/kernel.metallib");

    let unaligned_buf =
        "abcde;-1.5\nabdafdsfdsfs;32.5\ndkfsfdsfdsfdsfdsfds;-12.4\ndsfadsfdsfadsfjdkfjkdsjfkd;0.8\nabce;-3.8\n                     ".as_bytes();
    let (buf_prefix, buf, buf_suffix) = unsafe { unaligned_buf.align_to::<u128>() };
    let buf = unsafe {
        slice::from_raw_parts::<u8>(
            std::mem::transmute(buf.as_ptr()),
            buf.len() * size_of::<u128>() / size_of::<u8>(),
        )
    };
    println!("{} {} {}", buf_prefix.len(), buf.len(), buf_suffix.len());
    let res: Vec<i32> = vec![0; 10];
    autoreleasepool(|| {
        let device = &Device::system_default().unwrap();
        let cmd_q = device.new_command_queue();
        let cmd_buf = cmd_q.new_command_buffer();
        let kernel = device
            .new_library_with_file(&metallib_path)
            .unwrap()
            .get_function("debug", None)
            .unwrap();
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        let device_buf = device_buffer(device, &buf[..]);
        let res_buf = device_buffer(device, &res);

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&device_buf), 0);
        encoder.set_buffer(1, Some(&res_buf), 0);
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            (&(buf.len() as u32) as *const u32) as *const ffi::c_void,
        );

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    });

    println!("{}", std::str::from_utf8(buf).unwrap());
    // println!("{:064b}", res[0]);
    // println!("{:x} {:x} {:x} {:x}", res[0], res[1], res[2], res[3]);
    // println!("{:?}", res.into_iter().map(u64_to_str).collect::<Vec<_>>());
    // println!("{:?}", u64_to_str(res[0]));
    // println!("{:064b} {}", res[1], res[2]);
    println!("{:?}", res);
    // swar(&buf[..]);
}
