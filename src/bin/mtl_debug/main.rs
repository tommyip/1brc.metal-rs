#![allow(dead_code)]

use core::slice;
use std::{ffi, mem::size_of, ops::BitXor, path::PathBuf};

use metal::{objc::rc::autoreleasepool, Device, MTLSize};
use one_billion_row::{device_buffer, STATION_NAMES};
use ptr_hash::hash::Hasher;

#[repr(C, align(32))]
struct AlignedPerfectKeys([u8; 512 * 32]);

impl AlignedPerfectKeys {
    fn init() -> Self {
        let mut this = AlignedPerfectKeys([0; 512 * 32]);
        for name in STATION_NAMES {
            let offset = station_slot(name) * 32;
            this.0[offset..offset + name.len()].copy_from_slice(name.as_bytes());
        }
        this
    }
}

#[derive(Clone)]
pub struct FxHash;

impl FxHash {
    const K: u64 = 0x517cc1b727220a95;
}

impl Hasher<&str> for FxHash {
    type H = u64;

    fn hash(x: &&str, _seed: u64) -> Self::H {
        let mut h: u64 = 0;
        for chunk in x.as_bytes().chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            let i = u64::from_le_bytes(buf);
            h = h.rotate_left(5).bitxor(i).wrapping_mul(Self::K);
        }
        h
    }
}

fn station_slot(name: &str) -> usize {
    const C: u64 = 0x517cc1b727220a95;
    const PILOTS: [u8; 78] = [
        50, 110, 71, 13, 18, 27, 21, 14, 10, 16, 1, 14, 6, 11, 0, 2, 17, 2, 1, 4, 68, 79, 21, 0,
        22, 20, 60, 12, 30, 53, 62, 78, 27, 17, 2, 17, 13, 43, 21, 108, 19, 12, 25, 1, 55, 36, 1,
        0, 4, 184, 0, 21, 69, 25, 13, 177, 11, 97, 3, 29, 14, 104, 30, 4, 50, 23, 6, 102, 137, 10,
        227, 32, 29, 21, 7, 4, 244, 0,
    ];
    const REM_C1: u64 = 38;
    const REM_C2: u64 = 132;
    const C3: isize = -55;
    const P1: u64 = 11068046444225730560;

    let hash = FxHash::hash(&name, 0);
    let is_large = hash >= P1;
    let rem = if is_large { REM_C2 } else { REM_C1 };
    let bucket = (is_large as isize * C3) + ((hash as u128 * rem as u128) >> 64) as isize;
    let pilot = PILOTS[bucket as usize];
    return ((C as u128 * (hash ^ C.wrapping_mul(pilot as u64)) as u128) >> 64) as usize
        & ((1 << 9) - 1) as usize;
}

fn swar_find_semi(buf: &[u8]) {
    const BROADCAST_SEMICOLON: u64 = 0x3B3B3B3B3B3B3B3B;
    const BROADCAST_0X01: u64 = 0x0101010101010101;
    const BROADCAST_0X80: u64 = 0x8080808080808080;
    println!("swar find semi");

    let mut chunk = [0u8; 8];
    chunk.copy_from_slice(&buf[..8]);
    let word = u64::from_le_bytes(chunk);

    let diff = word ^ BROADCAST_SEMICOLON;
    let semi_bits = (diff - BROADCAST_0X01) & (!diff & BROADCAST_0X80);
    println!("semi input {:064b}", word);
    println!("semi bits  {:064b}", semi_bits);
    println!("semi pos   {}", semi_bits.trailing_zeros());
}

fn swar_find_dot(buf: &[u8]) {
    const DOT_BITS: u64 = 0x10101000;
    let mut chunk = [0u8; 8];
    chunk.copy_from_slice(&buf[..8]);
    let word = u64::from_le_bytes(chunk);

    let dot_pos = (!word & DOT_BITS).trailing_zeros();
    println!("dot pos {}", dot_pos);
    println!("dot pos / 8 = {}", dot_pos >> 3);
}

fn u64_to_str(x: &u64) -> Option<String> {
    std::str::from_utf8(&x.to_le_bytes())
        .ok()
        .map(|x| x.to_owned())
}

fn main() {
    let metallib_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/mtl_debug/kernel.metallib");

    let unaligned_buf = "             Panama City;24.3
Hanoi;22.4
Surabaya;42.0
Jos;18.6
Fianarantsoa;23.0
Willemstad;32.1
Abha;16.8
Jayapura;29.0
Alexandra;-14.6
Budapest;9.4
Garissa;29.4
Irkutsk;-2.8
Vilnius;-10.1
"
    .as_bytes();
    let (buf_prefix, buf, buf_suffix) = unsafe { unaligned_buf.align_to::<u128>() };
    let buf = unsafe {
        slice::from_raw_parts::<u8>(
            std::mem::transmute(buf.as_ptr()),
            buf.len() * size_of::<u128>() / size_of::<u8>(),
        )
    };
    println!("{} {} {}", buf_prefix.len(), buf.len(), buf_suffix.len());
    let res: Vec<u64> = vec![0; 12];

    let keys = AlignedPerfectKeys::init();
    println!("{}", unsafe { std::str::from_utf8_unchecked(&keys.0) });

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
        let keys_buf = device_buffer(device, &keys.0);
        let res_buf = device_buffer(device, &res);

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&device_buf), 0);
        encoder.set_buffer(1, Some(&keys_buf), 0);
        encoder.set_buffer(2, Some(&res_buf), 0);
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&(buf.len() as u32) as *const u32) as *const ffi::c_void,
        );

        encoder.dispatch_threads(MTLSize::new(512, 1, 1), MTLSize::new(1024, 1, 1));
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    });

    println!("{}", std::str::from_utf8(buf).unwrap());
    // println!("{:?}", res.iter().map(u64_to_str).collect::<Vec<_>>());
    println!("{:?}", res);
    // let expected_slot = names
    //     .into_iter()
    //     .map(|s| station_slot(s) as u64)
    //     .collect::<Vec<_>>();
    // println!("{:?}", expected_slot);
    // assert_eq!(res, expected_slot);
    // swar_find_semi("0;123456".as_bytes());
    // swar_find_dot("1.5\nabcd".as_bytes());
}
