pub mod gpu_baseline;

use std::{collections::HashMap, env, ffi, fmt, mem::size_of};

use metal::MTLResourceOptions;

pub const U64_SIZE: u64 = size_of::<u64>() as u64;
pub const I32_SIZE: u64 = size_of::<i32>() as u64;
pub const U32_SIZE: u64 = size_of::<u32>() as u64;

/// https://stackoverflow.com/a/28124775
fn round_to_positive(x: f32) -> f32 {
    let y = x.floor();
    if x == y {
        x
    } else {
        let z = (2.0 * x - y).floor();
        z.copysign(x)
    }
}

pub struct Station {
    min: i32,
    max: i32,
    sum: i32,
    count: i32,
}

#[derive(Default)]
pub struct Stations<'a> {
    inner: HashMap<&'a str, Station>,
}

impl fmt::Display for Station {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let min = self.min as f32 / 10.;
        let max = self.max as f32 / 10.;
        let mean = round_to_positive(((self.sum as f32 / 10.) / self.count as f32) * 10.) / 10.;
        f.write_fmt(format_args!("{:.1}/{:.1}/{:.1}", min, mean, max))
    }
}

impl<'a> fmt::Display for Stations<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut names = self.inner.keys().collect::<Vec<_>>();
        names.sort_unstable();
        for (i, name) in names.into_iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            let station = &self.inner[name];
            f.write_fmt(format_args!("{}={}", name, station))?;
        }
        f.write_str("}")
    }
}

pub fn device_buffer<T>(device: &metal::Device, buf: &[T]) -> metal::Buffer {
    device.new_buffer_with_bytes_no_copy(
        buf.as_ptr() as *const ffi::c_void,
        (buf.len() * size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    )
}

pub struct MetalCaptureGuard;

impl Drop for MetalCaptureGuard {
    fn drop(&mut self) {
        metal::CaptureManager::shared().stop_capture();
    }
}

pub fn metal_frame_capture(device: &metal::Device, output_url: &str) -> Option<MetalCaptureGuard> {
    if !env::var("METAL_CAPTURE_ENABLED")
        .ok()
        .is_some_and(|x| &x == "1")
    {
        return None;
    }
    let capture_manager = metal::CaptureManager::shared();
    let capture_descriptor = metal::CaptureDescriptor::new();
    capture_descriptor.set_capture_device(&device);
    capture_descriptor.set_output_url(output_url);
    capture_descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);

    capture_manager.start_capture(&capture_descriptor).unwrap();

    Some(MetalCaptureGuard)
}
