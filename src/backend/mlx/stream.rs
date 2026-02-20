// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! MLX device and stream management.
//!
//! MLX operations are dispatched to a device (CPU or GPU/Metal) via a stream.
//! On Apple Silicon with unified memory, arrays can be used by any device
//! without explicit transfers.

use super::ffi;
use std::sync::OnceLock;

/// The default stream used for all MLX operations.
/// Initialized once on first use.
static DEFAULT_STREAM: OnceLock<MlxStream> = OnceLock::new();

/// Safe wrapper around an MLX stream handle.
pub struct MlxStream {
    pub(crate) ptr: ffi::mlx_stream,
}

// MlxStream is created once and lives for the program duration.
// The underlying MLX stream is thread-safe.
unsafe impl Send for MlxStream {}
unsafe impl Sync for MlxStream {}

impl Drop for MlxStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::mlx_stream_free(self.ptr) };
        }
    }
}

/// Initialize the MLX backend with the given device type.
/// Should be called once at startup. Sets the default device and creates
/// the default stream.
///
/// If `use_gpu` is true and Metal is available, uses the GPU.
/// Otherwise falls back to CPU.
pub fn init_mlx(use_gpu: bool) {
    DEFAULT_STREAM.get_or_init(|| {
        let device_type = if use_gpu {
            let mut available = false;
            unsafe { ffi::mlx_metal_is_available(&mut available) };
            if available {
                ffi::mlx_device_type::MLX_GPU
            } else {
                eprintln!("Warning: Metal GPU not available, falling back to CPU");
                ffi::mlx_device_type::MLX_CPU
            }
        } else {
            ffi::mlx_device_type::MLX_CPU
        };

        let device = unsafe { ffi::mlx_device_new_type(device_type, 0) };
        unsafe { ffi::mlx_set_default_device(device) };

        let stream = unsafe { ffi::mlx_stream_new_device(device) };
        unsafe { ffi::mlx_device_free(device) };

        MlxStream { ptr: stream }
    });
}

/// Get the default stream for operations.
/// Panics if `init_mlx` has not been called.
pub fn default_stream() -> ffi::mlx_stream {
    DEFAULT_STREAM
        .get()
        .expect("MLX not initialized. Call init_mlx() first.")
        .ptr
}

/// Synchronize the default stream (wait for all pending operations).
pub fn synchronize() {
    let stream = default_stream();
    unsafe { ffi::mlx_synchronize(stream) };
}
