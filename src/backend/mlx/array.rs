// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Safe Rust wrapper around MLX arrays.
//!
//! `MlxArray` provides RAII memory management (Drop) and safe methods
//! for creating, manipulating, and reading MLX arrays.

use super::ffi;
use super::stream::default_stream;
use std::fmt;

/// Safe wrapper around an MLX array handle.
///
/// MLX arrays are lazily evaluated. Operations build a computation graph
/// that is only materialized when `eval()` is called (or implicitly when
/// reading data).
pub struct MlxArray {
    pub(crate) ptr: ffi::mlx_array,
}

// MLX arrays are reference-counted and thread-safe.
unsafe impl Send for MlxArray {}
unsafe impl Sync for MlxArray {}

impl Drop for MlxArray {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::mlx_array_free(self.ptr) };
        }
    }
}

impl Clone for MlxArray {
    fn clone(&self) -> Self {
        // MLX arrays are reference-counted internally.
        // Creating a new handle increments the reference count.
        let mut new_ptr = unsafe { ffi::mlx_array_new() };
        unsafe { ffi::mlx_array_set(&mut new_ptr, self.ptr) };
        Self { ptr: new_ptr }
    }
}

impl fmt::Debug for MlxArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ndim = self.ndim();
        let shape = self.shape();
        let dtype = self.dtype();
        write!(f, "MlxArray(shape={:?}, dtype={:?})", shape, dtype)
    }
}

impl MlxArray {
    /// Create an uninitialized array handle.
    pub(crate) fn empty() -> Self {
        let ptr = unsafe { ffi::mlx_array_new() };
        Self { ptr }
    }

    /// Wrap a raw MLX array pointer. Takes ownership.
    pub(crate) fn from_raw(ptr: ffi::mlx_array) -> Self {
        Self { ptr }
    }

    // -----------------------------------------------------------------------
    // Creation
    // -----------------------------------------------------------------------

    /// Create array from a float32 slice with the given shape.
    pub fn from_f32(data: &[f32], shape: &[i32]) -> Self {
        let mut res = Self::empty();
        let status = unsafe {
            ffi::mlx_array_new_data(
                &mut res.ptr,
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                ffi::mlx_dtype::MLX_FLOAT32,
            )
        };
        debug_assert_eq!(status, 0, "mlx_array_new_data failed");
        res
    }

    /// Create array from an i64 slice with the given shape.
    pub fn from_i64(data: &[i64], shape: &[i32]) -> Self {
        let mut res = Self::empty();
        let status = unsafe {
            ffi::mlx_array_new_data(
                &mut res.ptr,
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                ffi::mlx_dtype::MLX_INT64,
            )
        };
        debug_assert_eq!(status, 0, "mlx_array_new_data failed");
        res
    }

    /// Create array from an i32 slice with the given shape.
    pub fn from_i32(data: &[i32], shape: &[i32]) -> Self {
        let mut res = Self::empty();
        let status = unsafe {
            ffi::mlx_array_new_data(
                &mut res.ptr,
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                ffi::mlx_dtype::MLX_INT32,
            )
        };
        debug_assert_eq!(status, 0, "mlx_array_new_data failed");
        res
    }

    /// Create array from a bool slice with the given shape.
    pub fn from_bool(data: &[bool], shape: &[i32]) -> Self {
        let mut res = Self::empty();
        let status = unsafe {
            ffi::mlx_array_new_data(
                &mut res.ptr,
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                ffi::mlx_dtype::MLX_BOOL,
            )
        };
        debug_assert_eq!(status, 0, "mlx_array_new_data failed");
        res
    }

    /// Create a scalar float32 array.
    pub fn scalar_f32(val: f32) -> Self {
        let ptr = unsafe { ffi::mlx_array_new_float(val) };
        Self::from_raw(ptr)
    }

    /// Create a scalar int32 array.
    pub fn scalar_i32(val: i32) -> Self {
        let ptr = unsafe { ffi::mlx_array_new_int(val) };
        Self::from_raw(ptr)
    }

    /// Create a scalar bool array.
    pub fn scalar_bool(val: bool) -> Self {
        let ptr = unsafe { ffi::mlx_array_new_bool(val) };
        Self::from_raw(ptr)
    }

    /// Create a zeros array with the given shape and dtype.
    pub fn zeros(shape: &[i32], dtype: ffi::mlx_dtype) -> Self {
        let mut res = Self::empty();
        let s = default_stream();
        unsafe {
            ffi::mlx_zeros(
                &mut res.ptr,
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
                s,
            );
        }
        res
    }

    /// Create a ones array with the given shape and dtype.
    pub fn ones(shape: &[i32], dtype: ffi::mlx_dtype) -> Self {
        let mut res = Self::empty();
        let s = default_stream();
        unsafe {
            ffi::mlx_ones(
                &mut res.ptr,
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
                s,
            );
        }
        res
    }

    /// Create an arange array.
    pub fn arange(start: f64, stop: f64, step: f64, dtype: ffi::mlx_dtype) -> Self {
        let mut res = Self::empty();
        let s = default_stream();
        unsafe { ffi::mlx_arange(&mut res.ptr, start, stop, step, dtype, s) };
        res
    }

    /// Create a full array (all elements set to `val`).
    pub fn full(shape: &[i32], val: &MlxArray, dtype: ffi::mlx_dtype) -> Self {
        let mut res = Self::empty();
        let s = default_stream();
        unsafe {
            ffi::mlx_full(
                &mut res.ptr,
                shape.as_ptr(),
                shape.len() as i32,
                val.ptr,
                dtype,
                s,
            );
        }
        res
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    /// Number of dimensions.
    pub fn ndim(&self) -> i32 {
        unsafe { ffi::mlx_array_ndim(self.ptr) }
    }

    /// Shape along a specific dimension.
    pub fn shape_dim(&self, dim: i32) -> i32 {
        unsafe { ffi::mlx_array_shape(self.ptr, dim) }
    }

    /// Full shape as a vector.
    pub fn shape(&self) -> Vec<i32> {
        let ndim = self.ndim();
        (0..ndim).map(|i| self.shape_dim(i)).collect()
    }

    /// Total number of elements.
    pub fn size(&self) -> i32 {
        unsafe { ffi::mlx_array_size(self.ptr) }
    }

    /// Data type.
    pub fn dtype(&self) -> ffi::mlx_dtype {
        unsafe { ffi::mlx_array_dtype(self.ptr) }
    }

    // -----------------------------------------------------------------------
    // Evaluation and data access
    // -----------------------------------------------------------------------

    /// Evaluate the array (materialize the computation graph).
    /// Must be called before reading data.
    pub fn eval(&self) {
        let status = unsafe { ffi::mlx_array_eval(self.ptr) };
        debug_assert_eq!(status, 0, "mlx_array_eval failed");
    }

    /// Read float32 data. Calls eval automatically.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        self.eval();
        let size = self.size() as usize;
        let ptr = unsafe { ffi::mlx_array_data_float32(self.ptr) };
        assert!(!ptr.is_null(), "data pointer is null after eval");
        unsafe { std::slice::from_raw_parts(ptr, size).to_vec() }
    }

    /// Read int64 data. Calls eval automatically.
    pub fn to_vec_i64(&self) -> Vec<i64> {
        self.eval();
        let size = self.size() as usize;
        let ptr = unsafe { ffi::mlx_array_data_int64(self.ptr) };
        assert!(!ptr.is_null(), "data pointer is null after eval");
        unsafe { std::slice::from_raw_parts(ptr, size).to_vec() }
    }

    /// Read int32 data. Calls eval automatically.
    pub fn to_vec_i32(&self) -> Vec<i32> {
        self.eval();
        let size = self.size() as usize;
        let ptr = unsafe { ffi::mlx_array_data_int32(self.ptr) };
        assert!(!ptr.is_null(), "data pointer is null after eval");
        unsafe { std::slice::from_raw_parts(ptr, size).to_vec() }
    }

    /// Read a scalar float32 value. Calls eval automatically.
    pub fn item_f32(&self) -> f32 {
        self.eval();
        unsafe { ffi::mlx_array_item_float32(self.ptr) }
    }

    /// Read a scalar int64 value. Calls eval automatically.
    pub fn item_i64(&self) -> i64 {
        self.eval();
        unsafe { ffi::mlx_array_item_int64(self.ptr) }
    }

    /// Read a scalar int32 value. Calls eval automatically.
    pub fn item_i32(&self) -> i32 {
        self.eval();
        unsafe { ffi::mlx_array_item_int32(self.ptr) }
    }

    // -----------------------------------------------------------------------
    // Type conversion
    // -----------------------------------------------------------------------

    /// Cast array to a different dtype.
    pub fn astype(&self, dtype: ffi::mlx_dtype) -> Self {
        let mut res = Self::empty();
        let s = default_stream();
        unsafe { ffi::mlx_astype(&mut res.ptr, self.ptr, dtype, s) };
        res
    }
}
