// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Raw FFI declarations for the mlx-c library.
//!
//! These declarations are hand-written to match the mlx-c public C API.
//! They cover the subset needed by this crate: arrays, ops, I/O, devices,
//! streams, fast ML ops, FFT, maps, and vectors.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_float, c_int, c_void};

// ---------------------------------------------------------------------------
// Opaque types
// ---------------------------------------------------------------------------

/// Opaque MLX array handle.
pub type mlx_array = *mut c_void;

/// Opaque MLX stream handle.
pub type mlx_stream = *mut c_void;

/// Opaque MLX device handle.
pub type mlx_device = *mut c_void;

/// Opaque string-to-array map handle (for safetensors loading).
pub type mlx_map_string_to_array = *mut c_void;

/// Opaque string-to-string map handle (for metadata).
pub type mlx_map_string_to_string = *mut c_void;

/// Opaque map iterator handle.
pub type mlx_map_string_to_array_iterator = *mut c_void;

/// Opaque vector of arrays handle.
pub type mlx_vector_array = *mut c_void;

/// Opaque vector of ints handle.
pub type mlx_vector_int = *mut c_void;

/// Opaque MLX string handle.
pub type mlx_string = *mut c_void;

/// Opaque MLX closure handle.
pub type mlx_closure = *mut c_void;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// MLX data types.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mlx_dtype {
    MLX_BOOL = 0,
    MLX_UINT8 = 1,
    MLX_UINT16 = 2,
    MLX_UINT32 = 3,
    MLX_UINT64 = 4,
    MLX_INT8 = 5,
    MLX_INT16 = 6,
    MLX_INT32 = 7,
    MLX_INT64 = 8,
    MLX_FLOAT16 = 9,
    MLX_FLOAT32 = 10,
    MLX_FLOAT64 = 11,
    MLX_BFLOAT16 = 12,
    MLX_COMPLEX64 = 13,
}

/// MLX device types.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mlx_device_type {
    MLX_CPU = 0,
    MLX_GPU = 1,
}

extern "C" {
    // -----------------------------------------------------------------------
    // Array lifecycle
    // -----------------------------------------------------------------------

    pub fn mlx_array_new() -> mlx_array;
    pub fn mlx_array_free(arr: mlx_array) -> c_int;
    pub fn mlx_array_set(arr: *mut mlx_array, src: mlx_array) -> c_int;

    // -----------------------------------------------------------------------
    // Array creation
    // -----------------------------------------------------------------------

    /// Create array from raw data buffer.
    pub fn mlx_array_new_data(
        res: *mut mlx_array,
        data: *const c_void,
        shape: *const c_int,
        ndim: c_int,
        dtype: mlx_dtype,
    ) -> c_int;

    /// Create a scalar float32 array.
    pub fn mlx_array_new_float(val: c_float) -> mlx_array;

    /// Create a scalar int32 array.
    pub fn mlx_array_new_int(val: c_int) -> mlx_array;

    /// Create a scalar bool array.
    pub fn mlx_array_new_bool(val: bool) -> mlx_array;

    // -----------------------------------------------------------------------
    // Array metadata
    // -----------------------------------------------------------------------

    pub fn mlx_array_ndim(arr: mlx_array) -> c_int;
    pub fn mlx_array_shape(arr: mlx_array, dim: c_int) -> c_int;
    pub fn mlx_array_size(arr: mlx_array) -> c_int;
    pub fn mlx_array_dtype(arr: mlx_array) -> mlx_dtype;

    // -----------------------------------------------------------------------
    // Array evaluation and data access
    // -----------------------------------------------------------------------

    pub fn mlx_array_eval(arr: mlx_array) -> c_int;

    /// Get pointer to float32 data (must call eval first).
    pub fn mlx_array_data_float32(arr: mlx_array) -> *const c_float;

    /// Get pointer to int32 data (must call eval first).
    pub fn mlx_array_data_int32(arr: mlx_array) -> *const i32;

    /// Get pointer to int64 data (must call eval first).
    pub fn mlx_array_data_int64(arr: mlx_array) -> *const i64;

    /// Get pointer to bool data (must call eval first).
    pub fn mlx_array_data_bool(arr: mlx_array) -> *const bool;

    /// Get a scalar float32 value.
    pub fn mlx_array_item_float32(arr: mlx_array) -> c_float;

    /// Get a scalar int32 value.
    pub fn mlx_array_item_int32(arr: mlx_array) -> i32;

    /// Get a scalar int64 value.
    pub fn mlx_array_item_int64(arr: mlx_array) -> i64;

    // -----------------------------------------------------------------------
    // Device
    // -----------------------------------------------------------------------

    pub fn mlx_device_new_type(dtype: mlx_device_type, index: c_int) -> mlx_device;
    pub fn mlx_device_free(dev: mlx_device) -> c_int;
    pub fn mlx_get_default_device(res: *mut mlx_device) -> c_int;
    pub fn mlx_set_default_device(dev: mlx_device) -> c_int;
    pub fn mlx_metal_is_available(res: *mut bool) -> c_int;

    // -----------------------------------------------------------------------
    // Stream
    // -----------------------------------------------------------------------

    pub fn mlx_stream_new_device(dev: mlx_device) -> mlx_stream;
    pub fn mlx_stream_free(stream: mlx_stream) -> c_int;
    pub fn mlx_get_default_stream(dev: mlx_device, res: *mut mlx_stream) -> c_int;
    pub fn mlx_synchronize(stream: mlx_stream) -> c_int;

    // -----------------------------------------------------------------------
    // Core ops (ops.h)
    // -----------------------------------------------------------------------

    // Arithmetic
    pub fn mlx_add(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_subtract(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_multiply(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_divide(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_negative(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_abs(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_power(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_maximum(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_minimum(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_clip(
        res: *mut mlx_array,
        a: mlx_array,
        min: mlx_array,
        max: mlx_array,
        s: mlx_stream,
    ) -> c_int;

    // Matrix multiplication
    pub fn mlx_matmul(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;

    // Shape manipulation
    pub fn mlx_reshape(
        res: *mut mlx_array,
        a: mlx_array,
        shape: *const c_int,
        ndim: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_transpose(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_swapaxes(
        res: *mut mlx_array,
        a: mlx_array,
        axis1: c_int,
        axis2: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_expand_dims(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_squeeze(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_flatten(
        res: *mut mlx_array,
        a: mlx_array,
        start_axis: c_int,
        end_axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_slice(
        res: *mut mlx_array,
        a: mlx_array,
        start: *const c_int,
        nstart: c_int,
        stop: *const c_int,
        nstop: c_int,
        strides: *const c_int,
        nstrides: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_broadcast_to(
        res: *mut mlx_array,
        a: mlx_array,
        shape: *const c_int,
        ndim: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_contiguous(
        res: *mut mlx_array,
        a: mlx_array,
        allow_col_major: bool,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_as_strided(
        res: *mut mlx_array,
        a: mlx_array,
        shape: *const c_int,
        nshape: c_int,
        strides: *const i64,
        nstrides: c_int,
        offset: i64,
        s: mlx_stream,
    ) -> c_int;

    // Concatenation
    pub fn mlx_concatenate(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_stack(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    // Creation ops
    pub fn mlx_zeros(
        res: *mut mlx_array,
        shape: *const c_int,
        ndim: c_int,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_ones(
        res: *mut mlx_array,
        shape: *const c_int,
        ndim: c_int,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_full(
        res: *mut mlx_array,
        shape: *const c_int,
        ndim: c_int,
        val: mlx_array,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_arange(
        res: *mut mlx_array,
        start: f64,
        stop: f64,
        step: f64,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;

    // Indexing
    pub fn mlx_take(
        res: *mut mlx_array,
        a: mlx_array,
        indices: mlx_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_take_along_axis(
        res: *mut mlx_array,
        a: mlx_array,
        indices: mlx_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_gather(
        res: *mut mlx_array,
        a: mlx_array,
        indices: mlx_vector_array,
        axes: *const c_int,
        naxes: c_int,
        slice_sizes: *const c_int,
        nslice_sizes: c_int,
        s: mlx_stream,
    ) -> c_int;

    // Reduction
    pub fn mlx_sum(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_mean(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_var(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        keepdims: bool,
        ddof: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_sum_all(
        res: *mut mlx_array,
        a: mlx_array,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_mean_all(
        res: *mut mlx_array,
        a: mlx_array,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_argmax(
        res: *mut mlx_array,
        a: mlx_array,
        axis: c_int,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_argmin(
        res: *mut mlx_array,
        a: mlx_array,
        axis: c_int,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;

    // Math functions
    pub fn mlx_exp(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_log(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_sqrt(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_rsqrt(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_sin(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_cos(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_sigmoid(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_tanh(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;

    // Activation-related
    pub fn mlx_softmax(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        precise: bool,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_where(
        res: *mut mlx_array,
        condition: mlx_array,
        x: mlx_array,
        y: mlx_array,
        s: mlx_stream,
    ) -> c_int;

    // Comparison
    pub fn mlx_less(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_greater(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_equal(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_logical_or(
        res: *mut mlx_array,
        a: mlx_array,
        b: mlx_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_logical_not(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;

    // Triangular
    pub fn mlx_triu(res: *mut mlx_array, a: mlx_array, k: c_int, s: mlx_stream) -> c_int;
    pub fn mlx_tril(res: *mut mlx_array, a: mlx_array, k: c_int, s: mlx_stream) -> c_int;

    // Top-k
    pub fn mlx_topk(
        res: *mut mlx_array,
        a: mlx_array,
        k: c_int,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    // Sort / argsort
    pub fn mlx_sort(
        res: *mut mlx_array,
        a: mlx_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_argsort(
        res: *mut mlx_array,
        a: mlx_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    // Type conversion
    pub fn mlx_astype(
        res: *mut mlx_array,
        a: mlx_array,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;

    // Convolution
    pub fn mlx_conv1d(
        res: *mut mlx_array,
        input: mlx_array,
        weight: mlx_array,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_conv_transpose1d(
        res: *mut mlx_array,
        input: mlx_array,
        weight: mlx_array,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
        s: mlx_stream,
    ) -> c_int;

    // Pad
    pub fn mlx_pad(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        naxes: c_int,
        low_pad: *const c_int,
        nlow: c_int,
        high_pad: *const c_int,
        nhigh: c_int,
        val: mlx_array,
        s: mlx_stream,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Fast ML ops (fast.h)
    // -----------------------------------------------------------------------

    pub fn mlx_fast_rms_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        eps: c_float,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_fast_layer_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        bias: mlx_array,
        eps: c_float,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_fast_rope(
        res: *mut mlx_array,
        x: mlx_array,
        dims: c_int,
        traditional: bool,
        base: mlx_array,
        scale: c_float,
        offset: c_int,
        freqs: mlx_array,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_fast_scaled_dot_product_attention(
        res: *mut mlx_array,
        queries: mlx_array,
        keys: mlx_array,
        values: mlx_array,
        scale: c_float,
        mask: mlx_array,
        memory_efficient_threshold: c_int,
        s: mlx_stream,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // FFT (fft.h)
    // -----------------------------------------------------------------------

    pub fn mlx_fft_rfft(
        res: *mut mlx_array,
        a: mlx_array,
        n: c_int,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_fft_irfft(
        res: *mut mlx_array,
        a: mlx_array,
        n: c_int,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Random (random.h)
    // -----------------------------------------------------------------------

    pub fn mlx_random_categorical(
        res: *mut mlx_array,
        logits: mlx_array,
        axis: c_int,
        num_samples: c_int,
        key: mlx_array,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_random_key(key: *mut mlx_array, seed: u64) -> c_int;

    // -----------------------------------------------------------------------
    // I/O (io.h)
    // -----------------------------------------------------------------------

    pub fn mlx_load_safetensors(
        data: *mut mlx_map_string_to_array,
        metadata: *mut mlx_map_string_to_string,
        path: *const c_char,
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_save_safetensors(
        data: mlx_map_string_to_array,
        metadata: mlx_map_string_to_string,
        path: *const c_char,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Map (map.h)
    // -----------------------------------------------------------------------

    pub fn mlx_map_string_to_array_new() -> mlx_map_string_to_array;
    pub fn mlx_map_string_to_array_free(map: mlx_map_string_to_array) -> c_int;
    pub fn mlx_map_string_to_array_set(
        map: mlx_map_string_to_array,
        key: *const c_char,
        val: mlx_array,
    ) -> c_int;
    pub fn mlx_map_string_to_array_get(
        res: *mut mlx_array,
        map: mlx_map_string_to_array,
        key: *const c_char,
    ) -> c_int;
    pub fn mlx_map_string_to_array_size(map: mlx_map_string_to_array) -> c_int;
    pub fn mlx_map_string_to_array_iterator_new(
        map: mlx_map_string_to_array,
    ) -> mlx_map_string_to_array_iterator;
    pub fn mlx_map_string_to_array_iterator_next(
        it: mlx_map_string_to_array_iterator,
    ) -> bool;
    pub fn mlx_map_string_to_array_iterator_key(
        it: mlx_map_string_to_array_iterator,
        res: *mut mlx_string,
    ) -> c_int;
    pub fn mlx_map_string_to_array_iterator_value(
        it: mlx_map_string_to_array_iterator,
        res: *mut mlx_array,
    ) -> c_int;
    pub fn mlx_map_string_to_array_iterator_free(
        it: mlx_map_string_to_array_iterator,
    ) -> c_int;

    pub fn mlx_map_string_to_string_new() -> mlx_map_string_to_string;
    pub fn mlx_map_string_to_string_free(map: mlx_map_string_to_string) -> c_int;

    // -----------------------------------------------------------------------
    // Vector (vector.h)
    // -----------------------------------------------------------------------

    pub fn mlx_vector_array_new() -> mlx_vector_array;
    pub fn mlx_vector_array_free(vec: mlx_vector_array) -> c_int;
    pub fn mlx_vector_array_append(vec: mlx_vector_array, arr: mlx_array) -> c_int;
    pub fn mlx_vector_array_get(res: *mut mlx_array, vec: mlx_vector_array, idx: c_int) -> c_int;
    pub fn mlx_vector_array_size(vec: mlx_vector_array) -> c_int;

    // -----------------------------------------------------------------------
    // String (string.h)
    // -----------------------------------------------------------------------

    pub fn mlx_string_new() -> mlx_string;
    pub fn mlx_string_free(s: mlx_string) -> c_int;
    pub fn mlx_string_data(s: mlx_string) -> *const c_char;

    // -----------------------------------------------------------------------
    // Memory management
    // -----------------------------------------------------------------------

    pub fn mlx_get_active_memory() -> usize;
    pub fn mlx_get_peak_memory() -> usize;
    pub fn mlx_clear_cache() -> c_int;
}
