// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! MLX safetensors I/O.
//!
//! Provides weight loading from safetensors files using the native
//! MLX C API `mlx_load_safetensors`.

use super::array::MlxArray;
use super::ffi;
use super::stream::default_stream;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

/// Load all tensors from a safetensors file.
///
/// Returns a HashMap mapping tensor names to MlxArray values.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, MlxArray>, String> {
    let path_str = path.to_str().ok_or("Invalid UTF-8 in path")?;
    let c_path = CString::new(path_str).map_err(|e| format!("Invalid path: {e}"))?;

    let mut data: ffi::mlx_map_string_to_array = std::ptr::null_mut();
    let mut metadata: ffi::mlx_map_string_to_string = std::ptr::null_mut();

    let status = unsafe {
        ffi::mlx_load_safetensors(&mut data, &mut metadata, c_path.as_ptr(), default_stream())
    };

    if status != 0 {
        return Err(format!("Failed to load safetensors from {:?}", path));
    }

    // Iterate the map and collect into HashMap
    let mut result = HashMap::new();

    let iter = unsafe { ffi::mlx_map_string_to_array_iterator_new(data) };
    if iter.is_null() {
        unsafe { ffi::mlx_map_string_to_array_free(data) };
        if !metadata.is_null() {
            unsafe { ffi::mlx_map_string_to_string_free(metadata) };
        }
        return Err("Failed to create map iterator".to_string());
    }

    loop {
        let has_next = unsafe { ffi::mlx_map_string_to_array_iterator_next(iter) };
        if !has_next {
            break;
        }

        // Get key
        let mut key_str = unsafe { ffi::mlx_string_new() };
        let status = unsafe { ffi::mlx_map_string_to_array_iterator_key(iter, &mut key_str) };
        if status != 0 || key_str.is_null() {
            continue;
        }
        let key_ptr = unsafe { ffi::mlx_string_data(key_str) };
        let name = if key_ptr.is_null() {
            unsafe { ffi::mlx_string_free(key_str) };
            continue;
        } else {
            let name = unsafe { std::ffi::CStr::from_ptr(key_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { ffi::mlx_string_free(key_str) };
            name
        };

        // Get value
        let mut arr = MlxArray::empty();
        let status =
            unsafe { ffi::mlx_map_string_to_array_iterator_value(iter, &mut arr.ptr) };
        if status != 0 {
            continue;
        }

        result.insert(name, arr);
    }

    // Cleanup
    unsafe { ffi::mlx_map_string_to_array_iterator_free(iter) };
    unsafe { ffi::mlx_map_string_to_array_free(data) };
    if !metadata.is_null() {
        unsafe { ffi::mlx_map_string_to_string_free(metadata) };
    }

    Ok(result)
}

/// Load all tensors from multiple safetensors shards in a directory.
///
/// Looks for files matching `model*.safetensors` and merges all tensors.
pub fn load_safetensors_dir(dir: &Path) -> Result<HashMap<String, MlxArray>, String> {
    let mut all_tensors = HashMap::new();

    // Try single file first
    let single = dir.join("model.safetensors");
    if single.exists() {
        return load_safetensors(&single);
    }

    // Try sharded files
    let pattern_prefix = "model-";
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {:?}: {e}", dir))?;

    let mut shard_files: Vec<_> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            if name.starts_with(pattern_prefix) && name.ends_with(".safetensors") {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();

    shard_files.sort();

    if shard_files.is_empty() {
        return Err(format!("No safetensors files found in {:?}", dir));
    }

    for shard_path in &shard_files {
        let shard_tensors = load_safetensors(shard_path)?;
        all_tensors.extend(shard_tensors);
    }

    Ok(all_tensors)
}

/// Transpose Conv1d weights from PyTorch format [out, in, kernel] to
/// MLX format [out, kernel, in].
///
/// This should be called on conv weight tensors during loading when
/// the weights come from a PyTorch-trained model.
pub fn transpose_conv1d_weight(weight: &MlxArray) -> MlxArray {
    let shape = weight.shape();
    if shape.len() == 3 {
        // [out_channels, in_channels, kernel_size] -> [out_channels, kernel_size, in_channels]
        super::ops::transpose(weight, &[0, 2, 1])
    } else {
        weight.clone()
    }
}
