// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Unified tensor abstraction over tch (libtorch) and MLX backends.
//!
//! This module provides `Tensor`, `Device`, and `DType` types that work
//! identically regardless of which backend feature is enabled. All neural
//! network modules use these types instead of importing `tch` directly.

use std::path::Path;

// ---------------------------------------------------------------------------
// DType — data type abstraction
// ---------------------------------------------------------------------------

/// Tensor element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point (default for computation).
    Float32,
    /// 16-bit floating point.
    Float16,
    /// 16-bit brain floating point.
    BFloat16,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit signed integer.
    Int32,
    /// Boolean.
    Bool,
}

#[cfg(feature = "tch-backend")]
impl From<DType> for tch::Kind {
    fn from(dt: DType) -> Self {
        match dt {
            DType::Float32 => tch::Kind::Float,
            DType::Float16 => tch::Kind::Half,
            DType::BFloat16 => tch::Kind::BFloat16,
            DType::Int64 => tch::Kind::Int64,
            DType::Int32 => tch::Kind::Int,
            DType::Bool => tch::Kind::Bool,
        }
    }
}

#[cfg(feature = "tch-backend")]
impl From<tch::Kind> for DType {
    fn from(kind: tch::Kind) -> Self {
        match kind {
            tch::Kind::Float => DType::Float32,
            tch::Kind::Half => DType::Float16,
            tch::Kind::BFloat16 => DType::BFloat16,
            tch::Kind::Int64 => DType::Int64,
            tch::Kind::Int => DType::Int32,
            tch::Kind::Bool => DType::Bool,
            _ => DType::Float32,
        }
    }
}

#[cfg(feature = "mlx")]
impl From<DType> for crate::backend::mlx::ffi::mlx_dtype {
    fn from(dt: DType) -> Self {
        use crate::backend::mlx::ffi::mlx_dtype::*;
        match dt {
            DType::Float32 => MLX_FLOAT32,
            DType::Float16 => MLX_FLOAT16,
            DType::BFloat16 => MLX_BFLOAT16,
            DType::Int64 => MLX_INT64,
            DType::Int32 => MLX_INT32,
            DType::Bool => MLX_BOOL,
        }
    }
}

#[cfg(feature = "mlx")]
impl From<crate::backend::mlx::ffi::mlx_dtype> for DType {
    fn from(dt: crate::backend::mlx::ffi::mlx_dtype) -> Self {
        use crate::backend::mlx::ffi::mlx_dtype::*;
        match dt {
            MLX_FLOAT32 | MLX_FLOAT64 => DType::Float32,
            MLX_FLOAT16 => DType::Float16,
            MLX_BFLOAT16 => DType::BFloat16,
            MLX_INT64 => DType::Int64,
            MLX_INT32 | MLX_INT8 | MLX_INT16 => DType::Int32,
            MLX_BOOL => DType::Bool,
            _ => DType::Float32,
        }
    }
}

// ---------------------------------------------------------------------------
// Device — compute device abstraction
// ---------------------------------------------------------------------------

/// Compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU computation.
    Cpu,
    /// GPU computation (CUDA for tch, Metal for MLX).
    Gpu(usize),
}

impl Device {
    /// Default GPU device (index 0).
    pub fn gpu() -> Self {
        Device::Gpu(0)
    }
}

#[cfg(feature = "tch-backend")]
impl From<Device> for tch::Device {
    fn from(d: Device) -> Self {
        match d {
            Device::Cpu => tch::Device::Cpu,
            Device::Gpu(i) => tch::Device::Cuda(i),
        }
    }
}

#[cfg(feature = "tch-backend")]
impl From<tch::Device> for Device {
    fn from(d: tch::Device) -> Self {
        match d {
            tch::Device::Cpu => Device::Cpu,
            tch::Device::Cuda(i) => Device::Gpu(i),
            _ => Device::Cpu,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor — unified tensor type
// ---------------------------------------------------------------------------

/// Unified tensor type backed by either tch::Tensor or MLX array.
pub struct Tensor {
    #[cfg(feature = "tch-backend")]
    pub(crate) inner: tch::Tensor,

    #[cfg(feature = "mlx")]
    pub(crate) inner: crate::backend::mlx::array::MlxArray,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={:?})",
            self.size(),
            self.kind()
        )
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        #[cfg(feature = "tch-backend")]
        {
            Tensor {
                inner: self.inner.shallow_clone(),
            }
        }
        #[cfg(feature = "mlx")]
        {
            Tensor {
                inner: self.inner.clone(),
            }
        }
    }
}

// ===== tch backend implementation =====

#[cfg(feature = "tch-backend")]
#[allow(missing_docs)]
impl Tensor {
    /// Wrap a raw tch::Tensor.
    pub fn from_tch(t: tch::Tensor) -> Self {
        Tensor { inner: t }
    }

    /// Get the underlying tch::Tensor reference.
    pub fn as_tch(&self) -> &tch::Tensor {
        &self.inner
    }

    /// Consume and return the underlying tch::Tensor.
    pub fn into_tch(self) -> tch::Tensor {
        self.inner
    }

    // -- Creation --

    pub fn from_slice_f32(data: &[f32]) -> Self {
        Tensor::from_tch(tch::Tensor::from_slice(data))
    }

    pub fn from_slice_i64(data: &[i64]) -> Self {
        Tensor::from_tch(tch::Tensor::from_slice(data))
    }

    pub fn from_slice_i32(data: &[i32]) -> Self {
        Tensor::from_tch(tch::Tensor::from_slice(data))
    }

    pub fn zeros(shape: &[i64], dtype: DType, device: Device) -> Self {
        let opts = (tch::Kind::from(dtype), tch::Device::from(device));
        Tensor::from_tch(tch::Tensor::zeros(shape, opts))
    }

    pub fn ones(shape: &[i64], dtype: DType, device: Device) -> Self {
        let opts = (tch::Kind::from(dtype), tch::Device::from(device));
        Tensor::from_tch(tch::Tensor::ones(shape, opts))
    }

    pub fn full(shape: &[i64], val: f64, dtype: DType, device: Device) -> Self {
        let opts = (tch::Kind::from(dtype), tch::Device::from(device));
        Tensor::from_tch(tch::Tensor::full(shape, val, opts))
    }

    pub fn arange(start: i64, end: i64, device: Device) -> Self {
        let t = tch::Tensor::arange(end - start, (tch::Kind::Int64, tch::Device::from(device)))
            + start;
        Tensor::from_tch(t)
    }

    pub fn arange_f(start: f64, end: f64, step: f64, dtype: DType, device: Device) -> Self {
        let t = tch::Tensor::arange_start_step(
            start,
            end,
            step,
            (tch::Kind::from(dtype), tch::Device::from(device)),
        );
        Tensor::from_tch(t)
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Self {
        let inner: Vec<&tch::Tensor> = tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_tch(tch::Tensor::cat(&inner, dim))
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Self {
        let inner: Vec<&tch::Tensor> = tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_tch(tch::Tensor::stack(&inner, dim))
    }

    pub fn embedding(weight: &Tensor, indices: &Tensor) -> Self {
        Tensor::from_tch(tch::Tensor::embedding(
            &weight.inner,
            &indices.inner,
            -1,    // padding_idx (none)
            false, // scale_grad_by_freq
            false, // sparse
        ))
    }

    pub fn hann_window(size: i64, device: Device) -> Self {
        Tensor::from_tch(tch::Tensor::hann_window(
            size,
            (tch::Kind::Float, tch::Device::from(device)),
        ))
    }

    /// Load tensors from a safetensors file.
    pub fn load_safetensors(path: &Path) -> crate::error::Result<Vec<(String, Tensor)>> {
        let tensors = tch::Tensor::read_safetensors(path)?;
        Ok(tensors
            .into_iter()
            .map(|(name, t)| (name, Tensor::from_tch(t)))
            .collect())
    }

    // -- Shape --

    pub fn size(&self) -> Vec<i64> {
        self.inner.size()
    }

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    pub fn numel(&self) -> i64 {
        self.inner.numel() as i64
    }

    pub fn view(&self, shape: &[i64]) -> Self {
        Tensor::from_tch(self.inner.view(shape))
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        Tensor::from_tch(self.inner.reshape(shape))
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Self {
        Tensor::from_tch(self.inner.narrow(dim, start, len))
    }

    pub fn unsqueeze(&self, dim: i64) -> Self {
        Tensor::from_tch(self.inner.unsqueeze(dim))
    }

    pub fn squeeze_dim(&self, dim: i64) -> Self {
        Tensor::from_tch(self.inner.squeeze_dim(dim))
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        Tensor::from_tch(self.inner.transpose(dim0, dim1))
    }

    pub fn permute(&self, dims: &[i64]) -> Self {
        Tensor::from_tch(self.inner.permute(dims))
    }

    pub fn expand(&self, size: &[i64], implicit: bool) -> Self {
        Tensor::from_tch(self.inner.expand(size, implicit))
    }

    pub fn expand_as(&self, other: &Tensor) -> Self {
        Tensor::from_tch(self.inner.expand_as(&other.inner))
    }

    pub fn contiguous(&self) -> Self {
        Tensor::from_tch(self.inner.contiguous())
    }

    pub fn tr(&self) -> Self {
        Tensor::from_tch(self.inner.tr())
    }

    pub fn select(&self, dim: i64, index: i64) -> Self {
        Tensor::from_tch(self.inner.select(dim, index))
    }

    // -- Arithmetic --

    pub fn matmul(&self, other: &Tensor) -> Self {
        Tensor::from_tch(self.inner.matmul(&other.inner))
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        Tensor::from_tch(self.inner.pow_tensor_scalar(exp))
    }

    pub fn neg(&self) -> Self {
        Tensor::from_tch(self.inner.neg())
    }

    pub fn clamp(&self, min: f64, max: f64) -> Self {
        Tensor::from_tch(self.inner.clamp(min, max))
    }

    pub fn clamp_min(&self, min: f64) -> Self {
        Tensor::from_tch(self.inner.clamp_min(min))
    }

    // -- Activations --

    pub fn softmax(&self, dim: i64) -> Self {
        Tensor::from_tch(self.inner.softmax(dim, tch::Kind::Float))
    }

    pub fn relu(&self) -> Self {
        Tensor::from_tch(self.inner.relu())
    }

    pub fn gelu(&self) -> Self {
        Tensor::from_tch(self.inner.gelu("none"))
    }

    pub fn elu(&self) -> Self {
        Tensor::from_tch(self.inner.elu())
    }

    pub fn silu(&self) -> Self {
        Tensor::from_tch(self.inner.silu())
    }

    pub fn sigmoid(&self) -> Self {
        Tensor::from_tch(self.inner.sigmoid())
    }

    pub fn tanh(&self) -> Self {
        Tensor::from_tch(self.inner.tanh())
    }

    pub fn sin(&self) -> Self {
        Tensor::from_tch(self.inner.sin())
    }

    pub fn cos(&self) -> Self {
        Tensor::from_tch(self.inner.cos())
    }

    pub fn exp(&self) -> Self {
        Tensor::from_tch(self.inner.exp())
    }

    pub fn log(&self) -> Self {
        Tensor::from_tch(self.inner.log())
    }

    pub fn abs(&self) -> Self {
        Tensor::from_tch(self.inner.abs())
    }

    pub fn sqrt(&self) -> Self {
        Tensor::from_tch(self.inner.sqrt())
    }

    pub fn rsqrt(&self) -> Self {
        // tch doesn't have rsqrt directly; compute 1/sqrt
        let s = self.inner.sqrt();
        Tensor::from_tch(s.reciprocal())
    }

    // -- Reduction --

    pub fn mean_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        Tensor::from_tch(self.inner.mean_dim(dims, keepdim, tch::Kind::Float))
    }

    pub fn sum_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        Tensor::from_tch(self.inner.sum_dim_intlist(dims, keepdim, tch::Kind::Float))
    }

    pub fn var_dim(&self, dims: &[i64], unbiased: bool, keepdim: bool) -> Self {
        Tensor::from_tch(self.inner.var_dim(dims, unbiased, keepdim))
    }

    pub fn std_dim(&self, dims: &[i64], unbiased: bool, keepdim: bool) -> Self {
        // std = sqrt(var)
        self.var_dim(dims, unbiased, keepdim).sqrt()
    }

    pub fn mean_all(&self) -> Self {
        Tensor::from_tch(self.inner.mean(tch::Kind::Float))
    }

    // -- Indexing --

    pub fn index_select(&self, dim: i64, index: &Tensor) -> Self {
        Tensor::from_tch(self.inner.index_select(dim, &index.inner))
    }

    pub fn argmax(&self, dim: i64, keepdim: bool) -> Self {
        Tensor::from_tch(self.inner.argmax(dim, keepdim))
    }

    pub fn argmin(&self, dim: i64, keepdim: bool) -> Self {
        Tensor::from_tch(self.inner.argmin(dim, keepdim))
    }

    pub fn topk(&self, k: i64, dim: i64, largest: bool, sorted: bool) -> (Self, Self) {
        let (values, indices) = self.inner.topk(k, dim, largest, sorted);
        (Tensor::from_tch(values), Tensor::from_tch(indices))
    }

    pub fn multinomial(&self, num_samples: i64, replacement: bool) -> Self {
        Tensor::from_tch(self.inner.multinomial(num_samples, replacement))
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Self {
        Tensor::from_tch(self.inner.masked_fill(&mask.inner, value))
    }

    pub fn triu(&self, diagonal: i64) -> Self {
        Tensor::from_tch(self.inner.triu(diagonal))
    }

    pub fn tril(&self, diagonal: i64) -> Self {
        Tensor::from_tch(self.inner.tril(diagonal))
    }

    // -- Comparison --

    pub fn lt_tensor(&self, other: &Tensor) -> Self {
        Tensor::from_tch(self.inner.lt_tensor(&other.inner))
    }

    pub fn logical_or(&self, other: &Tensor) -> Self {
        Tensor::from_tch(self.inner.logical_or(&other.inner))
    }

    // -- Signal processing --

    pub fn conv1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self {
        let bias_inner = bias.map(|b| &b.inner);
        Tensor::from_tch(self.inner.conv1d(
            &weight.inner,
            bias_inner,
            stride,
            padding,
            dilation,
            groups,
        ))
    }

    pub fn conv_transpose1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        output_padding: &[i64],
        groups: i64,
        dilation: &[i64],
    ) -> Self {
        let bias_inner = bias.map(|b| &b.inner);
        Tensor::from_tch(self.inner.conv_transpose1d(
            &weight.inner,
            bias_inner,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        ))
    }

    pub fn constant_pad_nd(&self, pad: &[i64]) -> Self {
        Tensor::from_tch(self.inner.constant_pad_nd(pad))
    }

    pub fn reflection_pad1d(&self, pad: &[i64]) -> Self {
        Tensor::from_tch(self.inner.reflection_pad1d(pad))
    }

    pub fn stft(
        &self,
        n_fft: i64,
        hop_length: i64,
        win_length: i64,
        window: &Tensor,
        normalized: bool,
        onesided: bool,
        return_complex: bool,
    ) -> Self {
        Tensor::from_tch(self.inner.stft(
            n_fft,
            Some(hop_length),
            Some(win_length),
            Some(&window.inner),
            normalized,
            onesided,
            return_complex,
            false,
        ))
    }

    // -- Normalization --

    pub fn layer_norm(
        &self,
        normalized_shape: &[i64],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Self {
        Tensor::from_tch(self.inner.layer_norm(
            normalized_shape,
            weight.map(|w| &w.inner),
            bias.map(|b| &b.inner),
            eps,
            true,
        ))
    }

    // -- Misc --

    pub fn view_as(&self, other: &Tensor) -> Self {
        Tensor::from_tch(self.inner.view_as(&other.inner))
    }

    // -- Type / Device --

    pub fn to_dtype(&self, dtype: DType) -> Self {
        Tensor::from_tch(self.inner.to_kind(tch::Kind::from(dtype)))
    }

    pub fn to_device(&self, device: Device) -> Self {
        Tensor::from_tch(self.inner.to_device(tch::Device::from(device)))
    }

    pub fn kind(&self) -> DType {
        DType::from(self.inner.kind())
    }

    pub fn device(&self) -> Device {
        Device::from(self.inner.device())
    }

    pub fn shallow_clone(&self) -> Self {
        Tensor::from_tch(self.inner.shallow_clone())
    }

    // -- Data extraction --

    pub fn int64_value(&self, indices: &[i64]) -> i64 {
        self.inner.int64_value(indices)
    }

    pub fn f64_value(&self, indices: &[i64]) -> f64 {
        self.inner.double_value(indices)
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        let flat = self.inner.view(-1);
        let numel = flat.numel();
        let mut result = vec![0.0f32; numel];
        flat.to_kind(tch::Kind::Float)
            .copy_data(&mut result, numel);
        result
    }

    pub fn try_into_f64(&self) -> Result<f64, String> {
        let val: Result<f64, _> = (&self.inner).try_into();
        val.map_err(|e| format!("{e}"))
    }
}

// ===== MLX backend implementation =====

#[cfg(feature = "mlx")]
#[allow(missing_docs)]
impl Tensor {
    /// Wrap an MlxArray.
    pub fn from_mlx(a: crate::backend::mlx::array::MlxArray) -> Self {
        Tensor { inner: a }
    }

    /// Get the underlying MlxArray reference.
    pub fn as_mlx(&self) -> &crate::backend::mlx::array::MlxArray {
        &self.inner
    }

    // -- Creation --

    pub fn from_slice_f32(data: &[f32]) -> Self {
        let shape = [data.len() as i32];
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::from_f32(data, &shape))
    }

    pub fn from_slice_i64(data: &[i64]) -> Self {
        let shape = [data.len() as i32];
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::from_i64(data, &shape))
    }

    pub fn from_slice_i32(data: &[i32]) -> Self {
        let shape = [data.len() as i32];
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::from_i32(data, &shape))
    }

    pub fn zeros(shape: &[i64], dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::zeros(
            &shape_i32,
            dtype.into(),
        ))
    }

    pub fn ones(shape: &[i64], dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::ones(
            &shape_i32,
            dtype.into(),
        ))
    }

    pub fn full(shape: &[i64], val: f64, dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let val_arr = crate::backend::mlx::array::MlxArray::scalar_f32(val as f32);
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::full(
            &shape_i32,
            &val_arr,
            dtype.into(),
        ))
    }

    pub fn arange(start: i64, end: i64, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::arange(
            start as f64,
            end as f64,
            1.0,
            crate::backend::mlx::ffi::mlx_dtype::MLX_INT64,
        ))
    }

    pub fn arange_f(start: f64, end: f64, step: f64, dtype: DType, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::arange(
            start,
            end,
            step,
            dtype.into(),
        ))
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Self {
        let refs: Vec<&crate::backend::mlx::array::MlxArray> =
            tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::concatenate(&refs, dim as i32))
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Self {
        let refs: Vec<&crate::backend::mlx::array::MlxArray> =
            tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::stack(&refs, dim as i32))
    }

    pub fn embedding(weight: &Tensor, indices: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::take(
            &weight.inner,
            &indices.inner,
            0,
        ))
    }

    pub fn hann_window(size: i64, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::signal::hann_window(size as i32))
    }

    /// Load tensors from a safetensors file.
    pub fn load_safetensors(path: &Path) -> crate::error::Result<Vec<(String, Tensor)>> {
        let map = crate::backend::mlx::io::load_safetensors(path)
            .map_err(|e| crate::error::Qwen3TTSError::ModelLoad(e))?;
        Ok(map
            .into_iter()
            .map(|(name, arr)| (name, Tensor::from_mlx(arr)))
            .collect())
    }

    // -- Shape --

    pub fn size(&self) -> Vec<i64> {
        self.inner.shape().iter().map(|&s| s as i64).collect()
    }

    pub fn dim(&self) -> usize {
        self.inner.ndim() as usize
    }

    pub fn numel(&self) -> i64 {
        self.inner.size() as i64
    }

    pub fn view(&self, shape: &[i64]) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::reshape(&self.inner, &shape_i32))
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        self.view(shape)
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Self {
        let ndim = self.inner.ndim();
        let dim = if dim < 0 { ndim as i64 + dim } else { dim } as i32;
        let shape = self.inner.shape();
        let mut starts = vec![0i32; ndim as usize];
        let mut stops: Vec<i32> = shape.clone();
        let strides = vec![1i32; ndim as usize];
        starts[dim as usize] = start as i32;
        stops[dim as usize] = (start + len) as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::slice(
            &self.inner,
            &starts,
            &stops,
            &strides,
        ))
    }

    pub fn unsqueeze(&self, dim: i64) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim + 1
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::expand_dims(
            &self.inner,
            &[dim],
        ))
    }

    pub fn squeeze_dim(&self, dim: i64) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::squeeze(&self.inner, &[dim]))
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        let ndim = self.inner.ndim();
        let dim0 = if dim0 < 0 {
            ndim as i64 + dim0
        } else {
            dim0
        } as i32;
        let dim1 = if dim1 < 0 {
            ndim as i64 + dim1
        } else {
            dim1
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::swapaxes(
            &self.inner,
            dim0,
            dim1,
        ))
    }

    pub fn permute(&self, dims: &[i64]) -> Self {
        let dims_i32: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::transpose(
            &self.inner,
            &dims_i32,
        ))
    }

    pub fn expand(&self, size: &[i64], _implicit: bool) -> Self {
        let shape_i32: Vec<i32> = size.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::broadcast_to(
            &self.inner,
            &shape_i32,
        ))
    }

    pub fn expand_as(&self, other: &Tensor) -> Self {
        let shape = other.inner.shape();
        Tensor::from_mlx(crate::backend::mlx::ops::broadcast_to(&self.inner, &shape))
    }

    pub fn contiguous(&self) -> Self {
        // MLX uses unified memory; contiguous is effectively a no-op.
        self.clone()
    }

    pub fn tr(&self) -> Self {
        // Transpose 2D matrix
        self.transpose(-2, -1)
    }

    pub fn select(&self, dim: i64, index: i64) -> Self {
        let idx = crate::backend::mlx::array::MlxArray::scalar_i32(index as i32);
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        let taken = crate::backend::mlx::ops::take(&self.inner, &idx, dim);
        // Remove the selected dimension
        crate::backend::mlx::ops::squeeze(&taken, &[dim]);
        Tensor::from_mlx(crate::backend::mlx::ops::squeeze(&taken, &[dim]))
    }

    // -- Arithmetic --

    pub fn matmul(&self, other: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::matmul(&self.inner, &other.inner))
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        let exp_arr = crate::backend::mlx::array::MlxArray::scalar_f32(exp as f32);
        Tensor::from_mlx(crate::backend::mlx::ops::power(&self.inner, &exp_arr))
    }

    pub fn neg(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::negative(&self.inner))
    }

    pub fn clamp(&self, min: f64, max: f64) -> Self {
        let min_arr = crate::backend::mlx::array::MlxArray::scalar_f32(min as f32);
        let max_arr = crate::backend::mlx::array::MlxArray::scalar_f32(max as f32);
        Tensor::from_mlx(crate::backend::mlx::ops::clip(
            &self.inner,
            &min_arr,
            &max_arr,
        ))
    }

    pub fn clamp_min(&self, min: f64) -> Self {
        let min_arr = crate::backend::mlx::array::MlxArray::scalar_f32(min as f32);
        Tensor::from_mlx(crate::backend::mlx::ops::maximum(&self.inner, &min_arr))
    }

    // -- Activations --

    pub fn softmax(&self, dim: i64) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::softmax(&self.inner, &[dim]))
    }

    pub fn relu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::relu(&self.inner))
    }

    pub fn gelu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::gelu(&self.inner))
    }

    pub fn elu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::elu(&self.inner))
    }

    pub fn silu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::silu(&self.inner))
    }

    pub fn sigmoid(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::sigmoid(&self.inner))
    }

    pub fn tanh(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::tanh(&self.inner))
    }

    pub fn sin(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::sin(&self.inner))
    }

    pub fn cos(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::cos(&self.inner))
    }

    pub fn exp(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::exp(&self.inner))
    }

    pub fn log(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::log(&self.inner))
    }

    pub fn abs(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::abs(&self.inner))
    }

    pub fn sqrt(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::sqrt(&self.inner))
    }

    pub fn rsqrt(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::rsqrt(&self.inner))
    }

    // -- Reduction --

    pub fn mean_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        let dims_i32: Vec<i32> = dims.iter().map(|&d| {
            if d < 0 { self.inner.ndim() as i32 + d as i32 } else { d as i32 }
        }).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::mean(
            &self.inner,
            &dims_i32,
            keepdim,
        ))
    }

    pub fn sum_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        let dims_i32: Vec<i32> = dims.iter().map(|&d| {
            if d < 0 { self.inner.ndim() as i32 + d as i32 } else { d as i32 }
        }).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::sum(
            &self.inner,
            &dims_i32,
            keepdim,
        ))
    }

    pub fn var_dim(&self, dims: &[i64], unbiased: bool, keepdim: bool) -> Self {
        let ddof = if unbiased { 1 } else { 0 };
        let dims_i32: Vec<i32> = dims.iter().map(|&d| {
            if d < 0 { self.inner.ndim() as i32 + d as i32 } else { d as i32 }
        }).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::var(
            &self.inner,
            &dims_i32,
            keepdim,
            ddof,
        ))
    }

    pub fn std_dim(&self, dims: &[i64], unbiased: bool, keepdim: bool) -> Self {
        self.var_dim(dims, unbiased, keepdim).sqrt()
    }

    pub fn mean_all(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::mean_all(&self.inner, false))
    }

    // -- Indexing --

    pub fn index_select(&self, dim: i64, index: &Tensor) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::take(
            &self.inner,
            &index.inner,
            dim,
        ))
    }

    pub fn argmax(&self, dim: i64, keepdim: bool) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::argmax(&self.inner, dim, keepdim))
    }

    pub fn argmin(&self, dim: i64, keepdim: bool) -> Self {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::argmin(&self.inner, dim, keepdim))
    }

    pub fn topk(&self, k: i64, dim: i64, _largest: bool, _sorted: bool) -> (Self, Self) {
        let dim = if dim < 0 {
            self.inner.ndim() as i64 + dim
        } else {
            dim
        } as i32;
        // MLX topk returns the k largest values. For smallest, negate first.
        let values = crate::backend::mlx::ops::topk(&self.inner, k as i32, dim);
        // MLX topk doesn't return indices separately; use argsort to get them.
        let sorted_idx = crate::backend::mlx::ops::argsort(&self.inner, dim);
        // Take the last k indices (largest)
        let n = self.inner.shape_dim(dim);
        let idx_start = n - k as i32;
        let starts = vec![0i32; self.inner.ndim() as usize];
        let mut stops: Vec<i32> = self.inner.shape();
        stops[dim as usize] = k as i32;
        // For topk, we need to extract the top-k indices from argsort
        // Take last k elements along the sorted dimension
        let ndim = self.inner.ndim() as usize;
        let mut top_starts = vec![0i32; ndim];
        let mut top_stops = self.inner.shape();
        let top_strides = vec![1i32; ndim];
        top_starts[dim as usize] = n - k as i32;
        let top_indices = crate::backend::mlx::ops::slice(&sorted_idx, &top_starts, &top_stops, &top_strides);
        (Tensor::from_mlx(values), Tensor::from_mlx(top_indices))
    }

    pub fn multinomial(&self, num_samples: i64, _replacement: bool) -> Self {
        // MLX uses mlx_random_categorical which takes log-probabilities.
        // Convert probabilities to log-probs first.
        let log_probs = crate::backend::mlx::ops::log(&self.inner);
        let result = crate::backend::mlx::ops::random_categorical(
            &log_probs,
            -1, // last axis
            num_samples as i32,
        );
        Tensor::from_mlx(result)
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Self {
        let val = crate::backend::mlx::array::MlxArray::scalar_f32(value as f32);
        // where(mask, val, self) — fill with val where mask is true
        Tensor::from_mlx(crate::backend::mlx::ops::where_cond(
            &mask.inner,
            &val,
            &self.inner,
        ))
    }

    pub fn triu(&self, diagonal: i64) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::triu(
            &self.inner,
            diagonal as i32,
        ))
    }

    pub fn tril(&self, diagonal: i64) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::tril(
            &self.inner,
            diagonal as i32,
        ))
    }

    // -- Comparison --

    pub fn lt_tensor(&self, other: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::less(&self.inner, &other.inner))
    }

    pub fn logical_or(&self, other: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::logical_or(
            &self.inner,
            &other.inner,
        ))
    }

    // -- Signal processing --

    pub fn conv1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self {
        // MLX conv1d expects input [N, L, C_in], weight [C_out, K, C_in]
        // PyTorch expects input [N, C_in, L], weight [C_out, C_in, K]
        // We need to transpose input and weight, then transpose output back.
        let input_t = self.transpose(1, 2); // [N, C_in, L] -> [N, L, C_in]
        // Weight transpose: [C_out, C_in, K] -> [C_out, K, C_in]
        let weight_t = weight.transpose(1, 2);
        let result = crate::backend::mlx::ops::conv1d(
            &input_t.inner,
            &weight_t.inner,
            stride[0] as i32,
            padding[0] as i32,
            dilation[0] as i32,
            groups as i32,
        );
        // Transpose output back: [N, L_out, C_out] -> [N, C_out, L_out]
        let out = Tensor::from_mlx(result);
        let out = out.transpose(1, 2);
        if let Some(b) = bias {
            &out + &b.unsqueeze(-1)
        } else {
            out
        }
    }

    pub fn conv_transpose1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        _output_padding: &[i64],
        groups: i64,
        dilation: &[i64],
    ) -> Self {
        // Same transposition logic as conv1d
        let input_t = self.transpose(1, 2);
        let weight_t = weight.transpose(1, 2);
        let result = crate::backend::mlx::ops::conv_transpose1d(
            &input_t.inner,
            &weight_t.inner,
            stride[0] as i32,
            padding[0] as i32,
            dilation[0] as i32,
            groups as i32,
        );
        let out = Tensor::from_mlx(result);
        let out = out.transpose(1, 2);
        if let Some(b) = bias {
            &out + &b.unsqueeze(-1)
        } else {
            out
        }
    }

    pub fn constant_pad_nd(&self, pad: &[i64]) -> Self {
        let pad_i32: Vec<i32> = pad.iter().map(|&p| p as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::signal::constant_pad_nd(
            &self.inner,
            &pad_i32,
            0.0,
        ))
    }

    pub fn reflection_pad1d(&self, pad: &[i64]) -> Self {
        Tensor::from_mlx(crate::backend::mlx::signal::reflection_pad1d(
            &self.inner,
            pad[0] as i32,
            pad[1] as i32,
        ))
    }

    pub fn stft(
        &self,
        n_fft: i64,
        hop_length: i64,
        _win_length: i64,
        window: &Tensor,
        _normalized: bool,
        _onesided: bool,
        _return_complex: bool,
    ) -> Self {
        Tensor::from_mlx(crate::backend::mlx::signal::stft_magnitude(
            &self.inner,
            n_fft as i32,
            hop_length as i32,
            &window.inner,
        ))
    }

    // -- Normalization --

    pub fn layer_norm(
        &self,
        _normalized_shape: &[i64],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Self {
        if let Some(w) = weight {
            // Use MLX fused layer norm kernel for better performance
            Tensor::from_mlx(crate::backend::mlx::ops::fast_layer_norm(
                &self.inner,
                &w.inner,
                bias.map(|b| &b.inner),
                eps as f32,
            ))
        } else {
            // No weight: fall back to manual computation
            let mean = self.mean_dim(&[-1], true);
            let var = self.var_dim(&[-1], false, true);
            let normalized = &(self - &mean) / &(&var + eps).sqrt();
            if let Some(b) = bias {
                &normalized + b
            } else {
                normalized
            }
        }
    }

    // -- Misc --

    pub fn view_as(&self, other: &Tensor) -> Self {
        let shape: Vec<i64> = other.size();
        self.view(&shape)
    }

    // -- Type / Device --

    pub fn to_dtype(&self, dtype: DType) -> Self {
        Tensor::from_mlx(self.inner.astype(dtype.into()))
    }

    pub fn to_device(&self, _device: Device) -> Self {
        // MLX unified memory: arrays are accessible from any device.
        self.clone()
    }

    pub fn kind(&self) -> DType {
        DType::from(self.inner.dtype())
    }

    pub fn device(&self) -> Device {
        // MLX arrays don't track device; computation device is global.
        Device::Gpu(0)
    }

    pub fn shallow_clone(&self) -> Self {
        self.clone()
    }

    // -- Data extraction --

    pub fn int64_value(&self, _indices: &[i64]) -> i64 {
        // For scalar or single-element access
        self.inner.item_i64()
    }

    pub fn f64_value(&self, _indices: &[i64]) -> f64 {
        self.inner.item_f32() as f64
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        let f32_arr = self.inner.astype(crate::backend::mlx::ffi::mlx_dtype::MLX_FLOAT32);
        f32_arr.to_vec_f32()
    }

    pub fn try_into_f64(&self) -> Result<f64, String> {
        Ok(self.inner.item_f32() as f64)
    }
}

// ---------------------------------------------------------------------------
// Operator overloads (both backends)
// ---------------------------------------------------------------------------

// Add: Tensor + Tensor
impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner + &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::add(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        self + &rhs
    }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        &self + rhs
    }
}

impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        &self + &rhs
    }
}

// Add: Tensor + f64
impl std::ops::Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner + rhs) }
        #[cfg(feature = "mlx")]
        {
            let scalar = crate::backend::mlx::array::MlxArray::scalar_f32(rhs as f32);
            Tensor::from_mlx(crate::backend::mlx::ops::add(&self.inner, &scalar))
        }
    }
}

impl std::ops::Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor {
        &self + rhs
    }
}

// Sub: Tensor - Tensor
impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner - &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::subtract(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self - &rhs
    }
}

impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        &self - rhs
    }
}

impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        &self - &rhs
    }
}

// Mul: Tensor * Tensor
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner * &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::multiply(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        self * &rhs
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        &self * rhs
    }
}

impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        &self * &rhs
    }
}

// Mul: Tensor * f64
impl std::ops::Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner * rhs) }
        #[cfg(feature = "mlx")]
        {
            let scalar = crate::backend::mlx::array::MlxArray::scalar_f32(rhs as f32);
            Tensor::from_mlx(crate::backend::mlx::ops::multiply(&self.inner, &scalar))
        }
    }
}

impl std::ops::Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor {
        &self * rhs
    }
}

// Mul: f64 * Tensor (commutative)
impl std::ops::Mul<&Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        rhs * self
    }
}

impl std::ops::Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        &rhs * self
    }
}

// Div: Tensor / Tensor
impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner / &rhs.inner) }
        #[cfg(feature = "mlx")]
        { Tensor::from_mlx(crate::backend::mlx::ops::divide(&self.inner, &rhs.inner)) }
    }
}

impl std::ops::Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self / &rhs
    }
}

impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        &self / rhs
    }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        &self / &rhs
    }
}

// Div: Tensor / f64
impl std::ops::Div<f64> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor {
        #[cfg(feature = "tch-backend")]
        { Tensor::from_tch(&self.inner / rhs) }
        #[cfg(feature = "mlx")]
        {
            let scalar = crate::backend::mlx::array::MlxArray::scalar_f32(rhs as f32);
            Tensor::from_mlx(crate::backend::mlx::ops::divide(&self.inner, &scalar))
        }
    }
}

impl std::ops::Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor {
        &self / rhs
    }
}

// AddAssign: Tensor += Tensor
impl std::ops::AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        *self = &*self + rhs;
    }
}

impl std::ops::AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        *self = &*self + &rhs;
    }
}
