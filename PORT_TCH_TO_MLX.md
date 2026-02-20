# MLX Porting Skill Guide

Lessons learned from porting a PyTorch (tch-rs) neural network inference engine to Apple MLX via mlx-c FFI in Rust. These patterns apply broadly to any PyTorch-to-MLX porting effort.

## Architecture

The project uses a unified `Tensor` type in `src/tensor.rs` that wraps either `tch::Tensor` or `MlxArray` behind `#[cfg(feature)]` gates. Neural network layers in `src/layers.rs` and model code in `src/inference.rs` are backend-agnostic — they call methods on `Tensor` which dispatch to the appropriate backend.

### Key files
- `src/tensor.rs` — Unified tensor API (the main porting surface)
- `src/backend/mlx/ffi.rs` — Raw C FFI bindings to mlx-c
- `src/backend/mlx/ops.rs` — Safe Rust wrappers around FFI ops
- `src/backend/mlx/array.rs` — `MlxArray` type with RAII (Drop), Clone, creation, metadata, eval
- `src/backend/mlx/stream.rs` — Default stream management
- `src/backend/mlx/io.rs` — Safetensors weight loading
- `src/backend/mlx/signal.rs` — STFT and signal processing
- `build.rs` — CMake build for mlx-c submodule

## MLX-C FFI Gotchas

### String parameters must never be null
`mlx_fast_scaled_dot_product_attention` has a `mask_mode: *const c_char` param. Passing null causes a segfault. Always pass a valid C string, even if empty: `b"\0".as_ptr() as *const c_char`.

### Verify FFI signatures against mlx-c headers
Several bindings had wrong signatures discovered at runtime:
- `mlx_pad` — missing `mode: *const c_char` parameter between `val` and `stream` (use `"constant"`)
- `mlx_gather` — `naxes`/`nslice_sizes` should be `usize` not `c_int`
- `mlx_sort` — had an extra `axis` param; the actual C API function is `mlx_sort_axis`

### mlx_optional_float
Used by `mlx_fast_rope` for the base parameter. It's a struct with `float value` and `bool has_value`. Pass as a struct, not separate params.

## Conv Weight Format Translations

MLX convolutions expect different weight layouts than PyTorch:

| Operation | PyTorch layout | MLX layout | Transform |
|-----------|---------------|------------|-----------|
| conv1d | `[C_out, C_in, K]` | `[C_out, K, C_in]` | `transpose(1, 2)` |
| conv_transpose1d | `[C_in, C_out, K]` | `[C_out, K, C_in]` | `permute([1, 2, 0])` |

**Warning**: For conv_transpose1d, a simple `transpose(1, 2)` gives `[C_in, K, C_out]` which is WRONG. You need `permute([1, 2, 0])`.

## Conv Input/Output Format

MLX convolutions use channels-last:
- MLX: `[N, L, C]` (batch, length, channels)
- PyTorch: `[N, C, L]` (batch, channels, length)

You must transpose input before and output after every MLX conv call.

## RoPE (Rotary Position Embedding)

`mlx::fast::rope` uses **dim -2** as the sequence position dimension.
- Input `[batch, heads, seq, head_dim]`: dim -2 = seq — **correct**, pass directly
- Input `[batch, seq, heads, head_dim]`: dim -2 = heads — **wrong**, do NOT use this layout

## Safetensors Loading

`mlx_load_safetensors` must use a **CPU stream**. The Metal backend's `Load::eval_gpu` is not implemented and will throw. Create a CPU stream via:
```
mlx_device_new_type(MLX_CPU, 0) → mlx_stream_new_device
```
MLX auto-transfers arrays to GPU when they participate in GPU computations, so loading on CPU is fine.

## STFT Output Shape Mismatch

- MLX STFT returns `[n_frames, freq_bins]`
- PyTorch STFT returns `[freq_bins, n_frames]`

Fix: `swapaxes(0, 1)` in the MLX tensor STFT wrapper. Also ensure the MLX STFT implementation does NOT apply its own padding if the caller already handles center-padding.

## Tensor API Behavioral Differences

### expand / broadcast_to
PyTorch's `expand()` accepts `-1` meaning "keep this dimension unchanged". MLX's `broadcast_to` does not understand `-1` and will error. Fix: replace `-1` values with the actual current dimension size before calling `broadcast_to`.

### Scalar extraction (int64_value, f64_value)
PyTorch's `tensor.int64_value(&[i, j])` indexes directly. MLX's `item_i64()` only works on scalar (size-1) arrays. Fix: slice the array to the target indices first, then call `item_i64()` on the resulting scalar.

### multinomial / random_categorical
PyTorch's `multinomial(num_samples, replacement)` returns shape `[..., num_samples]`, preserving the trailing dimension. MLX's `random_categorical` drops the last dimension. Fix: `expand_dims(&result, &[-1])` after `random_categorical` to restore the trailing dim.

## Autoregressive Generation: Repetition Penalty

Both backends degenerate into infinite repetition with greedy decoding of codec tokens. With stochastic sampling (temperature > 0), the tch backend escapes randomly, but the MLX backend often gets stuck because small numerical differences in the CodePredictor produce slightly lower EOS logits (~-5 vs ~-3).

**Fix**: Apply `repetition_penalty` (default 1.05 from `generation_config.json`) to previously generated tokens:
- Positive logits: divide by penalty factor
- Negative logits: multiply by penalty factor

This is essential for convergence on MLX.

## Build & CI

- MLX feature: `cargo build --release --no-default-features --features mlx`
- Requires git submodule: `git submodule update --init --recursive` (for mlx-c)
- macOS only (aarch64 for Metal GPU); `build.rs` panics on non-macOS
- Links: Metal, Foundation, Accelerate, MetalPerformanceShaders frameworks + libc++
- CI: separate matrix entries for tch and mlx backends with conditional steps
- Both backends compile to the same binary paths — build and test sequentially to avoid overwrites

## Debugging Tips

- Add debug prints comparing tch vs MLX outputs at each step to find divergence points
- When MLX generates forever (no EOS), check: (1) repetition penalty implemented? (2) numerical differences in logits?
- Use `MlxArray::eval()` explicitly when you need to inspect intermediate values
- Compare shapes aggressively — many bugs manifest as shape mismatches in matmul or broadcast
