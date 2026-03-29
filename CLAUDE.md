# CLAUDE.md — Qwen3 TTS Rust Project Guide

## Project Purpose

Qwen3 TTS Rust (`qwen3_tts`) is a pure-Rust inference engine for the [Qwen3 TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) text-to-speech model family. It converts text to natural-sounding speech using transformer-based language models that generate codec tokens, which are then decoded to audio by a BigVGAN vocoder.

The crate supports two computation backends:

- **tch** (default) — PyTorch/libtorch via `tch-rs` bindings. Works on Linux x86_64, Linux ARM64, and macOS.
- **mlx** — Apple MLX via C FFI. macOS Apple Silicon only, uses Metal GPU acceleration.

## Supported Models

| Model | Capability | Notes |
|-------|-----------|-------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | Preset speaker voices | 9 built-in speakers |
| `Qwen3-TTS-12Hz-0.6B-Base` | Voice cloning (ICL) | Has speaker encoder weights |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | Preset voices + voice instruction control | Supports style instructions like "Speak urgently" |

Available speakers: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`.

## Build & Run

```bash
# tch backend (default) — requires libtorch or PyTorch
export LIBTORCH_USE_PYTORCH=1
cargo build --release

# MLX backend — requires git submodule
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx

# Generate speech
cargo run --release --bin tts -- models/Qwen3-TTS-12Hz-0.6B-CustomVoice "Hello world" Vivian english

# Voice cloning (ICL)
cargo run --release --bin voice_clone -- models/Qwen3-TTS-12Hz-0.6B-Base reference.wav "Text to say" english "Reference transcript"

# API server (OpenAI-compatible)
cargo run --release --bin api_server -- models/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8080
```

## Architecture

The crate follows a backend-agnostic design. All neural network code (layers, inference, vocoder) operates on a unified `Tensor` type that dispatches to tch or MLX at compile time via `#[cfg(feature)]` gates. The two features (`tch-backend` and `mlx`) are mutually exclusive.

### Inference Pipeline

```
Text → Tokenizer → Transformer (autoregressive codec token generation)
                          ↓
                    Codec tokens → BigVGAN Vocoder → Waveform (24kHz f32 PCM)
```

For voice cloning (ICL mode), the pipeline adds a reference audio encoding step:

```
Reference audio → AudioEncoder → Codec tokens ─┐
Reference transcript ──────────────────────────┤
Target text ───────────────────────────────────┤
                                               ↓
                              Transformer (conditioned on reference) → Vocoder → Audio
```

---

## File Descriptions

### Root

| File | Description |
|------|-------------|
| `Cargo.toml` | Package manifest. Defines `tch-backend` (default) and `mlx` features as mutually exclusive. |
| `build.rs` | Build script. When `mlx` feature is active, builds the mlx-c git submodule via CMake and links Metal/Accelerate/MPS frameworks. No-op for tch backend. |
| `install.sh` | Installation helper script. |

### `src/lib.rs`

Library root. Declares all public modules, enforces mutual exclusivity of backends via `compile_error!`, and re-exports main types (`Qwen3TTSModel`, `Language`, `Speaker`, `AudioInput`, etc.).

### `src/tensor.rs`

Unified tensor API — the main backend abstraction layer. Wraps either `tch::Tensor` or `MlxArray` behind cfg gates. Exposes a common interface for creation, arithmetic, indexing, reshaping, and extraction that all model code uses.

### `src/config.rs`

Configuration types deserialized from model JSON files (`config.json`, `generation_config.json`). Includes `Qwen3TTSConfig` (model architecture params), `GenerationConfig` (sampling params), `SpeakerEncoderConfig`, `VocoderConfig`, and `TokenizerType` detection.

### `src/types.rs`

Core domain types: `Language` (English/Chinese/Japanese/Korean/Auto), `Speaker`, `VoiceInstruction`, `AudioInput` (FilePath/Url/Base64), `GenerationOutput`, `DecodedAudio`, `EncodedAudio`, `VoiceClonePromptItem`.

### `src/error.rs`

Error types using `thiserror`. `Qwen3TTSError` covers model loading, tokenization, inference, audio processing, and I/O failures. Exports a `Result<T>` type alias.

### `src/tokenizer.rs`

Wraps HuggingFace `tokenizers` crate. Loads `tokenizer.json`, handles special token IDs (`<|audio_bos|>`, `<|audio_eos|>`, `<|text_eos|>`, codec tokens), and formats chat-template prompts for different generation modes.

### `src/model.rs`

High-level model API (`Qwen3TTSModel`). Provides `generate_custom_voice()`, `generate_voice_clone()`, and `generate_voice_design()` methods. Also contains `GenerationParams` builder and text formatting logic.

### `src/inference.rs`

Core inference engine (`TTSInference`). Loads transformer weights from safetensors, implements autoregressive token generation with KV cache, top-k/temperature sampling, repetition penalty, and codec token extraction. Provides `generate_with_params()`, `generate_with_instruct()`, and `generate_with_icl()`.

### `src/layers.rs`

Neural network layer implementations: `RMSNorm`, `Linear`, `Embedding`, `Conv1d`, `ConvTranspose1d`, `Snake` activation. All backend-agnostic using the unified `Tensor` type.

### `src/weights.rs`

Weight management. Loads `.safetensors` files from model directories, provides tensor lookup by name, handles device placement.

### `src/vocoder.rs`

BigVGAN vocoder that converts codec token embeddings to audio waveforms. Implements `ResBlock`, `AMPBlock`, upsampling layers with Snake activation, and the full BigVGAN generator architecture.

### `src/audio.rs`

Audio I/O and processing utilities. WAV file reading/writing (via `hound`), resampling (via `rubato`), base64 decoding, URL detection, peak normalization, and `AudioInput` parsing.

### `src/audio_encoder.rs`

Speech tokenizer that encodes reference audio into codec tokens for ICL voice cloning. Loads from `speech_tokenizer/model.safetensors`. Uses a VQ-VAE-style encoder with residual vector quantization.

### `src/speaker_encoder.rs`

ECAPA-TDNN speaker encoder that extracts x-vector speaker embeddings from reference audio. Used in voice cloning to condition generation on speaker identity. Only present in Base model variants.

### `src/api/` — OpenAI-Compatible REST API

| File | Description |
|------|-------------|
| `mod.rs` | Module declaration. |
| `types.rs` | Request/response types: `SpeechRequest`, `SseAudioDelta`, `SseAudioDone`, `ApiError`, `HealthResponse`, `ModelsResponse`. Maps OpenAI voice names (alloy→serena, echo→ryan, etc.) to Qwen3 speakers. |
| `routes.rs` | Axum route handlers. `POST /v1/audio/speech` dispatches to `speech_full` (returns complete WAV/PCM) or `speech_stream` (SSE with base64 PCM chunks). Voice cloning always uses ICL mode. `CloneData` stores speaker embedding as `Vec<f32>` for thread safety. |
| `chunking.rs` | Text chunking for streaming. First chunk splits aggressively at clause boundaries (≤60 chars) for low time-to-first-audio; subsequent chunks split at sentence boundaries. |

### `src/bin/` — Binary Targets

| File | Description |
|------|-------------|
| `tts.rs` | CLI for preset-voice speech generation. Args: `<model_path> <text> <speaker> <language> [instruction]`. |
| `voice_clone.rs` | CLI for ICL voice cloning. Args: `<model_path> <reference.wav> <text> <language> <reference_transcript>`. |
| `api_server.rs` | Axum HTTP server. Loads model, optional speaker encoder, optional audio encoder. Serves `/health`, `/v1/models`, `/v1/audio/speech`. |

### `src/backend/` — MLX Backend (only compiled with `--features mlx`)

| File | Description |
|------|-------------|
| `mod.rs` | Backend module root. |
| `mlx/ffi.rs` | Raw `extern "C"` FFI bindings to the mlx-c library. |
| `mlx/array.rs` | `MlxArray` type with RAII (Drop), Clone, creation helpers, metadata accessors, and eval. |
| `mlx/ops.rs` | Safe Rust wrappers around FFI operations (matmul, conv, softmax, etc.). |
| `mlx/stream.rs` | Default Metal/CPU stream management and MLX initialization. |
| `mlx/io.rs` | Safetensors weight loading via mlx-c (must use CPU stream). |
| `mlx/signal.rs` | STFT implementation for the vocoder's audio processing. |

### `examples/test_weights.rs`

Diagnostic tool that loads model weights and prints tensor shapes, counts, and basic statistics. Useful for verifying model files are intact.

---

## Voice Cloning: ICL-Only Architecture

All voice cloning (both CLI and API server) uses **ICL (In-Context Learning)** mode:

1. The **audio encoder** (`speech_tokenizer/model.safetensors`) encodes reference audio into codec tokens
2. The reference text transcript is tokenized alongside the codec tokens
3. The transformer generates target speech conditioned on both the reference audio and text
4. `audio_sample_text` (reference transcript) is **required** for all voice cloning requests

When the speaker encoder is available (Base models), it extracts an x-vector embedding from the reference audio. When absent (CustomVoice models), a **zero embedding** is used as a fallback. This allows voice cloning to work with any model that has an audio encoder.

**API server note:** `CloneData` stores speaker embeddings as `Vec<f32>` instead of `Tensor` because tch's raw C pointers are not `Send + Sync`, which is required by `tokio::task::spawn_blocking`. The tensor is reconstructed via `Tensor::from_slice_f32()` inside the blocking task.

---

## Porting tch to MLX: Implementation Notes

### Architecture

The unified `Tensor` type in `src/tensor.rs` wraps either `tch::Tensor` or `MlxArray` behind `#[cfg(feature)]` gates. Neural network layers in `src/layers.rs` and model code in `src/inference.rs` are backend-agnostic.

### MLX-C FFI Gotchas

- **String parameters must never be null.** `mlx_fast_scaled_dot_product_attention` has a `mask_mode: *const c_char` param. Passing null causes a segfault. Always use a valid C string, even empty: `b"\0".as_ptr() as *const c_char`.
- **Verify FFI signatures against mlx-c headers.** Several bindings had wrong signatures:
  - `mlx_pad` — missing `mode: *const c_char` parameter between `val` and `stream` (use `"constant"`)
  - `mlx_gather` — `naxes`/`nslice_sizes` should be `usize` not `c_int`
  - `mlx_sort` — the actual C API function is `mlx_sort_axis` (has an extra `axis` param)
- **`mlx_optional_float`** is a struct with `float value` and `bool has_value`. Used by `mlx_fast_rope` for the base parameter.

### Conv Weight Format Translations

MLX convolutions expect different weight layouts than PyTorch:

| Operation | PyTorch layout | MLX layout | Transform |
|-----------|---------------|------------|-----------|
| conv1d | `[C_out, C_in, K]` | `[C_out, K, C_in]` | `transpose(1, 2)` |
| conv_transpose1d | `[C_in, C_out, K]` | `[C_out, K, C_in]` | `permute([1, 2, 0])` |

**Warning**: For conv_transpose1d, a simple `transpose(1, 2)` gives `[C_in, K, C_out]` which is WRONG. You must use `permute([1, 2, 0])`.

MLX convolutions use channels-last (`[N, L, C]`) vs PyTorch channels-first (`[N, C, L]`). Transpose input/output around every MLX conv call.

### RoPE (Rotary Position Embedding)

`mlx::fast::rope` uses **dim -2** as the sequence position dimension:
- Input `[batch, heads, seq, head_dim]`: dim -2 = seq — **correct**
- Input `[batch, seq, heads, head_dim]`: dim -2 = heads — **wrong**

### Safetensors Loading

`mlx_load_safetensors` must use a **CPU stream**. The Metal backend's `Load::eval_gpu` is not implemented. MLX auto-transfers arrays to GPU when they participate in GPU computations.

### STFT Output Shape

- MLX STFT returns `[n_frames, freq_bins]`
- PyTorch STFT returns `[freq_bins, n_frames]`
- Fix: `swapaxes(0, 1)` in the MLX tensor STFT wrapper

### Tensor API Behavioral Differences

| Operation | PyTorch | MLX | Fix |
|-----------|---------|-----|-----|
| `expand()` / `broadcast_to` | Accepts `-1` (keep dim) | Does not accept `-1` | Replace `-1` with actual dim size |
| Scalar extraction | `tensor.int64_value(&[i, j])` indexes directly | `item_i64()` only on scalar arrays | Slice first, then `item_i64()` |
| `multinomial` / `random_categorical` | Returns `[..., num_samples]` | Drops last dimension | `expand_dims(&result, &[-1])` after call |

### Repetition Penalty (Critical for MLX)

Both backends degenerate into infinite repetition with greedy decoding. With stochastic sampling the tch backend escapes randomly, but MLX often gets stuck due to small numerical differences producing lower EOS logits.

**Fix**: Apply `repetition_penalty` (default 1.05 from `generation_config.json`):
- Positive logits: divide by penalty factor
- Negative logits: multiply by penalty factor

This is **essential** for convergence on MLX.

### Debugging Tips

- Add debug prints comparing tch vs MLX outputs at each pipeline step to find divergence points
- When MLX generates forever (no EOS), check: (1) repetition penalty implemented? (2) numerical differences in logits?
- Use `MlxArray::eval()` explicitly to inspect intermediate values
- Compare shapes aggressively — many bugs manifest as shape mismatches in matmul or broadcast

---

## Test Plan

### Prerequisites

```bash
# For tch backend
pip install torch==2.10.0
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

# HuggingFace tools
pip install huggingface_hub transformers

# Download models and generate tokenizer.json
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/Qwen3-TTS-12Hz-0.6B-CustomVoice', trust_remote_code=True)
tok.backend_tokenizer.save('models/Qwen3-TTS-12Hz-0.6B-CustomVoice/tokenizer.json')
"
```

### 1. Build Verification

- **1.1** `cargo build --release` — library + binaries (`tts`, `voice_clone`, `api_server`)
- **1.2** `cargo build --examples --release` — `test_weights` example
- **1.3** `cargo fmt --check` — no formatting issues
- **1.4** `cargo clippy --release -- -D warnings` — no lint warnings

### 2. Unit Tests

`cargo test --release` — no model weights required.

Tests cover: core types (Language, Speaker, AudioInput), audio utilities (URL detection, base64 detection, normalization), configuration (defaults, serialization), tokenizer config, model params builder, Snake activation, vocoder config defaults, text chunking strategies.

### 3. Integration Tests (Require Model Weights)

| Test | Model | Command | Pass Criteria |
|------|-------|---------|---------------|
| 3.1 Weight loading | 0.6B CustomVoice | `test_weights <model>` | Prints tensor info, "completed successfully" |
| 3.2 English TTS | 0.6B CustomVoice | `tts <model> "Hello!" Vivian english` | Valid WAV at 24kHz, duration > 0s |
| 3.3 Chinese TTS | 0.6B CustomVoice | `tts <model> "你好！" Vivian chinese` | Valid WAV at 24kHz |
| 3.4 Alternate speaker | 0.6B CustomVoice | `tts <model> "..." Ryan english` | Valid WAV (also used as reference for 3.5) |
| 3.5 Voice clone CLI | 0.6B Base | `voice_clone <model> ref.wav "..." english "ref text"` | ICL mode, valid WAV |
| 3.6 Voice clone API | 0.6B CustomVoice | `curl -d @clone_request.json` | HTTP 200, valid WAV, works without speaker encoder |
| 3.7 Instruction (urgent) | 1.7B CustomVoice | `tts <model> "..." Vivian english "Speak urgently"` | Valid WAV with urgent style |
| 3.8 Instruction (happy) | 1.7B CustomVoice | `tts <model> "..." Vivian english "Speak happily"` | Valid WAV with happy style |

### 4. API Server Tests (in CI integration workflow)

The integration CI starts the API server once and runs 4 tests:

1. **Preset voice, non-streaming WAV** — `POST /v1/audio/speech` with `voice: "alloy"`, `response_format: "wav"`
2. **Preset voice, streaming PCM** — same with `stream: true`, `response_format: "pcm"`
3. **Voice clone (ICL), non-streaming WAV** — includes `audio_sample` (base64) and `audio_sample_text`
4. **Voice clone (ICL), streaming PCM** — same with `stream: true`

Voice clone requests use `curl -d @clone_request.json` (file-based payload) to avoid shell argument length limits with base64-encoded audio.

### 5. Platform Matrix

| Platform | Runner | Backend | Notes |
|----------|--------|---------|-------|
| Linux x86_64 | `ubuntu-latest` | tch (libtorch) | Primary CI target |
| Linux ARM64 | `ubuntu-24.04-arm` | tch (libtorch) | ARM server target |
| macOS ARM64 | `macos-latest` | MLX (Metal GPU) | Apple Silicon target |

### 6. CI Structure

- **`.github/workflows/ci.yml`** — Build-only, triggered on push/PR. Compiles library + binaries + examples on all 3 platforms. No model downloads or inference.
- **`.github/workflows/integration.yml`** — Full integration tests, manual `workflow_dispatch` only. Downloads models, generates tokenizer, runs all integration tests including API server tests.

---

## Lessons Learned

1. **tch Tensor is not Send+Sync.** Raw C pointers in `tch::Tensor` prevent crossing thread boundaries. Store embeddings as `Vec<f32>` and reconstruct tensors inside `spawn_blocking` closures.

2. **Shell argument limits.** Base64-encoded audio can exceed OS argument length limits. Write payloads to JSON files and use `curl -d @file.json` in CI.

3. **ICL-only voice cloning simplifies the architecture.** Rather than maintaining separate x-vector-only and ICL code paths, using ICL universally with a zero-embedding fallback when the speaker encoder is absent keeps the code clean and works with all model variants.

4. **MLX numerical differences matter.** Small logit differences between backends cause the MLX backend to get stuck in repetition loops. Repetition penalty is essential, not optional.

5. **First-chunk-fast streaming.** Splitting the first text chunk at clause boundaries (≤60 chars) rather than sentence boundaries dramatically reduces time-to-first-audio in the streaming API.

6. **Verify FFI signatures at the source.** Auto-generated or hand-written FFI bindings often have subtle mismatches (wrong types, missing parameters). Always check against the C header files.
