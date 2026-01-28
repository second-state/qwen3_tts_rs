# Test Plan

This document describes the testing scenarios for the Qwen3 TTS Rust port. Tests are organized by category: build verification, unit tests, integration tests (end-to-end with model weights), and platform coverage.

## Prerequisites

Before running integration tests locally, ensure the following are set up:

```bash
# PyTorch (required by tch crate)
pip install torch==2.10.0
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

# HuggingFace tools (for model download and tokenizer generation)
pip install huggingface_hub transformers

# Download models
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# Generate tokenizer.json for each model
python3 -c "
from transformers import AutoTokenizer
for model in ['Qwen3-TTS-12Hz-0.6B-CustomVoice', 'Qwen3-TTS-12Hz-0.6B-Base']:
    path = f'models/{model}'
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.backend_tokenizer.save(f'{path}/tokenizer.json')
    print(f'Saved {path}/tokenizer.json')
"
```

---

## 1. Build Verification

### 1.1 Library build

Compile the library in release mode.

```bash
LIBTORCH_USE_PYTORCH=1 cargo build --release
```

**Pass criteria:** Exits 0 with no errors. Warnings are acceptable but should not include `error[...]`.

### 1.2 Examples build

Compile all example binaries.

```bash
LIBTORCH_USE_PYTORCH=1 cargo build --examples --release
```

**Pass criteria:** All three example binaries (`tts_demo`, `voice_clone_demo`, `test_weights`) compile without errors.

### 1.3 Formatting check

```bash
cargo fmt --check
```

**Pass criteria:** No diff output, exit code 0.

### 1.4 Clippy lint check

```bash
LIBTORCH_USE_PYTORCH=1 cargo clippy --release -- -D warnings
```

**Pass criteria:** No warnings or errors.

---

## 2. Unit Tests

Run all in-crate unit tests. These do not require model weights.

```bash
LIBTORCH_USE_PYTORCH=1 cargo test --release
```

### 2.1 Core types (`src/lib.rs`, `src/types.rs`)

| Test | Verifies |
|------|----------|
| `test_version` | `VERSION` constant is non-empty |
| `test_default_sample_rate` | `DEFAULT_SAMPLE_RATE == 24000` |
| `test_language_conversion` | `"english"`, `"EN"`, `"Auto"`, `"zh"` all parse to the correct `Language` variant |
| `test_speaker_creation` | `Speaker::new("Vivian")` round-trips through `.name()` |
| `test_audio_input_from_string` | `AudioInput::from("test.wav")` yields `FilePath` |

### 2.2 Audio utilities (`src/audio.rs`)

| Test | Verifies |
|------|----------|
| `test_is_url` | URL detection for `https://`, `http://`, and non-URL strings |
| `test_is_probably_base64` | Base64 data URI detection |
| `test_audio_input_from_str` | File path, URL, and base64 strings parse into the correct `AudioInput` variant |
| `test_normalize_audio` | Peak normalization scales samples correctly |

### 2.3 Configuration (`src/config.rs`)

| Test | Verifies |
|------|----------|
| `test_default_configs` | `Qwen3TTSConfig::default()` produces valid field values |
| `test_config_serialization` | Config round-trips through JSON serialization |

### 2.4 Tokenizer (`src/tokenizer.rs`)

| Test | Verifies |
|------|----------|
| `test_tokenizer_config_default` | Default tokenizer config has expected values |
| `test_tokenizer_type_detection` | Correctly identifies tokenizer type from config |

### 2.5 Model (`src/model.rs`)

| Test | Verifies |
|------|----------|
| `test_generation_params_builder` | Builder pattern for `GenerationParams` |
| `test_params_merge_with_defaults` | Params merge correctly with `GenerationConfig` defaults |
| `test_text_formatting` | Chat template text formatting |

### 2.6 Layers (`src/layers.rs`)

| Test | Verifies |
|------|----------|
| `test_snake_activation` | Snake activation function produces expected output values |

### 2.7 Vocoder (`src/vocoder.rs`)

| Test | Verifies |
|------|----------|
| `test_vocoder_config_default` | Default vocoder config values are valid |

### 2.8 Inference (`src/inference.rs`)

| Test | Verifies |
|------|----------|
| `test_inference_placeholder` | Placeholder assertion (module compiles) |

### 2.9 Weights (`src/weights.rs`)

| Test | Verifies |
|------|----------|
| `test_device_default` | CPU device initialization works |

**Pass criteria:** `cargo test` reports all tests passing with exit code 0.

---

## 3. Integration Tests (Require Model Weights)

These tests require downloaded model weights and run the full inference pipeline end-to-end. Each test produces a `.wav` output file.

### 3.1 Weight loading verification

Verify that model weights load correctly and tensor operations work.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --example test_weights --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

**Pass criteria:**
- Prints tensor count, names, shapes, and total parameters
- Embedding lookup for token 100 succeeds with valid mean/std
- If `speech_tokenizer/model.safetensors` exists, loads and reports speech tokenizer tensors
- Prints "Weight loading test completed successfully!"

### 3.2 English speech generation (CustomVoice)

Generate English speech with the Vivian speaker.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "Hello! This is a test of the Qwen3 TTS system running on CI." \
  Vivian \
  english
```

**Pass criteria:**
- Exits 0
- Produces `output.wav`
- Output is a valid WAV file at 24kHz
- Reported duration is > 0 seconds

### 3.3 Chinese speech generation (CustomVoice)

Generate Chinese speech with the Vivian speaker.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "你好！这是Qwen3语音合成系统的持续集成测试。" \
  Vivian \
  chinese
```

**Pass criteria:**
- Exits 0
- Produces `output.wav`
- Output is a valid WAV file at 24kHz
- Reported duration is > 0 seconds

### 3.4 Alternate speaker generation (CustomVoice)

Generate speech with the Ryan speaker to verify speaker selection and to produce reference audio for voice cloning tests.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "The quick brown fox jumps over the lazy dog." \
  Ryan \
  english
mv output.wav ryan_reference.wav
```

**Pass criteria:**
- Exits 0
- Produces `output.wav` (renamed to `ryan_reference.wav`)
- Output is a valid WAV file at 24kHz

### 3.5 Voice cloning - X-vector only (Base model)

Clone a voice from reference audio using speaker embedding extraction only.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --example voice_clone_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-Base \
  ryan_reference.wav \
  "This is a voice cloning test using Ryan as the reference speaker." \
  english
```

**Depends on:** Test 3.4 (produces `ryan_reference.wav`)

**Pass criteria:**
- Exits 0
- Prints "Mode: X-vector only"
- Reports speaker embedding shape `[1024]` and a non-zero norm
- Produces `output_voice_clone.wav`
- Output is a valid WAV file at 24kHz
- Reported duration is > 0 seconds

### 3.6 Voice cloning - ICL mode (Base model)

Clone a voice using ICL (In-Context Learning) with reference text transcript and codec token encoding.

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --example voice_clone_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-Base \
  ryan_reference.wav \
  "This is a voice cloning test with in-context learning." \
  english \
  "The quick brown fox jumps over the lazy dog."
```

**Depends on:** Test 3.4 (produces `ryan_reference.wav`)

**Pass criteria:**
- Exits 0
- Prints "Mode: ICL (In-Context Learning)"
- AudioEncoder loads from `speech_tokenizer/model.safetensors`
- Reports encoded codec frame count > 0
- Reports speaker embedding shape `[1024]` and a non-zero norm
- Produces `output_voice_clone.wav`
- Output is a valid WAV file at 24kHz
- Reported duration is > 0 seconds
- Reference portion is trimmed (printed "Trimming reference portion" message)

---

## 4. Platform Matrix

The CI runs all tests on three platform targets. Verify the full pipeline works on each:

| Platform | Runner | PyTorch install | Library path env |
|----------|--------|-----------------|------------------|
| Linux x86_64 | `ubuntu-latest` | `--index-url https://download.pytorch.org/whl/cpu` | `LD_LIBRARY_PATH` |
| Linux ARM64 | `ubuntu-24.04-arm` | Default index | `LD_LIBRARY_PATH` |
| macOS ARM64 | `macos-latest` | Default index | `DYLD_LIBRARY_PATH` |

**Pass criteria per platform:** All steps in sections 1-3 pass. Build artifacts are uploaded successfully.

---

## 5. Artifacts

The CI uploads the following artifacts per platform:

| Artifact | Source test |
|----------|------------|
| `target/release/examples/tts_demo` | Build (1.2) |
| `target/release/examples/voice_clone_demo` | Build (1.2) |
| `output_english.wav` | English generation (3.2) |
| `output_chinese.wav` | Chinese generation (3.3) |
| `ryan_reference.wav` | Ryan reference (3.4) |
| `output_voice_clone.wav` | X-vector voice clone (3.5) |
| `output_voice_clone_icl.wav` | ICL voice clone (3.6) |

**Pass criteria:** All 7 artifacts are present in the uploaded archive and are non-empty files.

---

## 6. Manual Verification (Optional)

These are subjective quality checks on the generated audio, not automated:

- [ ] `output_english.wav` sounds like intelligible English speech
- [ ] `output_chinese.wav` sounds like intelligible Chinese speech
- [ ] `ryan_reference.wav` sounds like a different speaker than Vivian
- [ ] `output_voice_clone.wav` sounds similar in timbre to `ryan_reference.wav`
- [ ] `output_voice_clone_icl.wav` sounds similar in timbre to `ryan_reference.wav`, and is at least as close as the x-vector-only output
- [ ] All outputs are free of obvious glitches, silence-only output, or excessive noise
