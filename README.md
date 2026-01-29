# Qwen3 TTS - Rust Port

[![Crates.io](https://img.shields.io/crates/v/qwen3_tts.svg)](https://crates.io/crates/qwen3_tts)
[![License](https://img.shields.io/crates/l/qwen3_tts.svg)](https://github.com/juntao/qwen3_tts_rs/blob/main/LICENSE)

A Rust implementation of the Qwen3 Text-to-Speech (TTS) model inference, using the [tch](https://github.com/LaurentMazare/tch-rs) crate for PyTorch/libtorch bindings.

## Prerequisites

### Install libtorch

The `tch` crate (v0.23) requires **PyTorch/libtorch 2.10.0**. You can set it up in one of two ways.

#### Option 1: Use pip-installed PyTorch (recommended)

Install PyTorch via pip and point the build system to it:

```bash
pip install torch==2.10.0
```

Then set the following environment variables before building:

```bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
```

#### Option 2: Download libtorch directly

Linux x86 CPU:

```bash
curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.10.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.10.0+cpu.zip
```

Linux x86 with CUDA 12.8:

```bash
curl -LO https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.10.0%2Bcu128.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.10.0+cu128.zip
```

macOS on Apple Silicon (M-series):

```bash
curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.10.0.zip
unzip libtorch-macos-arm64-2.10.0.zip
```

Then set environment variables (add to `~/.zprofile` or `~/.bash_profile` to persist):

```bash
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:$LD_LIBRARY_PATH
```

### Download the model

Download the Qwen3-TTS model weights. For speech synthesis with named speakers (CustomVoice 0.6B):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

For instruction-controlled voice synthesis (CustomVoice 1.7B, supports emotion/style control):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

For voice cloning from reference audio (Base):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
```

### Generate tokenizer.json

The Rust `tokenizers` crate requires a `tokenizer.json` file with the full tokenizer configuration (including the pre-tokenizer). Generate it from the Python tokenizer for each model you downloaded:

```bash
python3 -c "
from transformers import AutoTokenizer
for model in ['Qwen3-TTS-12Hz-0.6B-CustomVoice', 'Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-CustomVoice']:
    path = f'models/{model}'
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.backend_tokenizer.save(f'{path}/tokenizer.json')
    print(f'Saved {path}/tokenizer.json')
"
```

## Build

```bash
git clone https://github.com/juntao/qwen3_tts_rs.git
cd qwen3_tts_rs
cargo build --release
```

## Run

### TTS Demo

Generate speech from text:

```bash
cargo run --example tts_demo --release -- <model_path> [text] [speaker] [language] [instruction]
```

Example:

```bash
cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "Hello world, this is a test." \
  Vivian \
  english
```

This generates an `output.wav` file with 24kHz audio.

### Instruction-Controlled Voice (1.7B CustomVoice)

The 1.7B CustomVoice model supports instruction control to modulate voice characteristics like emotion, speaking style, and pace:

```bash
cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  "Breaking news! There has been a major development." \
  Vivian \
  english \
  "Speak in an urgent and excited voice"
```

Other instruction examples:
- `"Speak happily and joyfully"`
- `"Speak slowly and calmly"`
- `"Speak in a whisper"`
- `"Speak with a sad tone"`

### Voice Clone Demo

Clone a voice from a reference audio file using the Base model:

```bash
cargo run --example voice_clone_demo --release -- <model_path> <ref_audio> [text] [language] [ref_text]
```

**X-vector only mode** (no reference text):

```bash
cargo run --example voice_clone_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-Base \
  reference.wav \
  "Hello world, this is a voice cloning test." \
  english
```

**ICL mode** (with reference text, higher quality):

```bash
cargo run --example voice_clone_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-Base \
  reference.wav \
  "Hello world, this is a voice cloning test." \
  english \
  "This is the transcript of the reference audio."
```

When a reference text transcript is provided as the 5th argument, the demo uses **ICL (In-Context Learning) mode**, which encodes the reference audio into codec tokens and conditions generation on both the speaker embedding and the reference audio/text. This typically produces higher fidelity voice cloning compared to x-vector only mode.

Output is written to `output_voice_clone.wav`.

### Test Weight Loading

Verify that model weights load correctly:

```bash
cargo run --example test_weights --release -- models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### Available speakers (CustomVoice 0.6B)

Vivian, Serena, Ryan, Aiden, Uncle_fu, Ono_anna, Sohee, Eric, Dylan. See the model's `config.json` `spk_id` field for the full list.

### Available languages

`english`, `chinese`, and others defined in the model config.

## API Usage

Add `qwen3_tts` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
qwen3_tts = "0.1"
tch = "0.23"
```

### Named Characters (CustomVoice)

Generate speech using a predefined speaker voice:

```rust
use qwen3_tts::audio::write_wav_file;
use qwen3_tts::inference::TTSInference;
use std::path::Path;
use tch::Device;

fn main() -> anyhow::Result<()> {
    let inference = TTSInference::new(
        Path::new("models/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        Device::Cpu,
    )?;

    let (waveform, sample_rate) = inference.generate(
        "Hello! Welcome to the Qwen3 TTS system.",
        "Vivian",   // speaker name
        "english",  // language
    )?;

    write_wav_file("output.wav", &waveform, sample_rate)?;
    Ok(())
}
```

You can customize generation parameters (temperature, top_k, max codes):

```rust
let (waveform, sample_rate) = inference.generate_with_params(
    "This uses custom generation parameters.",
    "Ryan",
    "english",
    0.8,   // temperature
    30,    // top_k
    4096,  // max_codes
)?;
```

### Instruction-Controlled Voice (1.7B CustomVoice)

The 1.7B CustomVoice model supports instruction control to modulate voice characteristics:

```rust
use qwen3_tts::audio::write_wav_file;
use qwen3_tts::inference::TTSInference;
use std::path::Path;
use tch::Device;

fn main() -> anyhow::Result<()> {
    let inference = TTSInference::new(
        Path::new("models/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        Device::Cpu,
    )?;

    // Generate with instruction control
    let (waveform, sample_rate) = inference.generate_with_instruct(
        "Breaking news! There has been a major development.",
        "Vivian",   // speaker name
        "english",  // language
        Some("Speak in an urgent and excited voice"),  // instruction
        0.9,        // temperature
        50,         // top_k
        2048,       // max_codes
    )?;

    write_wav_file("urgent_news.wav", &waveform, sample_rate)?;

    // Without instruction (pass None for neutral voice)
    let (waveform, sample_rate) = inference.generate_with_instruct(
        "This is a normal announcement.",
        "Vivian",
        "english",
        None,  // no instruction
        0.9,
        50,
        2048,
    )?;

    write_wav_file("neutral.wav", &waveform, sample_rate)?;
    Ok(())
}
```

### Voice Cloning (X-vector)

Clone a voice from reference audio using the Base model with `TTSInference` and `SpeakerEncoder`:

```rust
use qwen3_tts::audio::{load_wav_file, resample, write_wav_file};
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;
use std::path::Path;
use tch::Device;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model_path = Path::new("models/Qwen3-TTS-12Hz-0.6B-Base");

    // Load model and speaker encoder
    let inference = TTSInference::new(model_path, device)?;
    let se_config = inference.config().speaker_encoder_config.clone();
    let speaker_encoder = SpeakerEncoder::load(inference.weights(), &se_config, device)?;

    // Load and resample reference audio to 24kHz
    let (samples, sr) = load_wav_file("reference.wav")?;
    let samples = if sr != se_config.sample_rate {
        resample(&samples, sr, se_config.sample_rate)?
    } else {
        samples
    };

    // Extract speaker embedding and generate speech
    let speaker_embedding = speaker_encoder.extract_embedding(&samples)?;
    let (waveform, sample_rate) = inference.generate_with_xvector(
        "Hello, this is a voice cloning test.",
        &speaker_embedding,
        "english",
        0.9,   // temperature
        50,    // top_k
        2048,  // max_codes
    )?;

    write_wav_file("output.wav", &waveform, sample_rate)?;
    Ok(())
}
```

### Voice Cloning (ICL - Higher Quality)

For higher fidelity voice cloning, use ICL mode which conditions on both the speaker embedding and reference audio codec tokens with their text transcription:

```rust
use qwen3_tts::audio::{load_wav_file, resample, write_wav_file};
use qwen3_tts::audio_encoder::AudioEncoder;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;
use std::path::Path;
use tch::Device;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model_path = Path::new("models/Qwen3-TTS-12Hz-0.6B-Base");

    // Load model and speaker encoder
    let inference = TTSInference::new(model_path, device)?;
    let se_config = inference.config().speaker_encoder_config.clone();
    let speaker_encoder = SpeakerEncoder::load(inference.weights(), &se_config, device)?;

    // Load and resample reference audio to 24kHz
    let (samples, sr) = load_wav_file("reference.wav")?;
    let samples = if sr != se_config.sample_rate {
        resample(&samples, sr, se_config.sample_rate)?
    } else {
        samples
    };

    // Extract speaker embedding
    let speaker_embedding = speaker_encoder.extract_embedding(&samples)?;

    // Encode reference audio to codec tokens
    let speech_tokenizer_path = model_path
        .join("speech_tokenizer")
        .join("model.safetensors");
    let audio_encoder = AudioEncoder::load(&speech_tokenizer_path, device)?;
    let ref_codes = audio_encoder.encode(&samples)?;

    // Generate with ICL (reference text + codec tokens + speaker embedding)
    let (waveform, sample_rate) = inference.generate_with_icl(
        "Hello, this is a voice cloning test.",
        "This is the transcript of the reference audio.",
        &ref_codes,
        &speaker_embedding,
        "english",
        0.9,   // temperature
        50,    // top_k
        2048,  // max_codes
    )?;

    write_wav_file("output.wav", &waveform, sample_rate)?;
    Ok(())
}
```

### Audio Utilities

The `audio` module provides helpers for reading and writing audio:

```rust
use qwen3_tts::audio::{load_wav_file, resample, write_wav_file};

// Read a WAV file
let (samples, sample_rate) = load_wav_file("input.wav")?;

// Write WAV to file
write_wav_file("output.wav", &samples, sample_rate)?;

// Resample between sample rates
let resampled = resample(&samples, 44100, 24000)?;
```

## Architecture

The inference pipeline follows the Python reference implementation:

```
Text → Tokenizer → Dual-stream Embeddings → TalkerModel (28-layer Transformer)
                                                    ↓
                                              codec_head → Code 0
                                                    ↓
                                        CodePredictor (5-layer Transformer) → Codes 1-15
                                                    ↓
                                              Vocoder → 24kHz Waveform
```

Key components:

| Module | Description |
|--------|-------------|
| `inference.rs` | TalkerModel + CodePredictor with dual-stream (text + codec) input |
| `speaker_encoder.rs` | ECAPA-TDNN speaker encoder for extracting x-vectors from reference audio |
| `audio_encoder.rs` | Mimi/SEANet speech tokenizer encoder for encoding audio to codec tokens (ICL mode) |
| `layers.rs` | RMSNorm, RoPE, GQA Attention with QK-Norm, SwiGLU MLP |
| `vocoder.rs` | ResidualVectorQuantizer, 8-layer pre-transformer, ConvNeXt upsampler, Snake decoder |
| `config.rs` | Model configuration deserialization from `config.json` |
| `audio.rs` | WAV file I/O and audio processing utilities |

## Dependencies

- **tch** (0.23): PyTorch/libtorch bindings for tensor ops and neural network inference
- **tokenizers** (0.21): HuggingFace tokenizers for BPE text tokenization
- **hound**: WAV file reading/writing
- **serde/serde_json**: Configuration deserialization
- **thiserror/anyhow**: Error handling

## License

Apache-2.0

## Credits

Based on the original Python implementation by the Alibaba Qwen team.
