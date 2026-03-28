# Qwen3 TTS - Rust CLI tools

[![Crates.io](https://img.shields.io/crates/v/qwen3_tts.svg)](https://crates.io/crates/qwen3_tts)
[![License](https://img.shields.io/crates/l/qwen3_tts.svg)](https://github.com/juntao/qwen3_tts_rs/blob/main/LICENSE)

A Rust implementation of the Qwen3 Text-to-Speech (TTS) model inference. This project provides two cross-platform CLI tools suitable for agentic skills for AI agents and bots.

- **tts** generates voice wav files from input text and named voice characters.
- **voice_clone** generates voice wav files from input text and reference audio files.

Supports two backends: **libtorch** (via the `tch` crate, cross-platform with optional CUDA) and **MLX** (Apple Silicon native via Metal GPU). Loads model weights directly from safetensors files and re-implements the complete neural network forward pass in Rust.

Learn more:
* [A Rust implementation / CLI](https://github.com/second-state/qwen3_asr_rs) for Qwen3's ASR (Automatic Speech Recognition or Speech-to-Text) models
* An OpenAI compatible [API server for audio / speech](https://github.com/second-state/qwen3_audio_api/tree/main/rust)
* An OpenClaw SKILL for voice generation. Copy and Paste to your lobster to [instsll it](https://raw.githubusercontent.com/second-state/qwen3_tts_rs/refs/heads/main/skills/install.md)

## Quick Start

Run the installer to download binaries, models, and reference audio for your platform:

```bash
curl -sSf https://raw.githubusercontent.com/second-state/qwen3_tts_rs/main/install.sh | bash
```

The installer detects your OS, CPU, and NVIDIA GPU (if present), then sets up everything in `./qwen3_tts_rs/`. Once complete, generate speech:

```bash
cd qwen3_tts_rs
./tts models/Qwen3-TTS-12Hz-0.6B-CustomVoice "Hello world, this is a test." Vivian english
```

Clone a voice from reference audio:

```bash
./voice_clone models/Qwen3-TTS-12Hz-0.6B-Base reference_audio/trump.wav \
  "Hello, this is a voice cloning test." english \
  "Angered and appalled millions of Americans across the political spectrum"
```

Output: `output.wav` / `output_voice_clone.wav` (24kHz audio)

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

## Supported Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 0.6B | [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 1.7B | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) |

## Usage

### Text-to-Speech

```bash
tts <model_path> [text] [speaker] [language] [instruction]
```

Example:

```bash
tts Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "Hello world, this is a test." \
  Vivian \
  english
```

### Instruction-Controlled Voice (1.7B CustomVoice)

The 1.7B CustomVoice model supports instruction control to modulate voice characteristics like emotion, speaking style, and pace:

```bash
tts Qwen3-TTS-12Hz-1.7B-CustomVoice \
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

### Voice Cloning

Clone a voice from a reference audio file. Voice cloning uses ICL (In-Context Learning) mode, which encodes the reference audio into codec tokens and conditions generation on both the speaker embedding and the reference audio/text transcript. This works with any model that includes a `speech_tokenizer/` directory (both Base and CustomVoice models).

```bash
voice_clone <model_path> <ref_audio> [text] [language] [ref_text]
```

**Preparing reference audio:** The reference audio must be a mono 24kHz 16-bit WAV file. Use ffmpeg to convert from other formats:

```bash
ffmpeg -i input.m4a -ac 1 -ar 24000 -sample_fmt s16 reference.wav
```

Sample reference audio files are provided in the `reference_audio/` directory.

**CLI example** (with reference text transcript):

```bash
voice_clone \
  Qwen3-TTS-12Hz-0.6B-Base \
  reference_audio/trump.wav \
  "Hello world, this is a voice cloning test." \
  english \
  "Angered and appalled millions of Americans across the political spectrum"
```

**API server:** When using the API server, voice cloning always uses ICL mode. Both `audio_sample` (base64-encoded WAV) and `audio_sample_text` (reference transcript) are required. The speaker encoder is used for the x-vector embedding when available (Base model); otherwise a zero embedding is substituted (CustomVoice model).

Output is written to `output_voice_clone.wav`.

### Available speakers (CustomVoice 0.6B)

Vivian, Serena, Ryan, Aiden, Uncle_fu, Ono_anna, Sohee, Eric, Dylan. See the model's `config.json` `spk_id` field for the full list.

### Available languages

`english`, `chinese`, and others defined in the model config.

## Build from Source

### Prerequisites

Install Python tools for downloading models and generating tokenizer files:

```bash
pip install huggingface_hub transformers
```

#### Download the model

Download the Qwen3-TTS model weights. For speech synthesis with named speakers (CustomVoice 0.6B):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

For instruction-controlled voice synthesis (CustomVoice 1.7B, supports emotion/style control):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

For voice cloning with speaker encoder x-vectors (Base):

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
```

#### Generate tokenizer.json

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

### Build for macOS (MLX)

Requires Apple Silicon Mac, Xcode (full installation for Metal shader compiler), and CMake.

Install dependencies:

```bash
brew install cmake ffmpeg
```

Build:

```bash
git clone https://github.com/second-state/qwen3_tts_rs.git
cd qwen3_tts_rs
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx
```

### Build for Linux (libtorch)

#### Download libtorch

Download and extract libtorch for your platform from [libtorch-releases](https://github.com/second-state/libtorch-releases/releases/tag/v2.7.1):

```bash
# Linux x86_64 (CPU)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-x86_64-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-x86_64-2.7.1.tar.gz

# Linux x86_64 (CUDA 12.6)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-x86_64-cuda12.6-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-x86_64-cuda12.6-2.7.1.tar.gz

# Linux ARM64 (CPU)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-aarch64-2.7.1.tar.gz

# Linux ARM64 (CUDA 12.6 / Jetson)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-cuda12.6-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-aarch64-cuda12.6-2.7.1.tar.gz
```

#### Set environment variables

```bash
export LIBTORCH=$(pwd)/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

Alternatively, if you already have Python/PyTorch, see [Troubleshooting: Using pip-installed PyTorch](#using-pip-installed-pytorch).

#### Install dependencies

```bash
sudo apt-get install -y ffmpeg
```

#### Build

```bash
git clone https://github.com/second-state/qwen3_tts_rs.git
cd qwen3_tts_rs
cargo build --release
```

This produces the CLI tools in `target/release/`:
- `tts` — text-to-speech generation
- `voice_clone` — voice cloning from reference audio
- `api_server` — OpenAI-compatible HTTP API server

## API Usage

Add `qwen3_tts` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
qwen3_tts = "0.1"
```

By default this uses the tch-backend. To use MLX instead:

```toml
[dependencies]
qwen3_tts = { version = "0.1", default-features = false, features = ["mlx"] }
```

### Named Characters (CustomVoice)

Generate speech using a predefined speaker voice:

```rust
use qwen3_tts::audio::write_wav_file;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::tensor::Device;
use std::path::Path;

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
use qwen3_tts::tensor::Device;

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
        "Speak in an urgent and excited voice",  // instruction
        0.9,        // temperature
        50,         // top_k
        2048,       // max_codes
    )?;

    write_wav_file("urgent_news.wav", &waveform, sample_rate)?;

    // Without instruction (pass empty string for neutral voice)
    let (waveform, sample_rate) = inference.generate_with_instruct(
        "This is a normal announcement.",
        "Vivian",
        "english",
        "",  // no instruction
        0.9,
        50,
        2048,
    )?;

    write_wav_file("neutral.wav", &waveform, sample_rate)?;
    Ok(())
}
```

### Voice Cloning (ICL)

Clone a voice from reference audio using ICL (In-Context Learning) mode. This works with any model that has a `speech_tokenizer/` directory. The speaker encoder provides an x-vector embedding when available (Base model); otherwise a zero embedding is used (CustomVoice model).

```rust
use qwen3_tts::audio::{load_wav_file, resample, write_wav_file};
use qwen3_tts::audio_encoder::AudioEncoder;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;
use qwen3_tts::tensor::{DType, Device, Tensor};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model_path = Path::new("models/Qwen3-TTS-12Hz-0.6B-CustomVoice");

    // Load model
    let inference = TTSInference::new(model_path, device)?;
    let se_config = inference.config().speaker_encoder_config.clone();

    // Load and resample reference audio to 24kHz
    let (samples, sr) = load_wav_file("reference.wav")?;
    let samples = if sr != se_config.sample_rate {
        resample(&samples, sr, se_config.sample_rate)?
    } else {
        samples
    };

    // Extract speaker embedding if speaker encoder is available, otherwise use zeros
    let speaker_embedding = match SpeakerEncoder::load(inference.weights(), &se_config, device) {
        Ok(speaker_encoder) => speaker_encoder.extract_embedding(&samples)?,
        Err(_) => Tensor::zeros(&[se_config.enc_dim as i64], DType::Float32, Device::Cpu),
    };

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

## Dependencies

- **tch** (0.23, optional): PyTorch/libtorch bindings — used by the `tch-backend` feature (default)
- **mlx-c** (submodule, optional): Apple MLX C bindings — used by the `mlx` feature
- **tokenizers** (0.21): HuggingFace tokenizers for BPE text tokenization
- **hound**: WAV file reading/writing
- **serde/serde_json**: Configuration deserialization
- **thiserror/anyhow**: Error handling

## License

Apache-2.0

## Troubleshooting

### Using pip-installed PyTorch

If you already have Python installed, you can use pip-installed PyTorch instead of downloading libtorch separately:

```bash
pip install torch==2.7.1
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
```

### Test weight loading

Verify that model weights load correctly by building and running the `test_weights` diagnostic example:

```bash
cargo build --examples --release
./target/release/examples/test_weights models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

## Credits

Based on the original Python implementation by the Alibaba Qwen team.
