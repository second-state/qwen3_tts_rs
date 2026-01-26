# Qwen3 TTS - Rust Port

A Rust implementation of the Qwen3 Text-to-Speech (TTS) model, ported from the original Python implementation.

## Overview

This crate provides a high-level API for generating speech from text using the Qwen3 TTS model family:

- **CustomVoice**: Use predefined speaker voices for consistent, reproducible speech synthesis
- **VoiceDesign**: Control voice characteristics using natural language descriptions
- **Base (VoiceClone)**: Clone voices from reference audio samples

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
qwen3_tts = "0.1"
```

## Quick Start

### CustomVoice Model

```rust
use qwen3_tts::{Qwen3TTSModel, Language, Speaker};

fn main() -> qwen3_tts::Result<()> {
    // Load a pretrained model
    let model = Qwen3TTSModel::from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")?;

    // Generate speech with a predefined speaker
    let output = model.generate_custom_voice(
        "Hello, welcome to Qwen TTS!",
        Speaker::new("Vivian"),
        Language::English,
        None,  // Optional instruction
        None,  // Use default generation params
    )?;

    // Save the output
    qwen3_tts::audio::write_wav_file("output.wav", &output.waveforms[0], output.sample_rate)?;

    Ok(())
}
```

### Voice Cloning

```rust
use qwen3_tts::{Qwen3TTSModel, AudioInput, Language};

fn main() -> qwen3_tts::Result<()> {
    let model = Qwen3TTSModel::from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")?;

    // Clone voice from reference audio
    let output = model.generate_voice_clone(
        "This is synthesized in the cloned voice.",
        Language::English,
        AudioInput::from("reference.wav"),
        Some("This is the reference transcript."),
        false, // Use ICL mode (requires ref_text)
        None,
    )?;

    qwen3_tts::audio::write_wav_file("cloned.wav", &output.waveforms[0], output.sample_rate)?;

    Ok(())
}
```

### Voice Design

```rust
use qwen3_tts::{Qwen3TTSModel, Language, VoiceInstruction};

fn main() -> qwen3_tts::Result<()> {
    let model = Qwen3TTSModel::from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")?;

    // Design voice with natural language
    let output = model.generate_voice_design(
        "Welcome to our service!",
        VoiceInstruction::new("A warm, friendly female voice with moderate speed"),
        Language::English,
        None,
    )?;

    qwen3_tts::audio::write_wav_file("designed.wav", &output.waveforms[0], output.sample_rate)?;

    Ok(())
}
```

## API Reference

### Main Types

| Type | Description |
|------|-------------|
| `Qwen3TTSModel` | Main TTS model wrapper with generation methods |
| `Qwen3TTSTokenizer` | Speech tokenizer for encoding/decoding audio |
| `AudioInput` | Flexible audio input (file path, URL, base64, waveform) |
| `VoiceClonePromptItem` | Voice clone prompt for reusable voice profiles |
| `GenerationParams` | Builder for generation parameters |
| `Language` | Language specification (Auto, English, Chinese, etc.) |
| `Speaker` | Speaker name for CustomVoice model |

### Configuration Types

| Type | Description |
|------|-------------|
| `Qwen3TTSConfig` | Main model configuration |
| `GenerationConfig` | Default generation parameters |
| `TokenizerType` | V1_25Hz or V2_12Hz tokenizer |
| `TTSModelType` | Base, CustomVoice, or VoiceDesign |

### Audio Utilities

```rust
use qwen3_tts::audio;

// Load audio from various sources
let samples = audio::load_audio(AudioInput::from("audio.wav"), 24000)?;

// Write audio to file
audio::write_wav_file("output.wav", &samples, 24000)?;

// Resample audio
let resampled = audio::resample(&samples, 16000, 24000)?;

// Normalize audio to [-1, 1]
audio::normalize_audio(&mut samples);
```

## Features

- `async`: Enable async support with tokio for URL fetching

```toml
[dependencies]
qwen3_tts = { version = "0.1", features = ["async"] }
```

## Architecture

The crate is organized into the following modules:

```
qwen3_tts/
├── audio.rs      # Audio loading, processing, and I/O
├── config.rs     # Configuration structs (mirrors Python configs)
├── error.rs      # Error types with thiserror
├── model.rs      # Qwen3TTSModel and generation methods
├── tokenizer.rs  # Qwen3TTSTokenizer for encode/decode
├── types.rs      # Common types (VoiceClonePromptItem, etc.)
└── lib.rs        # Public exports
```

## Comparison with Python API

| Python | Rust |
|--------|------|
| `Qwen3TTSModel.from_pretrained()` | `Qwen3TTSModel::from_pretrained()` |
| `model.generate_custom_voice()` | `model.generate_custom_voice()` |
| `model.generate_voice_design()` | `model.generate_voice_design()` |
| `model.generate_voice_clone()` | `model.generate_voice_clone()` |
| `model.create_voice_clone_prompt()` | `model.create_voice_clone_prompt()` |
| `VoiceClonePromptItem` | `VoiceClonePromptItem` |
| `Qwen3TTSTokenizer` | `Qwen3TTSTokenizer` |

## Dependencies

- **candle-core/candle-nn**: Tensor operations and neural network primitives
- **tokenizers**: Text tokenization (HuggingFace tokenizers)
- **hound**: WAV file reading/writing
- **rubato**: Audio resampling
- **serde/serde_json**: Configuration serialization
- **safetensors**: Model weight loading
- **hf-hub**: HuggingFace Hub integration
- **thiserror**: Error handling

## Status

This is a Rust port of the Python Qwen3 TTS implementation. Current status:

- [x] Project structure and Cargo.toml
- [x] Configuration types (matching Python configs)
- [x] Audio loading/processing utilities
- [x] Error handling with thiserror
- [x] VoiceClonePromptItem and types
- [x] Qwen3TTSTokenizer API structure
- [x] Qwen3TTSModel API structure
- [ ] Full model inference (requires weight loading)
- [ ] CUDA support
- [ ] Streaming generation

## License

Apache-2.0

## Credits

Based on the original Python implementation by the Alibaba Qwen team.
