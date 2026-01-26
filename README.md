# Qwen3 TTS - Rust Port

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

Download the Qwen3-TTS model weights. For example, the 0.6B CustomVoice model:

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### Generate tokenizer.json

The Rust `tokenizers` crate requires a `tokenizer.json` file with the full tokenizer configuration (including the pre-tokenizer). Generate it from the Python tokenizer:

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/Qwen3-TTS-12Hz-0.6B-CustomVoice', trust_remote_code=True)
tok.backend_tokenizer.save('models/Qwen3-TTS-12Hz-0.6B-CustomVoice/tokenizer.json')
print('Saved tokenizer.json')
"
```

## Build

```bash
git clone https://github.com/juntao/qwen3_tts_rs.git
cd qwen3_tts_rs
cargo build --release
```

If using pip-installed PyTorch (Option 1), build with:

```bash
LIBTORCH_USE_PYTORCH=1 cargo build --release
```

## Run

### TTS Demo

Generate speech from text:

```bash
cargo run --example tts_demo --release -- <model_path> [text] [speaker] [language]
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

### Test Weight Loading

Verify that model weights load correctly:

```bash
cargo run --example test_weights --release -- models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### Available speakers

Vivian, Serena, Ryan, Ethan, Chloe, Leo, Isabella, Alexander, Sophia, Benjamin, and more. See the model's `config.json` for the full list.

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
use qwen3_tts::model::Qwen3TTSModel;
use qwen3_tts::audio::write_wav_file;
use qwen3_tts::types::{Language, GenerationParams};

fn main() -> anyhow::Result<()> {
    let model = Qwen3TTSModel::from_pretrained("models/Qwen3-TTS-12Hz-0.6B-CustomVoice")?;

    let output = model.generate_custom_voice(
        "Hello! Welcome to the Qwen3 TTS system.",
        "Vivian",              // speaker name
        Language::English,
        None,                  // no instruct
        None,                  // default generation params
    )?;

    let waveform = output.waveform().unwrap();
    write_wav_file("output.wav", waveform, output.sample_rate)?;

    Ok(())
}
```

You can customize generation parameters:

```rust
let params = GenerationParams::default()
    .temperature(0.8)
    .top_k(30)
    .max_new_tokens(4096);

let output = model.generate_custom_voice(
    "This uses custom generation parameters.",
    "Ryan",
    Language::Chinese,
    None,
    Some(params),
)?;
```

To list available speakers and languages:

```rust
if let Some(speakers) = model.get_supported_speakers() {
    println!("Speakers: {:?}", speakers);
}
if let Some(languages) = model.get_supported_languages() {
    println!("Languages: {:?}", languages);
}
```

### Voice Cloning

Clone a voice from reference audio. Two modes are available:

**X-vector only mode** -- uses only the speaker embedding from the reference audio:

```rust
use qwen3_tts::types::AudioInput;

let output = model.generate_voice_clone(
    "This sentence will be spoken in the cloned voice.",
    Language::English,
    AudioInput::FilePath("reference.wav".into()),
    None,                  // no ref_text needed for x-vector mode
    true,                  // x_vector_only_mode
    None,
)?;
```

**In-Context Learning (ICL) mode** -- conditions on both the reference audio and its transcript for higher fidelity cloning:

```rust
let output = model.generate_voice_clone(
    "This sentence will be spoken in the cloned voice.",
    Language::English,
    AudioInput::FilePath("reference.wav".into()),
    Some("Transcript of the reference audio."),  // required for ICL mode
    false,                 // x_vector_only_mode = false enables ICL
    None,
)?;
```

To reuse a voice clone prompt across multiple generations:

```rust
let prompt = model.create_voice_clone_prompt(
    AudioInput::FilePath("reference.wav".into()),
    Some("Transcript of the reference audio."),
    false,
)?;

let output1 = model.generate_voice_clone_with_prompt(
    "First sentence in the cloned voice.",
    Language::English,
    &prompt,
    None,
)?;

let output2 = model.generate_voice_clone_with_prompt(
    "Second sentence in the same voice.",
    Language::English,
    &prompt,
    None,
)?;
```

Audio can be loaded from multiple sources:

```rust
// From a file path
let audio = AudioInput::FilePath("/path/to/audio.wav".into());

// From a base64-encoded string
let audio = AudioInput::Base64(base64_string);

// From raw samples
let audio = AudioInput::from_waveform(samples, 24000);
```

### Voice Design

Generate speech with a natural language description of the desired voice. This requires the VoiceDesign model variant:

```rust
use qwen3_tts::types::VoiceInstruction;

let model = Qwen3TTSModel::from_pretrained("models/Qwen3-TTS-12Hz-VoiceDesign")?;

let output = model.generate_voice_design(
    "Welcome to our customer support line.",
    VoiceInstruction::new("A warm, friendly female voice with moderate speed"),
    Language::English,
    None,
)?;

let waveform = output.waveform().unwrap();
write_wav_file("designed_voice.wav", waveform, output.sample_rate)?;
```

### Audio Utilities

The `audio` module provides helpers for reading and writing audio:

```rust
use qwen3_tts::audio::{write_wav_file, write_wav_bytes, load_audio, resample};
use qwen3_tts::types::AudioInput;

// Write WAV to file
write_wav_file("output.wav", &samples, 24000)?;

// Write WAV to in-memory bytes
let wav_bytes = write_wav_bytes(&samples, 24000)?;

// Load and resample audio to a target sample rate
let samples = load_audio(AudioInput::FilePath("input.wav".into()), 24000)?;

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
