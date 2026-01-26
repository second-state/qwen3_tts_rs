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

```bash
./target/release/qwen3_tts_demo <model_path> [text] [speaker] [language]
```

Example:

```bash
./target/release/qwen3_tts_demo \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "Hello world, this is a test." \
  Vivian \
  english
```

This generates an `output.wav` file with 24kHz audio.

### Available speakers

Vivian, Serena, Ryan, Ethan, Chloe, Leo, Isabella, Alexander, Sophia, Benjamin, and more. See the model's `config.json` for the full list.

### Available languages

`english`, `chinese`, and others defined in the model config.

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
