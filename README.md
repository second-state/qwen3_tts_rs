# Qwen3 TTS - Rust CLI tools

[![Crates.io](https://img.shields.io/crates/v/qwen3-tts-rs.svg)](https://crates.io/crates/qwen3-tts-rs)
[![License](https://img.shields.io/crates/l/qwen3-tts-rs.svg)](https://github.com/second-state/qwen3_tts_rs/blob/main/LICENSE)

A Rust implementation of the Qwen3 Text-to-Speech (TTS) model inference. Provides three cross-platform CLI tools suitable for agentic skills for AI agents and bots.

- **tts** — generate speech from text with named speaker voices
- **voice_clone** — clone a voice from reference audio
- **api_server** — OpenAI-compatible HTTP API server

Supports two backends: **libtorch** (via the `tch` crate, cross-platform with optional CUDA) and **MLX** (Apple Silicon native via Metal GPU).

Learn more:
* [A Rust implementation / CLI](https://github.com/second-state/qwen3_asr_rs) for Qwen3's ASR (Automatic Speech Recognition or Speech-to-Text) models
* An OpenAI compatible [API server for audio / speech](https://github.com/second-state/qwen3_audio_api/tree/main/rust)
* An OpenClaw SKILL for voice generation. Copy and Paste to your lobster to [install it](https://raw.githubusercontent.com/second-state/qwen3_tts_rs/refs/heads/main/skills/install.md)

## Quick Start

Install binaries, models, and reference audio for your platform:

```bash
curl -sSf https://raw.githubusercontent.com/second-state/qwen3_tts_rs/main/install.sh | bash
cd qwen3_tts_rs
```

The installer detects your OS, CPU, and NVIDIA GPU (if present), then sets up everything in `./qwen3_tts_rs/`.

### Text-to-Speech

Generate speech with a named speaker using the CustomVoice model:

```bash
./tts models/Qwen3-TTS-12Hz-0.6B-CustomVoice "Hello world, this is a test." Vivian english
# Output: output.wav (24 kHz)
```

### Voice Cloning

Clone a voice from reference audio using the Base model (ICL mode):

```bash
./voice_clone models/Qwen3-TTS-12Hz-0.6B-Base reference_audio/trump.wav \
  "Hello, this is a voice cloning test." english \
  "Angered and appalled millions of Americans across the political spectrum"
# Output: output_voice_clone.wav (24 kHz)
```

### API Server

Start the OpenAI-compatible API server with the CustomVoice model:

```bash
./api_server models/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8080
```

Then call the endpoint:

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "alloy"}' \
  -o output.wav
```

## Reference

### `tts` — Text-to-Speech

```
tts <model_path> [text] [speaker] [language] [instruction]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | (required) | Path to model directory |
| `text` | "Hello! This is a test..." | Text to synthesize (max ~4096 chars) |
| `speaker` | Vivian | Speaker name (see below) |
| `language` | english | Language: `english`, `chinese`, `japanese`, `korean` |
| `instruction` | (empty) | Voice style instruction (1.7B models only) |

Output: `output.wav` (24 kHz, 16-bit PCM)

**Available speakers** (CustomVoice models): Vivian, Serena, Ryan, Aiden, Uncle_fu, Ono_anna, Sohee, Eric, Dylan

**Instruction examples** (1.7B CustomVoice only):
- `"Speak in an urgent and excited voice"`
- `"Speak happily and joyfully"`
- `"Speak slowly and calmly"`
- `"Speak in a whisper"`

### `voice_clone` — Voice Cloning

```
voice_clone <model_path> <ref_audio> [text] [language] [ref_text]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | (required) | Path to model directory |
| `ref_audio` | (required) | Path to reference WAV file |
| `text` | "Hello! This is a test..." | Text to synthesize |
| `language` | english | Language |
| `ref_text` | (none) | Transcript of reference audio (enables ICL mode, higher quality) |

Output: `output_voice_clone.wav` (24 kHz, 16-bit PCM)

**Preparing reference audio:** Must be mono 24 kHz 16-bit WAV. Convert with ffmpeg:

```bash
ffmpeg -i input.m4a -ac 1 -ar 24000 -sample_fmt s16 reference.wav
```

**Example:**

```bash
./voice_clone models/Qwen3-TTS-12Hz-0.6B-Base reference_audio/trump.wav \
  "Hello, this is a voice cloning test." english \
  "Angered and appalled millions of Americans across the political spectrum"
```

### `api_server` — OpenAI-Compatible API

```
api_server <model_path> [--host 127.0.0.1] [--port 8080]
```

| Option | Default | Description |
|--------|---------|-------------|
| `model_path` | (required) | Path to model directory |
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 8080 | Listen port |

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/speech` | Generate speech (OpenAI-compatible) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

### API Request: `POST /v1/audio/speech`

```json
{
  "input": "Text to synthesize",
  "voice": "alloy",
  "model": "qwen3-tts",
  "response_format": "wav",
  "speed": 1.0,
  "stream": false,
  "language": "english",
  "instructions": "Speak urgently",
  "audio_sample": "<base64-encoded WAV>",
  "audio_sample_text": "Transcript of the reference audio"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | (required) | Text to synthesize (max 4096 chars) |
| `voice` | string | `"alloy"` | OpenAI name or Qwen3 speaker name (see mapping below) |
| `model` | string | — | Accepted for compatibility, ignored |
| `response_format` | string | `"wav"` | `"wav"` or `"pcm"` |
| `speed` | float | 1.0 | Speed multiplier (0.25–4.0) |
| `stream` | bool | false | Enable SSE streaming (requires `response_format: "pcm"`) |
| `language` | string | `"english"` | `english`, `chinese`, `japanese`, `korean`, `auto` |
| `instructions` | string | — | Voice style instruction (1.7B models only) |
| `audio_sample` | string | — | Base64-encoded reference WAV for voice cloning |
| `audio_sample_text` | string | — | Transcript of reference audio (required with `audio_sample`) |

**Voice name mapping** (OpenAI → Qwen3):

| OpenAI | Qwen3 |
|--------|-------|
| alloy | serena |
| echo | ryan |
| fable | vivian |
| onyx | eric |
| nova | ono_anna |
| shimmer | sohee |

You can also pass Qwen3 speaker names directly (e.g., `"voice": "vivian"`).

**Streaming:** When `stream: true` and `response_format: "pcm"`, the server returns Server-Sent Events with base64-encoded PCM chunks:

```
data: {"type":"speech.audio.delta","delta":"<base64 PCM>"}
data: {"type":"speech.audio.done"}
```

**Voice cloning via API:**

```bash
# Encode reference audio as base64
REF_B64=$(base64 < reference.wav)

curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"Hello from a cloned voice.\",
    \"voice\": \"alloy\",
    \"audio_sample\": \"$REF_B64\",
    \"audio_sample_text\": \"Transcript of the reference audio\"
  }" -o cloned.wav
```

## Build from Source

### macOS (MLX backend)

Requires Apple Silicon Mac, Xcode, and CMake.

```bash
brew install cmake
git clone https://github.com/second-state/qwen3_tts_rs.git
cd qwen3_tts_rs
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx
```

### Linux (libtorch backend)

**1. Download libtorch** from [libtorch-releases](https://github.com/second-state/libtorch-releases/releases/tag/v2.7.1):

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

**2. Set environment and build:**

```bash
export LIBTORCH=$(pwd)/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1

git clone https://github.com/second-state/qwen3_tts_rs.git
cd qwen3_tts_rs
cargo build --release
```

Alternatively, use pip-installed PyTorch instead of downloading libtorch:

```bash
pip install torch==2.7.1
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
```

### Download models and generate tokenizer

After building, download models and generate `tokenizer.json` for each:

```bash
pip install huggingface_hub transformers

huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice

python3 -c "
from transformers import AutoTokenizer
for model in ['Qwen3-TTS-12Hz-0.6B-CustomVoice', 'Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-CustomVoice']:
    path = f'models/{model}'
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tok.backend_tokenizer.save(f'{path}/tokenizer.json')
        print(f'Saved {path}/tokenizer.json')
    except Exception as e:
        print(f'Skipped {model}: {e}')
"
```

Binaries are in `target/release/`: `tts`, `voice_clone`, `api_server`.

### Rust library usage

Add to your `Cargo.toml`:

```toml
[dependencies]
qwen3-tts-rs = "0.2"
# Or for MLX backend:
# qwen3-tts-rs = { version = "0.2", default-features = false, features = ["mlx"] }
```

See the [API documentation on docs.rs](https://docs.rs/qwen3-tts-rs) for library usage examples.

## Performance (Apple M4 Mac, MLX backend)

Benchmarks with the **0.6B CustomVoice** model on Apple M4 MacBook Pro.

### CLI (`tts` / `voice_clone`)

| Test | Speaker | Language | Audio | Wall Time | RTF |
|------|---------|----------|-------|-----------|-----|
| Preset voice | Vivian | English | 5.92s | 10.93s | 1.85x |
| Preset voice | Ryan | English | 8.16s | 14.15s | 1.73x |
| Preset voice | Vivian | Chinese | 6.64s | 11.26s | 1.70x |
| Voice clone (ICL) | ref audio | English | 7.04s | 16.77s | 2.38x |

### API server (after warmup)

| Test | Voice | Mode | Audio | Wall Time | RTF |
|------|-------|------|-------|-----------|-----|
| Non-streaming WAV | alloy (serena) | full | 6.40s | 9.90s | 1.55x |
| Non-streaming WAV | echo (ryan) | full | 8.00s | 12.70s | 1.59x |
| Streaming PCM | alloy (serena) | stream | ~6.4s | 10.22s | ~1.60x |
| Voice clone WAV | alloy + ref | full | 9.04s | 20.11s | 2.22x |

**RTF** = Real-Time Factor (wall time / audio duration). Lower is better; < 1.0 means faster than real-time.

Test sentences: ~15–20 words in English ("The quick brown fox..." / "Scientists have discovered...") and Chinese.

## Architecture

```
Text → Tokenizer → Dual-stream Embeddings → TalkerModel (28-layer Transformer)
                                                    ↓
                                              codec_head → Code 0
                                                    ↓
                                        CodePredictor (5-layer Transformer) → Codes 1-15
                                                    ↓
                                              Vocoder → 24kHz Waveform
```

## License

Apache-2.0

## Credits

Based on the original Python implementation by the Alibaba Qwen team.
