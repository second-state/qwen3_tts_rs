---
name: audio-tts
description: Generate speech audio from text using Qwen3 TTS, or clone a voice from reference audio. Triggered when the user wants to convert text to speech, generate audio, read text aloud, or clone/mimic a voice. Supports multiple speakers, English and Chinese, and emotion/style control.
---

# Qwen3 TTS — Text-to-Speech and Voice Cloning

Generate speech audio from text, or clone a voice from a reference audio file.

## Binaries

* `{baseDir}/scripts/tts` — Text-to-speech generation with named speakers.
* `{baseDir}/scripts/voice_clone` — Voice cloning from a reference audio file.

## Models

* `{baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice` — Named speaker TTS (0.6B parameters).
* `{baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-Base` — Voice cloning from reference audio (0.6B parameters).

## When to Use Which Tool

- **`tts`** — When the user wants to generate speech from text using a named speaker (Vivian, Ryan, etc.). Supports English and Chinese.
- **`voice_clone`** — When the user wants to clone a specific voice from a reference audio file and generate new speech in that voice.

## Environment Setup (Linux only)

On Linux, the binaries require libtorch shared libraries. Set the library path before running:

```bash
export LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH
```

On macOS, no environment setup is needed (the binaries use the MLX backend).

## Text-to-Speech

Generate speech audio from text with a named speaker.

```bash
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/tts \
  {baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "<text>" \
  <speaker> \
  <language>
```

On macOS, omit the `LD_LIBRARY_PATH` prefix:

```bash
{baseDir}/scripts/tts \
  {baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "<text>" \
  <speaker> \
  <language>
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| model_path | Yes | Path to the model directory |
| text | Yes | The text to synthesize as speech |
| speaker | Yes | Speaker name (see Available Speakers below) |
| language | Yes | `english` or `chinese` |

### Available Speakers

Vivian, Serena, Ryan, Aiden, Uncle_fu, Ono_anna, Sohee, Eric, Dylan.

### Output

Generates `output.wav` (24kHz mono WAV) in the current working directory.

### Example

```bash
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/tts \
  {baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "Hello! Welcome to the Qwen3 text-to-speech system." \
  Vivian \
  english
```

## Voice Cloning

Clone a voice from a reference audio file and generate new speech.

```bash
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/voice_clone \
  {baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-Base \
  <reference_audio.wav> \
  "<text>" \
  <language> \
  ["<reference_text>"]
```

On macOS, omit the `LD_LIBRARY_PATH` prefix.

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| model_path | Yes | Path to the Base model directory |
| reference_audio | Yes | Path to reference WAV file (mono 24kHz 16-bit) |
| text | Yes | The text to synthesize in the cloned voice |
| language | Yes | `english` or `chinese` |
| reference_text | No | Transcript of the reference audio (enables ICL mode for higher quality) |

### Reference Audio Requirements

The reference audio must be a **mono 24kHz 16-bit WAV** file. Convert from other formats with ffmpeg:

```bash
ffmpeg -i input.m4a -ac 1 -ar 24000 -sample_fmt s16 reference.wav
```

### Output

Generates `output_voice_clone.wav` (24kHz mono WAV) in the current working directory.

### Example — X-vector Mode (no reference text)

```bash
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/voice_clone \
  {baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-Base \
  reference.wav \
  "This is a voice cloning test." \
  english
```

### Example — ICL Mode (with reference text, higher quality)

```bash
LD_LIBRARY_PATH={baseDir}/scripts/libtorch/lib:$LD_LIBRARY_PATH \
  {baseDir}/scripts/voice_clone \
  {baseDir}/scripts/models/Qwen3-TTS-12Hz-0.6B-Base \
  reference.wav \
  "This is a voice cloning test with in-context learning." \
  english \
  "The transcript of what was said in the reference audio."
```

When a reference text transcript is provided as the last argument, ICL (In-Context Learning) mode is used. This encodes the reference audio into codec tokens and conditions generation on both the speaker embedding and the reference audio/text, producing higher fidelity voice cloning.

## Workflow

### 1. Determine the Task

- If the user wants to generate speech from text with a named speaker, use `tts`.
- If the user wants to clone a voice from an audio file, use `voice_clone`.

### 2. Prepare Input

- For `tts`: Identify the text, speaker name, and language from the user's request. Default to `Vivian` and `english` if not specified.
- For `voice_clone`: Ensure the reference audio is a mono 24kHz 16-bit WAV. Convert if needed using ffmpeg.

### 3. Run the Command

Run the appropriate binary with `LD_LIBRARY_PATH` set (Linux only). Use the full paths to the binaries and model directories.

### 4. Return the Output

The output WAV file will be in the current working directory:
- `tts` produces `output.wav`
- `voice_clone` produces `output_voice_clone.wav`

Inform the user of the output file path.
