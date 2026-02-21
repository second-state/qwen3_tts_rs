# Qwen3 TTS Skill — Installation Guide

Follow these steps to install the Qwen3 TTS skill that provides text-to-speech and voice cloning capabilities.

## Prerequisites

- `curl` (for downloading)
- `unzip` and `tar` (for extraction)
- `bash` (shell)
- `git` (for cloning)
- `pip` with `huggingface_hub` and `transformers` (for model download and tokenizer generation)

## Quick Install (Recommended)

```bash
SKILL_DIR="${HOME}/.openclaw/skills/audio_tts"
mkdir -p "$SKILL_DIR"

# Clone the skill
git clone --depth 1 https://github.com/juntao/qwen3_tts_rs.git /tmp/qwen3-tts-repo
cp -r /tmp/qwen3-tts-repo/skills/* "$SKILL_DIR"
rm -rf /tmp/qwen3-tts-repo

# Download platform-specific binaries, libtorch, and models
"${SKILL_DIR}/bootstrap.sh"
```

After installation, verify it works:

```bash
# Test TTS (requires model download to complete)
~/.openclaw/skills/audio_tts/scripts/tts \
  ~/.openclaw/skills/audio_tts/scripts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "Hello, this is a test." \
  Vivian \
  english
ls -la output.wav
```

## Manual Installation

If automatic download fails, manually download the components:

1. Go to https://github.com/juntao/qwen3_tts_rs/releases/latest
2. Download the tarball for your platform:
   - `qwen3-tts-linux-x86_64-cpu.tar.gz` (Linux x86_64 CPU)
   - `qwen3-tts-linux-x86_64-cuda.tar.gz` (Linux x86_64 with CUDA)
   - `qwen3-tts-linux-aarch64-cpu.tar.gz` (Linux ARM64 CPU)
   - `qwen3-tts-macos-arm64.tar.gz` (macOS Apple Silicon, MLX backend)
3. Extract to `~/.openclaw/skills/audio_tts/scripts/`
4. Make executable:
   ```bash
   chmod +x ~/.openclaw/skills/audio_tts/scripts/tts
   chmod +x ~/.openclaw/skills/audio_tts/scripts/voice_clone
   ```
5. For Linux, download libtorch (see README for URLs) and extract to `~/.openclaw/skills/audio_tts/scripts/libtorch/`
6. Download models using `huggingface-cli` (see README)

## Troubleshooting

### Download Failed
Check network connectivity:
```bash
curl -I "https://github.com/juntao/qwen3_tts_rs/releases/latest"
```

### Unsupported Platform
Check your platform:
```bash
echo "OS: $(uname -s), Arch: $(uname -m)"
```

Supported: Linux (x86_64, aarch64) and macOS (Apple Silicon arm64).

### Missing libtorch (Linux only)
Ensure `LD_LIBRARY_PATH` includes the libtorch lib directory. The SKILL.md instructions set this automatically.
