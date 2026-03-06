#!/bin/bash
# Installer for Qwen3 TTS Rust — downloads binaries (with bundled libtorch), models, and reference audio
# Usage: curl -sSf https://raw.githubusercontent.com/second-state/qwen3_tts_rs/main/install.sh | bash

set -e

REPO="second-state/qwen3_tts_rs"
INSTALL_DIR="./qwen3_tts_rs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Detect platform
# ---------------------------------------------------------------------------
detect_platform() {
    case "$(uname -s)" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="darwin" ;;
        *)
            err "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)    ARCH="x86_64" ;;
        aarch64|arm64)   ARCH="aarch64" ;;
        *)
            err "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac

    # CUDA detection (Linux only — macOS uses Metal via MLX)
    CUDA_DRIVER=""
    if [ "$OS" = "linux" ]; then
        if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader &>/dev/null; then
            CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        fi
    fi

    info "Platform: ${OS} ${ARCH}${CUDA_DRIVER:+ (NVIDIA driver ${CUDA_DRIVER})}"
}

# ---------------------------------------------------------------------------
# 2. Resolve release asset name
# ---------------------------------------------------------------------------
resolve_asset() {
    case "${OS}-${ARCH}" in
        darwin-aarch64)  ASSET_NAME="qwen3-tts-macos-aarch64" ;;
        linux-x86_64)
            if [ -n "$CUDA_DRIVER" ]; then
                info "NVIDIA GPU detected. Choose build variant:"
                echo "  1) CUDA  (recommended for GPU)"
                echo "  2) CPU only"
                printf "Select variant [1]: "
                read -r variant </dev/tty
                variant="${variant:-1}"
                case "$variant" in
                    1) ASSET_NAME="qwen3-tts-linux-x86_64-cuda" ;;
                    2) ASSET_NAME="qwen3-tts-linux-x86_64" ;;
                    *) warn "Invalid choice, defaulting to CUDA."
                       ASSET_NAME="qwen3-tts-linux-x86_64-cuda" ;;
                esac
            else
                ASSET_NAME="qwen3-tts-linux-x86_64"
            fi
            ;;
        linux-aarch64)
            if [ -n "$CUDA_DRIVER" ]; then
                info "NVIDIA GPU detected on ARM64 (Jetson). Choose build variant:"
                echo "  1) CUDA  (recommended for Jetson)"
                echo "  2) CPU only"
                printf "Select variant [1]: "
                read -r variant </dev/tty
                variant="${variant:-1}"
                case "$variant" in
                    1) ASSET_NAME="qwen3-tts-linux-aarch64-cuda" ;;
                    2) ASSET_NAME="qwen3-tts-linux-aarch64" ;;
                    *) warn "Invalid choice, defaulting to CUDA."
                       ASSET_NAME="qwen3-tts-linux-aarch64-cuda" ;;
                esac
            else
                ASSET_NAME="qwen3-tts-linux-aarch64"
            fi
            ;;
        *)
            err "Unsupported platform: ${OS}-${ARCH}"
            exit 1
            ;;
    esac
    info "Release asset: ${ASSET_NAME}"
}

# ---------------------------------------------------------------------------
# 3. Download & extract release (libtorch is bundled for all Linux builds)
# ---------------------------------------------------------------------------
download_release() {
    local zip_name="${ASSET_NAME}.zip"
    local url="https://github.com/${REPO}/releases/latest/download/${zip_name}"

    info "Downloading release..."
    mkdir -p "${INSTALL_DIR}"

    local temp_dir
    temp_dir=$(mktemp -d)

    curl -fSL -o "${temp_dir}/${zip_name}" "$url"
    info "Extracting release..."
    unzip -q "${temp_dir}/${zip_name}" -d "${temp_dir}"

    cp -r "${temp_dir}/${ASSET_NAME}/"* "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/tts" "${INSTALL_DIR}/voice_clone"

    rm -rf "$temp_dir"
    ok "Binaries installed to ${INSTALL_DIR}/"
}

# ---------------------------------------------------------------------------
# 4. Model selection & download
# ---------------------------------------------------------------------------
download_models() {
    echo ""
    info "Select model size:"
    echo "  1) 0.6B (Recommended)"
    echo "  2) 1.7B"
    printf "Select model [1]: "
    read -r model_choice </dev/tty
    model_choice="${model_choice:-1}"

    if [ "$model_choice" = "2" ]; then
        MODEL_SIZE="1.7B"
        CUSTOM_VOICE_MODEL="Qwen3-TTS-12Hz-1.7B-CustomVoice"
    else
        MODEL_SIZE="0.6B"
        CUSTOM_VOICE_MODEL="Qwen3-TTS-12Hz-0.6B-CustomVoice"
    fi
    # Base model is always 0.6B (no 1.7B Base exists)
    BASE_MODEL="Qwen3-TTS-12Hz-0.6B-Base"

    local models_dir="${INSTALL_DIR}/models"
    mkdir -p "$models_dir"

    for model in "$CUSTOM_VOICE_MODEL" "$BASE_MODEL"; do
        local model_dir="${models_dir}/${model}"
        if [ -d "$model_dir" ] && [ -n "$(ls -A "$model_dir" 2>/dev/null)" ]; then
            ok "${model} already downloaded, skipping."
        else
            info "Downloading ${model}..."
            mkdir -p "$model_dir"

            local api_url="https://huggingface.co/api/models/Qwen/${model}"
            local hf_url="https://huggingface.co/Qwen/${model}/resolve/main"
            local files
            files=$(curl -fSL "$api_url" | grep -o '"rfilename":"[^"]*"' | sed 's/"rfilename":"//;s/"//')

            for file in $files; do
                case "$file" in
                    .gitattributes|README.md) continue ;;
                esac
                info "  ${file}..."
                mkdir -p "${model_dir}/$(dirname "$file")"
                curl -fSL -o "${model_dir}/${file}" "${hf_url}/${file}"
            done
            ok "${model} downloaded."
        fi

        # Copy pre-generated tokenizer from the release asset
        cp "${INSTALL_DIR}/tokenizers/${model}/tokenizer.json" "${model_dir}/tokenizer.json"
    done
    ok "Models ready."
}

# ---------------------------------------------------------------------------
# 5. Download reference audio
# ---------------------------------------------------------------------------
download_reference_audio() {
    local ref_dir="${INSTALL_DIR}/reference_audio"
    mkdir -p "$ref_dir"

    local base_url="https://raw.githubusercontent.com/${REPO}/main/reference_audio"

    info "Downloading reference audio..."
    for file in trump.wav trump.txt elon_musk.wav elon_musk.txt; do
        if [ ! -f "${ref_dir}/${file}" ]; then
            curl -fSL -o "${ref_dir}/${file}" "${base_url}/${file}"
        fi
    done
    ok "Reference audio saved to ${ref_dir}/"
}

# ---------------------------------------------------------------------------
# 6. Done — print sample commands
# ---------------------------------------------------------------------------
print_usage() {
    local cv_model_path="models/${CUSTOM_VOICE_MODEL}"
    local base_model_path="models/${BASE_MODEL}"

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN} Installation complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""

    echo "Text-to-Speech:"
    echo ""
    echo "  cd ${INSTALL_DIR}"
    echo "  ./tts ${cv_model_path} \"Hello! This is a test of Qwen3 TTS.\" Vivian english"
    echo ""

    echo "Voice Cloning (ICL mode):"
    echo ""
    echo "  cd ${INSTALL_DIR}"
    echo "  ./voice_clone ${base_model_path} reference_audio/trump.wav \\"
    echo "    \"Hello, this is a voice cloning test.\" english \\"
    echo "    \"Angered and appalled millions of Americans across the political spectrum\""
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    info "Qwen3 TTS Rust Installer"
    echo ""

    detect_platform
    resolve_asset
    download_release
    download_models
    download_reference_audio
    print_usage
}

main "$@"
