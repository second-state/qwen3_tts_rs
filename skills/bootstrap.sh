#!/bin/bash
# Bootstrap script for Qwen3 TTS skill
# Downloads platform-specific binaries, libtorch (Linux only), and models

set -e

REPO="second-state/qwen3_tts_rs"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SKILL_DIR}/scripts"
MODELS_DIR="${SCRIPTS_DIR}/models"

detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="darwin" ;;
        *)
            echo "Error: Unsupported operating system: $(uname -s)" >&2
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *)
            echo "Error: Unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac

    echo "${os}-${arch}"
}

detect_cuda() {
    # Check for NVIDIA GPU and CUDA toolkit
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        return 0
    fi
    if command -v nvcc &>/dev/null; then
        return 0
    fi
    if [ -d "/usr/local/cuda" ]; then
        return 0
    fi
    return 1
}

get_release_suffix() {
    local platform="$1"

    case "$platform" in
        linux-x86_64)
            if detect_cuda; then
                echo "linux-x86_64-cuda"
            else
                echo "linux-x86_64-cpu"
            fi
            ;;
        linux-aarch64)
            echo "linux-aarch64-cpu"
            ;;
        darwin-aarch64)
            echo "macos-arm64"
            ;;
        *)
            echo "Error: Unsupported platform: ${platform}" >&2
            exit 1
            ;;
    esac
}

download_binaries() {
    local suffix="$1"
    local asset_name="qwen3-tts-${suffix}.tar.gz"

    echo "=== Downloading binaries (${suffix}) ===" >&2

    mkdir -p "${SCRIPTS_DIR}"

    # Get download URL from latest release
    local api_url="https://api.github.com/repos/${REPO}/releases/latest"
    local download_url
    download_url=$(curl -sL "$api_url" | grep -o "https://github.com/${REPO}/releases/download/[^\"]*${asset_name}" | head -1)

    if [ -z "$download_url" ]; then
        echo "Error: Could not find release asset ${asset_name}" >&2
        echo "Check https://github.com/${REPO}/releases for available downloads." >&2
        exit 1
    fi

    local temp_dir
    temp_dir=$(mktemp -d)

    echo "Fetching from: ${download_url}" >&2
    curl -sL -o "${temp_dir}/${asset_name}" "$download_url"

    echo "Extracting binaries..." >&2
    tar xzf "${temp_dir}/${asset_name}" -C "${temp_dir}"

    # Copy binaries from extracted directory
    cp "${temp_dir}"/qwen3-tts-*/tts "${SCRIPTS_DIR}/"
    cp "${temp_dir}"/qwen3-tts-*/voice_clone "${SCRIPTS_DIR}/"
    chmod +x "${SCRIPTS_DIR}/tts" "${SCRIPTS_DIR}/voice_clone"

    # Copy mlx.metallib if present (macOS MLX builds)
    if ls "${temp_dir}"/qwen3-tts-*/mlx.metallib &>/dev/null; then
        cp "${temp_dir}"/qwen3-tts-*/mlx.metallib "${SCRIPTS_DIR}/"
        echo "mlx.metallib installed to ${SCRIPTS_DIR}" >&2
    fi

    rm -rf "$temp_dir"
    echo "Binaries installed to ${SCRIPTS_DIR}" >&2
}

download_libtorch() {
    local platform="$1"
    local suffix="$2"

    # macOS MLX backend does not need libtorch
    if [[ "$platform" == darwin-* ]]; then
        echo "=== Skipping libtorch (macOS uses MLX backend) ===" >&2
        return
    fi

    echo "=== Downloading libtorch ===" >&2

    local libtorch_url=""
    local archive_name=""

    case "$suffix" in
        linux-x86_64-cpu)
            libtorch_url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip"
            archive_name="libtorch.zip"
            ;;
        linux-x86_64-cuda)
            libtorch_url="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip"
            archive_name="libtorch.zip"
            ;;
        linux-aarch64-cpu)
            libtorch_url="https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz"
            archive_name="libtorch.tar.gz"
            ;;
    esac

    if [ -z "$libtorch_url" ]; then
        echo "Error: No libtorch URL for suffix ${suffix}" >&2
        exit 1
    fi

    local temp_dir
    temp_dir=$(mktemp -d)

    echo "Fetching from: ${libtorch_url}" >&2
    curl -sL -o "${temp_dir}/${archive_name}" "$libtorch_url"

    echo "Extracting libtorch..." >&2
    if [[ "$archive_name" == *.zip ]]; then
        unzip -q "${temp_dir}/${archive_name}" -d "${temp_dir}"
    else
        tar xzf "${temp_dir}/${archive_name}" -C "${temp_dir}"
    fi

    # Move libtorch into scripts/
    rm -rf "${SCRIPTS_DIR}/libtorch"
    mv "${temp_dir}/libtorch" "${SCRIPTS_DIR}/libtorch"

    rm -rf "$temp_dir"
    echo "libtorch installed to ${SCRIPTS_DIR}/libtorch" >&2
}

download_models() {
    echo "=== Downloading models ===" >&2

    mkdir -p "${MODELS_DIR}"

    # Check for huggingface-cli
    if ! command -v huggingface-cli &>/dev/null; then
        echo "Installing huggingface_hub..." >&2
        pip install -q huggingface_hub transformers
    fi

    # Download CustomVoice 0.6B
    local cv_dir="${MODELS_DIR}/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    if [ ! -d "$cv_dir" ] || [ -z "$(ls -A "$cv_dir" 2>/dev/null)" ]; then
        echo "Downloading Qwen3-TTS-12Hz-0.6B-CustomVoice..." >&2
        huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir "$cv_dir"
    else
        echo "CustomVoice 0.6B already downloaded, skipping." >&2
    fi

    # Download Base 0.6B
    local base_dir="${MODELS_DIR}/Qwen3-TTS-12Hz-0.6B-Base"
    if [ ! -d "$base_dir" ] || [ -z "$(ls -A "$base_dir" 2>/dev/null)" ]; then
        echo "Downloading Qwen3-TTS-12Hz-0.6B-Base..." >&2
        huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir "$base_dir"
    else
        echo "Base 0.6B already downloaded, skipping." >&2
    fi

    # Generate tokenizer.json files
    echo "Generating tokenizer.json files..." >&2
    python3 -c "
from transformers import AutoTokenizer
for model in ['Qwen3-TTS-12Hz-0.6B-CustomVoice', 'Qwen3-TTS-12Hz-0.6B-Base']:
    path = '${MODELS_DIR}/' + model
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.backend_tokenizer.save(path + '/tokenizer.json')
    print(f'Saved {path}/tokenizer.json')
"

    echo "Models installed to ${MODELS_DIR}" >&2
}

main() {
    local platform
    platform=$(detect_platform)
    echo "Detected platform: ${platform}" >&2

    local suffix
    suffix=$(get_release_suffix "$platform")
    echo "Release variant: ${suffix}" >&2

    download_binaries "$suffix"
    download_libtorch "$platform" "$suffix"
    download_models

    echo "" >&2
    echo "=== Installation complete ===" >&2
    echo "Installed files:" >&2
    ls -1 "${SCRIPTS_DIR}" | grep -v '^\.' >&2
}

main "$@"
