#!/bin/bash
# Bootstrap script for Qwen3 TTS skill
# Downloads platform-specific release (binaries + runtime deps) and models

set -e

REPO="second-state/qwen3_tts_rs"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SKILL_DIR}/scripts"
MODELS_DIR="${SCRIPTS_DIR}/models"

detect_platform() {
    local os arch

    case "$(uname -s)" in
    Linux*) os="linux" ;;
    Darwin*) os="darwin" ;;
    *)
        echo "Error: Unsupported operating system: $(uname -s)" >&2
        exit 1
        ;;
    esac

    case "$(uname -m)" in
    x86_64 | amd64) arch="x86_64" ;;
    aarch64 | arm64) arch="aarch64" ;;
    *)
        echo "Error: Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
    esac

    echo "${os}-${arch}"
}

get_asset_name() {
    local platform="$1"

    case "$platform" in
    linux-x86_64)
        echo "qwen3-tts-linux-x86_64"
        ;;
    linux-aarch64)
        echo "qwen3-tts-linux-aarch64"
        ;;
    darwin-aarch64)
        echo "qwen3-tts-macos-aarch64"
        ;;
    *)
        echo "Error: Unsupported platform: ${platform}" >&2
        exit 1
        ;;
    esac
}

download_release() {
    local asset_name="$1"
    local zip_name="${asset_name}.zip"

    echo "=== Downloading release (${asset_name}) ===" >&2

    mkdir -p "${SCRIPTS_DIR}"

    # Get download URL from latest release
    local api_url="https://api.github.com/repos/${REPO}/releases/latest"
    local download_url
    download_url=$(curl -sL "$api_url" | grep -o "https://github.com/${REPO}/releases/download/[^\"]*/${zip_name}" | head -1)

    if [ -z "$download_url" ]; then
        echo "Error: Could not find release asset ${zip_name}" >&2
        echo "Check https://github.com/${REPO}/releases for available downloads." >&2
        exit 1
    fi

    local temp_dir
    temp_dir=$(mktemp -d)

    echo "Fetching from: ${download_url}" >&2
    curl -sL -o "${temp_dir}/${zip_name}" "$download_url"

    echo "Extracting release..." >&2
    unzip -q "${temp_dir}/${zip_name}" -d "${temp_dir}"

    # Copy binaries
    cp "${temp_dir}/${asset_name}/tts" "${SCRIPTS_DIR}/tts"
    cp "${temp_dir}/${asset_name}/voice_clone" "${SCRIPTS_DIR}/voice_clone"
    chmod +x "${SCRIPTS_DIR}/tts" "${SCRIPTS_DIR}/voice_clone"

    # Copy mlx.metallib if present (macOS MLX backend)
    if [ -f "${temp_dir}/${asset_name}/mlx.metallib" ]; then
        cp "${temp_dir}/${asset_name}/mlx.metallib" "${SCRIPTS_DIR}/mlx.metallib"
    fi

    # Copy libtorch if present (Linux tch backend)
    if [ -d "${temp_dir}/${asset_name}/libtorch" ]; then
        rm -rf "${SCRIPTS_DIR}/libtorch"
        cp -r "${temp_dir}/${asset_name}/libtorch" "${SCRIPTS_DIR}/libtorch"
    fi

    rm -rf "$temp_dir"
    echo "Release installed to ${SCRIPTS_DIR}" >&2
}

download_models() {
    echo "=== Downloading models ===" >&2

    mkdir -p "${MODELS_DIR}"

    # Ensure huggingface_hub and transformers are available
    if ! command -v huggingface-cli &>/dev/null; then
        echo "Installing huggingface_hub and transformers..." >&2
        pip install -q huggingface_hub transformers
    fi

    for model in Qwen3-TTS-12Hz-0.6B-CustomVoice Qwen3-TTS-12Hz-0.6B-Base; do
        local model_dir="${MODELS_DIR}/${model}"
        if [ ! -d "$model_dir" ] || [ -z "$(ls -A "$model_dir" 2>/dev/null)" ]; then
            echo "Downloading ${model}..." >&2
            huggingface-cli download "Qwen/${model}" --local-dir "$model_dir"
        else
            echo "${model} already downloaded, skipping." >&2
        fi
    done

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

    local asset_name
    asset_name=$(get_asset_name "$platform")
    echo "Asset: ${asset_name}" >&2

    download_release "$asset_name"
    download_models

    echo "" >&2
    echo "=== Installation complete ===" >&2
    echo "Installed files:" >&2
    ls -1 "${SCRIPTS_DIR}" | grep -v '^\.' >&2
}

main "$@"
