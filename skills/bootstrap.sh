#!/bin/bash
# Bootstrap script for Qwen3 TTS skill
# Downloads platform-specific release (binaries + bundled libtorch) and models

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

has_nvidia_gpu() {
    command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader &>/dev/null
}

get_asset_name() {
    local platform="$1"

    case "$platform" in
    linux-x86_64)
        if has_nvidia_gpu; then
            echo "qwen3-tts-linux-x86_64-cuda"
        else
            echo "qwen3-tts-linux-x86_64"
        fi
        ;;
    linux-aarch64)
        if has_nvidia_gpu; then
            echo "qwen3-tts-linux-aarch64-cuda"
        else
            echo "qwen3-tts-linux-aarch64"
        fi
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

    # Copy bundled libtorch if present (all Linux builds)
    if [ -d "${temp_dir}/${asset_name}/libtorch" ]; then
        rm -rf "${SCRIPTS_DIR}/libtorch"
        cp -r "${temp_dir}/${asset_name}/libtorch" "${SCRIPTS_DIR}/libtorch"
    fi

    # Copy pre-generated tokenizers if present
    if [ -d "${temp_dir}/${asset_name}/tokenizers" ]; then
        rm -rf "${SCRIPTS_DIR}/tokenizers"
        cp -r "${temp_dir}/${asset_name}/tokenizers" "${SCRIPTS_DIR}/tokenizers"
    fi

    rm -rf "$temp_dir"
    echo "Release installed to ${SCRIPTS_DIR}" >&2
}

download_models() {
    echo "=== Downloading models ===" >&2

    mkdir -p "${MODELS_DIR}"

    for model in Qwen3-TTS-12Hz-0.6B-CustomVoice Qwen3-TTS-12Hz-0.6B-Base; do
        local model_dir="${MODELS_DIR}/${model}"
        if [ -d "$model_dir" ] && [ -n "$(ls -A "$model_dir" 2>/dev/null)" ]; then
            echo "${model} already downloaded, skipping." >&2
        else
            echo "Downloading ${model}..." >&2
            mkdir -p "$model_dir"

            local api_url="https://huggingface.co/api/models/Qwen/${model}"
            local hf_url="https://huggingface.co/Qwen/${model}/resolve/main"
            local files
            files=$(curl -fSL "$api_url" | grep -o '"rfilename":"[^"]*"' | sed 's/"rfilename":"//;s/"//')

            for file in $files; do
                case "$file" in
                    .gitattributes|README.md) continue ;;
                esac
                echo "  ${file}..." >&2
                mkdir -p "${model_dir}/$(dirname "$file")"
                curl -fSL -o "${model_dir}/${file}" "${hf_url}/${file}"
            done
            echo "${model} downloaded." >&2
        fi

        # Copy pre-generated tokenizer from the release asset
        cp "${SCRIPTS_DIR}/tokenizers/${model}/tokenizer.json" "${model_dir}/tokenizer.json"
    done

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
