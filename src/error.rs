// Copyright 2026 The Alibaba Qwen team.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Qwen3 TTS library.

use thiserror::Error;

/// Main error type for the Qwen3 TTS library.
#[derive(Error, Debug)]
pub enum Qwen3TTSError {
    /// Error during model loading
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Error during configuration parsing
    #[error("Configuration error: {0}")]
    Config(String),

    /// Error during audio processing
    #[error("Audio processing error: {0}")]
    Audio(String),

    /// Error during tokenization
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Error during model inference/generation
    #[error("Generation error: {0}")]
    Generation(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Unsupported feature or model type
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Network/HTTP error
    #[error("Network error: {0}")]
    Network(String),

    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Torch tensor error
    #[error("Torch error: {0}")]
    Torch(#[from] tch::TchError),

    /// Base64 decoding error
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    /// URL parsing error
    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),
}

/// Result type alias for Qwen3 TTS operations.
pub type Result<T> = std::result::Result<T, Qwen3TTSError>;
