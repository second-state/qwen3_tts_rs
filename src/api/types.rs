// Copyright 2026 Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Request/response types for the OpenAI-compatible TTS API.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

/// Maximum input text length in characters.
pub const MAX_INPUT_LENGTH: usize = 4096;

/// OpenAI-compatible speech synthesis request.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// Text to synthesize (max 4096 characters).
    pub input: String,

    /// Voice name. Accepts OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer)
    /// or Qwen3 speaker names directly (serena, vivian, ryan, etc.).
    #[serde(default = "default_voice")]
    pub voice: String,

    /// Model name (accepted for compatibility, ignored).
    #[serde(default)]
    pub model: Option<String>,

    /// Output audio format: "wav" (default) or "pcm".
    #[serde(default = "default_response_format")]
    pub response_format: String,

    /// Speed multiplier (0.25 to 4.0, default 1.0).
    #[serde(default = "default_speed")]
    pub speed: f32,

    /// Enable SSE streaming (default false). Requires response_format "pcm".
    #[serde(default)]
    pub stream: bool,

    /// Language (default: "english"). Supports: english, chinese, japanese, korean, auto.
    #[serde(default)]
    pub language: Option<String>,

    /// Voice instruction for controlling speech style (e.g. "Speak in an urgent voice").
    #[serde(default)]
    pub instructions: Option<String>,

    /// Base64-encoded reference audio for voice cloning.
    #[serde(default)]
    pub audio_sample: Option<String>,

    /// Transcript of the reference audio for ICL (in-context learning) mode.
    /// When provided together with audio_sample, enables higher-quality voice cloning.
    #[serde(default)]
    pub audio_sample_text: Option<String>,
}

fn default_voice() -> String {
    "alloy".to_string()
}

fn default_response_format() -> String {
    "wav".to_string()
}

fn default_speed() -> f32 {
    1.0
}

/// Map OpenAI voice names to Qwen3 speaker names.
/// If the voice is already a Qwen3 speaker name, pass it through.
pub fn map_voice(voice: &str) -> &str {
    match voice.to_lowercase().as_str() {
        "alloy" => "serena",
        "echo" => "ryan",
        "fable" => "vivian",
        "onyx" => "eric",
        "nova" => "ono_anna",
        "shimmer" => "sohee",
        // Pass through Qwen3 native speaker names
        other => {
            // Return original voice string for native names
            // We leak a check here to just return the input
            let _ = other;
            voice
        }
    }
}

/// SSE event for streaming audio.
#[derive(Debug, Serialize)]
pub struct SseAudioDelta {
    /// Event type: "speech.audio.delta"
    #[serde(rename = "type")]
    pub event_type: &'static str,
    /// Base64-encoded PCM audio data.
    pub delta: String,
}

/// SSE event signaling end of audio stream.
#[derive(Debug, Serialize)]
pub struct SseAudioDone {
    /// Event type: "speech.audio.done"
    #[serde(rename = "type")]
    pub event_type: &'static str,
}

/// SSE error event.
#[derive(Debug, Serialize)]
pub struct SseError {
    /// Event type: "error"
    #[serde(rename = "type")]
    pub event_type: &'static str,
    /// Error message.
    pub error: String,
}

/// Health check response.
#[derive(Serialize)]
pub struct HealthResponse {
    /// Status string.
    pub status: &'static str,
}

/// Model info for the /v1/models endpoint.
#[derive(Serialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: &'static str,
    /// Object type.
    pub object: &'static str,
    /// Model owner.
    pub owned_by: &'static str,
}

/// Models list response.
#[derive(Serialize)]
pub struct ModelsResponse {
    /// Object type.
    pub object: &'static str,
    /// List of models.
    pub data: Vec<ModelInfo>,
}

/// API error type with OpenAI-compatible error response format.
pub struct ApiError {
    /// HTTP status code.
    pub status: StatusCode,
    /// Error message.
    pub message: String,
}

impl ApiError {
    /// Create a 400 Bad Request error.
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    /// Create a 500 Internal Server Error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "code": null
            }
        });
        (self.status, axum::Json(body)).into_response()
    }
}

impl From<crate::error::Qwen3TTSError> for ApiError {
    fn from(e: crate::error::Qwen3TTSError) -> Self {
        ApiError::internal(e.to_string())
    }
}
