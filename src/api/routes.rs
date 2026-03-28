// Copyright 2026 Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Route handlers for the OpenAI-compatible TTS API.

use crate::api::chunking::{chunk_text_streaming, FIRST_CHUNK_MAX, REST_CHUNK_MAX};
use crate::api::types::*;
use crate::audio::{load_wav_bytes, resample, write_wav_bytes};
use crate::audio_encoder::AudioEncoder;
use crate::inference::TTSInference;
use crate::speaker_encoder::SpeakerEncoder;
use crate::tensor::{DType, Device, Tensor};

use axum::extract::State;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

/// Shared application state containing loaded models.
pub struct Models {
    /// Main TTS inference engine.
    pub inference: TTSInference,
    /// Speaker encoder for voice cloning (optional, loaded from Base models).
    pub speaker_encoder: Option<SpeakerEncoder>,
    /// Audio encoder for ICL voice cloning (optional, requires speech_tokenizer).
    pub audio_encoder: Option<AudioEncoder>,
}

/// Thread-safe shared state type.
pub type AppState = Arc<Mutex<Models>>;

/// GET /health
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// GET /v1/models
pub async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: "qwen3-tts",
            object: "model",
            owned_by: "qwen",
        }],
    })
}

/// POST /v1/audio/speech — dispatch to full or streaming handler.
pub async fn speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Result<Response, ApiError> {
    // Validate input
    if req.input.is_empty() {
        return Err(ApiError::bad_request("Input text is empty"));
    }
    if req.input.len() > MAX_INPUT_LENGTH {
        return Err(ApiError::bad_request(format!(
            "Input text exceeds maximum length of {} characters",
            MAX_INPUT_LENGTH
        )));
    }
    if req.speed < 0.25 || req.speed > 4.0 {
        return Err(ApiError::bad_request(
            "Speed must be between 0.25 and 4.0",
        ));
    }

    let format = req.response_format.to_lowercase();
    if format != "wav" && format != "pcm" {
        return Err(ApiError::bad_request(
            "Unsupported response_format. Use \"wav\" or \"pcm\"",
        ));
    }

    if req.stream {
        if format != "pcm" {
            return Err(ApiError::bad_request(
                "Streaming requires response_format \"pcm\"",
            ));
        }
        speech_stream(state, req).await
    } else {
        speech_full(state, req, &format).await
    }
}

/// Non-streaming speech generation — returns complete audio.
async fn speech_full(
    state: AppState,
    req: SpeechRequest,
    format: &str,
) -> Result<Response, ApiError> {
    let format = format.to_string();
    let speed = req.speed;

    let (waveform, sample_rate) =
        tokio::task::spawn_blocking(move || generate_audio(&state, &req))
            .await
            .map_err(|e| ApiError::internal(format!("Task join error: {}", e)))??;

    // Apply speed adjustment
    let waveform = apply_speed(&waveform, speed);

    if format == "wav" {
        let wav_bytes = write_wav_bytes(&waveform, sample_rate)
            .map_err(|e| ApiError::internal(format!("WAV encoding error: {}", e)))?;
        Ok((
            [(header::CONTENT_TYPE, "audio/wav")],
            wav_bytes,
        )
            .into_response())
    } else {
        // PCM: 16-bit signed LE mono
        let pcm = samples_to_pcm_bytes(&waveform);
        Ok((
            [(header::CONTENT_TYPE, "audio/pcm")],
            pcm,
        )
            .into_response())
    }
}

/// Streaming speech generation — returns SSE with base64 PCM chunks.
async fn speech_stream(
    state: AppState,
    req: SpeechRequest,
) -> Result<Response, ApiError> {
    let chunks = chunk_text_streaming(&req.input, FIRST_CHUNK_MAX, REST_CHUNK_MAX);
    if chunks.is_empty() {
        return Err(ApiError::bad_request("Input text produced no chunks"));
    }

    let voice = map_voice(&req.voice).to_string();
    let language = req
        .language
        .clone()
        .unwrap_or_else(|| "english".to_string());
    let instructions = req.instructions.clone().unwrap_or_default();
    let speed = req.speed;

    // Pre-process voice cloning data if provided
    let clone_data: Option<Arc<CloneData>> = if req.audio_sample.is_some() {
        Some(Arc::new(prepare_clone_data(&state, &req)?))
    } else {
        None
    };

    let stream = async_stream::stream! {
        for chunk in chunks {
            let state = state.clone();
            let voice = voice.clone();
            let language = language.clone();
            let instructions = instructions.clone();
            let clone_data = clone_data.clone();

            let result = tokio::task::spawn_blocking(move || {
                generate_chunk_audio(
                    &state,
                    &chunk,
                    &voice,
                    &language,
                    &instructions,
                    speed,
                    clone_data.as_deref(),
                )
            })
            .await;

            match result {
                Ok(Ok(pcm_bytes)) => {
                    let b64 = BASE64_STANDARD.encode(&pcm_bytes);
                    let delta = SseAudioDelta {
                        event_type: "speech.audio.delta",
                        delta: b64,
                    };
                    let data = serde_json::to_string(&delta).unwrap_or_default();
                    yield Ok::<_, Infallible>(Event::default().data(data));
                }
                Ok(Err(e)) => {
                    let err_event = SseError {
                        event_type: "error",
                        error: e.message,
                    };
                    let data = serde_json::to_string(&err_event).unwrap_or_default();
                    yield Ok(Event::default().data(data));
                }
                Err(e) => {
                    let err_event = SseError {
                        event_type: "error",
                        error: format!("Task join error: {}", e),
                    };
                    let data = serde_json::to_string(&err_event).unwrap_or_default();
                    yield Ok(Event::default().data(data));
                }
            }
        }

        // Final "done" event
        let done = SseAudioDone {
            event_type: "speech.audio.done",
        };
        let data = serde_json::to_string(&done).unwrap_or_default();
        yield Ok(Event::default().data(data));
    };

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Voice cloning data prepared from the request (always ICL mode).
struct CloneData {
    speaker_embedding: Tensor,
    ref_codes: Vec<Vec<i64>>,
    ref_text: String,
}

/// Prepare voice cloning data using ICL mode.
///
/// All voice cloning goes through ICL (in-context learning) which uses the audio encoder
/// to extract codec tokens from the reference audio. The speaker encoder is used for the
/// x-vector embedding when available; otherwise a zero embedding is used.
fn prepare_clone_data(state: &AppState, req: &SpeechRequest) -> Result<CloneData, ApiError> {
    let models = state.lock().map_err(|e| ApiError::internal(e.to_string()))?;

    let audio_encoder = models.audio_encoder.as_ref().ok_or_else(|| {
        ApiError::bad_request("Voice cloning requires an audio encoder (speech_tokenizer)")
    })?;

    let ref_text = req.audio_sample_text.as_ref().ok_or_else(|| {
        ApiError::bad_request(
            "Voice cloning requires audio_sample_text (transcript of the reference audio)",
        )
    })?;

    // Decode base64 audio
    let audio_b64 = req.audio_sample.as_ref().unwrap();
    let audio_bytes = BASE64_STANDARD
        .decode(audio_b64)
        .map_err(|e| ApiError::bad_request(format!("Invalid base64 audio_sample: {}", e)))?;

    // Load WAV and resample to 24kHz
    let (samples, sr) = load_wav_bytes(&audio_bytes)
        .map_err(|e| ApiError::bad_request(format!("Failed to decode audio_sample WAV: {}", e)))?;

    let se_sr = models.inference.config().speaker_encoder_config.sample_rate;
    let samples = if sr != se_sr {
        resample(&samples, sr, se_sr)
            .map_err(|e| ApiError::internal(format!("Resampling error: {}", e)))?
    } else {
        samples
    };

    // Extract speaker embedding if speaker encoder is available, otherwise use zeros
    let speaker_embedding = if let Some(speaker_encoder) = &models.speaker_encoder {
        speaker_encoder
            .extract_embedding(&samples)
            .map_err(|e| ApiError::internal(format!("Speaker embedding extraction failed: {}", e)))?
    } else {
        let enc_dim = models.inference.config().speaker_encoder_config.enc_dim as i64;
        Tensor::zeros(&[enc_dim], DType::Float32, Device::Cpu)
    };

    // Encode reference audio to codec tokens
    let ref_codes = audio_encoder
        .encode(&samples)
        .map_err(|e| ApiError::internal(format!("Audio encoding failed: {}", e)))?;

    Ok(CloneData {
        speaker_embedding,
        ref_codes,
        ref_text: ref_text.clone(),
    })
}

/// Generate audio for a single text (non-streaming path).
fn generate_audio(state: &AppState, req: &SpeechRequest) -> Result<(Vec<f32>, u32), ApiError> {
    let voice = map_voice(&req.voice);
    let language = req
        .language
        .as_deref()
        .unwrap_or("english");
    let instructions = req.instructions.as_deref().unwrap_or("");

    let models = state.lock().map_err(|e| ApiError::internal(e.to_string()))?;

    // Voice cloning path (ICL mode)
    if let Some(audio_b64) = &req.audio_sample {
        let ref_text = req.audio_sample_text.as_deref().ok_or_else(|| {
            ApiError::bad_request(
                "Voice cloning requires audio_sample_text (transcript of the reference audio)",
            )
        })?;
        return generate_with_clone(&models, &req.input, language, audio_b64, ref_text);
    }

    // Standard generation with instruction support
    if !instructions.is_empty() {
        models
            .inference
            .generate_with_instruct(&req.input, voice, language, instructions, 0.9, 50, 2048)
            .map_err(ApiError::from)
    } else {
        models
            .inference
            .generate_with_params(&req.input, voice, language, 0.9, 50, 2048)
            .map_err(ApiError::from)
    }
}

/// Generate audio using voice cloning (ICL mode).
fn generate_with_clone(
    models: &Models,
    text: &str,
    language: &str,
    audio_b64: &str,
    ref_text: &str,
) -> Result<(Vec<f32>, u32), ApiError> {
    let audio_encoder = models.audio_encoder.as_ref().ok_or_else(|| {
        ApiError::bad_request("Voice cloning requires an audio encoder (speech_tokenizer)")
    })?;

    let audio_bytes = BASE64_STANDARD
        .decode(audio_b64)
        .map_err(|e| ApiError::bad_request(format!("Invalid base64 audio_sample: {}", e)))?;

    let (samples, sr) = load_wav_bytes(&audio_bytes)
        .map_err(|e| ApiError::bad_request(format!("Failed to decode audio_sample WAV: {}", e)))?;

    let se_sr = models.inference.config().speaker_encoder_config.sample_rate;
    let samples = if sr != se_sr {
        resample(&samples, sr, se_sr)
            .map_err(|e| ApiError::internal(format!("Resampling error: {}", e)))?
    } else {
        samples
    };

    // Use speaker encoder if available, otherwise zero embedding
    let speaker_embedding = if let Some(speaker_encoder) = &models.speaker_encoder {
        speaker_encoder
            .extract_embedding(&samples)
            .map_err(|e| ApiError::internal(format!("Speaker embedding extraction failed: {}", e)))?
    } else {
        let enc_dim = models.inference.config().speaker_encoder_config.enc_dim as i64;
        Tensor::zeros(&[enc_dim], DType::Float32, Device::Cpu)
    };

    let ref_codes = audio_encoder
        .encode(&samples)
        .map_err(|e| ApiError::internal(format!("Audio encoding failed: {}", e)))?;

    models
        .inference
        .generate_with_icl(text, ref_text, &ref_codes, &speaker_embedding, language, 0.9, 50, 2048)
        .map_err(ApiError::from)
}

/// Generate audio for a single chunk in the streaming path.
fn generate_chunk_audio(
    state: &AppState,
    text: &str,
    voice: &str,
    language: &str,
    instructions: &str,
    speed: f32,
    clone_data: Option<&CloneData>,
) -> Result<Vec<u8>, ApiError> {
    let models = state.lock().map_err(|e| ApiError::internal(e.to_string()))?;

    let (waveform, _sample_rate) = if let Some(cd) = clone_data {
        // Voice cloning streaming (always ICL)
        models
            .inference
            .generate_with_icl(text, &cd.ref_text, &cd.ref_codes, &cd.speaker_embedding, language, 0.9, 50, 2048)
            .map_err(ApiError::from)?
    } else if !instructions.is_empty() {
        models
            .inference
            .generate_with_instruct(text, voice, language, instructions, 0.9, 50, 2048)
            .map_err(ApiError::from)?
    } else {
        models
            .inference
            .generate_with_params(text, voice, language, 0.9, 50, 2048)
            .map_err(ApiError::from)?
    };

    let waveform = apply_speed(&waveform, speed);
    Ok(samples_to_pcm_bytes(&waveform))
}

/// Convert f32 audio samples to 16-bit signed LE PCM bytes.
fn samples_to_pcm_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let scaled = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&scaled.to_le_bytes());
    }
    bytes
}

/// Apply speed adjustment via linear interpolation.
/// speed > 1.0 = faster (fewer samples), speed < 1.0 = slower (more samples).
fn apply_speed(samples: &[f32], speed: f32) -> Vec<f32> {
    if (speed - 1.0).abs() < 0.01 || samples.is_empty() {
        return samples.to_vec();
    }

    let new_len = (samples.len() as f64 / speed as f64) as usize;
    let mut output = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_pos = i as f64 * speed as f64;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        if idx + 1 < samples.len() {
            output.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }

    output
}
