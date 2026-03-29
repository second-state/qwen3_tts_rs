// Copyright 2026 Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible TTS API server for Qwen3 TTS.
//!
//! Usage:
//!   api_server <model_path> [--host 127.0.0.1] [--port 8080]
//!
//! Example:
//!   api_server ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8080

use clap::Parser;
use qwen_tts_rs::api::routes::{self, AppState, Models};
use qwen_tts_rs::audio_encoder::AudioEncoder;
use qwen_tts_rs::inference::TTSInference;
use qwen_tts_rs::speaker_encoder::SpeakerEncoder;
use qwen_tts_rs::tensor::Device;

use axum::routing::{get, post};
use axum::Router;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tower_http::trace::TraceLayer;

#[derive(Parser, Debug)]
#[command(name = "api_server", about = "OpenAI-compatible TTS API server")]
struct Args {
    /// Path to the model directory
    model_path: String,

    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let model_path = Path::new(&args.model_path);

    // Initialize MLX backend if feature enabled
    #[cfg(feature = "mlx")]
    {
        qwen_tts_rs::backend::mlx::stream::init_mlx(true);
        tracing::info!("MLX backend initialized (Metal GPU)");
    }

    // Select device
    let device = Device::Cpu;

    // Load main inference model
    tracing::info!("Loading TTS model from: {}", model_path.display());
    let inference = TTSInference::new(model_path, device)?;

    // Try to load speaker encoder for voice cloning
    let speaker_encoder = {
        let se_config = inference.config().speaker_encoder_config.clone();
        match SpeakerEncoder::load(inference.weights(), &se_config, device) {
            Ok(se) => {
                tracing::info!("Speaker encoder loaded (voice cloning enabled)");
                Some(se)
            }
            Err(e) => {
                tracing::warn!("Speaker encoder not available: {} (voice cloning disabled)", e);
                None
            }
        }
    };

    // Try to load audio encoder for ICL voice cloning
    let audio_encoder = {
        let speech_tokenizer_path = model_path
            .join("speech_tokenizer")
            .join("model.safetensors");
        if speech_tokenizer_path.exists() {
            match AudioEncoder::load(&speech_tokenizer_path, device) {
                Ok(ae) => {
                    tracing::info!("Audio encoder loaded (ICL voice cloning enabled)");
                    Some(ae)
                }
                Err(e) => {
                    tracing::warn!("Audio encoder not available: {} (ICL mode disabled)", e);
                    None
                }
            }
        } else {
            tracing::info!("No speech_tokenizer found (ICL mode disabled)");
            None
        }
    };

    // Build shared state
    let state: AppState = Arc::new(Mutex::new(Models {
        inference,
        speaker_encoder,
        audio_encoder,
    }));

    // Build router
    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/audio/speech", post(routes::speech))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let bind_addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting API server on {}", bind_addr);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
