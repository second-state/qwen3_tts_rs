// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! # Qwen3 TTS - Rust Port
//!
//! A Rust implementation of the Qwen3 Text-to-Speech (TTS) model.
//!
//! This crate provides a high-level API for generating speech from text using
//! the Qwen3 TTS model family, which includes:
//!
//! - **CustomVoice**: Use predefined speaker voices
//! - **VoiceDesign**: Control voice characteristics with natural language
//! - **Base (VoiceClone)**: Clone voices from reference audio
//!
//! ## Quick Start
//!
//! ```ignore
//! use qwen3_tts::{Qwen3TTSModel, Language, Speaker};
//!
//! // Load a pretrained model
//! let model = Qwen3TTSModel::from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")?;
//!
//! // Generate speech with a predefined speaker
//! let output = model.generate_custom_voice(
//!     "Hello, welcome to Qwen TTS!",
//!     Speaker::new("Vivian"),
//!     Language::English,
//!     None,
//!     None,
//! )?;
//!
//! // Save the output
//! qwen3_tts::audio::write_wav_file("output.wav", &output.waveforms[0], output.sample_rate)?;
//! ```
//!
//! ## Voice Cloning
//!
//! ```ignore
//! use qwen3_tts::{Qwen3TTSModel, AudioInput, Language};
//!
//! let model = Qwen3TTSModel::from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")?;
//!
//! // Clone voice from reference audio
//! let output = model.generate_voice_clone(
//!     "This is synthesized in the cloned voice.",
//!     Language::English,
//!     AudioInput::from("reference.wav"),
//!     Some("This is the reference transcript."),
//!     false, // Use ICL mode
//!     None,
//! )?;
//! ```
//!
//! ## Voice Design
//!
//! ```ignore
//! use qwen3_tts::{Qwen3TTSModel, Language, VoiceInstruction};
//!
//! let model = Qwen3TTSModel::from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")?;
//!
//! // Design voice with natural language
//! let output = model.generate_voice_design(
//!     "Welcome to our service!",
//!     VoiceInstruction::new("A warm, friendly female voice with moderate speed"),
//!     Language::English,
//!     None,
//! )?;
//! ```
//!
//! ## Features
//!
//! - `async`: Enable async support with tokio and reqwest for URL fetching

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

pub mod audio;
pub mod audio_encoder;
pub mod config;
pub mod error;
pub mod inference;
pub mod layers;
pub mod model;
pub mod speaker_encoder;
pub mod tokenizer;
pub mod types;
pub mod vocoder;
pub mod weights;

// Re-export main types at crate root for convenience
pub use audio::AudioInput;
pub use config::{GenerationConfig, Qwen3TTSConfig, TTSModelType, TokenizerType};
pub use error::{Qwen3TTSError, Result};
pub use model::{GenerationParams, Qwen3TTSModel};
pub use tokenizer::Qwen3TTSTokenizer;
pub use types::{
    DecodedAudio, EncodedAudio, GenerationOutput, Language, Speaker, VoiceClonePromptItem,
    VoiceInstruction,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default output sample rate in Hz.
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_default_sample_rate() {
        assert_eq!(DEFAULT_SAMPLE_RATE, 24000);
    }

    #[test]
    fn test_language_conversion() {
        // Test that "english" and "EN" both convert to Language::English
        let lang: Language = "english".into();
        assert_eq!(lang.as_str(), "english");

        let lang: Language = "EN".into();
        assert_eq!(lang.as_str(), "english");

        let lang: Language = "Auto".into();
        assert_eq!(lang.as_str(), "auto");

        let lang: Language = "zh".into();
        assert_eq!(lang.as_str(), "chinese");

        let lang = Language::Auto;
        assert_eq!(lang.as_str(), "auto");
    }

    #[test]
    fn test_speaker_creation() {
        let speaker = Speaker::new("Vivian");
        assert_eq!(speaker.name(), "Vivian");

        let speaker: Speaker = "John".into();
        assert_eq!(speaker.name(), "John");
    }

    #[test]
    fn test_audio_input_from_string() {
        let input = AudioInput::from("test.wav");
        match input {
            AudioInput::FilePath(path) => assert_eq!(path, "test.wav"),
            _ => panic!("Expected FilePath"),
        }
    }
}
