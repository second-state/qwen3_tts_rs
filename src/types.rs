// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Common types and data structures for Qwen3 TTS.

use serde::{Deserialize, Serialize};
use crate::tensor::Tensor;

/// Voice clone prompt item containing reference audio information.
///
/// This struct holds all the information needed to clone a voice from
/// a reference audio sample, including the encoded audio codes and
/// speaker embeddings.
#[derive(Debug)]
pub struct VoiceClonePromptItem {
    /// Reference audio codes from the speech tokenizer.
    /// Shape depends on tokenizer type:
    /// - V1 (25Hz): (T,) - 1D tensor
    /// - V2 (12Hz): (T, Q) - 2D tensor where Q is num_quantizers
    pub ref_code: Option<Tensor>,

    /// Speaker embedding (x-vector) extracted from reference audio.
    /// Shape: (D,) where D is the embedding dimension (typically 1024)
    pub ref_spk_embedding: Tensor,

    /// Whether to use only the x-vector for voice cloning.
    /// If true, ref_code is ignored and only speaker embedding is used.
    /// Mutually exclusive with ICL mode.
    pub x_vector_only_mode: bool,

    /// Whether to use In-Context Learning mode.
    /// If true, the model conditions on both ref_code and ref_text.
    /// Requires ref_text to be provided.
    pub icl_mode: bool,

    /// Reference transcript text (required when icl_mode is true).
    pub ref_text: Option<String>,
}

impl VoiceClonePromptItem {
    /// Create a new voice clone prompt item for x-vector only mode.
    pub fn x_vector_only(ref_spk_embedding: Tensor) -> Self {
        Self {
            ref_code: None,
            ref_spk_embedding,
            x_vector_only_mode: true,
            icl_mode: false,
            ref_text: None,
        }
    }

    /// Create a new voice clone prompt item for ICL mode.
    pub fn icl(ref_code: Tensor, ref_spk_embedding: Tensor, ref_text: String) -> Self {
        Self {
            ref_code: Some(ref_code),
            ref_spk_embedding,
            x_vector_only_mode: false,
            icl_mode: true,
            ref_text: Some(ref_text),
        }
    }
}

/// Encoded audio output from the speech tokenizer.
#[derive(Debug)]
pub struct EncodedAudio {
    /// Audio codes from vector quantization.
    /// Shape depends on tokenizer type:
    /// - V1 (25Hz): Vec of (T,) tensors
    /// - V2 (12Hz): Vec of (T, Q) tensors
    pub audio_codes: Vec<Tensor>,

    /// Speaker embeddings (x-vectors).
    /// Only present for V1 (25Hz) tokenizer.
    /// Shape: Vec of (D,) tensors
    pub xvectors: Option<Vec<Tensor>>,

    /// Reference mel spectrograms.
    /// Only present for V1 (25Hz) tokenizer.
    /// Shape: Vec of (T, M) tensors where M is mel_dim
    pub ref_mels: Option<Vec<Tensor>>,
}

/// Decoded audio output from the speech tokenizer.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Decoded waveforms as float32 samples.
    pub waveforms: Vec<Vec<f32>>,

    /// Output sample rate in Hz.
    pub sample_rate: u32,
}

/// Generation output from the TTS model.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated waveforms as float32 samples.
    pub waveforms: Vec<Vec<f32>>,

    /// Output sample rate in Hz.
    pub sample_rate: u32,
}

impl GenerationOutput {
    /// Get the first waveform (convenience method for single-item generation).
    pub fn waveform(&self) -> Option<&[f32]> {
        self.waveforms.first().map(|v| v.as_slice())
    }

    /// Get number of generated waveforms.
    pub fn len(&self) -> usize {
        self.waveforms.len()
    }

    /// Check if output is empty.
    pub fn is_empty(&self) -> bool {
        self.waveforms.is_empty()
    }
}

/// Language specification for TTS generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    /// Automatic language detection
    #[default]
    Auto,
    /// English
    English,
    /// Chinese (Mandarin)
    Chinese,
    /// Japanese
    Japanese,
    /// Korean
    Korean,
    /// Custom language code
    Custom(String),
}

impl Language {
    /// Convert to language code string.
    pub fn as_str(&self) -> &str {
        match self {
            Language::Auto => "auto",
            Language::English => "english",
            Language::Chinese => "chinese",
            Language::Japanese => "japanese",
            Language::Korean => "korean",
            Language::Custom(s) => s.as_str(),
        }
    }
}

impl From<&str> for Language {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "auto" => Language::Auto,
            "en" | "english" => Language::English,
            "zh" | "chinese" | "mandarin" => Language::Chinese,
            "ja" | "japanese" => Language::Japanese,
            "ko" | "korean" => Language::Korean,
            _ => Language::Custom(s.to_string()),
        }
    }
}

impl From<String> for Language {
    fn from(s: String) -> Self {
        Language::from(s.as_str())
    }
}

/// Speaker specification for CustomVoice model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Speaker(pub String);

impl Speaker {
    /// Create a new speaker with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the speaker name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Speaker {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for Speaker {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Voice design instruction for natural language voice control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInstruction(pub String);

impl VoiceInstruction {
    /// Create a new voice instruction.
    pub fn new(instruction: impl Into<String>) -> Self {
        Self(instruction.into())
    }

    /// Get the instruction text.
    pub fn text(&self) -> &str {
        &self.0
    }
}

impl From<&str> for VoiceInstruction {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for VoiceInstruction {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_conversion() {
        assert_eq!(Language::from("english").as_str(), "english");
        assert_eq!(Language::from("EN").as_str(), "english");
        assert_eq!(Language::from("Auto").as_str(), "auto");
        assert_eq!(Language::from("zh").as_str(), "chinese");
    }

    #[test]
    fn test_speaker() {
        let speaker = Speaker::new("Vivian");
        assert_eq!(speaker.name(), "Vivian");
    }
}
