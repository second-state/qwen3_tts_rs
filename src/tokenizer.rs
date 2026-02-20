// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Speech tokenizer for Qwen3 TTS.
//!
//! This module provides the `Qwen3TTSTokenizer` struct which wraps the
//! speech tokenizer models (V1 25Hz or V2 12Hz) for encoding audio to
//! discrete codes and decoding codes back to waveforms.

use crate::audio::{load_audio, AudioInput};
use crate::config::TokenizerType;
use crate::error::{Qwen3TTSError, Result};
use crate::types::{DecodedAudio, EncodedAudio};
use serde::Deserialize;
use std::path::Path;
use crate::tensor::{DType, Device, Tensor};

/// Configuration for the speech tokenizer.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    /// Model type identifier
    pub model_type: String,

    /// Input sample rate for encoding
    #[serde(default = "default_input_sr")]
    pub input_sample_rate: u32,

    /// Output sample rate for decoding
    #[serde(default = "default_output_sr")]
    pub output_sample_rate: u32,

    /// Encoder downsample rate (samples per code)
    #[serde(default = "default_downsample_rate")]
    pub encode_downsample_rate: u32,

    /// Decoder upsample rate (samples per code)
    #[serde(default = "default_upsample_rate")]
    pub decode_upsample_rate: u32,

    /// Codebook size for vector quantization
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,

    /// Number of quantizer groups
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: usize,
}

fn default_input_sr() -> u32 {
    24000
}
fn default_output_sr() -> u32 {
    24000
}
fn default_downsample_rate() -> u32 {
    960
} // 25Hz at 24kHz
fn default_upsample_rate() -> u32 {
    960
}
fn default_codebook_size() -> usize {
    32768
}
fn default_num_quantizers() -> usize {
    6
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            model_type: "qwen3_tts_tokenizer_12hz".to_string(),
            input_sample_rate: default_input_sr(),
            output_sample_rate: default_output_sr(),
            encode_downsample_rate: default_downsample_rate(),
            decode_upsample_rate: default_upsample_rate(),
            codebook_size: default_codebook_size(),
            num_quantizers: default_num_quantizers(),
        }
    }
}

/// Qwen3 TTS Speech Tokenizer.
///
/// This struct provides methods for encoding audio waveforms to discrete
/// codes and decoding codes back to waveforms.
///
/// # Example
///
/// ```ignore
/// use qwen3_tts::tokenizer::Qwen3TTSTokenizer;
///
/// let tokenizer = Qwen3TTSTokenizer::from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")?;
///
/// // Encode audio to codes
/// let encoded = tokenizer.encode("audio.wav")?;
///
/// // Decode codes back to audio
/// let decoded = tokenizer.decode(&encoded)?;
/// ```
pub struct Qwen3TTSTokenizer {
    /// Model configuration
    config: TokenizerConfig,

    /// Tokenizer type (25Hz or 12Hz)
    tokenizer_type: TokenizerType,

    /// Computation device (CPU/CUDA)
    device: Device,

    /// Model path for weight loading
    #[allow(dead_code)]
    model_path: Option<String>,
}

impl Qwen3TTSTokenizer {
    /// Load a tokenizer from a pretrained model path or HuggingFace repository.
    ///
    /// # Arguments
    ///
    /// * `pretrained_model_name_or_path` - HuggingFace repo ID or local directory path.
    ///
    /// # Returns
    ///
    /// A new `Qwen3TTSTokenizer` instance.
    pub fn from_pretrained(pretrained_model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_device(pretrained_model_name_or_path, Device::Cpu)
    }

    /// Load a tokenizer from a pretrained model with a specific device.
    pub fn from_pretrained_with_device(
        pretrained_model_name_or_path: &str,
        device: Device,
    ) -> Result<Self> {
        let model_path = if Path::new(pretrained_model_name_or_path).exists() {
            Some(pretrained_model_name_or_path.to_string())
        } else {
            // For non-local paths, we would need to download from HuggingFace
            // For now, return an error suggesting local path usage
            return Err(Qwen3TTSError::ModelLoad(format!(
                "Path '{}' does not exist. Please provide a local model path.",
                pretrained_model_name_or_path
            )));
        };

        // Load configuration
        let config_path = Path::new(pretrained_model_name_or_path).join("config.json");
        let config: TokenizerConfig = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)?
        } else {
            TokenizerConfig::default()
        };

        // Determine tokenizer type
        let tokenizer_type = if config.model_type.contains("25hz") {
            TokenizerType::V1_25Hz
        } else {
            TokenizerType::V2_12Hz
        };

        Ok(Self {
            config,
            tokenizer_type,
            device,
            model_path,
        })
    }

    /// Get the model type string.
    pub fn get_model_type(&self) -> &str {
        &self.config.model_type
    }

    /// Get the tokenizer type (25Hz or 12Hz).
    pub fn get_tokenizer_type(&self) -> TokenizerType {
        self.tokenizer_type
    }

    /// Get the expected input sample rate for encoding.
    pub fn get_input_sample_rate(&self) -> u32 {
        self.config.input_sample_rate
    }

    /// Get the output sample rate for decoded waveforms.
    pub fn get_output_sample_rate(&self) -> u32 {
        self.config.output_sample_rate
    }

    /// Get the encoder downsample rate (waveform samples per code step).
    pub fn get_encode_downsample_rate(&self) -> u32 {
        self.config.encode_downsample_rate
    }

    /// Get the decoder upsample rate (waveform samples per code step).
    pub fn get_decode_upsample_rate(&self) -> u32 {
        self.config.decode_upsample_rate
    }

    /// Load and preprocess audio for encoding.
    fn _prepare_audio(&self, audio: AudioInput) -> Result<Tensor> {
        let samples = load_audio(audio, self.config.input_sample_rate)?;
        let tensor = Tensor::from_slice_f32(&samples)
            .view(&[1, samples.len() as i64])
            .to_device(self.device);
        Ok(tensor)
    }

    /// Encode audio to discrete codes.
    ///
    /// # Arguments
    ///
    /// * `audios` - Audio input(s) to encode. Can be file paths, base64 strings,
    ///   URLs, or raw waveforms.
    /// * `sr` - Original sample rate (only required for raw waveform input).
    ///
    /// # Returns
    ///
    /// `EncodedAudio` containing:
    /// - `audio_codes`: Discrete codes from vector quantization
    /// - `xvectors`: Speaker embeddings (V1 only)
    /// - `ref_mels`: Reference mel spectrograms (V1 only)
    pub fn encode<I>(&self, audios: I, sr: Option<u32>) -> Result<EncodedAudio>
    where
        I: IntoIterator<Item = AudioInput>,
    {
        let audio_inputs: Vec<AudioInput> = audios.into_iter().collect();

        if audio_inputs.is_empty() {
            return Ok(EncodedAudio {
                audio_codes: vec![],
                xvectors: None,
                ref_mels: None,
            });
        }

        // Convert audio inputs with optional sample rate override
        let mut waveforms = Vec::with_capacity(audio_inputs.len());
        for input in audio_inputs {
            let input = match (input, sr) {
                (AudioInput::Waveform { samples, .. }, Some(override_sr)) => AudioInput::Waveform {
                    samples,
                    sample_rate: override_sr,
                },
                (other, _) => other,
            };
            let samples = load_audio(input, self.config.input_sample_rate)?;
            waveforms.push(samples);
        }

        // Placeholder: Actual encoding would happen here using the model
        // For now, we create dummy output to demonstrate the API
        let mut audio_codes = Vec::with_capacity(waveforms.len());
        let mut xvectors = if self.tokenizer_type == TokenizerType::V1_25Hz {
            Some(Vec::with_capacity(waveforms.len()))
        } else {
            None
        };
        let mut ref_mels = if self.tokenizer_type == TokenizerType::V1_25Hz {
            Some(Vec::with_capacity(waveforms.len()))
        } else {
            None
        };

        for waveform in &waveforms {
            // Calculate expected code length based on downsample rate
            let code_len = waveform.len() / self.config.encode_downsample_rate as usize;

            match self.tokenizer_type {
                TokenizerType::V1_25Hz => {
                    // V1: codes shape (T,), xvector shape (D,), ref_mel shape (T, M)
                    let codes = Tensor::zeros(&[code_len as i64], DType::Int64, self.device);
                    audio_codes.push(codes);

                    if let Some(ref mut xvecs) = xvectors {
                        let xvec = Tensor::zeros(&[1024], DType::Float32, self.device);
                        xvecs.push(xvec);
                    }

                    if let Some(ref mut mels) = ref_mels {
                        let mel = Tensor::zeros(&[code_len as i64, 80], DType::Float32, self.device);
                        mels.push(mel);
                    }
                }
                TokenizerType::V2_12Hz => {
                    // V2: codes shape (T, Q) where Q is num_quantizers
                    let codes = Tensor::zeros(
                        &[code_len as i64, self.config.num_quantizers as i64],
                        DType::Int64,
                        self.device,
                    );
                    audio_codes.push(codes);
                }
            }
        }

        Ok(EncodedAudio {
            audio_codes,
            xvectors,
            ref_mels,
        })
    }

    /// Encode a single audio file.
    pub fn encode_file(&self, path: &str) -> Result<EncodedAudio> {
        self.encode(
            std::iter::once(AudioInput::FilePath(path.to_string())),
            None,
        )
    }

    /// Decode audio codes back to waveforms.
    ///
    /// # Arguments
    ///
    /// * `encoded` - Encoded audio from `encode()`.
    ///
    /// # Returns
    ///
    /// `DecodedAudio` containing waveforms and sample rate.
    pub fn decode(&self, encoded: &EncodedAudio) -> Result<DecodedAudio> {
        if encoded.audio_codes.is_empty() {
            return Ok(DecodedAudio {
                waveforms: vec![],
                sample_rate: self.config.output_sample_rate,
            });
        }

        // Validate inputs for V1 tokenizer
        if self.tokenizer_type == TokenizerType::V1_25Hz {
            if encoded.xvectors.is_none() {
                return Err(Qwen3TTSError::InvalidInput(
                    "25Hz decode requires xvectors".to_string(),
                ));
            }
            if encoded.ref_mels.is_none() {
                return Err(Qwen3TTSError::InvalidInput(
                    "25Hz decode requires ref_mels".to_string(),
                ));
            }
        }

        // Placeholder: Actual decoding would happen here using the model
        // For now, we create dummy waveforms to demonstrate the API
        let mut waveforms = Vec::with_capacity(encoded.audio_codes.len());

        for codes in &encoded.audio_codes {
            // Calculate expected waveform length based on upsample rate
            let code_len = codes.size()[0] as usize;
            let wav_len = code_len * self.config.decode_upsample_rate as usize;

            // Generate silence placeholder
            let waveform = vec![0.0f32; wav_len];
            waveforms.push(waveform);
        }

        Ok(DecodedAudio {
            waveforms,
            sample_rate: self.config.output_sample_rate,
        })
    }

    /// Decode a single set of audio codes.
    pub fn decode_codes(&self, codes: &Tensor) -> Result<DecodedAudio> {
        let encoded = EncodedAudio {
            audio_codes: vec![codes.shallow_clone()],
            xvectors: None,
            ref_mels: None,
        };

        if self.tokenizer_type == TokenizerType::V1_25Hz {
            return Err(Qwen3TTSError::InvalidInput(
                "V1 tokenizer requires xvectors and ref_mels for decoding".to_string(),
            ));
        }

        self.decode(&encoded)
    }
}

impl std::fmt::Debug for Qwen3TTSTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TTSTokenizer")
            .field("tokenizer_type", &self.tokenizer_type)
            .field("input_sample_rate", &self.config.input_sample_rate)
            .field("output_sample_rate", &self.config.output_sample_rate)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_default() {
        let config = TokenizerConfig::default();
        assert_eq!(config.input_sample_rate, 24000);
        assert_eq!(config.output_sample_rate, 24000);
    }

    #[test]
    fn test_tokenizer_type_detection() {
        let mut config = TokenizerConfig::default();
        config.model_type = "qwen3_tts_tokenizer_25hz".to_string();

        let tokenizer_type = if config.model_type.contains("25hz") {
            TokenizerType::V1_25Hz
        } else {
            TokenizerType::V2_12Hz
        };

        assert_eq!(tokenizer_type, TokenizerType::V1_25Hz);
    }
}
