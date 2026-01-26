// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Main Qwen3 TTS model implementation.
//!
//! This module provides the `Qwen3TTSModel` struct which wraps the TTS
//! model and provides generation APIs for different use cases:
//! - `generate_custom_voice()`: Use predefined speakers
//! - `generate_voice_design()`: Use natural language voice instructions
//! - `generate_voice_clone()`: Clone voice from reference audio

use crate::audio::{load_audio, AudioInput};
use crate::config::{GenerationConfig, Qwen3TTSConfig, TTSModelType, TokenizerType};
use crate::error::{Qwen3TTSError, Result};
use crate::tokenizer::Qwen3TTSTokenizer;
use crate::types::{GenerationOutput, Language, Speaker, VoiceClonePromptItem, VoiceInstruction};
use std::collections::HashSet;
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// Builder for generation parameters.
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    /// Whether to use sampling
    pub do_sample: Option<bool>,
    /// Top-k sampling parameter
    pub top_k: Option<usize>,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f64>,
    /// Sampling temperature
    pub temperature: Option<f64>,
    /// Repetition penalty
    pub repetition_penalty: Option<f64>,
    /// Sub-talker sampling enabled
    pub subtalker_dosample: Option<bool>,
    /// Sub-talker top-k
    pub subtalker_top_k: Option<usize>,
    /// Sub-talker top-p
    pub subtalker_top_p: Option<f64>,
    /// Sub-talker temperature
    pub subtalker_temperature: Option<f64>,
    /// Maximum new tokens to generate
    pub max_new_tokens: Option<usize>,
    /// Whether to use non-streaming mode
    pub non_streaming_mode: Option<bool>,
}

impl GenerationParams {
    /// Create new generation parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sampling mode.
    pub fn do_sample(mut self, do_sample: bool) -> Self {
        self.do_sample = Some(do_sample);
        self
    }

    /// Set top-k sampling parameter.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set top-p (nucleus) sampling parameter.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    /// Set maximum new tokens.
    pub fn max_new_tokens(mut self, max_tokens: usize) -> Self {
        self.max_new_tokens = Some(max_tokens);
        self
    }

    /// Set non-streaming mode.
    pub fn non_streaming_mode(mut self, enabled: bool) -> Self {
        self.non_streaming_mode = Some(enabled);
        self
    }

    /// Merge with default generation config.
    fn merge_with_defaults(&self, defaults: &GenerationConfig) -> GenerationConfig {
        GenerationConfig {
            do_sample: self.do_sample.unwrap_or(defaults.do_sample),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            repetition_penalty: self.repetition_penalty.unwrap_or(defaults.repetition_penalty),
            subtalker_dosample: self.subtalker_dosample.unwrap_or(defaults.subtalker_dosample),
            subtalker_top_k: self.subtalker_top_k.unwrap_or(defaults.subtalker_top_k),
            subtalker_top_p: self.subtalker_top_p.unwrap_or(defaults.subtalker_top_p),
            subtalker_temperature: self
                .subtalker_temperature
                .unwrap_or(defaults.subtalker_temperature),
            max_new_tokens: self.max_new_tokens.unwrap_or(defaults.max_new_tokens),
        }
    }
}

/// Main Qwen3 TTS Model wrapper.
///
/// Provides generation APIs for different TTS use cases:
/// - CustomVoice: `generate_custom_voice()` with predefined speakers
/// - VoiceDesign: `generate_voice_design()` with natural language instructions
/// - Base (VoiceClone): `generate_voice_clone()` with reference audio
///
/// # Example
///
/// ```ignore
/// use qwen3_tts::model::Qwen3TTSModel;
///
/// let model = Qwen3TTSModel::from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")?;
///
/// // Generate speech with a predefined speaker
/// let output = model.generate_custom_voice(
///     "Hello, how are you today?",
///     "Vivian",
///     "EN",
///     None,
///     None,
/// )?;
///
/// // Save to file
/// qwen3_tts::audio::write_wav_file("output.wav", &output.waveforms[0], output.sample_rate)?;
/// ```
pub struct Qwen3TTSModel {
    /// Model configuration
    config: Qwen3TTSConfig,

    /// Default generation configuration
    generate_defaults: GenerationConfig,

    /// Speech tokenizer for encoding/decoding audio
    speech_tokenizer: Option<Qwen3TTSTokenizer>,

    /// Computation device
    device: Device,

    /// TTS model type (base, custom_voice, voice_design)
    tts_model_type: TTSModelType,

    /// Tokenizer type (25Hz or 12Hz)
    tokenizer_type: TokenizerType,

    /// Supported speakers (for custom_voice model)
    supported_speakers: Option<HashSet<String>>,

    /// Supported languages
    supported_languages: Option<HashSet<String>>,

    /// Speaker encoder sample rate
    speaker_encoder_sample_rate: u32,
}

impl Qwen3TTSModel {
    /// Load a model from a pretrained path or HuggingFace repository.
    ///
    /// # Arguments
    ///
    /// * `pretrained_model_name_or_path` - HuggingFace repo ID or local directory path.
    ///
    /// # Returns
    ///
    /// A new `Qwen3TTSModel` instance.
    pub fn from_pretrained(pretrained_model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_device(pretrained_model_name_or_path, Device::Cpu)
    }

    /// Load a model with a specific device.
    pub fn from_pretrained_with_device(
        pretrained_model_name_or_path: &str,
        device: Device,
    ) -> Result<Self> {
        let model_path = if Path::new(pretrained_model_name_or_path).exists() {
            pretrained_model_name_or_path.to_string()
        } else {
            // For non-local paths, return an error suggesting local path usage
            return Err(Qwen3TTSError::ModelLoad(format!(
                "Path '{}' does not exist. Please provide a local model path.",
                pretrained_model_name_or_path
            )));
        };

        // Load main config
        let config_path = Path::new(&model_path).join("config.json");
        let config: Qwen3TTSConfig = if config_path.exists() {
            Qwen3TTSConfig::from_file(&config_path)?
        } else {
            Qwen3TTSConfig::default()
        };

        // Load generation config if available
        let gen_config_path = Path::new(&model_path).join("generate_config.json");
        let generate_defaults = if gen_config_path.exists() {
            GenerationConfig::from_file(&gen_config_path)?
        } else {
            GenerationConfig::default()
        };

        // Extract model type info
        let tts_model_type = config.get_tts_model_type();
        let tokenizer_type = config.get_tokenizer_type();

        // Extract supported speakers from config
        let supported_speakers = config
            .talker_config
            .spk_id
            .as_ref()
            .map(|spk_map| spk_map.keys().map(|k| k.to_lowercase()).collect());

        // Extract supported languages from config
        let supported_languages = config
            .talker_config
            .codec_language_id
            .as_ref()
            .map(|lang_map| lang_map.keys().map(|k| k.to_lowercase()).collect());

        Ok(Self {
            config,
            generate_defaults,
            speech_tokenizer: None, // Would be loaded separately
            device,
            tts_model_type,
            tokenizer_type,
            supported_speakers,
            supported_languages,
            speaker_encoder_sample_rate: 24000,
        })
    }

    /// Get the TTS model type.
    pub fn get_tts_model_type(&self) -> TTSModelType {
        self.tts_model_type
    }

    /// Get the tokenizer type.
    pub fn get_tokenizer_type(&self) -> TokenizerType {
        self.tokenizer_type
    }

    /// Get supported speakers (for custom_voice model).
    pub fn get_supported_speakers(&self) -> Option<Vec<String>> {
        self.supported_speakers.as_ref().map(|s| {
            let mut speakers: Vec<_> = s.iter().cloned().collect();
            speakers.sort();
            speakers
        })
    }

    /// Get supported languages.
    pub fn get_supported_languages(&self) -> Option<Vec<String>> {
        self.supported_languages.as_ref().map(|s| {
            let mut languages: Vec<_> = s.iter().cloned().collect();
            languages.sort();
            languages
        })
    }

    /// Validate languages are supported.
    fn validate_languages(&self, languages: &[Language]) -> Result<()> {
        if let Some(ref supported) = self.supported_languages {
            for lang in languages {
                let lang_str = lang.as_str().to_lowercase();
                if lang_str != "auto" && !supported.contains(&lang_str) {
                    return Err(Qwen3TTSError::InvalidInput(format!(
                        "Unsupported language: {}. Supported: {:?}",
                        lang.as_str(),
                        self.get_supported_languages()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate speakers are supported.
    fn validate_speakers(&self, speakers: &[Speaker]) -> Result<()> {
        if let Some(ref supported) = self.supported_speakers {
            for speaker in speakers {
                let spk_str = speaker.name().to_lowercase();
                if !spk_str.is_empty() && !supported.contains(&spk_str) {
                    return Err(Qwen3TTSError::InvalidInput(format!(
                        "Unsupported speaker: {}. Supported: {:?}",
                        speaker.name(),
                        self.get_supported_speakers()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Build assistant text format for the model.
    fn build_assistant_text(&self, text: &str) -> String {
        format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            text
        )
    }

    /// Build reference text format for the model.
    fn build_ref_text(&self, text: &str) -> String {
        format!("<|im_start|>assistant\n{}<|im_end|>\n", text)
    }

    /// Build instruction text format for the model.
    fn build_instruct_text(&self, instruct: &str) -> String {
        format!("<|im_start|>user\n{}<|im_end|>\n", instruct)
    }

    /// Generate speech using the CustomVoice model with a predefined speaker.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize (single or batch)
    /// * `speaker` - Speaker name (must be in supported speakers list)
    /// * `language` - Language for synthesis
    /// * `instruct` - Optional instruction for voice control (not supported for 0b6 model)
    /// * `params` - Optional generation parameters
    ///
    /// # Returns
    ///
    /// `GenerationOutput` containing waveforms and sample rate.
    pub fn generate_custom_voice<T, S, L>(
        &self,
        text: T,
        speaker: S,
        language: L,
        instruct: Option<&str>,
        params: Option<GenerationParams>,
    ) -> Result<GenerationOutput>
    where
        T: AsRef<str>,
        S: Into<Speaker>,
        L: Into<Language>,
    {
        // Validate model type
        if self.tts_model_type != TTSModelType::CustomVoice {
            return Err(Qwen3TTSError::Unsupported(format!(
                "Model type {:?} does not support generate_custom_voice",
                self.tts_model_type
            )));
        }

        let texts = vec![text.as_ref().to_string()];
        let speakers = vec![speaker.into()];
        let languages = vec![language.into()];

        self.validate_languages(&languages)?;
        self.validate_speakers(&speakers)?;

        // Merge generation params
        let _gen_config = params
            .unwrap_or_default()
            .merge_with_defaults(&self.generate_defaults);

        // Build input text
        let _input_text = self.build_assistant_text(&texts[0]);

        // Build instruct text if provided
        let _instruct_text = instruct.map(|i| self.build_instruct_text(i));

        // Placeholder: Actual generation would happen here
        // For now, return a silent waveform
        let sample_rate = 24000;
        let duration_samples = sample_rate * 2; // 2 seconds of silence
        let waveform = vec![0.0f32; duration_samples as usize];

        tracing::info!(
            "Generated custom voice: text='{}', speaker='{}', language='{}'",
            texts[0],
            speakers[0].name(),
            languages[0].as_str()
        );

        Ok(GenerationOutput {
            waveforms: vec![waveform],
            sample_rate,
        })
    }

    /// Generate speech using the VoiceDesign model with natural language instructions.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language instruction describing desired voice
    /// * `language` - Language for synthesis
    /// * `params` - Optional generation parameters
    ///
    /// # Returns
    ///
    /// `GenerationOutput` containing waveforms and sample rate.
    pub fn generate_voice_design<T, I, L>(
        &self,
        text: T,
        instruct: I,
        language: L,
        params: Option<GenerationParams>,
    ) -> Result<GenerationOutput>
    where
        T: AsRef<str>,
        I: Into<VoiceInstruction>,
        L: Into<Language>,
    {
        // Validate model type
        if self.tts_model_type != TTSModelType::VoiceDesign {
            return Err(Qwen3TTSError::Unsupported(format!(
                "Model type {:?} does not support generate_voice_design",
                self.tts_model_type
            )));
        }

        let texts = vec![text.as_ref().to_string()];
        let instructs = vec![instruct.into()];
        let languages = vec![language.into()];

        self.validate_languages(&languages)?;

        // Merge generation params
        let _gen_config = params
            .unwrap_or_default()
            .merge_with_defaults(&self.generate_defaults);

        // Build input and instruction text
        let _input_text = self.build_assistant_text(&texts[0]);
        let _instruct_text = self.build_instruct_text(instructs[0].text());

        // Placeholder: Actual generation would happen here
        let sample_rate = 24000;
        let duration_samples = sample_rate * 2;
        let waveform = vec![0.0f32; duration_samples as usize];

        tracing::info!(
            "Generated voice design: text='{}', instruct='{}', language='{}'",
            texts[0],
            instructs[0].text(),
            languages[0].as_str()
        );

        Ok(GenerationOutput {
            waveforms: vec![waveform],
            sample_rate,
        })
    }

    /// Create a voice clone prompt from reference audio.
    ///
    /// # Arguments
    ///
    /// * `ref_audio` - Reference audio for voice cloning
    /// * `ref_text` - Transcript of reference audio (required for ICL mode)
    /// * `x_vector_only_mode` - If true, only use speaker embedding
    ///
    /// # Returns
    ///
    /// `VoiceClonePromptItem` that can be used with `generate_voice_clone`.
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio: AudioInput,
        ref_text: Option<&str>,
        x_vector_only_mode: bool,
    ) -> Result<VoiceClonePromptItem> {
        // Validate model type
        if self.tts_model_type != TTSModelType::Base {
            return Err(Qwen3TTSError::Unsupported(format!(
                "Model type {:?} does not support create_voice_clone_prompt",
                self.tts_model_type
            )));
        }

        // Validate ref_text requirement for ICL mode
        if !x_vector_only_mode && ref_text.is_none() {
            return Err(Qwen3TTSError::InvalidInput(
                "ref_text is required when x_vector_only_mode=false (ICL mode)".to_string(),
            ));
        }

        // Load and resample audio
        let samples = load_audio(ref_audio, self.speaker_encoder_sample_rate)?;

        // Placeholder: Extract speaker embedding
        // In a real implementation, this would use the speaker encoder model
        let spk_embedding = Tensor::zeros([1024], (Kind::Float, self.device));

        // Placeholder: Encode reference audio to codes
        // In a real implementation, this would use the speech tokenizer
        let ref_code = if !x_vector_only_mode {
            let code_len = samples.len() / 960; // Approximate based on downsample rate
            Some(Tensor::zeros([code_len as i64], (Kind::Int64, self.device)))
        } else {
            None
        };

        Ok(VoiceClonePromptItem {
            ref_code,
            ref_spk_embedding: spk_embedding,
            x_vector_only_mode,
            icl_mode: !x_vector_only_mode,
            ref_text: ref_text.map(|s| s.to_string()),
        })
    }

    /// Generate speech using voice cloning from reference audio.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `language` - Language for synthesis
    /// * `ref_audio` - Reference audio for voice cloning
    /// * `ref_text` - Transcript of reference audio (required for ICL mode)
    /// * `x_vector_only_mode` - If true, only use speaker embedding
    /// * `params` - Optional generation parameters
    ///
    /// # Returns
    ///
    /// `GenerationOutput` containing waveforms and sample rate.
    pub fn generate_voice_clone<T, L>(
        &self,
        text: T,
        language: L,
        ref_audio: AudioInput,
        ref_text: Option<&str>,
        x_vector_only_mode: bool,
        params: Option<GenerationParams>,
    ) -> Result<GenerationOutput>
    where
        T: AsRef<str>,
        L: Into<Language>,
    {
        // Validate model type
        if self.tts_model_type != TTSModelType::Base {
            return Err(Qwen3TTSError::Unsupported(format!(
                "Model type {:?} does not support generate_voice_clone",
                self.tts_model_type
            )));
        }

        let texts = vec![text.as_ref().to_string()];
        let languages = vec![language.into()];

        self.validate_languages(&languages)?;

        // Create voice clone prompt
        let prompt = self.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode)?;

        // Merge generation params
        let _gen_config = params
            .unwrap_or_default()
            .merge_with_defaults(&self.generate_defaults);

        // Build input text
        let _input_text = self.build_assistant_text(&texts[0]);

        // Build ref text if in ICL mode
        let _ref_text_formatted = prompt.ref_text.as_ref().map(|t| self.build_ref_text(t));

        // Placeholder: Actual generation would happen here
        let sample_rate = 24000;
        let duration_samples = sample_rate * 2;
        let waveform = vec![0.0f32; duration_samples as usize];

        tracing::info!(
            "Generated voice clone: text='{}', language='{}', icl_mode={}",
            texts[0],
            languages[0].as_str(),
            prompt.icl_mode
        );

        Ok(GenerationOutput {
            waveforms: vec![waveform],
            sample_rate,
        })
    }

    /// Generate speech using a pre-created voice clone prompt.
    ///
    /// This is useful when you want to reuse the same voice clone prompt
    /// for multiple generations.
    pub fn generate_voice_clone_with_prompt<T, L>(
        &self,
        text: T,
        language: L,
        prompt: &VoiceClonePromptItem,
        params: Option<GenerationParams>,
    ) -> Result<GenerationOutput>
    where
        T: AsRef<str>,
        L: Into<Language>,
    {
        // Validate model type
        if self.tts_model_type != TTSModelType::Base {
            return Err(Qwen3TTSError::Unsupported(format!(
                "Model type {:?} does not support generate_voice_clone",
                self.tts_model_type
            )));
        }

        let texts = vec![text.as_ref().to_string()];
        let languages = vec![language.into()];

        self.validate_languages(&languages)?;

        // Merge generation params
        let _gen_config = params
            .unwrap_or_default()
            .merge_with_defaults(&self.generate_defaults);

        // Build input text
        let _input_text = self.build_assistant_text(&texts[0]);

        // Placeholder: Actual generation would happen here
        let sample_rate = 24000;
        let duration_samples = sample_rate * 2;
        let waveform = vec![0.0f32; duration_samples as usize];

        Ok(GenerationOutput {
            waveforms: vec![waveform],
            sample_rate,
        })
    }
}

impl std::fmt::Debug for Qwen3TTSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TTSModel")
            .field("tts_model_type", &self.tts_model_type)
            .field("tokenizer_type", &self.tokenizer_type)
            .field("device", &self.device)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_params_builder() {
        let params = GenerationParams::new()
            .do_sample(true)
            .top_k(50)
            .top_p(0.9)
            .temperature(0.8)
            .max_new_tokens(1024);

        assert_eq!(params.do_sample, Some(true));
        assert_eq!(params.top_k, Some(50));
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.temperature, Some(0.8));
        assert_eq!(params.max_new_tokens, Some(1024));
    }

    #[test]
    fn test_params_merge_with_defaults() {
        let defaults = GenerationConfig::default();
        let params = GenerationParams::new().temperature(0.5);

        let merged = params.merge_with_defaults(&defaults);

        assert_eq!(merged.temperature, 0.5);
        assert_eq!(merged.top_k, defaults.top_k);
        assert_eq!(merged.top_p, defaults.top_p);
    }

    #[test]
    fn test_text_formatting() {
        let config = Qwen3TTSConfig::default();
        let model = Qwen3TTSModel {
            config,
            generate_defaults: GenerationConfig::default(),
            speech_tokenizer: None,
            device: Device::Cpu,
            tts_model_type: TTSModelType::Base,
            tokenizer_type: TokenizerType::V2_12Hz,
            supported_speakers: None,
            supported_languages: None,
            speaker_encoder_sample_rate: 24000,
        };

        let formatted = model.build_assistant_text("Hello");
        assert!(formatted.contains("<|im_start|>assistant"));
        assert!(formatted.contains("Hello"));
        assert!(formatted.contains("<|im_end|>"));
    }
}
