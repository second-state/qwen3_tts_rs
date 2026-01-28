// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Configuration structs for Qwen3 TTS models.
//!
//! This module contains configuration types that mirror the Python
//! `configuration_qwen3_tts.py` structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the ECAPA-TDNN speaker encoder.
///
/// The speaker encoder extracts speaker embeddings (x-vectors) from audio
/// for voice cloning and speaker identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEncoderConfig {
    /// Dimension of the input mel-spectrogram (default: 128)
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,

    /// Dimension of the final speaker embedding (default: 1024)
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,

    /// Output channels for each TDNN/SERes2Net layer
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,

    /// Kernel sizes for each layer
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,

    /// Dilations for each layer
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,

    /// Number of attention channels in AttentiveStatisticsPooling
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,

    /// Scale of the Res2NetBlock
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,

    /// Number of channels in squeeze part of SqueezeExcitationBlock
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,

    /// Sample rate for speaker encoder input (default: 24000)
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
}

fn default_mel_dim() -> usize {
    128
}
fn default_enc_dim() -> usize {
    1024
}
fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_enc_attention_channels() -> usize {
    128
}
fn default_enc_res2net_scale() -> usize {
    8
}
fn default_enc_se_channels() -> usize {
    128
}
fn default_sample_rate() -> u32 {
    24000
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            mel_dim: default_mel_dim(),
            enc_dim: default_enc_dim(),
            enc_channels: default_enc_channels(),
            enc_kernel_sizes: default_enc_kernel_sizes(),
            enc_dilations: default_enc_dilations(),
            enc_attention_channels: default_enc_attention_channels(),
            enc_res2net_scale: default_enc_res2net_scale(),
            enc_se_channels: default_enc_se_channels(),
            sample_rate: default_sample_rate(),
        }
    }
}

/// RoPE (Rotary Position Embedding) scaling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    /// Type of RoPE scaling: "default", "linear", "dynamic", "yarn", "longrope", "llama3"
    pub rope_type: Option<String>,

    /// Type field (alias for rope_type, some models use this name)
    #[serde(rename = "type", default)]
    pub type_field: Option<String>,

    /// Whether to interleave rotary embeddings
    #[serde(default)]
    pub interleaved: Option<bool>,

    /// M-RoPE section configuration
    #[serde(default)]
    pub mrope_section: Option<Vec<i64>>,

    /// Scaling factor for RoPE embeddings
    pub factor: Option<f64>,

    /// Original max position embeddings used during pretraining
    pub original_max_position_embeddings: Option<usize>,

    /// Attention factor for yarn and longrope
    pub attention_factor: Option<f64>,

    /// Beta fast for yarn
    pub beta_fast: Option<f64>,

    /// Beta slow for yarn
    pub beta_slow: Option<f64>,

    /// Short factor for longrope
    pub short_factor: Option<Vec<f64>>,

    /// Long factor for longrope
    pub long_factor: Option<Vec<f64>>,

    /// Low frequency factor for llama3
    pub low_freq_factor: Option<f64>,

    /// High frequency factor for llama3
    pub high_freq_factor: Option<f64>,
}

impl RopeScaling {
    /// Get the rope type, checking both rope_type and type fields.
    pub fn get_type(&self) -> Option<&str> {
        self.rope_type.as_deref().or(self.type_field.as_deref())
    }
}

/// Configuration for the TalkerCodePredictor (LLM-based codec predictor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkerCodePredictorConfig {
    /// Vocabulary size (default: 2048)
    #[serde(default = "default_code_vocab_size")]
    pub vocab_size: usize,

    /// Hidden dimension (default: 1024)
    #[serde(default = "default_code_hidden_size")]
    pub hidden_size: usize,

    /// Intermediate (MLP) dimension (default: 3072)
    #[serde(default = "default_code_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of transformer layers (default: 5)
    #[serde(default = "default_code_num_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads (default: 16)
    #[serde(default = "default_code_num_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads for GQA (default: 8)
    #[serde(default = "default_code_num_kv_heads")]
    pub num_key_value_heads: usize,

    /// Attention head dimension (default: 128)
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    /// Activation function (default: "silu")
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Maximum position embeddings (default: 32768)
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Initializer range (default: 0.02)
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,

    /// RMS norm epsilon (default: 1e-6)
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Whether to use KV cache
    #[serde(default = "default_true")]
    pub use_cache: bool,

    /// RoPE theta base (default: 10000.0)
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// RoPE scaling configuration
    pub rope_scaling: Option<RopeScaling>,

    /// Whether to use attention bias
    #[serde(default)]
    pub attention_bias: bool,

    /// Whether to use sliding window attention
    #[serde(default)]
    pub use_sliding_window: bool,

    /// Sliding window size (default: 4096)
    #[serde(default = "default_sliding_window")]
    pub sliding_window: Option<usize>,

    /// Max layers using full attention (default: 28)
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,

    /// Attention dropout (default: 0.0)
    #[serde(default)]
    pub attention_dropout: f64,

    /// Number of code groups (default: 32)
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
}

fn default_code_vocab_size() -> usize {
    2048
}
fn default_code_hidden_size() -> usize {
    1024
}
fn default_code_intermediate_size() -> usize {
    3072
}
fn default_code_num_layers() -> usize {
    5
}
fn default_code_num_heads() -> usize {
    16
}
fn default_code_num_kv_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_max_position_embeddings() -> usize {
    32768
}
fn default_initializer_range() -> f64 {
    0.02
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_true() -> bool {
    true
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_sliding_window() -> Option<usize> {
    Some(4096)
}
fn default_max_window_layers() -> usize {
    28
}
fn default_num_code_groups() -> usize {
    32
}

impl Default for TalkerCodePredictorConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_code_vocab_size(),
            hidden_size: default_code_hidden_size(),
            intermediate_size: default_code_intermediate_size(),
            num_hidden_layers: default_code_num_layers(),
            num_attention_heads: default_code_num_heads(),
            num_key_value_heads: default_code_num_kv_heads(),
            head_dim: default_head_dim(),
            hidden_act: default_hidden_act(),
            max_position_embeddings: default_max_position_embeddings(),
            initializer_range: default_initializer_range(),
            rms_norm_eps: default_rms_norm_eps(),
            use_cache: default_true(),
            rope_theta: default_rope_theta(),
            rope_scaling: None,
            attention_bias: false,
            use_sliding_window: false,
            sliding_window: default_sliding_window(),
            max_window_layers: default_max_window_layers(),
            attention_dropout: 0.0,
            num_code_groups: default_num_code_groups(),
        }
    }
}

/// Dialect value that can be either a boolean or a dialect name string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DialectValue {
    /// Boolean flag (false = standard, true = dialect)
    Bool(bool),
    /// Dialect name string (e.g., "sichuan_dialect", "beijing_dialect")
    Dialect(String),
}

impl DialectValue {
    /// Check if this is a dialect (either true or a named dialect).
    pub fn is_dialect(&self) -> bool {
        match self {
            DialectValue::Bool(b) => *b,
            DialectValue::Dialect(_) => true,
        }
    }

    /// Get the dialect name if specified.
    pub fn dialect_name(&self) -> Option<&str> {
        match self {
            DialectValue::Bool(_) => None,
            DialectValue::Dialect(s) => Some(s.as_str()),
        }
    }
}

/// Configuration for the Talker model (main TTS transformer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkerConfig {
    /// Code predictor sub-config
    #[serde(default)]
    pub code_predictor_config: TalkerCodePredictorConfig,

    /// Vocabulary size (default: 3072)
    #[serde(default = "default_talker_vocab_size")]
    pub vocab_size: usize,

    /// Hidden dimension (default: 1024)
    #[serde(default = "default_talker_hidden_size")]
    pub hidden_size: usize,

    /// Intermediate (MLP) dimension (default: 2048)
    #[serde(default = "default_talker_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of transformer layers (default: 20)
    #[serde(default = "default_talker_num_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads (default: 16)
    #[serde(default = "default_talker_num_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads for GQA (default: 2)
    #[serde(default = "default_talker_num_kv_heads")]
    pub num_key_value_heads: usize,

    /// Activation function (default: "silu")
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Maximum position embeddings (default: 32768)
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Initializer range (default: 0.02)
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,

    /// RMS norm epsilon (default: 1e-6)
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Whether to use KV cache
    #[serde(default = "default_true")]
    pub use_cache: bool,

    /// RoPE theta base (default: 10000.0)
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// RoPE scaling configuration
    pub rope_scaling: Option<RopeScaling>,

    /// Whether to use attention bias
    #[serde(default)]
    pub attention_bias: bool,

    /// Whether to use sliding window attention
    #[serde(default)]
    pub use_sliding_window: bool,

    /// Sliding window size (default: 4096)
    #[serde(default = "default_sliding_window")]
    pub sliding_window: Option<usize>,

    /// Attention dropout (default: 0.0)
    #[serde(default)]
    pub attention_dropout: f64,

    /// Number of code groups (default: 32)
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,

    /// Text encoder hidden size (default: 2048)
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,

    /// Codec EOS token ID (default: 4198)
    #[serde(default = "default_codec_eos_token_id")]
    pub codec_eos_token_id: u32,

    /// Codec think token ID (default: 4202)
    #[serde(default = "default_codec_think_id")]
    pub codec_think_id: u32,

    /// Codec no-think token ID (default: 4203)
    #[serde(default = "default_codec_nothink_id")]
    pub codec_nothink_id: u32,

    /// Codec think BOS token ID (default: 4204)
    #[serde(default = "default_codec_think_bos_id")]
    pub codec_think_bos_id: u32,

    /// Codec think EOS token ID (default: 4205)
    #[serde(default = "default_codec_think_eos_id")]
    pub codec_think_eos_id: u32,

    /// Codec pad token ID (default: 4196)
    #[serde(default = "default_codec_pad_id")]
    pub codec_pad_id: u32,

    /// Codec BOS token ID (default: 4197)
    #[serde(default = "default_codec_bos_id")]
    pub codec_bos_id: u32,

    /// Speaker ID mapping
    pub spk_id: Option<HashMap<String, u32>>,

    /// Speaker dialect flags (can be bool or dialect name string)
    #[serde(default)]
    pub spk_is_dialect: Option<HashMap<String, DialectValue>>,

    /// Codec language ID mapping
    pub codec_language_id: Option<HashMap<String, u32>>,
}

fn default_talker_vocab_size() -> usize {
    3072
}
fn default_talker_hidden_size() -> usize {
    1024
}
fn default_talker_intermediate_size() -> usize {
    2048
}
fn default_talker_num_layers() -> usize {
    20
}
fn default_talker_num_heads() -> usize {
    16
}
fn default_talker_num_kv_heads() -> usize {
    2
}
fn default_text_hidden_size() -> usize {
    2048
}
fn default_codec_eos_token_id() -> u32 {
    4198
}
fn default_codec_think_id() -> u32 {
    4202
}
fn default_codec_nothink_id() -> u32 {
    4203
}
fn default_codec_think_bos_id() -> u32 {
    4204
}
fn default_codec_think_eos_id() -> u32 {
    4205
}
fn default_codec_pad_id() -> u32 {
    4196
}
fn default_codec_bos_id() -> u32 {
    4197
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            code_predictor_config: TalkerCodePredictorConfig::default(),
            vocab_size: default_talker_vocab_size(),
            hidden_size: default_talker_hidden_size(),
            intermediate_size: default_talker_intermediate_size(),
            num_hidden_layers: default_talker_num_layers(),
            num_attention_heads: default_talker_num_heads(),
            num_key_value_heads: default_talker_num_kv_heads(),
            hidden_act: default_hidden_act(),
            max_position_embeddings: default_max_position_embeddings(),
            initializer_range: default_initializer_range(),
            rms_norm_eps: default_rms_norm_eps(),
            use_cache: default_true(),
            rope_theta: default_rope_theta(),
            rope_scaling: None,
            attention_bias: false,
            use_sliding_window: false,
            sliding_window: default_sliding_window(),
            attention_dropout: 0.0,
            num_code_groups: default_num_code_groups(),
            text_hidden_size: default_text_hidden_size(),
            codec_eos_token_id: default_codec_eos_token_id(),
            codec_think_id: default_codec_think_id(),
            codec_nothink_id: default_codec_nothink_id(),
            codec_think_bos_id: default_codec_think_bos_id(),
            codec_think_eos_id: default_codec_think_eos_id(),
            codec_pad_id: default_codec_pad_id(),
            codec_bos_id: default_codec_bos_id(),
            spk_id: None,
            spk_is_dialect: None,
            codec_language_id: None,
        }
    }
}

/// TTS model type variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TTSModelType {
    /// Base model for voice cloning
    #[default]
    Base,
    /// Custom voice model with predefined speakers
    CustomVoice,
    /// Voice design model with natural language instructions
    VoiceDesign,
}

/// Tokenizer type variants (25Hz or 12Hz).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerType {
    /// 25Hz tokenizer (V1)
    #[serde(alias = "qwen3_tts_tokenizer_25hz", alias = "25hz")]
    V1_25Hz,
    /// 12Hz tokenizer (V2)
    #[serde(alias = "qwen3_tts_tokenizer_12hz", alias = "12hz")]
    #[default]
    V2_12Hz,
}

/// Main configuration for Qwen3 TTS model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TTSConfig {
    /// Model type identifier
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// Talker model configuration
    #[serde(default)]
    pub talker_config: TalkerConfig,

    /// Speaker encoder configuration
    #[serde(default)]
    pub speaker_encoder_config: SpeakerEncoderConfig,

    /// Tokenizer type (25Hz or 12Hz)
    pub tokenizer_type: Option<String>,

    /// TTS model size (e.g., "0b6", "1b7")
    pub tts_model_size: Option<String>,

    /// TTS model type (base, custom_voice, voice_design)
    pub tts_model_type: Option<String>,

    /// IM start token ID (default: 151644)
    #[serde(default = "default_im_start_token_id")]
    pub im_start_token_id: u32,

    /// IM end token ID (default: 151645)
    #[serde(default = "default_im_end_token_id")]
    pub im_end_token_id: u32,

    /// TTS pad token ID (default: 151671)
    #[serde(default = "default_tts_pad_token_id")]
    pub tts_pad_token_id: u32,

    /// TTS BOS token ID (default: 151672)
    #[serde(default = "default_tts_bos_token_id")]
    pub tts_bos_token_id: u32,

    /// TTS EOS token ID (default: 151673)
    #[serde(default = "default_tts_eos_token_id")]
    pub tts_eos_token_id: u32,

    /// Assistant token ID (default: 77091)
    #[serde(default = "default_assistant_token_id")]
    pub assistant_token_id: u32,
}

fn default_model_type() -> String {
    "qwen3_tts".to_string()
}
fn default_im_start_token_id() -> u32 {
    151644
}
fn default_im_end_token_id() -> u32 {
    151645
}
fn default_tts_pad_token_id() -> u32 {
    151671
}
fn default_tts_bos_token_id() -> u32 {
    151672
}
fn default_tts_eos_token_id() -> u32 {
    151673
}
fn default_assistant_token_id() -> u32 {
    77091
}

impl Default for Qwen3TTSConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            talker_config: TalkerConfig::default(),
            speaker_encoder_config: SpeakerEncoderConfig::default(),
            tokenizer_type: None,
            tts_model_size: None,
            tts_model_type: None,
            im_start_token_id: default_im_start_token_id(),
            im_end_token_id: default_im_end_token_id(),
            tts_pad_token_id: default_tts_pad_token_id(),
            tts_bos_token_id: default_tts_bos_token_id(),
            tts_eos_token_id: default_tts_eos_token_id(),
            assistant_token_id: default_assistant_token_id(),
        }
    }
}

impl Qwen3TTSConfig {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &std::path::Path) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Get the parsed TTS model type.
    pub fn get_tts_model_type(&self) -> TTSModelType {
        match self.tts_model_type.as_deref() {
            Some("base") => TTSModelType::Base,
            Some("custom_voice") => TTSModelType::CustomVoice,
            Some("voice_design") => TTSModelType::VoiceDesign,
            _ => TTSModelType::Base,
        }
    }

    /// Get the parsed tokenizer type.
    pub fn get_tokenizer_type(&self) -> TokenizerType {
        match self.tokenizer_type.as_deref() {
            Some(s) if s.contains("25hz") => TokenizerType::V1_25Hz,
            Some(s) if s.contains("12hz") => TokenizerType::V2_12Hz,
            _ => TokenizerType::V2_12Hz,
        }
    }
}

/// Generation configuration for TTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Whether to use sampling (default: true)
    #[serde(default = "default_true")]
    pub do_sample: bool,

    /// Top-k sampling parameter (default: 50)
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Top-p (nucleus) sampling parameter (default: 1.0)
    #[serde(default = "default_top_p")]
    pub top_p: f64,

    /// Sampling temperature (default: 0.9)
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Repetition penalty (default: 1.05)
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,

    /// Sub-talker sampling enabled (default: true)
    #[serde(default = "default_true")]
    pub subtalker_dosample: bool,

    /// Sub-talker top-k (default: 50)
    #[serde(default = "default_top_k")]
    pub subtalker_top_k: usize,

    /// Sub-talker top-p (default: 1.0)
    #[serde(default = "default_top_p")]
    pub subtalker_top_p: f64,

    /// Sub-talker temperature (default: 0.9)
    #[serde(default = "default_temperature")]
    pub subtalker_temperature: f64,

    /// Maximum new tokens to generate (default: 2048)
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
}

fn default_top_k() -> usize {
    50
}
fn default_top_p() -> f64 {
    1.0
}
fn default_temperature() -> f64 {
    0.9
}
fn default_repetition_penalty() -> f64 {
    1.05
}
fn default_max_new_tokens() -> usize {
    2048
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            do_sample: true,
            top_k: default_top_k(),
            top_p: default_top_p(),
            temperature: default_temperature(),
            repetition_penalty: default_repetition_penalty(),
            subtalker_dosample: true,
            subtalker_top_k: default_top_k(),
            subtalker_top_p: default_top_p(),
            subtalker_temperature: default_temperature(),
            max_new_tokens: default_max_new_tokens(),
        }
    }
}

impl GenerationConfig {
    /// Load generation config from a JSON file.
    pub fn from_file(path: &std::path::Path) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs() {
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
        assert_eq!(config.im_start_token_id, 151644);

        let gen_config = GenerationConfig::default();
        assert!(gen_config.do_sample);
        assert_eq!(gen_config.top_k, 50);
    }

    #[test]
    fn test_config_serialization() {
        let config = Qwen3TTSConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: Qwen3TTSConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_type, config.model_type);
    }
}
