// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Inference implementation for Qwen3 TTS.
//!
//! This module provides actual model inference using loaded weights.
//! The architecture follows the Python reference:
//! - TalkerModel: 28-layer transformer with dual-stream (text + codec) input
//! - CodePredictor: 5-layer sub-transformer for generating codes 1-15
//! - Input format: text embeddings (projected 2048→1024) summed with codec embeddings (1024)

use crate::config::{Qwen3TTSConfig, TalkerCodePredictorConfig, TalkerConfig};
use crate::error::{Qwen3TTSError, Result};
use crate::layers::{Linear, RMSNorm, RotaryEmbedding, TransformerLayer};
use crate::vocoder::{load_vocoder_weights, Vocoder, VocoderConfig};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Kind, Tensor};
use tokenizers::Tokenizer;

/// Code predictor sub-transformer for generating codes 1-15 autoregressively.
///
/// Given the main model's hidden state and code 0 embedding, this model
/// generates the remaining 15 code groups one at a time, each conditioning
/// on previously generated codes.
pub struct CodePredictor {
    /// Embeddings for codes 1-15 (indexed 0-14)
    pub code_embeddings: Vec<Tensor>,
    /// Transformer layers (5 layers)
    layers: Vec<TransformerLayer>,
    /// Final layer norm
    norm: RMSNorm,
    /// LM heads for codes 1-15 (indexed 0-14)
    lm_heads: Vec<Linear>,
    /// Rotary position embedding
    rotary_emb: RotaryEmbedding,
    /// Optional projection from main model hidden size to code predictor hidden size
    /// Present in 1.7B+ models where talker hidden_size > code_predictor hidden_size
    small_to_mtp_projection: Option<Linear>,
    /// Hidden size
    _hidden_size: i64,
    /// Device
    device: Device,
}

impl CodePredictor {
    /// Load the code predictor from weights.
    pub fn load(
        weights: &HashMap<String, Tensor>,
        config: &TalkerCodePredictorConfig,
        device: Device,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size as i64;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads as i64;
        let num_kv_heads = config.num_key_value_heads as i64;
        let head_dim = config.head_dim as i64;
        let intermediate_size = config.intermediate_size as i64;
        let rms_norm_eps = config.rms_norm_eps;
        let num_code_groups = config.num_code_groups as i64;

        println!(
            "Loading CodePredictor with {} layers, hidden_size={}, num_heads={}, num_kv_heads={}",
            num_layers, hidden_size, num_heads, num_kv_heads
        );

        // Load codec embeddings (15 for codes 1-15)
        let mut code_embeddings = Vec::new();
        for i in 0..(num_code_groups - 1) {
            let key = format!("talker.code_predictor.model.codec_embedding.{}.weight", i);
            if let Some(tensor) = weights.get(&key) {
                code_embeddings.push(tensor.to_device(device).to_kind(Kind::Float));
            }
        }
        println!(
            "  Loaded {} code_predictor embeddings",
            code_embeddings.len()
        );

        // Load transformer layers
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let prefix = format!("talker.code_predictor.model.layers.{}", i);
            let layer = TransformerLayer::from_weights(
                weights,
                &prefix,
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                device,
            )
            .ok_or_else(|| {
                Qwen3TTSError::ModelLoad(format!("Failed to load code_predictor layer {}", i))
            })?;
            layers.push(layer);
        }
        println!("  Loaded {} code_predictor layers", layers.len());

        // Load final norm
        let norm_weight = weights
            .get("talker.code_predictor.model.norm.weight")
            .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing code_predictor norm.weight".into()))?
            .to_device(device)
            .to_kind(Kind::Float);
        let norm = RMSNorm::from_weights(norm_weight, rms_norm_eps);

        // Load LM heads (15 for codes 1-15)
        let mut lm_heads = Vec::new();
        for i in 0..(num_code_groups - 1) {
            let key = format!("talker.code_predictor.lm_head.{}.weight", i);
            if let Some(tensor) = weights.get(&key) {
                lm_heads.push(Linear::from_weights(
                    tensor.to_device(device).to_kind(Kind::Float),
                ));
            }
        }
        println!("  Loaded {} code_predictor LM heads", lm_heads.len());

        // Load optional small_to_mtp_projection (present in 1.7B+ models)
        let small_to_mtp_projection =
            if let Some(proj_weight) = weights.get("talker.code_predictor.small_to_mtp_projection.weight") {
                let proj_bias = weights.get("talker.code_predictor.small_to_mtp_projection.bias");
                let proj = if let Some(bias) = proj_bias {
                    Linear::from_weights_with_bias(
                        proj_weight.to_device(device).to_kind(Kind::Float),
                        bias.to_device(device).to_kind(Kind::Float),
                    )
                } else {
                    Linear::from_weights(proj_weight.to_device(device).to_kind(Kind::Float))
                };
                println!("  Loaded small_to_mtp_projection");
                Some(proj)
            } else {
                None
            };

        // Create rotary embedding
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings as i64,
            config.rope_theta,
            device,
        );

        Ok(Self {
            code_embeddings,
            layers,
            norm,
            lm_heads,
            rotary_emb,
            small_to_mtp_projection,
            _hidden_size: hidden_size,
            device,
        })
    }

    /// Generate codes 1-15 autoregressively.
    ///
    /// # Arguments
    /// * `main_hidden` - Hidden state from main model at last position [1, 1, main_hidden_size]
    /// * `code_0_embedding` - Embedding of code 0 from main codec_embedding [1, 1, main_hidden_size]
    /// * `temperature` - Sampling temperature (0 = greedy)
    /// * `top_k` - Top-k sampling parameter
    pub fn generate_codes(
        &self,
        main_hidden: &Tensor,
        code_0_embedding: &Tensor,
        temperature: f64,
        top_k: i64,
    ) -> Vec<i64> {
        let mut codes = Vec::new();

        // Start with [main_hidden, code_0_embedding] as input
        // Note: For 1.7B+ models, these are at main_hidden_size (2048), not code_predictor hidden_size (1024)
        let mut sequence = Tensor::cat(
            &[
                main_hidden.shallow_clone(),
                code_0_embedding.shallow_clone(),
            ],
            1,
        );

        for step in 0..self.lm_heads.len() {
            let seq_len = sequence.size()[1];

            // Create causal mask
            let mask = Tensor::zeros([seq_len, seq_len], (Kind::Float, self.device));
            let upper = Tensor::ones([seq_len, seq_len], (Kind::Bool, self.device)).triu(1);
            let causal_mask = mask.masked_fill(&upper, f64::NEG_INFINITY);
            let causal_mask = causal_mask.view([1, 1, seq_len, seq_len]);

            // Apply projection if present (1.7B+ models: project from 2048 to 1024)
            let projected_sequence = if let Some(ref proj) = self.small_to_mtp_projection {
                proj.forward(&sequence)
            } else {
                sequence.shallow_clone()
            };

            // Run through transformer layers
            let mut hidden = projected_sequence;
            for layer in &self.layers {
                hidden = layer.forward(&hidden, &self.rotary_emb, Some(&causal_mask));
            }

            // Apply final norm and get logits at last position
            let normed = self.norm.forward(&hidden);
            let last_hidden = normed.select(1, seq_len - 1).unsqueeze(0);
            let logits = self.lm_heads[step].forward(&last_hidden).squeeze_dim(0);

            // Sample or argmax
            let code = if temperature <= 0.0 {
                logits.argmax(-1, false).int64_value(&[0])
            } else {
                let logits = &logits / temperature;

                // Apply top-k filtering
                let logits = if top_k > 0 {
                    let vocab_size = logits.size()[logits.dim() - 1];
                    let k = top_k.min(vocab_size);
                    let (top_values, _) = logits.topk(k, -1, true, true);
                    let threshold = top_values.select(-1, k - 1);
                    let mask = logits.lt_tensor(&threshold.unsqueeze(-1));
                    logits.masked_fill(&mask, f64::NEG_INFINITY)
                } else {
                    logits
                };

                let probs = logits.softmax(-1, Kind::Float);
                probs.multinomial(1, true).int64_value(&[0, 0])
            };

            codes.push(code);

            // Embed the predicted code and append to sequence
            if step < self.code_embeddings.len() {
                let code_tensor = Tensor::from_slice(&[code]).to_device(self.device);
                let emb = self.code_embeddings[step]
                    .index_select(0, &code_tensor)
                    .unsqueeze(0); // [1, 1, hidden_size]
                sequence = Tensor::cat(&[sequence, emb], 1);
            }
        }

        codes
    }
}

/// Talker model for TTS generation.
///
/// This is the main 28-layer transformer that operates on dual-stream
/// (text + codec) embeddings and predicts code 0 at each generation step.
pub struct TalkerModel {
    /// Text embedding layer [151936, 2048]
    text_embedding: Tensor,
    /// Text projection FC1 (2048 -> 2048)
    text_proj_fc1_weight: Tensor,
    text_proj_fc1_bias: Tensor,
    /// Text projection FC2 (2048 -> 1024)
    text_proj_fc2_weight: Tensor,
    text_proj_fc2_bias: Tensor,
    /// Main codec embedding [3072, 1024]
    codec_embedding: Tensor,
    /// Transformer layers (28 layers)
    layers: Vec<TransformerLayer>,
    /// Final layer norm
    norm: RMSNorm,
    /// Main codec head for predicting code 0
    codec_head: Linear,
    /// Code predictor sub-transformer for codes 1-15
    code_predictor: CodePredictor,
    /// Rotary position embedding
    rotary_emb: RotaryEmbedding,
    /// Hidden size
    hidden_size: i64,
    /// Device
    device: Device,
}

impl TalkerModel {
    /// Load the Talker model from weights.
    pub fn load(
        weights: &HashMap<String, Tensor>,
        config: &TalkerConfig,
        device: Device,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size as i64;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads as i64;
        let num_kv_heads = config.num_key_value_heads as i64;
        let intermediate_size = config.intermediate_size as i64;
        let rms_norm_eps = config.rms_norm_eps;

        // Calculate head_dim from q_proj weight shape
        let head_dim =
            if let Some(q_proj) = weights.get("talker.model.layers.0.self_attn.q_proj.weight") {
                let q_out = q_proj.size()[0];
                q_out / num_heads
            } else {
                128
            };

        println!(
            "Loading TalkerModel with {} layers, hidden_size={}, num_heads={}, num_kv_heads={}, head_dim={}",
            num_layers, hidden_size, num_heads, num_kv_heads, head_dim
        );

        // Load text embedding [151936, 2048]
        let text_embedding = weights
            .get("talker.model.text_embedding.weight")
            .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing text_embedding.weight".into()))?
            .to_device(device)
            .to_kind(Kind::Float);
        println!("  Loaded text_embedding: {:?}", text_embedding.size());

        // Load text projection layers (2048 -> 1024)
        let text_proj_fc1_weight = weights
            .get("talker.text_projection.linear_fc1.weight")
            .ok_or_else(|| {
                Qwen3TTSError::ModelLoad("Missing text_projection.linear_fc1.weight".into())
            })?
            .to_device(device)
            .to_kind(Kind::Float);
        let text_proj_fc1_bias = weights
            .get("talker.text_projection.linear_fc1.bias")
            .ok_or_else(|| {
                Qwen3TTSError::ModelLoad("Missing text_projection.linear_fc1.bias".into())
            })?
            .to_device(device)
            .to_kind(Kind::Float);
        let text_proj_fc2_weight = weights
            .get("talker.text_projection.linear_fc2.weight")
            .ok_or_else(|| {
                Qwen3TTSError::ModelLoad("Missing text_projection.linear_fc2.weight".into())
            })?
            .to_device(device)
            .to_kind(Kind::Float);
        let text_proj_fc2_bias = weights
            .get("talker.text_projection.linear_fc2.bias")
            .ok_or_else(|| {
                Qwen3TTSError::ModelLoad("Missing text_projection.linear_fc2.bias".into())
            })?
            .to_device(device)
            .to_kind(Kind::Float);
        println!("  Loaded text_projection layers");

        // Load main codec embedding [3072, 1024]
        let codec_embedding = weights
            .get("talker.model.codec_embedding.weight")
            .ok_or_else(|| {
                Qwen3TTSError::ModelLoad("Missing talker.model.codec_embedding.weight".into())
            })?
            .to_device(device)
            .to_kind(Kind::Float);
        println!("  Loaded codec_embedding: {:?}", codec_embedding.size());

        // Load transformer layers
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let prefix = format!("talker.model.layers.{}", i);
            let layer = TransformerLayer::from_weights(
                weights,
                &prefix,
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                device,
            )
            .ok_or_else(|| {
                Qwen3TTSError::ModelLoad(format!("Failed to load transformer layer {}", i))
            })?;
            layers.push(layer);
        }
        println!("  Loaded {} transformer layers", layers.len());

        // Load final layer norm
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing norm.weight".into()))?
            .to_device(device)
            .to_kind(Kind::Float);
        let norm = RMSNorm::from_weights(norm_weight, rms_norm_eps);
        println!("  Loaded final norm");

        // Load codec_head for predicting code 0
        let codec_head_weight = weights
            .get("talker.codec_head.weight")
            .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing talker.codec_head.weight".into()))?
            .to_device(device)
            .to_kind(Kind::Float);
        let codec_head = Linear::from_weights(codec_head_weight);
        println!("  Loaded codec_head");

        // Load code predictor sub-transformer
        let code_predictor = CodePredictor::load(weights, &config.code_predictor_config, device)?;
        println!("  Loaded code_predictor");

        // Create rotary embedding
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings as i64,
            config.rope_theta,
            device,
        );

        Ok(Self {
            text_embedding,
            text_proj_fc1_weight,
            text_proj_fc1_bias,
            text_proj_fc2_weight,
            text_proj_fc2_bias,
            codec_embedding,
            layers,
            norm,
            codec_head,
            code_predictor,
            rotary_emb,
            hidden_size,
            device,
        })
    }

    /// Embed text token IDs through text_embedding → text_projection (2048 → 1024).
    fn embed_text(&self, token_ids: &[i64]) -> Tensor {
        let ids = Tensor::from_slice(token_ids).to_device(self.device);
        let embedded = self.text_embedding.index_select(0, &ids); // [N, 2048]

        // Text projection: FC1(2048→2048) with GELU, then FC2(2048→1024)
        let h = embedded.matmul(&self.text_proj_fc1_weight.tr()) + &self.text_proj_fc1_bias;
        let h = h.gelu("none");
        let projected = h.matmul(&self.text_proj_fc2_weight.tr()) + &self.text_proj_fc2_bias;

        projected.unsqueeze(0) // [1, N, 1024]
    }

    /// Embed codec token IDs through codec_embedding (3072 → 1024).
    fn embed_codec(&self, token_ids: &[i64]) -> Tensor {
        let ids = Tensor::from_slice(token_ids).to_device(self.device);
        self.codec_embedding.index_select(0, &ids).unsqueeze(0) // [1, N, 1024]
    }

    /// Build the full dual-stream input embeddings for custom_voice mode.
    ///
    /// The input sequence has positions where text embeddings and codec embeddings
    /// are summed together. The structure is:
    /// - Role prefix: 3 text-only positions
    /// - Codec prefix: think tokens + speaker + pad/bos (text_pad summed with codec tokens)
    /// - Text content: text tokens summed with codec_pad
    /// - Final: tts_pad + codec_bos
    #[allow(clippy::too_many_arguments)]
    pub fn build_input_embeddings(
        &self,
        text_token_ids: &[i64],
        speaker_id: i64,
        language_id: i64,
        tts_pad_id: i64,
        tts_bos_id: i64,
        tts_eos_id: i64,
        codec_think_id: i64,
        codec_think_bos_id: i64,
        codec_think_eos_id: i64,
        codec_pad_id: i64,
        codec_bos_id: i64,
    ) -> Tensor {
        // Phase 1: Role prefix (3 text-only positions)
        // Tokenize "<|im_start|>assistant\n" → these are the first 3 tokens of input_id
        // We need the actual token IDs for the role prefix
        let role_embed = self.embed_text(&text_token_ids[..3]); // [1, 3, 1024]

        // Phase 2: Codec prefix (6 positions for custom_voice)
        // Text side: tts_pad * 5 + tts_bos
        let tts_pad_embed = self.embed_text(&[tts_pad_id]); // [1, 1, 1024]
        let tts_bos_embed = self.embed_text(&[tts_bos_id]); // [1, 1, 1024]
        let tts_eos_embed = self.embed_text(&[tts_eos_id]); // [1, 1, 1024]

        // Codec prefix tokens: [think_id, think_bos_id, language_id, think_eos_id, speaker_id, codec_pad_id, codec_bos_id]
        let codec_prefix = self.embed_codec(&[
            codec_think_id,
            codec_think_bos_id,
            language_id,
            codec_think_eos_id,
            speaker_id,
            codec_pad_id,
            codec_bos_id,
        ]); // [1, 7, 1024]

        // Text side for codec prefix: tts_pad repeated 5 times + tts_bos
        let tts_pad_5 = tts_pad_embed.expand([1, 5, self.hidden_size], false); // [1, 5, 1024]
        let text_for_codec = Tensor::cat(&[tts_pad_5, tts_bos_embed.shallow_clone()], 1); // [1, 6, 1024]

        // Sum text + codec (first 6 of 7 codec prefix positions)
        let codec_prefix_6 = codec_prefix.narrow(1, 0, 6); // [1, 6, 1024]
        let phase2 = &text_for_codec + &codec_prefix_6; // [1, 6, 1024]

        // Phase 3: Text content (N positions + tts_eos)
        // text_token_ids[3..] contains the text content + tail tokens
        // In the full token sequence: token[3..N+3] = text, token[N+3..] = tail
        // We need text_token_ids[3..(len-5)] for content
        let text_start = 3;
        let text_end = text_token_ids.len().saturating_sub(5);
        let text_content_ids = &text_token_ids[text_start..text_end];
        let num_text_tokens = text_content_ids.len();

        let text_content_embed = if num_text_tokens > 0 {
            self.embed_text(text_content_ids) // [1, N, 1024]
        } else {
            Tensor::zeros([1, 0, self.hidden_size], (Kind::Float, self.device))
        };

        // Concatenate text content + tts_eos
        let text_with_eos = Tensor::cat(&[text_content_embed, tts_eos_embed.shallow_clone()], 1); // [1, N+1, 1024]

        // Codec side: codec_pad repeated (N+1) times
        let codec_pad_embed = self.embed_codec(&[codec_pad_id]); // [1, 1, 1024]
        let codec_pad_repeated =
            codec_pad_embed.expand([1, (num_text_tokens + 1) as i64, self.hidden_size], false);

        let phase3 = &text_with_eos + &codec_pad_repeated; // [1, N+1, 1024]

        // Phase 4: Final codec_bos
        let codec_bos_embed = codec_prefix.narrow(1, 6, 1); // Last position of codec prefix [1, 1, 1024]
        let phase4 = &tts_pad_embed + &codec_bos_embed; // [1, 1, 1024]

        // Concatenate all phases
        let input_embeddings = Tensor::cat(&[role_embed, phase2, phase3, phase4], 1);

        println!(
            "  Built input embeddings: {} positions (3 role + 6 codec_prefix + {} text + 1 eos + 1 bos)",
            input_embeddings.size()[1],
            num_text_tokens
        );

        input_embeddings
    }

    /// Build dual-stream input embeddings using a speaker embedding (x-vector)
    /// instead of a speaker ID. Used for voice cloning with the Base model.
    #[allow(clippy::too_many_arguments)]
    pub fn build_input_embeddings_with_xvector(
        &self,
        text_token_ids: &[i64],
        speaker_embedding: &Tensor,
        language_id: i64,
        tts_pad_id: i64,
        tts_bos_id: i64,
        tts_eos_id: i64,
        codec_think_id: i64,
        codec_think_bos_id: i64,
        codec_think_eos_id: i64,
        codec_pad_id: i64,
        codec_bos_id: i64,
    ) -> Tensor {
        // Phase 1: Role prefix (3 text-only positions)
        let role_embed = self.embed_text(&text_token_ids[..3]); // [1, 3, 1024]

        // Phase 2: Codec prefix (6 positions)
        // Text side: tts_pad * 5 + tts_bos
        let tts_pad_embed = self.embed_text(&[tts_pad_id]); // [1, 1, 1024]
        let tts_bos_embed = self.embed_text(&[tts_bos_id]); // [1, 1, 1024]
        let tts_eos_embed = self.embed_text(&[tts_eos_id]); // [1, 1, 1024]

        // Codec prefix tokens (without speaker_id — we inject x-vector at position 4)
        let codec_tokens_before_speaker = self.embed_codec(&[
            codec_think_id,
            codec_think_bos_id,
            language_id,
            codec_think_eos_id,
        ]); // [1, 4, 1024]

        // Speaker embedding: use provided x-vector directly (reshape to [1, 1, 1024])
        let spk_embed = speaker_embedding
            .to_kind(Kind::Float)
            .to_device(self.device);
        let spk_embed = if spk_embed.dim() == 1 {
            spk_embed.unsqueeze(0).unsqueeze(0) // [1024] → [1, 1, 1024]
        } else if spk_embed.dim() == 2 {
            spk_embed.unsqueeze(0) // [1, 1024] → [1, 1, 1024]
        } else {
            spk_embed
        };

        let codec_tokens_after_speaker = self.embed_codec(&[codec_pad_id, codec_bos_id]); // [1, 2, 1024]

        // Full codec prefix: [think, think_bos, language, think_eos, SPEAKER, pad, bos] = [1, 7, 1024]
        let codec_prefix = Tensor::cat(
            &[
                codec_tokens_before_speaker,
                spk_embed,
                codec_tokens_after_speaker,
            ],
            1,
        );

        // Text side for codec prefix: tts_pad repeated 5 times + tts_bos
        let tts_pad_5 = tts_pad_embed.expand([1, 5, self.hidden_size], false);
        let text_for_codec = Tensor::cat(&[tts_pad_5, tts_bos_embed.shallow_clone()], 1); // [1, 6, 1024]

        // Sum text + codec (first 6 of 7 codec prefix positions)
        let codec_prefix_6 = codec_prefix.narrow(1, 0, 6); // [1, 6, 1024]
        let phase2 = &text_for_codec + &codec_prefix_6; // [1, 6, 1024]

        // Phase 3: Text content (N positions + tts_eos)
        let text_start = 3;
        let text_end = text_token_ids.len().saturating_sub(5);
        let text_content_ids = &text_token_ids[text_start..text_end];
        let num_text_tokens = text_content_ids.len();

        let text_content_embed = if num_text_tokens > 0 {
            self.embed_text(text_content_ids)
        } else {
            Tensor::zeros([1, 0, self.hidden_size], (Kind::Float, self.device))
        };

        let text_with_eos = Tensor::cat(&[text_content_embed, tts_eos_embed.shallow_clone()], 1);

        let codec_pad_embed = self.embed_codec(&[codec_pad_id]);
        let codec_pad_repeated =
            codec_pad_embed.expand([1, (num_text_tokens + 1) as i64, self.hidden_size], false);

        let phase3 = &text_with_eos + &codec_pad_repeated;

        // Phase 4: Final codec_bos
        let codec_bos_embed = codec_prefix.narrow(1, 6, 1);
        let phase4 = &tts_pad_embed + &codec_bos_embed;

        // Concatenate all phases
        let input_embeddings = Tensor::cat(&[role_embed, phase2, phase3, phase4], 1);

        println!(
            "  Built input embeddings (xvector): {} positions (3 role + 6 codec_prefix + {} text + 1 eos + 1 bos)",
            input_embeddings.size()[1],
            num_text_tokens
        );

        input_embeddings
    }

    /// Build dual-stream input embeddings for ICL (In-Context Learning) voice cloning.
    ///
    /// This includes reference text + synthesis text as the text stream, and
    /// reference codec token embeddings as the codec stream. The speaker embedding
    /// (x-vector) is still used in the codec prefix.
    ///
    /// Structure:
    /// - Codec prefix (6 positions): same as x-vector mode with speaker embedding
    /// - ICL text: all ref_text + synth_text tokens + tts_eos, paired with codec_pad
    /// - ICL codec: codec_bos + ref codec frame embeddings (sum of 16 groups), paired with tts_pad
    /// - Final codec_bos to start generation
    #[allow(clippy::too_many_arguments)]
    pub fn build_input_embeddings_with_icl(
        &self,
        ref_text_token_ids: &[i64],
        synth_text_token_ids: &[i64],
        ref_codes: &[Vec<i64>],
        speaker_embedding: &Tensor,
        language_id: i64,
        tts_pad_id: i64,
        tts_bos_id: i64,
        tts_eos_id: i64,
        codec_think_id: i64,
        codec_think_bos_id: i64,
        codec_think_eos_id: i64,
        codec_pad_id: i64,
        codec_bos_id: i64,
    ) -> Tensor {
        // Phase 1: Role prefix (3 text-only positions)
        // First 3 tokens of ref_text are [im_start, assistant, \n] — standalone, no codec pairing
        let role_tokens = &ref_text_token_ids[..3];
        let role_embed = self.embed_text(role_tokens); // [1, 3, 1024]

        // Phase 2: Codec prefix with x-vector (6 summed positions)
        // Text side: tts_pad * 5 + tts_bos
        let tts_pad_embed = self.embed_text(&[tts_pad_id]); // [1, 1, 1024]
        let tts_bos_embed = self.embed_text(&[tts_bos_id]); // [1, 1, 1024]
        let tts_eos_embed = self.embed_text(&[tts_eos_id]); // [1, 1, 1024]

        // Codec prefix tokens (without speaker_id — inject x-vector at position 4)
        let codec_tokens_before_speaker = self.embed_codec(&[
            codec_think_id,
            codec_think_bos_id,
            language_id,
            codec_think_eos_id,
        ]); // [1, 4, 1024]

        // Speaker embedding
        let spk_embed = speaker_embedding
            .to_kind(Kind::Float)
            .to_device(self.device);
        let spk_embed = if spk_embed.dim() == 1 {
            spk_embed.unsqueeze(0).unsqueeze(0)
        } else if spk_embed.dim() == 2 {
            spk_embed.unsqueeze(0)
        } else {
            spk_embed
        };

        let codec_tokens_after_speaker = self.embed_codec(&[codec_pad_id, codec_bos_id]); // [1, 2, 1024]

        let codec_prefix = Tensor::cat(
            &[
                codec_tokens_before_speaker,
                spk_embed,
                codec_tokens_after_speaker,
            ],
            1,
        ); // [1, 7, 1024]

        // Text side for codec prefix: tts_pad * 5 + tts_bos
        let tts_pad_5 = tts_pad_embed.expand([1, 5, self.hidden_size], false);
        let text_for_codec = Tensor::cat(&[tts_pad_5, tts_bos_embed.shallow_clone()], 1); // [1, 6, 1024]

        // Sum text + codec (first 6 of 7 codec prefix positions)
        let codec_prefix_6 = codec_prefix.narrow(1, 0, 6);
        let phase2 = &text_for_codec + &codec_prefix_6; // [1, 6, 1024]

        // Phase 3: ICL text stream
        // Use pure text tokens only (strip role markers), matching Python:
        //   ref_id = ref_ids[:, 3:-2]  (strips im_start/assistant/\n and im_end/\n)
        //   text_id = input_id[:, 3:-5] (strips im_start/assistant/\n and im_end/\n/im_start/assistant/\n)
        let ref_pure = &ref_text_token_ids[3..ref_text_token_ids.len().saturating_sub(2)];
        let synth_pure = &synth_text_token_ids[3..synth_text_token_ids.len().saturating_sub(5)];
        let mut all_text_ids = Vec::new();
        all_text_ids.extend_from_slice(ref_pure);
        all_text_ids.extend_from_slice(synth_pure);
        let text_embed = self.embed_text(&all_text_ids); // [1, T_text, 1024]
        let text_embed = Tensor::cat(&[text_embed, tts_eos_embed], 1); // [1, T_text+1, 1024]
        let text_len = text_embed.size()[1];

        // Pair each text position with codec_pad
        let codec_pad_embed = self.embed_codec(&[codec_pad_id]); // [1, 1, 1024]
        let codec_pad_repeated = codec_pad_embed.expand([1, text_len, self.hidden_size], false);
        let phase3 = &text_embed + &codec_pad_repeated; // [1, T_text+1, 1024]

        // Phase 4: ICL codec stream
        // codec_bos + ref codec frame embeddings, paired with tts_pad
        let codec_bos_embed = self.embed_codec(&[codec_bos_id]); // [1, 1, 1024]

        // Compute ref codec frame embeddings: sum of all 16 codebook group embeddings per frame
        let num_ref_frames = ref_codes.len();
        let mut ref_codec_embeds = Vec::new();
        for frame_codes in ref_codes {
            let mut frame_embed =
                Tensor::zeros([1, 1, self.hidden_size], (Kind::Float, self.device));
            // Code 0: use main codec_embedding
            if !frame_codes.is_empty() {
                let code0_ids = Tensor::from_slice(&[frame_codes[0]]).to_device(self.device);
                let code0_embed = self
                    .codec_embedding
                    .index_select(0, &code0_ids)
                    .unsqueeze(0);
                frame_embed = &frame_embed + &code0_embed;
            }
            // Codes 1-15: use code_predictor embeddings
            for (i, &code) in frame_codes.iter().enumerate().skip(1) {
                if i - 1 < self.code_predictor.code_embeddings.len() {
                    let code_ids = Tensor::from_slice(&[code]).to_device(self.device);
                    let code_embed = self.code_predictor.code_embeddings[i - 1]
                        .index_select(0, &code_ids)
                        .unsqueeze(0);
                    frame_embed = &frame_embed + &code_embed;
                }
            }
            ref_codec_embeds.push(frame_embed);
        }

        // Concatenate: codec_bos + ref_codec_frames → [1, 1+R, 1024]
        let mut codec_parts = vec![codec_bos_embed];
        codec_parts.extend(ref_codec_embeds);
        let codec_stream = Tensor::cat(&codec_parts, 1); // [1, 1+R, 1024]
        let codec_len = codec_stream.size()[1];

        // Pair each codec position with tts_pad
        let tts_pad_repeated = tts_pad_embed.expand([1, codec_len, self.hidden_size], false);
        let phase4 = &codec_stream + &tts_pad_repeated; // [1, 1+R, 1024]

        // Phase 5: Final codec_bos to start generation
        let final_codec_bos = codec_prefix.narrow(1, 6, 1); // Last position of codec prefix [1, 1, 1024]
        let phase5 = &tts_pad_embed + &final_codec_bos; // [1, 1, 1024]

        // Concatenate all phases
        let input_embeddings = Tensor::cat(&[role_embed, phase2, phase3, phase4, phase5], 1);

        println!(
            "  Built ICL input embeddings: {} positions (3 role + 6 codec_prefix + {} text + {} codec + 1 bos)",
            input_embeddings.size()[1],
            text_len,
            codec_len
        );
        println!(
            "    ref_pure={} tokens, synth_pure={} tokens, ref_codec={} frames",
            ref_pure.len(),
            synth_pure.len(),
            num_ref_frames
        );

        input_embeddings
    }

    /// Run the transformer forward pass on embeddings.
    /// Returns NORMED hidden states (used for both codec_head and code predictor).
    fn forward_embeds(&self, embeddings: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let mut hidden = embeddings.shallow_clone();

        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.rotary_emb, attention_mask);
        }

        self.norm.forward(&hidden)
    }

    /// Predict code 0 from normed hidden states at the last position.
    fn predict_code_0(&self, normed_hidden: &Tensor, temperature: f64, top_k: i64) -> i64 {
        let last_hidden = normed_hidden.select(1, normed_hidden.size()[1] - 1);
        let logits = self.codec_head.forward(&last_hidden);

        if temperature <= 0.0 {
            logits.argmax(-1, false).int64_value(&[0])
        } else {
            let logits = &logits / temperature;

            let logits = if top_k > 0 {
                let vocab_size = logits.size()[logits.dim() - 1];
                let k = top_k.min(vocab_size);
                let (top_values, _) = logits.topk(k, -1, true, true);
                let threshold = top_values.select(-1, k - 1);
                let mask = logits.lt_tensor(&threshold.unsqueeze(-1));
                logits.masked_fill(&mask, f64::NEG_INFINITY)
            } else {
                logits
            };

            let probs = logits.softmax(-1, Kind::Float);
            probs.multinomial(1, true).int64_value(&[0, 0])
        }
    }

    /// Generate all codes (code 0 from codec_head, codes 1-15 from code predictor).
    pub fn generate_codes(
        &self,
        input_embeddings: &Tensor,
        max_codes: i64,
        temperature: f64,
        top_k: i64,
        eos_code: i64,
        tts_pad_embed: &Tensor,
    ) -> Vec<Vec<i64>> {
        let mut all_codes = Vec::new();
        let mut full_sequence = input_embeddings.shallow_clone();

        for step in 0..max_codes {
            let seq_len = full_sequence.size()[1];

            // Create causal mask
            let mask_zeros = Tensor::zeros([seq_len, seq_len], (Kind::Float, self.device));
            let upper = Tensor::ones([seq_len, seq_len], (Kind::Bool, self.device)).triu(1);
            let causal_mask = mask_zeros.masked_fill(&upper, f64::NEG_INFINITY);
            let causal_mask = causal_mask.view([1, 1, seq_len, seq_len]);

            // Run through transformer
            let normed_hidden = self.forward_embeds(&full_sequence, Some(&causal_mask));

            // Predict code 0 from main model
            let code_0 = self.predict_code_0(&normed_hidden, temperature, top_k);

            // Check EOS on code 0 only
            if code_0 == eos_code {
                println!("  EOS detected at step {}", step);
                break;
            }

            // Get main model's normed hidden state at last position for code predictor
            // (Python: past_hidden = hidden_states[:, -1:, :] where hidden_states = outputs.last_hidden_state after norm)
            // select(1, idx) on [1, seq_len, H] → [1, H], then unsqueeze(1) → [1, 1, H]
            let main_hidden = normed_hidden.select(1, seq_len - 1).unsqueeze(1); // [1, 1, hidden_size]

            // Embed code 0 through main codec_embedding
            let code_0_tensor = Tensor::from_slice(&[code_0]).to_device(self.device);
            let code_0_embed = self
                .codec_embedding
                .index_select(0, &code_0_tensor)
                .unsqueeze(0); // [1, 1, hidden_size]

            // Generate codes 1-15 using code predictor
            let predictor_codes =
                self.code_predictor
                    .generate_codes(&main_hidden, &code_0_embed, temperature, top_k);

            // Collect all 16 codes
            let mut frame_codes = vec![code_0];
            frame_codes.extend_from_slice(&predictor_codes);
            all_codes.push(frame_codes.clone());

            // Build next input: sum of all code embeddings + tts_pad (trailing text)
            // Code 0 embedding (from main codec_embedding)
            let mut code_embeds_sum = code_0_embed.shallow_clone();

            // Codes 1-15 embeddings (from code predictor embeddings)
            for (i, &code) in predictor_codes.iter().enumerate() {
                if i < self.code_predictor.code_embeddings.len() {
                    let ct = Tensor::from_slice(&[code]).to_device(self.device);
                    let emb = self.code_predictor.code_embeddings[i]
                        .index_select(0, &ct)
                        .unsqueeze(0); // [1, 1, hidden_size]
                    code_embeds_sum = &code_embeds_sum + &emb;
                }
            }

            // Add trailing text (tts_pad for non-streaming mode)
            let next_input = &code_embeds_sum + tts_pad_embed;

            // Append to sequence
            full_sequence = Tensor::cat(&[full_sequence, next_input], 1);

            if step % 10 == 0 {
                println!("  Generated {} code frames", step + 1);
            }
        }

        all_codes
    }
}

/// TTS Inference engine.
pub struct TTSInference {
    /// Text tokenizer
    tokenizer: Tokenizer,
    /// Model weights (kept for reference)
    #[allow(dead_code)]
    weights: HashMap<String, Tensor>,
    /// Talker model
    talker: TalkerModel,
    /// Vocoder for decoding audio codes to waveform
    vocoder: Option<Vocoder>,
    /// Model configuration
    config: Qwen3TTSConfig,
    /// Device
    device: Device,
}

impl TTSInference {
    /// Create a new TTS inference engine.
    pub fn new(model_path: &Path, device: Device) -> Result<Self> {
        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Qwen3TTSError::Tokenization(e.to_string()))?
        } else {
            let vocab_path = model_path.join("vocab.json");
            let merges_path = model_path.join("merges.txt");

            if vocab_path.exists() && merges_path.exists() {
                use tokenizers::models::bpe::BPE;
                let bpe = BPE::from_file(
                    &vocab_path.to_string_lossy(),
                    &merges_path.to_string_lossy(),
                )
                .build()
                .map_err(|e| Qwen3TTSError::Tokenization(e.to_string()))?;
                Tokenizer::new(bpe)
            } else {
                return Err(Qwen3TTSError::Tokenization(
                    "No tokenizer.json or vocab.json found".into(),
                ));
            }
        };

        // Load model config
        let config_path = model_path.join("config.json");
        let config = Qwen3TTSConfig::from_file(&config_path)?;

        // Load weights
        let weights_path = model_path.join("model.safetensors");
        let tensors = Tensor::read_safetensors(&weights_path)?;
        let weights: HashMap<String, Tensor> = tensors
            .into_iter()
            .map(|(name, tensor)| (name, tensor.to_device(device)))
            .collect();

        println!("Loaded {} weight tensors", weights.len());

        // Load the Talker model
        let talker = TalkerModel::load(&weights, &config.talker_config, device)?;

        // Try to load vocoder
        let vocoder = {
            let vocoder_path = model_path
                .join("speech_tokenizer")
                .join("model.safetensors");
            if vocoder_path.exists() {
                println!("Loading vocoder from: {}", vocoder_path.display());
                match load_vocoder_weights(&vocoder_path, device) {
                    Ok(vocoder_weights) => {
                        let vocoder_config = VocoderConfig::default();
                        match Vocoder::load(&vocoder_weights, vocoder_config, device) {
                            Ok(v) => {
                                println!("Vocoder loaded successfully");
                                Some(v)
                            }
                            Err(e) => {
                                println!("Warning: Failed to load vocoder: {}", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        println!("Warning: Failed to load vocoder weights: {}", e);
                        None
                    }
                }
            } else {
                println!(
                    "No vocoder found at {}, using placeholder audio",
                    vocoder_path.display()
                );
                None
            }
        };

        Ok(Self {
            tokenizer,
            weights,
            talker,
            vocoder,
            config,
            device,
        })
    }

    /// Tokenize input text.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Qwen3TTSError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Generate speech from text.
    pub fn generate(&self, text: &str, speaker: &str, language: &str) -> Result<(Vec<f32>, u32)> {
        self.generate_with_params(text, speaker, language, 0.9, 50, 2048)
    }

    /// Generate speech with configurable parameters.
    pub fn generate_with_params(
        &self,
        text: &str,
        speaker: &str,
        language: &str,
        temperature: f64,
        top_k: i64,
        max_codes: i64,
    ) -> Result<(Vec<f32>, u32)> {
        // Get speaker ID
        let speaker_id = self
            .config
            .talker_config
            .spk_id
            .as_ref()
            .and_then(|map| map.get(&speaker.to_lowercase()))
            .copied()
            .unwrap_or(0) as i64;

        // Get language ID
        let language_id = self
            .config
            .talker_config
            .codec_language_id
            .as_ref()
            .and_then(|map| map.get(&language.to_lowercase()))
            .copied()
            .unwrap_or(0) as i64;

        // Get codec special token IDs from config
        let codec_eos_id = self.config.talker_config.codec_eos_token_id as i64;
        let codec_think_id = self.config.talker_config.codec_think_id as i64;
        let codec_think_bos_id = self.config.talker_config.codec_think_bos_id as i64;
        let codec_think_eos_id = self.config.talker_config.codec_think_eos_id as i64;
        let codec_pad_id = self.config.talker_config.codec_pad_id as i64;
        let codec_bos_id = self.config.talker_config.codec_bos_id as i64;
        let tts_pad_id = self.config.tts_pad_token_id as i64;
        let tts_bos_id = self.config.tts_bos_token_id as i64;
        let tts_eos_id = self.config.tts_eos_token_id as i64;

        println!(
            "Speaker: {} (id={}), Language: {} (id={})",
            speaker, speaker_id, language, language_id
        );
        println!(
            "Codec tokens: eos={}, think={}, think_bos={}, think_eos={}, pad={}, bos={}",
            codec_eos_id,
            codec_think_id,
            codec_think_bos_id,
            codec_think_eos_id,
            codec_pad_id,
            codec_bos_id
        );
        println!(
            "TTS tokens: pad={}, bos={}, eos={}",
            tts_pad_id, tts_bos_id, tts_eos_id
        );

        // Build token sequence manually to ensure special tokens are correct.
        // The tokenizers crate doesn't recognize <|im_start|> etc. as special tokens,
        // so we manually construct: [im_start, assistant, \n] + text + [im_end, \n, im_start, assistant, \n]
        let im_start = self.config.im_start_token_id as i64;
        let im_end = self.config.im_end_token_id as i64;
        let assistant_id = self.config.assistant_token_id as i64;

        // Tokenize just the text content
        let text_tokens = self.tokenize(text)?;
        let text_ids: Vec<i64> = text_tokens.iter().map(|&id| id as i64).collect();

        // Tokenize "\n" to get the newline token ID
        let newline_tokens = self.tokenize("\n")?;
        let newline_id = newline_tokens.first().copied().unwrap_or(198) as i64;

        // Build full sequence:
        // [<|im_start|>, assistant, \n] + text_tokens + [<|im_end|>, \n, <|im_start|>, assistant, \n]
        let mut token_ids_i64 = vec![im_start, assistant_id, newline_id];
        token_ids_i64.extend_from_slice(&text_ids);
        token_ids_i64.extend_from_slice(&[im_end, newline_id, im_start, assistant_id, newline_id]);

        println!(
            "Token sequence: {} tokens, first 3 = {:?}, text = {} tokens, last 5 = {:?}",
            token_ids_i64.len(),
            &token_ids_i64[..3],
            text_ids.len(),
            &token_ids_i64[token_ids_i64.len().saturating_sub(5)..],
        );

        // Build dual-stream input embeddings
        println!("Building input embeddings...");
        let input_embeddings = self.talker.build_input_embeddings(
            &token_ids_i64,
            speaker_id,
            language_id,
            tts_pad_id,
            tts_bos_id,
            tts_eos_id,
            codec_think_id,
            codec_think_bos_id,
            codec_think_eos_id,
            codec_pad_id,
            codec_bos_id,
        );

        // Pre-compute tts_pad embedding for trailing text during generation
        let tts_pad_embed = self.talker.embed_text(&[tts_pad_id]); // [1, 1, 1024]

        // Generate codes autoregressively
        println!(
            "Generating audio codes (temp={}, top_k={}, max={})...",
            temperature, top_k, max_codes
        );
        let codes = self.talker.generate_codes(
            &input_embeddings,
            max_codes,
            temperature,
            top_k,
            codec_eos_id,
            &tts_pad_embed,
        );
        println!("Generated {} code frames", codes.len());

        // Convert codes to audio using vocoder
        let sample_rate = 24000u32;
        let waveform: Vec<f32>;

        if codes.is_empty() {
            println!("Warning: No codes generated, returning silence");
            let num_samples = sample_rate as usize * 2;
            waveform = vec![0.0; num_samples];
        } else if let Some(ref vocoder) = self.vocoder {
            println!("Decoding audio with vocoder...");

            let num_frames = codes.len();
            let num_quantizers = codes[0].len();

            // Clamp codes to valid codebook range [0, 2047]
            let codebook_size = 2048i64;
            let mut codes_flat: Vec<i64> = Vec::with_capacity(num_quantizers * num_frames);
            let mut out_of_range = 0;
            for q in 0..num_quantizers {
                for frame in &codes {
                    let code = frame[q];
                    if code < 0 || code >= codebook_size {
                        out_of_range += 1;
                        codes_flat.push(code.clamp(0, codebook_size - 1));
                    } else {
                        codes_flat.push(code);
                    }
                }
            }
            if out_of_range > 0 {
                println!(
                    "Warning: {} codes out of range [0, {}), clamped",
                    out_of_range, codebook_size
                );
            }

            // Create tensor [num_quantizers, seq_len] then unsqueeze to [1, num_quantizers, seq_len]
            let codes_tensor = Tensor::from_slice(&codes_flat)
                .view([num_quantizers as i64, num_frames as i64])
                .unsqueeze(0)
                .to_device(self.device);

            println!("Codes tensor shape: {:?}", codes_tensor.size());

            // Decode with vocoder
            let audio_tensor = vocoder.decode(&codes_tensor);
            println!("Vocoder output shape: {:?}", audio_tensor.size());

            // Convert tensor to Vec<f32>
            let audio_len = audio_tensor.numel();
            waveform =
                Vec::<f32>::try_from(audio_tensor.view([audio_len as i64]).to_kind(Kind::Float))
                    .unwrap_or_else(|_| vec![0.0; audio_len]);

            println!("Decoded {} audio samples", waveform.len());
        } else {
            println!("Warning: Vocoder not loaded, using placeholder synthesis");
            let samples_per_frame = 200;
            let codebook_size = self.config.talker_config.code_predictor_config.vocab_size as f32;
            let mut samples = Vec::new();

            for (frame_idx, frame_codes) in codes.iter().enumerate() {
                let primary_code = frame_codes.first().copied().unwrap_or(0) as f32;
                let base_freq = 100.0 + (primary_code / codebook_size) * 800.0;

                for sample_idx in 0..samples_per_frame {
                    let t =
                        (frame_idx * samples_per_frame + sample_idx) as f32 / sample_rate as f32;
                    let sample = (2.0 * std::f32::consts::PI * base_freq * t).sin() * 0.3;
                    samples.push(sample);
                }
            }
            waveform = samples;
        }

        println!(
            "Generated {} samples ({:.2} seconds)",
            waveform.len(),
            waveform.len() as f64 / sample_rate as f64
        );

        Ok((waveform, sample_rate))
    }

    /// Get a reference to the loaded model weights.
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }

    /// Get a reference to the model configuration.
    pub fn config(&self) -> &Qwen3TTSConfig {
        &self.config
    }

    /// Get the computation device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Generate speech using a speaker embedding (x-vector) for voice cloning.
    ///
    /// Similar to `generate_with_params` but uses a speaker embedding tensor
    /// extracted from reference audio instead of a named speaker.
    pub fn generate_with_xvector(
        &self,
        text: &str,
        speaker_embedding: &Tensor,
        language: &str,
        temperature: f64,
        top_k: i64,
        max_codes: i64,
    ) -> Result<(Vec<f32>, u32)> {
        // Get language ID
        let language_id = self
            .config
            .talker_config
            .codec_language_id
            .as_ref()
            .and_then(|map| map.get(&language.to_lowercase()))
            .copied()
            .unwrap_or(0) as i64;

        // Get codec special token IDs from config
        let codec_eos_id = self.config.talker_config.codec_eos_token_id as i64;
        let codec_think_id = self.config.talker_config.codec_think_id as i64;
        let codec_think_bos_id = self.config.talker_config.codec_think_bos_id as i64;
        let codec_think_eos_id = self.config.talker_config.codec_think_eos_id as i64;
        let codec_pad_id = self.config.talker_config.codec_pad_id as i64;
        let codec_bos_id = self.config.talker_config.codec_bos_id as i64;
        let tts_pad_id = self.config.tts_pad_token_id as i64;
        let tts_bos_id = self.config.tts_bos_token_id as i64;
        let tts_eos_id = self.config.tts_eos_token_id as i64;

        println!("Voice clone: Language: {} (id={})", language, language_id);

        // Build token sequence
        let im_start = self.config.im_start_token_id as i64;
        let im_end = self.config.im_end_token_id as i64;
        let assistant_id = self.config.assistant_token_id as i64;

        let text_tokens = self.tokenize(text)?;
        let text_ids: Vec<i64> = text_tokens.iter().map(|&id| id as i64).collect();

        let newline_tokens = self.tokenize("\n")?;
        let newline_id = newline_tokens.first().copied().unwrap_or(198) as i64;

        let mut token_ids_i64 = vec![im_start, assistant_id, newline_id];
        token_ids_i64.extend_from_slice(&text_ids);
        token_ids_i64.extend_from_slice(&[im_end, newline_id, im_start, assistant_id, newline_id]);

        println!(
            "Token sequence: {} tokens, text = {} tokens",
            token_ids_i64.len(),
            text_ids.len(),
        );

        // Build dual-stream input embeddings with speaker embedding
        println!("Building input embeddings with speaker x-vector...");
        let input_embeddings = self.talker.build_input_embeddings_with_xvector(
            &token_ids_i64,
            speaker_embedding,
            language_id,
            tts_pad_id,
            tts_bos_id,
            tts_eos_id,
            codec_think_id,
            codec_think_bos_id,
            codec_think_eos_id,
            codec_pad_id,
            codec_bos_id,
        );

        // Pre-compute tts_pad embedding for trailing text during generation
        let tts_pad_embed = self.talker.embed_text(&[tts_pad_id]);

        // Generate codes autoregressively
        println!(
            "Generating audio codes (temp={}, top_k={}, max={})...",
            temperature, top_k, max_codes
        );
        let codes = self.talker.generate_codes(
            &input_embeddings,
            max_codes,
            temperature,
            top_k,
            codec_eos_id,
            &tts_pad_embed,
        );
        println!("Generated {} code frames", codes.len());

        // Convert codes to audio using vocoder (same as generate_with_params)
        let sample_rate = 24000u32;
        let waveform: Vec<f32>;

        if codes.is_empty() {
            println!("Warning: No codes generated, returning silence");
            let num_samples = sample_rate as usize * 2;
            waveform = vec![0.0; num_samples];
        } else if let Some(ref vocoder) = self.vocoder {
            println!("Decoding audio with vocoder...");

            let num_frames = codes.len();
            let num_quantizers = codes[0].len();

            let codebook_size = 2048i64;
            let mut codes_flat: Vec<i64> = Vec::with_capacity(num_quantizers * num_frames);
            let mut out_of_range = 0;
            for q in 0..num_quantizers {
                for frame in &codes {
                    let code = frame[q];
                    if code < 0 || code >= codebook_size {
                        out_of_range += 1;
                        codes_flat.push(code.clamp(0, codebook_size - 1));
                    } else {
                        codes_flat.push(code);
                    }
                }
            }
            if out_of_range > 0 {
                println!(
                    "Warning: {} codes out of range [0, {}), clamped",
                    out_of_range, codebook_size
                );
            }

            let codes_tensor = Tensor::from_slice(&codes_flat)
                .view([num_quantizers as i64, num_frames as i64])
                .unsqueeze(0)
                .to_device(self.device);

            println!("Codes tensor shape: {:?}", codes_tensor.size());

            let audio_tensor = vocoder.decode(&codes_tensor);
            println!("Vocoder output shape: {:?}", audio_tensor.size());

            let audio_len = audio_tensor.numel();
            waveform =
                Vec::<f32>::try_from(audio_tensor.view([audio_len as i64]).to_kind(Kind::Float))
                    .unwrap_or_else(|_| vec![0.0; audio_len]);

            println!("Decoded {} audio samples", waveform.len());
        } else {
            println!("Warning: Vocoder not loaded, generating placeholder audio");
            let num_samples = sample_rate as usize * 2;
            waveform = vec![0.0; num_samples];
        }

        println!(
            "Generated {} samples ({:.2} seconds)",
            waveform.len(),
            waveform.len() as f64 / sample_rate as f64
        );

        Ok((waveform, sample_rate))
    }

    /// Generate speech using ICL (In-Context Learning) voice cloning.
    ///
    /// This mode conditions on both the speaker embedding AND reference audio codec
    /// tokens + text transcription for higher fidelity voice cloning.
    ///
    /// After generation, reference codes are prepended to generated codes, decoded
    /// together by the vocoder, and the reference audio portion is trimmed.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_icl(
        &self,
        text: &str,
        ref_text: &str,
        ref_codes: &[Vec<i64>],
        speaker_embedding: &Tensor,
        language: &str,
        temperature: f64,
        top_k: i64,
        max_codes: i64,
    ) -> Result<(Vec<f32>, u32)> {
        // Get language ID
        let language_id = self
            .config
            .talker_config
            .codec_language_id
            .as_ref()
            .and_then(|map| map.get(&language.to_lowercase()))
            .copied()
            .unwrap_or(0) as i64;

        // Get codec special token IDs
        let codec_eos_id = self.config.talker_config.codec_eos_token_id as i64;
        let codec_think_id = self.config.talker_config.codec_think_id as i64;
        let codec_think_bos_id = self.config.talker_config.codec_think_bos_id as i64;
        let codec_think_eos_id = self.config.talker_config.codec_think_eos_id as i64;
        let codec_pad_id = self.config.talker_config.codec_pad_id as i64;
        let codec_bos_id = self.config.talker_config.codec_bos_id as i64;
        let tts_pad_id = self.config.tts_pad_token_id as i64;
        let tts_bos_id = self.config.tts_bos_token_id as i64;
        let tts_eos_id = self.config.tts_eos_token_id as i64;

        println!(
            "ICL voice clone: Language: {} (id={})",
            language, language_id
        );
        println!("  Reference text: \"{}\"", ref_text);
        println!("  Synthesis text: \"{}\"", text);
        println!("  Reference codec frames: {}", ref_codes.len());

        // Build token sequences for ref text and synth text
        let im_start = self.config.im_start_token_id as i64;
        let im_end = self.config.im_end_token_id as i64;
        let assistant_id = self.config.assistant_token_id as i64;

        let newline_tokens = self.tokenize("\n")?;
        let newline_id = newline_tokens.first().copied().unwrap_or(198) as i64;

        // Reference text: <|im_start|>assistant\n{ref_text}<|im_end|>\n
        let ref_text_tokens = self.tokenize(ref_text)?;
        let ref_text_ids: Vec<i64> = ref_text_tokens.iter().map(|&id| id as i64).collect();
        let mut ref_token_ids = vec![im_start, assistant_id, newline_id];
        ref_token_ids.extend_from_slice(&ref_text_ids);
        ref_token_ids.extend_from_slice(&[im_end, newline_id]);

        // Synthesis text: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        let synth_tokens = self.tokenize(text)?;
        let synth_ids: Vec<i64> = synth_tokens.iter().map(|&id| id as i64).collect();
        let mut synth_token_ids = vec![im_start, assistant_id, newline_id];
        synth_token_ids.extend_from_slice(&synth_ids);
        synth_token_ids.extend_from_slice(&[
            im_end,
            newline_id,
            im_start,
            assistant_id,
            newline_id,
        ]);

        println!(
            "  ref_text tokens: {}, synth_text tokens: {}",
            ref_token_ids.len(),
            synth_token_ids.len()
        );

        // Build ICL input embeddings
        println!("Building ICL input embeddings...");
        let input_embeddings = self.talker.build_input_embeddings_with_icl(
            &ref_token_ids,
            &synth_token_ids,
            ref_codes,
            speaker_embedding,
            language_id,
            tts_pad_id,
            tts_bos_id,
            tts_eos_id,
            codec_think_id,
            codec_think_bos_id,
            codec_think_eos_id,
            codec_pad_id,
            codec_bos_id,
        );

        // Pre-compute tts_pad embedding for trailing text during generation
        let tts_pad_embed = self.talker.embed_text(&[tts_pad_id]);

        // Generate codes autoregressively
        println!(
            "Generating audio codes (temp={}, top_k={}, max={})...",
            temperature, top_k, max_codes
        );
        let generated_codes = self.talker.generate_codes(
            &input_embeddings,
            max_codes,
            temperature,
            top_k,
            codec_eos_id,
            &tts_pad_embed,
        );
        println!("Generated {} code frames", generated_codes.len());

        // Prepend reference codes to generated codes for joint decoding
        let ref_len = ref_codes.len();
        let gen_len = generated_codes.len();
        let total_len = ref_len + gen_len;
        println!(
            "Concatenating: {} ref + {} generated = {} total frames",
            ref_len, gen_len, total_len
        );

        let mut all_codes = Vec::with_capacity(total_len);
        all_codes.extend_from_slice(ref_codes);
        all_codes.extend_from_slice(&generated_codes);

        // Decode with vocoder
        let sample_rate = 24000u32;
        let waveform: Vec<f32>;

        if all_codes.is_empty() || gen_len == 0 {
            println!("Warning: No codes generated, returning silence");
            let num_samples = sample_rate as usize * 2;
            waveform = vec![0.0; num_samples];
        } else if let Some(ref vocoder) = self.vocoder {
            println!("Decoding concatenated audio with vocoder...");

            let num_frames = all_codes.len();
            let num_quantizers = all_codes[0].len();
            let codebook_size = 2048i64;

            let mut codes_flat: Vec<i64> = Vec::with_capacity(num_quantizers * num_frames);
            let mut out_of_range = 0;
            for q in 0..num_quantizers {
                for frame in &all_codes {
                    let code = frame[q];
                    if code < 0 || code >= codebook_size {
                        out_of_range += 1;
                        codes_flat.push(code.clamp(0, codebook_size - 1));
                    } else {
                        codes_flat.push(code);
                    }
                }
            }
            if out_of_range > 0 {
                println!(
                    "Warning: {} codes out of range [0, {}), clamped",
                    out_of_range, codebook_size
                );
            }

            let codes_tensor = Tensor::from_slice(&codes_flat)
                .view([num_quantizers as i64, num_frames as i64])
                .unsqueeze(0)
                .to_device(self.device);

            let audio_tensor = vocoder.decode(&codes_tensor);

            let audio_len = audio_tensor.numel();
            let full_waveform =
                Vec::<f32>::try_from(audio_tensor.view([audio_len as i64]).to_kind(Kind::Float))
                    .unwrap_or_else(|_| vec![0.0; audio_len]);

            // Trim reference portion from output
            // cut = ref_len / total_len * waveform_len
            let cut =
                (ref_len as f64 / total_len.max(1) as f64 * full_waveform.len() as f64) as usize;
            println!(
                "Trimming reference portion: cut {} of {} samples",
                cut,
                full_waveform.len()
            );
            waveform = full_waveform[cut..].to_vec();

            println!("Final output: {} samples", waveform.len());
        } else {
            println!("Warning: Vocoder not loaded, generating placeholder audio");
            let num_samples = sample_rate as usize * 2;
            waveform = vec![0.0; num_samples];
        }

        println!(
            "Generated {} samples ({:.2} seconds)",
            waveform.len(),
            waveform.len() as f64 / sample_rate as f64
        );

        Ok((waveform, sample_rate))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_inference_placeholder() {
        assert!(true);
    }
}
