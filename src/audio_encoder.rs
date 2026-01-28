// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Speech tokenizer encoder for converting audio waveforms to discrete codec tokens.
//!
//! This module implements the Mimi/EnCodec-style encoder used by Qwen3 TTS to encode
//! reference audio for ICL (In-Context Learning) voice cloning mode.
//!
//! Architecture: Conv Encoder (960x downsample) → Transformer (8 layers) → 2x Downsample → RVQ Encode
//! Total: 1920x downsample → 12.5 Hz frame rate at 24kHz input

use crate::error::{Qwen3TTSError, Result};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// Configuration for the audio encoder.
#[derive(Debug, Clone)]
pub struct AudioEncoderConfig {
    /// Input sample rate
    pub sample_rate: u32,
    /// Number of encoder transformer layers
    pub num_transformer_layers: usize,
    /// Transformer hidden size
    pub hidden_size: i64,
    /// Number of attention heads
    pub num_heads: i64,
    /// Intermediate size for FFN
    pub intermediate_size: i64,
    /// Codebook dimension
    pub codebook_dim: i64,
    /// Codebook size
    pub codebook_size: i64,
    /// Number of semantic quantizer layers
    pub num_semantic_quantizers: usize,
    /// Number of acoustic quantizer layers to use
    pub num_acoustic_quantizers: usize,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            num_transformer_layers: 8,
            hidden_size: 512,
            num_heads: 8,
            intermediate_size: 2048,
            codebook_dim: 256,
            codebook_size: 2048,
            num_semantic_quantizers: 1,
            num_acoustic_quantizers: 15,
        }
    }
}

/// A simple Conv1d layer with causal padding.
struct CausalConv1d {
    weight: Tensor,
    bias: Tensor,
    stride: i64,
    kernel_size: i64,
}

impl CausalConv1d {
    fn from_weights(weight: Tensor, bias: Tensor) -> Self {
        let kernel_size = weight.size()[2];
        let stride = 1; // Default; overridden by strided convs
        Self {
            weight,
            bias,
            stride,
            kernel_size,
        }
    }

    fn with_stride(mut self, stride: i64) -> Self {
        self.stride = stride;
        self
    }

    /// Forward pass with causal (left) padding.
    fn forward(&self, x: &Tensor) -> Tensor {
        let padding_total = self.kernel_size - self.stride;
        if padding_total > 0 {
            let padded = x.constant_pad_nd([padding_total, 0]);
            padded.conv1d(&self.weight, Some(&self.bias), [self.stride], [0], [1], 1)
        } else {
            x.conv1d(&self.weight, Some(&self.bias), [self.stride], [0], [1], 1)
        }
    }
}

/// Residual block: ELU → Conv1d(dim→dim/2, k=3) → ELU → Conv1d(dim/2→dim, k=1) + identity shortcut.
struct EncoderResBlock {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
}

impl EncoderResBlock {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        device: Device,
    ) -> Option<Self> {
        let conv1_weight = weights
            .get(&format!("{}.block.1.conv.weight", prefix))?
            .to_device(device);
        let conv1_bias = weights
            .get(&format!("{}.block.1.conv.bias", prefix))?
            .to_device(device);
        let conv2_weight = weights
            .get(&format!("{}.block.3.conv.weight", prefix))?
            .to_device(device);
        let conv2_bias = weights
            .get(&format!("{}.block.3.conv.bias", prefix))?
            .to_device(device);
        Some(Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // ELU → Conv1d(dim→dim/2, k=3, causal pad=2) → ELU → Conv1d(dim/2→dim, k=1, no pad)
        let h = x.elu();
        let h = h.constant_pad_nd([2, 0]); // causal pad for k=3, stride=1
        let h = h.conv1d(&self.conv1_weight, Some(&self.conv1_bias), [1], [0], [1], 1);
        let h = h.elu();
        let h = h.conv1d(&self.conv2_weight, Some(&self.conv2_bias), [1], [0], [1], 1);
        // Identity shortcut
        x + h
    }
}

/// Encoder transformer layer with LayerNorm, self-attention, LayerScale, and FFN.
struct EncoderTransformerLayer {
    // Pre-attention LayerNorm
    input_ln_weight: Tensor,
    input_ln_bias: Tensor,
    // Self-attention
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    // LayerScale after attention
    attn_layer_scale: Tensor,
    // Post-attention LayerNorm
    post_ln_weight: Tensor,
    post_ln_bias: Tensor,
    // FFN
    fc1_weight: Tensor,
    fc2_weight: Tensor,
    // LayerScale after MLP
    mlp_layer_scale: Tensor,
    // Config
    num_heads: i64,
    head_dim: i64,
}

impl EncoderTransformerLayer {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: i64,
        device: Device,
    ) -> Option<Self> {
        let hidden_size = weights
            .get(&format!("{}.input_layernorm.weight", prefix))?
            .size()[0];
        let head_dim = hidden_size / num_heads;

        Some(Self {
            input_ln_weight: weights
                .get(&format!("{}.input_layernorm.weight", prefix))?
                .to_device(device),
            input_ln_bias: weights
                .get(&format!("{}.input_layernorm.bias", prefix))?
                .to_device(device),
            q_proj: weights
                .get(&format!("{}.self_attn.q_proj.weight", prefix))?
                .to_device(device),
            k_proj: weights
                .get(&format!("{}.self_attn.k_proj.weight", prefix))?
                .to_device(device),
            v_proj: weights
                .get(&format!("{}.self_attn.v_proj.weight", prefix))?
                .to_device(device),
            o_proj: weights
                .get(&format!("{}.self_attn.o_proj.weight", prefix))?
                .to_device(device),
            attn_layer_scale: weights
                .get(&format!("{}.self_attn_layer_scale.scale", prefix))?
                .to_device(device),
            post_ln_weight: weights
                .get(&format!("{}.post_attention_layernorm.weight", prefix))?
                .to_device(device),
            post_ln_bias: weights
                .get(&format!("{}.post_attention_layernorm.bias", prefix))?
                .to_device(device),
            fc1_weight: weights
                .get(&format!("{}.mlp.fc1.weight", prefix))?
                .to_device(device),
            fc2_weight: weights
                .get(&format!("{}.mlp.fc2.weight", prefix))?
                .to_device(device),
            mlp_layer_scale: weights
                .get(&format!("{}.mlp_layer_scale.scale", prefix))?
                .to_device(device),
            num_heads,
            head_dim,
        })
    }

    fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
        let mean = x.mean_dim(&[-1i64][..], true, Kind::Float);
        let var = x.var_dim(&[-1i64][..], false, true);
        let normalized = (x - &mean) / (var + 1e-5).sqrt();
        &normalized * weight + bias
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [batch, seq_len, hidden_size]
        let batch = x.size()[0];
        let seq_len = x.size()[1];

        // Pre-attention LayerNorm
        let normed = Self::layer_norm(x, &self.input_ln_weight, &self.input_ln_bias);

        // Self-attention
        let q = normed
            .matmul(&self.q_proj.tr())
            .view([batch, seq_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [B, H, T, D]
        let k = normed
            .matmul(&self.k_proj.tr())
            .view([batch, seq_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = normed
            .matmul(&self.v_proj.tr())
            .view([batch, seq_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(-2, -1)) / scale;
        let attn_weights = attn_weights.softmax(-1, Kind::Float);
        let attn_output = attn_weights
            .matmul(&v)
            .permute([0, 2, 1, 3])
            .contiguous()
            .view([batch, seq_len, -1]);
        let attn_output = attn_output.matmul(&self.o_proj.tr());

        // LayerScale + residual
        let h = x + attn_output * &self.attn_layer_scale;

        // Post-attention LayerNorm
        let normed2 = Self::layer_norm(&h, &self.post_ln_weight, &self.post_ln_bias);

        // FFN: fc1 → GELU → fc2
        let mlp_out = normed2.matmul(&self.fc1_weight.tr()).gelu("none");
        let mlp_out = mlp_out.matmul(&self.fc2_weight.tr());

        // LayerScale + residual
        h + mlp_out * &self.mlp_layer_scale
    }
}

/// Codebook for encoding (nearest neighbor search).
struct EncoderCodebook {
    /// Normalized embeddings [codebook_size, dim]
    embeddings: Tensor,
}

impl EncoderCodebook {
    fn from_weights(embed_sum: &Tensor, cluster_usage: &Tensor) -> Self {
        let eps = 1e-5;
        let usage = cluster_usage.clamp_min(eps).unsqueeze(-1);
        let embeddings = embed_sum / &usage;
        Self { embeddings }
    }

    /// Encode: find nearest codebook vector for each input frame.
    /// x: [batch, seq_len, dim]
    /// Returns: codes [batch, seq_len] and quantized [batch, seq_len, dim]
    fn encode(&self, x: &Tensor) -> (Tensor, Tensor) {
        // Compute squared distances: ||x - e||² = ||x||² - 2*x·e + ||e||²
        let x_sq = x
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float); // [B, T, 1]
        let e_sq = self
            .embeddings
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
            .tr(); // [1, codebook_size]
        let dot = x.matmul(&self.embeddings.tr()); // [B, T, codebook_size]
        let distances: Tensor = &x_sq - 2.0 * &dot + &e_sq; // [B, T, codebook_size]

        let codes = distances.argmin(-1, false); // [B, T]

        // Look up quantized values
        let quantized =
            Tensor::embedding(&self.embeddings, &codes.view([-1]), -1, false, false).view_as(x);

        (codes, quantized)
    }
}

/// Single VQ layer with input/output projections.
struct EncoderVQLayer {
    codebook: EncoderCodebook,
}

/// Residual Vector Quantizer for encoding.
struct EncoderRVQ {
    /// Input projection (1x1 conv as linear)
    input_proj_weight: Tensor,
    /// Output projection (used in decode direction, kept for weight loading completeness)
    #[allow(dead_code)]
    output_proj_weight: Tensor,
    /// VQ layers
    layers: Vec<EncoderVQLayer>,
    /// Number of layers to actually use
    num_valid_layers: usize,
}

impl EncoderRVQ {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_layers: usize,
        num_valid: usize,
        device: Device,
    ) -> Option<Self> {
        let input_proj_weight = weights
            .get(&format!("{}.input_proj.weight", prefix))?
            .squeeze_dim(-1)
            .to_device(device);
        let output_proj_weight = weights
            .get(&format!("{}.output_proj.weight", prefix))?
            .squeeze_dim(-1)
            .to_device(device);

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let embed_sum = weights
                .get(&format!("{}.layers.{}.codebook.embed_sum", prefix, i))?
                .to_device(device);
            let cluster_usage = weights
                .get(&format!("{}.layers.{}.codebook.cluster_usage", prefix, i))?
                .to_device(device);
            layers.push(EncoderVQLayer {
                codebook: EncoderCodebook::from_weights(&embed_sum, &cluster_usage),
            });
        }

        Some(Self {
            input_proj_weight,
            output_proj_weight,
            layers,
            num_valid_layers: num_valid,
        })
    }

    /// Encode features to codes using residual vector quantization.
    /// x: [batch, dim, seq_len] (conv format)
    /// Returns: codes [batch, num_valid_layers, seq_len]
    fn encode(&self, x: &Tensor) -> Tensor {
        let batch = x.size()[0];
        let seq_len = x.size()[2];

        // Input projection: [B, dim, T] → [B, T, dim] → matmul → [B, T, codebook_dim]
        let x_transposed = x.transpose(1, 2); // [B, T, dim]
        let mut residual = x_transposed.matmul(&self.input_proj_weight.tr()); // [B, T, codebook_dim]

        let num_use = self.num_valid_layers.min(self.layers.len());
        let mut all_codes = Vec::with_capacity(num_use);

        for layer in self.layers.iter().take(num_use) {
            let (codes, quantized) = layer.codebook.encode(&residual);
            all_codes.push(codes); // [B, T]
            residual = &residual - &quantized;
        }

        // Stack: list of [B, T] → [num_layers, B, T] → [B, num_layers, T]
        Tensor::stack(&all_codes, 0)
            .permute([1, 0, 2])
            .contiguous()
            .view([batch, num_use as i64, seq_len])
    }
}

/// The complete audio encoder.
pub struct AudioEncoder {
    // Conv encoder layers
    conv_layers: Vec<EncoderConvLayer>,
    // Transformer layers
    transformer_layers: Vec<EncoderTransformerLayer>,
    // Downsample
    downsample_weight: Tensor,
    // Quantizers
    semantic_rvq: EncoderRVQ,
    acoustic_rvq: EncoderRVQ,
    // Config
    config: AudioEncoderConfig,
    device: Device,
}

/// Types of conv encoder layers.
enum EncoderConvLayer {
    /// Regular convolution (with optional stride)
    Conv(CausalConv1d),
    /// Residual block
    ResBlock(EncoderResBlock),
    /// ELU activation (no parameters)
    Elu,
}

impl AudioEncoder {
    /// Load the audio encoder from a safetensors file.
    pub fn load(weights_path: &Path, device: Device) -> Result<Self> {
        let config = AudioEncoderConfig::default();

        println!("Loading audio encoder from: {}", weights_path.display());
        let tensors = Tensor::read_safetensors(weights_path)?;
        let weights: HashMap<String, Tensor> = tensors
            .into_iter()
            .map(|(name, tensor)| (name, tensor.to_device(device)))
            .collect();

        // Count encoder keys
        let enc_count = weights.keys().filter(|k| k.starts_with("encoder.")).count();
        println!("  Found {} encoder weight tensors", enc_count);

        // Load conv encoder layers (0-14)
        let mut conv_layers = Vec::new();

        // Layer 0: Conv1d(1→64, k=7, stride=1)
        let w = weights
            .get("encoder.encoder.layers.0.conv.weight")
            .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing encoder conv layer 0".into()))?
            .shallow_clone();
        let b = weights
            .get("encoder.encoder.layers.0.conv.bias")
            .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing encoder conv layer 0 bias".into()))?
            .shallow_clone();
        conv_layers.push(EncoderConvLayer::Conv(CausalConv1d::from_weights(w, b)));

        // Layer 1: ResBlock(64)
        conv_layers.push(EncoderConvLayer::ResBlock(
            EncoderResBlock::from_weights(&weights, "encoder.encoder.layers.1", device)
                .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing resblock 1".into()))?,
        ));

        // Layer 2: ELU
        conv_layers.push(EncoderConvLayer::Elu);

        // Layer 3: Conv1d(64→128, k=8, stride=4)
        let w = weights
            .get("encoder.encoder.layers.3.conv.weight")
            .unwrap()
            .shallow_clone();
        let b = weights
            .get("encoder.encoder.layers.3.conv.bias")
            .unwrap()
            .shallow_clone();
        conv_layers.push(EncoderConvLayer::Conv(
            CausalConv1d::from_weights(w, b).with_stride(4),
        ));

        // Layer 4: ResBlock(128)
        conv_layers.push(EncoderConvLayer::ResBlock(
            EncoderResBlock::from_weights(&weights, "encoder.encoder.layers.4", device)
                .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing resblock 4".into()))?,
        ));

        // Layer 5: ELU
        conv_layers.push(EncoderConvLayer::Elu);

        // Layer 6: Conv1d(128→256, k=10, stride=5)
        let w = weights
            .get("encoder.encoder.layers.6.conv.weight")
            .unwrap()
            .shallow_clone();
        let b = weights
            .get("encoder.encoder.layers.6.conv.bias")
            .unwrap()
            .shallow_clone();
        conv_layers.push(EncoderConvLayer::Conv(
            CausalConv1d::from_weights(w, b).with_stride(5),
        ));

        // Layer 7: ResBlock(256)
        conv_layers.push(EncoderConvLayer::ResBlock(
            EncoderResBlock::from_weights(&weights, "encoder.encoder.layers.7", device)
                .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing resblock 7".into()))?,
        ));

        // Layer 8: ELU
        conv_layers.push(EncoderConvLayer::Elu);

        // Layer 9: Conv1d(256→512, k=12, stride=6)
        let w = weights
            .get("encoder.encoder.layers.9.conv.weight")
            .unwrap()
            .shallow_clone();
        let b = weights
            .get("encoder.encoder.layers.9.conv.bias")
            .unwrap()
            .shallow_clone();
        conv_layers.push(EncoderConvLayer::Conv(
            CausalConv1d::from_weights(w, b).with_stride(6),
        ));

        // Layer 10: ResBlock(512)
        conv_layers.push(EncoderConvLayer::ResBlock(
            EncoderResBlock::from_weights(&weights, "encoder.encoder.layers.10", device)
                .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing resblock 10".into()))?,
        ));

        // Layer 11: ELU
        conv_layers.push(EncoderConvLayer::Elu);

        // Layer 12: Conv1d(512→1024, k=16, stride=8)
        let w = weights
            .get("encoder.encoder.layers.12.conv.weight")
            .unwrap()
            .shallow_clone();
        let b = weights
            .get("encoder.encoder.layers.12.conv.bias")
            .unwrap()
            .shallow_clone();
        conv_layers.push(EncoderConvLayer::Conv(
            CausalConv1d::from_weights(w, b).with_stride(8),
        ));

        // Layer 13: ELU
        conv_layers.push(EncoderConvLayer::Elu);

        // Layer 14: Conv1d(1024→512, k=3, stride=1)
        let w = weights
            .get("encoder.encoder.layers.14.conv.weight")
            .unwrap()
            .shallow_clone();
        let b = weights
            .get("encoder.encoder.layers.14.conv.bias")
            .unwrap()
            .shallow_clone();
        conv_layers.push(EncoderConvLayer::Conv(CausalConv1d::from_weights(w, b)));

        println!("  Loaded {} conv encoder layers", conv_layers.len());

        // Load transformer layers
        let mut transformer_layers = Vec::new();
        for i in 0..config.num_transformer_layers {
            let prefix = format!("encoder.encoder_transformer.layers.{}", i);
            let layer =
                EncoderTransformerLayer::from_weights(&weights, &prefix, config.num_heads, device)
                    .ok_or_else(|| {
                        Qwen3TTSError::ModelLoad(format!(
                            "Failed to load encoder transformer layer {}",
                            i
                        ))
                    })?;
            transformer_layers.push(layer);
        }
        println!("  Loaded {} transformer layers", transformer_layers.len());

        // Load downsample
        let downsample_weight = weights
            .get("encoder.downsample.conv.weight")
            .ok_or_else(|| Qwen3TTSError::ModelLoad("Missing encoder downsample".into()))?
            .shallow_clone();
        println!("  Loaded downsample: {:?}", downsample_weight.size());

        // Load quantizers
        let semantic_rvq = EncoderRVQ::from_weights(
            &weights,
            "encoder.quantizer.semantic_residual_vector_quantizer",
            1, // 1 semantic layer
            1,
            device,
        )
        .ok_or_else(|| Qwen3TTSError::ModelLoad("Failed to load semantic RVQ".into()))?;
        println!(
            "  Loaded semantic RVQ: {} layers",
            semantic_rvq.layers.len()
        );

        let acoustic_rvq = EncoderRVQ::from_weights(
            &weights,
            "encoder.quantizer.acoustic_residual_vector_quantizer",
            31, // 31 total acoustic layers
            15, // use first 15
            device,
        )
        .ok_or_else(|| Qwen3TTSError::ModelLoad("Failed to load acoustic RVQ".into()))?;
        println!(
            "  Loaded acoustic RVQ: {} layers (using {})",
            acoustic_rvq.layers.len(),
            acoustic_rvq.num_valid_layers
        );

        Ok(Self {
            conv_layers,
            transformer_layers,
            downsample_weight,
            semantic_rvq,
            acoustic_rvq,
            config,
            device,
        })
    }

    /// Encode audio waveform to codec tokens.
    ///
    /// Input: f32 audio samples at 24kHz
    /// Output: Vec of frames, each frame is 16 codec codes (1 semantic + 15 acoustic)
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<i64>>> {
        let waveform = Tensor::from_slice(samples)
            .to_device(self.device)
            .to_kind(Kind::Float)
            .unsqueeze(0) // [1, T]
            .unsqueeze(0); // [1, 1, T]

        println!("  Audio encoder input: {:?}", waveform.size());

        // Conv encoder
        let mut h = waveform;
        for layer in &self.conv_layers {
            h = match layer {
                EncoderConvLayer::Conv(conv) => conv.forward(&h),
                EncoderConvLayer::ResBlock(block) => block.forward(&h),
                EncoderConvLayer::Elu => h.elu(),
            };
        }
        println!("  After conv encoder: {:?}", h.size());

        // Transpose for transformer: [B, C, T] → [B, T, C]
        let mut h = h.transpose(1, 2);
        println!("  Before transformer: {:?}", h.size());

        // Transformer
        for layer in &self.transformer_layers {
            h = layer.forward(&h);
        }

        // Back to conv format: [B, T, C] → [B, C, T]
        let h = h.transpose(1, 2);
        println!("  After transformer: {:?}", h.size());

        // Downsample: Conv1d(512→512, k=4, stride=2) with causal padding
        let pad = 4 - 2; // kernel_size - stride = 2
        let h = h.constant_pad_nd([pad, 0]);
        let h = h.conv1d(&self.downsample_weight, None::<&Tensor>, [2], [0], [1], 1);
        println!("  After downsample: {:?}", h.size());

        // Quantize
        // Semantic: 1 code per frame
        let semantic_codes = self.semantic_rvq.encode(&h); // [1, 1, T]
                                                           // Acoustic: 15 codes per frame
        let acoustic_codes = self.acoustic_rvq.encode(&h); // [1, 15, T]

        // Concatenate: [1, 16, T]
        let all_codes = Tensor::cat(&[semantic_codes, acoustic_codes], 1);
        println!("  Encoded codes: {:?}", all_codes.size());

        // Convert to Vec<Vec<i64>>: frames × 16
        let num_quantizers = all_codes.size()[1] as usize;
        let num_frames = all_codes.size()[2] as usize;

        let mut frames = Vec::with_capacity(num_frames);
        for t in 0..num_frames {
            let mut frame_codes = Vec::with_capacity(num_quantizers);
            for q in 0..num_quantizers {
                let code = all_codes.int64_value(&[0, q as i64, t as i64]);
                frame_codes.push(code);
            }
            frames.push(frame_codes);
        }

        println!(
            "  Encoded {} frames × {} quantizers ({:.2} Hz)",
            num_frames,
            num_quantizers,
            self.config.sample_rate as f64 * num_frames as f64 / samples.len() as f64
        );

        Ok(frames)
    }
}
