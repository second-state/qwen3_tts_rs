// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! ECAPA-TDNN speaker encoder for extracting speaker embeddings (x-vectors).
//!
//! Processes reference audio through mel-spectrogram extraction and an
//! ECAPA-TDNN network to produce a 1024-dimensional speaker embedding
//! used for voice cloning with the Base model.

use crate::config::SpeakerEncoderConfig;
use crate::error::{Qwen3TTSError, Result};
use std::collections::HashMap;
use crate::tensor::{Tensor, Device, DType};

// Mel spectrogram parameters (from Qwen3 TTS Python reference)
const MEL_N_FFT: i64 = 1024;
const MEL_HOP_SIZE: i64 = 256;
const MEL_WIN_SIZE: i64 = 1024;
const MEL_N_MELS: i64 = 128;
const MEL_SAMPLE_RATE: i64 = 24000;
const MEL_FMIN: f64 = 0.0;
const MEL_FMAX: f64 = 12000.0;

/// 1D convolution layer (weight + bias) stored as tensors.
struct Conv1d {
    weight: Tensor, // [out_channels, in_channels, kernel_size]
    bias: Tensor,   // [out_channels]
}

impl Conv1d {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str, device: Device) -> Result<Self> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .ok_or_else(|| Qwen3TTSError::ModelLoad(format!("Missing {}.weight", prefix)))?
            .to_device(device)
            .to_dtype(DType::Float32);
        let bias = weights
            .get(&format!("{}.bias", prefix))
            .ok_or_else(|| Qwen3TTSError::ModelLoad(format!("Missing {}.bias", prefix)))?
            .to_device(device)
            .to_dtype(DType::Float32);
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor, dilation: i64, groups: i64) -> Tensor {
        let kernel_size = self.weight.size()[2];
        // Calculate padding to maintain sequence length (same padding with reflect mode)
        let padding = (kernel_size - 1) * dilation / 2;
        // Apply reflect padding first, then convolution with no padding
        // This matches Python's nn.Conv1d(padding="same", padding_mode="reflect")
        let padded = x.reflection_pad1d(&[padding, padding]);
        padded.conv1d(&self.weight, Some(&self.bias), &[1], &[0], &[dilation], groups)
    }

    fn forward_no_pad(&self, x: &Tensor) -> Tensor {
        x.conv1d(&self.weight, Some(&self.bias), &[1], &[0], &[1], 1)
    }
}

/// Res2Net block: splits channels into groups with cascaded dilated convolutions.
struct Res2NetBlock {
    blocks: Vec<Conv1d>,
    scale: usize,
}

impl Res2NetBlock {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        scale: usize,
        device: Device,
    ) -> Result<Self> {
        // scale=8 means 8 groups; first is identity, so 7 convolution blocks
        let num_convs = scale - 1;
        let mut blocks = Vec::with_capacity(num_convs);
        for i in 0..num_convs {
            let conv = Conv1d::load(weights, &format!("{}.blocks.{}.conv", prefix, i), device)?;
            blocks.push(conv);
        }
        Ok(Self { blocks, scale })
    }

    fn forward(&self, x: &Tensor, dilation: i64) -> Tensor {
        let channels = x.size()[1];
        let group_size = channels / self.scale as i64;

        // Split into groups along channel dimension
        let groups: Vec<Tensor> = (0..self.scale as i64)
            .map(|i| x.narrow(1, i * group_size, group_size))
            .collect();

        let mut outputs = Vec::with_capacity(self.scale);

        // First group: identity
        outputs.push(groups[0].shallow_clone());

        // Remaining groups: cascaded dilated convolutions
        for i in 1..self.scale {
            let input = if i == 1 {
                groups[i].shallow_clone()
            } else {
                &groups[i] + &outputs[i - 1]
            };
            let out = self.blocks[i - 1].forward(&input, dilation, 1).relu();
            outputs.push(out);
        }

        Tensor::cat(&outputs, 1)
    }
}

/// Squeeze-Excitation block for channel attention.
struct SEBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SEBlock {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str, device: Device) -> Result<Self> {
        let conv1 = Conv1d::load(weights, &format!("{}.conv1", prefix), device)?;
        let conv2 = Conv1d::load(weights, &format!("{}.conv2", prefix), device)?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // Global average pooling over time
        let s = x.mean_dim(&[2i64], true);
        let s = self.conv1.forward_no_pad(&s).relu();
        let s = self.conv2.forward_no_pad(&s).sigmoid();
        x * &s
    }
}

/// SE-Res2Net block: TDNN → Res2Net → TDNN → SE with residual.
struct SERes2NetBlock {
    tdnn1: Conv1d,
    res2net_block: Res2NetBlock,
    tdnn2: Conv1d,
    se_block: SEBlock,
}

impl SERes2NetBlock {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        scale: usize,
        device: Device,
    ) -> Result<Self> {
        let tdnn1 = Conv1d::load(weights, &format!("{}.tdnn1.conv", prefix), device)?;
        let res2net_block =
            Res2NetBlock::load(weights, &format!("{}.res2net_block", prefix), scale, device)?;
        let tdnn2 = Conv1d::load(weights, &format!("{}.tdnn2.conv", prefix), device)?;
        let se_block = SEBlock::load(weights, &format!("{}.se_block", prefix), device)?;
        Ok(Self {
            tdnn1,
            res2net_block,
            tdnn2,
            se_block,
        })
    }

    fn forward(&self, x: &Tensor, dilation: i64) -> Tensor {
        let residual = x.shallow_clone();
        let h = self.tdnn1.forward(x, 1, 1).relu();
        let h = self.res2net_block.forward(&h, dilation);
        let h = self.tdnn2.forward(&h, 1, 1).relu();
        let h = self.se_block.forward(&h);
        &h + &residual
    }
}

/// Attentive Statistics Pooling.
struct Asp {
    tdnn: Conv1d,
    conv: Conv1d,
}

impl Asp {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str, device: Device) -> Result<Self> {
        let tdnn = Conv1d::load(weights, &format!("{}.tdnn.conv", prefix), device)?;
        let conv = Conv1d::load(weights, &format!("{}.conv", prefix), device)?;
        Ok(Self { tdnn, conv })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [B, C, T]
        let mean = x.mean_dim(&[2i64], true); // [B, C, 1]
        let std = x.std_dim(&[2i64], true, true); // [B, C, 1]

        // Expand mean and std to time dimension
        let t = x.size()[2];
        let mean_expanded = mean.expand_as(x);
        let std_expanded = std.expand(&[-1, -1, t], false);

        // Concatenate input, mean, std → [B, 3C, T]
        let attn_input = Tensor::cat(&[x.shallow_clone(), mean_expanded, std_expanded], 1);

        // Attention network
        let attn = self.tdnn.forward_no_pad(&attn_input).tanh();
        let attn = self.conv.forward_no_pad(&attn).softmax(2); // [B, C, T]

        // Weighted statistics
        let weighted_mean = (x * &attn).sum_dim(&[2i64], false); // [B, C]
        let weighted_var = ((x - &weighted_mean.unsqueeze(2)).pow_scalar(2.0) * &attn)
            .sum_dim(&[2i64], false); // [B, C]
        let weighted_std = (weighted_var + 1e-6).sqrt();

        // Concatenate mean and std → [B, 2C]
        Tensor::cat(&[weighted_mean, weighted_std], 1)
    }
}

/// ECAPA-TDNN speaker encoder.
///
/// Extracts 1024-dimensional speaker embeddings (x-vectors) from audio.
pub struct SpeakerEncoder {
    initial_conv: Conv1d,
    se_res2net_blocks: Vec<SERes2NetBlock>,
    mfa_conv: Conv1d,
    asp: Asp,
    fc: Conv1d,
    mel_basis: Tensor,
    hann_window: Tensor,
    device: Device,
    dilations: Vec<i64>,
}

impl SpeakerEncoder {
    /// Load speaker encoder from model weights.
    pub fn load(
        weights: &HashMap<String, Tensor>,
        config: &SpeakerEncoderConfig,
        device: Device,
    ) -> Result<Self> {
        println!("Loading SpeakerEncoder...");

        // Load initial convolution
        let initial_conv = Conv1d::load(weights, "speaker_encoder.blocks.0.conv", device)?;
        println!("  Loaded initial conv");

        // Load SE-Res2Net blocks (blocks 1, 2, 3)
        let mut se_res2net_blocks = Vec::new();
        let dilations: Vec<i64> = config.enc_dilations[1..4]
            .iter()
            .map(|&d| d as i64)
            .collect();

        for i in 1..=3 {
            let block = SERes2NetBlock::load(
                weights,
                &format!("speaker_encoder.blocks.{}", i),
                config.enc_res2net_scale,
                device,
            )?;
            se_res2net_blocks.push(block);
        }
        println!("  Loaded {} SE-Res2Net blocks", se_res2net_blocks.len());

        // Load MFA convolution
        let mfa_conv = Conv1d::load(weights, "speaker_encoder.mfa.conv", device)?;
        println!("  Loaded MFA conv");

        // Load Asp
        let asp = Asp::load(weights, "speaker_encoder.asp", device)?;
        println!("  Loaded Asp");

        // Load final FC
        let fc = Conv1d::load(weights, "speaker_encoder.fc", device)?;
        println!("  Loaded FC");

        // Pre-compute mel filterbank and Hann window
        let mel_basis = create_mel_filterbank(
            MEL_SAMPLE_RATE as f64,
            MEL_N_FFT,
            MEL_N_MELS,
            MEL_FMIN,
            MEL_FMAX,
            device,
        );
        let hann_window = Tensor::hann_window(MEL_WIN_SIZE, device);

        println!("SpeakerEncoder loaded successfully");

        Ok(Self {
            initial_conv,
            se_res2net_blocks,
            mfa_conv,
            asp,
            fc,
            mel_basis,
            hann_window,
            device,
            dilations,
        })
    }

    /// Extract speaker embedding from audio samples.
    ///
    /// Input: f32 audio samples at 24kHz.
    /// Output: 1024-dim speaker embedding tensor (raw fc output, not normalized).
    pub fn extract_embedding(&self, audio_samples: &[f32]) -> Result<Tensor> {
        let waveform = Tensor::from_slice_f32(audio_samples)
            .to_device(self.device)
            .to_dtype(DType::Float32);

        // Compute mel spectrogram
        let mel = self.compute_mel_spectrogram(&waveform)?;
        // mel shape: [n_mels, T]

        // Add batch dimension: [1, n_mels, T]
        let mel = mel.unsqueeze(0);

        // Forward through ECAPA-TDNN
        let embedding = self.forward(&mel);

        Ok(embedding)
    }

    fn compute_mel_spectrogram(&self, waveform: &Tensor) -> Result<Tensor> {
        // Pad waveform for STFT (center padding with reflect mode, matching Python)
        let pad = MEL_N_FFT / 2;
        // reflection_pad1d requires 3D input [B, C, T]
        let waveform_3d = waveform.unsqueeze(0).unsqueeze(0);
        let padded_3d = waveform_3d.reflection_pad1d(&[pad, pad]);
        let padded = padded_3d.squeeze_dim(0).squeeze_dim(0);

        // STFT (no center padding since we padded manually)
        let stft = padded.stft(
            MEL_N_FFT,
            MEL_HOP_SIZE,
            MEL_WIN_SIZE,
            &self.hann_window,
            false, // normalized
            true,  // onesided
            true,  // return_complex
        );

        // Magnitude spectrogram
        let magnitude = stft.abs().pow_scalar(2.0); // [freq_bins, T]

        // Apply mel filterbank: [n_mels, freq_bins] @ [freq_bins, T] → [n_mels, T]
        let mel_spec = self.mel_basis.matmul(&magnitude);

        // Log compression (dynamic range compression)
        let mel_spec = mel_spec.clamp_min(1e-5).log();

        Ok(mel_spec)
    }

    fn forward(&self, mel: &Tensor) -> Tensor {
        // mel: [B, 128, T]

        // Initial TDNN
        let x = self.initial_conv.forward(mel, 1, 1).relu();

        // SE-Res2Net blocks with residual outputs for MFA
        let mut block_outputs = Vec::new();
        let mut h = x;
        for (i, block) in self.se_res2net_blocks.iter().enumerate() {
            h = block.forward(&h, self.dilations[i]);
            block_outputs.push(h.shallow_clone());
        }

        // Multi-layer Feature Aggregation: concatenate block outputs
        let cat = Tensor::cat(&block_outputs, 1); // [B, 1536, T]
        let mfa_out = self.mfa_conv.forward(&cat, 1, 1).relu();

        // Attentive Statistics Pooling → [B, 3072]
        let pooled = self.asp.forward(&mfa_out);

        // Final FC: [B, 3072] → [B, 1024]
        // fc weight is [1024, 3072, 1], so we need to add a time dim
        let pooled = pooled.unsqueeze(2); // [B, 3072, 1]
        let embedding = self.fc.forward_no_pad(&pooled).squeeze_dim(2); // [B, 1024]

        // Return first (and only) batch element
        // No L2 normalization — the Python reference does not normalize,
        // and the model expects embeddings at the natural scale of the fc output.
        embedding.squeeze_dim(0) // [1024]
    }
}

/// Create mel filterbank matrix using slaney-style mel scale.
///
/// Returns a [n_mels, n_fft/2 + 1] tensor.
fn create_mel_filterbank(
    sr: f64,
    n_fft: i64,
    n_mels: i64,
    fmin: f64,
    fmax: f64,
    device: Device,
) -> Tensor {
    let n_freqs = n_fft / 2 + 1;

    // Slaney mel scale (librosa default)
    let min_mel = hz_to_mel_slaney(fmin);
    let max_mel = hz_to_mel_slaney(fmax);

    // Evenly spaced mel points
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| min_mel + (max_mel - min_mel) * i as f64 / (n_points - 1) as f64)
        .collect();

    // Convert mel points back to Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f64> = hz_points.iter().map(|&h| h * n_fft as f64 / sr).collect();

    // Build filterbank
    let mut filterbank = vec![0.0f32; (n_mels * n_freqs) as usize];

    for i in 0..n_mels as usize {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        // Slaney normalization: 2.0 / (right_hz - left_hz)
        let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);

        for j in 0..n_freqs as usize {
            let freq = j as f64;
            let val = if freq >= left && freq <= center && center > left {
                (freq - left) / (center - left)
            } else if freq > center && freq <= right && right > center {
                (right - freq) / (right - center)
            } else {
                0.0
            };
            filterbank[i * n_freqs as usize + j] = (val * enorm) as f32;
        }
    }

    Tensor::from_slice_f32(&filterbank)
        .reshape(&[n_mels, n_freqs])
        .to_device(device)
}

/// Convert Hz to mel (slaney/librosa default scale).
fn hz_to_mel_slaney(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0; // 66.667 Hz
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15.0
    let logstep = (6.4f64).ln() / 27.0;

    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

/// Convert mel to Hz (slaney/librosa default scale).
fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    }
}
