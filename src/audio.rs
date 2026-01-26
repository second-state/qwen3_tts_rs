// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Audio loading and processing utilities.
//!
//! This module provides functions for loading audio from various sources
//! (file paths, URLs, base64 strings) and resampling to target sample rates.

use crate::error::{Qwen3TTSError, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use hound::{WavReader, WavSpec, WavWriter};
use rubato::{FftFixedIn, Resampler};
use std::io::Cursor;
use url::Url;

/// Supported audio input types.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Path to a WAV file
    FilePath(String),
    /// Base64-encoded audio data
    Base64(String),
    /// URL to fetch audio from
    Url(String),
    /// Raw waveform samples with sample rate
    Waveform {
        /// Audio sample values
        samples: Vec<f32>,
        /// Sample rate in Hz
        sample_rate: u32,
    },
}

impl AudioInput {
    /// Create from a string, automatically detecting the type.
    pub fn from_str(s: &str) -> Self {
        if is_url(s) {
            AudioInput::Url(s.to_string())
        } else if is_probably_base64(s) {
            AudioInput::Base64(s.to_string())
        } else {
            AudioInput::FilePath(s.to_string())
        }
    }

    /// Create from raw waveform samples.
    pub fn from_waveform(samples: Vec<f32>, sample_rate: u32) -> Self {
        AudioInput::Waveform { samples, sample_rate }
    }
}

impl From<&str> for AudioInput {
    fn from(s: &str) -> Self {
        AudioInput::from_str(s)
    }
}

impl From<String> for AudioInput {
    fn from(s: String) -> Self {
        AudioInput::from_str(&s)
    }
}

impl From<(Vec<f32>, u32)> for AudioInput {
    fn from((samples, sample_rate): (Vec<f32>, u32)) -> Self {
        AudioInput::Waveform { samples, sample_rate }
    }
}

/// Check if a string looks like a URL.
fn is_url(s: &str) -> bool {
    if let Ok(url) = Url::parse(s) {
        matches!(url.scheme(), "http" | "https") && url.host().is_some()
    } else {
        false
    }
}

/// Check if a string is probably base64-encoded audio.
fn is_probably_base64(s: &str) -> bool {
    // Data URL format
    if s.starts_with("data:audio") {
        return true;
    }
    // Heuristic: no path separators and long enough to be audio data
    if !s.contains('/') && !s.contains('\\') && s.len() > 256 {
        return true;
    }
    false
}

/// Decode base64 audio data to raw bytes.
fn decode_base64_to_bytes(b64: &str) -> Result<Vec<u8>> {
    let data = if b64.contains(',') && b64.trim().starts_with("data:") {
        // Extract base64 portion after the comma
        b64.split(',').nth(1).unwrap_or(b64)
    } else {
        b64
    };

    BASE64_STANDARD
        .decode(data)
        .map_err(|e| Qwen3TTSError::Base64(e))
}

/// Load audio from a WAV file path.
pub fn load_wav_file(path: &str) -> Result<(Vec<f32>, u32)> {
    let reader = WavReader::open(path)
        .map_err(|e| Qwen3TTSError::Audio(format!("Failed to open WAV file: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert stereo to mono if necessary
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(chunk[0])) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Load audio from WAV bytes.
pub fn load_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let cursor = Cursor::new(bytes);
    let reader = WavReader::new(cursor)
        .map_err(|e| Qwen3TTSError::Audio(format!("Failed to parse WAV data: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert stereo to mono if necessary
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(chunk[0])) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Fetch audio from a URL.
///
/// Currently returns an error as URL fetching is not yet implemented.
pub fn fetch_audio_from_url(_url: &str) -> Result<Vec<u8>> {
    Err(Qwen3TTSError::Unsupported(
        "URL fetching is not yet implemented".to_string()
    ))
}

/// Load audio from an AudioInput, resampling to the target sample rate.
pub fn load_audio(input: AudioInput, target_sr: u32) -> Result<Vec<f32>> {
    let (samples, source_sr) = match input {
        AudioInput::FilePath(path) => load_wav_file(&path)?,
        AudioInput::Base64(b64) => {
            let bytes = decode_base64_to_bytes(&b64)?;
            load_wav_bytes(&bytes)?
        }
        AudioInput::Url(url) => {
            let bytes = fetch_audio_from_url(&url)?;
            load_wav_bytes(&bytes)?
        }
        AudioInput::Waveform { samples, sample_rate } => (samples, sample_rate),
    };

    // Resample if necessary
    if source_sr != target_sr {
        resample(&samples, source_sr, target_sr)
    } else {
        Ok(samples)
    }
}

/// Load multiple audio inputs.
pub fn load_audio_batch<I>(inputs: I, target_sr: u32) -> Result<Vec<Vec<f32>>>
where
    I: IntoIterator<Item = AudioInput>,
{
    inputs.into_iter().map(|input| load_audio(input, target_sr)).collect()
}

/// Resample audio from one sample rate to another.
pub fn resample(samples: &[f32], source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    if source_sr == target_sr {
        return Ok(samples.to_vec());
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(
        source_sr as usize,
        target_sr as usize,
        chunk_size,
        2, // sub-chunks
        1, // channels
    )
    .map_err(|e| Qwen3TTSError::Audio(format!("Failed to create resampler: {}", e)))?;

    let mut output = Vec::new();
    let mut input_buffer = vec![vec![0.0f32; chunk_size]];
    let mut output_buffer = resampler.output_buffer_allocate(true);

    let mut pos = 0;
    while pos < samples.len() {
        let end = (pos + chunk_size).min(samples.len());
        let actual_len = end - pos;

        // Copy samples to input buffer
        input_buffer[0][..actual_len].copy_from_slice(&samples[pos..end]);
        // Pad with zeros if needed
        if actual_len < chunk_size {
            input_buffer[0][actual_len..].fill(0.0);
        }

        let (_, out_len) = resampler
            .process_into_buffer(&input_buffer, &mut output_buffer, None)
            .map_err(|e| Qwen3TTSError::Audio(format!("Resampling failed: {}", e)))?;

        output.extend_from_slice(&output_buffer[0][..out_len]);
        pos += chunk_size;
    }

    Ok(output)
}

/// Normalize audio to [-1, 1] range.
pub fn normalize_audio(samples: &mut [f32]) {
    let max_val = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_val > 0.0 && max_val != 1.0 {
        for sample in samples.iter_mut() {
            *sample /= max_val;
        }
    }
}

/// Write audio to a WAV file.
pub fn write_wav_file(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| Qwen3TTSError::Audio(format!("Failed to create WAV file: {}", e)))?;

    // Convert f32 samples to i16
    for &sample in samples {
        // Clamp to [-1, 1] and scale to i16 range
        let clamped = sample.clamp(-1.0, 1.0);
        let scaled = (clamped * 32767.0) as i16;
        writer.write_sample(scaled)
            .map_err(|e| Qwen3TTSError::Audio(format!("Failed to write sample: {}", e)))?;
    }

    writer.finalize()
        .map_err(|e| Qwen3TTSError::Audio(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Write audio to WAV bytes.
pub fn write_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut buffer, spec)
            .map_err(|e| Qwen3TTSError::Audio(format!("Failed to create WAV writer: {}", e)))?;

        for &sample in samples {
            writer.write_sample(sample)
                .map_err(|e| Qwen3TTSError::Audio(format!("Failed to write sample: {}", e)))?;
        }

        writer.finalize()
            .map_err(|e| Qwen3TTSError::Audio(format!("Failed to finalize WAV: {}", e)))?;
    }

    Ok(buffer.into_inner())
}

/// Encode audio to base64 WAV format.
pub fn encode_audio_base64(samples: &[f32], sample_rate: u32) -> Result<String> {
    let bytes = write_wav_bytes(samples, sample_rate)?;
    Ok(BASE64_STANDARD.encode(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_url() {
        assert!(is_url("https://example.com/audio.wav"));
        assert!(is_url("http://example.com/audio.wav"));
        assert!(!is_url("/path/to/file.wav"));
        assert!(!is_url("file.wav"));
    }

    #[test]
    fn test_is_probably_base64() {
        assert!(is_probably_base64("data:audio/wav;base64,AAAA"));
        assert!(!is_probably_base64("/path/to/file.wav"));
    }

    #[test]
    fn test_audio_input_from_str() {
        match AudioInput::from_str("/path/to/file.wav") {
            AudioInput::FilePath(_) => {}
            _ => panic!("Expected FilePath"),
        }

        match AudioInput::from_str("https://example.com/audio.wav") {
            AudioInput::Url(_) => {}
            _ => panic!("Expected Url"),
        }
    }

    #[test]
    fn test_normalize_audio() {
        let mut samples = vec![0.5, -1.0, 0.25, -0.5];
        normalize_audio(&mut samples);
        assert!((samples[1] - (-1.0)).abs() < 1e-6);
        assert!((samples[0] - 0.5).abs() < 1e-6);
    }
}
