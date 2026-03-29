// Copyright 2026 Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Audio encoding for multiple output formats (MP3, FLAC, OGG Opus, WAV, PCM).

use crate::audio::write_wav_bytes;
use crate::error::{Qwen3TTSError, Result};

/// Encode audio samples to the requested format.
///
/// Returns `(encoded_bytes, content_type)`.
pub fn encode_audio(samples: &[f32], sample_rate: u32, format: &str) -> Result<(Vec<u8>, &'static str)> {
    match format {
        "wav" => {
            let bytes = write_wav_bytes(samples, sample_rate)?;
            Ok((bytes, "audio/wav"))
        }
        "pcm" => {
            let bytes = encode_pcm_i16(samples);
            Ok((bytes, "audio/pcm"))
        }
        "mp3" => {
            let bytes = encode_mp3(samples, sample_rate)?;
            Ok((bytes, "audio/mpeg"))
        }
        "flac" => {
            let bytes = encode_flac(samples, sample_rate)?;
            Ok((bytes, "audio/flac"))
        }
        "ogg" | "opus" => {
            let bytes = encode_ogg_opus(samples, sample_rate)?;
            Ok((bytes, "audio/ogg"))
        }
        _ => Err(Qwen3TTSError::Audio(format!(
            "Unsupported format '{}'. Supported: wav, pcm, mp3, flac, ogg, opus",
            format
        ))),
    }
}

/// Encode f32 samples to raw 16-bit signed little-endian PCM bytes.
fn encode_pcm_i16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let scaled = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&scaled.to_le_bytes());
    }
    bytes
}

/// Encode to MP3 (128 kbps CBR).
///
/// Resamples from native sample rate to 44100 Hz (MPEG-1 standard rate)
/// for maximum player compatibility.
fn encode_mp3(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use mp3lame_encoder::{Builder, FlushNoGap, MonoPcm};

    // Resample to 44100 Hz for MPEG-1 Layer III compatibility
    let samples_44k = if sample_rate != 44100 {
        resample_linear(samples, sample_rate, 44100)
    } else {
        samples.to_vec()
    };

    let mut builder = Builder::new().ok_or_else(|| {
        Qwen3TTSError::Audio("Failed to create MP3 encoder".to_string())
    })?;
    builder.set_num_channels(1).map_err(|e| {
        Qwen3TTSError::Audio(format!("MP3 encoder config error: {:?}", e))
    })?;
    builder.set_sample_rate(44100).map_err(|e| {
        Qwen3TTSError::Audio(format!("MP3 encoder config error: {:?}", e))
    })?;
    builder.set_brate(mp3lame_encoder::Bitrate::Kbps128).map_err(|e| {
        Qwen3TTSError::Audio(format!("MP3 encoder config error: {:?}", e))
    })?;
    builder.set_quality(mp3lame_encoder::Quality::Best).map_err(|e| {
        Qwen3TTSError::Audio(format!("MP3 encoder config error: {:?}", e))
    })?;
    let mut encoder = builder.build().map_err(|e| {
        Qwen3TTSError::Audio(format!("Failed to build MP3 encoder: {:?}", e))
    })?;

    let pcm: Vec<i16> = samples_44k
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

    let input = MonoPcm(&pcm);
    let mut mp3_out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));

    encoder.encode_to_vec(input, &mut mp3_out).map_err(|e| {
        Qwen3TTSError::Audio(format!("MP3 encoding error: {:?}", e))
    })?;

    encoder.flush_to_vec::<FlushNoGap>(&mut mp3_out).map_err(|e| {
        Qwen3TTSError::Audio(format!("MP3 flush error: {:?}", e))
    })?;

    Ok(mp3_out)
}

/// Encode to FLAC (lossless, 16-bit).
fn encode_flac(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use flacenc::bitsink::ByteSink;
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let pcm_i32: Vec<i32> = samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i32)
        .collect();

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_enc, e)| Qwen3TTSError::Audio(format!("FLAC config error: {:?}", e)))?;

    let source = flacenc::source::MemSource::from_samples(&pcm_i32, 1, 16, sample_rate as usize);
    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| Qwen3TTSError::Audio(format!("FLAC encoding error: {}", e)))?;

    let mut sink = ByteSink::new();
    flac_stream
        .write(&mut sink)
        .map_err(|e| Qwen3TTSError::Audio(format!("FLAC write error: {:?}", e)))?;

    Ok(sink.into_inner())
}

/// Encode to OGG Opus (20ms frames, 48 kHz).
fn encode_ogg_opus(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use audiopus::coder::Encoder as OpusEncoder;
    use audiopus::{Application, Channels, SampleRate as OpusSampleRate};

    // Opus requires 48 kHz
    let samples_48k = if sample_rate != 48000 {
        resample_linear(samples, sample_rate, 48000)
    } else {
        samples.to_vec()
    };

    let encoder = OpusEncoder::new(OpusSampleRate::Hz48000, Channels::Mono, Application::Voip)
        .map_err(|e| Qwen3TTSError::Audio(format!("Opus encoder creation error: {}", e)))?;

    // 20ms frames at 48kHz = 960 samples
    let frame_size = 960;
    let mut ogg_buf = Vec::new();
    let serial = 1u32;

    {
        let mut writer = ogg::PacketWriter::new(&mut ogg_buf);

        // OpusHead identification header (RFC 7845)
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(1); // channel count
        head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
        head.extend_from_slice(&48000u32.to_le_bytes()); // input sample rate
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // channel mapping family
        writer
            .write_packet(head, serial, ogg::PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| Qwen3TTSError::Audio(format!("OGG write error: {}", e)))?;

        // OpusTags comment header
        let vendor = b"qwen3-tts-rs";
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes()); // no user comments
        writer
            .write_packet(tags, serial, ogg::PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| Qwen3TTSError::Audio(format!("OGG write error: {}", e)))?;

        // Encode audio frames
        let mut opus_out = vec![0u8; 4000];
        let mut granule: u64 = 0;
        let total_frames = (samples_48k.len() + frame_size - 1) / frame_size;

        for (i, chunk) in samples_48k.chunks(frame_size).enumerate() {
            let frame: Vec<f32> = if chunk.len() < frame_size {
                let mut padded = chunk.to_vec();
                padded.resize(frame_size, 0.0);
                padded
            } else {
                chunk.to_vec()
            };

            let encoded_len = encoder.encode_float(&frame, &mut opus_out).map_err(|e| {
                Qwen3TTSError::Audio(format!("Opus encoding error: {}", e))
            })?;

            granule += frame_size as u64;
            let end_info = if i == total_frames - 1 {
                ogg::PacketWriteEndInfo::EndStream
            } else {
                ogg::PacketWriteEndInfo::NormalPacket
            };

            writer
                .write_packet(
                    opus_out[..encoded_len].to_vec(),
                    serial,
                    end_info,
                    granule,
                )
                .map_err(|e| Qwen3TTSError::Audio(format!("OGG write error: {}", e)))?;
        }
    }

    Ok(ogg_buf)
}

/// Linear interpolation resampling between sample rates.
fn resample_linear(samples: &[f32], source_sr: u32, target_sr: u32) -> Vec<f32> {
    if source_sr == target_sr || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = target_sr as f64 / source_sr as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        if idx + 1 < samples.len() {
            output.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }

    output
}
