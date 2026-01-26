// Copyright 2026 The Alibaba Qwen team.
// SPDX-License-Identifier: Apache-2.0

//! Demo application for Qwen3 TTS Rust port.

use qwen3_tts::audio::write_wav_file;
use qwen3_tts::inference::TTSInference;
use std::path::Path;
use tch::Device;

fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_path> [text] [speaker] [language]", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  {} ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice \"Hello, world!\" Vivian english",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = &args[1];
    let text = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Hello! This is a test of the Qwen3 TTS system.");
    let speaker = args.get(3).map(|s| s.as_str()).unwrap_or("Vivian");
    let language = args.get(4).map(|s| s.as_str()).unwrap_or("english");

    println!("=== Qwen3 TTS Rust Demo ===");
    println!();
    println!("Loading model from: {}", model_path);

    // Create inference engine
    let inference = TTSInference::new(Path::new(model_path), Device::Cpu)?;

    println!();
    println!("Generating speech...");
    println!("  Text: \"{}\"", text);
    println!("  Speaker: {}", speaker);
    println!("  Language: {}", language);
    println!();

    // Generate audio
    let (waveform, sample_rate) = inference.generate(text, speaker, language)?;

    // Write output
    let output_path = "output.wav";
    println!();
    println!("Writing output to: {}", output_path);

    write_wav_file(output_path, &waveform, sample_rate)?;

    println!("Done! Generated {} samples at {} Hz", waveform.len(), sample_rate);
    println!("Duration: {:.2} seconds", waveform.len() as f64 / sample_rate as f64);

    Ok(())
}
