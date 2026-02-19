// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Demo: Clone a voice from reference audio and generate speech.
//!
//! Usage:
//!   cargo run --example voice_clone_demo --release -- <model_path> <ref_audio> [text] [language] [ref_text]
//!
//! Without ref_text (x-vector only mode):
//!   cargo run --example voice_clone_demo --release -- \
//!     models/Qwen3-TTS-12Hz-0.6B-Base ref.wav \
//!     "Hello world, this is a voice cloning test." english
//!
//! With ref_text (ICL mode, higher quality):
//!   cargo run --example voice_clone_demo --release -- \
//!     models/Qwen3-TTS-12Hz-0.6B-Base ref.wav \
//!     "Hello world, this is a voice cloning test." english \
//!     "This is the transcript of the reference audio."

use qwen3_tts::audio::write_wav_file;
use qwen3_tts::audio_encoder::AudioEncoder;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;
use qwen3_tts::tensor::Device;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_path> <ref_audio> [text] [language] [ref_text]",
            args[0]
        );
        eprintln!();
        eprintln!("Examples:");
        eprintln!(
            "  {} models/Qwen3-TTS-12Hz-0.6B-Base ref.wav \"Hello world!\" english",
            args[0]
        );
        eprintln!();
        eprintln!("  With reference text (ICL mode):");
        eprintln!(
            "  {} models/Qwen3-TTS-12Hz-0.6B-Base ref.wav \"Hello!\" english \"Ref transcript\"",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ref_audio_path = &args[2];
    let text = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("Hello! This is a test of voice cloning with Qwen3 TTS.");
    let language = args.get(4).map(|s| s.as_str()).unwrap_or("english");
    let ref_text = args.get(5).map(|s| s.as_str());

    let icl_mode = ref_text.is_some();

    println!("=== Qwen3 TTS Voice Clone Demo ===");
    if icl_mode {
        println!("Mode: ICL (In-Context Learning)");
    } else {
        println!("Mode: X-vector only");
    }
    println!();

    // Step 1: Load model
    println!("Loading model from: {}", model_path);
    let device = Device::Cpu;
    let inference = TTSInference::new(Path::new(model_path), device)?;

    // Step 2: Load speaker encoder from the same weights
    println!();
    println!("Loading speaker encoder...");
    let se_config = inference.config().speaker_encoder_config.clone();
    let speaker_encoder = SpeakerEncoder::load(inference.weights(), &se_config, device)?;

    // Step 3: Load reference audio
    println!();
    println!("Loading reference audio: {}", ref_audio_path);
    let (ref_samples, ref_sr) = qwen3_tts::audio::load_wav_file(ref_audio_path)?;
    println!(
        "  {} samples at {} Hz ({:.2}s)",
        ref_samples.len(),
        ref_sr,
        ref_samples.len() as f64 / ref_sr as f64
    );

    // Resample to 24kHz if needed
    let ref_samples = if ref_sr != se_config.sample_rate {
        println!(
            "  Resampling from {} Hz to {} Hz...",
            ref_sr, se_config.sample_rate
        );
        qwen3_tts::audio::resample(&ref_samples, ref_sr, se_config.sample_rate)?
    } else {
        ref_samples
    };

    // Step 4: Extract speaker embedding
    println!("Extracting speaker embedding...");
    let speaker_embedding = speaker_encoder.extract_embedding(&ref_samples)?;
    println!("  Speaker embedding shape: {:?}", speaker_embedding.size());
    let emb_sq = &speaker_embedding * &speaker_embedding;
    let emb_norm = emb_sq.mean_all().try_into_f64().unwrap_or(0.0).sqrt()
        * (speaker_embedding.numel() as f64).sqrt();
    println!("  Speaker embedding norm: {:.4}", emb_norm);

    // Step 5: Generate speech
    println!();
    println!("Generating speech...");
    println!("  Text: \"{}\"", text);
    println!("  Language: {}", language);
    if let Some(rt) = ref_text {
        println!("  Reference text: \"{}\"", rt);
    }
    println!();

    let (waveform, sample_rate) = if icl_mode {
        // ICL mode: encode reference audio to codec tokens, then generate with ICL
        let ref_text_str = ref_text.unwrap();

        println!("=== ICL Mode: Encoding reference audio ===");
        let speech_tokenizer_path = Path::new(model_path)
            .join("speech_tokenizer")
            .join("model.safetensors");
        let audio_encoder = AudioEncoder::load(&speech_tokenizer_path, device)?;

        println!();
        println!("Encoding reference audio to codec tokens...");
        let ref_codes = audio_encoder.encode(&ref_samples)?;
        println!("  Encoded {} codec frames", ref_codes.len());

        println!();
        println!("=== Generating with ICL ===");
        inference.generate_with_icl(
            text,
            ref_text_str,
            &ref_codes,
            &speaker_embedding,
            language,
            0.9,
            50,
            2048,
        )?
    } else {
        // X-vector only mode
        inference.generate_with_xvector(text, &speaker_embedding, language, 0.9, 50, 2048)?
    };

    // Step 6: Write output
    let output_path = "output_voice_clone.wav";
    println!();
    println!("Writing output to: {}", output_path);
    write_wav_file(output_path, &waveform, sample_rate)?;

    println!(
        "Done! Generated {} samples at {} Hz",
        waveform.len(),
        sample_rate
    );
    println!(
        "Duration: {:.2} seconds",
        waveform.len() as f64 / sample_rate as f64
    );

    Ok(())
}
