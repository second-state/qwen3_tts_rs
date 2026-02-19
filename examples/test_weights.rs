// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Test weight loading from safetensors files.
//!
//! Usage:
//!   cargo run --example test_weights -- <model_path>
//!
//! Example:
//!   cargo run --example test_weights -- ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice

use qwen3_tts::tensor::{DType, Tensor};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_path>", args[0]);
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);

    // Load main model weights
    let model_safetensors = model_path.join("model.safetensors");
    println!("Loading model weights from: {:?}", model_safetensors);

    let tensors = Tensor::load_safetensors(&model_safetensors)?;
    println!("Loaded {} tensors from main model", tensors.len());

    // Print first 20 tensor names and shapes
    println!("\nFirst 20 tensors:");
    for (i, (name, tensor)) in tensors.iter().take(20).enumerate() {
        let size: Vec<i64> = tensor.size();
        let kind = tensor.kind();
        let numel = tensor.numel();
        println!(
            "  {:2}. {} {:?} {:?} ({} elements)",
            i + 1,
            name,
            size,
            kind,
            numel
        );
    }

    // Calculate total parameters
    let total_params: i64 = tensors.iter().map(|(_, t)| t.numel()).sum();
    println!(
        "\nTotal parameters: {} ({:.2}M)",
        total_params,
        total_params as f64 / 1_000_000.0
    );

    // Load speech tokenizer weights
    let tokenizer_safetensors = model_path.join("speech_tokenizer/model.safetensors");
    if tokenizer_safetensors.exists() {
        println!(
            "\n---\nLoading speech tokenizer weights from: {:?}",
            tokenizer_safetensors
        );

        let tokenizer_tensors = Tensor::load_safetensors(&tokenizer_safetensors)?;
        println!(
            "Loaded {} tensors from speech tokenizer",
            tokenizer_tensors.len()
        );

        // Print first 10 tensor names and shapes
        println!("\nFirst 10 speech tokenizer tensors:");
        for (i, (name, tensor)) in tokenizer_tensors.iter().take(10).enumerate() {
            let size: Vec<i64> = tensor.size();
            let kind = tensor.kind();
            println!("  {:2}. {} {:?} {:?}", i + 1, name, size, kind);
        }

        let tokenizer_params: i64 = tokenizer_tensors
            .iter()
            .map(|(_, t)| t.numel())
            .sum();
        println!(
            "\nSpeech tokenizer parameters: {} ({:.2}M)",
            tokenizer_params,
            tokenizer_params as f64 / 1_000_000.0
        );
    }

    // Test basic tensor operation
    println!("\n---\nTesting basic tensor operations...");

    let embed_name = "model.embed_tokens.weight";
    if let Some((_, embed_tensor)) = tensors.iter().find(|(n, _)| n == embed_name) {
        let embed_size = embed_tensor.size();
        println!("Embedding tensor shape: {:?}", embed_size);

        let token_id = 100i64;
        let embedding = embed_tensor.select(0, token_id);
        let emb_size = embedding.size();
        println!("Embedding for token {}: shape {:?}", token_id, emb_size);

        let mean = embedding.to_dtype(DType::Float32).mean_all();
        println!("  Mean: {:.6}", mean.try_into_f64().unwrap_or(0.0));
    }

    println!("\nWeight loading test completed successfully!");

    Ok(())
}
