// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Model weight loading from safetensors format using tch.
//!
//! This module provides utilities for loading pre-trained weights
//! from HuggingFace safetensors format.

use crate::error::{Qwen3TTSError, Result};
use std::path::Path;
use tch::{nn, Device, Tensor};

/// Load model weights from a safetensors file into a VarStore.
pub fn load_safetensors(vs: &mut nn::VarStore, path: &Path) -> Result<()> {
    vs.load(path)?;
    Ok(())
}

/// Load a single tensor from a safetensors file.
pub fn load_tensor(path: &Path, tensor_name: &str, device: Device) -> Result<Tensor> {
    let tensors = tch::Tensor::read_safetensors(path)?;

    for (name, tensor) in tensors {
        if name == tensor_name {
            return Ok(tensor.to_device(device));
        }
    }

    Err(Qwen3TTSError::ModelLoad(format!(
        "Tensor '{}' not found in {:?}",
        tensor_name, path
    )))
}

/// Load all tensors from a safetensors file.
pub fn load_all_tensors(path: &Path, device: Device) -> Result<Vec<(String, Tensor)>> {
    let tensors = tch::Tensor::read_safetensors(path)?;

    let result: Vec<(String, Tensor)> = tensors
        .into_iter()
        .map(|(name, tensor)| (name, tensor.to_device(device)))
        .collect();

    Ok(result)
}

/// Container for Qwen3 TTS model weights.
pub struct Qwen3TTSWeights {
    /// Path to the model directory
    pub model_path: std::path::PathBuf,
    /// Device to load tensors on
    pub device: Device,
    /// VarStore containing all model weights
    pub vs: nn::VarStore,
}

impl Qwen3TTSWeights {
    /// Load model weights from a directory.
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);

        // Load main model weights
        let model_safetensors = model_path.join("model.safetensors");
        if model_safetensors.exists() {
            vs.load(&model_safetensors)?;
        }

        Ok(Self {
            model_path: model_path.to_path_buf(),
            device,
            vs,
        })
    }

    /// Get a tensor by name from the VarStore.
    pub fn get(&self, name: &str) -> Option<Tensor> {
        self.vs.variables().get(name).map(|t| t.shallow_clone())
    }

    /// Get the root path for the model.
    pub fn root(&self) -> nn::Path<'_> {
        self.vs.root()
    }
}

/// Speech tokenizer weights.
pub struct SpeechTokenizerWeights {
    /// Path to the speech tokenizer directory
    pub tokenizer_path: std::path::PathBuf,
    /// Device to load tensors on
    pub device: Device,
    /// VarStore containing speech tokenizer weights
    pub vs: nn::VarStore,
}

impl SpeechTokenizerWeights {
    /// Load speech tokenizer weights from a directory.
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        let tokenizer_path = model_path.join("speech_tokenizer");
        let mut vs = nn::VarStore::new(device);

        // Load speech tokenizer weights
        let tokenizer_safetensors = tokenizer_path.join("model.safetensors");
        if tokenizer_safetensors.exists() {
            vs.load(&tokenizer_safetensors)?;
        }

        Ok(Self {
            tokenizer_path,
            device,
            vs,
        })
    }

    /// Get the root path for the tokenizer.
    pub fn root(&self) -> nn::Path<'_> {
        self.vs.root()
    }
}

/// Print summary of loaded weights.
pub fn print_weight_summary(path: &Path) -> Result<()> {
    let tensors = tch::Tensor::read_safetensors(path)?;

    println!("Loaded {} tensors from {:?}", tensors.len(), path);
    println!("First 20 tensors:");

    for (i, (name, tensor)) in tensors.iter().take(20).enumerate() {
        let size: Vec<i64> = tensor.size();
        let kind = tensor.kind();
        println!("  {}: {} {:?} {:?}", i, name, size, kind);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_default() {
        let device = Device::Cpu;
        assert_eq!(device, Device::Cpu);
    }
}
