// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The own llama-family decoder body (inference plan P1).
//!
//! HF-config-driven: [`config::DecoderConfig`] parses a HuggingFace
//! `config.json` directly, so one body runs the open llama-family class
//! (TinyLlama, Llama 3.2, SmolLM) from the standard artifact triple
//! (`config.json` / `tokenizer.json` / `model.safetensors`) — the same
//! triple `embed`'s BERT uses, and the byte shape the eidetic model
//! corridor stores.
//!
//! Built from burn-nn primitives (`RmsNorm`, `RotaryEncoding`, `SwiGlu`)
//! against the workspace burn pin; the GQA attention wiring in
//! [`attention`] is the piece burn-nn does not provide, informed by
//! tracel-ai/models' llama-burn (MIT OR Apache-2.0) as a reference
//! implementation.
//!
//! Slice status: config + attention + layer + model stack + safetensors
//! loader (this module). The KV-cached generation loop and the
//! `InferenceProvider` impl are the next slices.

pub mod attention;
pub mod config;
pub mod generate;
#[cfg(feature = "decoder-wgpu")]
pub mod gpu_probe;
pub mod layer;
pub mod loader;
#[cfg(feature = "decoder-lora")]
pub mod lora;
pub mod model;
pub mod provider;
pub mod sample;
pub mod tensors;
#[cfg(feature = "decoder-lora")]
pub mod train;
#[cfg(feature = "decoder-autodiff")]
pub mod train_autodiff;
#[cfg(feature = "decoder-lora")]
pub mod train_autodiff_settings;

pub use attention::{DecoderAttention, LayerKvCache, LlamaRotaryEncoding};
pub use config::DecoderConfig;
pub use generate::{
    AsyncGenerationOutcome, TokenPicker, generate_ids, generate_ids_with, generate_ids_with_async,
    generate_ids_with_async_controlled,
};
pub use layer::{DecoderLayer, LoadedDecoderLayer};
pub use loader::load_decoder_from_bytes;
pub use model::{DecoderModel, KvCache, LoadedDecoder};

#[cfg(feature = "decoder-wgpu")]
use burn::tensor::Device;
#[cfg(feature = "decoder-wgpu")]
pub use gpu_probe::{DecoderGpuKind, GpuAdapterFacts, GpuDeviceType, probe_gpu_adapter};
#[cfg(feature = "decoder-lora")]
pub use lora::{PEFT_LORA_NDARRAY_LOADER, PeftLoraAdapterLoader};
pub use provider::{DecoderGeneration, DecoderProvider};
pub use sample::Sampler;
#[cfg(feature = "decoder-lora")]
pub use train::{
    LoraTrainerSettings, TRAINED_ADAPTER_FORMAT_VERSION, TRAINED_PEFT_VERSION, TrainedLoraAdapter,
    TrainingCase, expected_token_rank, ranking_tally, train_peft_lora,
};
// The v1 vocabulary rides with the loader, not with the trainer: a consumer
// must be able to name a v1 adapter's version and settings on a build that
// will never train one. Only the function is behind `decoder-autodiff`.
#[cfg(feature = "decoder-autodiff")]
pub use train_autodiff::train_peft_lora_autodiff;
#[cfg(feature = "decoder-lora")]
pub use train_autodiff_settings::{
    AutodiffLoraSettings, TRAINED_ADAPTER_FORMAT_VERSION_AUTODIFF, TRAINED_PEFT_VERSION_AUTODIFF,
};

/// The device vocabulary hosts select from, re-exported so a composition
/// layer never names `burn` itself.
pub use burn::tensor::Device as DecoderDevice;

/// The decoder on the wgpu backend — the concrete inference lane hosts use
/// so they never name `burn` themselves.
#[cfg(feature = "decoder-wgpu")]
pub type WgpuDecoderProvider = DecoderProvider;

/// Load a llama-family checkpoint (HF artifact triple as bytes) on the wgpu
/// backend. The host-facing entry point for real local inference.
#[cfg(feature = "decoder-wgpu")]
pub fn load_wgpu_provider(
    config_bytes: &[u8],
    tokenizer_bytes: &[u8],
    weights_bytes: &[u8],
    model_id: impl Into<String>,
) -> Result<WgpuDecoderProvider, crate::infer::provider::InferError> {
    DecoderProvider::from_bytes(
        config_bytes,
        tokenizer_bytes,
        weights_bytes,
        model_id,
        "burn-wgpu",
        &Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0)),
    )
}

/// Deterministic-weight helpers shared by the decoder tests: same values
/// on every backend by construction (the `embed::bert::wgpu_parity`
/// pattern).
#[cfg(test)]
pub(crate) mod test_support {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::config::DecoderConfig;

    pub fn det_vec(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i + salt * 7919) as f32) * 0.618_034).sin() * 0.05)
            .collect()
    }

    pub fn t2(a: usize, b: usize, salt: usize, dev: &Device) -> Tensor<2> {
        Tensor::from_data(TensorData::new(det_vec(a * b, salt), [a, b]), dev)
    }

    pub fn t1_ones(n: usize, dev: &Device) -> Tensor<1> {
        Tensor::ones([n], dev)
    }

    /// Tiny dims for fast tests: hidden 8, 4 heads (head_dim 2), GQA 2.
    pub fn tiny_config() -> DecoderConfig {
        DecoderConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 16,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            tie_word_embeddings: false,
            eos_token_id: Vec::new(),
        }
    }
}
