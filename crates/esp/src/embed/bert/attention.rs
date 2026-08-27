// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! BERT multi-head self-attention.
//!
//! ```text
//! Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
//! ```
//!
//! Combined into [`BertAttention`] which wraps [`BertSelfAttention`]
//! (Q/K/V + scaled-dot-product) and [`BertSelfOutput`] (output projection +
//! residual + LayerNorm), mirroring HuggingFace's `BertAttention` shape
//! exactly so safetensors weight names map one-to-one.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Tensor};

use super::config::BertConfig;

/// Multi-head scaled-dot-product self-attention. Outputs raw context vectors;
/// the residual + LayerNorm step lives in [`BertSelfOutput`].
#[derive(Module, Debug)]
pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl BertSelfAttention {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        let head_dim = config.head_dim();
        Self {
            query: LinearConfig::new(config.hidden_size, config.hidden_size).init(device),
            key: LinearConfig::new(config.hidden_size, config.hidden_size).init(device),
            value: LinearConfig::new(config.hidden_size, config.hidden_size).init(device),
            num_heads: config.num_attention_heads,
            head_dim,
        }
    }

    /// Construct from pre-loaded Q/K/V weights. Note: HF stores Linear
    /// weights as `[out, in]`; if a future check shows Burn expects
    /// `[in, out]`, transpose at the safetensors-extraction boundary.
    pub fn from_loaded(
        config: &BertConfig,
        loaded: &super::loaded::LoadedBertLayer,
        device: &Device,
    ) -> Self {
        Self {
            query: super::construct::linear_from_loaded(
                loaded.q_w.clone(),
                loaded.q_b.clone(),
                device,
            ),
            key: super::construct::linear_from_loaded(
                loaded.k_w.clone(),
                loaded.k_b.clone(),
                device,
            ),
            value: super::construct::linear_from_loaded(
                loaded.v_w.clone(),
                loaded.v_b.clone(),
                device,
            ),
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        }
    }

    /// Forward pass.
    ///
    /// - `hidden`: `[batch, seq_len, hidden_size]`
    /// - returns: `[batch, seq_len, hidden_size]`
    pub fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, hidden_size] = hidden.dims();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        let q = self.query.forward(hidden.clone());
        let k = self.key.forward(hidden.clone());
        let v = self.value.forward(hidden);

        // Reshape for multi-head: [B, S, H] -> [B, S, num_heads, head_dim] ->
        // [B, num_heads, S, head_dim]
        let q = q
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);

        // Scaled-dot-product: Q @ K^T / sqrt(d_k)
        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn = softmax(scores, 3);

        // Apply attention to V: [B, num_heads, S, head_dim]
        let context = attn.matmul(v);

        // Reshape back: [B, num_heads, S, head_dim] -> [B, S, num_heads, head_dim]
        // -> [B, S, hidden_size]
        context
            .swap_dims(1, 2)
            .reshape([batch, seq_len, hidden_size])
    }
}

/// Output projection + residual + LayerNorm. Mirrors HF's `BertSelfOutput`
/// so safetensors keys (`output.dense`, `output.LayerNorm`) map directly.
#[derive(Module, Debug)]
pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        Self {
            dense: LinearConfig::new(config.hidden_size, config.hidden_size).init(device),
            layer_norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
        }
    }

    pub fn from_loaded(
        config: &BertConfig,
        loaded: &super::loaded::LoadedBertLayer,
        device: &Device,
    ) -> Self {
        Self {
            dense: super::construct::linear_from_loaded(
                loaded.attn_out_w.clone(),
                loaded.attn_out_b.clone(),
                device,
            ),
            layer_norm: super::construct::layer_norm_from_loaded(
                loaded.attn_ln_gamma.clone(),
                loaded.attn_ln_beta.clone(),
                config.layer_norm_eps,
                device,
            ),
        }
    }

    /// `LayerNorm(dense(attn) + residual)`.
    pub fn forward(&self, attn: Tensor<3>, residual: Tensor<3>) -> Tensor<3> {
        let projected = self.dense.forward(attn);
        self.layer_norm.forward(projected + residual)
    }
}

/// Combined attention block: self-attention → output projection.
#[derive(Module, Debug)]
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        Self {
            self_attention: BertSelfAttention::new(config, device),
            output: BertSelfOutput::new(config, device),
        }
    }

    pub fn from_loaded(
        config: &BertConfig,
        loaded: &super::loaded::LoadedBertLayer,
        device: &Device,
    ) -> Self {
        Self {
            self_attention: BertSelfAttention::from_loaded(config, loaded, device),
            output: BertSelfOutput::from_loaded(config, loaded, device),
        }
    }

    pub fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let attn = self.self_attention.forward(hidden.clone());
        self.output.forward(attn, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::bert::config::MINILM_L6_V2;

    // backend chosen per call site via Device

    fn config() -> BertConfig {
        let mut c = MINILM_L6_V2.clone();
        c.hidden_act = "gelu".to_string();
        c
    }

    fn rand_input(batch: usize, seq: usize, hidden: usize) -> Tensor<3> {
        let device = Device::ndarray();
        Tensor::random(
            [batch, seq, hidden],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        )
    }

    #[test]
    fn self_attention_returns_input_shape() {
        let device = Device::ndarray();
        let attn: BertSelfAttention = BertSelfAttention::new(&config(), &device);
        let input = rand_input(2, 8, 384);
        let out = attn.forward(input);
        assert_eq!(out.dims(), [2, 8, 384]);
    }

    #[test]
    fn self_output_returns_input_shape() {
        let device = Device::ndarray();
        let so: BertSelfOutput = BertSelfOutput::new(&config(), &device);
        let attn = rand_input(2, 8, 384);
        let residual = rand_input(2, 8, 384);
        let out = so.forward(attn, residual);
        assert_eq!(out.dims(), [2, 8, 384]);
    }

    #[test]
    fn full_attention_block_returns_input_shape() {
        let device = Device::ndarray();
        let block: BertAttention = BertAttention::new(&config(), &device);
        let input = rand_input(2, 8, 384);
        let out = block.forward(input);
        assert_eq!(out.dims(), [2, 8, 384]);
    }

    #[test]
    fn attention_handles_single_token_sequence() {
        let device = Device::ndarray();
        let block: BertAttention = BertAttention::new(&config(), &device);
        let input = rand_input(1, 1, 384);
        let out = block.forward(input);
        assert_eq!(out.dims(), [1, 1, 384]);
    }

    #[test]
    fn attention_no_nans_on_random_input() {
        let device = Device::ndarray();
        let block: BertAttention = BertAttention::new(&config(), &device);
        let input = rand_input(2, 16, 384);
        let out = block.forward(input);
        let data = out.into_data();
        let values = data.to_vec::<f32>().unwrap();
        assert!(values.iter().all(|x| !x.is_nan()), "produced NaN values");
        assert!(
            values.iter().all(|x| !x.is_infinite()),
            "produced Inf values"
        );
    }
}
