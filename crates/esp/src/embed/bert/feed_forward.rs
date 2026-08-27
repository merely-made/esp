// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! BERT feed-forward block: intermediate (Linear + GELU) + output
//! (Linear + residual + LayerNorm).
//!
//! Mirrors HuggingFace's `BertIntermediate` + `BertOutput` shape so
//! safetensors keys map directly: `intermediate.dense`, `output.dense`,
//! `output.LayerNorm`.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::{Device, Tensor};

use super::config::BertConfig;

/// Expansion projection: hidden_size → intermediate_size, GELU.
#[derive(Module, Debug)]
pub struct BertIntermediate {
    dense: Linear,
}

impl BertIntermediate {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        Self {
            dense: LinearConfig::new(config.hidden_size, config.intermediate_size).init(device),
        }
    }

    pub fn from_loaded(loaded: &super::loaded::LoadedBertLayer, device: &Device) -> Self {
        Self {
            dense: super::construct::linear_from_loaded(
                loaded.inter_w.clone(),
                loaded.inter_b.clone(),
                device,
            ),
        }
    }

    /// `gelu(dense(x))`. Input `[B, S, hidden]`, output `[B, S, intermediate]`.
    pub fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        gelu(self.dense.forward(hidden))
    }
}

/// Contraction projection + residual + LayerNorm:
/// intermediate_size → hidden_size, then add+norm.
#[derive(Module, Debug)]
pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        Self {
            dense: LinearConfig::new(config.intermediate_size, config.hidden_size).init(device),
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
                loaded.out_w.clone(),
                loaded.out_b.clone(),
                device,
            ),
            layer_norm: super::construct::layer_norm_from_loaded(
                loaded.out_ln_gamma.clone(),
                loaded.out_ln_beta.clone(),
                config.layer_norm_eps,
                device,
            ),
        }
    }

    /// `LayerNorm(dense(intermediate) + residual)`.
    /// `intermediate`: `[B, S, intermediate]`, `residual`: `[B, S, hidden]`.
    pub fn forward(&self, intermediate: Tensor<3>, residual: Tensor<3>) -> Tensor<3> {
        let projected = self.dense.forward(intermediate);
        self.layer_norm.forward(projected + residual)
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

    fn rand(shape: [usize; 3]) -> Tensor<3> {
        let device = Device::ndarray();
        Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device)
    }

    #[test]
    fn intermediate_expands_to_intermediate_size() {
        let device = Device::ndarray();
        let cfg = config();
        let layer: BertIntermediate = BertIntermediate::new(&cfg, &device);
        let input = rand([2, 8, cfg.hidden_size]);
        let out = layer.forward(input);
        assert_eq!(out.dims(), [2, 8, cfg.intermediate_size]);
    }

    #[test]
    fn output_contracts_to_hidden_size() {
        let device = Device::ndarray();
        let cfg = config();
        let layer: BertOutput = BertOutput::new(&cfg, &device);
        let intermediate = rand([2, 8, cfg.intermediate_size]);
        let residual = rand([2, 8, cfg.hidden_size]);
        let out = layer.forward(intermediate, residual);
        assert_eq!(out.dims(), [2, 8, cfg.hidden_size]);
    }

    #[test]
    fn intermediate_no_nans() {
        let device = Device::ndarray();
        let layer: BertIntermediate = BertIntermediate::new(&config(), &device);
        let input = rand([1, 4, 384]);
        let out = layer.forward(input);
        let v = out.into_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| !x.is_nan() && !x.is_infinite()));
    }
}
