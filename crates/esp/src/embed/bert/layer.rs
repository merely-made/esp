// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! BERT transformer layer — composes attention with feed-forward.
//!
//! ```text
//! BertLayer(x):
//!   y = BertAttention(x)                  // self-attn + residual + norm
//!   z = BertOutput(BertIntermediate(y), y) // FFN + residual + norm
//!   return z
//! ```
//!
//! Mirrors HuggingFace's `BertLayer` shape: `attention.{self,output}`,
//! `intermediate`, `output` — names align with safetensors keys.

use burn::module::Module;
use burn::tensor::{Device, Tensor};

use super::attention::BertAttention;
use super::config::BertConfig;
use super::feed_forward::{BertIntermediate, BertOutput};

#[derive(Module, Debug)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        Self {
            attention: BertAttention::new(config, device),
            intermediate: BertIntermediate::new(config, device),
            output: BertOutput::new(config, device),
        }
    }

    pub fn from_loaded(
        config: &BertConfig,
        loaded: &super::loaded::LoadedBertLayer,
        device: &Device,
    ) -> Self {
        Self {
            attention: BertAttention::from_loaded(config, loaded, device),
            intermediate: BertIntermediate::from_loaded(loaded, device),
            output: BertOutput::from_loaded(config, loaded, device),
        }
    }

    /// Forward through one full transformer block.
    /// `[B, S, hidden]` → `[B, S, hidden]`.
    pub fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let attended = self.attention.forward(hidden);
        let expanded = self.intermediate.forward(attended.clone());
        self.output.forward(expanded, attended)
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
    fn layer_returns_input_shape() {
        let device = Device::ndarray();
        let layer: BertLayer = BertLayer::new(&config(), &device);
        let input = rand([2, 8, 384]);
        let out = layer.forward(input);
        assert_eq!(out.dims(), [2, 8, 384]);
    }

    #[test]
    fn layer_no_nans_on_random_input() {
        let device = Device::ndarray();
        let layer: BertLayer = BertLayer::new(&config(), &device);
        let input = rand([1, 16, 384]);
        let out = layer.forward(input);
        let v = out.into_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| !x.is_nan() && !x.is_infinite()));
    }

    #[test]
    fn layer_handles_long_sequence() {
        let device = Device::ndarray();
        let layer: BertLayer = BertLayer::new(&config(), &device);
        let input = rand([1, 128, 384]);
        let out = layer.forward(input);
        assert_eq!(out.dims(), [1, 128, 384]);
    }
}
