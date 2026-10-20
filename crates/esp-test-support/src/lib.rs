// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Tiny deterministic models shared by ESP and its integration-test consumers.
//!
//! This development-only package supplies test data without a model runtime.
//! ESP and downstream integration tests share one decoder helper implementation.
//! The bundled template and MoE assets are versioned qualification snapshots;
//! they do not acquire model weights or depend on the ESP runtime package.

pub mod decoder;

/// ESP's literal SmolLM2 instruction template, before test-specific shortening.
pub const SMOLLM2_INSTRUCT_V1: &[u8] = include_bytes!("../fixtures/smollm2-instruct-v1.json");

/// Configuration for the tiny Qwen2 mixture-of-experts qualification model.
pub const QWEN2_MOE_CONFIG_JSON: &str = include_str!("../fixtures/qwen2_moe/config.json");

/// Safetensors weights matching [`QWEN2_MOE_CONFIG_JSON`].
pub const QWEN2_MOE_WEIGHTS: &[u8] = include_bytes!("../fixtures/qwen2_moe/model.safetensors");

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::SafeTensors;

    #[test]
    fn decoder_assets_have_matching_vocabulary_and_weight_shapes() {
        let config: serde_json::Value = serde_json::from_str(decoder::CONFIG_JSON).unwrap();
        let tokenizer: serde_json::Value =
            serde_json::from_str(&decoder::tokenizer_json()).unwrap();
        assert_eq!(
            tokenizer["model"]["vocab"].as_object().unwrap().len() as u64,
            config["vocab_size"].as_u64().unwrap()
        );
        let bytes = decoder::weights_bytes();
        let weights = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(
            weights.tensor("model.embed_tokens.weight").unwrap().shape(),
            &[32, 8]
        );
        assert_eq!(weights.tensor("lm_head.weight").unwrap().shape(), &[32, 8]);
        assert_eq!(
            weights
                .tensor("model.layers.1.self_attn.k_proj.weight")
                .unwrap()
                .shape(),
            &[4, 8]
        );
    }

    #[test]
    fn packaged_moe_assets_and_instruction_template_are_readable() {
        let config: serde_json::Value = serde_json::from_str(QWEN2_MOE_CONFIG_JSON).unwrap();
        let weights = SafeTensors::deserialize(QWEN2_MOE_WEIGHTS).unwrap();
        assert_eq!(config["model_type"], "qwen2_moe");
        assert_eq!(
            weights.tensor("model.embed_tokens.weight").unwrap().shape(),
            &[32, 16]
        );
        let template: serde_json::Value = serde_json::from_slice(SMOLLM2_INSTRUCT_V1).unwrap();
        assert_eq!(template["schema"], "esp.prompt-template/v1");
    }
}
