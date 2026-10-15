// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! HF-config-driven decoder configuration.
//!
//! Parsed straight from a HuggingFace `config.json`, so the Llama block
//! runs from the HF layout directly. This is the config the loader pairs
//! with `model.safetensors` + `tokenizer.json`, the same artifact triple
//! `embed`'s BERT uses.
//!
//! The decoder implements one block: pre-norm RMSNorm, bias-free
//! grouped-query attention, split-half RoPE over the full head with
//! `rope_theta` only, and a SiLU-gated MLP. A config that names another
//! family, or asks for a feature outside that block, is refused by name
//! ([`InferError::Unsupported`]) instead of loading and computing
//! something other than what the checkpoint was trained for. Reference
//! receipts exist for TinyLlama and SmolLM2; Llama 3.1 and later are
//! refused until `rope_scaling` is implemented.

use serde::Deserialize;
use serde_json::Value;

use crate::infer::provider::InferError;

/// `model_type` values this decoder runs.
const SUPPORTED_MODEL_TYPES: &[&str] = &["llama"];

/// `architectures` entries this decoder runs.
const SUPPORTED_ARCHITECTURES: &[&str] = &["LlamaForCausalLM"];

/// The config fields that decide whether a checkpoint is a Llama block at
/// all. Every field is optional so synthetic test configs without them
/// still parse; a field that is present must name something the decoder
/// implements.
#[derive(Debug, Deserialize)]
struct ArchitectureFields {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    text_config: Option<Value>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    rope_scaling: Option<Value>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    mlp_bias: Option<bool>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    #[serde(default)]
    attn_logit_softcapping: Option<f64>,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    #[serde(default)]
    num_experts: Option<u64>,
    #[serde(default)]
    num_local_experts: Option<u64>,
    #[serde(default)]
    n_routed_experts: Option<u64>,
    #[serde(default)]
    num_experts_per_tok: Option<u64>,
}

fn unsupported(feature: &str, detail: impl Into<String>) -> InferError {
    InferError::Unsupported {
        feature: format!("config.{feature}"),
        detail: detail.into(),
    }
}

impl ArchitectureFields {
    /// Refusals that must come before the dimension fields are parsed: a
    /// multimodal or foreign config would otherwise surface as a misleading
    /// missing-field error.
    fn refuse_foreign(&self) -> Result<(), InferError> {
        if self.text_config.is_some() {
            return Err(unsupported(
                "text_config",
                "multimodal configs nest the language model; only a flat decoder config runs",
            ));
        }
        if let Some(model_type) = &self.model_type
            && !SUPPORTED_MODEL_TYPES.contains(&model_type.as_str())
        {
            return Err(unsupported(
                "model_type",
                format!("{model_type:?}; supported: {SUPPORTED_MODEL_TYPES:?}"),
            ));
        }
        if let Some(architectures) = &self.architectures
            && let Some(other) = architectures
                .iter()
                .find(|a| !SUPPORTED_ARCHITECTURES.contains(&a.as_str()))
        {
            return Err(unsupported(
                "architectures",
                format!("{other:?}; supported: {SUPPORTED_ARCHITECTURES:?}"),
            ));
        }
        Ok(())
    }

    /// Refusals for features outside the implemented block.
    fn refuse_features(&self, config: &DecoderConfig) -> Result<(), InferError> {
        for (field, act) in [
            ("hidden_act", &self.hidden_act),
            ("hidden_activation", &self.hidden_activation),
        ] {
            // HF treats "swish" and "silu" as the same function.
            if let Some(act) = act
                && act != "silu"
                && act != "swish"
            {
                return Err(unsupported(
                    field,
                    format!("{act:?}; only the SiLU-gated MLP is implemented"),
                ));
            }
        }
        if let Some(scaling) = &self.rope_scaling {
            return Err(unsupported(
                "rope_scaling",
                format!("{scaling}; only unscaled RoPE (rope_theta) is implemented"),
            ));
        }
        if let Some(factor) = self.partial_rotary_factor
            && factor != 1.0
        {
            return Err(unsupported(
                "partial_rotary_factor",
                format!("{factor}; RoPE is applied to the full head"),
            ));
        }
        if let Some(head_dim) = self.head_dim
            && head_dim != config.head_dim()
        {
            return Err(unsupported(
                "head_dim",
                format!(
                    "{head_dim} differs from hidden_size / num_attention_heads = {}",
                    config.head_dim()
                ),
            ));
        }
        for (field, bias) in [
            ("attention_bias", self.attention_bias),
            ("mlp_bias", self.mlp_bias),
        ] {
            if bias == Some(true) {
                return Err(unsupported(field, "projections are bias-free"));
            }
        }
        if let Some(window) = self.sliding_window
            && self.use_sliding_window != Some(false)
            && window < config.max_position_embeddings
        {
            return Err(unsupported(
                "sliding_window",
                format!(
                    "{window} is shorter than max_position_embeddings {}; attention is full causal",
                    config.max_position_embeddings
                ),
            ));
        }
        for (field, cap) in [
            ("attn_logit_softcapping", self.attn_logit_softcapping),
            ("final_logit_softcapping", self.final_logit_softcapping),
        ] {
            if let Some(cap) = cap {
                return Err(unsupported(
                    field,
                    format!("{cap}; logit soft-capping is not implemented"),
                ));
            }
        }
        for (field, experts) in [
            ("num_experts", self.num_experts),
            ("num_local_experts", self.num_local_experts),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
        ] {
            if let Some(n) = experts
                && n > 0
            {
                return Err(unsupported(
                    field,
                    format!("{n}; mixture-of-experts layers are not implemented"),
                ));
            }
        }
        Ok(())
    }
}

/// Architecture hyperparameters for a llama-family decoder, in HF
/// `config.json` field names.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Grouped-query attention: number of shared key/value heads. Absent in
    /// pre-GQA configs, where it equals `num_attention_heads` (plain MHA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Whether the LM head shares the token-embedding matrix (TinyLlama:
    /// false; Llama 3.2 1B and SmolLM: true).
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// End-of-sequence token id(s). HF configs carry either one int
    /// (TinyLlama: 2) or a list (Llama 3.2); both parse.
    #[serde(default, deserialize_with = "one_or_many")]
    pub eos_token_id: Vec<u32>,
}

/// Accept `2`, `[2, 3]`, or absent.
fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<u32>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(u32),
        Many(Vec<u32>),
    }
    Ok(match Option::<OneOrMany>::deserialize(d)? {
        None => Vec::new(),
        Some(OneOrMany::One(id)) => vec![id],
        Some(OneOrMany::Many(ids)) => ids,
    })
}

fn default_rms_norm_eps() -> f64 {
    1.0e-5
}

fn default_rope_theta() -> f32 {
    10_000.0
}

impl DecoderConfig {
    /// Parse a HF `config.json` byte buffer, refusing any family or feature
    /// the decoder does not implement.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, InferError> {
        let parse =
            |e: serde_json::Error| InferError::InvalidConfig(format!("config.json parse: {e}"));
        let architecture: ArchitectureFields = serde_json::from_slice(bytes).map_err(parse)?;
        architecture.refuse_foreign()?;
        let config: Self = serde_json::from_slice(bytes).map_err(parse)?;
        config.validate()?;
        architecture.refuse_features(&config)?;
        Ok(config)
    }

    /// Key/value head count, with the pre-GQA fallback.
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Query heads served by each key/value head.
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.kv_heads()
    }

    fn validate(&self) -> Result<(), InferError> {
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(InferError::InvalidConfig(format!(
                "hidden_size {} not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if !self.num_attention_heads.is_multiple_of(self.kv_heads()) {
            return Err(InferError::InvalidConfig(format!(
                "num_attention_heads {} not divisible by num_key_value_heads {}",
                self.num_attention_heads,
                self.kv_heads()
            )));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(InferError::InvalidConfig(format!(
                "head_dim {} must be even for rotary encoding",
                self.head_dim()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The relevant subset of TinyLlama-1.1B-Chat-v1.0's real config.json,
    /// including fields we ignore, to prove HF configs parse as-is.
    const TINYLLAMA_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 22,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "eos_token_id": 2,
        "vocab_size": 32000
    }"#;

    #[test]
    fn parses_real_tinyllama_config() {
        let c = DecoderConfig::from_json_bytes(TINYLLAMA_CONFIG.as_bytes()).unwrap();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_hidden_layers, 22);
        assert_eq!(c.kv_heads(), 4);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.kv_group_size(), 8);
        assert!(!c.tie_word_embeddings);
        assert_eq!(c.eos_token_id, vec![2]);
    }

    #[test]
    fn eos_list_form_parses() {
        let json = r#"{
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "max_position_embeddings": 16, "eos_token_id": [128001, 128008]
        }"#;
        let c = DecoderConfig::from_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(c.eos_token_id, vec![128001, 128008]);
    }

    #[test]
    fn missing_kv_heads_falls_back_to_mha() {
        let json = r#"{
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "max_position_embeddings": 16
        }"#;
        let c = DecoderConfig::from_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(c.kv_heads(), 4);
        assert_eq!(c.kv_group_size(), 1);
        assert_eq!(c.rms_norm_eps, 1.0e-5);
        assert_eq!(c.rope_theta, 10_000.0);
    }

    /// A small valid config with `extra` merged in.
    fn tiny_with(extra: serde_json::Value) -> Vec<u8> {
        let mut base = serde_json::json!({
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 16
        });
        for (k, v) in extra.as_object().unwrap() {
            base[k] = v.clone();
        }
        serde_json::to_vec(&base).unwrap()
    }

    fn refused_field(extra: serde_json::Value) -> String {
        match DecoderConfig::from_json_bytes(&tiny_with(extra)) {
            Err(InferError::Unsupported { feature, .. }) => feature,
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn real_llama_block_fields_are_accepted() {
        // The fields Llama-block configs carry, at their supported values.
        // SmolLM2-135M-Instruct's real config states model_type, the
        // architecture, hidden_act, a null rope_scaling and both biases
        // false; head_dim, when a config states it, must equal
        // hidden / heads.
        let json = serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_act": "silu", "rope_scaling": null, "attention_bias": false,
            "mlp_bias": false, "head_dim": 2, "pretraining_tp": 1,
            "sliding_window": null
        });
        DecoderConfig::from_json_bytes(&tiny_with(json)).unwrap();
    }

    #[test]
    fn foreign_model_type_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"model_type": "qwen2"})),
            "config.model_type"
        );
    }

    #[test]
    fn foreign_architecture_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"architectures": ["MistralForCausalLM"]})),
            "config.architectures"
        );
    }

    #[test]
    fn multimodal_nesting_refused_before_missing_fields() {
        // Gemma 3's multimodal config nests the decoder under text_config and
        // has no top-level vocab_size; the refusal names the nesting.
        let json = br#"{"model_type": "llama", "text_config": {"hidden_size": 8}}"#;
        match DecoderConfig::from_json_bytes(json) {
            Err(InferError::Unsupported { feature, .. }) => {
                assert_eq!(feature, "config.text_config")
            },
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn rope_scaling_refused() {
        // Llama 3.1 to 3.3 carry this; unscaled RoPE would answer wrongly.
        let scaling = serde_json::json!({"rope_scaling": {
            "rope_type": "llama3", "factor": 32.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }});
        assert_eq!(refused_field(scaling), "config.rope_scaling");
    }

    #[test]
    fn partial_rotary_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"partial_rotary_factor": 0.75})),
            "config.partial_rotary_factor"
        );
        DecoderConfig::from_json_bytes(&tiny_with(
            serde_json::json!({"partial_rotary_factor": 1.0}),
        ))
        .unwrap();
    }

    #[test]
    fn explicit_head_dim_mismatch_refused() {
        // Qwen3 and Mistral NeMo state head_dim apart from hidden / heads.
        assert_eq!(
            refused_field(serde_json::json!({"head_dim": 4})),
            "config.head_dim"
        );
    }

    #[test]
    fn biases_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"attention_bias": true})),
            "config.attention_bias"
        );
        assert_eq!(
            refused_field(serde_json::json!({"mlp_bias": true})),
            "config.mlp_bias"
        );
    }

    #[test]
    fn effective_sliding_window_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"sliding_window": 8})),
            "config.sliding_window"
        );
        // Disabled, or no shorter than the context, it cannot take effect.
        DecoderConfig::from_json_bytes(&tiny_with(
            serde_json::json!({"sliding_window": 8, "use_sliding_window": false}),
        ))
        .unwrap();
        DecoderConfig::from_json_bytes(&tiny_with(serde_json::json!({"sliding_window": 16})))
            .unwrap();
    }

    #[test]
    fn non_silu_activation_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"hidden_act": "gelu"})),
            "config.hidden_act"
        );
        assert_eq!(
            refused_field(serde_json::json!({"hidden_activation": "gelu_pytorch_tanh"})),
            "config.hidden_activation"
        );
        DecoderConfig::from_json_bytes(&tiny_with(serde_json::json!({"hidden_act": "swish"})))
            .unwrap();
    }

    #[test]
    fn soft_capping_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"attn_logit_softcapping": 50.0})),
            "config.attn_logit_softcapping"
        );
        assert_eq!(
            refused_field(serde_json::json!({"final_logit_softcapping": 30.0})),
            "config.final_logit_softcapping"
        );
    }

    #[test]
    fn experts_refused() {
        for field in [
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "num_experts_per_tok",
        ] {
            let mut extra = serde_json::Map::new();
            extra.insert(field.to_string(), 8.into());
            assert_eq!(
                refused_field(serde_json::Value::Object(extra)),
                format!("config.{field}")
            );
        }
    }

    #[test]
    fn inconsistent_heads_rejected() {
        let json = r#"{
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 3, "max_position_embeddings": 16
        }"#;
        let err = DecoderConfig::from_json_bytes(json.as_bytes()).unwrap_err();
        assert!(matches!(err, InferError::InvalidConfig(_)));
    }
}
