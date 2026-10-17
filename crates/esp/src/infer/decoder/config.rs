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
//! Supported blocks use pre-norm RMSNorm, grouped-query attention, split-half
//! RoPE and SwiGLU. Declared variants add linear/Llama3 scaling, independent
//! head width, sliding windows, biases, QK norms and Qwen2 routed/shared
//! experts. Unknown families, scaling modes and unread tensors remain errors.

use serde::Deserialize;
use serde_json::Value;

use crate::infer::provider::InferError;

/// `model_type` values this decoder runs.
const SUPPORTED_MODEL_TYPES: &[&str] = &["llama", "mistral", "qwen2", "qwen3", "qwen2_moe"];

/// `architectures` entries this decoder runs.
const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM",
];

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "one")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}
fn one() -> usize {
    1
}

/// Supported static frequency transforms. Dynamic and YaRN variants remain
/// refused rather than using a checkpoint with the wrong positional encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    Linear {
        factor: f32,
    },
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: usize,
    },
}

impl RopeScaling {
    fn parse(value: &Value) -> Result<Self, InferError> {
        let object = value
            .as_object()
            .ok_or_else(|| unsupported("rope_scaling", "expected an object"))?;
        let kind = object
            .get("rope_type")
            .or_else(|| object.get("type"))
            .and_then(Value::as_str)
            .ok_or_else(|| unsupported("rope_scaling", "missing rope_type"))?;
        if object
            .get("type")
            .zip(object.get("rope_type"))
            .is_some_and(|(a, b)| a != b)
        {
            return Err(unsupported(
                "rope_scaling",
                "conflicting type and rope_type",
            ));
        }
        let number = |key: &str| -> Result<f32, InferError> {
            let n = object.get(key).and_then(Value::as_f64).unwrap_or(f64::NAN) as f32;
            if !n.is_finite() || n <= 0.0 {
                return Err(unsupported(
                    "rope_scaling",
                    format!("{key} must be positive and finite"),
                ));
            }
            Ok(n)
        };
        let allowed: &[&str] = match kind {
            "linear" => &["type", "rope_type", "factor"],
            "llama3" => &[
                "type",
                "rope_type",
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            ],
            _ => {
                return Err(unsupported(
                    "rope_scaling",
                    format!("{kind:?}; supported: linear, llama3"),
                ));
            },
        };
        if object.keys().any(|key| !allowed.contains(&key.as_str())) {
            return Err(unsupported(
                "rope_scaling",
                "unimplemented scaling parameters",
            ));
        }
        let factor = number("factor")?;
        if factor < 1.0 {
            return Err(unsupported("rope_scaling", "factor must be at least one"));
        }
        match kind {
            "linear" => Ok(Self::Linear { factor }),
            "llama3" => {
                let low_freq_factor = number("low_freq_factor")?;
                let high_freq_factor = number("high_freq_factor")?;
                let original_max_position_embeddings = object
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| *n > 0)
                    .ok_or_else(|| {
                        unsupported(
                            "rope_scaling",
                            "original context must be a positive integer",
                        )
                    })?;
                if high_freq_factor <= low_freq_factor {
                    return Err(unsupported(
                        "rope_scaling",
                        "high frequency factor must exceed low frequency factor",
                    ));
                }
                Ok(Self::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings,
                })
            },
            _ => unreachable!(),
        }
    }

    pub(crate) fn apply(&self, frequency: f32) -> f32 {
        match *self {
            Self::Linear { factor } => frequency / factor,
            Self::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                let wavelength = std::f32::consts::TAU / frequency;
                let original = original_max_position_embeddings as f32;
                if wavelength < original / high_freq_factor {
                    return frequency;
                }
                if wavelength > original / low_freq_factor {
                    return frequency / factor;
                }
                let smooth = (original / wavelength - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                (1.0 - smooth) * frequency / factor + smooth * frequency
            },
        }
    }
}

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
    mlp_bias: Option<bool>,
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
        if let Some(factor) = self.partial_rotary_factor
            && factor != 1.0
        {
            return Err(unsupported(
                "partial_rotary_factor",
                format!("{factor}; RoPE is applied to the full head"),
            ));
        }
        if self.mlp_bias == Some(true) {
            return Err(unsupported("mlp_bias", "MLP projections are bias-free"));
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
            if config.moe.is_some() && matches!(field, "num_experts" | "num_experts_per_tok") {
                continue;
            }
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
    #[serde(skip)]
    pub moe: Option<MoeConfig>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    #[serde(default)]
    pub max_window_layers: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(skip)]
    pub rope_scaling: Option<RopeScaling>,
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
        let mut config: Self = serde_json::from_slice(bytes).map_err(parse)?;
        if let Some(architectures) = &architecture.architectures {
            for name in architectures {
                let family = match name.as_str() {
                    "LlamaForCausalLM" => "llama",
                    "MistralForCausalLM" => "mistral",
                    "Qwen2ForCausalLM" => "qwen2",
                    "Qwen3ForCausalLM" => "qwen3",
                    "Qwen2MoeForCausalLM" => "qwen2_moe",
                    _ => unreachable!("foreign architecture refused"),
                };
                if config
                    .model_type
                    .as_deref()
                    .is_some_and(|kind| kind != family)
                {
                    return Err(unsupported(
                        "architectures",
                        "architecture and model_type disagree",
                    ));
                }
                config.model_type = Some(family.into());
            }
        }
        config.rope_scaling = architecture
            .rope_scaling
            .as_ref()
            .map(RopeScaling::parse)
            .transpose()?;
        if config.model_type.as_deref() == Some("qwen2_moe") {
            config.moe = Some(serde_json::from_slice(bytes).map_err(parse)?);
        }
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
        self.head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    pub fn query_width(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    pub fn qkv_bias(&self) -> bool {
        matches!(self.model_type.as_deref(), Some("qwen2" | "qwen2_moe")) || self.attention_bias
    }

    pub fn output_bias(&self) -> bool {
        !matches!(self.model_type.as_deref(), Some("qwen2" | "qwen2_moe")) && self.attention_bias
    }

    pub fn qk_norm(&self) -> bool {
        self.model_type.as_deref() == Some("qwen3")
    }

    pub fn layer_is_moe(&self, index: usize) -> bool {
        self.moe.as_ref().is_some_and(|m| {
            !m.mlp_only_layers.contains(&index) && (index + 1).is_multiple_of(m.decoder_sparse_step)
        })
    }

    pub fn layer_window(&self, index: usize) -> Option<usize> {
        if let Some(types) = &self.layer_types {
            return (types[index] == "sliding_attention")
                .then_some(self.sliding_window)
                .flatten();
        }
        if self.use_sliding_window == Some(false) {
            return None;
        }
        if matches!(
            self.model_type.as_deref(),
            Some("qwen2" | "qwen3" | "qwen2_moe")
        ) {
            return (self.use_sliding_window == Some(true)
                && index >= self.max_window_layers.unwrap_or(28))
            .then_some(self.sliding_window)
            .flatten();
        }
        self.sliding_window
    }

    /// Query heads served by each key/value head.
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.kv_heads()
    }

    fn validate(&self) -> Result<(), InferError> {
        if self.num_attention_heads == 0
            || (self.head_dim.is_none()
                && !self.hidden_size.is_multiple_of(self.num_attention_heads))
        {
            return Err(InferError::InvalidConfig(format!(
                "hidden_size {} not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.kv_heads() == 0 || !self.num_attention_heads.is_multiple_of(self.kv_heads()) {
            return Err(InferError::InvalidConfig(format!(
                "num_attention_heads {} not divisible by num_key_value_heads {}",
                self.num_attention_heads,
                self.kv_heads()
            )));
        }
        if self.head_dim() == 0 || !self.head_dim().is_multiple_of(2) {
            return Err(InferError::InvalidConfig(format!(
                "head_dim {} must be even for rotary encoding",
                self.head_dim()
            )));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.num_hidden_layers == 0
            || self.max_position_embeddings == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || self.sliding_window == Some(0)
            || self
                .num_attention_heads
                .checked_mul(self.head_dim())
                .is_none()
            || self.kv_heads().checked_mul(self.head_dim()).is_none()
        {
            return Err(InferError::InvalidConfig(
                "zero, non-finite or overflowing decoder dimensions/settings".into(),
            ));
        }
        if let Some(types) = &self.layer_types
            && (types.len() != self.num_hidden_layers
                || types.iter().any(|kind| {
                    kind != "full_attention" && kind != "sliding_attention"
                        || kind == "sliding_attention" && self.sliding_window.is_none()
                }))
        {
            return Err(unsupported(
                "layer_types",
                "expected one supported attention type per layer and a window for sliding layers",
            ));
        }
        if let Some(moe) = &self.moe
            && (moe.num_experts == 0
                || moe.num_experts_per_tok == 0
                || moe.num_experts_per_tok > moe.num_experts
                || moe.moe_intermediate_size == 0
                || moe.shared_expert_intermediate_size == 0
                || moe.decoder_sparse_step == 0
                || moe
                    .mlp_only_layers
                    .iter()
                    .any(|index| *index >= self.num_hidden_layers)
                || moe
                    .num_experts
                    .checked_mul(moe.moe_intermediate_size)
                    .and_then(|n| n.checked_mul(self.hidden_size))
                    .is_none())
        {
            return Err(unsupported(
                "experts",
                "invalid expert count, routing, intermediate size or sparse-layer selection",
            ));
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
            refused_field(serde_json::json!({"model_type": "gemma2"})),
            "config.model_type"
        );
    }

    #[test]
    fn foreign_architecture_refused() {
        assert_eq!(
            refused_field(serde_json::json!({"architectures": ["GemmaForCausalLM"]})),
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
    fn llama3_scaling_is_validated_and_unknown_scaling_is_refused() {
        let scaling = serde_json::json!({"rope_scaling": {
            "rope_type": "llama3", "factor": 32.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }});
        assert!(matches!(
            DecoderConfig::from_json_bytes(&tiny_with(scaling))
                .unwrap()
                .rope_scaling,
            Some(RopeScaling::Llama3 { .. })
        ));
        assert_eq!(
            refused_field(serde_json::json!({"rope_scaling":{"rope_type":"yarn", "factor": 4}})),
            "config.rope_scaling"
        );
        assert_eq!(
            refused_field(
                serde_json::json!({"rope_scaling":{"rope_type":"llama3", "factor": 4,"low_freq_factor":4,"high_freq_factor":1,"original_max_position_embeddings":8192}})
            ),
            "config.rope_scaling"
        );
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
    fn independent_query_width_and_zero_kv_heads_are_validated() {
        let c =
            DecoderConfig::from_json_bytes(&tiny_with(serde_json::json!({"head_dim": 4}))).unwrap();
        assert_eq!(c.query_width(), 16);
        assert!(
            DecoderConfig::from_json_bytes(&tiny_with(
                serde_json::json!({"num_key_value_heads":0})
            ))
            .is_err()
        );
    }

    #[test]
    fn biases_refused() {
        assert!(
            DecoderConfig::from_json_bytes(&tiny_with(serde_json::json!({"attention_bias":true})))
                .unwrap()
                .output_bias()
        );
        assert_eq!(
            refused_field(serde_json::json!({"mlp_bias": true})),
            "config.mlp_bias"
        );
    }

    #[test]
    fn effective_and_disabled_sliding_windows() {
        let c = DecoderConfig::from_json_bytes(&tiny_with(serde_json::json!({"sliding_window":8})))
            .unwrap();
        assert_eq!(c.layer_window(0), Some(8));
        // Disabled, or no shorter than the context, it cannot take effect.
        DecoderConfig::from_json_bytes(&tiny_with(
            serde_json::json!({"sliding_window": 8, "use_sliding_window": false}),
        ))
        .unwrap();
        DecoderConfig::from_json_bytes(&tiny_with(serde_json::json!({"sliding_window": 16})))
            .unwrap();
        assert_eq!(
            refused_field(serde_json::json!({"layer_types":["mystery","full_attention"]})),
            "config.layer_types"
        );
        assert_eq!(
            refused_field(
                serde_json::json!({"model_type":"qwen3","architectures":["LlamaForCausalLM"]})
            ),
            "config.architectures"
        );
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
