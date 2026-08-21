//! HF-config-driven decoder configuration.
//!
//! Parsed straight from a HuggingFace `config.json` (unknown fields
//! ignored), so one body runs the whole open llama-family class —
//! TinyLlama, Llama 3.2, SmolLM — from the HF layout directly. This is
//! the config the loader slice will pair with `model.safetensors` +
//! `tokenizer.json`, the same artifact triple `embed`'s BERT uses.

use serde::Deserialize;

use crate::infer::provider::InferError;

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
    /// Parse a HF `config.json` byte buffer.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, InferError> {
        let config: Self = serde_json::from_slice(bytes)
            .map_err(|e| InferError::InvalidConfig(format!("config.json parse: {e}")))?;
        config.validate()?;
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
