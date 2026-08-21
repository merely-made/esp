//! BERT configuration. Maps directly onto the HuggingFace `config.json`
//! shape so loading is a one-to-one parse when the loader lands.

use serde::{Deserialize, Serialize};

/// BERT architecture configuration.
///
/// Field names match the HuggingFace `config.json` keys for direct deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    /// Hidden activation. Always "gelu" in practice; reserved for future
    /// alternatives.
    pub hidden_act: String,
    /// Pad token id used in attention masking.
    pub pad_token_id: usize,
}

impl BertConfig {
    /// Per-attention-head dimension: `hidden_size / num_attention_heads`.
    /// Panics if not evenly divisible (a misconfigured model).
    pub fn head_dim(&self) -> usize {
        assert!(
            self.hidden_size.is_multiple_of(self.num_attention_heads),
            "BertConfig: hidden_size ({}) must be divisible by num_attention_heads ({})",
            self.hidden_size,
            self.num_attention_heads
        );
        self.hidden_size / self.num_attention_heads
    }
}

// ─── Open-source-first embedding-model presets ──────────────────────────────
//
// All presets below are Apache 2.0 or MIT licensed. They define the
// architectural shape (vocab/hidden/layers/heads/intermediate/max_pos) of
// each model so callers can validate a loaded `config.json` against the
// expected preset, OR construct a random-weight model with the right shape
// for testing. The actual weights still come from each model's
// `model.safetensors` via `BertEmbeddingProvider::load(model_dir)`.
//
// **Drop-in compatibility**: these presets all share BERT's vanilla
// architecture (token + position + token-type embeddings, multi-head
// scaled-dot-product attention with absolute positional encoding,
// FFN with GELU, LayerNorm pre-residual). Models with architectural
// extensions (e.g. Nomic's rotary attention, Matryoshka resizing) are
// noted in the docs but not yet supported as drop-in presets — they
// require additional work in `bert/embeddings.rs` and `bert/attention.rs`.
//
// **License-first ordering**: Apache 2.0 / MIT models surfaced as
// presets here; Meta-licensed models (Llama family) deliberately not
// surfaced as presets because their licenses aren't OSI-approved.

/// Configuration for `sentence-transformers/all-MiniLM-L6-v2` — the
/// recommended first-bundled embedding model (Apache 2.0, ~22 MB on disk,
/// 384-dim output, 6 layers).
///
/// Empirically validated: Burn-side BERT matches HF reference vectors
/// within 1e-4 across reference texts. See `bert/validation.rs`.
pub const MINILM_L6_V2: BertConfig = BertConfig {
    vocab_size: 30_522,
    hidden_size: 384,
    num_hidden_layers: 6,
    num_attention_heads: 12,
    intermediate_size: 1_536,
    max_position_embeddings: 512,
    type_vocab_size: 2,
    layer_norm_eps: 1.0e-12,
    hidden_act: String::new(), // "gelu" — populated by deserialization
    pad_token_id: 0,
};

/// Configuration for `Snowflake/snowflake-arctic-embed-xs` — Apache 2.0,
/// 384-dim output, 22 MB on disk. Same hidden-size as MiniLM but a deeper
/// 12-layer stack; trained with retrieval-tuned objectives that often
/// outperform MiniLM on search tasks despite the same parameter count
/// being in a thinner range. Drop-in compatible with `BertEmbeddingProvider`.
pub const SNOWFLAKE_ARCTIC_EMBED_XS: BertConfig = BertConfig {
    vocab_size: 30_522,
    hidden_size: 384,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 1_536,
    max_position_embeddings: 512,
    type_vocab_size: 2,
    layer_norm_eps: 1.0e-12,
    hidden_act: String::new(),
    pad_token_id: 0,
};

/// Configuration for `TaylorAI/bge-micro-v2` — MIT, 384-dim output,
/// ~17 MB on disk. The smallest viable BERT-class embedding model;
/// only 3 transformer layers. Trade quality for size — useful where
/// disk/RAM are tight (mobile, low-power, CDN-bundled). Drop-in
/// compatible with `BertEmbeddingProvider`.
pub const BGE_MICRO_V2: BertConfig = BertConfig {
    vocab_size: 30_522,
    hidden_size: 384,
    num_hidden_layers: 3,
    num_attention_heads: 12,
    intermediate_size: 1_536,
    max_position_embeddings: 512,
    type_vocab_size: 2,
    layer_norm_eps: 1.0e-12,
    hidden_act: String::new(),
    pad_token_id: 0,
};

// ─── Notes on near-term-but-not-yet-drop-in models ──────────────────────────
//
// Surfaced here for documentation; not yet supported as `BertConfig` presets
// because they require architectural extensions beyond vanilla BERT.
//
// - **`nomic-ai/nomic-embed-text-v1.5`** (Apache 2.0, 768-dim, 12 layers,
//   ~137 MB). Uses *rotary positional embeddings* instead of absolute
//   position lookups, plus *Matryoshka representation learning* (variable
//   output dimension at inference time). Both are real architectural
//   changes — would require new `BertEmbeddings::forward` paths and
//   `BertSelfAttention` rotary-Q/K rotation. Worth wiring when a longer
//   context (8 K) or multilingual story becomes a real product pull.
//
// - **`sentence-transformers/LaBSE`** (Apache 2.0, multilingual, 768-dim,
//   ~470 MB). Vanilla BERT architecture — should work as a drop-in once
//   we want a multilingual default. Larger; not the bundled default but
//   a strong on-demand option.
//
// - **`thenlper/gte-small`**, **`intfloat/e5-small-v2`**, **`mixedbread-ai/
//   mxbai-embed-xsmall-v1`** — all BERT-class, all Apache 2.0 or MIT,
//   all should drop in. Add presets opportunistically as users want them.

#[cfg(test)]
mod tests {
    use super::*;

    fn minilm_clone() -> BertConfig {
        let mut c = MINILM_L6_V2.clone();
        c.hidden_act = "gelu".to_string();
        c
    }

    #[test]
    fn minilm_l6_v2_shape() {
        let c = minilm_clone();
        assert_eq!(c.vocab_size, 30_522);
        assert_eq!(c.hidden_size, 384);
        assert_eq!(c.num_hidden_layers, 6);
        assert_eq!(c.num_attention_heads, 12);
        assert_eq!(c.intermediate_size, 1_536);
        assert_eq!(c.max_position_embeddings, 512);
    }

    #[test]
    fn head_dim_is_integer() {
        let c = minilm_clone();
        assert_eq!(c.head_dim(), 32);
        // 384 / 12 = 32 — confirms even division.
    }

    #[test]
    fn config_serde_roundtrip() {
        let c = minilm_clone();
        let json = serde_json::to_string(&c).unwrap();
        let back: BertConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn config_deserialises_huggingface_style() {
        // A config.json shape from HF is a strict superset of our fields;
        // here we test the subset we read.
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 1536,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
            "pad_token_id": 0
        }"#;
        let c: BertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.head_dim(), 32);
    }

    #[test]
    fn snowflake_arctic_embed_xs_shape() {
        let c = SNOWFLAKE_ARCTIC_EMBED_XS.clone();
        assert_eq!(c.hidden_size, 384);
        assert_eq!(c.num_hidden_layers, 12); // deeper than MiniLM-L6
        assert_eq!(c.head_dim(), 32);
    }

    #[test]
    fn bge_micro_v2_shape() {
        let c = BGE_MICRO_V2.clone();
        assert_eq!(c.hidden_size, 384);
        assert_eq!(c.num_hidden_layers, 3); // smallest viable BERT
        assert_eq!(c.head_dim(), 32);
    }

    #[test]
    fn open_source_presets_share_compatible_dimensions() {
        // All three presets produce 384-dim output — interchangeable in
        // any consumer that uses the EmbeddingProvider trait without
        // re-indexing. Switching presets means swapping which model dir
        // `BertEmbeddingProvider::load` points at; existing vectors
        // computed with one preset are invalid against another (the
        // numerical space differs even at matching dimensionality), so
        // re-embedding is required, but the trait surface is unchanged.
        assert_eq!(
            MINILM_L6_V2.hidden_size,
            SNOWFLAKE_ARCTIC_EMBED_XS.hidden_size
        );
        assert_eq!(MINILM_L6_V2.hidden_size, BGE_MICRO_V2.hidden_size);
    }

    #[test]
    fn open_source_presets_serde_roundtrip() {
        for preset in [&MINILM_L6_V2, &SNOWFLAKE_ARCTIC_EMBED_XS, &BGE_MICRO_V2] {
            let mut c = preset.clone();
            c.hidden_act = "gelu".to_string();
            let json = serde_json::to_string(&c).unwrap();
            let back: BertConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(c, back);
        }
    }
}
