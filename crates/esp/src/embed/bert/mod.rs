// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Burn-backed BERT embedding provider.
//!
//! A full BERT-class sentence embedder on [`burn`], behind the `bert` feature
//! (`bert-wgpu` adds GPU execution). The pipeline is `text → word-piece tokenize →
//! embeddings + positions + LayerNorm → N encoder layers (multi-head attention +
//! GELU feed-forward) → sentence pooling ([CLS] or mean) → L2-normalize`, exposed
//! through the [`EmbeddingProvider`](crate::embed::EmbeddingProvider) trait as
//! [`BertEmbeddingProvider`]. Weights load from safetensors (HuggingFace layout).
//!
//! Modules: [`config`] (model configs — MiniLM-L6, BGE-micro, arctic-xs),
//! [`embeddings`] / [`attention`] / [`feed_forward`] / [`layer`] / [`encoder`] /
//! [`model`] (the network), [`tokenizer`] (word-piece), [`loader`] +
//! [`safetensors_io`] + [`loaded`] (weight loading from bytes or a directory),
//! [`provider`] (the trait impl), and [`validation`] (an ignored end-to-end check
//! against a reference `sentence-transformers` run).
//!
//! An unloaded provider's [`embed`](BertEmbeddingProvider::embed) returns
//! [`ModelNotLoaded`](crate::embed::EmbedError::ModelNotLoaded); a loaded one produces
//! real vectors.

pub mod attention;
pub mod config;
pub mod construct;
pub mod embeddings;
pub mod encoder;
pub mod feed_forward;
pub mod layer;
pub mod loaded;
pub mod loader;
pub mod model;
pub mod provider;
pub mod safetensors_io;
pub mod tokenizer;
pub mod validation;
#[cfg(all(test, feature = "bert-wgpu"))]
mod wgpu_parity;

pub use attention::{BertAttention, BertSelfAttention, BertSelfOutput};
pub use config::{BGE_MICRO_V2, BertConfig, MINILM_L6_V2, SNOWFLAKE_ARCTIC_EMBED_XS};
pub use embeddings::BertEmbeddings;
pub use encoder::BertEncoder;
pub use feed_forward::{BertIntermediate, BertOutput};
pub use layer::BertLayer;
pub use loader::{LoaderError, ModelArtifacts};
pub use model::{BertModel, Pooling};
#[cfg(feature = "bert-wgpu")]
pub use provider::load_wgpu;
pub use provider::{BertEmbeddingProvider, from_bytes_cpu, load_cpu};
pub use tokenizer::{BertTokenizer, EncodedBatch};
