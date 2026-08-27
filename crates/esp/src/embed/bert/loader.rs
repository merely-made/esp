// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Load a BERT model from a HuggingFace-style on-disk layout.
//!
//! Expected layout (matching `sentence-transformers/all-MiniLM-L6-v2`):
//!
//! ```text
//! <model_dir>/
//!   config.json          → BertConfig
//!   tokenizer.json       → BertTokenizer
//!   model.safetensors    → weights
//! ```
//!
//! This module ships:
//!
//! - **`ModelArtifacts`** — struct holding the parsed config, tokenizer,
//!   and the available safetensors tensor names (for verification).
//! - **`load_artifacts(model_dir)`** — parses `config.json` and
//!   `tokenizer.json`, opens `model.safetensors` and enumerates its
//!   tensor names. Returns the artifacts ready for inspection.
//! - **`load_into_model(artifacts, model)`** — **stub**. The
//!   safetensors → Burn `Param<Tensor>` injection is its own focused
//!   slice; this function returns [`LoaderError::WeightLoadingNotImplemented`]
//!   today.
//!
//! ## HuggingFace → Burn tensor name map
//!
//! When the weight-injection slice lands, it needs to map each HF
//! safetensors key to the corresponding Burn module field path. The map
//! follows directly from the module hierarchy this crate already
//! implements:
//!
//! | HuggingFace key | Burn field path |
//! | --- | --- |
//! | `embeddings.word_embeddings.weight` | `embeddings.word_embeddings.weight` |
//! | `embeddings.position_embeddings.weight` | `embeddings.position_embeddings.weight` |
//! | `embeddings.token_type_embeddings.weight` | `embeddings.token_type_embeddings.weight` |
//! | `embeddings.LayerNorm.weight` | `embeddings.layer_norm.gamma` |
//! | `embeddings.LayerNorm.bias` | `embeddings.layer_norm.beta` |
//! | `encoder.layer.{i}.attention.self.query.weight` | `encoder.layers[i].attention.self_attention.query.weight` |
//! | `encoder.layer.{i}.attention.self.query.bias` | `encoder.layers[i].attention.self_attention.query.bias` |
//! | `encoder.layer.{i}.attention.self.key.{weight,bias}` | `…attention.self_attention.key.{…}` |
//! | `encoder.layer.{i}.attention.self.value.{weight,bias}` | `…attention.self_attention.value.{…}` |
//! | `encoder.layer.{i}.attention.output.dense.{weight,bias}` | `…attention.output.dense.{…}` |
//! | `encoder.layer.{i}.attention.output.LayerNorm.{weight,bias}` | `…attention.output.layer_norm.{gamma,beta}` |
//! | `encoder.layer.{i}.intermediate.dense.{weight,bias}` | `…intermediate.dense.{…}` |
//! | `encoder.layer.{i}.output.dense.{weight,bias}` | `…output.dense.{…}` |
//! | `encoder.layer.{i}.output.LayerNorm.{weight,bias}` | `…output.layer_norm.{gamma,beta}` |
//!
//! Caveat: PyTorch `Linear.weight` is stored as `[out, in]`; Burn's
//! `Linear.weight` may be transposed. Verify with one tensor before
//! trusting the bulk load.

use std::path::{Path, PathBuf};

use safetensors::SafeTensors;

use super::config::BertConfig;
use super::tokenizer::BertTokenizer;
use crate::embed::provider::EmbedError;
use burn::tensor::Device;

/// Reasons artifact loading or weight injection could fail.
#[derive(Debug)]
pub enum LoaderError {
    /// The model directory was missing one of the expected files.
    MissingFile(PathBuf),
    /// `config.json` could not be parsed as a [`BertConfig`].
    InvalidConfig(String),
    /// The tokenizer file could not be parsed.
    InvalidTokenizer(String),
    /// The safetensors file could not be opened or parsed.
    InvalidWeights(String),
    /// I/O error reading one of the files.
    Io(std::io::Error),
    /// Weight injection (safetensors → Burn `Param<Tensor>`) is not yet
    /// implemented; the validation slice fills this in.
    WeightLoadingNotImplemented,
}

impl From<std::io::Error> for LoaderError {
    fn from(e: std::io::Error) -> Self {
        LoaderError::Io(e)
    }
}

impl From<LoaderError> for EmbedError {
    fn from(e: LoaderError) -> Self {
        EmbedError::Backend(format!("loader error: {e:?}"))
    }
}

/// Parsed artifacts ready for model-injection. The actual weight bytes
/// remain inside `weights_path` until injection is implemented; this
/// struct holds the *names* of available tensors as a sanity surface.
#[derive(Debug)]
pub struct ModelArtifacts {
    pub config: BertConfig,
    pub tokenizer: BertTokenizer,
    pub weights_path: PathBuf,
    pub tensor_names: Vec<String>,
}

/// Read `config.json`, `tokenizer.json`, and enumerate
/// `model.safetensors` tensor names from a model directory.
pub fn load_artifacts(model_dir: impl AsRef<Path>) -> Result<ModelArtifacts, LoaderError> {
    let dir = model_dir.as_ref();
    let config_path = dir.join("config.json");
    let tokenizer_path = dir.join("tokenizer.json");
    let weights_path = dir.join("model.safetensors");

    if !config_path.exists() {
        return Err(LoaderError::MissingFile(config_path));
    }
    if !tokenizer_path.exists() {
        return Err(LoaderError::MissingFile(tokenizer_path));
    }
    if !weights_path.exists() {
        return Err(LoaderError::MissingFile(weights_path));
    }

    let config_bytes = std::fs::read(&config_path)?;
    let tokenizer_bytes = std::fs::read(&tokenizer_path)?;
    let weights_bytes = std::fs::read(&weights_path)?;

    artifacts_from_bytes_with_weights_path(
        &config_bytes,
        &tokenizer_bytes,
        &weights_bytes,
        weights_path,
    )
}

/// Build [`ModelArtifacts`] from in-memory bytes — the path through which
/// host-resolved model blobs flow.
///
/// Accepts the three HuggingFace artifacts (config.json, tokenizer.json,
/// model.safetensors) as byte buffers. The returned `ModelArtifacts` carries
/// an empty `weights_path`; weight injection ([`load_into_model_from_bytes`])
/// works directly off the buffer rather than re-reading from disk.
pub fn artifacts_from_bytes(
    config_bytes: &[u8],
    tokenizer_bytes: &[u8],
    weights_bytes: &[u8],
) -> Result<ModelArtifacts, LoaderError> {
    artifacts_from_bytes_with_weights_path(
        config_bytes,
        tokenizer_bytes,
        weights_bytes,
        PathBuf::new(),
    )
}

fn artifacts_from_bytes_with_weights_path(
    config_bytes: &[u8],
    tokenizer_bytes: &[u8],
    weights_bytes: &[u8],
    weights_path: PathBuf,
) -> Result<ModelArtifacts, LoaderError> {
    let config: BertConfig = serde_json::from_slice(config_bytes)
        .map_err(|e| LoaderError::InvalidConfig(format!("{e}")))?;

    let tokenizer = BertTokenizer::from_bytes(
        tokenizer_bytes,
        config.max_position_embeddings,
        config.pad_token_id as u32,
    )
    .map_err(|e| match e {
        EmbedError::InvalidConfig(msg) => LoaderError::InvalidTokenizer(msg),
        other => LoaderError::InvalidTokenizer(format!("{other:?}")),
    })?;

    let safetensors = SafeTensors::deserialize(weights_bytes)
        .map_err(|e| LoaderError::InvalidWeights(format!("{e}")))?;
    let tensor_names = safetensors.names().into_iter().map(String::from).collect();

    Ok(ModelArtifacts {
        config,
        tokenizer,
        weights_path,
        tensor_names,
    })
}

/// Specification for one expected tensor in a HF BERT safetensors file:
/// the dotted HF name and the rank-erased shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Enumerate every tensor a HF BERT safetensors file should contain for
/// the given config.
///
/// The list is the canonical input to weight loading: it defines which HF
/// names the loader expects and what shape each should have. For
/// `MINILM_L6_V2` (6 layers, hidden=384, intermediate=1536) this returns
/// 5 + 6×16 = **101 tensors**.
pub fn expected_tensors(config: &BertConfig) -> Vec<TensorSpec> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let mut out = vec![
        TensorSpec {
            name: "embeddings.word_embeddings.weight".to_string(),
            shape: vec![config.vocab_size, h],
        },
        TensorSpec {
            name: "embeddings.position_embeddings.weight".to_string(),
            shape: vec![config.max_position_embeddings, h],
        },
        TensorSpec {
            name: "embeddings.token_type_embeddings.weight".to_string(),
            shape: vec![config.type_vocab_size, h],
        },
        TensorSpec {
            name: "embeddings.LayerNorm.weight".to_string(),
            shape: vec![h],
        },
        TensorSpec {
            name: "embeddings.LayerNorm.bias".to_string(),
            shape: vec![h],
        },
    ];
    for i in 0..config.num_hidden_layers {
        let p = format!("encoder.layer.{i}");
        // Q/K/V projections.
        for proj in ["query", "key", "value"] {
            out.push(TensorSpec {
                name: format!("{p}.attention.self.{proj}.weight"),
                shape: vec![h, h],
            });
            out.push(TensorSpec {
                name: format!("{p}.attention.self.{proj}.bias"),
                shape: vec![h],
            });
        }
        // Self-attention output (dense + LayerNorm).
        out.push(TensorSpec {
            name: format!("{p}.attention.output.dense.weight"),
            shape: vec![h, h],
        });
        out.push(TensorSpec {
            name: format!("{p}.attention.output.dense.bias"),
            shape: vec![h],
        });
        out.push(TensorSpec {
            name: format!("{p}.attention.output.LayerNorm.weight"),
            shape: vec![h],
        });
        out.push(TensorSpec {
            name: format!("{p}.attention.output.LayerNorm.bias"),
            shape: vec![h],
        });
        // Intermediate (expansion).
        out.push(TensorSpec {
            name: format!("{p}.intermediate.dense.weight"),
            shape: vec![inter, h],
        });
        out.push(TensorSpec {
            name: format!("{p}.intermediate.dense.bias"),
            shape: vec![inter],
        });
        // Output (contraction + LayerNorm).
        out.push(TensorSpec {
            name: format!("{p}.output.dense.weight"),
            shape: vec![h, inter],
        });
        out.push(TensorSpec {
            name: format!("{p}.output.dense.bias"),
            shape: vec![h],
        });
        out.push(TensorSpec {
            name: format!("{p}.output.LayerNorm.weight"),
            shape: vec![h],
        });
        out.push(TensorSpec {
            name: format!("{p}.output.LayerNorm.bias"),
            shape: vec![h],
        });
    }
    out
}

/// Reasons the weights file fails validation against the config's
/// expected tensor list.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationIssue {
    Missing(String),
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    UnsupportedDtype {
        name: String,
        actual: String,
    },
}

/// Validate a safetensors weights file against the expected tensor
/// list for a config. Returns the list of issues; an empty list means
/// the file is structurally compatible.
///
/// Path-based wrapper: reads `artifacts.weights_path` from disk, then
/// delegates to [`validate_weights_from_bytes`].
pub fn validate_weights(artifacts: &ModelArtifacts) -> Result<Vec<ValidationIssue>, LoaderError> {
    let bytes = std::fs::read(&artifacts.weights_path)?;
    validate_weights_from_bytes(&artifacts.config, &bytes)
}

/// Validate a safetensors weights byte buffer against the expected tensor
/// list for a config. Returns the list of issues; an empty list means the
/// buffer is structurally compatible.
pub fn validate_weights_from_bytes(
    config: &BertConfig,
    weights_bytes: &[u8],
) -> Result<Vec<ValidationIssue>, LoaderError> {
    let safetensors = SafeTensors::deserialize(weights_bytes)
        .map_err(|e| LoaderError::InvalidWeights(format!("{e}")))?;

    let expected = expected_tensors(config);
    let mut issues = Vec::new();
    for spec in &expected {
        let view = match safetensors.tensor(&spec.name) {
            Ok(v) => v,
            Err(_) => {
                issues.push(ValidationIssue::Missing(spec.name.clone()));
                continue;
            }
        };
        let actual_shape: Vec<usize> = view.shape().to_vec();
        if actual_shape != spec.shape {
            issues.push(ValidationIssue::ShapeMismatch {
                name: spec.name.clone(),
                expected: spec.shape.clone(),
                actual: actual_shape,
            });
        }
        if !matches!(
            view.dtype(),
            safetensors::tensor::Dtype::F32 | safetensors::tensor::Dtype::F16
        ) {
            issues.push(ValidationIssue::UnsupportedDtype {
                name: spec.name.clone(),
                actual: format!("{:?}", view.dtype()),
            });
        }
    }
    Ok(issues)
}

/// Load a complete `BertModel` from disk.
///
/// Steps:
///
/// 1. Validate the safetensors file structure against
///    [`expected_tensors`] for the artifacts' config.
/// 2. Extract every tensor into a typed [`super::loaded::LoadedBert`]
///    bundle via [`super::loaded::extract_all_tensors`].
/// 3. Construct the model via [`super::loaded::LoadedBert::into_model`],
///    which composes per-layer `from_loaded` constructors that bypass
///    Burn's random initialisation and inject the loaded tensors via
///    direct `Param::from_tensor` field assignment on Burn's nn
///    primitives (`Linear.weight`, `Embedding.weight`,
///    `LayerNorm.gamma/beta`).
///
/// Caveats not yet hard-verified:
///
/// - **PyTorch `Linear.weight` orientation**: HF stores `[out, in]`;
///   Burn expects `[in, out]`. We extract with `[in_features, out_features]`
///   from `expected_tensors` (matching Burn's order), so this is a
///   transpose that needs verification on the first real forward pass
///   against HF reference. If MILe forward outputs disagree, transpose
///   the `*_w` Linear tensors at the extraction boundary in
///   [`super::loaded::extract_all_tensors`].
/// - **LayerNorm field naming**: HF `weight`/`bias`; Burn `gamma`/`beta`.
///   Already mapped at the [`super::construct::layer_norm_from_loaded`]
///   boundary.
pub fn load_into_model(
    artifacts: &ModelArtifacts,
    device: &Device,
) -> Result<super::model::BertModel, LoaderError> {
    let bytes = std::fs::read(&artifacts.weights_path)?;
    load_into_model_from_bytes(&artifacts.config, &bytes, device)
}

/// Same as [`load_into_model`] but takes the safetensors weights as an
/// in-memory byte buffer. The path host-resolved model blobs flow
/// through.
pub fn load_into_model_from_bytes(
    config: &BertConfig,
    weights_bytes: &[u8],
    device: &Device,
) -> Result<super::model::BertModel, LoaderError> {
    let issues = validate_weights_from_bytes(config, weights_bytes)?;
    if !issues.is_empty() {
        return Err(LoaderError::InvalidWeights(format!(
            "weights file does not match config — {} issues: {:?}",
            issues.len(),
            issues.iter().take(5).collect::<Vec<_>>()
        )));
    }
    let loaded = super::loaded::extract_all_tensors_from_bytes(config, weights_bytes, device)?;
    Ok(loaded.into_model(device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::bert::config::MINILM_L6_V2;

    fn minilm_config() -> BertConfig {
        let mut c = MINILM_L6_V2.clone();
        c.hidden_act = "gelu".to_string();
        c
    }

    #[test]
    fn missing_directory_returns_missing_file() {
        let result = load_artifacts("/nonexistent/path/to/model");
        assert!(matches!(result, Err(LoaderError::MissingFile(_))));
    }

    #[test]
    fn loader_error_implements_into_embed_error() {
        let _e: EmbedError = LoaderError::WeightLoadingNotImplemented.into();
    }

    #[test]
    fn expected_tensors_count_matches_minilm_l6() {
        // 5 embedding-layer tensors + 6 layers × 16 tensors-per-layer = 101.
        let specs = expected_tensors(&minilm_config());
        assert_eq!(specs.len(), 5 + 6 * 16);
    }

    #[test]
    fn expected_tensors_includes_canonical_names() {
        let specs = expected_tensors(&minilm_config());
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"embeddings.word_embeddings.weight"));
        assert!(names.contains(&"embeddings.LayerNorm.bias"));
        assert!(names.contains(&"encoder.layer.0.attention.self.query.weight"));
        assert!(names.contains(&"encoder.layer.5.output.LayerNorm.bias"));
    }

    #[test]
    fn expected_tensors_word_embedding_shape() {
        let specs = expected_tensors(&minilm_config());
        let word_emb = specs
            .iter()
            .find(|s| s.name == "embeddings.word_embeddings.weight")
            .unwrap();
        assert_eq!(word_emb.shape, vec![30_522, 384]);
    }

    #[test]
    fn expected_tensors_attention_qkv_shapes_match_hidden() {
        let specs = expected_tensors(&minilm_config());
        let qkv: Vec<&TensorSpec> = specs
            .iter()
            .filter(|s| {
                s.name.starts_with("encoder.layer.0.attention.self.") && s.name.ends_with(".weight")
            })
            .collect();
        assert_eq!(qkv.len(), 3); // query, key, value
        for spec in qkv {
            assert_eq!(spec.shape, vec![384, 384]);
        }
    }

    #[test]
    fn expected_tensors_intermediate_shape_is_inter_x_hidden() {
        let specs = expected_tensors(&minilm_config());
        let inter = specs
            .iter()
            .find(|s| s.name == "encoder.layer.0.intermediate.dense.weight")
            .unwrap();
        // [intermediate_size, hidden_size] = [1536, 384] (HF convention).
        assert_eq!(inter.shape, vec![1536, 384]);
    }

    #[test]
    fn expected_tensors_output_dense_shape_contracts() {
        let specs = expected_tensors(&minilm_config());
        let out_dense = specs
            .iter()
            .find(|s| s.name == "encoder.layer.0.output.dense.weight")
            .unwrap();
        // [hidden_size, intermediate_size] (contraction direction).
        assert_eq!(out_dense.shape, vec![384, 1536]);
    }
}
