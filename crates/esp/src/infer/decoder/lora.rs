// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! PEFT LoRA adapter loading for the llama-family decoder.
//!
//! The implementation keeps base safetensors immutable, decodes a fresh
//! session-owned tensor set, and adds each selected adapter delta in session
//! order. It currently accepts the ordinary PEFT LoRA subset exercised by the
//! forcing fixture: bias-free A/B tensors on llama attention projections.

use std::collections::HashSet;

use burn::tensor::{Device, Tensor};
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::config::DecoderConfig;
use super::loader::load_decoder_tensors_from_bytes;
use super::model::{DecoderModel, LoadedDecoder};
use super::provider::DecoderProvider;
use super::tensors::extract;
use crate::infer::provider::InferError;
use crate::infer::session::{AdapterArtifact, AdapterLoader, BoundModelSession, ModelSession};

/// Runtime id recorded by the native NdArray forcing fixture.
pub const PEFT_LORA_NDARRAY_LOADER: &str = "burn-ndarray/peft-lora-v1";
const SUPPORTED_CAPABILITIES: &[&str] = &["peft-lora", "llama-attention-projections"];

#[derive(Debug, Deserialize)]
struct PeftLoraConfig {
    base_model_name_or_path: String,
    peft_type: String,
    peft_version: String,
    r: usize,
    lora_alpha: f32,
    #[serde(default)]
    target_modules: Vec<String>,
    #[serde(default)]
    bias: String,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_qalora: bool,
    #[serde(default)]
    rank_pattern: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    alpha_pattern: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    modules_to_save: Option<serde_json::Value>,
}

fn tensor<'a>(tensors: &'a SafeTensors<'_>, name: &str) -> Result<TensorView<'a>, InferError> {
    tensors
        .tensor(name)
        .map_err(|_| InferError::InvalidWeights(format!("missing adapter tensor: {name}")))
}

/// The name-only half of [`dimensions`]: is this a llama attention
/// projection the loader will accept?
///
/// Trainer settings are validated before any `config.json` is parsed, so the
/// name check has to stand without the shapes. The two are bound together by
/// `dimensions_and_names_agree` below; neither may grow an arm alone.
pub(crate) fn supported_target_module(module: &str) -> bool {
    matches!(module, "q_proj" | "k_proj" | "v_proj" | "o_proj")
}

pub(crate) fn dimensions(
    config: &DecoderConfig,
    module: &str,
) -> Result<(usize, usize), InferError> {
    let h = config.hidden_size;
    let kv = config.kv_heads() * config.head_dim();
    match module {
        "q_proj" | "o_proj" => Ok((h, h)),
        "k_proj" | "v_proj" => Ok((h, kv)),
        other => Err(InferError::InvalidConfig(format!(
            "PEFT LoRA target module {other:?} is not supported; expected llama q/k/v/o projections"
        ))),
    }
}

pub(crate) fn add_delta(base: Tensor<2>, a: Tensor<2>, b: Tensor<2>, scale: f32) -> Tensor<2> {
    // PEFT stores A [rank, in] and B [out, rank]. Burn linears store
    // [in, out], so the resident delta is A^T @ B^T.
    base + a.transpose().matmul(b.transpose()).mul_scalar(scale)
}

fn apply_peft_lora(
    decoder_config: &DecoderConfig,
    loaded: &mut LoadedDecoder,
    session: &ModelSession,
    selection_scale: f32,
    artifact: &AdapterArtifact<'_>,
    device: &Device,
) -> Result<(), InferError> {
    let config: PeftLoraConfig = serde_json::from_slice(artifact.config_bytes)
        .map_err(|error| InferError::InvalidConfig(format!("PEFT adapter config: {error}")))?;
    let manifest = artifact.manifest;

    for capability in &manifest.runtime_compat.minimum_capabilities {
        if !SUPPORTED_CAPABILITIES.contains(&capability.as_str()) {
            return Err(InferError::InvalidConfig(format!(
                "unsupported adapter capability {capability}"
            )));
        }
    }

    if manifest.adapter_format != "peft-lora" {
        return Err(InferError::InvalidConfig(format!(
            "unsupported adapter format {}",
            manifest.adapter_format
        )));
    }
    if !config.peft_type.eq_ignore_ascii_case("lora") {
        return Err(InferError::InvalidConfig(format!(
            "unsupported PEFT type {}",
            config.peft_type
        )));
    }
    let expected_format_version = format!("peft-{}", config.peft_version);
    if manifest.adapter_format_version != expected_format_version {
        return Err(InferError::InvalidConfig(format!(
            "PEFT version {} does not match manifest format version {}",
            config.peft_version, manifest.adapter_format_version
        )));
    }
    if config.base_model_name_or_path != session.model_id {
        return Err(InferError::InvalidConfig(format!(
            "PEFT base model {} does not match session model {}",
            config.base_model_name_or_path, session.model_id
        )));
    }
    if config.r != usize::from(manifest.rank)
        || config.lora_alpha.to_bits() != manifest.alpha.to_bits()
    {
        return Err(InferError::InvalidConfig(format!(
            "PEFT rank/alpha ({}/{}) does not match manifest ({}/{})",
            config.r, config.lora_alpha, manifest.rank, manifest.alpha
        )));
    }
    if config.r == 0 || !config.lora_alpha.is_finite() {
        return Err(InferError::InvalidConfig(
            "PEFT rank must be positive and alpha finite".into(),
        ));
    }
    if config.fan_in_fan_out
        || config.use_dora
        || config.use_rslora
        || config.use_qalora
        || config.bias != "none"
        || !config.rank_pattern.is_empty()
        || !config.alpha_pattern.is_empty()
        || config.modules_to_save.is_some()
    {
        return Err(InferError::InvalidConfig(
            "adapter requires an unsupported PEFT variant (bias, fan-in/out, DoRA, RSLoRA, QALoRA, per-module rank/alpha, or modules_to_save)".into(),
        ));
    }

    let config_targets: HashSet<&str> = config.target_modules.iter().map(String::as_str).collect();
    for module in &manifest.target_modules {
        dimensions(decoder_config, module)?;
        if !config_targets.contains(module.as_str()) {
            return Err(InferError::InvalidConfig(format!(
                "manifest target module {module} is absent from adapter_config.json"
            )));
        }
    }

    let tensors = SafeTensors::deserialize(artifact.weight_bytes)
        .map_err(|error| InferError::InvalidWeights(format!("adapter safetensors: {error}")))?;
    let expected_tensor_count =
        decoder_config.num_hidden_layers * manifest.target_modules.len() * 2;
    if tensors.len() != expected_tensor_count {
        return Err(InferError::InvalidWeights(format!(
            "adapter tensor count mismatch: expected {expected_tensor_count}, got {}",
            tensors.len()
        )));
    }

    let lora_scale = manifest.alpha / f32::from(manifest.rank) * selection_scale;
    for (layer_index, layer) in loaded.layers.iter_mut().enumerate() {
        for module in &manifest.target_modules {
            let (in_features, out_features) = dimensions(decoder_config, module)?;
            let prefix = format!("base_model.model.model.layers.{layer_index}.self_attn.{module}");
            let a = extract::<2>(
                &tensor(&tensors, &format!("{prefix}.lora_A.weight"))?,
                [config.r, in_features],
                device,
            )?;
            let b = extract::<2>(
                &tensor(&tensors, &format!("{prefix}.lora_B.weight"))?,
                [out_features, config.r],
                device,
            )?;
            match module.as_str() {
                "q_proj" => layer.q_w = add_delta(layer.q_w.clone(), a, b, lora_scale),
                "k_proj" => layer.k_w = add_delta(layer.k_w.clone(), a, b, lora_scale),
                "v_proj" => layer.v_w = add_delta(layer.v_w.clone(), a, b, lora_scale),
                "o_proj" => layer.o_w = add_delta(layer.o_w.clone(), a, b, lora_scale),
                _ => unreachable!("validated by dimensions"),
            }
        }
    }
    Ok(())
}

/// Session factory for the ESP Burn decoder plus ordinary PEFT LoRA tensors.
pub struct PeftLoraAdapterLoader<'a> {
    base_model_ref: eidetic::ManifestId,
    tokenizer_ref: eidetic::ManifestId,
    config_bytes: &'a [u8],
    tokenizer_bytes: &'a [u8],
    base_weights: &'a [u8],
    model_id: &'a str,
    loader: &'a str,
    device: Device,
}

impl<'a> PeftLoraAdapterLoader<'a> {
    /// Construct a loader over immutable base artifacts and a host-selected device.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_model_ref: eidetic::ManifestId,
        tokenizer_ref: eidetic::ManifestId,
        config_bytes: &'a [u8],
        tokenizer_bytes: &'a [u8],
        base_weights: &'a [u8],
        model_id: &'a str,
        loader: &'a str,
        device: &Device,
    ) -> Self {
        Self {
            base_model_ref,
            tokenizer_ref,
            config_bytes,
            tokenizer_bytes,
            base_weights,
            model_id,
            loader,
            device: device.clone(),
        }
    }
}

impl AdapterLoader for PeftLoraAdapterLoader<'_> {
    type Provider = DecoderProvider;

    fn load_session(
        &self,
        session: ModelSession,
        artifacts: &[AdapterArtifact<'_>],
    ) -> Result<BoundModelSession<Self::Provider>, InferError> {
        session.validate_adapters(artifacts)?;
        if session.base_model_ref != self.base_model_ref
            || session.tokenizer_ref != self.tokenizer_ref
            || session.model_id != self.model_id
            || session.loader != self.loader
        {
            return Err(InferError::InvalidConfig(
                "session base manifest, tokenizer, model id, or loader does not match PEFT loader base"
                    .into(),
            ));
        }
        let config = DecoderConfig::from_json_bytes(self.config_bytes)?;
        let mut loaded = load_decoder_tensors_from_bytes(&config, self.base_weights, &self.device)?;
        for ((selection, artifact), index) in
            session.adapters.iter().zip(artifacts.iter()).zip(0usize..)
        {
            apply_peft_lora(
                &config,
                &mut loaded,
                &session,
                selection.scale,
                artifact,
                &self.device,
            )
            .map_err(|error| match error {
                InferError::InvalidConfig(message) => {
                    InferError::InvalidConfig(format!("adapter {index}: {message}"))
                }
                InferError::InvalidWeights(message) => {
                    InferError::InvalidWeights(format!("adapter {index}: {message}"))
                }
                other => other,
            })?;
        }
        let tokenizer = Tokenizer::from_bytes(self.tokenizer_bytes)
            .map_err(|error| InferError::InvalidConfig(format!("tokenizer.json parse: {error}")))?;
        let model = DecoderModel::from_loaded(config, loaded, &self.device);
        let provider = DecoderProvider::from_parts(model, tokenizer, self.model_id, self.loader);
        BoundModelSession::bind(session, provider).map_err(InferError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::tests::det_loaded;
    use super::super::test_support::tiny_config;
    use super::*;
    use eidetic::{AdapterRuntimeCompat, Hash, ManifestId, ModelAdapterManifest};
    use safetensors::tensor::Dtype;

    fn adapter_bytes(config: &DecoderConfig, rank: usize) -> Vec<u8> {
        let mut buffers = Vec::new();
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("base_model.model.model.layers.{layer}.self_attn.q_proj");
            let a: Vec<f32> = (0..rank * config.hidden_size)
                .map(|index| (index + 1) as f32 * 0.001)
                .collect();
            let b: Vec<f32> = (0..config.hidden_size * rank)
                .map(|index| (index + 3) as f32 * -0.002)
                .collect();
            buffers.push((
                format!("{prefix}.lora_A.weight"),
                vec![rank, config.hidden_size],
                a.iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ));
            buffers.push((
                format!("{prefix}.lora_B.weight"),
                vec![config.hidden_size, rank],
                b.iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ));
        }
        let views: Vec<(&str, TensorView<'_>)> = buffers
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize(views, &None).unwrap()
    }

    #[test]
    fn dimensions_and_names_agree() {
        let config = tiny_config();
        for module in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "lm_head",
            "",
            "Q_PROJ",
        ] {
            assert_eq!(
                supported_target_module(module),
                dimensions(&config, module).is_ok(),
                "the name-only check and dimensions() disagree about {module:?}"
            );
        }
    }

    #[test]
    fn applies_real_low_rank_product_to_only_declared_projection() {
        let config = tiny_config();
        let device = Device::ndarray();
        let mut loaded = det_loaded(&config, &device);
        let before_q = loaded.layers[0].q_w.clone().into_data();
        let before_v = loaded.layers[0].v_w.clone().into_data();
        let base_ref = ManifestId::of_blob(b"base");
        let tokenizer_ref = ManifestId::of_blob(b"tokenizer");
        let template_hash = Hash::of(b"template");
        let adapter_ref = ManifestId::of_blob(b"adapter manifest");
        let session = ModelSession {
            base_model_ref: base_ref,
            model_id: "fixture/model".into(),
            tokenizer_ref,
            prompt_template_hash: template_hash,
            quantization: None,
            loader: PEFT_LORA_NDARRAY_LOADER.into(),
            adapters: vec![crate::infer::session::AdapterSelection {
                manifest_ref: adapter_ref,
                scale: 1.0,
            }],
        };
        let config_bytes = br#"{
            "base_model_name_or_path":"fixture/model",
            "peft_type":"LORA",
            "peft_version":"test",
            "r":2,
            "lora_alpha":4.0,
            "target_modules":["q_proj"],
            "bias":"none"
        }"#;
        let weights = adapter_bytes(&config, 2);
        let manifest = ModelAdapterManifest {
            name: "fixture".into(),
            base_model_ref: base_ref,
            adapter_blob: ManifestId::of_blob(&weights),
            adapter_config_blob: ManifestId::of_blob(config_bytes),
            adapter_format: "peft-lora".into(),
            adapter_format_version: "peft-test".into(),
            runtime_compat: AdapterRuntimeCompat {
                minimum_capabilities: vec!["peft-lora".into()],
                known_loaders: vec![PEFT_LORA_NDARRAY_LOADER.into()],
                converter_lineage: vec![],
            },
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["q_proj".into()],
            tokenizer_ref,
            prompt_template_hash: template_hash,
            quantization_assumption: None,
            training_corpus_root: None,
            training_method: serde_json::Value::Null,
            eval_results: None,
        };
        let artifact = AdapterArtifact {
            manifest_ref: adapter_ref,
            manifest: &manifest,
            config_bytes,
            weight_bytes: &weights,
        };
        apply_peft_lora(&config, &mut loaded, &session, 1.0, &artifact, &device).unwrap();
        assert_ne!(loaded.layers[0].q_w.clone().into_data(), before_q);
        assert_eq!(loaded.layers[0].v_w.clone().into_data(), before_v);
    }
}
