// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Local-only resolution of an inference row from the owner's model library.

use super::decoder::WeightFormat;
use super::decoder::streaming::LayerSource;
use super::decoder::{DecoderDevice, DecoderProvider};
use super::layer_library::{
    ExpertWeightIndex, FormattedLayerWeightIndex, LayerArtifact, LayerWeightIndex,
};
use super::prompt::{PromptTemplate, SessionProvider};
use super::{BoundModelSession, InferError, ModelSession};
use eidetic::typed::{TypedPayload, load_typed};
use eidetic::{ManifestId, ModelLibrary, NoFetcher, SchemaRef, Store};
use serde::{Deserialize, Serialize};

const INFERENCE_MODEL_SCHEMA_PAYLOAD: &[u8] = br#"{"format":"mere-native","schema_id":"esp.inference-model/v1","body":{"version":1,"description":"An exact model manifest paired with its prompt-template artifact.","required":["model","template"],"fields":{"model":{"type":"string"},"template":{"type":"string"}},"additional_properties":false}}"#;

/// One selectable row binds a model manifest to its declared prompt artifact.
/// Changing either reference changes the row and the resulting session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceModelRow {
    pub model: ManifestId,
    pub template: ManifestId,
}

impl TypedPayload for InferenceModelRow {
    fn schema_ref() -> SchemaRef {
        SchemaRef::from_id(ManifestId::of_blob(INFERENCE_MODEL_SCHEMA_PAYLOAD))
    }
}

fn failure(error: impl std::fmt::Display) -> InferError {
    InferError::InvalidConfig(format!("model library: {error}"))
}

/// Resolve without a network fetcher. Call on the inference actor's thread;
/// release the store lock after this returns, before any generation starts.
pub async fn load_library_session(
    store: &mut dyn Store,
    row: ManifestId,
    device: &DecoderDevice,
    loader: &str,
    max_tokens: usize,
) -> Result<SessionProvider<DecoderProvider>, InferError> {
    resolve_library_session(store, row)
        .await?
        .load(device, loader, max_tokens)
}

/// Byte buffers resolved while borrowing the library. Device construction
/// happens after this value leaves the store's lock.
pub struct ResolvedInferenceRow {
    model_ref: ManifestId,
    model: eidetic::models::ResolvedModel,
    template_bytes: Vec<u8>,
    layered: Option<LayerWeightIndex>,
    formatted: Option<FormattedLayerWeightIndex>,
    experts: Option<ExpertWeightIndex>,
}

/// Resolve and validate the row and template without constructing a device.
pub async fn resolve_library_session(
    store: &mut dyn Store,
    row: ManifestId,
) -> Result<ResolvedInferenceRow, InferError> {
    resolve_library_row(store, row, false).await
}

/// Diagnostic owner path for measuring an imported candidate. Resident
/// selection must use `resolve_library_session`, which requires qualification.
pub async fn resolve_library_candidate(
    store: &mut dyn Store,
    row: ManifestId,
) -> Result<ResolvedInferenceRow, InferError> {
    resolve_library_row(store, row, true).await
}

async fn resolve_library_row(
    store: &mut dyn Store,
    row: ManifestId,
    candidate: bool,
) -> Result<ResolvedInferenceRow, InferError> {
    let row = load_typed::<InferenceModelRow>(store, &mut NoFetcher, row)
        .await
        .map_err(failure)?
        .ok_or_else(|| failure(format!("inference row {row} is missing")))?;
    let manifest = eidetic::manifest::load_manifest(store, row.template)
        .await
        .map_err(failure)?
        .ok_or_else(|| failure(format!("template {} is missing", row.template)))?;
    if manifest.schema != PromptTemplate::schema_ref() {
        return Err(failure("template artifact has the wrong schema"));
    }
    let template_bytes = eidetic::manifest::resolve_blob(store, &mut NoFetcher, &manifest)
        .await
        .map_err(failure)?;
    PromptTemplate::from_bytes(&template_bytes)?;
    let model_manifest = ModelLibrary::load_model(store, row.model)
        .await
        .map_err(failure)?
        .ok_or_else(|| failure(format!("model {} is missing", row.model)))?;
    let weight_manifest = eidetic::manifest::load_manifest(store, model_manifest.weight_blob)
        .await
        .map_err(failure)?
        .ok_or_else(|| failure("weight manifest missing"))?;
    let formatted = if weight_manifest.schema == FormattedLayerWeightIndex::schema_ref() {
        if weight_manifest.id != model_manifest.weight_blob
            || weight_manifest.content_hash != model_manifest.weight_blob.0
        {
            return Err(failure("formatted index identity mismatch"));
        }
        let index = load_typed::<FormattedLayerWeightIndex>(
            store,
            &mut NoFetcher,
            model_manifest.weight_blob,
        )
        .await
        .map_err(failure)?
        .ok_or_else(|| failure("formatted index missing"))?;
        match &index.qualification {
            Some(qualification) => qualification.validate()?,
            None if !candidate => {
                return Err(failure("weight format candidate has not been measured"));
            },
            None => (),
        }
        Some(index)
    } else {
        None
    };
    let experts = if weight_manifest.schema == ExpertWeightIndex::schema_ref() {
        if weight_manifest.id != model_manifest.weight_blob
            || weight_manifest.content_hash != model_manifest.weight_blob.0
        {
            return Err(failure("expert index identity mismatch"));
        }
        Some(
            load_typed::<ExpertWeightIndex>(store, &mut NoFetcher, model_manifest.weight_blob)
                .await
                .map_err(failure)?
                .ok_or_else(|| failure("expert index missing"))?,
        )
    } else {
        None
    };
    let layered = if let Some(index) = &experts {
        Some(index.weights.clone())
    } else if let Some(index) = &formatted {
        Some(index.weights.clone())
    } else if weight_manifest.schema == LayerWeightIndex::schema_ref() {
        if weight_manifest.id != model_manifest.weight_blob
            || weight_manifest.content_hash != model_manifest.weight_blob.0
        {
            return Err(failure(
                "layer index manifest does not match its content reference",
            ));
        }
        let index =
            load_typed::<LayerWeightIndex>(store, &mut NoFetcher, model_manifest.weight_blob)
                .await
                .map_err(failure)?
                .ok_or_else(|| failure("layer index missing"))?;
        Some(index)
    } else {
        None
    };
    if let Some(index) = &layered {
        index.options.validate()?;
        let config = super::decoder::DecoderConfig::from_json_bytes(
            &serde_json::to_vec(&model_manifest.config).map_err(failure)?,
        )?;
        if index.layers.len() != config.num_hidden_layers {
            return Err(failure("layer count disagrees with config"));
        }
        match (&config.moe, &experts) {
            (Some(moe), Some(experts))
                if experts.experts.len() == config.num_hidden_layers
                    && experts.experts.iter().enumerate().all(|(i, bank)| {
                        bank.len()
                            == if config.layer_is_moe(i) {
                                moe.num_experts
                            } else {
                                0
                            }
                    }) =>
            {
                ()
            },
            (None, None) => (),
            _ => {
                return Err(failure(
                    "expert index dimensions disagree with decoder config",
                ));
            },
        }
    }
    let resolved = if let Some(index) = &layered {
        let tokenizer_manifest =
            eidetic::manifest::load_manifest(store, model_manifest.tokenizer_blob)
                .await
                .map_err(failure)?
                .ok_or_else(|| failure("tokenizer missing"))?;
        eidetic::models::ResolvedModel {
            components: eidetic::ModelComponents {
                config_bytes: serde_json::to_vec(&model_manifest.config).map_err(failure)?,
                tokenizer_bytes: eidetic::manifest::resolve_blob(
                    store,
                    &mut NoFetcher,
                    &tokenizer_manifest,
                )
                .await
                .map_err(failure)?,
                weight_bytes: index
                    .shared
                    .read(store, index.options.max_shared_bytes)
                    .await?,
            },
            manifest: model_manifest,
        }
    } else {
        ModelLibrary::resolve_components(store, &mut NoFetcher, row.model)
            .await
            .map_err(failure)?
            .ok_or_else(|| failure("model missing"))?
    };
    Ok(ResolvedInferenceRow {
        model_ref: row.model,
        model: resolved,
        template_bytes,
        layered,
        formatted,
        experts,
    })
}

impl ResolvedInferenceRow {
    /// Validated geometry for host placement and transport bounds.
    pub fn decoder_config(&self) -> Result<super::decoder::DecoderConfig, InferError> {
        super::decoder::DecoderConfig::from_json_bytes(&self.model.components.config_bytes)
    }

    /// Full runtime weights and declared context KV. Hosts separately reserve
    /// backend, activations, RoPE, allocator and artifact transfer memory.
    pub fn resident_payload_bytes(&self) -> Result<u64, InferError> {
        let c = self.decoder_config()?;
        let format = self.weight_format().unwrap_or_default();
        let h = c.hidden_size as u128;
        let matrix = h.saturating_mul(c.vocab_size as u128);
        let scalar = if format == WeightFormat::F32 { 4 } else { 2 };
        let head = match format {
            WeightFormat::Q8Block32 => matrix.saturating_add(matrix / 32 * 4),
            WeightFormat::Q4Block32 => (matrix / 2).saturating_add(matrix / 32 * 4),
            _ => matrix.saturating_mul(scalar),
        };
        let shared = matrix
            .saturating_mul(scalar)
            .saturating_add(h.saturating_mul(scalar))
            .saturating_add(if c.tie_word_embeddings { 0 } else { head });
        let layers = self
            .stage_layer_costs()?
            .iter()
            .map(|b| *b as u128)
            .sum::<u128>();
        u64::try_from(layers.saturating_add(shared))
            .map_err(|_| failure("resident payload size overflow"))
    }
    /// Resident layer payload including KV at the declared context limit.
    /// Hosts reserve remaining shared/backend memory before assigning ranges.
    pub fn stage_layer_costs(&self) -> Result<Vec<u64>, InferError> {
        let config =
            super::decoder::DecoderConfig::from_json_bytes(&self.model.components.config_bytes)?;
        let bytes = super::decoder::stage::layer_resident_bytes(
            &config,
            self.weight_format().unwrap_or_default(),
        )?;
        Ok(vec![bytes; config.num_hidden_layers])
    }
    pub fn weight_format(&self) -> Option<WeightFormat> {
        self.formatted.as_ref().map(|index| index.format)
    }

    /// A host retains the model-library owner while this source is in use.
    pub fn layer_artifacts(&self) -> Option<Vec<LayerArtifact>> {
        self.layered.as_ref().map(|index| index.layers.clone())
    }

    /// The owner supplies the store borrow, while this row supplies exact
    /// artifact identities and bounds. No store lock spans tensor execution.
    pub fn artifact_source<R>(&self, reader: R) -> Option<std::sync::Arc<dyn LayerSource>>
    where
        R: Fn(&LayerArtifact, usize) -> Result<Vec<u8>, InferError> + Send + Sync + 'static,
    {
        use std::sync::Arc;
        let layers = self.layer_artifacts()?;
        let experts = self.experts.as_ref().map(|index| index.experts.clone());
        let reader = Arc::new(reader);
        let layer_reader = reader.clone();
        Some(Arc::new(super::decoder::streaming::with_expert_reader(
            move |index: usize, limit: usize| {
                layer_reader(
                    layers
                        .get(index)
                        .ok_or_else(|| failure("layer outside index"))?,
                    limit,
                )
            },
            move |layer: usize, expert: usize, limit: usize| {
                reader(
                    experts
                        .as_ref()
                        .and_then(|bank| bank.get(layer))
                        .and_then(|bank| bank.get(expert))
                        .ok_or_else(|| failure("expert outside index"))?,
                    limit,
                )
            },
        )))
    }

    fn execution_loader(&self, loader: &str) -> String {
        if self.experts.is_some() {
            format!("{loader}/expert-stream-v1/f32")
        } else if let Some(index) = &self.formatted {
            format!("{loader}/layer-stream-v2/{}", index.format.id())
        } else if self.layered.is_some() {
            format!("{loader}/layer-stream-v1")
        } else {
            loader.into()
        }
    }
    /// Exact identity before a deferred device load, including execution format.
    pub fn session(&self, loader: &str) -> ModelSession {
        ModelSession {
            base_model_ref: self.model_ref,
            model_id: self.model.manifest.model_id.clone(),
            tokenizer_ref: self.model.manifest.tokenizer_blob,
            prompt_template_hash: eidetic::Hash::of(&self.template_bytes),
            quantization: self
                .formatted
                .as_ref()
                .filter(|i| i.format != WeightFormat::F32)
                .map(|i| i.format.id().into()),
            loader: self.execution_loader(loader),
            adapters: Vec::new(),
        }
    }

    fn validate_qualification(&self, loader: &str) -> Result<(), InferError> {
        if let Some(qualification) = self
            .formatted
            .as_ref()
            .and_then(|i| i.qualification.as_ref())
        {
            if qualification.measured_loader != self.execution_loader(loader) {
                return Err(failure(
                    "weight format has not been qualified on this loader",
                ));
            }
        }
        Ok(())
    }

    /// Declared capability for a deferred loader. Tensor shape validation still
    /// occurs at device load; the host must surface that call's failures.
    pub fn capability(&self, loader: &str) -> Result<super::ModelCapability, InferError> {
        self.validate_qualification(loader)?;
        let config =
            super::decoder::DecoderConfig::from_json_bytes(&self.model.components.config_bytes)?;
        Ok(super::ModelCapability {
            model_id: self.model.manifest.model_id.clone(),
            context_window: config.max_position_embeddings,
            quantization: self
                .formatted
                .as_ref()
                .filter(|i| i.format != WeightFormat::F32)
                .map(|i| i.format.id().into()),
            loader: self.execution_loader(loader),
            streaming: true,
        })
    }

    /// Load on the caller-selected device, with the library borrow released.
    pub fn load(
        &self,
        device: &DecoderDevice,
        loader: &str,
        max_tokens: usize,
    ) -> Result<SessionProvider<DecoderProvider>, InferError> {
        self.load_with_layer_source(device, loader, max_tokens, None)
    }

    pub fn load_with_layer_source(
        &self,
        device: &DecoderDevice,
        loader: &str,
        max_tokens: usize,
        source: Option<Box<dyn LayerSource>>,
    ) -> Result<SessionProvider<DecoderProvider>, InferError> {
        self.load_execution(device, loader, max_tokens, source, None, false)
    }

    /// First rung of the same indexed session: all layers stay on this device.
    pub fn load_resident_with_layer_source(
        &self,
        device: &DecoderDevice,
        loader: &str,
        max_tokens: usize,
        source: Box<dyn LayerSource>,
    ) -> Result<SessionProvider<DecoderProvider>, InferError> {
        self.load_execution(device, loader, max_tokens, Some(source), None, true)
    }

    /// The stage chain is a placement of this exact layer-indexed session.
    /// It does not change the format, backend class or checkpoint identity.
    pub fn load_with_stages(
        &self,
        device: &DecoderDevice,
        loader: &str,
        max_tokens: usize,
        stages: Box<dyn super::decoder::stage::StageExecutor>,
    ) -> Result<SessionProvider<DecoderProvider>, InferError> {
        if stages.session_id() != Some(self.session(loader).id()?.to_string().as_str()) {
            return Err(failure("stage chain belongs to another exact session"));
        }
        self.load_execution(
            device,
            loader,
            max_tokens,
            Some(Box::new(|_, _| {
                Err(InferError::Backend(
                    "staged provider must not stream local layers".into(),
                ))
            })),
            Some(stages),
            false,
        )
    }

    /// Load only this range from the library owner's verified source. The
    /// caller keeps its mesh lease and hardware guard through drain and drop.
    pub fn load_stage(
        &self,
        device: &DecoderDevice,
        loader: &str,
        range: super::decoder::stage::LayerRange,
        source: &dyn LayerSource,
    ) -> Result<super::decoder::stage::DecoderStage, InferError> {
        self.validate_qualification(loader)?;
        let index = self
            .layered
            .as_ref()
            .ok_or_else(|| failure("stages require a layer index"))?;
        let stage = super::decoder::stage::DecoderStage::from_source(
            super::decoder::DecoderConfig::from_json_bytes(&self.model.components.config_bytes)?,
            range,
            source,
            index.options.clone(),
            self.weight_format().unwrap_or_default(),
            device,
        )?;
        if range.end == index.layers.len() {
            let tensors =
                safetensors::SafeTensors::deserialize(&self.model.components.weight_bytes)
                    .map_err(failure)?;
            if self.experts.is_some() {
                let config = self.decoder_config()?;
                let shared = super::decoder::loader::load_shared_shards(
                    &config,
                    &tensors,
                    device,
                    index.options.max_layer_bytes,
                )?;
                return stage.with_output_shards(shared);
            }
            let shared = super::decoder::loader::load_decoder_shared_tensors_with_format(
                &super::decoder::DecoderConfig::from_json_bytes(
                    &self.model.components.config_bytes,
                )?,
                &tensors,
                device,
                self.weight_format().unwrap_or_default(),
            )?;
            stage.with_output(shared)
        } else {
            Ok(stage)
        }
    }

    fn load_execution(
        &self,
        device: &DecoderDevice,
        loader: &str,
        max_tokens: usize,
        source: Option<Box<dyn LayerSource>>,
        stages: Option<Box<dyn super::decoder::stage::StageExecutor>>,
        resident: bool,
    ) -> Result<SessionProvider<DecoderProvider>, InferError> {
        self.validate_qualification(loader)?;
        let provider = if let Some(index) = &self.layered {
            DecoderProvider::from_layer_source_with_format(
                &self.model.components.config_bytes,
                &self.model.components.tokenizer_bytes,
                &self.model.components.weight_bytes,
                source
                    .ok_or_else(|| failure("streamed model requires its library layer source"))?,
                index.options.clone(),
                &self.model.manifest.model_id,
                self.execution_loader(loader),
                device,
                self.formatted
                    .as_ref()
                    .map(|i| i.format)
                    .unwrap_or_default(),
            )?
        } else {
            DecoderProvider::from_bytes(
                &self.model.components.config_bytes,
                &self.model.components.tokenizer_bytes,
                &self.model.components.weight_bytes,
                &self.model.manifest.model_id,
                loader,
                device,
            )?
        }
        .with_literal_prompt()
        .with_template_eos(&PromptTemplate::from_bytes(&self.template_bytes)?.eos);
        let provider = if resident {
            provider.with_resident_layers()?
        } else {
            provider
        };
        let provider = if let Some(stages) = stages {
            provider.with_stages(stages)?
        } else {
            provider
        };
        let session = self.session(loader);
        SessionProvider::new(
            BoundModelSession::bind(session, provider)?,
            self.template_bytes.clone(),
            max_tokens,
        )
    }
}
