// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Local-only resolution of an inference row from the owner's model library.

use super::decoder::{DecoderDevice, DecoderProvider};
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
}

/// Resolve and validate the row and template without constructing a device.
pub async fn resolve_library_session(
    store: &mut dyn Store,
    row: ManifestId,
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
    let resolved = ModelLibrary::resolve_components(store, &mut NoFetcher, row.model)
        .await
        .map_err(failure)?
        .ok_or_else(|| failure(format!("model {} is missing", row.model)))?;
    Ok(ResolvedInferenceRow {
        model_ref: row.model,
        model: resolved,
        template_bytes,
    })
}

impl ResolvedInferenceRow {
    /// Exact identity before a deferred device load. The decoder currently
    /// accepts only unquantized model weights.
    pub fn session(&self, loader: &str) -> ModelSession {
        ModelSession {
            base_model_ref: self.model_ref,
            model_id: self.model.manifest.model_id.clone(),
            tokenizer_ref: self.model.manifest.tokenizer_blob,
            prompt_template_hash: eidetic::Hash::of(&self.template_bytes),
            quantization: None,
            loader: loader.into(),
            adapters: Vec::new(),
        }
    }

    /// Declared capability for a deferred loader. Tensor shape validation still
    /// occurs at device load; the host must surface that call's failures.
    pub fn capability(&self, loader: &str) -> Result<super::ModelCapability, InferError> {
        let config =
            super::decoder::DecoderConfig::from_json_bytes(&self.model.components.config_bytes)?;
        Ok(super::ModelCapability {
            model_id: self.model.manifest.model_id.clone(),
            context_window: config.max_position_embeddings,
            quantization: None,
            loader: loader.into(),
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
        let provider = DecoderProvider::from_bytes(
            &self.model.components.config_bytes,
            &self.model.components.tokenizer_bytes,
            &self.model.components.weight_bytes,
            &self.model.manifest.model_id,
            loader,
            device,
        )?
        .with_literal_prompt();
        let session = self.session(loader);
        SessionProvider::new(
            BoundModelSession::bind(session, provider)?,
            self.template_bytes.clone(),
            max_tokens,
        )
    }
}
