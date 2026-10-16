// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Content-addressed, per-layer model-library storage. Old loaders cannot
//! mistake this index for a safetensors checkpoint and silently omit layers.

use super::decoder::WeightFormat;
use super::{InferError, decoder::streaming::StreamingOptions};
use eidetic::typed::{TypedPayload, save_typed};
use eidetic::{ManifestId, NoFetcher, SchemaRef, Store};
use serde::{Deserialize, Serialize};

const SCHEMA: &[u8] = br#"{"format":"mere-native","schema_id":"esp.layer-weights/v1","body":{"version":1,"description":"Exact shared and ordered per-layer safetensors artifacts with execution bounds.","required":["shared","layers","options"],"additional_properties":false}}"#;

const FORMAT_SCHEMA: &[u8] = br#"{"format":"mere-native","schema_id":"esp.layer-weights/v2","body":{"version":2,"description":"Explicit execution format and measured qualification over exact layer artifacts.","additional_properties":false}}"#;

/// The bf16 comparison uses the exact rendered prompts recorded here. Logit
/// metrics describe the next token; generated ids expose later divergence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptDeviation {
    pub rendered_prompt: String,
    pub baseline_tokens: Vec<u32>,
    pub format_tokens: Vec<u32>,
    pub logit_rmse: f64,
    pub logit_max_abs: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormatQualification {
    pub baseline_model: ManifestId,
    pub baseline_loader: String,
    pub measured_loader: String,
    pub prompts: Vec<PromptDeviation>,
    pub encoded_weight_bytes: u64,
    pub weight_payload_bytes: u64,
    pub streaming_resident_payload_bytes: u64,
    pub largest_layer_bytes: u64,
    pub logical_stream_bytes: u64,
    pub elapsed_seconds: f64,
}

impl FormatQualification {
    pub fn validate(&self) -> Result<(), InferError> {
        if self.baseline_loader.is_empty()
            || self.measured_loader.is_empty()
            || self.prompts.is_empty()
            || self.encoded_weight_bytes == 0
            || self.weight_payload_bytes == 0
            || self.streaming_resident_payload_bytes == 0
            || self.largest_layer_bytes == 0
            || self.logical_stream_bytes == 0
            || !self.elapsed_seconds.is_finite()
            || self.elapsed_seconds <= 0.0
            || self.prompts.iter().any(|p| {
                p.rendered_prompt.is_empty()
                    || p.baseline_tokens.is_empty()
                    || p.format_tokens.is_empty()
                    || !p.logit_rmse.is_finite()
                    || p.logit_rmse < 0.0
                    || !p.logit_max_abs.is_finite()
                    || p.logit_max_abs < p.logit_rmse
            })
        {
            return Err(failure("incomplete or non-finite format qualification"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormattedLayerWeightIndex {
    pub weights: LayerWeightIndex,
    pub format: WeightFormat,
    pub source_artifacts: Vec<ManifestId>,
    /// A candidate cannot be selected by the resident until measured.
    pub qualification: Option<FormatQualification>,
}

impl TypedPayload for FormattedLayerWeightIndex {
    fn schema_ref() -> SchemaRef {
        SchemaRef::from_id(ManifestId::of_blob(FORMAT_SCHEMA))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerArtifact {
    pub blob: ManifestId,
    pub byte_size: u64,
}

impl LayerArtifact {
    /// Check the size before asking the store for a payload, then check its
    /// exact content reference. No fetcher can acquire missing model bytes.
    pub async fn read(&self, store: &mut dyn Store, limit: usize) -> Result<Vec<u8>, InferError> {
        let manifest = eidetic::manifest::load_manifest(store, self.blob)
            .await
            .map_err(failure)?
            .ok_or_else(|| failure(format!("missing layer artifact {}", self.blob)))?;
        if self.byte_size > limit as u64
            || manifest.byte_size != self.byte_size
            || manifest.id != self.blob
            || manifest.content_hash != self.blob.0
            || manifest.schema != eidetic::models::OpaqueBlob::schema_ref()
        {
            return Err(failure(
                "layer artifact identity, schema or byte limit mismatch",
            ));
        }
        let bytes = eidetic::manifest::resolve_blob(store, &mut NoFetcher, &manifest)
            .await
            .map_err(failure)?;
        if bytes.len() as u64 != self.byte_size || ManifestId::of_blob(&bytes) != self.blob {
            return Err(failure("layer artifact bytes do not match their reference"));
        }
        Ok(bytes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerWeightIndex {
    pub shared: LayerArtifact,
    pub layers: Vec<LayerArtifact>,
    pub options: StreamingOptions,
}

impl TypedPayload for LayerWeightIndex {
    fn schema_ref() -> SchemaRef {
        SchemaRef::from_id(ManifestId::of_blob(SCHEMA))
    }
}

fn failure(error: impl std::fmt::Display) -> InferError {
    InferError::InvalidWeights(format!("layer library: {error}"))
}

async fn save<T: TypedPayload>(
    store: &mut dyn Store,
    payload: &T,
) -> Result<ManifestId, InferError> {
    save_typed(
        store,
        payload,
        vec![],
        eidetic::PrivacyClass::LocalOnly,
        eidetic::ProvenanceRecord::self_imported("local"),
        eidetic::TrustEnvelope::self_asserted(),
        eidetic::Timestamp::ZERO,
    )
    .await
    .map_err(failure)
}

async fn save_artifact(store: &mut dyn Store, bytes: Vec<u8>) -> Result<LayerArtifact, InferError> {
    let byte_size = bytes.len() as u64;
    let blob = save(store, &eidetic::models::OpaqueBlob(bytes)).await?;
    Ok(LayerArtifact { blob, byte_size })
}

/// Explicit host import from a local HF checkpoint (single or sharded).
/// Only one encoded layer is borrowed at a time. Partial imports leave
/// content-addressed blobs, never a selectable incomplete model manifest.
#[cfg(not(target_family = "wasm"))]
pub async fn import_layered_model(
    store: &mut dyn Store,
    directory: &std::path::Path,
    model_id: &str,
    license: &str,
    options: StreamingOptions,
) -> Result<ManifestId, InferError> {
    import_layers(store, directory, model_id, license, options, None).await
}

/// Import a candidate. Quantization occurs once here; the resident never
/// quantizes while reading layers. Publish its measurement with
/// `qualify_formatted_model` before selecting it through an inference row.
#[cfg(not(target_family = "wasm"))]
pub async fn import_formatted_model(
    store: &mut dyn Store,
    directory: &std::path::Path,
    model_id: &str,
    license: &str,
    options: StreamingOptions,
    format: WeightFormat,
) -> Result<ManifestId, InferError> {
    import_layers(store, directory, model_id, license, options, Some(format)).await
}

#[cfg(not(target_family = "wasm"))]
async fn import_layers(
    store: &mut dyn Store,
    directory: &std::path::Path,
    model_id: &str,
    license: &str,
    options: StreamingOptions,
    format: Option<WeightFormat>,
) -> Result<ManifestId, InferError> {
    options.validate()?;
    let config_bytes = std::fs::read(directory.join("config.json")).map_err(failure)?;
    let config = super::decoder::DecoderConfig::from_json_bytes(&config_bytes)?;
    let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(failure)?;
    let mut reader = super::decoder::checkpoint::CheckpointReader::open(directory, &config)?;
    let mut source_artifacts = Vec::new();
    let mut convert = |bytes: Vec<u8>, limit: usize| -> Result<Vec<u8>, InferError> {
        source_artifacts.push(ManifestId::of_blob(&bytes));
        let bytes = match format {
            Some(format) => super::decoder::weight_format::convert_artifact(&bytes, format)?,
            None => bytes,
        };
        if bytes.len() > limit {
            return Err(failure("converted artifact exceeds its byte limit"));
        }
        Ok(bytes)
    };
    let shared = save_artifact(
        store,
        convert(
            reader.shared_bytes(options.max_shared_bytes)?,
            options.max_shared_bytes,
        )?,
    )
    .await?;
    let mut layers = Vec::new();
    for index in 0..config.num_hidden_layers {
        layers.push(
            save_artifact(
                store,
                convert(
                    reader.layer_bytes(index, options.max_layer_bytes)?,
                    options.max_layer_bytes,
                )?,
            )
            .await?,
        );
    }
    let weights = LayerWeightIndex {
        shared,
        layers,
        options,
    };
    let weight_blob = match format {
        Some(format) => {
            save(
                store,
                &FormattedLayerWeightIndex {
                    weights,
                    format,
                    source_artifacts,
                    qualification: None,
                },
            )
            .await?
        },
        None => save(store, &weights).await?,
    };
    let tokenizer_blob = save(
        store,
        &eidetic::models::OpaqueBlob(
            std::fs::read(directory.join("tokenizer.json")).map_err(failure)?,
        ),
    )
    .await?;
    save(
        store,
        &eidetic::ModelManifest {
            model_id: model_id.into(),
            architecture: "llama".into(),
            license: Some(license.into()),
            config: config_json,
            weight_blob,
            tokenizer_blob,
        },
    )
    .await
}

/// Attach an immutable, backend-specific measurement to an imported
/// candidate. The baseline must be a bf16 row of the same source model.
/// The original candidate and its artifacts remain content addressed.
pub async fn qualify_formatted_model(
    store: &mut dyn Store,
    candidate: ManifestId,
    qualification: FormatQualification,
) -> Result<ManifestId, InferError> {
    use eidetic::{ModelLibrary, typed::load_typed};
    qualification.validate()?;
    let mut model = ModelLibrary::load_model(store, candidate)
        .await
        .map_err(failure)?
        .ok_or_else(|| failure("candidate missing"))?;
    let baseline = ModelLibrary::load_model(store, qualification.baseline_model)
        .await
        .map_err(failure)?
        .ok_or_else(|| failure("bf16 baseline missing"))?;
    let mut index =
        load_typed::<FormattedLayerWeightIndex>(store, &mut NoFetcher, model.weight_blob)
            .await
            .map_err(failure)?
            .ok_or_else(|| failure("candidate format index missing"))?;
    let baseline_index =
        load_typed::<FormattedLayerWeightIndex>(store, &mut NoFetcher, baseline.weight_blob)
            .await
            .map_err(failure)?
            .ok_or_else(|| failure("baseline format index missing"))?;
    if baseline_index.format != WeightFormat::Bf16
        || index.source_artifacts != baseline_index.source_artifacts
        || model.config != baseline.config
        || model.tokenizer_blob != baseline.tokenizer_blob
        || model.model_id != baseline.model_id
    {
        return Err(failure(
            "qualification baseline is not this model's bf16 row",
        ));
    }
    let encoded = index.weights.shared.byte_size
        + index
            .weights
            .layers
            .iter()
            .map(|a| a.byte_size)
            .sum::<u64>();
    let largest = index
        .weights
        .layers
        .iter()
        .map(|a| a.byte_size)
        .max()
        .unwrap_or(0);
    if qualification.encoded_weight_bytes != encoded || qualification.largest_layer_bytes != largest
    {
        return Err(failure("qualification byte costs disagree with artifacts"));
    }
    index.qualification = Some(qualification);
    model.weight_blob = save(store, &index).await?;
    save(store, &model).await
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::super::decoder::loader::tests::{serialize_f32, tiny_tensor_table};
    use super::super::decoder::test_support::tiny_config;
    use super::super::decoder::{DecoderDevice, DecoderProvider};
    use super::super::library::{InferenceModelRow, resolve_library_session};
    use super::super::{GenerationRequest, InferenceProvider};
    use super::*;
    use std::sync::{Arc, Mutex};

    fn config_json(window: usize) -> serde_json::Value {
        serde_json::json!({
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "vocab_size":32,"hidden_size":8,"intermediate_size":16,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
            "max_position_embeddings":window,"rms_norm_eps":1e-5,"rope_theta":10000.0,
            "tie_word_embeddings":false
        })
    }

    fn fixture(window: usize) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config_json(window)).unwrap(),
        )
        .unwrap();
        let vocab: serde_json::Map<_, _> = (0..32)
            .map(|i| (format!("t{i}"), serde_json::json!(i)))
            .collect();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            serde_json::to_vec(&serde_json::json!({
                "version":"1.0", "pre_tokenizer":{"type":"Whitespace"},
                "model":{"type":"WordLevel","vocab":vocab,"unk_token":"t0"}
            }))
            .unwrap(),
        )
        .unwrap();
        let table = tiny_tensor_table(&tiny_config(), true);
        let mut weight_map = serde_json::Map::new();
        for shard in 0..2 {
            let name = format!("model-0000{shard}-of-00002.safetensors");
            let entries: Vec<_> = table
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == shard)
                .map(|(_, entry)| {
                    weight_map.insert(entry.0.clone(), serde_json::json!(name));
                    entry.clone()
                })
                .collect();
            std::fs::write(directory.path().join(name), serialize_f32(&entries)).unwrap();
        }
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":weight_map})).unwrap(),
        )
        .unwrap();
        directory
    }

    fn imported(
        directory: &std::path::Path,
    ) -> (
        Arc<Mutex<eidetic::MemoryBackend>>,
        super::super::library::ResolvedInferenceRow,
    ) {
        let mut store = eidetic::MemoryBackend::default();
        let model = pollster::block_on(import_layered_model(
            &mut store,
            directory,
            "fixture/layered",
            "MPL-2.0",
            StreamingOptions {
                query_chunk: 3,
                ..Default::default()
            },
        ))
        .unwrap();
        let template = super::super::prompt::PromptTemplate::from_bytes(include_bytes!(
            "../../templates/smollm2-instruct-v1.json"
        ))
        .unwrap();
        let template = pollster::block_on(save(&mut store, &template)).unwrap();
        let row =
            pollster::block_on(save(&mut store, &InferenceModelRow { model, template })).unwrap();
        let resolved = pollster::block_on(resolve_library_session(&mut store, row)).unwrap();
        (Arc::new(Mutex::new(store)), resolved)
    }

    #[test]
    fn sharded_library_matches_resident_and_resumes_at_every_boundary() {
        let directory = fixture(256);
        let (store, resolved) = imported(directory.path());
        let artifacts = resolved.layer_artifacts().unwrap();
        let source = move |index: usize, limit: usize| {
            pollster::block_on(artifacts[index].read(&mut *store.lock().unwrap(), limit))
        };
        let streamed = resolved
            .load_with_layer_source(
                &DecoderDevice::ndarray(),
                "burn-ndarray",
                8,
                Some(Box::new(source)),
            )
            .unwrap();
        let resident = DecoderProvider::from_bytes(
            &serde_json::to_vec(&config_json(256)).unwrap(),
            &std::fs::read(directory.path().join("tokenizer.json")).unwrap(),
            &serialize_f32(&tiny_tensor_table(&tiny_config(), true)),
            "fixture/layered",
            "burn-ndarray",
            &DecoderDevice::ndarray(),
        )
        .unwrap()
        .with_literal_prompt();
        assert!(streamed.capability().loader.ends_with("/layer-stream-v1"));
        assert!(
            resolved
                .load(&DecoderDevice::ndarray(), "burn-ndarray", 8)
                .is_err()
        );
        for temperature in [0.0, 0.8] {
            let request = GenerationRequest {
                prompt: "t3 t7 t11 t5 t9".into(),
                max_tokens: 8,
                temperature,
                seed: Some(42),
                top_p: Some(0.95),
                ..Default::default()
            };
            let expected = resident
                .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
                .unwrap();
            let provider = streamed.bound().provider();
            let full = provider
                .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
                .unwrap();
            assert_eq!(full.token_ids, expected.token_ids);
            assert_eq!(full.text, expected.text);
            for boundary in 0..8 {
                let seen = std::cell::Cell::new(0usize);
                let paused = provider
                    .generate_checkpointed(
                        &request,
                        None,
                        &mut |cp| {
                            seen.set(cp.token_ids.len());
                            Ok(())
                        },
                        &mut || seen.get() >= boundary,
                    )
                    .unwrap();
                assert_eq!(paused.token_ids.len(), boundary);
                let resumed = provider
                    .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
                    .unwrap();
                assert_eq!(
                    resumed.token_ids, full.token_ids,
                    "temperature {temperature}, pause {boundary}"
                );
                assert_eq!(resumed.text, full.text);
            }
        }
        assert!(
            streamed
                .bound()
                .provider()
                .streaming_stats()
                .unwrap()
                .layer_reads
                > 0
        );
    }

    #[test]
    fn direct_provider_refuses_shared_artifacts_over_budget() {
        let directory = fixture(256);
        let mut reader = super::super::decoder::checkpoint::CheckpointReader::open(
            directory.path(),
            &tiny_config(),
        )
        .unwrap();
        let shared = reader.shared_bytes(1024 * 1024).unwrap();
        let result = DecoderProvider::from_layer_source(
            &serde_json::to_vec(&config_json(256)).unwrap(),
            &std::fs::read(directory.path().join("tokenizer.json")).unwrap(),
            &shared,
            Box::new(|_, _| Err(InferError::Backend("unused source".into()))),
            StreamingOptions {
                max_shared_bytes: 1,
                ..Default::default()
            },
            "fixture/layered",
            "burn-ndarray/layer-stream-v1",
            &DecoderDevice::ndarray(),
        );
        assert!(
            matches!(result, Err(InferError::InvalidWeights(message)) if message.contains("shared artifact"))
        );
    }

    #[test]
    fn corrupted_or_oversized_layer_is_refused_before_generation() {
        let directory = fixture(256);
        let (store, resolved) = imported(directory.path());
        let artifact = resolved.layer_artifacts().unwrap().remove(0);
        let mut store = store.lock().unwrap();
        assert!(pollster::block_on(artifact.read(&mut *store, 1)).is_err());
        let manifest =
            pollster::block_on(eidetic::manifest::load_manifest(&mut *store, artifact.blob))
                .unwrap()
                .unwrap();
        let key = manifest
            .sources
            .iter()
            .find_map(|source| match source {
                eidetic::BlobSource::Local { key } => Some(key),
                _ => None,
            })
            .unwrap();
        pollster::block_on(store.put(key, b"corrupted")).unwrap();
        assert!(pollster::block_on(artifact.read(&mut *store, usize::MAX)).is_err());
    }

    #[test]
    fn shard_directory_and_missing_entries_are_refused() {
        let directory = fixture(256);
        let path = directory.path().join("model.safetensors.index.json");
        let mut index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        index["weight_map"]["model.norm.weight"] = serde_json::json!("../outside.safetensors");
        std::fs::write(&path, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(
            super::super::decoder::checkpoint::CheckpointReader::open(
                directory.path(),
                &tiny_config()
            )
            .is_err()
        );
        let directory = fixture(256);
        let path = directory.path().join("model.safetensors.index.json");
        let mut index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        index["weight_map"]
            .as_object_mut()
            .unwrap()
            .remove("model.norm.weight");
        std::fs::write(&path, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(
            super::super::decoder::checkpoint::CheckpointReader::open(
                directory.path(),
                &tiny_config()
            )
            .is_err()
        );
    }

    #[test]
    fn failed_layer_read_commits_no_generated_token() {
        let directory = fixture(256);
        let (_store, resolved) = imported(directory.path());
        let provider = resolved
            .load_with_layer_source(
                &DecoderDevice::ndarray(),
                "burn-ndarray",
                8,
                Some(Box::new(|_, _| {
                    Err(InferError::Backend("missing layer".into()))
                })),
            )
            .unwrap();
        let mut committed = Vec::new();
        let result = provider.generate_checkpointed(
            &GenerationRequest {
                prompt: "t3 t7".into(),
                max_tokens: 8,
                ..Default::default()
            },
            None,
            &mut |cp| {
                committed.push(cp.token_ids.clone());
                Ok(())
            },
            &mut || false,
        );
        assert!(result.is_err());
        assert_eq!(committed, vec![Vec::<u32>::new()]);
    }

    #[cfg(feature = "decoder-flex")]
    #[test]
    fn formatted_rows_require_measurement_and_refuse_format_change_on_resume() {
        use super::super::library::resolve_library_candidate;
        let directory = fixture(256);
        let mut store = eidetic::MemoryBackend::default();
        let template = super::super::prompt::PromptTemplate::from_bytes(include_bytes!(
            "../../templates/smollm2-instruct-v1.json"
        ))
        .unwrap();
        let template = pollster::block_on(save(&mut store, &template)).unwrap();
        let mut candidates = Vec::new();
        for format in [WeightFormat::Bf16, WeightFormat::F16] {
            let model = pollster::block_on(import_formatted_model(
                &mut store,
                directory.path(),
                "fixture/halves",
                "MPL-2.0",
                StreamingOptions::default(),
                format,
            ))
            .unwrap();
            let row = pollster::block_on(save(&mut store, &InferenceModelRow { model, template }))
                .unwrap();
            assert!(pollster::block_on(resolve_library_session(&mut store, row)).is_err());
            let resolved = pollster::block_on(resolve_library_candidate(&mut store, row)).unwrap();
            candidates.push((model, resolved));
        }
        assert_ne!(
            candidates[0].1.session("burn-flex").id().unwrap(),
            candidates[1].1.session("burn-flex").id().unwrap()
        );
        let store = Arc::new(Mutex::new(store));
        let provider = |resolved: &super::super::library::ResolvedInferenceRow| {
            let artifacts = resolved.layer_artifacts().unwrap();
            let store = store.clone();
            resolved
                .load_with_layer_source(
                    &DecoderDevice::flex(),
                    "burn-flex",
                    8,
                    Some(Box::new(move |i: usize, limit| {
                        pollster::block_on(artifacts[i].read(&mut *store.lock().unwrap(), limit))
                    })),
                )
                .unwrap()
        };
        let bf16 = provider(&candidates[0].1);
        let f16 = provider(&candidates[1].1);
        let request = GenerationRequest {
            prompt: "t3 t7 t11".into(),
            max_tokens: 4,
            seed: Some(42),
            temperature: 0.8,
            ..Default::default()
        };
        let boundary = std::cell::Cell::new(0);
        let paused = bf16
            .generate_checkpointed(
                &request,
                None,
                &mut |cp| {
                    boundary.set(cp.token_ids.len());
                    Ok(())
                },
                &mut || boundary.get() >= 2,
            )
            .unwrap();
        assert_eq!(paused.token_ids.len(), 2);
        assert!(
            f16.generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
                .is_err()
        );
        let resumed = bf16
            .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
            .unwrap();
        let full = bf16
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        assert_eq!(resumed.token_ids, full.token_ids);
        let index = {
            let mut store = store.lock().unwrap();
            let model = pollster::block_on(eidetic::ModelLibrary::load_model(
                &mut *store,
                candidates[1].0,
            ))
            .unwrap()
            .unwrap();
            pollster::block_on(eidetic::typed::load_typed::<FormattedLayerWeightIndex>(
                &mut *store,
                &mut NoFetcher,
                model.weight_blob,
            ))
            .unwrap()
            .unwrap()
        };
        let mut measurement = FormatQualification {
            baseline_model: candidates[0].0,
            baseline_loader: bf16.capability().loader.clone(),
            measured_loader: f16.capability().loader.clone(),
            prompts: vec![PromptDeviation {
                rendered_prompt: "fixed fixture prompt".into(),
                baseline_tokens: full.token_ids.clone(),
                format_tokens: vec![1],
                logit_rmse: 0.0,
                logit_max_abs: 0.0,
            }],
            encoded_weight_bytes: index.weights.shared.byte_size
                + index
                    .weights
                    .layers
                    .iter()
                    .map(|a| a.byte_size)
                    .sum::<u64>(),
            largest_layer_bytes: index
                .weights
                .layers
                .iter()
                .map(|a| a.byte_size)
                .max()
                .unwrap(),
            weight_payload_bytes: 1,
            streaming_resident_payload_bytes: 1,
            logical_stream_bytes: 1,
            elapsed_seconds: 1.0,
        };
        let mut store = store.lock().unwrap();
        let qualified = pollster::block_on(qualify_formatted_model(
            &mut *store,
            candidates[1].0,
            measurement.clone(),
        ))
        .unwrap();
        let row = pollster::block_on(save(
            &mut *store,
            &InferenceModelRow {
                model: qualified,
                template,
            },
        ))
        .unwrap();
        let resolved = pollster::block_on(resolve_library_session(&mut *store, row)).unwrap();
        assert_eq!(resolved.weight_format(), Some(WeightFormat::F16));
        assert!(
            resolved
                .load_with_layer_source(
                    &DecoderDevice::flex(),
                    "burn-wgpu",
                    8,
                    Some(Box::new(|_, _| unreachable!()))
                )
                .is_err()
        );
        measurement.prompts[0].logit_rmse = f64::NAN;
        assert!(
            pollster::block_on(qualify_formatted_model(
                &mut *store,
                candidates[1].0,
                measurement
            ))
            .is_err()
        );
    }
}
