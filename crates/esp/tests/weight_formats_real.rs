// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0
#![cfg(all(
    feature = "decoder-flex",
    feature = "model-session",
    not(target_family = "wasm")
))]

use eidetic::{
    ManifestId,
    typed::{TypedPayload, load_typed, save_typed},
};
use esp::infer::decoder::streaming::StreamingOptions;
use esp::infer::decoder::{DecoderDevice, DecoderProvider, WeightFormat};
use esp::infer::layer_library::{
    FormatQualification, FormattedLayerWeightIndex, PromptDeviation, import_formatted_model,
    qualify_formatted_model,
};
use esp::infer::library::{InferenceModelRow, resolve_library_candidate, resolve_library_session};
use esp::infer::prompt::{PromptTemplate, SessionProvider};
use esp::infer::{GenerationRequest, InferError, InferenceProvider};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

fn save<T: TypedPayload>(store: &mut dyn eidetic::Store, value: &T) -> ManifestId {
    pollster::block_on(save_typed(
        store,
        value,
        vec![],
        eidetic::PrivacyClass::LocalOnly,
        eidetic::ProvenanceRecord::self_imported("local"),
        eidetic::TrustEnvelope::self_asserted(),
        eidetic::Timestamp::ZERO,
    ))
    .unwrap()
}

fn load(
    store: Arc<Mutex<muniment::RedbBackend>>,
    row: ManifestId,
    device: &DecoderDevice,
    loader: &str,
    candidate: bool,
) -> SessionProvider<DecoderProvider> {
    let resolved = {
        let mut store = store.lock().unwrap();
        if candidate {
            pollster::block_on(resolve_library_candidate(&mut *store, row)).unwrap()
        } else {
            pollster::block_on(resolve_library_session(&mut *store, row)).unwrap()
        }
    };
    let artifacts = resolved.layer_artifacts().unwrap();
    let source = move |index: usize, limit: usize| {
        pollster::block_on(artifacts[index].read(&mut *store.lock().unwrap(), limit))
    };
    resolved
        .load_with_layer_source(device, loader, 16, Some(Box::new(source)))
        .unwrap()
}

#[test]
#[ignore = "recorded half-format receipts and pinned ESP_SMOLLM2_DIR"]
fn saved_half_imports_keep_their_exact_artifact_identity() {
    let work = PathBuf::from(std::env::var_os("ESP_FORMAT_WORK_DIR").unwrap());
    let directory = PathBuf::from(std::env::var_os("ESP_SMOLLM2_DIR").unwrap());
    let mut store = muniment::RedbBackend::open(work.join("model-formats.redb")).unwrap();
    let mut verified = Vec::new();
    for (name, format) in [("bf16", WeightFormat::Bf16), ("f16", WeightFormat::F16)] {
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(work.join(format!("{name}-burn-flex.json"))).unwrap(),
        )
        .unwrap();
        let expected =
            ManifestId(eidetic::Hash::parse(receipt["candidate_model"].as_str().unwrap()).unwrap());
        let model = pollster::block_on(import_formatted_model(
            &mut store,
            &directory,
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "Apache-2.0",
            StreamingOptions::default(),
            format,
        ))
        .unwrap();
        assert_eq!(
            model, expected,
            "{name} content identity after import guards"
        );
        let manifest = pollster::block_on(eidetic::ModelLibrary::load_model(&mut store, model))
            .unwrap()
            .unwrap();
        let index = pollster::block_on(load_typed::<FormattedLayerWeightIndex>(
            &mut store,
            &mut eidetic::NoFetcher,
            manifest.weight_blob,
        ))
        .unwrap()
        .unwrap();
        verified.push(serde_json::json!({"format":format,"candidate_model":model.to_string(),
            "shared_encoded_bytes":index.weights.shared.byte_size,"encoded_layer_pass_bytes":index.weights.layers.iter().map(|a|a.byte_size).sum::<u64>()}));
    }
    std::fs::write(
        work.join("verified-half-artifacts.json"),
        serde_json::to_vec_pretty(&verified).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "pinned SmolLM2 snapshot, ESP_FORMAT_WORK_DIR, ESP_WEIGHT_FORMAT and ESP_WEIGHT_LOADER"]
fn measure_real_weight_format() {
    let work = PathBuf::from(std::env::var_os("ESP_FORMAT_WORK_DIR").expect("ESP_FORMAT_WORK_DIR"));
    let directory = PathBuf::from(std::env::var_os("ESP_SMOLLM2_DIR").expect("ESP_SMOLLM2_DIR"));
    let format_name = std::env::var("ESP_WEIGHT_FORMAT").unwrap();
    let format: WeightFormat = format_name.parse().unwrap();
    let loader = std::env::var("ESP_WEIGHT_LOADER").unwrap();
    let device = if loader == "burn-flex" {
        DecoderDevice::flex()
    } else {
        assert_eq!(loader, "burn-wgpu");
        #[cfg(feature = "decoder-wgpu")]
        {
            let kind = esp::infer::decoder::DecoderGpuKind::DiscreteGpu(0);
            let facts = esp::infer::decoder::probe_gpu_adapter(kind.clone()).unwrap();
            assert!(facts.matched_requested_class);
            assert_eq!(
                facts.device_type,
                esp::infer::decoder::GpuDeviceType::Discrete
            );
            DecoderDevice::wgpu(kind)
        }
        #[cfg(not(feature = "decoder-wgpu"))]
        {
            panic!("decoder-wgpu required")
        }
    };
    format.validate_device(&device).unwrap();
    let reference: serde_json::Value =
        serde_json::from_slice(include_bytes!("smollm2_360m_reference.json")).unwrap();
    for (name, hash) in reference["artifacts"].as_object().unwrap() {
        // The largest source is 724 MB; hash using a bounded reader.
        use std::io::Read;
        let mut file = std::fs::File::open(directory.join(name)).unwrap();
        let mut digest = Sha256::new();
        let mut bytes = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut bytes).unwrap();
            if n == 0 {
                break;
            }
            digest.update(&bytes[..n]);
        }
        assert_eq!(
            digest
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            hash.as_str().unwrap()
        );
    }
    std::fs::create_dir_all(&work).unwrap();
    let store = Arc::new(Mutex::new(
        muniment::RedbBackend::open(work.join("model-formats.redb")).unwrap(),
    ));
    let template =
        PromptTemplate::from_bytes(include_bytes!("../templates/smollm2-instruct-v1.json"))
            .unwrap();
    let import_start = Instant::now();
    let (candidate, row, template_ref, index) = {
        let mut store = store.lock().unwrap();
        let model = pollster::block_on(import_formatted_model(
            &mut *store,
            &directory,
            reference["model_id"].as_str().unwrap(),
            "Apache-2.0",
            StreamingOptions::default(),
            format,
        ))
        .unwrap();
        let template_ref = save(&mut *store, &template);
        let row = save(
            &mut *store,
            &InferenceModelRow {
                model,
                template: template_ref,
            },
        );
        assert!(pollster::block_on(resolve_library_session(&mut *store, row)).is_err());
        let manifest = pollster::block_on(eidetic::ModelLibrary::load_model(&mut *store, model))
            .unwrap()
            .unwrap();
        let index = pollster::block_on(load_typed::<FormattedLayerWeightIndex>(
            &mut *store,
            &mut eidetic::NoFetcher,
            manifest.weight_blob,
        ))
        .unwrap()
        .unwrap();
        (model, row, template_ref, index)
    };
    let import_seconds = import_start.elapsed().as_secs_f64();
    let encoded_weight_bytes = index.weights.shared.byte_size
        + index
            .weights
            .layers
            .iter()
            .map(|a| a.byte_size)
            .sum::<u64>();
    let largest_layer_bytes = index
        .weights
        .layers
        .iter()
        .map(|a| a.byte_size)
        .max()
        .unwrap();
    let (weight_payload_bytes, streaming_resident_payload_bytes) = {
        let mut store = store.lock().unwrap();
        let payload = |bytes: Vec<u8>| {
            safetensors::SafeTensors::deserialize(&bytes)
                .unwrap()
                .tensors()
                .iter()
                .map(|(_, v)| v.data().len() as u64)
                .sum::<u64>()
        };
        let shared = payload(
            pollster::block_on(
                index
                    .weights
                    .shared
                    .read(&mut *store, index.weights.options.max_shared_bytes),
            )
            .unwrap(),
        );
        let mut total = shared;
        let mut largest = 0;
        for layer in &index.weights.layers {
            let bytes = payload(
                pollster::block_on(layer.read(&mut *store, index.weights.options.max_layer_bytes))
                    .unwrap(),
            );
            total += bytes;
            largest = largest.max(bytes);
        }
        (total, shared + largest)
    };
    let baseline: Option<serde_json::Value> =
        if format == WeightFormat::Bf16 && loader == "burn-flex" {
            None
        } else {
            Some(
                serde_json::from_slice(&std::fs::read(work.join("bf16-burn-flex.json")).unwrap())
                    .unwrap(),
            )
        };
    let provider = load(store.clone(), row, &device, &loader, true);
    let start = Instant::now();
    let mut raw = Vec::new();
    let mut deviations = Vec::new();
    let mut first_checkpoint = None;
    for (i, expected) in reference["rows"].as_array().unwrap().iter().enumerate() {
        let question = expected["question"].as_str().unwrap();
        let rendered = template.render(question, "").unwrap();
        let logits = provider
            .bound()
            .provider()
            .next_token_logits(&rendered)
            .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        let request = GenerationRequest {
            prompt: question.into(),
            max_tokens: 16,
            ..Default::default()
        };
        let row_start = Instant::now();
        let generated = provider
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        assert_eq!(
            serde_json::json!(generated.prompt_ids),
            expected["prompt_ids"]
        );
        let baseline_tokens = baseline
            .as_ref()
            .map(|b| serde_json::from_value::<Vec<u32>>(b["rows"][i]["token_ids"].clone()).unwrap())
            .unwrap_or_else(|| generated.token_ids.clone());
        let baseline_logits: Vec<f32> = baseline
            .as_ref()
            .map(|b| serde_json::from_value(b["rows"][i]["logits"].clone()).unwrap())
            .unwrap_or_else(|| logits.clone());
        assert_eq!(baseline_logits.len(), logits.len());
        let errors: Vec<_> = logits
            .iter()
            .zip(&baseline_logits)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .collect();
        deviations.push(PromptDeviation {
            rendered_prompt: rendered,
            baseline_tokens,
            format_tokens: generated.token_ids.clone(),
            logit_rmse: (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt(),
            logit_max_abs: errors.into_iter().fold(0.0, f64::max),
        });
        raw.push(serde_json::json!({"question":question,"token_ids":generated.token_ids,"text":generated.text,"logits":logits,"seconds":row_start.elapsed().as_secs_f64()}));
        if i == 0 {
            first_checkpoint = Some(generated);
        }
    }
    let request = GenerationRequest {
        prompt: reference["rows"][0]["question"].as_str().unwrap().into(),
        max_tokens: 16,
        ..Default::default()
    };
    let boundary = std::cell::Cell::new(0);
    let paused = provider
        .generate_checkpointed(
            &request,
            None,
            &mut |cp| {
                boundary.set(cp.token_ids.len());
                Ok(())
            },
            &mut || boundary.get() >= 3,
        )
        .unwrap();
    assert_eq!(paused.token_ids.len(), 3);
    let resumed = provider
        .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
        .unwrap();
    assert_eq!(
        resumed.token_ids,
        first_checkpoint.as_ref().unwrap().token_ids
    );
    let elapsed_seconds = start.elapsed().as_secs_f64();
    let stats = provider.bound().provider().streaming_stats().unwrap();
    let candidate_session = provider.session_identity().unwrap();
    let measured_loader = provider.capability().loader.clone();
    drop(provider);
    device.sync().unwrap();
    let baseline_model = baseline
        .as_ref()
        .map(|b| ManifestId(eidetic::Hash::parse(b["candidate_model"].as_str().unwrap()).unwrap()))
        .unwrap_or(candidate);
    let qualification = FormatQualification {
        baseline_model,
        baseline_loader: format!("burn-flex/layer-stream-v2/{}", WeightFormat::Bf16.id()),
        measured_loader,
        prompts: deviations,
        encoded_weight_bytes,
        weight_payload_bytes,
        streaming_resident_payload_bytes,
        largest_layer_bytes,
        logical_stream_bytes: stats.encoded_bytes_read,
        elapsed_seconds,
    };
    let qualified_model = pollster::block_on(qualify_formatted_model(
        &mut *store.lock().unwrap(),
        candidate,
        qualification.clone(),
    ))
    .unwrap();
    let qualified_row = save(
        &mut *store.lock().unwrap(),
        &InferenceModelRow {
            model: qualified_model,
            template: template_ref,
        },
    );
    let provider = load(store.clone(), qualified_row, &device, &loader, false);
    // Qualification changes the model envelope, so candidate checkpoints also
    // cannot be resumed silently under the newly published row.
    assert_ne!(provider.session_identity().unwrap(), candidate_session);
    let err = provider
        .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
        .unwrap_err();
    assert!(matches!(err, InferError::InvalidRequest(_)), "{err:?}");
    if let Some(baseline) = &baseline {
        let baseline_row: ManifestId =
            ManifestId(eidetic::Hash::parse(baseline["qualified_row"].as_str().unwrap()).unwrap());
        let bf16 = load(
            store.clone(),
            baseline_row,
            &DecoderDevice::flex(),
            "burn-flex",
            false,
        );
        assert_ne!(bf16.session_identity(), provider.session_identity());
        assert!(
            bf16.generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
                .is_err()
        );
    }
    let receipt = serde_json::json!({"schema":"mere.weight-format-measurement/v1", "source":reference["artifacts"],
        "candidate_model":candidate.to_string(),"qualified_model":qualified_model.to_string(),"qualified_row":qualified_row.to_string(),
        "format":format,"loader":loader,"import_seconds":import_seconds,"qualification":qualification,"rows":raw,
        "shared_encoded_bytes":index.weights.shared.byte_size,"encoded_layer_pass_bytes":index.weights.layers.iter().map(|a|a.byte_size).sum::<u64>(),"statistics":stats,
        "pause_resume_matches":true,"format_change_resume_refused":true,"candidate_selection_refused":true,
        "memory_scope":"payload counts exclude allocator overhead, activations, KV, temporary kernel storage and OS cache; host measurement records actual process and global GPU peaks"});
    let output = work.join(format!("{format_name}-{loader}.json"));
    std::fs::write(output, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    eprintln!(
        "format {} qualified: {} bytes, {:.3}s generation",
        format.id(),
        encoded_weight_bytes,
        elapsed_seconds
    );
}
