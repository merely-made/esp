// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use eidetic::typed::{TypedPayload, save_typed};
use eidetic::{ManifestId, ModelLibrary, PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope};
use esp::infer::decoder::DecoderDevice;
use esp::infer::library::{InferenceModelRow, load_library_session};
use esp::infer::prompt::PromptTemplate;
use esp::infer::{GenerationRequest, InferenceProvider};
use sha2::{Digest, Sha256};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::Instant;

fn save<T: TypedPayload>(store: &mut dyn eidetic::Store, value: &T) -> ManifestId {
    pollster::block_on(save_typed(
        store,
        value,
        vec![],
        PrivacyClass::LocalOnly,
        ProvenanceRecord::self_imported("local"),
        TrustEnvelope::self_asserted(),
        Timestamp::ZERO,
    ))
    .unwrap()
}

#[test]
#[ignore = "requires ESP_SMOLLM2_DIR with pinned 724MB checkpoint; run in release mode"]
fn library_cpu_matches_independent_greedy_reference() {
    library_matches_reference(
        DecoderDevice::ndarray(),
        "burn-ndarray",
        serde_json::Value::Null,
    );
}

#[cfg(feature = "decoder-wgpu")]
#[test]
#[ignore = "requires pinned ESP_SMOLLM2_DIR and a discrete GPU"]
fn library_gpu_matches_independent_greedy_reference() {
    use esp::infer::decoder::{DecoderGpuKind, GpuDeviceType, probe_gpu_adapter};
    let kind = DecoderGpuKind::DiscreteGpu(0);
    let adapter = probe_gpu_adapter(kind.clone()).unwrap();
    assert_eq!(adapter.device_type, GpuDeviceType::Discrete);
    assert!(adapter.matched_requested_class);
    library_matches_reference(
        DecoderDevice::wgpu(kind),
        "burn-wgpu",
        serde_json::to_value(adapter).unwrap(),
    );
}

fn library_matches_reference(device: DecoderDevice, loader: &str, adapter: serde_json::Value) {
    let directory = PathBuf::from(std::env::var_os("ESP_SMOLLM2_DIR").expect("ESP_SMOLLM2_DIR"));
    let reference: serde_json::Value =
        serde_json::from_slice(include_bytes!("smollm2_360m_reference.json")).unwrap();
    let read = |name: &str| {
        let bytes = std::fs::read(directory.join(name)).unwrap();
        let hash: String = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(
            hash,
            reference["artifacts"][name].as_str().unwrap(),
            "exact {name}"
        );
        bytes
    };
    let config = read("config.json");
    let tokenizer = read("tokenizer.json");
    let weights = read("model.safetensors");
    read("tokenizer_config.json");
    let mut store = muniment::MemoryBackend::default();
    let model = pollster::block_on(ModelLibrary::save_model_with_components(
        &mut store,
        reference["model_id"].as_str().unwrap(),
        "llama",
        "Apache-2.0",
        serde_json::from_slice(&config).unwrap(),
        &weights,
        &tokenizer,
        vec![],
        vec![],
        PrivacyClass::LocalOnly,
        ProvenanceRecord::self_imported("local"),
        TrustEnvelope::self_asserted(),
        Timestamp::ZERO,
    ))
    .unwrap();
    drop(weights);
    let template =
        PromptTemplate::from_bytes(include_bytes!("../templates/smollm2-instruct-v1.json"))
            .unwrap();
    let template_id = save(&mut store, &template);
    let row = save(
        &mut store,
        &InferenceModelRow {
            model,
            template: template_id,
        },
    );
    let start = Instant::now();
    let provider =
        pollster::block_on(load_library_session(&mut store, row, &device, loader, 16)).unwrap();
    let load_seconds = start.elapsed().as_secs_f64();
    // The store is not needed during generation.
    drop(store);
    let identity = provider.session_identity().unwrap();
    let mut rows = vec![];
    for expected in reference["rows"].as_array().unwrap() {
        let question = expected["question"].as_str().unwrap();
        let request = GenerationRequest {
            prompt: question.into(),
            max_tokens: 16,
            ..Default::default()
        };
        let prepared = provider.prepare(&request, "").unwrap();
        assert_eq!(prepared.session_id, provider.bound().session_id());
        assert_eq!(
            prepared.request.prompt,
            expected["prompt"].as_str().unwrap()
        );
        let start = Instant::now();
        let generated = provider
            .bound()
            .provider()
            .generate_streaming_observed(
                &prepared.request,
                &mut |_| ControlFlow::Continue(()),
                &mut |_| {},
            )
            .unwrap();
        let seconds = start.elapsed().as_secs_f64();
        assert_eq!(
            serde_json::json!(generated.token_ids),
            expected["token_ids"],
            "greedy ids for {question}"
        );
        assert_eq!(generated.text, expected["text"].as_str().unwrap());
        assert_eq!(
            provider.generate(&request).unwrap(),
            generated.text,
            "actor-compatible provider path"
        );
        let boundary = std::cell::Cell::new(0usize);
        let uninterrupted = provider
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        assert_eq!(uninterrupted.token_ids, generated.token_ids);
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
        let resumed = provider
            .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
            .unwrap();
        let first_difference = uninterrupted
            .token_ids
            .iter()
            .zip(&resumed.token_ids)
            .position(|(a, b)| a != b)
            .or_else(|| {
                (uninterrupted.token_ids.len() != resumed.token_ids.len())
                    .then_some(uninterrupted.token_ids.len().min(resumed.token_ids.len()))
            });
        assert!(
            first_difference.is_none(),
            "prefill resume differed at {first_difference:?}; same-backend qualification requires a numerical trace if this changes"
        );
        rows.push(serde_json::json!({"question": question, "token_ids": generated.token_ids, "text": generated.text, "seconds": seconds,
            "resume_at": paused.token_ids.len(), "resumed_token_ids": resumed.token_ids, "draws": resumed.draws, "first_differing_token": first_difference}));
    }
    let receipt = serde_json::json!({"backend": loader, "adapter": adapter, "model": reference["model_id"], "revision": reference["revision"],
        "row": row, "session": identity, "load_seconds": load_seconds, "rows": rows});
    println!("{receipt}");
    if let Some(path) = std::env::var_os("ESP_INFERENCE_RECEIPT") {
        std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    }
}
