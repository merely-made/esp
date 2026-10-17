// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
#![cfg(all(
    feature = "decoder-flex",
    feature = "model-session",
    not(target_family = "wasm")
))]

use eidetic::{
    ManifestId,
    typed::{TypedPayload, save_typed},
};
use esp::infer::decoder::{DecoderDevice, streaming::StreamingOptions};
use esp::infer::layer_library::import_layered_model;
use esp::infer::library::{InferenceModelRow, resolve_library_session};
use esp::infer::prompt::PromptTemplate;
use esp::infer::{GenerationRequest, InferenceProvider};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
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

fn verify(path: &Path, expected: &str) {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let hash: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(hash, expected, "pinned {}", path.display());
}

fn run(name: &str, device: DecoderDevice, loader: &str) {
    let work = PathBuf::from(
        std::env::var_os("ESP_ARCHITECTURE_WORK_DIR").expect("explicit architecture fixture root"),
    );
    let (reference, template): (&[u8], &[u8]) = match name {
        "qwen2" => (
            include_bytes!("qwen2_05b_reference.json"),
            include_bytes!("../templates/qwen2-instruct-v1.json"),
        ),
        "qwen3" => (
            include_bytes!("qwen3_06b_reference.json"),
            include_bytes!("../templates/qwen3-nonthinking-v1.json"),
        ),
        _ => unreachable!(),
    };
    let reference: serde_json::Value = serde_json::from_slice(reference).unwrap();
    let model_id = reference["model_id"].as_str().unwrap();
    let source = PathBuf::from(std::env::var_os("USERPROFILE").unwrap())
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", model_id.replace('/', "--")))
        .join("snapshots")
        .join(reference["revision"].as_str().unwrap());
    for (file, hash) in reference["artifacts"].as_object().unwrap() {
        verify(&source.join(file), hash.as_str().unwrap());
    }
    let template_hash: String = Sha256::digest(template)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(
        template_hash,
        reference["template_sha256"].as_str().unwrap()
    );
    std::fs::create_dir_all(work.join(name)).unwrap();
    let mut store = muniment::RedbBackend::open(work.join(name).join("model.redb")).unwrap();
    let options = StreamingOptions {
        query_chunk: 32,
        max_layer_bytes: 256 * 1024 * 1024,
        max_shared_bytes: 2 * 1024 * 1024 * 1024,
    };
    let model = pollster::block_on(import_layered_model(
        &mut store,
        &source,
        model_id,
        "Apache-2.0",
        options,
    ))
    .unwrap();
    let template = save(&mut store, &PromptTemplate::from_bytes(template).unwrap());
    let row = save(&mut store, &InferenceModelRow { model, template });
    let resolved = pollster::block_on(resolve_library_session(&mut store, row)).unwrap();
    let artifacts = Arc::new(resolved.layer_artifacts().unwrap());
    let store = Arc::new(Mutex::new(store));
    let mut results = Vec::new();
    let mut identity = None;
    for resident in [true, false] {
        let source = {
            let artifacts = artifacts.clone();
            let store = store.clone();
            move |index: usize, limit: usize| {
                pollster::block_on(artifacts[index].read(&mut *store.lock().unwrap(), limit))
            }
        };
        let start = Instant::now();
        let provider = if resident {
            resolved.load_resident_with_layer_source(&device, loader, 16, Box::new(source))
        } else {
            resolved.load_with_layer_source(&device, loader, 16, Some(Box::new(source)))
        }
        .unwrap();
        let session = serde_json::to_value(provider.session_identity()).unwrap();
        if let Some(old) = &identity {
            assert_eq!(old, &session);
        } else {
            identity = Some(session);
        }
        let mut rows = Vec::new();
        for expected in reference["rows"].as_array().unwrap() {
            let request = GenerationRequest {
                prompt: expected["question"].as_str().unwrap().into(),
                max_tokens: 16,
                ..Default::default()
            };
            let prepared = provider.prepare(&request, "").unwrap();
            assert_eq!(
                prepared.request.prompt,
                expected["prompt"].as_str().unwrap()
            );
            let generated = provider
                .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
                .unwrap();
            assert!(generated.complete);
            assert_eq!(
                serde_json::json!(generated.prompt_ids),
                expected["prompt_ids"]
            );
            assert_eq!(
                serde_json::json!(generated.token_ids),
                expected["token_ids"],
                "{name}/{resident}"
            );
            rows.push(serde_json::json!({"question":request.prompt,"token_ids":generated.token_ids,"text":generated.text}));
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
            serde_json::json!(resumed.token_ids),
            reference["rows"][0]["token_ids"]
        );
        results.push(serde_json::json!({"resident":resident,"seconds":start.elapsed().as_secs_f64(),"rows":rows,"resume_boundary":3,"resume_matched":true}));
        drop(provider);
        device.sync().unwrap();
    }
    let receipt = serde_json::json!({"model":model_id,"revision":reference["revision"],"row":row.to_string(),"session":identity,"loader":loader,"results":results,
        "scope":"Pinned real model-library row and data template on one physical device; explicit resident and streamed kernel placement, not host budget admission or fleet scale"});
    let path = work.join(format!("{name}-{loader}-result.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    eprintln!("qualified {model_id} via {loader}: {}", path.display());
}

#[test]
#[ignore = "requires pinned cached checkpoints and ESP_ARCHITECTURE_WORK_DIR"]
fn qwen2_cpu_model_row() {
    run("qwen2", DecoderDevice::flex(), "burn-flex");
}
#[test]
#[ignore = "requires pinned cached checkpoints and ESP_ARCHITECTURE_WORK_DIR"]
fn qwen3_cpu_model_row() {
    run("qwen3", DecoderDevice::flex(), "burn-flex");
}
#[cfg(feature = "decoder-wgpu")]
#[test]
#[ignore = "requires pinned cached checkpoints, discrete GPU and ESP_ARCHITECTURE_WORK_DIR"]
fn qwen2_gpu_model_row() {
    run(
        "qwen2",
        DecoderDevice::wgpu(esp::infer::decoder::DecoderGpuKind::DiscreteGpu(0)),
        "burn-wgpu",
    );
}
#[cfg(feature = "decoder-wgpu")]
#[test]
#[ignore = "requires pinned cached checkpoints, discrete GPU and ESP_ARCHITECTURE_WORK_DIR"]
fn qwen3_gpu_model_row() {
    run(
        "qwen3",
        DecoderDevice::wgpu(esp::infer::decoder::DecoderGpuKind::DiscreteGpu(0)),
        "burn-wgpu",
    );
}

#[test]
fn expert_library_identity_and_read_failures_are_enforced() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/support/architectures");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
    let expected = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == "qwen2_moe")
        .unwrap();
    let folder = tempfile::tempdir().unwrap();
    for file in ["config.json", "model.safetensors"] {
        std::fs::copy(root.join("qwen2_moe").join(file), folder.path().join(file)).unwrap();
    }
    let vocab: serde_json::Map<String, serde_json::Value> = (0..32)
        .map(|id| (format!("t{id}"), serde_json::json!(id)))
        .collect();
    let tokenizer = serde_json::json!({"version":"1.0","pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordLevel","vocab":vocab,"unk_token":"t0"}});
    std::fs::write(
        folder.path().join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    let mut store = muniment::MemoryBackend::default();
    let options = StreamingOptions {
        query_chunk: 2,
        max_layer_bytes: 1024 * 1024,
        max_shared_bytes: 1024 * 1024,
    };
    let model = pollster::block_on(import_layered_model(
        &mut store,
        folder.path(),
        "fixture/qwen2-moe",
        "MPL-2.0",
        options,
    ))
    .unwrap();
    let manifest = pollster::block_on(eidetic::ModelLibrary::load_model(&mut store, model))
        .unwrap()
        .unwrap();
    assert_eq!(manifest.architecture, "qwen2_moe");
    let mut template =
        PromptTemplate::from_bytes(include_bytes!("../templates/qwen2-instruct-v1.json")).unwrap();
    template.system_prompt.clear();
    template.user.prefix.clear();
    template.user.suffix.clear();
    template.assistant.prefix.clear();
    template.eos.clear();
    template.stop.clear();
    let template = save(&mut store, &template);
    let row = save(&mut store, &InferenceModelRow { model, template });
    let resolved = pollster::block_on(resolve_library_session(&mut store, row)).unwrap();
    let store = Arc::new(Mutex::new(store));
    let source = {
        let store = store.clone();
        resolved
            .artifact_source(move |artifact, limit| {
                pollster::block_on(artifact.read(&mut *store.lock().unwrap(), limit))
            })
            .unwrap()
    };
    let request = GenerationRequest {
        prompt: expected["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| format!("t{}", id.as_u64().unwrap()))
            .collect::<Vec<_>>()
            .join(" "),
        max_tokens: 4,
        ..Default::default()
    };
    let provider = resolved
        .load_with_layer_source(
            &DecoderDevice::flex(),
            "burn-flex",
            4,
            Some(Box::new(source.clone())),
        )
        .unwrap();
    let result = provider
        .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
        .unwrap();
    assert_eq!(
        serde_json::json!(result.token_ids),
        expected["generated_ids"]
    );
    let session = provider.session_identity();
    drop(provider);
    let provider = resolved
        .load_resident_with_layer_source(
            &DecoderDevice::flex(),
            "burn-flex",
            4,
            Box::new(source.clone()),
        )
        .unwrap();
    assert_eq!(provider.session_identity(), session);
    assert_eq!(
        serde_json::json!(
            provider
                .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
                .unwrap()
                .token_ids
        ),
        expected["generated_ids"]
    );
    drop(provider);
    let device = DecoderDevice::flex();
    let stage = resolved
        .load_stage(
            &device,
            "burn-flex",
            esp::infer::decoder::stage::LayerRange { first: 0, end: 2 },
            source.as_ref(),
        )
        .unwrap();
    let provider = resolved
        .load_with_stages(&device, "burn-flex", 4, Box::new(stage))
        .unwrap();
    assert_eq!(provider.session_identity(), session);
    assert_eq!(
        serde_json::json!(
            provider
                .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
                .unwrap()
                .token_ids
        ),
        expected["generated_ids"]
    );
    drop(provider);
    let bad = esp::infer::decoder::streaming::with_expert_reader(
        move |index, limit| source.read_layer(index, limit),
        |_, _, _| {
            Err(esp::infer::InferError::InvalidWeights(
                "unavailable expert".into(),
            ))
        },
    );
    let provider = resolved
        .load_with_layer_source(&DecoderDevice::flex(), "burn-flex", 4, Some(Box::new(bad)))
        .unwrap();
    let mut committed = 0;
    assert!(
        provider
            .generate_checkpointed(
                &request,
                None,
                &mut |cp| {
                    committed = committed.max(cp.token_ids.len());
                    Ok(())
                },
                &mut || false
            )
            .is_err()
    );
    assert_eq!(committed, 0);
}
