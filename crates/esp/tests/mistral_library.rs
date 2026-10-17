// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the Mozilla Public
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

fn digest(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 4 * 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

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

fn run(device: DecoderDevice, loader: &str) {
    let work = PathBuf::from(
        std::env::var_os("ESP_ARCHITECTURE_WORK_DIR").expect("explicit I9 work root"),
    );
    let reference: serde_json::Value =
        serde_json::from_slice(include_bytes!("mistral_7b_reference.json")).unwrap();
    assert_eq!(reference["model_id"], "mistralai/Mistral-7B-Instruct-v0.1");
    assert_eq!(
        reference["revision"],
        "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca"
    );
    assert_eq!(reference["max_tokens"], 8);
    assert_eq!(reference["dtype"], "torch.float32");
    let source = PathBuf::from(std::env::var_os("USERPROFILE").unwrap())
        .join(".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca");
    let template_bytes = include_bytes!("../templates/mistral-instruct-v01.json");
    let template = PromptTemplate::from_bytes(template_bytes).unwrap();
    let template_hash: String = Sha256::digest(template_bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(reference["template_sha256"], template_hash);
    let options = StreamingOptions {
        query_chunk: 32,
        max_layer_bytes: 512 * 1024 * 1024,
        max_shared_bytes: 2 * 1024 * 1024 * 1024,
    };
    let identity = serde_json::json!({"artifacts":reference["artifacts"],"template":template,"options":options});
    let folder = work.join("mistral");
    std::fs::create_dir_all(&folder).unwrap();
    let mut store = muniment::RedbBackend::open(folder.join("model.redb")).unwrap();
    let cached = std::fs::read(folder.join("row-cache.json"))
        .ok()
        .map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).unwrap());
    let row: ManifestId = if let Some(cache) = cached.filter(|cache| cache["inputs"] == identity) {
        serde_json::from_value(cache["row"].clone()).unwrap()
    } else {
        for (file, hash) in reference["artifacts"].as_object().unwrap() {
            assert!(!file.contains(['/', '\\', ':']));
            assert_eq!(digest(&source.join(file)), hash.as_str().unwrap());
        }
        eprintln!("importing pinned Mistral library row");
        let model = pollster::block_on(import_layered_model(
            &mut store,
            &source,
            "mistralai/Mistral-7B-Instruct-v0.1",
            "Apache-2.0",
            options.clone(),
        ))
        .unwrap();
        let template = save(&mut store, &template);
        let row = save(&mut store, &InferenceModelRow { model, template });
        std::fs::write(
            folder.join("row-cache.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"inputs":identity,"row":row})).unwrap(),
        )
        .unwrap();
        row
    };
    let resolved = pollster::block_on(resolve_library_session(&mut store, row)).unwrap();
    let store = Arc::new(Mutex::new(store));
    let reader = store.clone();
    let source = resolved
        .artifact_source(move |artifact, limit| {
            pollster::block_on(artifact.read(&mut *reader.lock().unwrap(), limit))
        })
        .unwrap();
    let provider = resolved
        .load_with_layer_source(&device, loader, 8, Some(Box::new(source)))
        .unwrap();
    let session = provider.session_identity();
    let mut rows = Vec::new();
    for expected in reference["rows"].as_array().unwrap() {
        let request = GenerationRequest {
            prompt: expected["question"].as_str().unwrap().into(),
            max_tokens: 8,
            ..Default::default()
        };
        assert_eq!(
            provider.prepare(&request, "").unwrap().request.prompt,
            expected["prompt"].as_str().unwrap()
        );
        let start = Instant::now();
        let result = provider
            .generate_checkpointed(
                &request,
                None,
                &mut |checkpoint| {
                    eprintln!(
                        "{loader} {} token={} seconds={:.3}",
                        request.prompt,
                        checkpoint.token_ids.len(),
                        start.elapsed().as_secs_f64()
                    );
                    Ok(())
                },
                &mut || false,
            )
            .unwrap();
        assert!(result.complete);
        assert!(!result.token_ids.is_empty());
        assert_eq!(serde_json::json!(result.prompt_ids), expected["prompt_ids"]);
        assert_eq!(serde_json::json!(result.token_ids), expected["token_ids"]);
        rows.push(serde_json::json!({"question":request.prompt,"token_ids":result.token_ids,"text":result.text,"seconds":start.elapsed().as_secs_f64()}));
    }
    let request = GenerationRequest {
        prompt: reference["rows"][1]["question"].as_str().unwrap().into(),
        max_tokens: 8,
        ..Default::default()
    };
    let count = std::cell::Cell::new(0);
    let paused = provider
        .generate_checkpointed(
            &request,
            None,
            &mut |checkpoint| {
                count.set(checkpoint.token_ids.len());
                Ok(())
            },
            &mut || count.get() >= 3,
        )
        .unwrap();
    assert_eq!(paused.token_ids.len(), 3);
    assert!(!paused.complete);
    let paused = serde_json::from_slice(&serde_json::to_vec(&paused).unwrap()).unwrap();
    let resumed = provider
        .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
        .unwrap();
    assert!(resumed.complete);
    assert_eq!(
        serde_json::json!(resumed.token_ids),
        reference["rows"][1]["token_ids"]
    );
    let stats = provider.bound().provider().streaming_stats().unwrap();
    assert!(stats.layer_reads > 0);
    assert!(stats.largest_encoded_layer <= options.max_layer_bytes as u64);
    assert_eq!(stats.expert_reads, 0);
    std::fs::write(work.join(format!("mistral-{loader}-result.json")), serde_json::to_vec_pretty(&serde_json::json!({
        "model":reference["model_id"],"revision":reference["revision"],"row":row,"session":session,"loader":loader,"dtype":"f32","options":options,
        "rows":rows,"reads":stats,"resume":{"boundary":3,"serialized_checkpoint":true,"token_ids":resumed.token_ids},
        "scope":"Streamed trained-family model-library row on one computer; no whole-model residence, installed resident or largest-per-family claim"
    })).unwrap()).unwrap();
    drop(provider);
    device.sync().unwrap();
    assert_eq!(Arc::strong_count(&store), 1);
    drop(store);
    drop(muniment::RedbBackend::open(folder.join("model.redb")).unwrap());
}

#[test]
#[ignore = "requires pinned Mistral checkpoint and ESP_ARCHITECTURE_WORK_DIR"]
fn streamed_mistral_cpu_matches_reference() {
    run(DecoderDevice::flex(), "burn-flex");
}

#[cfg(feature = "decoder-wgpu")]
#[test]
#[ignore = "requires pinned Mistral checkpoint, explicit work root and a discrete GPU"]
fn streamed_mistral_gpu_matches_reference() {
    use esp::infer::decoder::{DecoderGpuKind, probe_gpu_adapter};
    let adapter = probe_gpu_adapter(DecoderGpuKind::DiscreteGpu(0)).unwrap();
    assert!(adapter.matched_requested_class, "{adapter}");
    run(
        DecoderDevice::wgpu(DecoderGpuKind::DiscreteGpu(0)),
        "burn-wgpu",
    );
}
