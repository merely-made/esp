// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
#![cfg(all(
    feature = "decoder-wgpu",
    feature = "model-session",
    not(target_family = "wasm")
))]

use eidetic::{
    ManifestId,
    typed::{TypedPayload, save_typed},
};
use esp::infer::decoder::{
    DecoderConfig, DecoderDevice, DecoderGpuKind, checkpoint::CheckpointReader,
    streaming::StreamingOptions,
};
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
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

#[test]
#[ignore = "requires the pinned 115 GB Qwen2 MoE snapshot, explicit work root and a discrete GPU"]
fn larger_than_ram_moe_completes_with_selected_expert_streaming() {
    let work = PathBuf::from(
        std::env::var_os("ESP_ARCHITECTURE_WORK_DIR").expect("explicit I9 work root"),
    );
    let source=PathBuf::from(std::env::var_os("USERPROFILE").unwrap()).join(".cache/huggingface/hub/models--Qwen--Qwen2-57B-A14B-Instruct/snapshots/50896d66b39f1425d63720541a66c7df13e053c0");
    let reference: serde_json::Value =
        serde_json::from_slice(include_bytes!("qwen2_57b_moe_reference.json")).unwrap();
    assert_eq!(
        reference["revision"],
        "50896d66b39f1425d63720541a66c7df13e053c0"
    );
    let config =
        DecoderConfig::from_json_bytes(&std::fs::read(source.join("config.json")).unwrap())
            .unwrap();
    let reader = CheckpointReader::open(&source, &config).unwrap();
    let full_payload = reader.f32_weight_payload_bytes().unwrap();
    assert!(
        full_payload > 200_000_000_000,
        "actual checkpoint tensor shapes, not a synthetic size claim"
    );
    drop(reader);
    let folder = work.join("qwen2-moe");
    std::fs::create_dir_all(&folder).unwrap();
    let mut store = muniment::RedbBackend::open(folder.join("model.redb")).unwrap();
    let template =
        PromptTemplate::from_bytes(include_bytes!("../templates/qwen2-instruct-v1.json")).unwrap();
    let identity = serde_json::json!({"artifacts":reference["artifacts"],"template":template,"architecture":"qwen2_moe","expert_index_version":1});
    let cached = std::fs::read(folder.join("row-cache.json"))
        .ok()
        .map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).unwrap());
    let row: ManifestId = if let Some(cached) = cached.filter(|cache| cache["inputs"] == identity) {
        serde_json::from_value(cached["row"].clone()).unwrap()
    } else {
        for (file, hash) in reference["artifacts"].as_object().unwrap() {
            assert!(!file.contains(['/', '\\', ':']));
            eprintln!("verifying {file}");
            assert_eq!(digest(&source.join(file)), hash.as_str().unwrap());
        }
        let start = Instant::now();
        let options = StreamingOptions {
            query_chunk: 8,
            max_layer_bytes: 1024 * 1024 * 1024,
            max_shared_bytes: 3 * 1024 * 1024 * 1024,
        };
        eprintln!("importing independently addressed core layers and experts");
        let model = pollster::block_on(import_layered_model(
            &mut store,
            &source,
            "Qwen/Qwen2-57B-A14B-Instruct",
            "Apache-2.0",
            options,
        ))
        .unwrap();
        let template = save(&mut store, &template);
        let row = save(&mut store, &InferenceModelRow { model, template });
        std::fs::write(folder.join("row-cache.json"),serde_json::to_vec_pretty(&serde_json::json!({"inputs":identity,"row":row,"import_seconds":start.elapsed().as_secs_f64()})).unwrap()).unwrap();
        row
    };
    let resolved = pollster::block_on(resolve_library_session(&mut store, row)).unwrap();
    let store = Arc::new(Mutex::new(store));
    let device = DecoderDevice::wgpu(DecoderGpuKind::DiscreteGpu(0));
    let adapter = esp::infer::decoder::probe_gpu_adapter(DecoderGpuKind::DiscreteGpu(0)).unwrap();
    assert!(
        adapter.matched_requested_class,
        "requested discrete GPU: {adapter}"
    );
    let load = || {
        let store = store.clone();
        let source = resolved
            .artifact_source(move |artifact, limit| {
                pollster::block_on(artifact.read(&mut *store.lock().unwrap(), limit))
            })
            .unwrap();
        resolved
            .load_with_layer_source(&device, "burn-wgpu", 16, Some(Box::new(source)))
            .unwrap()
    };
    let provider = load();
    let shared = provider.bound().provider().shared_matrix_layout().unwrap();
    assert!(shared.embedding_shards > 1 && shared.output_shards > 1);
    assert!(shared.largest_tensor_bytes <= 1024 * 1024 * 1024);
    let mut results = Vec::new();
    for expected in reference["rows"].as_array().unwrap() {
        let request = GenerationRequest {
            prompt: expected["question"].as_str().unwrap().into(),
            max_tokens: 16,
            ..Default::default()
        };
        let before = provider.bound().provider().streaming_stats().unwrap();
        let start = Instant::now();
        let mut events = Vec::new();
        let mut last_reads = before.expert_reads;
        let result=provider.generate_checkpointed(&request,None,&mut |checkpoint|{
            let stats=provider.bound().provider().streaming_stats().unwrap();
            if checkpoint.token_ids.len()>1 && !checkpoint.complete {
                assert_eq!(stats.expert_reads-last_reads,(config.num_hidden_layers*config.moe.as_ref().unwrap().num_experts_per_tok) as u64);
            }
            if !checkpoint.token_ids.is_empty() {events.push(serde_json::json!({"at_token":checkpoint.token_ids.len(),"expert_reads":stats.expert_reads,"seconds":start.elapsed().as_secs_f64()}));}
            last_reads=stats.expert_reads;
            Ok(())
        },&mut ||false).unwrap();
        assert!(result.complete);
        assert!(!result.token_ids.is_empty());
        assert_eq!(serde_json::json!(result.prompt_ids), expected["prompt_ids"]);
        assert_eq!(
            serde_json::json!(result.token_ids),
            expected["token_ids"],
            "native f32 and independent Transformers reference"
        );
        let stats = provider.bound().provider().streaming_stats().unwrap();
        assert!(stats.largest_encoded_expert < 1024 * 1024 * 1024);
        results.push(serde_json::json!({"question":request.prompt,"token_ids":result.token_ids,"text":result.text,
            "seconds":start.elapsed().as_secs_f64(),"before":before,"after":stats,"token_events":events}));
        eprintln!("completed {}: {}", request.prompt, result.text);
    }
    let request = GenerationRequest {
        prompt: reference["rows"][0]["question"].as_str().unwrap().into(),
        max_tokens: 16,
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
    let before_restart_reads = provider.bound().provider().streaming_stats();
    let session = provider.session_identity();
    drop(provider);
    device.sync().unwrap();
    let provider = load();
    assert_eq!(provider.session_identity(), session);
    let resumed = provider
        .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
        .unwrap();
    assert_eq!(
        serde_json::json!(resumed.token_ids),
        reference["rows"][0]["token_ids"]
    );
    let receipt = serde_json::json!({"model":reference["model_id"],"revision":reference["revision"],"reference_dtype":reference["dtype"],
        "native_dtype":"f32","f32_weight_payload_bytes":full_payload,"row":row,"session":provider.session_identity(),"shared_matrix_layout":shared,"adapter":adapter,
        "results":results,"resume":{"at_token":3,"token_ids":resumed.token_ids,"serialized_checkpoint_and_reloaded_provider":true,"before_restart_reads":before_restart_reads},"final_reads":provider.bound().provider().streaming_stats(),
        "scope":"One physical computer/GPU; library-backed core and selected-expert streaming; no installed resident or physical fleet claim"});
    std::fs::write(
        work.join("qwen2-moe-result.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    drop(provider);
    device.sync().unwrap();
}
