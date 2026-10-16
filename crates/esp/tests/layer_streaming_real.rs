// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(
    feature = "decoder",
    feature = "model-session",
    not(target_family = "wasm")
))]

use eidetic::typed::save_typed;
use eidetic::{PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope};
use esp::infer::decoder::streaming::StreamingOptions;
use esp::infer::decoder::{DecoderConfig, DecoderDevice};
use esp::infer::layer_library::import_layered_model;
use esp::infer::library::{InferenceModelRow, resolve_library_session};
use esp::infer::prompt::PromptTemplate;
use esp::infer::{GenerationRequest, InferenceProvider};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[test]
#[ignore = "requires pinned ESP_SMOLLM2_DIR and ESP_STREAMING_WORK_DIR"]
fn real_cpu_layer_streaming_matches_reference() {
    run(DecoderDevice::ndarray(), "burn-ndarray");
}

#[cfg(feature = "decoder-wgpu")]
#[test]
#[ignore = "requires pinned ESP_SMOLLM2_DIR, ESP_STREAMING_WORK_DIR and a discrete GPU"]
fn real_gpu_layer_streaming_matches_reference() {
    use esp::infer::decoder::{DecoderGpuKind, GpuDeviceType, probe_gpu_adapter};
    let kind = DecoderGpuKind::DiscreteGpu(0);
    let facts = probe_gpu_adapter(kind.clone()).unwrap();
    assert_eq!(facts.device_type, GpuDeviceType::Discrete);
    assert!(facts.matched_requested_class);
    eprintln!("streamed inference adapter: {facts}");
    run(DecoderDevice::wgpu(kind), "burn-wgpu");
}

fn verify(path: &Path, expected: &str) {
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut bytes = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut bytes).unwrap();
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    let actual: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(actual, expected, "exact {}", path.display());
}

fn run(device: DecoderDevice, loader: &str) {
    let directory = PathBuf::from(
        std::env::var_os("ESP_STREAMING_MODEL_DIR")
            .or_else(|| std::env::var_os("ESP_SMOLLM2_DIR"))
            .expect("ESP_STREAMING_MODEL_DIR or ESP_SMOLLM2_DIR"),
    );
    let work =
        PathBuf::from(std::env::var_os("ESP_STREAMING_WORK_DIR").expect("ESP_STREAMING_WORK_DIR"));
    std::fs::create_dir_all(&work).unwrap();
    let reference_bytes = std::env::var_os("ESP_STREAMING_REFERENCE")
        .map(|path| std::fs::read(path).unwrap())
        .unwrap_or_else(|| include_bytes!("smollm2_360m_reference.json").to_vec());
    let reference: serde_json::Value = serde_json::from_slice(&reference_bytes).unwrap();
    let max_tokens = reference["max_tokens"].as_u64().unwrap_or(16) as usize;
    for (file, hash) in reference["artifacts"].as_object().unwrap() {
        verify(&directory.join(file), hash.as_str().unwrap());
    }
    let mut store = muniment::RedbBackend::open(work.join("model-layered.redb")).unwrap();
    let options = StreamingOptions::default();
    let model = pollster::block_on(import_layered_model(
        &mut store,
        &directory,
        reference["model_id"].as_str().unwrap(),
        "Apache-2.0",
        options.clone(),
    ))
    .unwrap();
    let template_bytes = std::env::var_os("ESP_STREAMING_TEMPLATE")
        .map(|path| std::fs::read(path).unwrap())
        .unwrap_or_else(|| include_bytes!("../templates/smollm2-instruct-v1.json").to_vec());
    let template = PromptTemplate::from_bytes(&template_bytes).unwrap();
    let template = pollster::block_on(save_typed(
        &mut store,
        &template,
        vec![],
        PrivacyClass::LocalOnly,
        ProvenanceRecord::self_imported("local"),
        TrustEnvelope::self_asserted(),
        Timestamp::ZERO,
    ))
    .unwrap();
    let row = pollster::block_on(save_typed(
        &mut store,
        &InferenceModelRow { model, template },
        vec![],
        PrivacyClass::LocalOnly,
        ProvenanceRecord::self_imported("local"),
        TrustEnvelope::self_asserted(),
        Timestamp::ZERO,
    ))
    .unwrap();
    let resolved = pollster::block_on(resolve_library_session(&mut store, row)).unwrap();
    let artifacts = resolved.layer_artifacts().unwrap();
    let bytes_per_pass: u64 = artifacts.iter().map(|artifact| artifact.byte_size).sum();
    let store = Arc::new(Mutex::new(store));
    let source = move |index: usize, limit: usize| {
        pollster::block_on(artifacts[index].read(&mut *store.lock().unwrap(), limit))
    };
    let provider = resolved
        .load_with_layer_source(&device, loader, max_tokens, Some(Box::new(source)))
        .unwrap();
    let decoder = provider.bound().provider();
    let mut rows = Vec::new();
    for expected in reference["rows"].as_array().unwrap() {
        let request = GenerationRequest {
            prompt: expected["question"].as_str().unwrap().into(),
            max_tokens,
            ..Default::default()
        };
        let before = decoder.streaming_stats().unwrap();
        let start = Instant::now();
        let result = provider
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let after = decoder.streaming_stats().unwrap();
        assert_eq!(serde_json::json!(result.token_ids), expected["token_ids"]);
        assert_eq!(serde_json::json!(result.prompt_ids), expected["prompt_ids"]);
        assert_eq!(result.text, expected["text"].as_str().unwrap());
        rows.push(serde_json::json!({"question":request.prompt,"token_ids":result.token_ids,"text":result.text,
            "seconds":seconds,"seconds_per_token":seconds/result.token_ids.len() as f64,
            "layer_reads":after.layer_reads-before.layer_reads,
            "logical_encoded_bytes_read":after.encoded_bytes_read-before.encoded_bytes_read,
            "largest_encoded_layer":after.largest_encoded_layer}));
    }
    let request = GenerationRequest {
        prompt: reference["rows"][0]["question"].as_str().unwrap().into(),
        max_tokens,
        ..Default::default()
    };
    let boundary = std::cell::Cell::new(0usize);
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

    // Long prefill is a separate row when the model's declared window fits it.
    let config =
        DecoderConfig::from_json_bytes(&std::fs::read(directory.join("config.json")).unwrap())
            .unwrap();
    let long = if config.max_position_embeddings >= 8192 {
        let request = GenerationRequest {
            prompt: " the".repeat(4096),
            max_tokens: 1,
            ..Default::default()
        };
        let start = Instant::now();
        let result = provider
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        let seconds = start.elapsed().as_secs_f64();
        assert!(result.prompt_ids.len() >= 4096);
        assert!(result.complete);
        serde_json::json!({"prompt_tokens":result.prompt_ids.len(),"token_ids":result.token_ids,
            "seconds":seconds,"largest_attention_score_block_bytes":config.num_attention_heads * options.query_chunk * result.prompt_ids.len() * 4,
            "reference_compared":false})
    } else {
        serde_json::Value::Null
    };
    let receipt = serde_json::json!({
        "schema":"mere.resident-inference/layer-streaming-software/v1", "model":reference["model_id"],
        "revision":reference["revision"],"artifacts":reference["artifacts"],"row":row.to_string(),
        "session":provider.session_identity(),"loader":provider.capability().loader,"options":options,
        "bytes_per_layer_pass":bytes_per_pass,"rows":rows,
        "pause_resume":{"boundary":3,"token_ids":resumed.token_ids,"matched_reference":true},
        "long_prefill":long,
        "statistics":decoder.streaming_stats(),"device_capacity_measured_by_this_test":false,
        "limits":["logical artifact reads do not measure physical disk traffic", "device capacity and peak process memory require the separate host measurement"]
    });
    let encoded = serde_json::to_vec_pretty(&receipt).unwrap();
    std::fs::write(work.join(format!("{loader}-software.json")), &encoded).unwrap();
    println!("{}", String::from_utf8(encoded).unwrap());
}
