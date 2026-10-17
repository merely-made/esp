// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;
use burn::tensor::{Device, Int, Tensor, TensorData};
use stage::{DecoderStage, LayerRange, StageExecutor};
use std::path::PathBuf;
use std::sync::Arc;
use streaming::StreamingOptions;

#[derive(Debug)]
struct Chain(Vec<DecoderStage>);
impl StageExecutor for Chain {
    fn returns_logits(&self) -> bool {
        true
    }
    fn forward(
        &self,
        mut hidden: Tensor<3>,
        start: usize,
    ) -> Result<Tensor<3>, crate::infer::InferError> {
        for stage in &self.0 {
            hidden = stage.forward(hidden, start)?;
        }
        Ok(hidden)
    }
}

fn check(name: &str) {
    check_on(name, Device::ndarray());
}

fn check_on(name: &str, device: Device) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/support/architectures");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
    let row = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == name)
        .unwrap();
    let folder = root.join(name);
    let config =
        DecoderConfig::from_json_bytes(&std::fs::read(folder.join("config.json")).unwrap())
            .unwrap();
    let weights = std::fs::read(folder.join("model.safetensors")).unwrap();
    use sha2::{Digest, Sha256};
    let digest: String = Sha256::digest(&weights)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, row["weights_sha256"].as_str().unwrap());
    let resident = loader::load_decoder_from_bytes(&config, &weights, &device).unwrap();
    let options = StreamingOptions {
        query_chunk: 2,
        max_layer_bytes: 1024 * 1024,
        max_shared_bytes: 1024 * 1024,
    };
    let mut reader = checkpoint::CheckpointReader::open(&folder, &config).unwrap();
    let shared = reader.shared_bytes(options.max_shared_bytes).unwrap();
    let shared_tensors = safetensors::SafeTensors::deserialize(&shared).unwrap();
    let artifacts = Arc::new(
        (0..config.num_hidden_layers)
            .map(|index| reader.layer_bytes(index, options.max_layer_bytes).unwrap())
            .collect::<Vec<_>>(),
    );
    let experts = Arc::new(
        (0..config.num_hidden_layers)
            .map(|layer| {
                if config.layer_is_moe(layer) {
                    (0..config.moe.as_ref().unwrap().num_experts)
                        .map(|expert| {
                            reader
                                .expert_bytes(layer, expert, options.max_layer_bytes)
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>(),
    );
    let make_source = || {
        let artifacts = artifacts.clone();
        let experts = experts.clone();
        streaming::with_expert_reader(
            move |index: usize, _limit: usize| Ok(artifacts[index].clone()),
            move |layer: usize, expert: usize, _limit: usize| Ok(experts[layer][expert].clone()),
        )
    };
    let source = make_source();
    let streamed = DecoderModel::from_layer_source(
        config.clone(),
        loader::load_decoder_shared_tensors(&config, &shared_tensors, &device).unwrap(),
        Box::new(source),
        options.clone(),
        &device,
    )
    .unwrap();
    let source = make_source();
    let first = DecoderStage::from_source(
        config.clone(),
        LayerRange { first: 0, end: 1 },
        &source,
        options.clone(),
        WeightFormat::F32,
        &device,
    )
    .unwrap();
    let shard_bound = config.hidden_size * 4 * 7;
    let last = DecoderStage::from_source(
        config.clone(),
        LayerRange { first: 1, end: 2 },
        &source,
        options.clone(),
        WeightFormat::F32,
        &device,
    )
    .unwrap()
    .with_output_shards(
        loader::load_shared_shards(&config, &shared_tensors, &device, shard_bound).unwrap(),
    )
    .unwrap();
    let staged = DecoderModel::from_layer_source(
        config.clone(),
        loader::load_decoder_shared_tensors(&config, &shared_tensors, &device).unwrap(),
        Box::new(source),
        options,
        &device,
    )
    .unwrap()
    .with_stages(Box::new(Chain(vec![first, last])))
    .unwrap();
    let sharded = DecoderModel::from_sharded_shared_source(
        config.clone(),
        loader::load_shared_shards(&config, &shared_tensors, &device, shard_bound).unwrap(),
        Box::new(make_source()),
        StreamingOptions {
            query_chunk: 2,
            max_layer_bytes: 1024 * 1024,
            max_shared_bytes: 1024 * 1024,
        },
        &device,
    )
    .unwrap();
    assert!(sharded.shared_matrix_layout().unwrap().largest_tensor_bytes <= shard_bound as u64);
    let ids: Vec<i32> = row["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    for (mode, model) in [
        ("resident", resident),
        ("streamed", streamed),
        ("staged", staged),
        ("shared-row-shards", sharded),
    ] {
        let mut cache = model.new_cache();
        let mut next =
            Tensor::<2, Int>::from_data(TensorData::new(ids.clone(), [1, ids.len()]), &device);
        let mut generated = Vec::new();
        let mut prior_expert_reads = 0;
        for step in 0..4 {
            let logits = model
                .try_forward_cached_last(next, &mut cache)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            if let Some(moe) = &config.moe
                && mode == "streamed"
            {
                let stats = model.streaming_stats().unwrap();
                if step > 0 {
                    assert_eq!(
                        stats.expert_reads - prior_expert_reads,
                        (moe.num_experts_per_tok
                            * (0..config.num_hidden_layers)
                                .filter(|i| config.layer_is_moe(*i))
                                .count()) as u64,
                        "single-token forward reads exactly its selected experts"
                    );
                }
                assert!(stats.largest_encoded_expert <= 1024 * 1024);
                prior_expert_reads = stats.expert_reads;
            }
            if step == 0 {
                let expected = row["last_logits"].as_array().unwrap();
                let max = logits
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (*a - b.as_f64().unwrap() as f32).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max < 2e-5,
                    "{name}/{mode}: independent logits max error {max}"
                );
            }
            let id = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as i32;
            generated.push(id);
            next = Tensor::<2, Int>::from_data([[id]], &device);
        }
        assert_eq!(
            serde_json::json!(generated),
            row["generated_ids"],
            "{name}/{mode}: cached tokens versus full-recompute Transformers"
        );
    }
}

#[test]
fn original_llama_reference() {
    check("llama");
}
#[test]
fn llama3_scaled_reference_past_original_context() {
    check("llama3");
}
#[test]
fn linear_scaled_reference() {
    check("linear");
}
#[test]
fn mistral_window_and_independent_head_reference() {
    check("mistral");
}
#[test]
fn qwen2_biases_and_mixed_windows_reference() {
    check("qwen2");
}
#[test]
fn qwen3_head_norms_and_mixed_windows_reference() {
    check("qwen3");
}
#[test]
fn qwen2_moe_selected_experts_and_shared_gate_reference() {
    check("qwen2_moe");
}
#[test]
fn qwen2_moe_normalized_routes_and_hybrid_layers_reference() {
    check("qwen2_moe_normalized");
}

#[cfg(feature = "decoder-wgpu")]
#[test]
#[ignore = "requires a native discrete GPU"]
fn gpu_architecture_streaming_and_stages_match_reference() {
    for name in [
        "llama",
        "llama3",
        "linear",
        "mistral",
        "qwen2",
        "qwen3",
        "qwen2_moe",
        "qwen2_moe_normalized",
    ] {
        check_on(name, Device::wgpu(DecoderGpuKind::DiscreteGpu(0)));
    }
}
