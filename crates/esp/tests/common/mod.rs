// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The trainer forcing fixture, shared by the v0 and v1 receipts.
//!
//! One corpus, one synthetic base model, one tokenizer. Both receipts must
//! see the *same* task for their step counts and held-out tallies to be
//! comparable, so the fixture lives here rather than being copied into each
//! receipt where the two could drift apart unnoticed.

// Each receipt uses a subset of this fixture; the module is compiled into
// every test binary that names it.
#![allow(dead_code)]

use eidetic::models::{OpaqueBlob, TrainingCorpus};
use eidetic::typed::{load_typed, save_typed};
use eidetic::{ManifestId, NoFetcher, PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope};
use esp::infer::decoder::TrainingCase;
use muniment::MemoryBackend;
use safetensors::tensor::{Dtype, TensorView};

// ── Fixture identity ────────────────────────────────────────────────────────

pub const MODEL_ID: &str = "fixture/trainer-forcing";
pub const PROMPT_TEMPLATE: &[u8] = b"{{ prompt }}";

/// The deterministic task family: every prompt ends with the trigger token
/// and the expected next token is fixed, while the held-out prefixes are
/// disjoint from the training prefixes. The receipt's point is the artifact
/// dataflow and a strict held-out improvement, not benchmark difficulty.
pub const TRIGGER: &str = "t29";
pub const EXPECTED: &str = "t7";
/// The vocabulary id of [`EXPECTED`], for direct rank reads.
pub const EXPECTED_ID: usize = 7;
pub const TRAIN_PREFIXES: [&str; 6] = [
    "t3 t11 t5",
    "t18 t2 t26",
    "t9 t14 t1",
    "t22 t6 t13",
    "t4 t27 t10",
    "t15 t8 t21",
];
pub const EVAL_PREFIXES: [&str; 6] = [
    "t12 t25 t3",
    "t7 t19 t30",
    "t24 t1 t16",
    "t5 t28 t9",
    "t17 t20 t2",
    "t31 t10 t23",
];

// ── Tiny synthetic base model (the eidetic_corridor artifact shape) ─────────

pub const CONFIG_JSON: &str = r#"{
    "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "max_position_embeddings": 16,
    "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

pub fn tokenizer_json() -> String {
    let vocab: Vec<String> = (0..32).map(|i| format!("\"t{i}\": {i}")).collect();
    format!(
        r#"{{
            "version": "1.0",
            "pre_tokenizer": {{ "type": "Whitespace" }},
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }}
        }}"#,
        vocab.join(", ")
    )
}

pub fn det_vec(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + salt * 7919) as f32) * 0.618_034).sin() * 0.05)
        .collect()
}

pub fn base_weights() -> Vec<u8> {
    let (h, kv, inter, vocab) = (8usize, 4usize, 16usize, 32usize);
    let mut table: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, salt: usize| {
        let n: usize = shape.iter().product();
        table.push((name, shape, det_vec(n, salt)));
    };
    push("model.embed_tokens.weight".into(), vec![vocab, h], 7);
    for i in 0..2usize {
        let p = format!("model.layers.{i}");
        let s = 100 * (i + 1);
        push(format!("{p}.input_layernorm.weight"), vec![h], s);
        push(format!("{p}.self_attn.q_proj.weight"), vec![h, h], s + 1);
        push(format!("{p}.self_attn.k_proj.weight"), vec![kv, h], s + 2);
        push(format!("{p}.self_attn.v_proj.weight"), vec![kv, h], s + 3);
        push(format!("{p}.self_attn.o_proj.weight"), vec![h, h], s + 4);
        push(
            format!("{p}.post_attention_layernorm.weight"),
            vec![h],
            s + 8,
        );
        push(format!("{p}.mlp.gate_proj.weight"), vec![inter, h], s + 5);
        push(format!("{p}.mlp.up_proj.weight"), vec![inter, h], s + 6);
        push(format!("{p}.mlp.down_proj.weight"), vec![h, inter], s + 7);
    }
    push("model.norm.weight".into(), vec![h], 9);
    push("lm_head.weight".into(), vec![vocab, h], 8);

    let buffers: Vec<(String, Vec<usize>, Vec<u8>)> = table
        .iter()
        .map(|(name, shape, values)| {
            (
                name.clone(),
                shape.clone(),
                values.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views: Vec<(&str, TensorView<'_>)> = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize(views, &None).unwrap()
}

// ── Corpus cases ────────────────────────────────────────────────────────────

pub fn training_case(prefix: &str) -> TrainingCase {
    TrainingCase {
        prompt: format!("{prefix} {TRIGGER}"),
        expected_token: EXPECTED.to_string(),
    }
}

pub fn case_blob(prefix: &str) -> OpaqueBlob {
    OpaqueBlob(serde_json::to_vec(&training_case(prefix)).unwrap())
}

pub fn save_cases(store: &mut MemoryBackend, name: &str, prefixes: &[&str]) -> Vec<ManifestId> {
    let mut ids: Vec<ManifestId> = prefixes
        .iter()
        .map(|prefix| {
            pollster::block_on(save_typed(
                store,
                &case_blob(prefix),
                vec![],
                PrivacyClass::LocalOnly,
                ProvenanceRecord::self_imported(name),
                TrustEnvelope::self_asserted(),
                Timestamp(0),
            ))
            .expect("save case codicil")
        })
        .collect();
    ids.sort_by_key(ToString::to_string);
    ids
}

pub fn load_cases(store: &mut MemoryBackend, ids: &[ManifestId]) -> Vec<TrainingCase> {
    ids.iter()
        .map(|id| {
            let blob = pollster::block_on(load_typed::<OpaqueBlob>(store, &mut NoFetcher, *id))
                .expect("load case codicil")
                .expect("case codicil present");
            serde_json::from_slice(&blob.0).expect("case JSON")
        })
        .collect()
}

/// The canonical corpus: disjoint, strictly ordered source partitions.
pub fn corpus(store: &mut MemoryBackend, name: &str) -> TrainingCorpus {
    let corpus = TrainingCorpus {
        training_source_codicils: save_cases(store, name, &TRAIN_PREFIXES),
        evaluation_source_codicils: save_cases(store, name, &EVAL_PREFIXES),
    };
    corpus.validate().expect("corpus valid");
    corpus
}
