// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The trainer forcing fixture (distillery v0 plan §9).
//!
//! One local, deterministic ranking receipt that materializes a canonical
//! `TrainingCorpus`, derives real PEFT LoRA adapter weights from the corpus's
//! training partition alone, publishes the adapter manifest with that corpus
//! as its provenance, and writes an `EvalReport` in which the adapter strictly
//! beats the unchanged baseline on the same fixed held-out cases.
//!
//! The receipt fixes the first trainer resource input/output shape: inputs
//! are the base-model manifest, the tokenizer blob, and the corpus's training
//! partition plus explicit hyperparameters; outputs are the adapter
//! weight/config blobs, the adapter manifest, and the evaluation report. The
//! training itself is `esp::infer::decoder::train` — the one shared trainer
//! implementation — and both evaluated sessions are built by the real
//! `PeftLoraAdapterLoader` from bytes resolved back out of the store. ESP
//! owns the tensor execution, Eidetic owns the artifacts; no Mesh job, lease,
//! checkpoint, or Distillery authority is involved.

#![cfg(feature = "decoder-lora")]

use burn::tensor::Device;
use eidetic::models::{EvalMetric, EvalReport, OpaqueBlob, TrainingCorpus};
use eidetic::typed::{load_typed, save_typed};
use eidetic::{
    AdapterRuntimeCompat, Hash, ManifestId, ModelAdapterManifest, ModelLibrary, NoFetcher,
    PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope,
};
use esp::infer::decoder::{
    LoraTrainerSettings, PEFT_LORA_NDARRAY_LOADER, PeftLoraAdapterLoader,
    TRAINED_ADAPTER_FORMAT_VERSION, TrainingCase, expected_token_rank, ranking_tally,
    train_peft_lora,
};
use esp::infer::{AdapterArtifact, AdapterLoader, AdapterSelection, ModelSession};
use muniment::MemoryBackend;
use safetensors::tensor::{Dtype, TensorView};

// ── Fixture identity ────────────────────────────────────────────────────────

const MODEL_ID: &str = "fixture/trainer-forcing";
const PROMPT_TEMPLATE: &[u8] = b"{{ prompt }}";
const ADAPTER_NAME: &str = "trainer-forcing-fixture";

/// The deterministic task family: every prompt ends with the trigger token
/// and the expected next token is fixed, while the held-out prefixes are
/// disjoint from the training prefixes. The receipt's point is the artifact
/// dataflow and a strict held-out improvement, not benchmark difficulty.
const TRIGGER: &str = "t29";
const EXPECTED: &str = "t7";
const TRAIN_PREFIXES: [&str; 6] = [
    "t3 t11 t5",
    "t18 t2 t26",
    "t9 t14 t1",
    "t22 t6 t13",
    "t4 t27 t10",
    "t15 t8 t21",
];
const EVAL_PREFIXES: [&str; 6] = [
    "t12 t25 t3",
    "t7 t19 t30",
    "t24 t1 t16",
    "t5 t28 t9",
    "t17 t20 t2",
    "t31 t10 t23",
];

fn trainer_settings() -> LoraTrainerSettings {
    LoraTrainerSettings {
        rank: 1,
        alpha: 8.0,
        target_module: "v_proj".into(),
        steps: 40,
        initial_step_length: 1.0,
        minimum_step_length: 1.0e-4,
        epsilon: 0.02,
    }
}

// ── Tiny synthetic base model (the eidetic_corridor artifact shape) ─────────

const CONFIG_JSON: &str = r#"{
    "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "max_position_embeddings": 16,
    "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

fn tokenizer_json() -> String {
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

fn det_vec(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + salt * 7919) as f32) * 0.618_034).sin() * 0.05)
        .collect()
}

fn base_weights() -> Vec<u8> {
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

fn case_blob(prefix: &str) -> OpaqueBlob {
    let case = TrainingCase {
        prompt: format!("{prefix} {TRIGGER}"),
        expected_token: EXPECTED.to_string(),
    };
    OpaqueBlob(serde_json::to_vec(&case).unwrap())
}

fn save_cases(store: &mut MemoryBackend, prefixes: &[&str]) -> Vec<ManifestId> {
    let mut ids: Vec<ManifestId> = prefixes
        .iter()
        .map(|prefix| {
            pollster::block_on(save_typed(
                store,
                &case_blob(prefix),
                vec![],
                PrivacyClass::LocalOnly,
                ProvenanceRecord::self_imported(ADAPTER_NAME),
                TrustEnvelope::self_asserted(),
                Timestamp(0),
            ))
            .expect("save case codicil")
        })
        .collect();
    ids.sort_by_key(ToString::to_string);
    ids
}

fn load_cases(store: &mut MemoryBackend, ids: &[ManifestId]) -> Vec<TrainingCase> {
    ids.iter()
        .map(|id| {
            let blob = pollster::block_on(load_typed::<OpaqueBlob>(store, &mut NoFetcher, *id))
                .expect("load case codicil")
                .expect("case codicil present");
            serde_json::from_slice(&blob.0).expect("case JSON")
        })
        .collect()
}

// ── The receipt ─────────────────────────────────────────────────────────────

#[test]
fn adapter_trained_from_corpus_beats_baseline_on_held_out_ranking() {
    let device = Device::ndarray();
    let mut store = MemoryBackend::default();
    let provenance = || ProvenanceRecord::self_imported(ADAPTER_NAME);

    // 1. Base model triple through the eidetic model corridor.
    let tokenizer_bytes = tokenizer_json();
    let base_model_ref = pollster::block_on(ModelLibrary::save_model_with_components(
        &mut store,
        MODEL_ID,
        "llama",
        "MIT",
        serde_json::from_str(CONFIG_JSON).unwrap(),
        &base_weights(),
        tokenizer_bytes.as_bytes(),
        Vec::new(),
        Vec::new(),
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(0),
    ))
    .expect("save base model");
    let resolved = pollster::block_on(ModelLibrary::resolve_components(
        &mut store,
        &mut NoFetcher,
        base_model_ref,
    ))
    .expect("resolve base model")
    .expect("base model present");
    let tokenizer_ref = resolved.manifest.tokenizer_blob;

    // 2. Canonical corpus: disjoint, strictly ordered source partitions.
    let corpus = TrainingCorpus {
        training_source_codicils: save_cases(&mut store, &TRAIN_PREFIXES),
        evaluation_source_codicils: save_cases(&mut store, &EVAL_PREFIXES),
    };
    corpus.validate().expect("corpus valid");
    let corpus_ref = pollster::block_on(save_typed(
        &mut store,
        &corpus,
        vec![],
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(1),
    ))
    .expect("save corpus");

    // 3. The shared deterministic trainer over the training partition only.
    let settings = trainer_settings();
    let train_cases = load_cases(&mut store, &corpus.training_source_codicils);
    let trained = train_peft_lora(
        &resolved.components.config_bytes,
        &resolved.components.tokenizer_bytes,
        &resolved.components.weight_bytes,
        MODEL_ID,
        &train_cases,
        &settings,
        &device,
    )
    .expect("train adapter");
    assert!(
        trained.trained_loss < trained.initial_loss,
        "training must reduce the training loss: {} -> {}",
        trained.initial_loss,
        trained.trained_loss
    );

    // 4. Publish the adapter: the trained blobs and the manifest whose
    //    provenance names the corpus.
    let save_blob = |store: &mut MemoryBackend, bytes: &[u8]| {
        pollster::block_on(save_typed(
            store,
            &OpaqueBlob(bytes.to_vec()),
            vec![],
            PrivacyClass::LocalOnly,
            provenance(),
            TrustEnvelope::self_asserted(),
            Timestamp(2),
        ))
        .expect("save adapter blob")
    };
    let adapter_blob = save_blob(&mut store, &trained.adapter_safetensors);
    let adapter_config_blob = save_blob(&mut store, &trained.adapter_config_json);
    let manifest = ModelAdapterManifest {
        name: ADAPTER_NAME.into(),
        base_model_ref,
        adapter_blob,
        adapter_config_blob,
        adapter_format: "peft-lora".into(),
        adapter_format_version: TRAINED_ADAPTER_FORMAT_VERSION.into(),
        runtime_compat: AdapterRuntimeCompat {
            minimum_capabilities: vec!["peft-lora".into()],
            known_loaders: vec![PEFT_LORA_NDARRAY_LOADER.into()],
            converter_lineage: vec![],
        },
        rank: settings.rank,
        alpha: settings.alpha,
        target_modules: vec![settings.target_module.clone()],
        tokenizer_ref,
        prompt_template_hash: Hash::of(PROMPT_TEMPLATE),
        quantization_assumption: None,
        training_corpus_root: Some(corpus_ref),
        training_method: serde_json::json!({
            "trainer": "esp-trainer-v0",
            "objective": "next-token cross-entropy",
            "settings": serde_json::to_value(&settings).unwrap(),
            "inputs": {
                "base_model_ref": base_model_ref.to_string(),
                "tokenizer_ref": tokenizer_ref.to_string(),
                "corpus_partition": "training_source_codicils",
            },
            "outputs": ["adapter_blob", "adapter_config_blob", "eval_report"],
        }),
        eval_results: None,
    };
    let adapter_ref = pollster::block_on(save_typed(
        &mut store,
        &manifest,
        vec![],
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(3),
    ))
    .expect("save adapter manifest");

    // 5. Evaluate baseline and adapter through the real session loader, from
    //    bytes resolved back out of the store.
    let stored_manifest = pollster::block_on(load_typed::<ModelAdapterManifest>(
        &mut store,
        &mut NoFetcher,
        adapter_ref,
    ))
    .expect("load adapter manifest")
    .expect("adapter manifest present");
    let load_blob = |store: &mut MemoryBackend, id: ManifestId| {
        pollster::block_on(load_typed::<OpaqueBlob>(store, &mut NoFetcher, id))
            .expect("load adapter blob")
            .expect("adapter blob present")
            .0
    };
    let stored_weights = load_blob(&mut store, stored_manifest.adapter_blob);
    let stored_config = load_blob(&mut store, stored_manifest.adapter_config_blob);
    let loader = PeftLoraAdapterLoader::new(
        base_model_ref,
        tokenizer_ref,
        &resolved.components.config_bytes,
        &resolved.components.tokenizer_bytes,
        &resolved.components.weight_bytes,
        MODEL_ID,
        PEFT_LORA_NDARRAY_LOADER,
        &device,
    );
    let session = |adapters: Vec<AdapterSelection>| ModelSession {
        base_model_ref,
        model_id: MODEL_ID.into(),
        tokenizer_ref,
        prompt_template_hash: Hash::of(PROMPT_TEMPLATE),
        quantization: None,
        loader: PEFT_LORA_NDARRAY_LOADER.into(),
        adapters,
    };
    let baseline_session = loader
        .load_session(session(vec![]), &[])
        .expect("baseline session");
    let adapted = || {
        loader
            .load_session(
                session(vec![AdapterSelection {
                    manifest_ref: adapter_ref,
                    scale: 1.0,
                }]),
                &[AdapterArtifact {
                    manifest_ref: adapter_ref,
                    manifest: &stored_manifest,
                    config_bytes: &stored_config,
                    weight_bytes: &stored_weights,
                }],
            )
            .expect("adapted session")
    };
    let adapted_session = adapted();

    let eval_cases = load_cases(&mut store, &corpus.evaluation_source_codicils);
    let expected_id = 7usize;
    for (label, session) in [
        ("baseline", &baseline_session),
        ("adapter", &adapted_session),
    ] {
        let ranks: Vec<usize> = eval_cases
            .iter()
            .map(|case| {
                let logits = session.provider().next_token_logits(&case.prompt).unwrap();
                expected_token_rank(&logits, expected_id)
            })
            .collect();
        println!("{label} held-out expected-token ranks: {ranks:?}");
    }
    // Ranking cutoff 3: the rank-1 value-projection adapter tilts the tiny
    // model's near-uniform logits decisively into the top 3 (baseline ranks
    // sit at 7-8) without demanding a fragile single-case top-1 margin.
    let limit = 3u32;
    let baseline = ranking_tally(
        baseline_session.provider(),
        &resolved.components.tokenizer_bytes,
        &eval_cases,
        limit,
    )
    .expect("baseline tally");
    let adapter = ranking_tally(
        adapted_session.provider(),
        &resolved.components.tokenizer_bytes,
        &eval_cases,
        limit,
    )
    .expect("adapter tally");
    assert_eq!(baseline.total, eval_cases.len() as u64);
    assert!(
        baseline.passed < baseline.total,
        "receipt is trivial: the baseline already passes every held-out case ({baseline:?})"
    );
    assert!(
        adapter.passed > baseline.passed,
        "adapter must strictly beat the baseline: baseline {baseline:?}, adapter {adapter:?}"
    );

    // Determinism: a freshly loaded adapted session reproduces the tally.
    assert_eq!(
        ranking_tally(
            adapted().provider(),
            &resolved.components.tokenizer_bytes,
            &eval_cases,
            limit,
        )
        .expect("repeat tally"),
        adapter,
        "adapter tally must be reproducible from a fresh session"
    );

    // 6. The immutable receipt, round-tripped and provenance-checked.
    let report = EvalReport {
        base_model_ref,
        adapter_ref,
        corpus_ref,
        metric: EvalMetric::RankingAt { limit },
        baseline,
        adapter,
    };
    let report_ref = pollster::block_on(save_typed(
        &mut store,
        &report,
        vec![],
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(4),
    ))
    .expect("save eval report");
    let stored_report = pollster::block_on(load_typed::<EvalReport>(
        &mut store,
        &mut NoFetcher,
        report_ref,
    ))
    .expect("load eval report")
    .expect("eval report present");
    assert_eq!(stored_report, report);
    assert!(
        stored_report
            .adapter_beats_baseline()
            .expect("comparable report"),
        "stored receipt must show the strict improvement"
    );
    stored_report
        .validate_for_adapter(adapter_ref, &stored_manifest)
        .expect("receipt provenance links match the adapter manifest");

    println!(
        "trainer forcing receipt: loss {:.4} -> {:.4}, baseline {}/{} vs adapter {}/{} at ranking@{limit}",
        trained.initial_loss,
        trained.trained_loss,
        report.baseline.passed,
        report.baseline.total,
        report.adapter.passed,
        report.adapter.total,
    );
}
