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
use eidetic::models::{EvalMetric, EvalReport, OpaqueBlob};
use eidetic::typed::{load_typed, save_typed};
use eidetic::{
    AdapterRuntimeCompat, Hash, ManifestId, ModelAdapterManifest, ModelLibrary, NoFetcher,
    PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope,
};
use esp::infer::decoder::{
    LoraTrainerSettings, PEFT_LORA_NDARRAY_LOADER, PeftLoraAdapterLoader,
    TRAINED_ADAPTER_FORMAT_VERSION, expected_token_rank, ranking_tally, train_peft_lora,
};
use esp::infer::{AdapterArtifact, AdapterLoader, AdapterSelection, ModelSession};
use muniment::MemoryBackend;

mod common;
use common::{
    CONFIG_JSON, EXPECTED_ID, MODEL_ID, PROMPT_TEMPLATE, base_weights, corpus as fixture_corpus,
    load_cases, tokenizer_json,
};

const ADAPTER_NAME: &str = "trainer-forcing-fixture";

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
    let corpus = fixture_corpus(&mut store, ADAPTER_NAME);
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
    let started = std::time::Instant::now();
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
    let elapsed = started.elapsed();
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
    let expected_id = EXPECTED_ID;
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
        "trainer forcing receipt: loss {:.4} -> {:.4} in {} steps ({elapsed:?}), \
         baseline {}/{} vs adapter {}/{} at ranking@{limit}",
        trained.initial_loss,
        trained.trained_loss,
        settings.steps,
        report.baseline.passed,
        report.baseline.total,
        report.adapter.passed,
        report.adapter.total,
    );
}
