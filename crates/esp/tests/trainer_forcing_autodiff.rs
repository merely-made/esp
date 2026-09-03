// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The v1 trainer forcing receipt (autodiff LoRA trainer plan, Phase 1).
//!
//! The v0 receipt (`trainer_forcing.rs`) with one substitution: the
//! finite-difference trainer becomes the autodiff trainer. Everything else —
//! the corpus, the synthetic base model, the eidetic model corridor, the
//! adapter manifest, the `PeftLoraAdapterLoader`, the `EvalReport` and its
//! provenance check — is the same code on the same fixture, which is what
//! makes the two receipts comparable.
//!
//! What the receipt has to show beyond v0's: the held-out strict improvement
//! at `RankingAt { limit: 3 }` reached in **fewer steps than v0's 40**, and
//! the adapter tally reproducing from a fresh session. The step count is the
//! measurement — a real gradient should not need forty line-searched
//! descents to move a rank-1 delta.

#![cfg(feature = "decoder-autodiff")]

use burn::tensor::Device;
use eidetic::models::{EvalMetric, EvalReport, OpaqueBlob};
use eidetic::typed::{load_typed, save_typed};
use eidetic::{
    AdapterRuntimeCompat, Hash, ManifestId, ModelAdapterManifest, ModelLibrary, NoFetcher,
    PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope,
};
use esp::infer::decoder::{
    AutodiffLoraSettings, PEFT_LORA_NDARRAY_LOADER, PeftLoraAdapterLoader,
    TRAINED_ADAPTER_FORMAT_VERSION_AUTODIFF, TRAINED_PEFT_VERSION_AUTODIFF, expected_token_rank,
    ranking_tally, train_peft_lora_autodiff,
};
use esp::infer::{AdapterArtifact, AdapterLoader, AdapterSelection, ModelSession};
use muniment::MemoryBackend;

mod common;
use common::{
    CONFIG_JSON, EXPECTED_ID, MODEL_ID, PROMPT_TEMPLATE, base_weights, corpus as fixture_corpus,
    load_cases, tokenizer_json,
};

const ADAPTER_NAME: &str = "trainer-forcing-autodiff-fixture";

/// v0's step budget, the number this receipt has to come in under.
const V0_STEPS: u32 = 40;

/// Twelve Adam steps at a modest learning rate.
///
/// Not tuned to a maximum. On this fixture the held-out tally is a coarse,
/// non-monotone function of the step count — swept at this learning rate it
/// reads 3, 3, 2, 6, 3, 3, 4 out of 6 at 8, 12, 16, 20, 24, 30 and 36 steps —
/// because a rank-1 delta on a near-uniform eight-hidden model reorders the
/// top of the logit row in jumps. Every one of those counts beats the
/// baseline's 0/6, so the receipt asserts the strict improvement and nothing
/// finer; picking the count that happened to read 6/6 would be reading noise
/// as a result. Twelve is chosen for sitting well inside that stable region
/// and well under v0's forty.
fn trainer_settings() -> AutodiffLoraSettings {
    AutodiffLoraSettings {
        rank: 1,
        alpha: 8.0,
        target_modules: vec!["v_proj".into()],
        steps: 12,
        learning_rate: 0.2,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1.0e-8,
        weight_decay: 0.0,
    }
}

#[test]
fn autodiff_adapter_beats_baseline_on_held_out_ranking_in_fewer_steps() {
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

    // 2. The same canonical corpus the v0 receipt trains on.
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

    // 3. The autodiff trainer over the training partition only.
    let settings = trainer_settings();
    assert!(
        settings.steps < V0_STEPS,
        "the v1 receipt must cost fewer steps than v0's {V0_STEPS}"
    );
    let train_cases = load_cases(&mut store, &corpus.training_source_codicils);
    let started = std::time::Instant::now();
    let trained = train_peft_lora_autodiff(
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

    // 4. Publish the adapter under the v1 version strings.
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
        adapter_format_version: TRAINED_ADAPTER_FORMAT_VERSION_AUTODIFF.into(),
        runtime_compat: AdapterRuntimeCompat {
            minimum_capabilities: vec!["peft-lora".into()],
            known_loaders: vec![PEFT_LORA_NDARRAY_LOADER.into()],
            converter_lineage: vec![],
        },
        rank: settings.rank,
        alpha: settings.alpha,
        target_modules: settings.target_modules.clone(),
        tokenizer_ref,
        prompt_template_hash: Hash::of(PROMPT_TEMPLATE),
        quantization_assumption: None,
        training_corpus_root: Some(corpus_ref),
        training_method: serde_json::json!({
            "trainer": TRAINED_PEFT_VERSION_AUTODIFF,
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
    for (label, session) in [
        ("baseline", &baseline_session),
        ("adapter", &adapted_session),
    ] {
        let ranks: Vec<usize> = eval_cases
            .iter()
            .map(|case| {
                let logits = session.provider().next_token_logits(&case.prompt).unwrap();
                expected_token_rank(&logits, EXPECTED_ID)
            })
            .collect();
        println!("{label} held-out expected-token ranks: {ranks:?}");
    }
    // The v0 receipt's cutoff, unchanged: the two receipts are only
    // comparable if they are scored the same way.
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
        "autodiff trainer forcing receipt: loss {:.4} -> {:.4} in {} steps ({elapsed:?}), \
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
