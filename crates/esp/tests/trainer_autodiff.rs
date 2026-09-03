// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The v1 autodiff trainer against the unchanged adapter loader.
//!
//! Three claims, each on the shared trainer forcing fixture: training reduces
//! the loss; what it produces is an ordinary PEFT adapter that survives
//! `apply_peft_lora`'s checks under the v1 version strings and moves the
//! model's logits; and the two constraints v1 lifts over v0 — per-case
//! expected tokens and several target projections — really work rather than
//! merely being accepted by validation.
//!
//! The loader is not modified for any of this. That is the point: the trainer
//! version is a value in `adapter_config.json`, not a change to the bytes.

#![cfg(feature = "decoder-autodiff")]

use burn::tensor::Device;
use eidetic::{AdapterRuntimeCompat, Hash, ManifestId, ModelAdapterManifest};
use esp::infer::decoder::{
    AutodiffLoraSettings, PEFT_LORA_NDARRAY_LOADER, PeftLoraAdapterLoader,
    TRAINED_ADAPTER_FORMAT_VERSION_AUTODIFF, TrainedLoraAdapter, TrainingCase,
    train_peft_lora_autodiff,
};
use esp::infer::{AdapterArtifact, AdapterLoader, AdapterSelection, ModelSession};

mod common;
use common::{
    CONFIG_JSON, EXPECTED, MODEL_ID, PROMPT_TEMPLATE, TRAIN_PREFIXES, TRIGGER, base_weights,
    tokenizer_json, training_case,
};

const ADAPTER_NAME: &str = "trainer-autodiff-fixture";

fn settings(target_modules: &[&str], steps: u32) -> AutodiffLoraSettings {
    AutodiffLoraSettings {
        rank: 1,
        alpha: 8.0,
        target_modules: target_modules.iter().map(|m| (*m).to_string()).collect(),
        steps,
        learning_rate: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1.0e-8,
        weight_decay: 0.0,
    }
}

fn train_cases() -> Vec<TrainingCase> {
    TRAIN_PREFIXES.iter().map(|p| training_case(p)).collect()
}

fn train(
    settings: &AutodiffLoraSettings,
    cases: &[TrainingCase],
    device: &Device,
) -> TrainedLoraAdapter {
    train_peft_lora_autodiff(
        CONFIG_JSON.as_bytes(),
        tokenizer_json().as_bytes(),
        &base_weights(),
        MODEL_ID,
        cases,
        settings,
        device,
    )
    .expect("train adapter")
}

/// The manifest the loader checks the adapter against. Built directly rather
/// than through the eidetic corridor: the corridor is the forcing receipt's
/// subject, and here the subject is the loader's admission checks.
fn manifest(settings: &AutodiffLoraSettings, trained: &TrainedLoraAdapter) -> ModelAdapterManifest {
    let base_model_ref = ManifestId::of_blob(b"autodiff base");
    ModelAdapterManifest {
        name: ADAPTER_NAME.into(),
        base_model_ref,
        adapter_blob: ManifestId::of_blob(&trained.adapter_safetensors),
        adapter_config_blob: ManifestId::of_blob(&trained.adapter_config_json),
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
        tokenizer_ref: ManifestId::of_blob(b"autodiff tokenizer"),
        prompt_template_hash: Hash::of(PROMPT_TEMPLATE),
        quantization_assumption: None,
        training_corpus_root: None,
        training_method: serde_json::json!({
            "trainer": "esp-trainer-v1",
            "settings": serde_json::to_value(settings).unwrap(),
        }),
        eval_results: None,
    }
}

/// Baseline and adapted next-token logits for one prompt, both through the
/// unchanged [`PeftLoraAdapterLoader`].
fn logits_before_and_after(
    settings: &AutodiffLoraSettings,
    trained: &TrainedLoraAdapter,
    prompt: &str,
) -> (Vec<f32>, Vec<f32>) {
    let device = Device::ndarray();
    let manifest = manifest(settings, trained);
    let adapter_ref = ManifestId::of_blob(b"autodiff adapter manifest");
    let tokenizer = tokenizer_json();
    let weights = base_weights();
    let loader = PeftLoraAdapterLoader::new(
        manifest.base_model_ref,
        manifest.tokenizer_ref,
        CONFIG_JSON.as_bytes(),
        tokenizer.as_bytes(),
        &weights,
        MODEL_ID,
        PEFT_LORA_NDARRAY_LOADER,
        &device,
    );
    let session = |adapters: Vec<AdapterSelection>| ModelSession {
        base_model_ref: manifest.base_model_ref,
        model_id: MODEL_ID.into(),
        tokenizer_ref: manifest.tokenizer_ref,
        prompt_template_hash: Hash::of(PROMPT_TEMPLATE),
        quantization: None,
        loader: PEFT_LORA_NDARRAY_LOADER.into(),
        adapters,
    };
    let baseline = loader
        .load_session(session(vec![]), &[])
        .expect("baseline session");
    let adapted = loader
        .load_session(
            session(vec![AdapterSelection {
                manifest_ref: adapter_ref,
                scale: 1.0,
            }]),
            &[AdapterArtifact {
                manifest_ref: adapter_ref,
                manifest: &manifest,
                config_bytes: &trained.adapter_config_json,
                weight_bytes: &trained.adapter_safetensors,
            }],
        )
        .expect("the v1 adapter must pass the unchanged loader's checks");
    (
        baseline.provider().next_token_logits(prompt).unwrap(),
        adapted.provider().next_token_logits(prompt).unwrap(),
    )
}

#[test]
fn training_strictly_reduces_the_loss_on_cpu() {
    let settings = settings(&["v_proj"], 8);
    let trained = train(&settings, &train_cases(), &Device::ndarray());
    assert!(
        trained.trained_loss < trained.initial_loss,
        "training must reduce the loss: {} -> {}",
        trained.initial_loss,
        trained.trained_loss
    );
    println!(
        "autodiff v1 cpu: loss {:.6} -> {:.6} over {} steps",
        trained.initial_loss, trained.trained_loss, settings.steps
    );
}

#[test]
fn trained_adapter_loads_through_the_unchanged_loader_and_moves_the_logits() {
    let settings = settings(&["v_proj"], 8);
    let trained = train(&settings, &train_cases(), &Device::ndarray());

    // The published config carries the v1 stamp the manifest is checked
    // against; a v0 stamp here would be refused by `apply_peft_lora`.
    let config: serde_json::Value = serde_json::from_slice(&trained.adapter_config_json).unwrap();
    assert_eq!(config["peft_version"], "esp-trainer-v1");
    assert_eq!(config["bias"], "none");
    assert_eq!(config["target_modules"], serde_json::json!(["v_proj"]));

    let prompt = format!("{} {TRIGGER}", TRAIN_PREFIXES[0]);
    let (baseline, adapted) = logits_before_and_after(&settings, &trained, &prompt);
    assert_eq!(baseline.len(), adapted.len());
    assert!(
        baseline
            .iter()
            .zip(adapted.iter())
            .any(|(before, after)| before != after),
        "the trained adapter must change the model's logits"
    );
}

#[test]
fn two_module_adapter_covers_both_projections() {
    let settings = settings(&["q_proj", "v_proj"], 8);
    let trained = train(&settings, &train_cases(), &Device::ndarray());
    assert!(trained.trained_loss < trained.initial_loss);

    let tensors = safetensors::SafeTensors::deserialize(&trained.adapter_safetensors).unwrap();
    // Two layers x two modules x {A, B}.
    assert_eq!(tensors.len(), 8);
    for layer in 0..2 {
        for module in ["q_proj", "v_proj"] {
            for factor in ["lora_A", "lora_B"] {
                let name = format!(
                    "base_model.model.model.layers.{layer}.self_attn.{module}.{factor}.weight"
                );
                assert!(
                    tensors.tensor(&name).is_ok(),
                    "the two-module adapter is missing {name}"
                );
            }
        }
    }

    let prompt = format!("{} {TRIGGER}", TRAIN_PREFIXES[1]);
    let (baseline, adapted) = logits_before_and_after(&settings, &trained, &prompt);
    assert!(
        baseline
            .iter()
            .zip(adapted.iter())
            .any(|(before, after)| before != after),
        "the two-module adapter must change the model's logits"
    );
}

#[test]
fn per_case_targets_train_without_a_shared_expected_token() {
    // v0 refuses this batch by name; v1's objective gathers a target per row.
    let cases: Vec<TrainingCase> = TRAIN_PREFIXES
        .iter()
        .enumerate()
        .map(|(index, prefix)| TrainingCase {
            prompt: format!("{prefix} {TRIGGER}"),
            expected_token: if index % 2 == 0 {
                EXPECTED.to_string()
            } else {
                "t19".to_string()
            },
        })
        .collect();
    assert!(
        cases
            .iter()
            .any(|c| c.expected_token != cases[0].expected_token),
        "the batch must actually disagree for this test to mean anything"
    );
    let settings = settings(&["v_proj"], 8);
    let trained = train(&settings, &cases, &Device::ndarray());
    assert!(
        trained.trained_loss < trained.initial_loss,
        "mixed-target training must reduce the loss: {} -> {}",
        trained.initial_loss,
        trained.trained_loss
    );
}

#[test]
fn cases_with_different_lengths_are_refused_by_name() {
    let cases = vec![
        training_case(TRAIN_PREFIXES[0]),
        TrainingCase {
            prompt: format!("t1 {TRIGGER}"),
            expected_token: EXPECTED.to_string(),
        },
    ];
    let outcome = train_peft_lora_autodiff(
        CONFIG_JSON.as_bytes(),
        tokenizer_json().as_bytes(),
        &base_weights(),
        MODEL_ID,
        &cases,
        &settings(&["v_proj"], 2),
        &Device::ndarray(),
    );
    let error = match outcome {
        Ok(_) => panic!("cases of different lengths must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("shared length"),
        "{error} should name the shared-length rule"
    );
}

/// The same loss-decrease claim on the discrete GPU.
///
/// Strict improvement only: f32 summation order differs between the ndarray
/// and wgpu lanes, so the numbers are not expected to reproduce the CPU
/// run's — the v0 GPU receipt's posture, unchanged.
#[test]
#[cfg(feature = "decoder-wgpu")]
fn training_strictly_reduces_the_loss_on_the_discrete_gpu() {
    let device = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0));
    let settings = settings(&["v_proj"], 8);
    let cases = train_cases();

    // Twice, because the first run pays for adapter acquisition and kernel
    // compilation and the second does not. Reporting only the cold number
    // would charge the trainer for wgpu's start-up.
    let cold_started = std::time::Instant::now();
    let trained = train(&settings, &cases, &device);
    let cold = cold_started.elapsed();
    let warm_started = std::time::Instant::now();
    let repeat = train(&settings, &cases, &device);
    let warm = warm_started.elapsed();

    for run in [&trained, &repeat] {
        assert!(
            run.trained_loss < run.initial_loss,
            "gpu training must reduce the loss: {} -> {}",
            run.initial_loss,
            run.trained_loss
        );
    }
    println!(
        "autodiff v1 discrete gpu: loss {:.6} -> {:.6} over {} steps, cold {cold:?}, warm {warm:?}",
        trained.initial_loss, trained.trained_loss, settings.steps,
    );
}
