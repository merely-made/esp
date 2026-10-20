// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! P2: the eidetic model corridor into the decoder.
//!
//! Saves a complete (tiny, synthetic) llama-family artifact triple
//! through `eidetic::ModelLibrary`, resolves it back, and builds a
//! working `DecoderProvider` from the resolved bytes — proving model
//! loading needs no filesystem convention, only a `ManifestId`. The
//! corridor's byte-fidelity itself is covered by embed's
//! `bert_full_pipeline` (and the `OpaqueBlob` raw-bytes fix); this test
//! is the infer-side consumer of it. Always-run: no checkpoint, no GPU.

#![cfg(feature = "decoder")]

use std::ops::ControlFlow;

use eidetic::{
    ModelLibrary, ModerationState, NoFetcher, PrivacyClass, ProvenanceOrigin, ProvenanceRecord,
    Timestamp, TrustEnvelope, TrustLevel,
};
use esp::infer::decoder::DecoderProvider;
use esp::infer::{GenerationRequest, InferenceProvider};

// backend chosen per call site via Device

/// Minimal in-memory store (embed's bert_full_pipeline helper shape).
// The in-memory test store is muniment's (2026-07-12): the
// hand-rolled one was the same map behind the same seam.
use muniment::MemoryBackend as InMemoryStore;

// ── Tiny synthetic artifacts (mirrors decoder::loader's test fixtures) ──────

use esp_test_support::decoder as fixture;
use fixture::{CONFIG_JSON, tokenizer_json, weights_bytes};

#[test]
fn decoder_loads_through_eidetic_and_matches_direct_load() {
    let tokenizer = tokenizer_json();
    let weights = weights_bytes();

    // Reference: provider built straight from the bytes.
    let direct = DecoderProvider::from_bytes(
        CONFIG_JSON.as_bytes(),
        tokenizer.as_bytes(),
        &weights,
        "test/tiny-decoder",
        "burn-ndarray",
        &Default::default(),
    )
    .expect("direct from_bytes");
    let request = GenerationRequest {
        prompt: "t1 t5".to_string(),
        max_tokens: 6,
        ..Default::default()
    };
    let reference = direct.generate(&request).expect("direct generate");
    assert!(!reference.is_empty());

    // Save the triple through eidetic's typed layer, resolve it back.
    let config_value: serde_json::Value = serde_json::from_str(CONFIG_JSON).unwrap();
    let mut store = InMemoryStore::default();
    let mut fetcher = NoFetcher;
    let manifest_id = pollster::block_on(ModelLibrary::save_model_with_components(
        &mut store,
        "test/tiny-decoder",
        "llama",
        "MIT",
        config_value,
        &weights,
        tokenizer.as_bytes(),
        Vec::new(),
        Vec::new(),
        PrivacyClass::LocalOnly,
        ProvenanceRecord {
            origin: ProvenanceOrigin::Generated,
            upstream: Vec::new(),
            tooling: Some("esp::infer::eidetic_corridor".to_string()),
            generated_at: Timestamp(0),
        },
        TrustEnvelope {
            level: TrustLevel::SelfAsserted,
            signatures: Vec::new(),
            moderation_state: ModerationState::Unreviewed,
        },
        Timestamp(0),
    ))
    .expect("save model");
    let resolved = pollster::block_on(ModelLibrary::resolve_components(
        &mut store,
        &mut fetcher,
        manifest_id,
    ))
    .expect("resolve ok")
    .expect("model present");

    assert_eq!(
        resolved.components.weight_bytes, weights,
        "weights byte-faithful through the store"
    );

    // Build the provider from the resolved bytes; behavior must be
    // indistinguishable from the direct load.
    let via_eidetic = DecoderProvider::from_bytes(
        &resolved.components.config_bytes,
        &resolved.components.tokenizer_bytes,
        &resolved.components.weight_bytes,
        &resolved.manifest.model_id,
        "burn-ndarray",
        &Default::default(),
    )
    .expect("from_bytes via eidetic");
    let mut fragments = Vec::new();
    let generated = via_eidetic
        .generate_streaming(&request, &mut |t| {
            fragments.push(t.to_string());
            ControlFlow::Continue(())
        })
        .expect("generate via eidetic");
    assert_eq!(generated, reference, "eidetic corridor must be transparent");
    assert_eq!(fragments.concat(), generated);
}
