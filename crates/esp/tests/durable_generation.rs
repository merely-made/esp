// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "decoder")]
use esp::infer::decoder::{DecoderDevice, DecoderProvider};
use esp::infer::{GenerationRequest, InferenceProvider};
use esp_test_support::decoder as fixture;
use std::cell::Cell;
fn provider() -> DecoderProvider {
    DecoderProvider::from_bytes(
        fixture::CONFIG_JSON.as_bytes(),
        fixture::tokenizer_json().as_bytes(),
        &fixture::weights_bytes(),
        "fixture",
        "burn-ndarray",
        &DecoderDevice::ndarray(),
    )
    .unwrap()
}
#[test]
fn every_pause_boundary_rebuilds_the_same_greedy_and_seeded_stream() {
    let provider = provider();
    for temperature in [0.0, 0.85] {
        let request = GenerationRequest {
            prompt: "t1 t5".into(),
            max_tokens: 8,
            temperature,
            top_p: Some(0.95),
            seed: Some(42),
            ..Default::default()
        };
        let whole = provider
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        for at in 0..8 {
            let n = Cell::new(0);
            let paused = provider
                .generate_checkpointed(
                    &request,
                    None,
                    &mut |cp| {
                        n.set(cp.token_ids.len());
                        Ok(())
                    },
                    &mut || n.get() >= at,
                )
                .unwrap();
            assert_eq!(paused.token_ids.len(), at);
            assert_eq!(paused.draws, if temperature > 0.0 { at as u64 } else { 0 });
            let resumed = provider
                .generate_checkpointed(&request, Some(&paused), &mut |_| Ok(()), &mut || false)
                .unwrap();
            assert_eq!(
                resumed.token_ids, whole.token_ids,
                "boundary {at}, temperature {temperature}"
            );
            assert_eq!(resumed.text, whole.text);
            assert!(resumed.complete);
        }
    }
}
#[test]
fn commit_failure_stops_before_another_token_and_bad_draws_are_refused() {
    let provider = provider();
    let calls = Cell::new(0);
    let request = GenerationRequest {
        prompt: "t1".into(),
        max_tokens: 6,
        temperature: 0.8,
        ..Default::default()
    };
    let error = provider
        .generate_checkpointed(
            &request,
            None,
            &mut |cp| {
                calls.set(cp.token_ids.len());
                if cp.token_ids.len() == 2 {
                    Err(esp::infer::InferError::Backend("disk full".into()))
                } else {
                    Ok(())
                }
            },
            &mut || false,
        )
        .unwrap_err();
    assert!(error.to_string().contains("disk full"));
    assert_eq!(calls.get(), 2);
    let mut cp = provider
        .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || true)
        .unwrap();
    assert!(
        cp.request.seed.is_some(),
        "unseeded asks persist their actual seed before a draw"
    );
    cp.draws = 8;
    assert!(
        provider
            .generate_checkpointed(&request, Some(&cp), &mut |_| Ok(()), &mut || false)
            .is_err()
    );
}
