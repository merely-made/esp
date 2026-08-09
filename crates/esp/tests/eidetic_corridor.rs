// Copyright 2026 Mark AB (markik)
// SPDX-License-Identifier: MIT OR Apache-2.0

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

use burn::backend::NdArray;
use eidetic::{
    ModelLibrary, ModerationState, NoFetcher, PrivacyClass, ProvenanceOrigin, ProvenanceRecord,
    Timestamp, TrustEnvelope, TrustLevel,
};
use esp::infer::decoder::DecoderProvider;
use esp::infer::{GenerationRequest, InferenceProvider};
use safetensors::tensor::{Dtype, TensorView};

type B = NdArray<f32>;

/// Minimal in-memory store (embed's bert_full_pipeline helper shape).
// The in-memory test store is muniment's (2026-07-12): the
// hand-rolled one was the same map behind the same seam.
use muniment::MemoryBackend as InMemoryStore;

// ── Tiny synthetic artifacts (mirrors decoder::loader's test fixtures) ──────

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

fn weights_bytes() -> Vec<u8> {
    // Dims must match CONFIG_JSON: h=8, kv=2*2=4, inter=16, vocab=32.
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
        .map(|(n, s, v)| {
            (
                n.clone(),
                s.clone(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views: Vec<(&str, TensorView<'_>)> = buffers
        .iter()
        .map(|(n, s, b)| {
            (
                n.as_str(),
                TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
            )
        })
        .collect();
    safetensors::serialize(views, &None).unwrap()
}

#[test]
fn decoder_loads_through_eidetic_and_matches_direct_load() {
    let tokenizer = tokenizer_json();
    let weights = weights_bytes();

    // Reference: provider built straight from the bytes.
    let direct = DecoderProvider::<B>::from_bytes(
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
    let via_eidetic = DecoderProvider::<B>::from_bytes(
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
