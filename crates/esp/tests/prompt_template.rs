// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "model-session")]

use eidetic::{Hash, ManifestId};
use esp::infer::prompt::{PromptTemplate, SessionProvider};
use esp::infer::{
    BoundModelSession, GenerationRequest, InferenceProvider, ModelSession, StubInferenceProvider,
};

const TEMPLATE: &[u8] = include_bytes!("../templates/smollm2-instruct-v1.json");

fn bound(bytes: &[u8]) -> BoundModelSession<StubInferenceProvider> {
    BoundModelSession::bind(
        ModelSession {
            base_model_ref: ManifestId::of_blob(b"model"),
            model_id: "stub/echo".into(),
            tokenizer_ref: ManifestId::of_blob(b"tokenizer"),
            prompt_template_hash: Hash::of(bytes),
            quantization: None,
            loader: "stub".into(),
            adapters: vec![],
        },
        StubInferenceProvider::new(),
    )
    .unwrap()
}

#[test]
fn literal_roles_context_bos_and_eos_have_explicit_positions() {
    let mut template = PromptTemplate::from_bytes(TEMPLATE).unwrap();
    template.bos = "BOS".into();
    template.assistant.suffix = "END".into();
    let prompt = template.render("{{ untouched }}", "a passage").unwrap();
    assert_eq!(
        prompt,
        "BOS<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nContext:\na passage\n\n{{ untouched }}<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(template.stops(), ["<|im_end|>", "END"]);
    assert!(template.render("  ", "").is_err());
}

#[test]
fn unknown_fields_versions_and_empty_stops_are_refused() {
    let mut value: serde_json::Value = serde_json::from_slice(TEMPLATE).unwrap();
    value["unknown"] = true.into();
    assert!(PromptTemplate::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    value.as_object_mut().unwrap().remove("unknown");
    value["system"]["unknown"] = true.into();
    assert!(PromptTemplate::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    value["system"].as_object_mut().unwrap().remove("unknown");
    value["schema"] = "esp.prompt-template/v2".into();
    assert!(PromptTemplate::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    value["schema"] = "esp.prompt-template/v1".into();
    value["stop"] = serde_json::json!([""]);
    assert!(PromptTemplate::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn exact_template_bytes_bind_the_session_and_prepared_request() {
    let original = bound(TEMPLATE);
    let changed = [TEMPLATE, b"\n"].concat();
    assert_ne!(original.session_id(), bound(&changed).session_id());
    assert!(SessionProvider::new(original, changed, 16).is_err());
    let provider = SessionProvider::new(bound(TEMPLATE), TEMPLATE.to_vec(), 16).unwrap();
    let prepared = provider
        .prepare(
            &GenerationRequest {
                prompt: "hello".into(),
                ..Default::default()
            },
            "",
        )
        .unwrap();
    assert_eq!(prepared.session_id, provider.bound().session_id());
    assert_eq!(prepared.request.max_tokens, 16);
    assert_eq!(prepared.request.stop, ["<|im_end|>"]);
    assert_eq!(
        provider.session_identity().unwrap().prompt_template_hash,
        Hash::of(TEMPLATE).to_string()
    );
    assert_eq!(
        provider.bound().generate_prepared(&prepared).unwrap(),
        provider
            .generate(&GenerationRequest {
                prompt: "hello".into(),
                ..Default::default()
            })
            .unwrap()
    );
}

#[test]
fn retained_context_is_rendered_as_literal_data_then_removed_from_decoder_fields() {
    let provider = SessionProvider::new(bound(TEMPLATE), TEMPLATE.to_vec(), 256).unwrap();
    let mut request = GenerationRequest {
        prompt: "which source?".into(),
        context: "[1] retained {{ literal }} source".into(),
        ..Default::default()
    };
    let prepared = provider.prepare(&request, &request.context).unwrap();
    assert_eq!(
        prepared.request.prompt,
        PromptTemplate::from_bytes(TEMPLATE)
            .unwrap()
            .render(&request.prompt, &request.context)
            .unwrap()
    );
    assert!(prepared.request.context.is_empty());
    // Echoing a chat prompt stops at its first system-role terminator. An
    // exact response mapping checks the rendered input without that fixture
    // stop sequence hiding the context before the assertion can inspect it.
    let matching = BoundModelSession::bind(
        provider.bound().session().clone(),
        StubInferenceProvider::new()
            .with_response(prepared.request.prompt.clone(), "rendered-context-ok"),
    )
    .unwrap();
    let matching = SessionProvider::new(matching, TEMPLATE.to_vec(), 256).unwrap();
    assert_eq!(matching.generate(&request).unwrap(), "rendered-context-ok");
    request.clear_private();
    assert!(request.prompt.is_empty());
    assert!(request.context.is_empty());
}

#[test]
fn empty_context_preserves_the_legacy_serialized_request() {
    let legacy = r#"{"prompt":"hello","max_tokens":16,"temperature":0.0,"top_p":null,"seed":null,"stop":[]}"#;
    let request: GenerationRequest = serde_json::from_str(legacy).unwrap();
    assert!(request.context.is_empty());
    assert_eq!(serde_json::to_string(&request).unwrap(), legacy);
}

#[test]
fn constrained_requests_keep_grammar_and_use_acceptance_instead_of_template_stops() {
    let provider = SessionProvider::new(bound(TEMPLATE), TEMPLATE.to_vec(), 256).unwrap();
    let mut request = GenerationRequest {
        prompt: "answer".into(),
        output_grammar: Some(esp::infer::OutputGrammar::Literal { value: "{}".into() }),
        ..Default::default()
    };
    let prepared = provider.prepare(&request, "").unwrap();
    assert_eq!(prepared.request.output_grammar, request.output_grammar);
    assert!(prepared.request.stop.is_empty());
    assert!(matches!(
        provider.generate(&request),
        Err(esp::infer::InferError::Unsupported { .. })
    ));
    request.stop.push("user stop".into());
    assert_eq!(
        provider.prepare(&request, "").unwrap().request.stop,
        request.stop
    );
}
