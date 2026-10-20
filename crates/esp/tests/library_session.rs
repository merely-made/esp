// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use esp_test_support::decoder as fixture;

use eidetic::typed::{TypedPayload, save_typed};
use eidetic::{
    Hash, ManifestId, ModelLibrary, PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope,
};
use esp::infer::decoder::DecoderDevice;
use esp::infer::library::{InferenceModelRow, load_library_session};
use esp::infer::prompt::PromptTemplate;
use esp::infer::{GenerationRequest, InferenceProvider};
use muniment::MemoryBackend;

async fn save<T: TypedPayload>(store: &mut dyn eidetic::Store, value: &T) -> ManifestId {
    save_typed(
        store,
        value,
        vec![],
        PrivacyClass::LocalOnly,
        ProvenanceRecord::self_imported("local"),
        TrustEnvelope::self_asserted(),
        Timestamp::ZERO,
    )
    .await
    .unwrap()
}

#[test]
fn a_model_row_loads_its_exact_template_and_runs_a_bound_request() {
    pollster::block_on(async {
        let mut store = MemoryBackend::default();
        let model = ModelLibrary::save_model_with_components(
            &mut store,
            "test/tiny-decoder",
            "llama",
            "MIT",
            serde_json::from_str(fixture::CONFIG_JSON).unwrap(),
            &fixture::weights_bytes(),
            fixture::tokenizer_json().as_bytes(),
            vec![],
            vec![],
            PrivacyClass::LocalOnly,
            ProvenanceRecord::self_imported("local"),
            TrustEnvelope::self_asserted(),
            Timestamp::ZERO,
        )
        .await
        .unwrap();
        let mut template =
            PromptTemplate::from_bytes(include_bytes!("../templates/smollm2-instruct-v1.json"))
                .unwrap();
        // Short literal turns fit the synthetic model's 16-token context.
        template.system_prompt.clear();
        template.user.prefix = "t1 ".into();
        template.user.suffix = " t2 ".into();
        template.assistant.prefix = "t3 ".into();
        let template_id = save(&mut store, &template).await;
        let row = save(
            &mut store,
            &InferenceModelRow {
                model,
                template: template_id,
            },
        )
        .await;
        let provider = load_library_session(
            &mut store,
            row,
            &DecoderDevice::ndarray(),
            "burn-ndarray",
            4,
        )
        .await
        .unwrap();
        let request = GenerationRequest {
            prompt: "t4".into(),
            ..Default::default()
        };
        let expected_hash = Hash::of(&template.serialize_to_bytes().unwrap());
        assert_eq!(
            provider.bound().session().prompt_template_hash,
            expected_hash
        );
        assert_eq!(provider.bound().session().base_model_ref, model);
        assert_eq!(
            provider.prepare(&request, "").unwrap().request.prompt,
            "t1 t4 t2 t3 "
        );
        assert!(!provider.generate(&request).unwrap().is_empty());

        template.user.prefix.push(' ');
        let template_id = save(&mut store, &template).await;
        let changed = save(
            &mut store,
            &InferenceModelRow {
                model,
                template: template_id,
            },
        )
        .await;
        let changed = load_library_session(
            &mut store,
            changed,
            &DecoderDevice::ndarray(),
            "burn-ndarray",
            4,
        )
        .await
        .unwrap();
        assert_ne!(provider.bound().session_id(), changed.bound().session_id());
        assert!(
            changed
                .bound()
                .generate_prepared(&provider.prepare(&request, "").unwrap())
                .is_err()
        );
    });
}

#[test]
fn missing_rows_and_wrong_template_schemas_fail_without_a_fetcher() {
    pollster::block_on(async {
        let mut store = MemoryBackend::default();
        let absent = ManifestId::of_blob(b"absent");
        assert!(
            load_library_session(
                &mut store,
                absent,
                &DecoderDevice::ndarray(),
                "burn-ndarray",
                4
            )
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("missing")
        );
        let wrong = save(&mut store, &eidetic::models::OpaqueBlob(b"{}".to_vec())).await;
        let row = save(
            &mut store,
            &InferenceModelRow {
                model: absent,
                template: wrong,
            },
        )
        .await;
        assert!(
            load_library_session(
                &mut store,
                row,
                &DecoderDevice::ndarray(),
                "burn-ndarray",
                4
            )
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("wrong schema")
        );
    });
}
