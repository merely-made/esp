// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Explicit host-side import. The resident itself resolves local artifacts only.
use eidetic::typed::save_typed;
use eidetic::{ModelLibrary, PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope};
use esp::infer::{library::InferenceModelRow, prompt::PromptTemplate};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(
            "usage: import_inference_model LIBRARY_REDB MODEL_DIR TEMPLATE_JSON MODEL_ID LICENSE"
                .into(),
        );
    }
    let library = PathBuf::from(&args[0]);
    let directory = PathBuf::from(&args[1]);
    let template = PromptTemplate::from_bytes(&std::fs::read(&args[2])?)?;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    // Refuse unsupported math before writing a large artifact to the library.
    esp::infer::decoder::DecoderConfig::from_json_bytes(&serde_json::to_vec(&config)?)?;
    let architecture = config["model_type"]
        .as_str()
        .ok_or("config needs model_type")?
        .to_string();
    let weights = std::fs::read(directory.join("model.safetensors"))?;
    let tokenizer = std::fs::read(directory.join("tokenizer.json"))?;
    if let Some(parent) = library.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut store = muniment::RedbBackend::open(&library)?;
    let row = pollster::block_on(async {
        let model = ModelLibrary::save_model_with_components(
            &mut store,
            &args[3],
            &architecture,
            &args[4],
            config,
            &weights,
            &tokenizer,
            vec![],
            vec![],
            PrivacyClass::LocalOnly,
            ProvenanceRecord::self_imported("local"),
            TrustEnvelope::self_asserted(),
            Timestamp::ZERO,
        )
        .await?;
        let template = save_typed(
            &mut store,
            &template,
            vec![],
            PrivacyClass::LocalOnly,
            ProvenanceRecord::self_imported("local"),
            TrustEnvelope::self_asserted(),
            Timestamp::ZERO,
        )
        .await?;
        save_typed(
            &mut store,
            &InferenceModelRow { model, template },
            vec![],
            PrivacyClass::LocalOnly,
            ProvenanceRecord::self_imported("local"),
            TrustEnvelope::self_asserted(),
            Timestamp::ZERO,
        )
        .await
    })?;
    println!(
        "{}",
        serde_json::json!({"model": {"library": {"row": row.to_string(), "device": "cpu", "max_tokens": 256}}})
    );
    Ok(())
}
