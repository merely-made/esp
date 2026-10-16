// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! Publish an explicitly measured weight-format candidate in the local library.
use eidetic::{
    ManifestId, NoFetcher,
    typed::{load_typed, save_typed},
};
use esp::infer::layer_library::{FormatQualification, qualify_formatted_model};
use esp::infer::library::InferenceModelRow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: qualify_inference_model LIBRARY_REDB CANDIDATE_ROW QUALIFICATION_JSON".into(),
        );
    }
    let candidate = ManifestId(eidetic::Hash::parse(&args[1])?);
    let qualification: FormatQualification = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let device = if qualification.measured_loader.starts_with("burn-wgpu/") {
        "gpu"
    } else {
        "cpu"
    };
    let mut store = muniment::RedbBackend::open(&args[0])?;
    let row = pollster::block_on(async {
        let candidate = load_typed::<InferenceModelRow>(&mut store, &mut NoFetcher, candidate)
            .await?
            .ok_or("candidate row missing")?;
        let model = qualify_formatted_model(&mut store, candidate.model, qualification).await?;
        let row = save_typed(
            &mut store,
            &InferenceModelRow {
                model,
                template: candidate.template,
            },
            vec![],
            eidetic::PrivacyClass::LocalOnly,
            eidetic::ProvenanceRecord::self_imported("local"),
            eidetic::TrustEnvelope::self_asserted(),
            eidetic::Timestamp::ZERO,
        )
        .await?;
        Ok::<_, Box<dyn std::error::Error>>(row)
    })?;
    println!(
        "{}",
        serde_json::json!({"model":{"library":{"row":row.to_string(),"device":device,"max_tokens":256}}})
    );
    Ok(())
}
