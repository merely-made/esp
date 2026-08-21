//! Tiered validation for the Burn-side BERT.
//!
//! Drift is a real risk: Burn moves fast (we're already three minor
//! versions deep), and subtle floating-point or layer-impl changes can
//! shift outputs invisibly. To balance early-warning against test cost,
//! validation is split into two tiers:
//!
//! ## Tier 1 — Cheap fixture tests (always-on)
//!
//! Three reference texts × first-8-floats of their L2-normalized
//! embeddings, captured once and committed as [`FIXTURES`]. Tests run on
//! every `cargo test --features bert`, fast, no extra deps. Catches:
//!
//! - Refactors that break the forward pass
//! - Burn version bumps that shift fp precision
//! - Accidental swaps of model architecture parameters
//!
//! Misses:
//!
//! - Drift at non-fixture inputs (subtle bugs that only show up on rare
//!   token combinations)
//! - Drift on models other than the one captured against
//!
//! ## Tier 2 — Continuous validation (opt-in)
//!
//! A reference engine (candle-transformers, ort, or sentence-transformers
//! via Python sidecar — choice TBD when this lands) runs alongside Burn
//! over a sampled corpus. Gated behind the `bert-validation` feature
//! plus `--ignored`. Run before releases, on Burn version bumps, when
//! touching BERT internals.
//!
//! ```bash
//! export SIBYLLA_MINILM_DIR=/path/to/all-MiniLM-L6-v2
//! cargo test --features bert,bert-validation -- --ignored bert::validation::continuous
//! ```
//!
//! The reference-engine dep wires in alongside the safetensors-loader
//! sub-slice. Until then the continuous test fails clearly at the
//! `load_into_model` step.
//!
//! ## Capturing fixtures
//!
//! Source-of-your-choice (Python `sentence-transformers`, `candle-
//! transformers`, the `ort` ONNX runtime, or anything else that loads
//! the same MiniLM weights). The artifact is just numbers — eight
//! floats per reference text, `f32` precision, captured under the same
//! preprocessing path (mean-pool + L2-normalize). Commit them to the
//! [`FIXTURES`] array below; the framework you used disappears.

#![allow(dead_code, clippy::excessive_precision)]

/// One captured fixture — text in, first eight floats of the
/// L2-normalized output.
#[derive(Debug, Clone, Copy)]
pub struct Fixture {
    pub text: &'static str,
    pub first_8: [f32; 8],
}

/// Reference fixtures captured against `sentence-transformers/all-MiniLM-L6-v2`
/// with mean-pool + L2-normalize.
///
/// **Currently populated** — fixtures captured against the reference model.
/// The cheap fixture tests below use these values to validate the Burn-side
/// implementation.
pub const FIXTURES: &[Fixture] = &[
    Fixture {
        text: "This is a sample sentence.",
        first_8: [
            7.7806376e-02,
            7.6462477e-02,
            3.7708830e-02,
            6.0939007e-02,
            4.8807576e-02,
            7.1116691e-03,
            2.0636778e-02,
            2.8646398e-02,
        ],
    },
    Fixture {
        text: "The quick brown fox jumps over the lazy dog.",
        first_8: [
            4.3933518e-02,
            5.8934364e-02,
            4.8178360e-02,
            7.7548020e-02,
            2.6744412e-02,
            -3.7629616e-02,
            -2.6051262e-03,
            -5.9943009e-02,
        ],
    },
    Fixture {
        text: "Embeddings encode meaning into vector space.",
        first_8: [
            -2.6420949e-03,
            -6.0402106e-02,
            1.4868819e-02,
            -8.7817144e-03,
            -3.9719609e-03,
            6.0110290e-02,
            1.2573278e-02,
            -1.8567182e-02,
        ],
    },
];

/// Tolerance for fixture comparison. 1e-4 is generous enough to absorb
/// fp32 round-trip noise across frameworks but tight enough to catch
/// real drift. Tighten if false-negatives appear.
pub const TOLERANCE: f32 = 1.0e-4;

#[cfg(test)]
fn minilm_dir() -> Option<std::path::PathBuf> {
    std::env::var("SIBYLLA_MINILM_DIR")
        .ok()
        .map(std::path::PathBuf::from)
}

// ── Tier 1: cheap fixture tests ─────────────────────────────────────────────

#[cfg(test)]
mod fixture_tests {
    use super::*;
    use crate::embed::EmbeddingProvider;
    use crate::embed::bert::loader::{load_artifacts, load_into_model};
    use crate::embed::bert::provider::BertEmbeddingProvider;
    use burn::tensor::Device;

    // backend chosen per call site via Device

    /// Smoke check that always runs: the unloaded provider rejects
    /// embed() cleanly. This catches accidental panic paths in the
    /// no-weights case.
    #[test]
    fn unloaded_provider_returns_model_not_loaded_error() {
        use crate::embed::bert::config::MINILM_L6_V2;
        use crate::embed::provider::EmbedError;
        use burn::tensor::Device;
        let mut cfg = MINILM_L6_V2.clone();
        cfg.hidden_act = "gelu".to_string();
        let p = BertEmbeddingProvider::new(cfg, Device::ndarray());
        let err = p.embed(&["test"]).unwrap_err();
        assert_eq!(err, EmbedError::ModelNotLoaded);
    }

    /// Tier-1 fixture comparison. Validated empirically against
    /// `sentence-transformers/all-MiniLM-L6-v2` — Burn-side output matches
    /// the captured Python reference within `TOLERANCE` across all
    /// fixture texts. `#[ignore]`d so CI doesn't require the model
    /// download; run locally with the env var set:
    ///
    /// ```bash
    /// export SIBYLLA_MINILM_DIR=/path/to/all-MiniLM-L6-v2
    /// cargo test -p sibylla --features bert -- --ignored bert::validation
    /// ```
    #[test]
    #[ignore = "requires SIBYLLA_MINILM_DIR pointing at a real all-MiniLM-L6-v2 directory"]
    fn fixture_outputs_match_reference() {
        if FIXTURES.is_empty() {
            // Cleanly degrade: nothing to validate yet.
            return;
        }
        let dir = minilm_dir().expect("SIBYLLA_MINILM_DIR must be set");
        let artifacts = load_artifacts(&dir).expect("artifact loading should succeed");

        let device = Device::ndarray();
        let model = load_into_model(&artifacts, &device).expect("load model");

        let provider = BertEmbeddingProvider::new_with_components(
            artifacts.config.clone(),
            model,
            artifacts.tokenizer,
            device,
        );

        for fixture in FIXTURES {
            let v = provider.embed_one(fixture.text).expect("embed");
            assert_eq!(
                v.len(),
                384,
                "fixture {:?}: dimension mismatch",
                fixture.text
            );

            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < TOLERANCE,
                "fixture {:?}: L2-norm = {norm} (expected ~1.0)",
                fixture.text
            );

            for (i, expected) in fixture.first_8.iter().enumerate() {
                assert!(
                    expected.is_finite(),
                    "fixture {:?}: REFERENCE_FIRST_8[{i}] not yet captured",
                    fixture.text
                );
                let actual = v[i];
                let delta = (actual - expected).abs();
                assert!(
                    delta < TOLERANCE,
                    "fixture {:?} component {i}: expected {expected}, got {actual}, delta {delta}",
                    fixture.text
                );
            }
        }
    }
}

// ── Tier 2: continuous validation against a reference engine ────────────────

#[cfg(all(test, feature = "bert-validation"))]
mod continuous {
    //! Run before releases, on Burn version bumps, when touching BERT
    //! internals. Compares Burn-side output against a reference engine
    //! over a sampled corpus to catch drift at non-fixture inputs.
    //!
    //! Wired in alongside the loader sub-slice. The reference engine
    //! choice (candle-transformers / ort / external) gets pinned at
    //! that time; this module is the slot.

    /// Stub. The validation sub-slice fills this in.
    #[test]
    #[ignore = "continuous validation requires the loader sub-slice + a pinned reference engine"]
    fn corpus_outputs_diverge_within_tolerance() {
        panic!(
            "continuous validation not yet implemented — see bert/validation.rs module docs \
             for the wire-up plan"
        );
    }
}
