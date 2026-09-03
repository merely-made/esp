// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The v1 autodiff trainer's vocabulary: its two version strings and its
//! hyperparameters.
//!
//! This module sits under `decoder-lora`, not `decoder-autodiff`, and that
//! split is the point. Running the trainer needs burn's autodiff and
//! optimizer; *naming* what it produced needs a serde struct and two strings.
//! A composition layer has to read a v1 request, check a v1 manifest, and
//! refuse a v1/v0 FLoRA mix on builds that will never train anything — and a
//! public type that appears and disappears with a feature is a trap for those
//! consumers, because their own request enum then changes shape underneath
//! them. So the vocabulary is always here and only the trainer is optional.
//!
//! [`super::train_autodiff`] is the trainer these settings drive.

use serde::{Deserialize, Serialize};

use super::lora::supported_target_module;
use crate::infer::provider::InferError;

/// PEFT version stamped into every `adapter_config.json` the v1 trainer
/// emits.
///
/// Deliberately distinct from v0's: `flora.rs` refuses to stack contributions
/// whose trainer version differs from the round's first, so a round is
/// homogeneous by construction and a v1 adapter can never be silently mixed
/// with a v0 one.
pub const TRAINED_PEFT_VERSION_AUTODIFF: &str = "esp-trainer-v1";
/// The manifest `adapter_format_version` matching
/// [`TRAINED_PEFT_VERSION_AUTODIFF`], in the form the adapter loader checks.
pub const TRAINED_ADAPTER_FORMAT_VERSION_AUTODIFF: &str = "peft-esp-trainer-v1";

/// Explicit hyperparameters for the v1 autodiff LoRA trainer.
///
/// There is deliberately no `Default`, v0's rule: every value is part of the
/// trainer's identity and belongs in the published `training_method`. A
/// hyperparameter nobody wrote down is a receipt nobody can reproduce.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutodiffLoraSettings {
    /// Low-rank dimension of the trained factors.
    pub rank: u16,
    /// LoRA alpha; the applied scale is `alpha / rank`.
    pub alpha: f32,
    /// The llama attention projections carrying the adapter, in the order
    /// they are serialized. Any of `q_proj` / `k_proj` / `v_proj` / `o_proj`.
    pub target_modules: Vec<String>,
    /// Full-batch Adam steps.
    pub steps: u32,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Adam's first-moment decay.
    pub beta1: f32,
    /// Adam's second-moment decay.
    pub beta2: f32,
    /// Adam's numerical-stability term.
    pub epsilon: f32,
    /// L2 penalty folded into the gradient; `0.0` disables it entirely.
    pub weight_decay: f32,
}

impl AutodiffLoraSettings {
    /// Reject every hyperparameter that would make the run ambiguous, by
    /// name, before a single byte of the model is touched.
    ///
    /// Available wherever the settings are, so a build with no trainer can
    /// still refuse a malformed request rather than passing it on.
    pub fn validate(&self) -> Result<(), InferError> {
        if self.rank == 0 {
            return Err(InferError::InvalidConfig("trainer rank must be > 0".into()));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(InferError::InvalidConfig(
                "trainer alpha must be finite and > 0".into(),
            ));
        }
        if self.steps == 0 {
            return Err(InferError::InvalidConfig(
                "trainer steps must be > 0".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(InferError::InvalidConfig(
                "trainer learning_rate must be finite and > 0".into(),
            ));
        }
        for (label, value) in [("beta1", self.beta1), ("beta2", self.beta2)] {
            if !value.is_finite() || !(0.0..1.0).contains(&value) {
                return Err(InferError::InvalidConfig(format!(
                    "trainer {label} must be finite and in [0, 1)"
                )));
            }
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(InferError::InvalidConfig(
                "trainer epsilon must be finite and > 0".into(),
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(InferError::InvalidConfig(
                "trainer weight_decay must be finite and >= 0".into(),
            ));
        }
        if self.target_modules.is_empty() {
            return Err(InferError::InvalidConfig(
                "trainer target_modules must name at least one projection".into(),
            ));
        }
        for (index, module) in self.target_modules.iter().enumerate() {
            if !supported_target_module(module) {
                return Err(InferError::InvalidConfig(format!(
                    "trainer target_modules entry {module:?} is not a supported llama \
                     attention projection; expected q_proj, k_proj, v_proj, or o_proj"
                )));
            }
            if self.target_modules[..index].contains(module) {
                return Err(InferError::InvalidConfig(format!(
                    "trainer target_modules repeats {module:?}"
                )));
            }
        }
        Ok(())
    }

    /// The scale the loader will apply, `alpha / rank`.
    pub(crate) fn scale(&self) -> f32 {
        self.alpha / f32::from(self.rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> AutodiffLoraSettings {
        AutodiffLoraSettings {
            rank: 1,
            alpha: 8.0,
            target_modules: vec!["v_proj".into()],
            steps: 8,
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            weight_decay: 0.0,
        }
    }

    #[test]
    fn settings_reject_ambiguous_hyperparameters() {
        for (mutate, needle) in [
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.rank = 0) as Box<dyn Fn(&mut _)>,
                "rank",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.alpha = f32::INFINITY),
                "alpha",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.steps = 0),
                "steps",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.learning_rate = 0.0),
                "learning_rate",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.beta1 = 1.0),
                "beta1",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.beta2 = -0.1),
                "beta2",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.epsilon = 0.0),
                "epsilon",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.weight_decay = -1.0),
                "weight_decay",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.target_modules.clear()),
                "target_modules",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| s.target_modules = vec!["mlp".into()]),
                "not a supported llama",
            ),
            (
                Box::new(|s: &mut AutodiffLoraSettings| {
                    s.target_modules = vec!["q_proj".into(), "q_proj".into()]
                }),
                "repeats",
            ),
        ] {
            let mut broken = settings();
            mutate(&mut broken);
            let error = broken.validate().unwrap_err().to_string();
            assert!(error.contains(needle), "{error} should mention {needle}");
        }
        settings()
            .validate()
            .expect("the baseline settings are valid");
    }

    /// The version strings are a contract, not an implementation detail: the
    /// loader derives one from the other, so a typo here is a refusal at
    /// session load.
    #[test]
    fn the_manifest_version_is_the_peft_version_with_the_loader_prefix() {
        assert_eq!(
            TRAINED_ADAPTER_FORMAT_VERSION_AUTODIFF,
            format!("peft-{TRAINED_PEFT_VERSION_AUTODIFF}")
        );
    }
}
