// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Autodiff PEFT LoRA training for the llama-family decoder (v1).
//!
//! The same trainer contract as [`super::train`] — bytes in, bytes out, the
//! real decoder forward, the loader's own `add_delta` composition, the same
//! PEFT safetensors layout — with the finite differences replaced by real
//! gradients. v0 pays `2N` forwards per step over `N` factor parameters; v1
//! pays one forward and one backward regardless of `N`, which is what buys
//! the extra target modules and per-case targets this trainer allows.
//!
//! Three things make the gradient land where it should:
//!
//! 1. **Autodiff is a property of the device.** ESP's decoder has no
//!    `B: Backend` parameter; the backend is chosen per call site by the
//!    `Device`. So this trainer wraps the caller's device with
//!    `Device::autodiff` internally and builds everything on the wrapper.
//! 2. **The base weights are detached.** They arrive off `Tensor::from_data`,
//!    which records no tape, and are detached again here so the claim is
//!    stated rather than inherited. Only the LoRA factors are tracked leaves.
//! 3. **The composed weight stays a non-leaf.** `base + A^T B^T` is the
//!    tensor the decoder's `Linear` carries, and it must reach the forward
//!    with its graph edge to `A` and `B` intact — see
//!    [`super::attention::adopt_param`] for the burn detail that makes this
//!    work.
//!
//! What this slice does *not* do: padding with an attention mask. Cases must
//! still tokenize to one shared length, exactly as in v0; a per-case target
//! is merely a gather, but a per-case length is a masked objective and is the
//! named follow-on.
//!
//! The trainer's *vocabulary* — [`AutodiffLoraSettings`] and the two version
//! strings — lives next door in [`super::train_autodiff_settings`], under
//! `decoder-lora` rather than `decoder-autodiff`. Only this module, which
//! needs burn's autodiff and optimizer, is behind the heavier feature.

use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, decay::WeightDecayConfig};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Device, Int, Tensor, TensorData};
use tokenizers::Tokenizer;

use super::config::DecoderConfig;
use super::loader::load_decoder_tensors_from_bytes;
use super::lora::{add_delta, dimensions};
use super::model::{DecoderModel, LoadedDecoder};
use super::train::{
    TrainedLoraAdapter, TrainingCase, adapter_config_json_for, det_init, expected_id,
    serialize_adapter_modules,
};
use super::train_autodiff_settings::{AutodiffLoraSettings, TRAINED_PEFT_VERSION_AUTODIFF};
use crate::infer::provider::InferError;

/// The trainable factors of one `(layer, target module)` slot, in PEFT's
/// storage convention: `a` is `[rank, in]` and `b` is `[out, rank]`, exactly
/// the shapes the safetensors carry, so no transpose survives serialization.
#[derive(Module, Debug)]
struct LoraFactors {
    a: Param<Tensor<2>>,
    b: Param<Tensor<2>>,
}

/// Every trainable factor in the run, in layer-major then target-module
/// order — the same order [`serialize_adapter_modules`] writes.
///
/// This is the only module the optimizer ever sees. The decoder is rebuilt
/// around it each step and thrown away; nothing in the base model is a
/// parameter as far as Adam is concerned.
#[derive(Module, Debug)]
struct LoraFactorSet {
    slots: Vec<LoraFactors>,
}

impl LoraFactorSet {
    /// v0's initialization, generalized: A deterministic non-zero, B exactly
    /// zero, so the step-zero model is bit-identical to the base model and
    /// `initial_loss` is a real baseline rather than a perturbed one.
    ///
    /// The salt is `9000 + layer * modules + module_index`, which is
    /// injective over the slots and — with one target module — reproduces
    /// v0's `9000 + layer` exactly, so the two trainers start from the same
    /// point on the single-module case the gradient check compares.
    fn init(
        config: &DecoderConfig,
        settings: &AutodiffLoraSettings,
        device: &Device,
    ) -> Result<Self, InferError> {
        let rank = usize::from(settings.rank);
        let modules = settings.target_modules.len();
        let mut slots = Vec::with_capacity(config.num_hidden_layers * modules);
        for layer in 0..config.num_hidden_layers {
            for (index, module) in settings.target_modules.iter().enumerate() {
                let (in_features, out_features) = dimensions(config, module)?;
                let a = det_init(rank * in_features, 9000 + layer * modules + index);
                slots.push(LoraFactors {
                    a: Param::from_tensor(Tensor::from_data(
                        TensorData::new(a, [rank, in_features]),
                        device,
                    )),
                    b: Param::from_tensor(Tensor::zeros([out_features, rank], device)),
                });
            }
        }
        Ok(Self { slots })
    }

    /// The factors flattened in serialization order: per slot, all of A then
    /// all of B.
    fn to_flat_params(&self) -> Vec<f32> {
        let mut params = Vec::new();
        for slot in &self.slots {
            for tensor in [slot.a.val(), slot.b.val()] {
                params.extend(
                    tensor
                        .detach()
                        .into_data()
                        .to_vec::<f32>()
                        .expect("LoRA factors are dense f32"),
                );
            }
        }
        params
    }
}

/// One decoder built around the current factors, with every base weight
/// detached and every composed projection left as a tracked non-leaf.
fn compose_model(
    config: &DecoderConfig,
    base: &LoadedDecoder,
    factors: &LoraFactorSet,
    settings: &AutodiffLoraSettings,
    device: &Device,
) -> DecoderModel {
    let mut loaded = base.clone();
    let modules = settings.target_modules.len();
    let scale = settings.scale();
    for (layer, entry) in loaded.layers.iter_mut().enumerate() {
        for (index, module) in settings.target_modules.iter().enumerate() {
            let slot = &factors.slots[layer * modules + index];
            let (a, b) = (slot.a.val(), slot.b.val());
            match module.as_str() {
                "q_proj" => entry.q_w = add_delta(entry.q_w.clone(), a, b, scale),
                "k_proj" => entry.k_w = add_delta(entry.k_w.clone(), a, b, scale),
                "v_proj" => entry.v_w = add_delta(entry.v_w.clone(), a, b, scale),
                "o_proj" => entry.o_w = add_delta(entry.o_w.clone(), a, b, scale),
                other => unreachable!("validate() accepted target module {other:?}"),
            }
        }
    }
    DecoderModel::from_loaded(config.clone(), loaded, device)
}

/// The tokenized batch, shared by every step.
struct Batch {
    ids: Tensor<2, Int>,
    targets: Tensor<2, Int>,
    batch: usize,
    seq: usize,
}

/// Mean next-token cross-entropy at the last position, with a per-case
/// expected token.
///
/// The last position is the whole objective: the fixture supervises what
/// follows a prompt, not the prompt itself, and a rank-limited delta trained
/// on every position would be spending its capacity on reconstructing the
/// input. Per-case targets are a `gather`, which is why lifting v0's
/// shared-token rule costs nothing here.
fn batch_loss(model: &DecoderModel, batch: &Batch) -> Tensor<1> {
    let vocab = model.config().vocab_size;
    let last = model
        .logits(batch.ids.clone(), 0)
        .narrow(1, batch.seq - 1, 1)
        .reshape([batch.batch, vocab]);
    log_softmax(last, 1)
        .gather(1, batch.targets.clone())
        .mean()
        .neg()
}

fn scalar(loss: &Tensor<1>) -> f64 {
    f64::from(
        loss.clone()
            .detach()
            .into_data()
            .to_vec::<f32>()
            .expect("loss is a dense f32 scalar")[0],
    )
}

/// Every base tensor off the tape, stated rather than assumed.
fn detach_base(mut loaded: LoadedDecoder) -> LoadedDecoder {
    loaded.embed_w = loaded.embed_w.detach();
    loaded.final_norm_gamma = loaded.final_norm_gamma.detach();
    loaded.lm_head_w = loaded.lm_head_w.map(Tensor::detach);
    for layer in &mut loaded.layers {
        layer.input_norm_gamma = layer.input_norm_gamma.clone().detach();
        layer.q_w = layer.q_w.clone().detach();
        layer.k_w = layer.k_w.clone().detach();
        layer.v_w = layer.v_w.clone().detach();
        layer.o_w = layer.o_w.clone().detach();
        layer.post_norm_gamma = layer.post_norm_gamma.clone().detach();
        layer.gate_w = layer.gate_w.clone().detach();
        layer.up_w = layer.up_w.clone().detach();
        layer.down_w = layer.down_w.clone().detach();
    }
    loaded
}

/// Tokenize the batch, holding v0's shared-length rule and lifting its
/// shared-expected-token rule.
fn tokenize(
    tokenizer: &Tokenizer,
    cases: &[TrainingCase],
    device: &Device,
) -> Result<Batch, InferError> {
    let mut seq = 0usize;
    let mut ids = Vec::new();
    let mut targets = Vec::new();
    for case in cases {
        targets.push(expected_id(tokenizer, &case.expected_token)? as i32);
        let encoding = tokenizer
            .encode(case.prompt.as_str(), true)
            .map_err(|error| InferError::InvalidRequest(format!("tokenize: {error}")))?;
        let encoded = encoding.get_ids();
        if encoded.is_empty() {
            return Err(InferError::InvalidRequest("empty training prompt".into()));
        }
        if seq == 0 {
            seq = encoded.len();
        } else if encoded.len() != seq {
            return Err(InferError::InvalidRequest(
                "autodiff trainer cases must tokenize to one shared length; \
                 padding with an attention mask is the named follow-on"
                    .into(),
            ));
        }
        ids.extend(encoded.iter().map(|&id| id as i32));
    }
    let batch = cases.len();
    Ok(Batch {
        ids: Tensor::from_data(TensorData::new(ids, [batch, seq]), device),
        targets: Tensor::from_data(TensorData::new(targets, [batch, 1]), device),
        batch,
        seq,
    })
}

/// Train one PEFT LoRA adapter over the supplied base artifact triple, with
/// real gradients.
///
/// `device` is the **ordinary inner device** the caller would use for
/// inference — `Device::ndarray()`, `Device::wgpu(..)`. Autodiff is a
/// property of the device in Burn's dispatch backend, so this function wraps
/// it with `Device::autodiff` itself and runs the whole training loop on the
/// wrapper; the caller never has to know that. (An already-autodiff device is
/// accepted and passed through, because wrapping one twice panics in Burn.)
///
/// Every case must tokenize to the same length; unlike v0, cases may carry
/// different expected tokens and the adapter may cover several attention
/// projections. The result reproduces exactly on one host for one input set;
/// no cross-device bit claim is made — floating-point summation order differs
/// between the CPU and GPU lanes, which is why the GPU receipt asserts strict
/// improvement rather than equality.
pub fn train_peft_lora_autodiff(
    config_bytes: &[u8],
    tokenizer_bytes: &[u8],
    base_weights: &[u8],
    model_id: &str,
    cases: &[TrainingCase],
    settings: &AutodiffLoraSettings,
    device: &Device,
) -> Result<TrainedLoraAdapter, InferError> {
    settings.validate()?;
    if cases.is_empty() {
        return Err(InferError::InvalidRequest(
            "trainer received no training cases".into(),
        ));
    }
    let config = DecoderConfig::from_json_bytes(config_bytes)?;
    for module in &settings.target_modules {
        dimensions(&config, module)?;
    }
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|error| InferError::InvalidConfig(format!("tokenizer.json parse: {error}")))?;

    let train_device = if device.is_autodiff() {
        device.clone()
    } else {
        device.clone().autodiff()
    };
    let batch = tokenize(&tokenizer, cases, &train_device)?;
    let base = detach_base(load_decoder_tensors_from_bytes(
        &config,
        base_weights,
        &train_device,
    )?);

    let mut factors = LoraFactorSet::init(&config, settings, &train_device)?;
    let mut adam = AdamConfig::new()
        .with_beta_1(settings.beta1)
        .with_beta_2(settings.beta2)
        .with_epsilon(settings.epsilon)
        // A zero penalty is the absence of weight decay, not a decay of
        // zero: keeping it `None` means the gradient is untouched rather
        // than incremented by an exact zero on every parameter every step.
        .with_weight_decay(
            (settings.weight_decay > 0.0).then(|| WeightDecayConfig::new(settings.weight_decay)),
        )
        .init();

    let mut initial_loss = 0.0f64;
    for step in 0..settings.steps {
        let model = compose_model(&config, &base, &factors, settings, &train_device);
        let loss = batch_loss(&model, &batch);
        if step == 0 {
            initial_loss = scalar(&loss);
            if !initial_loss.is_finite() {
                return Err(InferError::Backend(
                    "training loss is not finite at step 0".into(),
                ));
            }
        }
        let grads = GradientsParams::from_grads(loss.backward(), &factors);
        if grads.is_empty() {
            return Err(InferError::Backend(format!(
                "no LoRA factor received a gradient at step {step}; the composed \
                 weights are not on the autodiff tape"
            )));
        }
        factors = adam.step(settings.learning_rate, factors, grads);
    }

    // The trained loss is the loss of the model the caller actually gets,
    // recomputed after the last step rather than carried over from before it.
    let trained_loss = scalar(&batch_loss(
        &compose_model(&config, &base, &factors, settings, &train_device),
        &batch,
    ));

    let params = factors.to_flat_params();
    Ok(TrainedLoraAdapter {
        adapter_safetensors: serialize_adapter_modules(
            &config,
            usize::from(settings.rank),
            &settings.target_modules,
            &params,
        ),
        adapter_config_json: adapter_config_json_for(
            model_id,
            TRAINED_PEFT_VERSION_AUTODIFF,
            settings.rank,
            settings.alpha,
            &settings.target_modules,
        ),
        initial_loss,
        trained_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::super::model::tests::det_loaded;
    use super::super::test_support::tiny_config;
    use super::super::train::Objective;
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

    /// Gradient check on the tiny fixture at the initial point.
    ///
    /// The autodiff gradient of every factor parameter is compared against a
    /// central difference of **v0's own `Objective::loss`** — the same
    /// function the finite-difference trainer descends — so a disagreement
    /// means the two trainers do not share an objective, which is the thing
    /// worth catching.
    ///
    /// Tolerance. The forward is f32 (the decoder's dtype), so the loss
    /// carries roughly 1e-6 of absolute noise; a central difference divides
    /// that by `2h`, giving about 5e-5 of noise floor at `h = 0.01`, plus an
    /// `O(h^2)` truncation term. The bound below is
    /// `1e-4 + 0.005 * |finite difference|`: absolute where the gradient is
    /// near zero (noise-dominated), relative where it is not
    /// (truncation-dominated). Observed max error on this fixture is 7.8e-7,
    /// two orders inside the absolute term, so the bound is headroom for f32
    /// rather than a threshold the run sits against — and it is far too
    /// tight to pass a wrong gradient, since a transposed factor or a
    /// dropped `scale` is off by whole factors, not by half a percent.
    #[test]
    fn autodiff_gradient_matches_v0_central_difference() {
        let config = tiny_config();
        let inner = Device::ndarray();
        let train_device = inner.clone().autodiff();
        let settings = settings();
        let module = settings.target_modules[0].clone();
        let (in_features, out_features) = dimensions(&config, &module).unwrap();
        let rank = usize::from(settings.rank);

        // One shared expected token and one shared length: v0's Objective
        // only speaks that dialect, and the point here is the gradient.
        let batch_ids: Vec<i32> = vec![1, 5, 9, 2, 7, 3, 11, 4, 6, 8];
        let (batch, seq) = (2usize, 5usize);
        let expected = 7usize;

        let base_inner = detach_base(det_loaded(&config, &inner));
        let objective = Objective {
            config: &config,
            base: &base_inner,
            target_module: &module,
            rank,
            in_features,
            out_features,
            scale: settings.alpha / f32::from(settings.rank),
            batch_ids: batch_ids.clone(),
            batch,
            seq,
            expected_id: expected,
            device: inner.clone(),
        };

        let factors = LoraFactorSet::init(&config, &settings, &train_device).unwrap();
        let base_train = detach_base(det_loaded(&config, &train_device));
        let model = compose_model(&config, &base_train, &factors, &settings, &train_device);
        let batch_tensors = Batch {
            ids: Tensor::from_data(TensorData::new(batch_ids, [batch, seq]), &train_device),
            targets: Tensor::from_data(
                TensorData::new(vec![expected as i32; batch], [batch, 1]),
                &train_device,
            ),
            batch,
            seq,
        };
        let loss = batch_loss(&model, &batch_tensors);
        assert!(
            (scalar(&loss) - objective.loss(&factors.to_flat_params())).abs() < 1.0e-4,
            "the two trainers must agree on the loss before their gradients can be compared"
        );
        let grads = loss.backward();
        let autodiff: Vec<f32> = factors
            .slots
            .iter()
            .flat_map(|slot| {
                let mut values = Vec::new();
                for param in [&slot.a, &slot.b] {
                    values.extend(
                        param
                            .val()
                            .grad(&grads)
                            .expect("every LoRA factor must receive a gradient")
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap(),
                    );
                }
                values
            })
            .collect();

        let mut params = factors.to_flat_params();
        assert_eq!(autodiff.len(), params.len());
        let h = 0.01f32;
        let mut worst = 0.0f64;
        for index in 0..params.len() {
            let original = params[index];
            params[index] = original + h;
            let plus = objective.loss(&params);
            params[index] = original - h;
            let minus = objective.loss(&params);
            params[index] = original;
            let finite = (plus - minus) / (2.0 * f64::from(h));
            let error = (f64::from(autodiff[index]) - finite).abs();
            worst = worst.max(error);
            assert!(
                error <= 1.0e-4 + 0.005 * finite.abs(),
                "parameter {index}: autodiff {} vs central difference {finite}",
                autodiff[index]
            );
        }
        println!(
            "gradient check: {} parameters, max error {worst:e}",
            params.len()
        );
    }

    #[test]
    fn factor_set_reproduces_v0_initialization_for_one_module() {
        let config = tiny_config();
        let device = Device::ndarray().autodiff();
        let settings = settings();
        let factors = LoraFactorSet::init(&config, &settings, &device).unwrap();
        let (in_features, out_features) = dimensions(&config, &settings.target_modules[0]).unwrap();
        let rank = usize::from(settings.rank);
        for layer in 0..config.num_hidden_layers {
            assert_eq!(
                factors.slots[layer]
                    .a
                    .val()
                    .detach()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                det_init(rank * in_features, 9000 + layer),
                "layer {layer} A must start where v0 starts"
            );
            assert!(
                factors.slots[layer]
                    .b
                    .val()
                    .detach()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0),
                "layer {layer} B must start at zero so step zero is the base model"
            );
            assert_eq!(factors.slots[layer].b.val().dims(), [out_features, rank]);
        }
    }
}
