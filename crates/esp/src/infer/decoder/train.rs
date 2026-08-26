//! Deterministic PEFT LoRA training for the llama-family decoder (v0).
//!
//! This is the tensor-execution half of the trainer forcing receipt
//! (distillery v0 plan §9): full-batch central finite differences under a
//! backtracking line search over the LoRA factors of one attention
//! projection, with every loss read from the real decoder forward and the
//! delta composed by the same operation the adapter loader applies. It stops
//! at ESP's boundary: bytes in, bytes out. Eidetic owns the artifacts a
//! caller makes from the result, and Mesh owns any job carrying a run.

use burn::tensor::{Device, Int, Tensor, TensorData};
use eidetic::models::EvalTally;
use safetensors::tensor::{Dtype, TensorView};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::config::DecoderConfig;
use super::loader::load_decoder_tensors_from_bytes;
use super::lora::{add_delta, dimensions};
use super::model::{DecoderModel, LoadedDecoder};
use super::provider::DecoderProvider;
use crate::infer::provider::InferError;

/// PEFT version stamped into every `adapter_config.json` this trainer emits.
pub const TRAINED_PEFT_VERSION: &str = "esp-trainer-v0";
/// The manifest `adapter_format_version` matching [`TRAINED_PEFT_VERSION`],
/// in the form the adapter loader checks.
pub const TRAINED_ADAPTER_FORMAT_VERSION: &str = "peft-esp-trainer-v0";

/// One supervised next-token case: a rendered prompt and its expected token.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingCase {
    /// The full rendered prompt whose next token is supervised.
    pub prompt: String,
    /// The tokenizer token expected immediately after the prompt.
    pub expected_token: String,
}

/// Explicit hyperparameters for the v0 finite-difference LoRA trainer.
///
/// There is deliberately no `Default`: every value is part of the trainer's
/// deterministic identity and belongs in the published `training_method`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraTrainerSettings {
    /// Low-rank dimension of the trained factors.
    pub rank: u16,
    /// LoRA alpha; the applied scale is `alpha / rank`.
    pub alpha: f32,
    /// The single llama attention projection carrying the adapter.
    pub target_module: String,
    /// Full-batch descent steps.
    pub steps: u32,
    /// First trial length for the backtracking line search.
    pub initial_step_length: f32,
    /// Smallest trial length before a step is skipped.
    pub minimum_step_length: f32,
    /// Central-difference probe width.
    pub epsilon: f32,
}

impl LoraTrainerSettings {
    fn validate(&self) -> Result<(), InferError> {
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
        for (label, value) in [
            ("initial_step_length", self.initial_step_length),
            ("minimum_step_length", self.minimum_step_length),
            ("epsilon", self.epsilon),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(InferError::InvalidConfig(format!(
                    "trainer {label} must be finite and > 0"
                )));
            }
        }
        if self.initial_step_length < self.minimum_step_length {
            return Err(InferError::InvalidConfig(
                "trainer initial step length is below the minimum".into(),
            ));
        }
        Ok(())
    }
}

/// The trained adapter artifacts and the loss trajectory endpoints.
pub struct TrainedLoraAdapter {
    /// PEFT-named safetensors bytes for the trained factors.
    pub adapter_safetensors: Vec<u8>,
    /// The matching PEFT `adapter_config.json` bytes.
    pub adapter_config_json: Vec<u8>,
    /// Training loss before the first step.
    pub initial_loss: f64,
    /// Training loss after the last accepted step.
    pub trained_loss: f64,
}

/// Deterministic non-zero start for the A factors; B starts at zero so the
/// step-zero model is exactly the base model.
fn det_init(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + salt * 7919) as f32) * 0.618_034).sin() * 0.1)
        .collect()
}

struct Objective<'a> {
    config: &'a DecoderConfig,
    base: &'a LoadedDecoder,
    target_module: &'a str,
    rank: usize,
    in_features: usize,
    out_features: usize,
    scale: f32,
    batch_ids: Vec<i32>,
    batch: usize,
    seq: usize,
    expected_id: usize,
    device: Device,
}

impl Objective<'_> {
    fn factors(&self, params: &[f32], layer: usize) -> (Tensor<2>, Tensor<2>) {
        let per_layer = self.rank * (self.in_features + self.out_features);
        let offset = layer * per_layer;
        let a_len = self.rank * self.in_features;
        let a = Tensor::from_data(
            TensorData::new(
                params[offset..offset + a_len].to_vec(),
                [self.rank, self.in_features],
            ),
            &self.device,
        );
        let b = Tensor::from_data(
            TensorData::new(
                params[offset + a_len..offset + per_layer].to_vec(),
                [self.out_features, self.rank],
            ),
            &self.device,
        );
        (a, b)
    }

    /// Mean next-token cross-entropy of the expected token, through the real
    /// decoder forward with the loader's own delta composition.
    fn loss(&self, params: &[f32]) -> f64 {
        let mut loaded = self.base.clone();
        for (layer, entry) in loaded.layers.iter_mut().enumerate() {
            let (a, b) = self.factors(params, layer);
            match self.target_module {
                "q_proj" => entry.q_w = add_delta(entry.q_w.clone(), a, b, self.scale),
                "k_proj" => entry.k_w = add_delta(entry.k_w.clone(), a, b, self.scale),
                "v_proj" => entry.v_w = add_delta(entry.v_w.clone(), a, b, self.scale),
                "o_proj" => entry.o_w = add_delta(entry.o_w.clone(), a, b, self.scale),
                other => unreachable!("dimensions() validated target module {other:?}"),
            }
        }
        let model = DecoderModel::from_loaded(self.config.clone(), loaded, &self.device);
        let logits = model
            .logits(
                Tensor::<2, Int>::from_data(
                    TensorData::new(self.batch_ids.clone(), [self.batch, self.seq]),
                    &self.device,
                ),
                0,
            )
            .into_data()
            .to_vec::<f32>()
            .expect("decoder logits are dense f32");
        let (vocab, seq) = (self.config.vocab_size, self.seq);
        let mut loss = 0.0f64;
        for case in 0..self.batch {
            let row = &logits[(case * seq + (seq - 1)) * vocab..(case * seq + seq) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let log_sum_exp = row
                .iter()
                .map(|&logit| (f64::from(logit) - max).exp())
                .sum::<f64>()
                .ln()
                + max;
            loss += log_sum_exp - f64::from(row[self.expected_id]);
        }
        loss / self.batch as f64
    }
}

/// Train one PEFT LoRA adapter over the supplied base artifact triple.
///
/// Every case must tokenize to the same length and share one expected token;
/// the shared token is what makes a rank-limited low-rank delta a coherent
/// v0 objective. The result reproduces exactly on one host for one input
/// set; no cross-device bit claim is made.
pub fn train_peft_lora(
    config_bytes: &[u8],
    tokenizer_bytes: &[u8],
    base_weights: &[u8],
    model_id: &str,
    cases: &[TrainingCase],
    settings: &LoraTrainerSettings,
    device: &Device,
) -> Result<TrainedLoraAdapter, InferError> {
    settings.validate()?;
    if cases.is_empty() {
        return Err(InferError::InvalidRequest(
            "trainer received no training cases".into(),
        ));
    }
    let config = DecoderConfig::from_json_bytes(config_bytes)?;
    let (in_features, out_features) = dimensions(&config, &settings.target_module)?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|error| InferError::InvalidConfig(format!("tokenizer.json parse: {error}")))?;
    let base = load_decoder_tensors_from_bytes(&config, base_weights, device)?;

    let expected_id = expected_id(&tokenizer, &cases[0].expected_token)?;
    let mut seq = 0usize;
    let mut batch_ids = Vec::new();
    for case in cases {
        if case.expected_token != cases[0].expected_token {
            return Err(InferError::InvalidRequest(
                "v0 trainer cases must share one expected token".into(),
            ));
        }
        let encoding = tokenizer
            .encode(case.prompt.as_str(), true)
            .map_err(|error| InferError::InvalidRequest(format!("tokenize: {error}")))?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(InferError::InvalidRequest("empty training prompt".into()));
        }
        if seq == 0 {
            seq = ids.len();
        } else if ids.len() != seq {
            return Err(InferError::InvalidRequest(
                "v0 trainer cases must tokenize to one shared length".into(),
            ));
        }
        batch_ids.extend(ids.iter().map(|&id| id as i32));
    }

    let rank = usize::from(settings.rank);
    let objective = Objective {
        config: &config,
        base: &base,
        target_module: &settings.target_module,
        rank,
        in_features,
        out_features,
        scale: settings.alpha / f32::from(settings.rank),
        batch_ids,
        batch: cases.len(),
        seq,
        expected_id,
        device: device.clone(),
    };

    let per_layer = rank * (in_features + out_features);
    let mut params = vec![0.0f32; config.num_hidden_layers * per_layer];
    for layer in 0..config.num_hidden_layers {
        let a = det_init(rank * in_features, 9000 + layer);
        params[layer * per_layer..layer * per_layer + a.len()].copy_from_slice(&a);
    }

    let initial_loss = objective.loss(&params);
    let mut current_loss = initial_loss;
    let mut step_length = settings.initial_step_length;
    for step in 0..settings.steps {
        let mut gradient = vec![0.0f32; params.len()];
        for j in 0..params.len() {
            let original = params[j];
            params[j] = original + settings.epsilon;
            let plus = objective.loss(&params);
            params[j] = original - settings.epsilon;
            let minus = objective.loss(&params);
            params[j] = original;
            gradient[j] = ((plus - minus) / (2.0 * f64::from(settings.epsilon))) as f32;
        }
        let norm = gradient
            .iter()
            .map(|g| f64::from(*g) * f64::from(*g))
            .sum::<f64>()
            .sqrt() as f32;
        if norm == 0.0 {
            return Err(InferError::Backend(format!(
                "training gradient vanished at step {step}"
            )));
        }
        let mut trial = step_length * 2.0;
        while trial >= settings.minimum_step_length {
            let candidate: Vec<f32> = params
                .iter()
                .zip(gradient.iter())
                .map(|(param, g)| param - trial * g / norm)
                .collect();
            let candidate_loss = objective.loss(&candidate);
            if candidate_loss < current_loss {
                params = candidate;
                current_loss = candidate_loss;
                step_length = trial;
                break;
            }
            trial *= 0.5;
        }
    }

    Ok(TrainedLoraAdapter {
        adapter_safetensors: serialize_adapter(
            &config,
            settings,
            &params,
            in_features,
            out_features,
        ),
        adapter_config_json: adapter_config_json(model_id, settings),
        initial_loss,
        trained_loss: current_loss,
    })
}

fn expected_id(tokenizer: &Tokenizer, token: &str) -> Result<usize, InferError> {
    tokenizer
        .token_to_id(token)
        .map(|id| id as usize)
        .ok_or_else(|| {
            InferError::InvalidRequest(format!("expected token {token:?} is not in the vocabulary"))
        })
}

fn serialize_adapter(
    config: &DecoderConfig,
    settings: &LoraTrainerSettings,
    params: &[f32],
    in_features: usize,
    out_features: usize,
) -> Vec<u8> {
    let rank = usize::from(settings.rank);
    let per_layer = rank * (in_features + out_features);
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for layer in 0..config.num_hidden_layers {
        let prefix = format!(
            "base_model.model.model.layers.{layer}.self_attn.{}",
            settings.target_module
        );
        let offset = layer * per_layer;
        let a_len = rank * in_features;
        buffers.push((
            format!("{prefix}.lora_A.weight"),
            vec![rank, in_features],
            params[offset..offset + a_len]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        ));
        buffers.push((
            format!("{prefix}.lora_B.weight"),
            vec![out_features, rank],
            params[offset + a_len..offset + per_layer]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        ));
    }
    let views: Vec<(&str, TensorView<'_>)> = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .expect("trained factor shapes are consistent"),
            )
        })
        .collect();
    safetensors::serialize(views, &None).expect("trained adapter serializes")
}

fn adapter_config_json(model_id: &str, settings: &LoraTrainerSettings) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "base_model_name_or_path": model_id,
        "peft_type": "LORA",
        "peft_version": TRAINED_PEFT_VERSION,
        "r": settings.rank,
        "lora_alpha": settings.alpha,
        "target_modules": [settings.target_module],
        "bias": "none",
    }))
    .expect("adapter config serializes")
}

/// 1-based rank of the expected token in one next-token logit row; ties
/// resolve in the expected token's favor, deterministically.
pub fn expected_token_rank(logits: &[f32], expected_id: usize) -> usize {
    1 + logits
        .iter()
        .filter(|&&logit| logit > logits[expected_id])
        .count()
}

/// Count how many cases place their expected token within `limit` by
/// next-token logit rank.
pub fn ranking_tally(
    provider: &DecoderProvider,
    tokenizer_bytes: &[u8],
    cases: &[TrainingCase],
    limit: u32,
) -> Result<EvalTally, InferError> {
    if limit == 0 {
        return Err(InferError::InvalidRequest(
            "ranking limit must be > 0".into(),
        ));
    }
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|error| InferError::InvalidConfig(format!("tokenizer.json parse: {error}")))?;
    let mut passed = 0u64;
    for case in cases {
        let id = expected_id(&tokenizer, &case.expected_token)?;
        let logits = provider.next_token_logits(&case.prompt)?;
        if expected_token_rank(&logits, id) <= limit as usize {
            passed += 1;
        }
    }
    Ok(EvalTally {
        passed,
        total: cases.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> LoraTrainerSettings {
        LoraTrainerSettings {
            rank: 1,
            alpha: 8.0,
            target_module: "v_proj".into(),
            steps: 1,
            initial_step_length: 1.0,
            minimum_step_length: 1.0e-4,
            epsilon: 0.02,
        }
    }

    #[test]
    fn settings_reject_ambiguous_hyperparameters() {
        for (mutate, needle) in [
            (
                Box::new(|s: &mut LoraTrainerSettings| s.rank = 0) as Box<dyn Fn(&mut _)>,
                "rank",
            ),
            (
                Box::new(|s: &mut LoraTrainerSettings| s.alpha = 0.0),
                "alpha",
            ),
            (Box::new(|s: &mut LoraTrainerSettings| s.steps = 0), "steps"),
            (
                Box::new(|s: &mut LoraTrainerSettings| s.epsilon = f32::NAN),
                "epsilon",
            ),
            (
                Box::new(|s: &mut LoraTrainerSettings| s.initial_step_length = 1.0e-6),
                "below the minimum",
            ),
        ] {
            let mut broken = settings();
            mutate(&mut broken);
            let error = broken.validate().unwrap_err().to_string();
            assert!(error.contains(needle), "{error} should mention {needle}");
        }
    }

    #[test]
    fn ranks_resolve_ties_toward_the_expected_token() {
        assert_eq!(expected_token_rank(&[0.5, 0.5, 0.1], 1), 1);
        assert_eq!(expected_token_rank(&[0.9, 0.5, 0.7], 1), 3);
    }
}
