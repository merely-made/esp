// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! [`DecoderProvider`] — the decoder body behind the [`InferenceProvider`]
//! seam.
//!
//! Streaming detokenization decodes the full generated id sequence each
//! step and emits the string delta, holding a fragment back whenever the
//! decode is not a clean prefix extension (mid-byte BPE boundaries); the
//! held text arrives with the next token. Stop sequences truncate before
//! the completing fragment is emitted, matching `StubInferenceProvider`
//! semantics.
//!
//! Token choice per request: `temperature == 0.0` = greedy;
//! `temperature > 0` = seeded temperature/top-p sampling with the
//! seeding policy documented in [`super::sample`] (explicit seed =
//! reproducible; no seed = a fresh one is drawn and traced).

use std::ops::ControlFlow;

use tokenizers::Tokenizer;

use super::config::DecoderConfig;
use super::generate::{TokenPicker, generate_ids_with_async_controlled, try_generate_ids_with};
use super::loader::load_decoder_from_bytes;
use super::model::DecoderModel;
use super::sample::{Sampler, SplitMix64};
use crate::infer::provider::{GenerationRequest, InferError, InferenceProvider, ModelCapability};
use burn::tensor::{Device, Int, Tensor, TensorData};

/// Numerical observations for this provider's checkpoint generation calls.
/// Counts contain no prompt, token text or score values.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ScoreHealth {
    pub checked_rows: u64,
    pub invalid_rows: u64,
    pub prefix_recomputations: u64,
    pub failed_recoveries: u64,
}
#[derive(Default)]
struct ScoreCounters {
    checked_rows: std::sync::atomic::AtomicU64,
    invalid_rows: std::sync::atomic::AtomicU64,
    prefix_recomputations: std::sync::atomic::AtomicU64,
    failed_recoveries: std::sync::atomic::AtomicU64,
}
impl ScoreCounters {
    fn observe(&self, row: &[f32]) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        self.checked_rows.fetch_add(1, Relaxed);
        let invalid = row.iter().any(|v| !v.is_finite());
        if invalid {
            self.invalid_rows.fetch_add(1, Relaxed);
        }
        invalid
    }
}

/// A llama-family decoder wired to the `InferenceProvider` seam.
pub struct DecoderProvider {
    model: DecoderModel,
    tokenizer: Tokenizer,
    capability: ModelCapability,
    add_special_tokens: bool,
    eos_token_ids: Vec<u32>,
    score_counters: ScoreCounters,
    #[cfg(test)]
    score_faults: std::sync::Mutex<std::collections::BTreeMap<usize, Vec<f32>>>,
}

/// Decoder-specific details that the portable [`InferenceProvider`] seam does
/// not expose. Distillery's measurement harness uses the generated ids to
/// report token throughput rather than mistaking detokenized text fragments
/// for tokens.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DecoderGeneration {
    /// Complete emitted text, identical to the return from
    /// [`InferenceProvider::generate_streaming`].
    pub text: String,
    /// Generated token ids, excluding the prompt and EOS.
    pub token_ids: Vec<u32>,
    /// Number of non-empty detokenized fragments delivered to the caller.
    pub emitted_fragments: usize,
    /// Whether the caller's fragment callback requested an early stop.
    pub stopped_by_callback: bool,
    /// Whether an async host cancelled before the next token was emitted.
    pub cooperatively_cancelled: bool,
}

struct StreamObserver<'a> {
    provider: &'a DecoderProvider,
    request: &'a GenerationRequest,
    on_fragment: &'a mut dyn FnMut(&str) -> ControlFlow<()>,
    on_generated_token: &'a mut dyn FnMut(u32),
    generated_ids: Vec<u32>,
    emitted: String,
    emitted_fragments: usize,
    stopped_by_callback: bool,
    stopped_by_string: bool,
    decode_error: Option<InferError>,
}

impl Drop for StreamObserver<'_> {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.generated_ids.zeroize();
        self.emitted.zeroize();
    }
}

impl<'a> StreamObserver<'a> {
    fn new(
        provider: &'a DecoderProvider,
        request: &'a GenerationRequest,
        on_fragment: &'a mut dyn FnMut(&str) -> ControlFlow<()>,
        on_generated_token: &'a mut dyn FnMut(u32),
    ) -> Self {
        Self {
            provider,
            request,
            on_fragment,
            on_generated_token,
            generated_ids: Vec::new(),
            emitted: String::new(),
            emitted_fragments: 0,
            stopped_by_callback: false,
            stopped_by_string: false,
            decode_error: None,
        }
    }

    fn observe(&mut self, token: u32) -> ControlFlow<()> {
        (self.on_generated_token)(token);
        self.generated_ids.push(token);
        let full = match self.provider.decode(&self.generated_ids) {
            Ok(full) => full,
            Err(error) => {
                self.decode_error = Some(error);
                return ControlFlow::Break(());
            }
        };
        // Hold back non-prefix decodes (mid-byte BPE boundary); the text
        // arrives with a later token.
        if !full.starts_with(&self.emitted) || full.len() == self.emitted.len() {
            return ControlFlow::Continue(());
        }
        let stop_at = self
            .request
            .stop
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| full.find(stop))
            .min();
        self.stopped_by_string = stop_at.is_some();
        // Hold a suffix which could become a stop across later fragments.
        // Every prefix length is a UTF-8 boundary; content is never sliced
        // through a character.
        let withheld = self
            .request
            .stop
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .map(|(index, _)| index)
                    .filter(|index| *index > 0)
                    .filter(|index| full.ends_with(&stop[..*index]))
            })
            .max()
            .unwrap_or(0);
        let end = stop_at.unwrap_or(full.len() - withheld);
        if end > self.emitted.len() {
            let delta = &full[self.emitted.len()..end];
            self.stopped_by_callback = (self.on_fragment)(delta).is_break();
            self.emitted_fragments += 1;
            self.emitted = full[..end].to_string();
        }
        if self.stopped_by_callback || self.stopped_by_string {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn finish(
        mut self,
        token_ids: Vec<u32>,
        cooperatively_cancelled: bool,
    ) -> Result<DecoderGeneration, InferError> {
        if let Some(error) = self.decode_error.take() {
            return Err(error);
        }
        // At a normal token limit an incomplete stop prefix is ordinary text.
        // A cancellation never flushes held bytes after the stop boundary.
        if !self.stopped_by_callback && !self.stopped_by_string && !cooperatively_cancelled {
            let full = self.provider.decode(&self.generated_ids)?;
            if full.starts_with(&self.emitted) && full.len() > self.emitted.len() {
                self.stopped_by_callback =
                    (self.on_fragment)(&full[self.emitted.len()..]).is_break();
                self.emitted_fragments += 1;
                self.emitted = full;
            }
        }
        Ok(DecoderGeneration {
            text: std::mem::take(&mut self.emitted),
            token_ids,
            emitted_fragments: self.emitted_fragments,
            stopped_by_callback: self.stopped_by_callback,
            cooperatively_cancelled,
        })
    }
}

/// A corrupt backend row must never become a token decision. Rebuild the
/// exact committed prefix once per generation call; deterministic corruption
/// still fails closed. This does not alter checkpoints, grammar or RNG draws.
fn finite_score_row(
    row: Vec<f32>,
    recovered: &mut bool,
    recompute: impl FnOnce() -> Result<Vec<f32>, InferError>,
) -> Result<Vec<f32>, InferError> {
    if row.iter().all(|v| v.is_finite()) {
        return Ok(row);
    }
    if *recovered {
        return Err(InferError::Backend(
            "non-finite model scores after numerical recovery was already used".into(),
        ));
    }
    *recovered = true;
    let fresh = recompute()?;
    if fresh.len() != row.len() || fresh.iter().any(|v| !v.is_finite()) {
        return Err(InferError::Backend(
            "recomputed model scores are non-finite or have changed vocabulary size".into(),
        ));
    }
    Ok(fresh)
}

impl DecoderProvider {
    /// Owner-admitted dense layer payload retained only for this provider.
    pub fn with_layer_cache(mut self, budget: u64) -> Result<Self, InferError> {
        self.model = self.model.with_layer_cache(budget)?;
        Ok(self)
    }
    /// Keep all verified indexed layers resident under host admission.
    pub fn with_resident_layers(mut self) -> Result<Self, InferError> {
        self.model = self.model.with_resident_layers()?;
        Ok(self)
    }
    /// Use a session-bound remote layer chain with this provider's ordinary
    /// embedding, output head, tokenizer and checkpoint generation loop.
    pub fn with_stages(
        mut self,
        stages: Box<dyn super::stage::StageExecutor>,
    ) -> Result<Self, InferError> {
        self.model = self.model.with_stages(stages)?;
        Ok(self)
    }
    /// Assemble from already-built parts (tests; pre-loaded models).
    pub fn from_parts(
        model: DecoderModel,
        tokenizer: Tokenizer,
        model_id: impl Into<String>,
        loader: impl Into<String>,
    ) -> Self {
        let capability = ModelCapability {
            model_id: model_id.into(),
            context_window: model.config().max_position_embeddings,
            quantization: None,
            loader: loader.into(),
            streaming: true,
        };
        let eos_token_ids = model.config().eos_token_id.clone();
        Self {
            model,
            tokenizer,
            capability,
            add_special_tokens: true,
            eos_token_ids,
            score_counters: Default::default(),
            #[cfg(test)]
            score_faults: Default::default(),
        }
    }

    /// The declared template owns BOS/EOS and role tokens. Disable tokenizer
    /// postprocessing so it cannot silently prepend a second BOS.
    pub fn with_literal_prompt(mut self) -> Self {
        self.add_special_tokens = false;
        self
    }

    /// An exact template may declare an end-of-turn token absent from the
    /// base config (for example Qwen2's instruct generation defaults). It is
    /// session data, so finish before committing/emitting that control token.
    pub fn with_template_eos(mut self, eos: &str) -> Self {
        if let Some(id) = self.tokenizer.token_to_id(eos)
            && !self.eos_token_ids.contains(&id)
        {
            self.eos_token_ids.push(id);
        }
        self
    }

    /// Monotonic counts for score rows consumed by checkpoint generation.
    pub fn score_health(&self) -> ScoreHealth {
        use std::sync::atomic::Ordering::Relaxed;
        ScoreHealth {
            checked_rows: self.score_counters.checked_rows.load(Relaxed),
            invalid_rows: self.score_counters.invalid_rows.load(Relaxed),
            prefix_recomputations: self.score_counters.prefix_recomputations.load(Relaxed),
            failed_recoveries: self.score_counters.failed_recoveries.load(Relaxed),
        }
    }

    pub fn streaming_stats(&self) -> Option<super::streaming::StreamingReadStats> {
        self.model.streaming_stats()
    }

    pub fn shared_matrix_layout(&self) -> Option<super::model::SharedMatrixLayout> {
        self.model.shared_matrix_layout()
    }

    /// Build from the HF artifact triple as byte buffers — the shape
    /// eidetic's `ModelComponents` resolves to (P2), and what a model
    /// directory reads into.
    pub fn from_bytes(
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        weights_bytes: &[u8],
        model_id: impl Into<String>,
        loader: impl Into<String>,
        device: &Device,
    ) -> Result<Self, InferError> {
        let config = DecoderConfig::from_json_bytes(config_bytes)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| InferError::InvalidConfig(format!("tokenizer.json parse: {e}")))?;
        let model = load_decoder_from_bytes(&config, weights_bytes, device)?;
        Ok(Self::from_parts(model, tokenizer, model_id, loader))
    }

    /// Build a provider from the resident shared artifact and a source of
    /// independently verified layers. The ordinary tokenizer and generation
    /// checkpoint paths are used unchanged.
    pub fn from_layer_source(
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        shared_bytes: &[u8],
        source: Box<dyn super::streaming::LayerSource>,
        options: super::streaming::StreamingOptions,
        model_id: impl Into<String>,
        loader: impl Into<String>,
        device: &Device,
    ) -> Result<Self, InferError> {
        Self::from_layer_source_with_format(
            config_bytes,
            tokenizer_bytes,
            shared_bytes,
            source,
            options,
            model_id,
            loader,
            device,
            super::WeightFormat::F32,
        )
    }

    pub fn from_layer_source_with_format(
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        shared_bytes: &[u8],
        source: Box<dyn super::streaming::LayerSource>,
        options: super::streaming::StreamingOptions,
        model_id: impl Into<String>,
        loader: impl Into<String>,
        device: &Device,
        format: super::WeightFormat,
    ) -> Result<Self, InferError> {
        options.validate()?;
        if shared_bytes.len() > options.max_shared_bytes {
            return Err(InferError::InvalidWeights(
                "shared artifact exceeds its byte limit".into(),
            ));
        }
        let config = DecoderConfig::from_json_bytes(config_bytes)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|error| InferError::InvalidConfig(format!("tokenizer: {error}")))?;
        let tensors = safetensors::SafeTensors::deserialize(shared_bytes)
            .map_err(|error| InferError::InvalidWeights(format!("shared tensors: {error}")))?;
        if tensors.names().iter().any(|name| {
            !matches!(
                name.as_str(),
                "model.embed_tokens.weight" | "model.norm.weight" | "lm_head.weight"
            )
        }) {
            return Err(InferError::InvalidWeights(
                "unexpected tensor in shared artifact".into(),
            ));
        }
        format.validate_device(device)?;
        let model = if config.moe.is_some() {
            if format != super::WeightFormat::F32 {
                return Err(InferError::Unsupported {
                    feature: "MoE format".into(),
                    detail: "expert execution currently requires f32".into(),
                });
            }
            let shared = super::loader::load_shared_shards(
                &config,
                &tensors,
                device,
                options.max_layer_bytes,
            )?;
            DecoderModel::from_sharded_shared_source(config, shared, source, options, device)?
        } else {
            let shared = super::loader::load_decoder_shared_tensors_with_format(
                &config, &tensors, device, format,
            )?;
            DecoderModel::from_layer_source_with_format(
                config, shared, source, options, device, format,
            )?
        };
        let mut provider = Self::from_parts(model, tokenizer, model_id, loader);
        provider.capability.quantization =
            (format != super::WeightFormat::F32).then(|| format.id().into());
        Ok(provider)
    }

    /// Full vocabulary logits for the token immediately following a rendered
    /// prompt. This decoder-specific diagnostic supports numerical adapter and
    /// backend receipts without widening the portable provider trait.
    pub fn next_token_logits(&self, rendered_prompt: &str) -> Result<Vec<f32>, InferError> {
        if rendered_prompt.is_empty() {
            return Err(InferError::InvalidRequest("empty prompt".to_string()));
        }
        let encoding = self
            .tokenizer
            .encode(rendered_prompt, self.add_special_tokens)
            .map_err(|error| InferError::InvalidRequest(format!("tokenize: {error}")))?;
        let prompt_ids = encoding.get_ids();
        if prompt_ids.len() > self.capability.context_window {
            return Err(InferError::PromptTooLong {
                length: prompt_ids.len(),
                limit: self.capability.context_window,
            });
        }
        let input: Vec<i32> = prompt_ids.iter().map(|&token| token as i32).collect();
        let logits = self.model.try_forward_cached_last(
            Tensor::<2, Int>::from_data(
                TensorData::new(input, [1, prompt_ids.len()]),
                &self.model.device(),
            ),
            &mut self.model.new_cache(),
        )?;
        let [_, sequence, vocabulary] = logits.dims();
        logits
            .slice([0..1, (sequence - 1)..sequence, 0..vocabulary])
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|error| InferError::Backend(format!("decode next-token logits: {error}")))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, InferError> {
        self.tokenizer
            // A declared template's stop strings can themselves be special
            // tokens. Keep them visible to the stop matcher in literal mode.
            .decode(ids, self.add_special_tokens)
            .map_err(|e| InferError::Backend(format!("detokenize: {e}")))
    }

    fn prepare_generation(
        &self,
        request: &GenerationRequest,
    ) -> Result<(Vec<u32>, usize, TokenPicker), InferError> {
        if request.prompt.is_empty() {
            return Err(InferError::InvalidRequest("empty prompt".to_string()));
        }
        if request.max_tokens == 0 {
            return Err(InferError::InvalidRequest("max_tokens is 0".to_string()));
        }
        if request.temperature < 0.0 || !request.temperature.is_finite() {
            return Err(InferError::InvalidRequest(format!(
                "temperature must be finite and >= 0, got {}",
                request.temperature
            )));
        }
        let picker = if request.temperature == 0.0 {
            TokenPicker::Greedy
        } else {
            let seed = request.seed.unwrap_or_else(|| {
                let (_, drawn) = SplitMix64::from_entropy();
                // Surfaced so an unseeded run is still reproducible after
                // the fact: re-request with `seed: Some(this)`.
                tracing::debug!(target: "infer", seed = drawn, "sampling seed drawn");
                drawn
            });
            TokenPicker::Sampled(Sampler::new(request.temperature, request.top_p, seed))
        };

        let encoding = self
            .tokenizer
            .encode(request.prompt.as_str(), self.add_special_tokens)
            .map_err(|e| InferError::InvalidRequest(format!("tokenize: {e}")))?;
        let prompt_ids = encoding.get_ids().to_vec();
        let window = self.capability.context_window;
        if prompt_ids.len() >= window {
            return Err(InferError::PromptTooLong {
                length: prompt_ids.len(),
                limit: window,
            });
        }
        let max_new = request.max_tokens.min(window - prompt_ids.len());
        Ok((prompt_ids, max_new, picker))
    }

    /// Generate with decoder-specific token observation in addition to the
    /// portable text-fragment callback. `on_generated_token` is observational;
    /// cancellation retains the `InferenceProvider` callback semantics.
    pub fn generate_streaming_observed(
        &self,
        request: &GenerationRequest,
        on_fragment: &mut dyn FnMut(&str) -> ControlFlow<()>,
        on_generated_token: &mut dyn FnMut(u32),
    ) -> Result<DecoderGeneration, InferError> {
        if request.output_grammar.is_some() {
            let cancelled = std::cell::Cell::new(false);
            let mut observer = StreamObserver::new(self, request, on_fragment, on_generated_token);
            let mut observed = 0;
            let cp = self.generate_checkpointed(
                request,
                None,
                &mut |cp| {
                    for &id in &cp.token_ids[observed..] {
                        if observer.observe(id).is_break() {
                            cancelled.set(true);
                        }
                    }
                    observed = cp.token_ids.len();
                    Ok(())
                },
                &mut || cancelled.get(),
            )?;
            return observer.finish(cp.token_ids.clone(), false);
        }
        let (prompt_ids, max_new, mut picker) = self.prepare_generation(request)?;
        let mut observer = StreamObserver::new(self, request, on_fragment, on_generated_token);
        let token_ids = try_generate_ids_with(
            &self.model,
            &prompt_ids,
            max_new,
            &self.eos_token_ids,
            &mut picker,
            &mut |token| observer.observe(token),
        )?;
        observer.finish(token_ids, false)
    }

    /// Promise-backed counterpart to [`Self::generate_streaming_observed`].
    /// Use this for browser WebGPU, where reading a generated token must yield
    /// to the host event loop instead of blocking the worker.
    pub async fn generate_streaming_observed_async(
        &self,
        request: &GenerationRequest,
        on_fragment: &mut dyn FnMut(&str) -> ControlFlow<()>,
        on_generated_token: &mut dyn FnMut(u32),
    ) -> Result<DecoderGeneration, InferError> {
        self.generate_streaming_observed_async_controlled(
            request,
            on_fragment,
            on_generated_token,
            &mut || false,
        )
        .await
    }

    /// Promise-backed generation with cooperative cancellation checked before
    /// a newly read token crosses the observer boundary.
    pub async fn generate_streaming_observed_async_controlled(
        &self,
        request: &GenerationRequest,
        on_fragment: &mut dyn FnMut(&str) -> ControlFlow<()>,
        on_generated_token: &mut dyn FnMut(u32),
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<DecoderGeneration, InferError> {
        if request.output_grammar.is_some() {
            return Err(InferError::Unsupported {
                feature: "output_grammar".into(),
                detail: "async constrained decoding is not implemented".into(),
            });
        }
        let (prompt_ids, max_new, mut picker) = self.prepare_generation(request)?;
        let mut observer = StreamObserver::new(self, request, on_fragment, on_generated_token);
        let outcome = generate_ids_with_async_controlled(
            &self.model,
            &prompt_ids,
            max_new,
            &self.eos_token_ids,
            &mut picker,
            should_cancel,
            &mut |token| observer.observe(token),
        )
        .await
        .map_err(InferError::Backend)?;
        observer.finish(outcome.token_ids, outcome.cancelled)
    }
}

impl InferenceProvider for DecoderProvider {
    fn generate_checkpointed(
        &self,
        request: &GenerationRequest,
        resume: Option<&crate::infer::provider::GenerationCheckpoint>,
        commit: &mut dyn FnMut(
            &crate::infer::provider::GenerationCheckpoint,
        ) -> Result<(), InferError>,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<crate::infer::provider::GenerationCheckpoint, InferError> {
        use crate::infer::provider::GenerationCheckpoint;
        if resume.is_some_and(|cp| {
            let mut supplied = zeroize::Zeroizing::new(request.clone());
            if supplied.temperature > 0.0 && supplied.seed.is_none() {
                supplied.seed = cp.request.seed;
            }
            cp.request != *supplied
        }) {
            return Err(InferError::InvalidRequest(
                "checkpoint request differs from supplied request".into(),
            ));
        }
        let mut request = zeroize::Zeroizing::new(
            resume.map_or_else(|| request.clone(), |cp| cp.request.clone()),
        );
        if request.temperature > 0.0 && request.seed.is_none() {
            request.seed = Some(SplitMix64::from_entropy().1);
        }
        let (prompt_ids, max_new, mut picker) = self.prepare_generation(&request)?;
        let mut cp = resume.cloned().unwrap_or_else(|| GenerationCheckpoint {
            placements: Vec::new(),
            session: None,
            request: (*request).clone(),
            prompt_ids: prompt_ids.clone(),
            token_ids: Vec::new(),
            draws: 0,
            text: String::new(),
            complete: false,
        });
        let sampled = request.temperature > 0.0;
        if cp.prompt_ids != prompt_ids
            || cp.token_ids.len() > max_new
            || cp
                .token_ids
                .iter()
                .any(|&id| id as usize >= self.model.config().vocab_size)
            || (!cp.complete
                && cp.draws
                    != if sampled {
                        cp.token_ids.len() as u64
                    } else {
                        0
                    })
        {
            return Err(InferError::InvalidRequest(
                "invalid generation checkpoint tokens or draw position".into(),
            ));
        }
        let mut constraint = request
            .output_grammar
            .as_ref()
            .map(|grammar| {
                if !request.stop.is_empty() {
                    return Err(InferError::InvalidRequest(
                        "stop strings cannot truncate constrained output".into(),
                    ));
                }
                super::constraint::Constraint::new(
                    grammar,
                    &self.tokenizer,
                    self.model.config().vocab_size,
                )
            })
            .transpose()?;
        if let Some(grammar) = &mut constraint {
            for &id in &cp.token_ids {
                grammar.accept(id)?;
            }
            if cp.complete && (!grammar.complete() || cp.text != self.decode(&cp.token_ids)?) {
                return Err(InferError::InvalidRequest(
                    "complete checkpoint violates output grammar".into(),
                ));
            }
        }
        if cp.complete {
            return Ok(cp);
        }
        if let TokenPicker::Sampled(sampler) = &mut picker {
            sampler.skip_draws(cp.draws);
        }
        let mut discard = |_: &str| ControlFlow::Continue(());
        let mut unobserved = |_: u32| {};
        let mut observer = StreamObserver::new(self, &request, &mut discard, &mut unobserved);
        for &token in &cp.token_ids {
            let _ = observer.observe(token);
        }
        if observer.emitted != cp.text || observer.stopped_by_string {
            return Err(InferError::InvalidRequest(
                "checkpoint text or stop state does not match token IDs".into(),
            ));
        }
        commit(&cp)?;
        if stop() {
            return Ok(cp);
        }
        let mut prefix = prompt_ids;
        prefix.extend_from_slice(&cp.token_ids);
        if prefix.len() >= self.capability.context_window {
            return Err(InferError::PromptTooLong {
                length: prefix.len(),
                limit: self.capability.context_window,
            });
        }
        let device = self.model.device();
        let mut cache = self.model.new_cache();
        let input: Tensor<2, Int> = Tensor::from_data(
            TensorData::new(
                prefix.iter().map(|&token| token as i32).collect::<Vec<_>>(),
                [1, prefix.len()],
            ),
            &device,
        );
        let mut logits = self.model.try_forward_cached_last(input, &mut cache)?;
        let mut score_recovery_used = false;
        while cp.token_ids.len() < max_new {
            // No draw is spent on a paused, cancelled or stopped device.
            if stop() {
                return Ok(cp);
            }
            let forced = if let Some(grammar) = &constraint {
                grammar.forced(&self.tokenizer)?
            } else {
                vec![]
            };
            let tokens = if forced.is_empty() {
                let token = if let Some(grammar) = &constraint {
                    let [_, seq, vocab] = logits.dims();
                    let mut row = logits
                        .clone()
                        .slice([0..1, (seq - 1)..seq, 0..vocab])
                        .into_data()
                        .try_to_vec::<f32>()
                        .map_err(|e| InferError::Backend(e.to_string()))?;
                    #[cfg(test)]
                    if let Some(fault) = self
                        .score_faults
                        .lock()
                        .unwrap()
                        .remove(&cp.token_ids.len())
                    {
                        row = fault;
                    }
                    if row.iter().any(|value| !value.is_finite()) && stop() {
                        return Ok(cp);
                    }
                    let invalid = self.score_counters.observe(&row);
                    let checked = finite_score_row(row, &mut score_recovery_used, || {
                        self.score_counters
                            .prefix_recomputations
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Scores correspond to the next token after exactly
                        // these retained IDs. No token has been selected yet.
                        let mut ids = cp.prompt_ids.clone();
                        ids.extend_from_slice(&cp.token_ids);
                        tracing::warn!(
                            output_tokens = cp.token_ids.len(),
                            "non-finite decoder scores; recomputing committed prefix once"
                        );
                        cache = self.model.new_cache();
                        let input = Tensor::<2, Int>::from_data(
                            TensorData::new(
                                ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
                                [1, ids.len()],
                            ),
                            &device,
                        );
                        let fresh = self
                            .model
                            .try_forward_cached_last(input, &mut cache)?
                            .into_data()
                            .try_to_vec::<f32>()
                            .map_err(|error| {
                                InferError::Backend(format!("read recomputed scores: {error}"))
                            })?;
                        self.score_counters.observe(&fresh);
                        Ok(fresh)
                    });
                    if invalid && checked.is_err() {
                        self.score_counters
                            .failed_recoveries
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    row = checked?;
                    if stop() {
                        return Ok(cp);
                    }
                    grammar.mask(&mut row)?;
                    match &mut picker {
                        TokenPicker::Greedy => {
                            row.iter()
                                .enumerate()
                                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                                .unwrap()
                                .0 as u32
                        }
                        TokenPicker::Sampled(sampler) => sampler.sample(&row),
                    }
                } else {
                    picker.pick(&logits)
                };
                vec![token]
            } else {
                // Forced syntax consumes the same recorded sampling positions
                // on replay. It carries no model-selected semantic decision.
                if let TokenPicker::Sampled(sampler) = &mut picker {
                    sampler.skip_draws(forced.len() as u64);
                }
                forced
            };
            let mut evaluated = Vec::new();
            for token in tokens.into_iter().take(max_new - cp.token_ids.len()) {
                if stop() {
                    return Ok(cp);
                }
                cp.draws += u64::from(sampled);
                if self.eos_token_ids.contains(&token) {
                    cp.complete = true;
                } else {
                    if let Some(grammar) = &mut constraint {
                        grammar.accept(token)?;
                    }
                    cp.token_ids.push(token);
                    evaluated.push(token);
                    let _ = observer.observe(token);
                    if let Some(error) = observer.decode_error.take() {
                        return Err(error);
                    }
                    cp.complete = constraint.as_ref().map_or(
                        observer.stopped_by_string || cp.token_ids.len() == max_new,
                        |g| g.complete(),
                    );
                }
                cp.text = observer.emitted.clone();
                if cp.complete && !observer.stopped_by_string {
                    cp.text = self.decode(&cp.token_ids)?;
                }
                commit(&cp)?;
                if cp.complete || stop() {
                    return Ok(cp);
                }
            }
            if cp.token_ids.len() == max_new {
                break;
            }
            let step: Tensor<2, Int> = Tensor::from_data(
                TensorData::new(
                    evaluated.iter().map(|id| *id as i32).collect::<Vec<_>>(),
                    [1, evaluated.len()],
                ),
                &device,
            );
            logits = self.model.try_forward_cached_last(step, &mut cache)?;
        }
        if constraint.is_some() {
            return Err(InferError::InvalidRequest(
                "token limit reached before output grammar completed".into(),
            ));
        }

        cp.complete = true;
        commit(&cp)?;
        Ok(cp)
    }

    fn capability(&self) -> &ModelCapability {
        &self.capability
    }

    fn generate_streaming(
        &self,
        request: &GenerationRequest,
        on_token: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<String, InferError> {
        self.generate_streaming_observed(request, on_token, &mut |_| {})
            .map(|generation| generation.text)
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::tests::det_loaded;
    use super::super::test_support::tiny_config;
    use super::*;
    use burn::tensor::Device;

    // backend chosen per call site via Device

    /// A WordLevel tokenizer over the tiny vocab (`t0`..`t31`), built as
    /// real tokenizer.json bytes so the provider exercises the same
    /// parse path a HF checkpoint does.
    fn word_level_tokenizer() -> Tokenizer {
        let vocab: Vec<String> = (0..32).map(|i| format!("\"t{i}\": {i}")).collect();
        let json = format!(
            r#"{{
                "version": "1.0",
                "pre_tokenizer": {{ "type": "Whitespace" }},
                "model": {{
                    "type": "WordLevel",
                    "vocab": {{ {} }},
                    "unk_token": "t0"
                }}
            }}"#,
            vocab.join(", ")
        );
        Tokenizer::from_bytes(json.as_bytes()).expect("valid test tokenizer")
    }

    fn provider() -> DecoderProvider {
        let config = tiny_config();
        let dev = Device::ndarray();
        let model = DecoderModel::from_loaded(config.clone(), det_loaded(&config, &dev), &dev);
        DecoderProvider::from_parts(model, word_level_tokenizer(), "test/tiny", "burn-ndarray")
    }

    fn request(prompt: &str, max_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            ..Default::default()
        }
    }

    #[test]
    fn corrupt_scores_recompute_once_before_any_token_choice() {
        let mut used = false;
        let healthy = finite_score_row(vec![f32::NAN, f32::INFINITY], &mut used, || {
            Ok(vec![1., 2.])
        })
        .unwrap();
        assert!(used);
        assert_eq!(healthy, vec![1., 2.]);
        assert!(
            finite_score_row(vec![f32::NAN], &mut used, || panic!(
                "recovery budget must not be spent twice"
            ))
            .is_err()
        );
        let mut used = false;
        assert!(
            finite_score_row(vec![f32::NAN], &mut used, || Ok(vec![f32::NEG_INFINITY])).is_err()
        );
        assert!(used);
        let mut used = false;
        assert!(finite_score_row(vec![f32::NAN], &mut used, || Ok(vec![1., 2.])).is_err());
    }

    #[test]
    fn valid_scores_preserve_values_and_do_not_recompute() {
        let mut used = false;
        let scores = vec![-2., 0., 3.];
        assert_eq!(
            finite_score_row(scores.clone(), &mut used, || panic!(
                "finite rows must retain the ordinary execution path"
            ))
            .unwrap(),
            scores
        );
        assert!(!used);
    }

    #[test]
    fn numerical_recovery_preserves_committed_prefix_and_sampling_draws() {
        use crate::infer::OutputGrammar as G;
        let mut config = tiny_config();
        config.vocab_size = 256;
        config.max_position_embeddings = 128;
        let dev = Device::ndarray();
        let model = DecoderModel::from_loaded(config.clone(), det_loaded(&config, &dev), &dev);
        let provider = DecoderProvider::from_parts(
            model,
            super::super::constraint::tests::tokenizer(),
            "test/bytes",
            "burn-ndarray",
        );
        for temperature in [0., 0.7] {
            let mut r = request("x", 64);
            r.temperature = temperature;
            r.seed = Some(17);
            r.output_grammar = Some(G::Sequence {
                items: vec![
                    G::Literal {
                        value: "{\"ok\":".into(),
                    },
                    G::Choice {
                        alternatives: vec![
                            G::Literal {
                                value: "true}".into(),
                            },
                            G::Literal {
                                value: "false}".into(),
                            },
                        ],
                    },
                ],
            });
            let full = provider
                .generate_checkpointed(&r, None, &mut |_| Ok(()), &mut || false)
                .unwrap();
            provider
                .score_faults
                .lock()
                .unwrap()
                .insert(6, vec![f32::NAN; 256]);
            let recovered = provider
                .generate_checkpointed(&r, None, &mut |_| Ok(()), &mut || false)
                .unwrap();
            assert_eq!(full, recovered);
            assert!(provider.score_faults.lock().unwrap().is_empty());
        }
        let health = provider.score_health();
        assert_eq!(health.invalid_rows, 2);
        assert_eq!(health.prefix_recomputations, 2);
        assert_eq!(health.failed_recoveries, 0);
    }

    #[test]
    fn repeated_numerical_fault_stops_before_a_second_token_decision() {
        use crate::infer::OutputGrammar as G;
        let mut config = tiny_config();
        config.vocab_size = 256;
        config.max_position_embeddings = 128;
        let dev = Device::ndarray();
        let model = DecoderModel::from_loaded(config.clone(), det_loaded(&config, &dev), &dev);
        let provider = DecoderProvider::from_parts(
            model,
            super::super::constraint::tests::tokenizer(),
            "test/bytes",
            "burn-ndarray",
        );
        let mut r = request("x", 64);
        let choice = G::Choice {
            alternatives: vec![
                G::Literal {
                    value: "true".into(),
                },
                G::Literal {
                    value: "false".into(),
                },
            ],
        };
        r.output_grammar = Some(G::Sequence {
            items: vec![
                G::Literal {
                    value: "{\"first\":".into(),
                },
                choice.clone(),
                G::Literal {
                    value: ",\"second\":".into(),
                },
                choice,
                G::Literal { value: "}".into() },
            ],
        });
        let full = provider
            .generate_checkpointed(&r, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        let second = full.text.find("\"second\":").unwrap() + 9;
        provider
            .score_faults
            .lock()
            .unwrap()
            .extend([(9, vec![f32::NAN; 256]), (second, vec![f32::NAN; 256])]);
        let mut last = None;
        let result = provider.generate_checkpointed(
            &r,
            None,
            &mut |cp| {
                last = Some(cp.clone());
                Ok(())
            },
            &mut || false,
        );
        assert!(matches!(result, Err(InferError::Backend(_))));
        let last = last.unwrap();
        assert!(!last.complete);
        assert_eq!(last.token_ids, full.token_ids[..second]);
        assert_eq!(last.text, full.text[..second]);
        assert!(provider.score_faults.lock().unwrap().is_empty());
        let health = provider.score_health();
        assert_eq!(health.invalid_rows, 2);
        assert_eq!(health.prefix_recomputations, 1);
        assert_eq!(health.failed_recoveries, 1);
    }

    #[test]
    fn constrained_checkpoint_replay_and_truncation_fail_closed() {
        use crate::infer::OutputGrammar as G;
        let mut config = tiny_config();
        config.vocab_size = 256;
        config.max_position_embeddings = 128;
        let dev = Device::ndarray();
        let model = DecoderModel::from_loaded(config.clone(), det_loaded(&config, &dev), &dev);
        let provider = DecoderProvider::from_parts(
            model,
            super::super::constraint::tests::tokenizer(),
            "test/bytes",
            "burn-ndarray",
        );
        let mut r = request("x", 64);
        r.output_grammar = Some(G::Sequence {
            items: vec![
                G::Literal {
                    value: "{\"ok\":".into(),
                },
                G::Choice {
                    alternatives: vec![
                        G::Literal {
                            value: "true}".into(),
                        },
                        G::Literal {
                            value: "false}".into(),
                        },
                    ],
                },
            ],
        });
        let full = provider
            .generate_checkpointed(&r, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        assert!(full.complete);
        serde_json::from_str::<serde_json::Value>(&full.text).unwrap();
        let mut corrupt = full.clone();
        corrupt.text.push('x');
        assert!(
            provider
                .generate_checkpointed(&r, Some(&corrupt), &mut |_| Ok(()), &mut || false)
                .is_err()
        );
        let pause = std::cell::Cell::new(false);
        let partial = provider
            .generate_checkpointed(
                &r,
                None,
                &mut |cp| {
                    if cp.token_ids.len() == 4 {
                        pause.set(true);
                    }
                    Ok(())
                },
                &mut || pause.get(),
            )
            .unwrap();
        assert!(!partial.complete);
        let resumed = provider
            .generate_checkpointed(&r, Some(&partial), &mut |_| Ok(()), &mut || false)
            .unwrap();
        assert_eq!(resumed.text, full.text);
        assert_eq!(resumed.token_ids, full.token_ids);
        let mut changed = r.clone();
        changed.output_grammar = None;
        assert!(
            provider
                .generate_checkpointed(&changed, Some(&partial), &mut |_| Ok(()), &mut || false)
                .is_err()
        );
        r.temperature = 0.8;
        r.seed = Some(42);
        let sampled = provider
            .generate_checkpointed(&r, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        pause.set(false);
        let partial = provider
            .generate_checkpointed(
                &r,
                None,
                &mut |cp| {
                    if cp.token_ids.len() == 4 {
                        pause.set(true);
                    }
                    Ok(())
                },
                &mut || pause.get(),
            )
            .unwrap();
        let resumed = provider
            .generate_checkpointed(&r, Some(&partial), &mut |_| Ok(()), &mut || false)
            .unwrap();
        assert_eq!(sampled.token_ids, resumed.token_ids);
        assert_eq!(sampled.draws, resumed.draws);
        r.max_tokens = 2;
        assert!(
            provider
                .generate_checkpointed(&r, None, &mut |_| Ok(()), &mut || false)
                .is_err()
        );
    }

    #[test]
    fn template_end_token_finishes_before_committing_its_id() {
        let request = request("t1", 4);
        let ordinary = provider()
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        let first = *ordinary.token_ids.first().unwrap();
        let stopped = provider()
            .with_template_eos(&format!("t{first}"))
            .generate_checkpointed(&request, None, &mut |_| Ok(()), &mut || false)
            .unwrap();
        assert!(stopped.token_ids.is_empty());
        assert!(stopped.text.is_empty());
    }

    #[test]
    fn a_stop_across_fragments_never_leaks_its_prefix() {
        let provider = provider();
        let mut request = request("t1", 4);
        request.stop = vec!["t1 t2".into()];
        let mut text = String::new();
        let mut sink = |fragment: &str| {
            text.push_str(fragment);
            ControlFlow::Continue(())
        };
        let mut token = |_| {};
        let mut observer = StreamObserver::new(&provider, &request, &mut sink, &mut token);
        assert!(observer.observe(1).is_continue());
        assert!(observer.observe(2).is_break());
        assert_eq!(observer.finish(vec![1, 2], false).unwrap().text, "");
        assert_eq!(text, "");
    }

    #[test]
    fn a_normal_limit_flushes_an_incomplete_stop_but_cancellation_does_not() {
        let provider = provider();
        let mut request = request("t1", 4);
        request.stop = vec!["t1 t2".into()];
        for cancelled in [false, true] {
            let mut text = String::new();
            let mut sink = |fragment: &str| {
                text.push_str(fragment);
                ControlFlow::Continue(())
            };
            let mut token = |_| {};
            let mut observer = StreamObserver::new(&provider, &request, &mut sink, &mut token);
            assert!(observer.observe(1).is_continue());
            observer.finish(vec![1], cancelled).unwrap();
            assert_eq!(text, if cancelled { "" } else { "t1" });
        }
    }

    #[test]
    fn streams_deltas_that_concatenate_to_the_result() {
        let p = provider();
        let mut fragments = Vec::new();
        let full = p
            .generate_streaming(&request("t1 t5", 6), &mut |t| {
                fragments.push(t.to_string());
                ControlFlow::Continue(())
            })
            .unwrap();
        assert!(!full.is_empty());
        assert_eq!(fragments.concat(), full);
        // WordLevel vocab: every emitted word is one of ours.
        for word in full.split_whitespace() {
            assert!(word.starts_with('t'), "unexpected token text {word:?}");
        }
        // Deterministic (greedy).
        assert_eq!(p.generate(&request("t1 t5", 6)).unwrap(), full);
    }

    #[test]
    fn observed_generation_counts_model_tokens_not_text_fragments() {
        let p = provider();
        let mut observed = Vec::new();
        let generation = p
            .generate_streaming_observed(
                &request("t1 t5", 6),
                &mut |_| ControlFlow::Continue(()),
                &mut |token| observed.push(token),
            )
            .unwrap();
        assert_eq!(generation.token_ids, observed);
        assert_eq!(generation.token_ids.len(), 6);
        assert!(!generation.stopped_by_callback);
        assert!(!generation.cooperatively_cancelled);
        assert!(generation.emitted_fragments <= generation.token_ids.len());
    }

    #[test]
    fn async_control_cancels_before_the_next_observed_token() {
        use std::cell::Cell;

        let p = provider();
        let observed = Cell::new(0usize);
        let mut fragments = Vec::new();
        let generation = pollster::block_on(p.generate_streaming_observed_async_controlled(
            &request("t1 t5", 6),
            &mut |fragment| {
                fragments.push(fragment.to_string());
                ControlFlow::Continue(())
            },
            &mut |_| observed.set(observed.get() + 1),
            &mut || observed.get() == 1,
        ))
        .expect("controlled generation");
        assert!(generation.cooperatively_cancelled);
        assert!(!generation.stopped_by_callback);
        assert_eq!(generation.token_ids.len(), 1);
        assert_eq!(observed.get(), 1);
        assert_eq!(fragments.concat(), generation.text);
    }

    #[test]
    fn stop_sequence_truncates_before_emission() {
        let p = provider();
        let free = p.generate(&request("t1 t5", 4)).unwrap();
        let second_word = free.split_whitespace().nth(1).map(str::to_string);
        let Some(stop) = second_word else {
            panic!("test model generated fewer than 2 tokens: {free:?}");
        };
        let mut req = request("t1 t5", 4);
        req.stop = vec![stop.clone()];
        let stopped = p.generate(&req).unwrap();
        assert!(
            !stopped.contains(&stop),
            "stop sequence leaked: {stopped:?}"
        );
        assert!(free.starts_with(&stopped));
    }

    #[test]
    fn prompt_too_long_names_the_window() {
        let p = provider(); // context_window = 16
        let long: Vec<String> = (0..20).map(|i| format!("t{}", i % 32)).collect();
        let err = p.generate(&request(&long.join(" "), 4)).unwrap_err();
        assert!(matches!(
            err,
            InferError::PromptTooLong {
                length: 20,
                limit: 16
            }
        ));
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_unseeded_runs() {
        let p = provider();
        let mut req = request("t1 t5", 6);
        req.temperature = 0.8;
        req.top_p = Some(0.95);
        req.seed = Some(42);
        let a = p.generate(&req).unwrap();
        let b = p.generate(&req).unwrap();
        assert_eq!(a, b, "same seed must reproduce the stream");
        req.seed = None;
        assert!(p.generate(&req).is_ok(), "unseeded sampling draws a seed");
    }

    #[test]
    fn invalid_temperature_rejected() {
        let p = provider();
        for bad in [-0.5f32, f32::NAN, f32::INFINITY] {
            let mut req = request("t1", 4);
            req.temperature = bad;
            assert!(matches!(
                p.generate(&req).unwrap_err(),
                InferError::InvalidRequest(_)
            ));
        }
    }

    #[test]
    fn capability_reflects_model_and_loader() {
        let p = provider();
        let c = p.capability();
        assert_eq!(c.context_window, 16);
        assert_eq!(c.loader, "burn-ndarray");
        assert!(c.streaming);
    }
}
