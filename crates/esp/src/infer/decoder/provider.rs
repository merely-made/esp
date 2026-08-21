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
use super::generate::{TokenPicker, generate_ids_with};
use super::loader::load_decoder_from_bytes;
use super::model::DecoderModel;
use super::sample::{Sampler, SplitMix64};
use crate::infer::provider::{GenerationRequest, InferError, InferenceProvider, ModelCapability};
use burn::tensor::Device;

/// A llama-family decoder wired to the `InferenceProvider` seam.
pub struct DecoderProvider {
    model: DecoderModel,
    tokenizer: Tokenizer,
    capability: ModelCapability,
}

impl DecoderProvider {
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
        Self {
            model,
            tokenizer,
            capability,
        }
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

    fn decode(&self, ids: &[u32]) -> Result<String, InferError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| InferError::Backend(format!("detokenize: {e}")))
    }
}

impl InferenceProvider for DecoderProvider {
    fn capability(&self) -> &ModelCapability {
        &self.capability
    }

    fn generate_streaming(
        &self,
        request: &GenerationRequest,
        on_token: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<String, InferError> {
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
        let mut picker = if request.temperature == 0.0 {
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
            .encode(request.prompt.as_str(), true)
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

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        let mut decode_error: Option<InferError> = None;

        generate_ids_with(
            &self.model,
            &prompt_ids,
            max_new,
            &self.model.config().eos_token_id,
            &mut picker,
            &mut |token| {
                generated_ids.push(token);
                let full = match self.decode(&generated_ids) {
                    Ok(f) => f,
                    Err(e) => {
                        decode_error = Some(e);
                        return ControlFlow::Break(());
                    }
                };
                // Hold back non-prefix decodes (mid-byte BPE boundary);
                // the text arrives with a later token.
                if !full.starts_with(&emitted) || full.len() == emitted.len() {
                    return ControlFlow::Continue(());
                }
                if request.stop.iter().any(|s| full.contains(s.as_str())) {
                    // Truncate before the completing fragment, like
                    // StubInferenceProvider: nothing past the stop is emitted.
                    return ControlFlow::Break(());
                }
                let delta = full[emitted.len()..].to_string();
                let flow = on_token(&delta);
                emitted = full;
                flow
            },
        );

        if let Some(error) = decode_error {
            return Err(error);
        }
        Ok(emitted)
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
