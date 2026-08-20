//! The token-level generation loop: prefill the prompt once, then
//! KV-cached single-token decode steps.
//!
//! Token choice is a [`TokenPicker`]: greedy argmax (deterministic, the
//! validation-fixture path) or seeded temperature/top-p sampling
//! ([`super::sample::Sampler`]). Both run over the same cached forward.

use std::ops::ControlFlow;

use burn::tensor::{Device, ElementConversion, Int, Tensor};

use super::model::DecoderModel;
use super::sample::Sampler;

/// How the next token is chosen from a logits block.
pub enum TokenPicker {
    /// Argmax — deterministic.
    Greedy,
    /// Seeded temperature/top-p sampling.
    Sampled(Sampler),
}

impl TokenPicker {
    fn pick(&mut self, logits: &Tensor<3>) -> u32 {
        match self {
            TokenPicker::Greedy => argmax_last(logits),
            TokenPicker::Sampled(sampler) => {
                let [_, seq, vocab] = logits.dims();
                let row = logits
                    .clone()
                    .slice([0..1, (seq - 1)..seq, 0..vocab])
                    .into_data()
                    .to_vec::<f32>()
                    .expect("logits row");
                sampler.sample(&row)
            }
        }
    }
}

/// Greedy-pick the token at the last position of a logits block.
/// `into_scalar().elem()` converts from the backend's int element type
/// (i64 on ndarray, i32 on the wgpu instantiation) instead of assuming
/// one.
fn argmax_last(logits: &Tensor<3>) -> u32 {
    let [batch, seq, vocab] = logits.dims();
    debug_assert_eq!(batch, 1, "generation is single-sequence");
    let last = logits
        .clone()
        .slice([0..1, (seq - 1)..seq, 0..vocab])
        .argmax(2);
    let id: i64 = last.into_scalar::<i64>();
    id as u32
}

/// Greedy [`generate_ids_with`].
pub fn generate_ids(
    model: &DecoderModel,
    prompt_ids: &[u32],
    max_new: usize,
    eos: &[u32],
    on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
) -> Vec<u32> {
    generate_ids_with(
        model,
        prompt_ids,
        max_new,
        eos,
        &mut TokenPicker::Greedy,
        on_token,
    )
}

/// Generate up to `max_new` tokens after `prompt_ids` with a KV cache,
/// choosing each token via `picker`. Each generated id goes through
/// `on_token` (return `Break` to stop after it); generation also stops
/// on any id in `eos`. Returns the generated ids (prompt excluded, eos
/// excluded).
pub fn generate_ids_with(
    model: &DecoderModel,
    prompt_ids: &[u32],
    max_new: usize,
    eos: &[u32],
    picker: &mut TokenPicker,
    on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
) -> Vec<u32> {
    assert!(!prompt_ids.is_empty(), "prompt must not be empty");
    let device = model.device();
    let mut cache = model.new_cache();

    let prompt: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
    let input: Tensor<2, Int> = Tensor::from_data(
        burn::tensor::TensorData::new(prompt, [1, prompt_ids.len()]),
        &device,
    );
    let mut logits = model.forward_cached(input, &mut cache);

    let mut out = Vec::new();
    for _ in 0..max_new {
        let token = picker.pick(&logits);
        if eos.contains(&token) {
            break;
        }
        out.push(token);
        if on_token(token).is_break() {
            break;
        }
        let step: Tensor<2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![token as i32], [1, 1]),
            &device,
        );
        logits = model.forward_cached(step, &mut cache);
    }
    out
}

/// Reference implementation for the tests: no cache, full recompute of
/// the whole sequence every step. Slow and simple on purpose.
#[cfg(test)]
pub(crate) fn generate_ids_uncached(
    model: &DecoderModel,
    prompt_ids: &[u32],
    max_new: usize,
    eos: &[u32],
) -> Vec<u32> {
    let device = model.device();
    let mut ids: Vec<u32> = prompt_ids.to_vec();
    let mut out = Vec::new();
    for _ in 0..max_new {
        let all: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let input: Tensor<2, Int> =
            Tensor::from_data(burn::tensor::TensorData::new(all, [1, ids.len()]), &device);
        let logits = model.logits(input, 0);
        let token = argmax_last(&logits);
        if eos.contains(&token) {
            break;
        }
        out.push(token);
        ids.push(token);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::model::DecoderModel;
    use super::super::model::tests::det_loaded;
    use super::super::test_support::tiny_config;
    use super::*;
    
    // backend chosen per call site via Device

    fn model() -> DecoderModel {
        let config = tiny_config();
        let dev = Device::ndarray();
        DecoderModel::from_loaded(config.clone(), det_loaded(&config, &dev), &dev)
    }

    /// THE correctness lock for the KV cache: cached incremental decode
    /// must produce exactly the same token sequence as naive full
    /// recompute.
    #[test]
    fn cached_generation_equals_full_recompute() {
        let m = model();
        let prompt = [1u32, 5, 9, 2];
        let cached = generate_ids(&m, &prompt, 8, &[], &mut |_| ControlFlow::Continue(()));
        let uncached = generate_ids_uncached(&m, &prompt, 8, &[]);
        assert_eq!(cached, uncached, "KV-cached decode diverged from naive");
        assert_eq!(cached.len(), 8);
    }

    #[test]
    fn generation_is_deterministic() {
        let m = model();
        let prompt = [3u32, 7];
        let a = generate_ids(&m, &prompt, 6, &[], &mut |_| ControlFlow::Continue(()));
        let b = generate_ids(&m, &prompt, 6, &[], &mut |_| ControlFlow::Continue(()));
        assert_eq!(a, b);
    }

    #[test]
    fn eos_stops_generation_without_emitting_it() {
        let m = model();
        let prompt = [1u32, 5, 9, 2];
        // Discover what the model would emit, then make its second token
        // the eos and expect exactly the first token back.
        let free = generate_ids(&m, &prompt, 3, &[], &mut |_| ControlFlow::Continue(()));
        let stopped = generate_ids(&m, &prompt, 3, &[free[1]], &mut |_| {
            ControlFlow::Continue(())
        });
        assert_eq!(stopped, vec![free[0]]);
    }

    #[test]
    fn seeded_sampled_generation_is_reproducible() {
        let m = model();
        let prompt = [1u32, 5];
        let run = |seed: u64| {
            generate_ids_with(
                &m,
                &prompt,
                6,
                &[],
                &mut TokenPicker::Sampled(Sampler::new(0.8, Some(0.95), seed)),
                &mut |_| ControlFlow::Continue(()),
            )
        };
        assert_eq!(run(42), run(42), "same seed, same stream");
        assert_eq!(run(42).len(), 6);
    }

    #[test]
    fn callback_break_stops_after_current_token() {
        let m = model();
        let prompt = [1u32, 5];
        let mut seen = Vec::new();
        let out = generate_ids(&m, &prompt, 8, &[], &mut |t| {
            seen.push(t);
            if seen.len() == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(out.len(), 2);
        assert_eq!(seen, out);
    }
}
