//! Real-checkpoint validation for the decoder (inference plan P1's
//! payoff receipts). All `#[ignore]`d — they need `VATES_TINYLLAMA_DIR`
//! pointing at a directory with TinyLlama-1.1B-Chat-v1.0's `config.json`
//! / `tokenizer.json` / `model.safetensors` (HF layout, bf16). Run:
//!
//! ```bash
//! VATES_TINYLLAMA_DIR=C:/t/models/TinyLlama-1.1B-Chat-v1.0 \
//!     cargo test -p esp --features decoder-wgpu --release \
//!     --test tinyllama_real -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "decoder")]

use std::ops::ControlFlow;
use std::path::PathBuf;

use esp::infer::decoder::DecoderProvider;
use esp::infer::{GenerationRequest, InferenceProvider};


fn model_dir() -> Option<PathBuf> {
    std::env::var("VATES_TINYLLAMA_DIR").ok().map(PathBuf::from)
}

fn load_provider(device: burn::tensor::Device, loader: &str) -> DecoderProvider {
    let dir = model_dir().expect("VATES_TINYLLAMA_DIR must be set");
    let config = std::fs::read(dir.join("config.json")).expect("read config.json");
    let tokenizer = std::fs::read(dir.join("tokenizer.json")).expect("read tokenizer.json");
    let weights = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let t = std::time::Instant::now();
    let provider = DecoderProvider::from_bytes(
        &config,
        &tokenizer,
        &weights,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        loader,
        &device,
    )
    .expect("load TinyLlama");
    println!("model load ({loader}): {} ms", t.elapsed().as_millis());
    provider
}

fn request(prompt: &str, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_string(),
        max_tokens,
        ..Default::default()
    }
}

/// The semantic fixture: a knowledge-completion prompt whose greedy
/// continuation any competent checkpoint answers one way. Proves the
/// whole chain — bf16 load, name map, GQA, RoPE, KV cache, tokenizer —
/// is numerically right; garbage weights produce garbage tokens, not
/// "Paris".
#[test]
#[ignore = "requires VATES_TINYLLAMA_DIR (2.2GB checkpoint)"]
fn greedy_continuation_answers_paris() {
    let provider = load_provider(burn::tensor::Device::ndarray(), "burn-ndarray");
    let out = provider
        .generate(&request("The capital of France is", 8))
        .expect("generate");
    println!("continuation: {out:?}");
    assert!(
        out.contains("Paris"),
        "expected Paris in the greedy continuation, got {out:?}"
    );
}

/// Streaming deltas concatenate to the same text the collected call
/// returns, on the real BPE tokenizer (byte-level boundaries included).
#[test]
#[ignore = "requires VATES_TINYLLAMA_DIR (2.2GB checkpoint)"]
fn streaming_matches_collected_on_real_tokenizer() {
    let provider = load_provider(burn::tensor::Device::ndarray(), "burn-ndarray");
    let req = request("The capital of France is", 8);
    let mut fragments = Vec::new();
    let streamed = provider
        .generate_streaming(&req, &mut |t| {
            fragments.push(t.to_string());
            ControlFlow::Continue(())
        })
        .expect("generate");
    assert_eq!(fragments.concat(), streamed);
    assert_eq!(provider.generate(&req).expect("generate"), streamed);
}

/// Tokens/sec, CPU vs GPU: prefill + 16 greedy tokens on each backend.
#[cfg(feature = "decoder-wgpu")]
#[test]
#[ignore = "requires VATES_TINYLLAMA_DIR (2.2GB checkpoint) and a GPU"]
fn timing_tokens_per_second_cpu_vs_gpu() {
    let gpu_device = burn::tensor::Device::wgpu(burn::tensor::DeviceKind::default());
    let prompt = "The old lighthouse keeper climbed the stairs and";
    let tokens = 16usize;

    let cpu = load_provider(burn::tensor::Device::ndarray(), "burn-ndarray");
    let t = std::time::Instant::now();
    let out_cpu = cpu
        .generate(&request(prompt, tokens))
        .expect("cpu generate");
    let cpu_ms = t.elapsed().as_millis();

    let gpu = load_provider(gpu_device.clone(), "burn-wgpu");
    // Warmup: one short generation so kernel compilation is not billed.
    let _ = gpu.generate(&request(prompt, 2)).expect("gpu warmup");
    let t = std::time::Instant::now();
    let out_gpu = gpu
        .generate(&request(prompt, tokens))
        .expect("gpu generate");
    let gpu_ms = t.elapsed().as_millis();

    println!(
        "cpu: {tokens} tokens in {cpu_ms} ms ({:.2} tok/s): {out_cpu:?}",
        tokens as f64 * 1000.0 / cpu_ms as f64
    );
    println!(
        "gpu: {tokens} tokens in {gpu_ms} ms ({:.2} tok/s): {out_gpu:?}",
        tokens as f64 * 1000.0 / gpu_ms as f64
    );
    assert_eq!(out_cpu, out_gpu, "greedy output must match across backends");
}
