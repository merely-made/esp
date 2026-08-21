//! Wgpu ↔ NdArray parity and timing for the BERT model (burn brief
//! Lane 1, `bert-wgpu`). Weights are synthesized deterministically on the
//! host so both backends run the identical model — no model files needed.
//! Parity uses tiny dims; the `#[ignore]`d timing test uses real MiniLM
//! dims and should run in release:
//!
//! ```bash
//! cargo test -p sibylla --features bert-wgpu --release \
//!     -- --ignored timing --nocapture
//! ```

use burn::tensor::{Device, Int, Tensor, TensorData};

use super::config::{BertConfig, MINILM_L6_V2};
use super::loaded::{LoadedBert, LoadedBertLayer, LoadedEmbeddings};
use super::model::Pooling;

fn cpu_device() -> Device {
    Device::ndarray()
}
fn gpu_device() -> Device {
    Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0))
}

/// Deterministic pseudo-weights: smooth, small, salt-separated so no two
/// tensors are identical. Same values on every backend by construction.
fn det_vec(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + salt * 7919) as f32) * 0.618_034).sin() * 0.02)
        .collect()
}

fn t1(n: usize, salt: usize, dev: &Device) -> Tensor<1> {
    Tensor::from_data(TensorData::new(det_vec(n, salt), [n]), dev)
}

fn t2(a: usize, b: usize, salt: usize, dev: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(det_vec(a * b, salt), [a, b]), dev)
}

/// Deterministic sibling of `loaded.rs`'s test-only `synth_loaded`:
/// identical structure, but values come from `det_vec` instead of the
/// backend RNG so CPU and GPU builds carry the same weights.
fn det_loaded(config: &BertConfig, dev: &Device) -> LoadedBert {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let mut salt = 0usize;
    let mut next = || {
        salt += 1;
        salt
    };

    let embeddings = LoadedEmbeddings {
        word: t2(config.vocab_size, h, next(), dev),
        position: t2(config.max_position_embeddings, h, next(), dev),
        token_type: t2(config.type_vocab_size, h, next(), dev),
        ln_gamma: Tensor::<1>::ones([h], dev),
        ln_beta: Tensor::<1>::zeros([h], dev),
    };
    let layers = (0..config.num_hidden_layers)
        .map(|_| LoadedBertLayer {
            q_w: t2(h, h, next(), dev),
            q_b: t1(h, next(), dev),
            k_w: t2(h, h, next(), dev),
            k_b: t1(h, next(), dev),
            v_w: t2(h, h, next(), dev),
            v_b: t1(h, next(), dev),
            attn_out_w: t2(h, h, next(), dev),
            attn_out_b: t1(h, next(), dev),
            attn_ln_gamma: Tensor::<1>::ones([h], dev),
            attn_ln_beta: Tensor::<1>::zeros([h], dev),
            inter_w: t2(h, inter, next(), dev),
            inter_b: t1(inter, next(), dev),
            out_w: t2(inter, h, next(), dev),
            out_b: t1(h, next(), dev),
            out_ln_gamma: Tensor::<1>::ones([h], dev),
            out_ln_beta: Tensor::<1>::zeros([h], dev),
        })
        .collect();
    LoadedBert {
        config: config.clone(),
        embeddings,
        layers,
    }
}

fn tiny_config() -> BertConfig {
    BertConfig {
        vocab_size: 32,
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 16,
        type_vocab_size: 2,
        layer_norm_eps: 1.0e-12,
        hidden_act: "gelu".to_string(),
        pad_token_id: 0,
    }
}

/// Run a sentence forward on backend `B` over a fixed token batch.
fn sentence_on(config: &BertConfig, ids: &[Vec<i32>], dev: &Device) -> Vec<f32> {
    let model = det_loaded(config, dev).into_model(dev);
    let (batch, seq) = (ids.len(), ids[0].len());
    let flat: Vec<i32> = ids.iter().flatten().copied().collect();
    let input_ids: Tensor<2, Int> = Tensor::from_data(TensorData::new(flat, [batch, seq]), dev);
    model
        .forward_sentence(input_ids, Pooling::Mean, true)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

fn token_batch(config: &BertConfig, batch: usize, seq: usize) -> Vec<Vec<i32>> {
    (0..batch)
        .map(|b| {
            (0..seq)
                .map(|s| ((b * 31 + s * 7 + 1) % config.vocab_size) as i32)
                .collect()
        })
        .collect()
}

#[test]
fn bert_sentence_parity_ndarray_wgpu() {
    let cfg = tiny_config();
    let ids = token_batch(&cfg, 3, 5);
    let cpu = sentence_on(&cfg, &ids, &cpu_device());
    let gpu = sentence_on(&cfg, &ids, &gpu_device());
    assert_eq!(cpu.len(), gpu.len());
    let max_diff = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 2.0e-3,
        "cpu/gpu embeddings diverged: max_diff={max_diff}"
    );
}

/// CPU-vs-GPU timing at real MiniLM-L6 dims (384 hidden, 6 layers) with a
/// deterministic synthetic model. Includes device→host readback; the GPU
/// gets a warmup forward so kernel compilation is not billed. Model build
/// time is excluded from both numbers.
#[test]
#[ignore]
fn timing_bert_cpu_vs_gpu() {
    let cfg = MINILM_L6_V2.clone();
    for (batch, seq) in [(1usize, 32usize), (8, 64), (32, 128)] {
        let ids = token_batch(&cfg, batch, seq);

        let dev = cpu_device();
        let model = det_loaded(&cfg, &dev).into_model(&dev);
        let flat: Vec<i32> = ids.iter().flatten().copied().collect();
        let input: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(flat.clone(), [batch, seq]), &dev);
        let t = std::time::Instant::now();
        let _ = model
            .forward_sentence(input, Pooling::Mean, true)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let cpu_us = t.elapsed().as_micros();

        let dev = gpu_device();
        let model = det_loaded(&cfg, &dev).into_model(&dev);
        let input: Tensor<2, Int> = Tensor::from_data(TensorData::new(flat, [batch, seq]), &dev);
        let _warm = model
            .forward_sentence(input.clone(), Pooling::Mean, true)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let t = std::time::Instant::now();
        let _ = model
            .forward_sentence(input, Pooling::Mean, true)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let gpu_us = t.elapsed().as_micros();

        println!("batch={batch} seq={seq}: ndarray={cpu_us}us wgpu={gpu_us}us");
    }
}
