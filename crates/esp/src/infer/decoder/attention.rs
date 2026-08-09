//! Grouped-query attention with rotary position encoding.
//!
//! The one llama-family piece burn-nn does not hand us whole: q/k/v/o
//! projections (bias-free), RoPE on q and k, key/value heads shared
//! across query-head groups, causal mask, softmax-weighted values. GQA
//! wiring informed by tracel-ai/models' llama-burn (MIT OR Apache-2.0),
//! rebuilt here on burn-nn primitives against the workspace burn pin.
//!
//! Weight convention matches `embed::bert`: `Linear` weights arrive as
//! `[in_features, out_features]`, already transposed from HF's
//! `[out, in]` at the extraction boundary.
//!
//! Two forward shapes share one implementation: the full-sequence
//! prefill (empty cache, square causal mask) and the cached single-token
//! decode step (queries attend everything cached, no mask needed). The
//! cached path is proven equal to naive full recompute by the
//! generation tests.

use burn::nn::{Linear, LinearConfig, RotaryEncoding};
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Tensor, backend::Backend};

use super::config::DecoderConfig;

/// Per-layer key/value cache: `[batch, kv_heads, seq_so_far, head_dim]`,
/// stored pre-GQA-expansion (memory-optimal; expansion happens per step).
#[derive(Debug, Default)]
pub struct LayerKvCache<B: Backend> {
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
}

impl<B: Backend> LayerKvCache<B> {
    /// Append this step's keys/values; returns the full cached pair.
    fn append(&mut self, k_new: Tensor<B, 4>, v_new: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = match self.k.take() {
            Some(prev) => Tensor::cat(vec![prev, k_new], 2),
            None => k_new,
        };
        let v = match self.v.take() {
            Some(prev) => Tensor::cat(vec![prev, v_new], 2),
            None => v_new,
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        (k, v)
    }

    fn cached_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }
}

/// Build a bias-free `Linear` from a pre-loaded `[in, out]` weight.
pub(crate) fn linear_no_bias_from_loaded<B: Backend>(
    weight: Tensor<B, 2>,
    device: &B::Device,
) -> Linear<B> {
    let [d_in, d_out] = weight.dims();
    let mut linear = LinearConfig::new(d_in, d_out).with_bias(false).init(device);
    linear.weight = burn::module::Param::from_tensor(weight);
    linear
}

/// One attention block of the decoder.
#[derive(Debug)]
pub struct DecoderAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> DecoderAttention<B> {
    /// Assemble from pre-loaded projection weights (`[in, out]`):
    /// `q/o: [hidden, hidden]`, `k/v: [hidden, kv_heads * head_dim]`.
    pub fn from_loaded(
        config: &DecoderConfig,
        q_w: Tensor<B, 2>,
        k_w: Tensor<B, 2>,
        v_w: Tensor<B, 2>,
        o_w: Tensor<B, 2>,
        device: &B::Device,
    ) -> Self {
        Self {
            q_proj: linear_no_bias_from_loaded(q_w, device),
            k_proj: linear_no_bias_from_loaded(k_w, device),
            v_proj: linear_no_bias_from_loaded(v_w, device),
            o_proj: linear_no_bias_from_loaded(o_w, device),
            n_heads: config.num_attention_heads,
            n_kv_heads: config.kv_heads(),
            head_dim: config.head_dim(),
        }
    }

    /// Full-sequence causal forward (prefill without retaining a cache).
    /// `x: [batch, seq, hidden]`; `rope` is the model-owned rotary table
    /// (one per model, shared by all layers); `start` is the absolute
    /// position of `x`'s first token.
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RotaryEncoding<B>, start: usize) -> Tensor<B, 3> {
        self.forward_cached(x, rope, &mut LayerKvCache::default(), start)
    }

    /// Cached causal forward: appends this step's keys/values to `cache`
    /// and attends the query block over everything cached. Two supported
    /// shapes — prefill (empty cache, any `seq`) and incremental decode
    /// (`seq == 1` over a warm cache); a multi-token block over a warm
    /// cache is rejected, since nothing in the generation loop produces
    /// it.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        cache: &mut LayerKvCache<B>,
        start: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq, hidden] = x.dims();
        let past = cache.cached_len();
        assert!(
            past == 0 || seq == 1,
            "multi-token block over a warm cache is unsupported (past={past}, seq={seq})"
        );
        let device = x.device();

        // Project and split heads: [b, s, n*hd] -> [b, n, s, hd].
        let split = |t: Tensor<B, 3>, n: usize| -> Tensor<B, 4> {
            t.reshape([batch, seq, n, self.head_dim]).swap_dims(1, 2)
        };
        let q = split(self.q_proj.forward(x.clone()), self.n_heads);
        let k = split(self.k_proj.forward(x.clone()), self.n_kv_heads);
        let v = split(self.v_proj.forward(x), self.n_kv_heads);

        // Rotate q and this step's k by absolute position; cached keys
        // were rotated when they were appended.
        let q = rope.apply(q, start);
        let k = rope.apply(k, start);

        let (k_all, v_all) = cache.append(k, v);
        let total = past + seq;

        // GQA: each kv head serves a contiguous group of query heads
        // (HF convention: q head i reads kv head i / group). Tile each kv
        // head `group` times along a fresh axis, then fold into the head
        // axis.
        let group = self.n_heads / self.n_kv_heads;
        let expand_kv = |t: Tensor<B, 4>| -> Tensor<B, 4> {
            if group == 1 {
                return t;
            }
            t.reshape([batch, self.n_kv_heads, 1, total * self.head_dim])
                .repeat_dim(2, group)
                .reshape([batch, self.n_heads, total, self.head_dim])
        };
        let k_all = expand_kv(k_all);
        let v_all = expand_kv(v_all);

        // Scaled dot-product; causal mask only when the query block has
        // more than one position (a single decode step attends its whole
        // past by construction).
        let mut scores = q
            .matmul(k_all.swap_dims(2, 3))
            .div_scalar((self.head_dim as f32).sqrt());
        if seq > 1 {
            let mask = Tensor::<B, 2, Bool>::tril_mask([seq, total], past as i64, &device)
                .unsqueeze::<4>()
                .expand([batch, self.n_heads, seq, total]);
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
        let attn = softmax(scores, 3);

        // Weighted values back to [b, s, hidden], through the output proj.
        let out = attn
            .matmul(v_all)
            .swap_dims(1, 2)
            .reshape([batch, seq, hidden]);
        self.o_proj.forward(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{t2, tiny_config};
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::RotaryEncodingConfig;

    type B = NdArray<f32>;

    fn rope(config: &DecoderConfig) -> RotaryEncoding<B> {
        RotaryEncodingConfig::new(config.max_position_embeddings, config.head_dim())
            .with_theta(config.rope_theta)
            .init(&Default::default())
    }

    fn attention(config: &DecoderConfig) -> DecoderAttention<B> {
        let dev = Default::default();
        let h = config.hidden_size;
        let kv = config.kv_heads() * config.head_dim();
        DecoderAttention::from_loaded(
            config,
            t2(h, h, 11, &dev),
            t2(h, kv, 12, &dev),
            t2(h, kv, 13, &dev),
            t2(h, h, 14, &dev),
            &dev,
        )
    }

    #[test]
    fn forward_preserves_shape() {
        let config = tiny_config();
        let attn = attention(&config);
        let x = t2::<B>(5, config.hidden_size, 99, &Default::default()).reshape([
            1,
            5,
            config.hidden_size,
        ]);
        let out = attn.forward(x, &rope(&config), 0);
        assert_eq!(out.dims(), [1, 5, config.hidden_size]);
    }

    /// The causal-mask probe: perturbing the LAST token must not change
    /// any earlier position's output. Catches both a missing mask and an
    /// inverted one.
    #[test]
    fn causal_mask_blocks_future_positions() {
        let config = tiny_config();
        let attn = attention(&config);
        let dev = Default::default();
        let h = config.hidden_size;

        let base = t2::<B>(5, h, 99, &dev).reshape([1, 5, h]);
        let perturbed = {
            let bump = Tensor::<B, 3>::ones([1, 1, h], &dev);
            let last = base.clone().slice([0..1, 4..5, 0..h]) + bump;
            Tensor::cat(vec![base.clone().slice([0..1, 0..4, 0..h]), last], 1)
        };

        let rope = rope(&config);
        let a = attn
            .forward(base, &rope, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = perturbed_out(&attn, perturbed, &rope);

        let per_pos = h;
        for pos in 0..4 {
            let (lo, hi) = (pos * per_pos, (pos + 1) * per_pos);
            assert!(
                a[lo..hi]
                    .iter()
                    .zip(&b[lo..hi])
                    .all(|(x, y)| (x - y).abs() < 1.0e-6),
                "position {pos} changed when only the future token moved"
            );
        }
        let (lo, hi) = (4 * per_pos, 5 * per_pos);
        assert!(
            a[lo..hi]
                .iter()
                .zip(&b[lo..hi])
                .any(|(x, y)| (x - y).abs() > 1.0e-6),
            "the perturbed position itself must change"
        );
    }

    fn perturbed_out(
        attn: &DecoderAttention<B>,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
    ) -> Vec<f32> {
        attn.forward(x, rope, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    /// GQA equivalence: a GQA attention (kv_heads < heads) must equal a
    /// plain MHA attention whose k/v weights duplicate each kv head's
    /// block once per group. Locks in the grouping convention.
    #[test]
    fn gqa_matches_mha_with_duplicated_kv_blocks() {
        let dev: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let mut gqa_cfg = tiny_config(); // hidden 8, 4 heads -> head_dim 2
        gqa_cfg.num_key_value_heads = Some(2);
        let mut mha_cfg = gqa_cfg.clone();
        mha_cfg.num_key_value_heads = Some(4);

        let h = gqa_cfg.hidden_size;
        let hd = gqa_cfg.head_dim();
        let kv_w_gqa = t2::<B>(h, 2 * hd, 21, &dev);
        let kv_v_gqa = t2::<B>(h, 2 * hd, 22, &dev);
        // Duplicate each kv head's [h, hd] column-block group-fold times:
        // kv0 kv0 kv1 kv1.
        let dup = |t: &Tensor<B, 2>| -> Tensor<B, 2> {
            let b0 = t.clone().slice([0..h, 0..hd]);
            let b1 = t.clone().slice([0..h, hd..2 * hd]);
            Tensor::cat(vec![b0.clone(), b0, b1.clone(), b1], 1)
        };

        let q_w = t2::<B>(h, h, 23, &dev);
        let o_w = t2::<B>(h, h, 24, &dev);
        let gqa = DecoderAttention::from_loaded(
            &gqa_cfg,
            q_w.clone(),
            kv_w_gqa.clone(),
            kv_v_gqa.clone(),
            o_w.clone(),
            &dev,
        );
        let mha =
            DecoderAttention::from_loaded(&mha_cfg, q_w, dup(&kv_w_gqa), dup(&kv_v_gqa), o_w, &dev);

        let x = t2::<B>(6, h, 30, &dev).reshape([1, 6, h]);
        let rope = rope(&gqa_cfg);
        let a = gqa
            .forward(x.clone(), &rope, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = mha
            .forward(x, &rope, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1.0e-5),
            "GQA and duplicated-kv MHA diverged"
        );
    }
}
