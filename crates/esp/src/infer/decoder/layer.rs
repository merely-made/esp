//! One decoder layer: pre-norm attention and pre-norm SwiGLU MLP, each
//! with a residual — the llama-family block, assembled from burn-nn's
//! `RmsNorm` + `SwiGlu` plus this crate's [`DecoderAttention`].
//!
//! HF weight-name mapping (applied by the loader slice; this module takes
//! tensors): `input_layernorm.weight` → `input_norm.gamma`,
//! `self_attn.{q,k,v,o}_proj.weight` → attention (transposed to
//! `[in, out]`), `post_attention_layernorm.weight` → `post_norm.gamma`,
//! `mlp.gate_proj` → SwiGlu inner (the silu side), `mlp.up_proj` → SwiGlu
//! outer, `mlp.down_proj` → `down`.

use burn::module::Param;
use burn::nn::{Linear, RmsNorm, RmsNormConfig, SwiGlu, SwiGluConfig};
use burn::tensor::{Device, Tensor};

use super::attention::{DecoderAttention, LlamaRotaryEncoding, linear_no_bias_from_loaded};
use super::config::DecoderConfig;

/// Build an `RmsNorm` whose gamma is the supplied tensor.
pub(crate) fn rms_norm_from_loaded(gamma: Tensor<1>, epsilon: f64, device: &Device) -> RmsNorm {
    let [size] = gamma.dims();
    let mut norm = RmsNormConfig::new(size).with_epsilon(epsilon).init(device);
    norm.gamma = Param::from_tensor(gamma);
    norm
}

/// Build a `SwiGlu` from pre-loaded gate/up weights (`[hidden, inter]`,
/// bias-free). Gate drives the silu side (`linear_inner`), up the
/// element-wise side (`linear_outer`) — the HF llama convention.
pub(crate) fn swiglu_from_loaded(gate_w: Tensor<2>, up_w: Tensor<2>, device: &Device) -> SwiGlu {
    let [d_in, d_out] = gate_w.dims();
    let mut swiglu = SwiGluConfig::new(d_in, d_out).with_bias(false).init(device);
    swiglu.linear_inner = linear_no_bias_from_loaded(gate_w, device);
    swiglu.linear_outer = linear_no_bias_from_loaded(up_w, device);
    swiglu
}

/// One transformer layer of the decoder.
#[derive(Debug)]
pub struct DecoderLayer {
    input_norm: RmsNorm,
    attention: DecoderAttention,
    post_norm: RmsNorm,
    mlp: SwiGlu,
    down: Linear,
}

/// The pre-loaded tensors for one layer, in Burn `[in, out]` convention.
#[derive(Clone)]
pub struct LoadedDecoderLayer {
    pub input_norm_gamma: Tensor<1>,
    pub q_w: Tensor<2>,
    pub k_w: Tensor<2>,
    pub v_w: Tensor<2>,
    pub o_w: Tensor<2>,
    pub post_norm_gamma: Tensor<1>,
    pub gate_w: Tensor<2>,
    pub up_w: Tensor<2>,
    pub down_w: Tensor<2>,
}

impl DecoderLayer {
    pub fn from_loaded(
        config: &DecoderConfig,
        loaded: LoadedDecoderLayer,
        device: &Device,
    ) -> Self {
        Self {
            input_norm: rms_norm_from_loaded(loaded.input_norm_gamma, config.rms_norm_eps, device),
            attention: DecoderAttention::from_loaded(
                config, loaded.q_w, loaded.k_w, loaded.v_w, loaded.o_w, device,
            ),
            post_norm: rms_norm_from_loaded(loaded.post_norm_gamma, config.rms_norm_eps, device),
            mlp: swiglu_from_loaded(loaded.gate_w, loaded.up_w, device),
            down: linear_no_bias_from_loaded(loaded.down_w, device),
        }
    }

    /// `x: [batch, seq, hidden]` → same shape.
    pub fn forward(&self, x: Tensor<3>, rope: &LlamaRotaryEncoding, start: usize) -> Tensor<3> {
        self.forward_cached(
            x,
            rope,
            &mut super::attention::LayerKvCache::default(),
            start,
        )
    }

    /// Cached variant: threads the layer's KV cache through attention.
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        rope: &LlamaRotaryEncoding,
        cache: &mut super::attention::LayerKvCache,
        start: usize,
    ) -> Tensor<3> {
        let h = x.clone()
            + self
                .attention
                .forward_cached(self.input_norm.forward(x), rope, cache, start);
        let mlp = self
            .down
            .forward(self.mlp.forward(self.post_norm.forward(h.clone())));
        h + mlp
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{t1_ones, t2, tiny_config};
    use super::*;

    // backend chosen per call site via Device

    pub(crate) fn det_layer(config: &DecoderConfig, salt: usize, device: &Device) -> DecoderLayer {
        let h = config.hidden_size;
        let kv = config.kv_heads() * config.head_dim();
        let inter = config.intermediate_size;
        DecoderLayer::from_loaded(
            config,
            LoadedDecoderLayer {
                input_norm_gamma: t1_ones(h, device),
                q_w: t2(h, h, salt + 1, device),
                k_w: t2(h, kv, salt + 2, device),
                v_w: t2(h, kv, salt + 3, device),
                o_w: t2(h, h, salt + 4, device),
                post_norm_gamma: t1_ones(h, device),
                gate_w: t2(h, inter, salt + 5, device),
                up_w: t2(h, inter, salt + 6, device),
                down_w: t2(inter, h, salt + 7, device),
            },
            device,
        )
    }

    #[test]
    fn layer_forward_shape_and_finite_and_deterministic() {
        let config = tiny_config();
        let dev = Device::ndarray();
        let layer = det_layer(&config, 40, &dev);
        let rope = LlamaRotaryEncoding::new(
            config.max_position_embeddings,
            config.head_dim(),
            config.rope_theta,
            &dev,
        );
        let x = t2(5, config.hidden_size, 99, &dev).reshape([1, 5, config.hidden_size]);

        let a = layer
            .forward(x.clone(), &rope, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(a.len(), 5 * config.hidden_size);
        assert!(a.iter().all(|v| v.is_finite()), "NaN/Inf in layer output");

        let b = layer
            .forward(x, &rope, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(a, b, "same input must produce identical output");
    }
}

#[cfg(all(test, feature = "decoder-wgpu"))]
mod tests_wgpu {
    use super::super::test_support::{det_vec, tiny_config};
    use super::*;
    use burn::tensor::TensorData;

    fn layer_out(config: &DecoderConfig, dev: &Device) -> Vec<f32> {
        let dev = dev.clone();
        let h = config.hidden_size;
        let kv = config.kv_heads() * config.head_dim();
        let inter = config.intermediate_size;
        let t2 = |a: usize, b: usize, salt: usize| -> Tensor<2> {
            Tensor::from_data(TensorData::new(det_vec(a * b, salt), [a, b]), &dev)
        };
        let layer = DecoderLayer::from_loaded(
            config,
            LoadedDecoderLayer {
                input_norm_gamma: Tensor::ones([h], &dev),
                q_w: t2(h, h, 41),
                k_w: t2(h, kv, 42),
                v_w: t2(h, kv, 43),
                o_w: t2(h, h, 44),
                post_norm_gamma: Tensor::ones([h], &dev),
                gate_w: t2(h, inter, 45),
                up_w: t2(h, inter, 46),
                down_w: t2(inter, h, 47),
            },
            &dev,
        );
        let rope = LlamaRotaryEncoding::new(
            config.max_position_embeddings,
            config.head_dim(),
            config.rope_theta,
            &dev,
        );
        let x: Tensor<3> = Tensor::from_data(TensorData::new(det_vec(5 * h, 99), [1, 5, h]), &dev);
        layer
            .forward(x, &rope, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn layer_parity_ndarray_wgpu() {
        let config = tiny_config();
        let cpu = layer_out(&config, &Device::ndarray());
        let gpu = layer_out(
            &config,
            &Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0)),
        );
        let max_diff = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 2.0e-3,
            "cpu/gpu layer outputs diverged: {max_diff}"
        );
    }
}
