//! BERT feed-forward block: intermediate (Linear + GELU) + output
//! (Linear + residual + LayerNorm).
//!
//! Mirrors HuggingFace's `BertIntermediate` + `BertOutput` shape so
//! safetensors keys map directly: `intermediate.dense`, `output.dense`,
//! `output.LayerNorm`.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::{Tensor, backend::Backend};

use super::config::BertConfig;

/// Expansion projection: hidden_size → intermediate_size, GELU.
#[derive(Module, Debug)]
pub struct BertIntermediate<B: Backend> {
    dense: Linear<B>,
}

impl<B: Backend> BertIntermediate<B> {
    pub fn new(config: &BertConfig, device: &B::Device) -> Self {
        Self {
            dense: LinearConfig::new(config.hidden_size, config.intermediate_size).init(device),
        }
    }

    pub fn from_loaded(loaded: &super::loaded::LoadedBertLayer<B>, device: &B::Device) -> Self {
        Self {
            dense: super::construct::linear_from_loaded(
                loaded.inter_w.clone(),
                loaded.inter_b.clone(),
                device,
            ),
        }
    }

    /// `gelu(dense(x))`. Input `[B, S, hidden]`, output `[B, S, intermediate]`.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        gelu(self.dense.forward(hidden))
    }
}

/// Contraction projection + residual + LayerNorm:
/// intermediate_size → hidden_size, then add+norm.
#[derive(Module, Debug)]
pub struct BertOutput<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> BertOutput<B> {
    pub fn new(config: &BertConfig, device: &B::Device) -> Self {
        Self {
            dense: LinearConfig::new(config.intermediate_size, config.hidden_size).init(device),
            layer_norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
        }
    }

    pub fn from_loaded(
        config: &BertConfig,
        loaded: &super::loaded::LoadedBertLayer<B>,
        device: &B::Device,
    ) -> Self {
        Self {
            dense: super::construct::linear_from_loaded(
                loaded.out_w.clone(),
                loaded.out_b.clone(),
                device,
            ),
            layer_norm: super::construct::layer_norm_from_loaded(
                loaded.out_ln_gamma.clone(),
                loaded.out_ln_beta.clone(),
                config.layer_norm_eps,
                device,
            ),
        }
    }

    /// `LayerNorm(dense(intermediate) + residual)`.
    /// `intermediate`: `[B, S, intermediate]`, `residual`: `[B, S, hidden]`.
    pub fn forward(&self, intermediate: Tensor<B, 3>, residual: Tensor<B, 3>) -> Tensor<B, 3> {
        let projected = self.dense.forward(intermediate);
        self.layer_norm.forward(projected + residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::bert::config::MINILM_L6_V2;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn config() -> BertConfig {
        let mut c = MINILM_L6_V2.clone();
        c.hidden_act = "gelu".to_string();
        c
    }

    fn rand(shape: [usize; 3]) -> Tensor<B, 3> {
        let device = Default::default();
        Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device)
    }

    #[test]
    fn intermediate_expands_to_intermediate_size() {
        let device = Default::default();
        let cfg = config();
        let layer: BertIntermediate<B> = BertIntermediate::new(&cfg, &device);
        let input = rand([2, 8, cfg.hidden_size]);
        let out = layer.forward(input);
        assert_eq!(out.dims(), [2, 8, cfg.intermediate_size]);
    }

    #[test]
    fn output_contracts_to_hidden_size() {
        let device = Default::default();
        let cfg = config();
        let layer: BertOutput<B> = BertOutput::new(&cfg, &device);
        let intermediate = rand([2, 8, cfg.intermediate_size]);
        let residual = rand([2, 8, cfg.hidden_size]);
        let out = layer.forward(intermediate, residual);
        assert_eq!(out.dims(), [2, 8, cfg.hidden_size]);
    }

    #[test]
    fn intermediate_no_nans() {
        let device = Default::default();
        let layer: BertIntermediate<B> = BertIntermediate::new(&config(), &device);
        let input = rand([1, 4, 384]);
        let out = layer.forward(input);
        let v = out.into_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| !x.is_nan() && !x.is_infinite()));
    }
}
