//! BERT encoder — a stack of [`BertLayer`]s.
//!
//! Mirrors HF's `BertEncoder` shape so safetensors `encoder.layer.{i}.…`
//! keys map to `layers[i]` directly when the loader slice lands.

use burn::module::Module;
use burn::tensor::{Device, Tensor};

use super::config::BertConfig;
use super::layer::BertLayer;

#[derive(Module, Debug)]
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| BertLayer::new(config, device))
            .collect();
        Self { layers }
    }

    pub fn from_loaded(
        config: &BertConfig,
        loaded_layers: &[super::loaded::LoadedBertLayer],
        device: &Device,
    ) -> Self {
        let layers = loaded_layers
            .iter()
            .map(|l| BertLayer::from_loaded(config, l, device))
            .collect();
        Self { layers }
    }

    /// Number of stacked layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward through the full stack. Input/output shape `[B, S, hidden]`.
    pub fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        self.layers
            .iter()
            .fold(hidden, |acc, layer| layer.forward(acc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::bert::config::MINILM_L6_V2;
    
    // backend chosen per call site via Device

    fn config() -> BertConfig {
        let mut c = MINILM_L6_V2.clone();
        c.hidden_act = "gelu".to_string();
        c
    }

    fn rand(shape: [usize; 3]) -> Tensor<3> {
        let device = Device::ndarray();
        Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device)
    }

    #[test]
    fn encoder_stacks_correct_number_of_layers() {
        let device = Device::ndarray();
        let encoder: BertEncoder = BertEncoder::new(&config(), &device);
        assert_eq!(encoder.num_layers(), 6);
    }

    #[test]
    fn encoder_returns_input_shape() {
        let device = Device::ndarray();
        let encoder: BertEncoder = BertEncoder::new(&config(), &device);
        let input = rand([2, 8, 384]);
        let out = encoder.forward(input);
        assert_eq!(out.dims(), [2, 8, 384]);
    }

    #[test]
    fn encoder_no_nans_through_full_stack() {
        let device = Device::ndarray();
        let encoder: BertEncoder = BertEncoder::new(&config(), &device);
        let input = rand([1, 16, 384]);
        let out = encoder.forward(input);
        let v = out.into_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| !x.is_nan() && !x.is_infinite()));
    }
}
