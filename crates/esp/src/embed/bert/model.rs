//! `BertModel` — embeddings + encoder + sentence pooling.
//!
//! For sentence-transformers compatibility this model produces sentence
//! embeddings via **mean-pooling** over the sequence dimension, optionally
//! followed by L2 normalisation. (The original HF BERT pre-trains with
//! `[CLS]` pooling; sentence-transformers retrains with mean-pooling, and
//! it's the form that matches `all-MiniLM-L6-v2`.)

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor};

use super::config::BertConfig;
use super::embeddings::BertEmbeddings;
use super::encoder::BertEncoder;

/// Sentence-pooling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Mean-pool along the sequence dimension. Used by sentence-transformers.
    Mean,
    /// Take the first token (`[CLS]`) representation. Used by classic BERT.
    Cls,
}

#[derive(Module, Debug)]
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
}

impl BertModel {
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        Self {
            embeddings: BertEmbeddings::new(config, device),
            encoder: BertEncoder::new(config, device),
        }
    }

    /// Construct from a [`super::loaded::LoadedBert`] bundle. This is the
    /// canonical "I have weights, build me a model" entry point used by
    /// [`super::loader::load_into_model`].
    pub fn from_loaded(loaded: &super::loaded::LoadedBert, device: &Device) -> Self {
        Self {
            embeddings: BertEmbeddings::from_loaded(&loaded.config, &loaded.embeddings, device),
            encoder: BertEncoder::from_loaded(&loaded.config, &loaded.layers, device),
        }
    }

    /// Token-level forward: `[B, S]` int → `[B, S, hidden]` float.
    pub fn forward_tokens(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        let embedded = self.embeddings.forward(input_ids);
        self.encoder.forward(embedded)
    }

    /// Sentence-level forward: `[B, S]` int → `[B, hidden]` float.
    /// Pools across the sequence dimension; optionally L2-normalises.
    pub fn forward_sentence(
        &self,
        input_ids: Tensor<2, Int>,
        pooling: Pooling,
        l2_normalize: bool,
    ) -> Tensor<2> {
        let token_repr = self.forward_tokens(input_ids);
        let pooled = match pooling {
            Pooling::Mean => mean_pool(token_repr),
            Pooling::Cls => cls_pool(token_repr),
        };
        if l2_normalize {
            l2_normalize_rows(pooled)
        } else {
            pooled
        }
    }
}

/// Mean-pool along the sequence (dim 1). `[B, S, H]` → `[B, H]`.
fn mean_pool(token_repr: Tensor<3>) -> Tensor<2> {
    let [batch, _, hidden] = token_repr.dims();
    token_repr.mean_dim(1).reshape([batch, hidden])
}

/// Take the first token's representation. `[B, S, H]` → `[B, H]`.
fn cls_pool(token_repr: Tensor<3>) -> Tensor<2> {
    let [batch, _, hidden] = token_repr.dims();
    token_repr
        .slice([0..batch, 0..1, 0..hidden])
        .reshape([batch, hidden])
}

/// L2-normalise along the last dim. `[B, H]` rows → unit norm.
fn l2_normalize_rows(rows: Tensor<2>) -> Tensor<2> {
    let squared = rows.clone() * rows.clone();
    let sum = squared.sum_dim(1);
    let norm = sum.sqrt().clamp_min(f32::EPSILON);
    rows / norm
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

    fn ids(rows: &[&[i64]]) -> Tensor<2, Int> {
        let device = Device::ndarray();
        let max_len = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let batch = rows.len();
        let mut flat = Vec::with_capacity(batch * max_len);
        for r in rows {
            flat.extend_from_slice(r);
            for _ in r.len()..max_len {
                flat.push(0);
            }
        }
        Tensor::from_data(
            burn::tensor::TensorData::new(flat, [batch, max_len]),
            &device,
        )
    }

    #[test]
    fn forward_tokens_returns_per_token_shape() {
        let device = Device::ndarray();
        let model: BertModel = BertModel::new(&config(), &device);
        let input = ids(&[&[101, 1023, 2057, 6010, 102]]);
        let out = model.forward_tokens(input);
        assert_eq!(out.dims(), [1, 5, 384]);
    }

    #[test]
    fn forward_sentence_mean_pool_returns_correct_shape() {
        let device = Device::ndarray();
        let model: BertModel = BertModel::new(&config(), &device);
        let input = ids(&[&[101, 1023, 2057, 6010, 102], &[101, 7592, 2088, 102]]);
        let out = model.forward_sentence(input, Pooling::Mean, true);
        assert_eq!(out.dims(), [2, 384]);
    }

    #[test]
    fn forward_sentence_cls_pool_returns_correct_shape() {
        let device = Device::ndarray();
        let model: BertModel = BertModel::new(&config(), &device);
        let input = ids(&[&[101, 1023, 2057, 6010, 102]]);
        let out = model.forward_sentence(input, Pooling::Cls, false);
        assert_eq!(out.dims(), [1, 384]);
    }

    #[test]
    fn l2_normalized_rows_have_unit_norm() {
        let device = Device::ndarray();
        let model: BertModel = BertModel::new(&config(), &device);
        let input = ids(&[&[101, 1023, 2057, 6010, 102]]);
        let out = model.forward_sentence(input, Pooling::Mean, true);
        let v = out.into_data().to_vec::<f32>().unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-4, "norm = {norm}");
    }

    #[test]
    fn no_nans_through_full_stack() {
        let device = Device::ndarray();
        let model: BertModel = BertModel::new(&config(), &device);
        let input = ids(&[&[101, 1023, 2057, 6010, 102]]);
        let out = model.forward_sentence(input, Pooling::Mean, true);
        let v = out.into_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| !x.is_nan() && !x.is_infinite()));
    }
}
