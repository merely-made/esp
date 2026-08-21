//! BERT input embeddings: word + position + token-type, summed and
//! layer-normed.
//!
//! ```text
//! emb(input_ids) = LayerNorm(
//!     word_embeddings(input_ids)
//!   + position_embeddings(0..seq_len)
//!   + token_type_embeddings(0)         // single-segment input
//! )
//! ```

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig};
use burn::tensor::{Device, Int, Tensor};

use super::config::BertConfig;

/// BERT embedding layer combining word, position, and token-type
/// embeddings under a final LayerNorm.
#[derive(Module, Debug)]
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    /// Initialize from a config with random weights. Real weights load via
    /// the future `loader.rs` slice.
    pub fn new(config: &BertConfig, device: &Device) -> Self {
        Self {
            word_embeddings: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device),
            position_embeddings: EmbeddingConfig::new(
                config.max_position_embeddings,
                config.hidden_size,
            )
            .init(device),
            token_type_embeddings: EmbeddingConfig::new(config.type_vocab_size, config.hidden_size)
                .init(device),
            layer_norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
        }
    }

    /// Construct from pre-loaded tensors. Tensors are cloned internally
    /// (cheap — Burn tensors are Arc-shared); the loaded bundle stays intact.
    pub fn from_loaded(
        config: &BertConfig,
        loaded: &super::loaded::LoadedEmbeddings,
        device: &Device,
    ) -> Self {
        Self {
            word_embeddings: super::construct::embedding_from_loaded(loaded.word.clone(), device),
            position_embeddings: super::construct::embedding_from_loaded(
                loaded.position.clone(),
                device,
            ),
            token_type_embeddings: super::construct::embedding_from_loaded(
                loaded.token_type.clone(),
                device,
            ),
            layer_norm: super::construct::layer_norm_from_loaded(
                loaded.ln_gamma.clone(),
                loaded.ln_beta.clone(),
                config.layer_norm_eps,
                device,
            ),
        }
    }

    /// Forward pass.
    ///
    /// - `input_ids`: `[batch, seq_len]` integer tensor of token ids.
    /// - returns: `[batch, seq_len, hidden_size]` float tensor.
    pub fn forward(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Position ids: [0, 1, ..., seq_len-1] broadcast across the batch.
        let position_ids: Tensor<2, Int> = Tensor::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch_size);

        // Single-segment input → all token type ids = 0.
        let token_type_ids: Tensor<2, Int> = Tensor::zeros([batch_size, seq_len], &device);

        let word_emb = self.word_embeddings.forward(input_ids);
        let pos_emb = self.position_embeddings.forward(position_ids);
        let type_emb = self.token_type_embeddings.forward(token_type_ids);

        let sum = word_emb + pos_emb + type_emb;
        self.layer_norm.forward(sum)
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

    #[test]
    fn embeddings_construct_without_panic() {
        let device = Device::ndarray();
        let _emb: BertEmbeddings = BertEmbeddings::new(&config(), &device);
    }

    #[test]
    fn forward_returns_correct_shape() {
        let device = Device::ndarray();
        let emb: BertEmbeddings = BertEmbeddings::new(&config(), &device);
        // batch=2, seq_len=5
        let input_ids: Tensor<2, Int> = Tensor::from_data(
            [[101, 1023, 2057, 6010, 102], [101, 7592, 2088, 102, 0]],
            &device,
        );
        let out = emb.forward(input_ids);
        assert_eq!(out.dims(), [2, 5, 384]);
    }

    #[test]
    fn forward_handles_single_token() {
        let device = Device::ndarray();
        let emb: BertEmbeddings = BertEmbeddings::new(&config(), &device);
        let input_ids: Tensor<2, Int> = Tensor::from_data([[101]], &device);
        let out = emb.forward(input_ids);
        assert_eq!(out.dims(), [1, 1, 384]);
    }
}
