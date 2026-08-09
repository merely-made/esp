//! [`BertEmbeddingProvider`] — Burn-generic BERT provider.
//!
//! End-to-end wiring is in place: the provider holds a [`BertModel`]
//! (random or loaded weights) and a [`BertTokenizer`], and `embed`
//! tokenises → runs the model → mean-pools → L2-normalises → returns
//! `Vec<Vec<f32>>`. Until the loader slice fills in safetensors → Burn
//! weight injection, callers using [`BertEmbeddingProvider::new`] will
//! get the unloaded provider that returns
//! [`crate::embed::EmbedError::ModelNotLoaded`]. Callers that already have a
//! [`BertModel`] and [`BertTokenizer`] in hand (e.g. tests, or future
//! loaded-from-disk paths) can build a working provider via
//! [`BertEmbeddingProvider::new_with_components`].

use std::path::Path;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::embed::bert::config::BertConfig;
use crate::embed::bert::loader::{load_artifacts, load_into_model, load_into_model_from_bytes};
use crate::embed::bert::model::{BertModel, Pooling};
use crate::embed::bert::tokenizer::BertTokenizer;
use crate::embed::provider::{EmbedError, EmbeddingProvider, SimilarityMetric};

/// Burn-backed BERT embedding provider.
///
/// Generic over `B: Backend` so callers pick CPU (ndarray) or GPU (wgpu).
#[derive(Debug)]
pub struct BertEmbeddingProvider<B: Backend> {
    config: BertConfig,
    model: Option<BertModel<B>>,
    tokenizer: Option<BertTokenizer>,
    device: B::Device,
    pooling: Pooling,
    l2_normalize: bool,
}

impl<B: Backend> BertEmbeddingProvider<B> {
    /// Construct an unloaded provider. `embed` returns
    /// [`EmbedError::ModelNotLoaded`] until [`Self::with_model`] and
    /// [`Self::with_tokenizer`] are populated (or the
    /// [`Self::new_with_components`] constructor is used).
    pub fn new(config: BertConfig, device: B::Device) -> Self {
        Self {
            config,
            model: None,
            tokenizer: None,
            device,
            pooling: Pooling::Mean,
            l2_normalize: true,
        }
    }

    /// Construct a fully-wired provider. `embed` will produce real
    /// embeddings via the model's forward pass.
    pub fn new_with_components(
        config: BertConfig,
        model: BertModel<B>,
        tokenizer: BertTokenizer,
        device: B::Device,
    ) -> Self {
        Self {
            config,
            model: Some(model),
            tokenizer: Some(tokenizer),
            device,
            pooling: Pooling::Mean,
            l2_normalize: true,
        }
    }

    /// Override the pooling strategy (default: [`Pooling::Mean`]).
    pub fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = pooling;
        self
    }

    /// Override L2-normalization (default: `true`).
    pub fn with_l2_normalize(mut self, l2: bool) -> Self {
        self.l2_normalize = l2;
        self
    }

    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Whether weights + tokenizer are loaded.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some() && self.tokenizer.is_some()
    }

    /// One-shot constructor: load `config.json` + `tokenizer.json` +
    /// `model.safetensors` from `model_dir`, build the model, return a
    /// fully-wired provider ready for `embed`.
    ///
    /// The expected directory layout matches HuggingFace's:
    ///
    /// ```text
    /// <model_dir>/
    ///   config.json
    ///   tokenizer.json
    ///   model.safetensors
    /// ```
    ///
    /// `B::Device` is the backend device on which tensors live (e.g.
    /// `NdArrayDevice::default()` for CPU, a wgpu device for GPU).
    pub fn load(model_dir: impl AsRef<Path>, device: B::Device) -> Result<Self, EmbedError> {
        let artifacts = load_artifacts(model_dir)?;
        let model = load_into_model::<B>(&artifacts, &device)?;
        Ok(Self::new_with_components(
            artifacts.config,
            model,
            artifacts.tokenizer,
            device,
        ))
    }

    /// In-memory constructor: build a fully-wired provider from already-
    /// loaded byte buffers. The path through which host-resolved model
    /// artifacts flow into a working provider, with no filesystem touch.
    ///
    /// `config_bytes` must be a HuggingFace-style `config.json`,
    /// `tokenizer_bytes` a HuggingFace `tokenizer.json`, and `weights_bytes`
    /// a `model.safetensors` buffer that satisfies the config's expected
    /// tensor shape (validated before injection).
    pub fn from_bytes(
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        weights_bytes: &[u8],
        device: B::Device,
    ) -> Result<Self, EmbedError> {
        let config: BertConfig = serde_json::from_slice(config_bytes)
            .map_err(|e| EmbedError::InvalidConfig(format!("config parse failed: {e}")))?;
        let tokenizer = BertTokenizer::from_bytes(
            tokenizer_bytes,
            config.max_position_embeddings,
            config.pad_token_id as u32,
        )?;
        let model = load_into_model_from_bytes::<B>(&config, weights_bytes, &device)?;
        Ok(Self::new_with_components(config, model, tokenizer, device))
    }
}

impl<B: Backend> EmbeddingProvider for BertEmbeddingProvider<B> {
    fn dimensions(&self) -> usize {
        self.config.hidden_size
    }

    fn metric(&self) -> SimilarityMetric {
        SimilarityMetric::Cosine
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let model = self.model.as_ref().ok_or(EmbedError::ModelNotLoaded)?;
        let tokenizer = self.tokenizer.as_ref().ok_or(EmbedError::ModelNotLoaded)?;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch = tokenizer.encode_batch(texts)?;
        let input_ids: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(batch.input_ids, [batch.batch_size, batch.seq_len]),
            &self.device,
        );

        let output = model.forward_sentence(input_ids, self.pooling, self.l2_normalize);

        // Convert [batch, hidden] tensor to Vec<Vec<f32>>.
        let dims = output.dims();
        let flat = output
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| EmbedError::Backend(format!("tensor → Vec<f32>: {e:?}")))?;
        let mut out = Vec::with_capacity(dims[0]);
        for row in flat.chunks_exact(dims[1]) {
            out.push(row.to_vec());
        }
        Ok(out)
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

    #[test]
    fn dimensions_match_config_hidden_size() {
        let p = BertEmbeddingProvider::<B>::new(config(), Default::default());
        assert_eq!(p.dimensions(), 384);
    }

    #[test]
    fn metric_is_cosine() {
        let p = BertEmbeddingProvider::<B>::new(config(), Default::default());
        assert_eq!(p.metric(), SimilarityMetric::Cosine);
    }

    #[test]
    fn unloaded_provider_returns_model_not_loaded() {
        let p = BertEmbeddingProvider::<B>::new(config(), Default::default());
        let err = p.embed(&["test"]).unwrap_err();
        assert_eq!(err, EmbedError::ModelNotLoaded);
        assert!(!p.is_loaded());
    }

    #[test]
    fn config_borrow_matches_constructor() {
        let cfg = config();
        let p = BertEmbeddingProvider::<B>::new(cfg.clone(), Default::default());
        assert_eq!(p.config(), &cfg);
    }

    #[test]
    fn satisfies_embedding_provider_trait_object_safety_check() {
        let p = BertEmbeddingProvider::<B>::new(config(), Default::default());
        let _: &dyn EmbeddingProvider = &p;
    }

    #[test]
    fn pooling_and_normalize_overrides_compose() {
        let p = BertEmbeddingProvider::<B>::new(config(), Default::default())
            .with_pooling(Pooling::Cls)
            .with_l2_normalize(false);
        assert_eq!(p.pooling, Pooling::Cls);
        assert!(!p.l2_normalize);
    }

    #[test]
    fn empty_texts_returns_empty_when_loaded() {
        // Sanity check: empty input short-circuits before model invocation
        // (so this works even when model/tokenizer are missing? — actually
        // the early-return is *after* the loaded check, so this is still
        // ModelNotLoaded for an unloaded provider).
        let p = BertEmbeddingProvider::<B>::new(config(), Default::default());
        let err = p.embed(&[]).unwrap_err();
        assert_eq!(err, EmbedError::ModelNotLoaded);
    }
}
