// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Embedding provider trait + similarity-metric vocabulary.

use serde::{Deserialize, Serialize};

/// Similarity metric an embedding provider declares for its output space.
///
/// The choice affects how a vector index should compare vectors. Most
/// sentence-transformer-class models are L2-normalized and use Cosine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// `cos(θ) = (a · b) / (‖a‖ ‖b‖)`. Higher is more similar.
    /// Suitable for L2-normalized embeddings.
    Cosine,
    /// `‖a − b‖`. Lower is more similar. Suitable for un-normalized embeddings.
    Euclidean,
    /// `a · b`. Higher is more similar. Suitable when magnitudes carry meaning.
    DotProduct,
}

impl SimilarityMetric {
    /// Whether higher-is-more-similar (true) or lower-is-more-similar (false).
    pub fn higher_is_better(self) -> bool {
        matches!(self, Self::Cosine | Self::DotProduct)
    }
}

/// Reasons an embedding call could fail.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbedError {
    /// Input text exceeded the model's token/character limit.
    InputTooLong { length: usize, limit: usize },
    /// The provider type can hold a model but no weights have been loaded yet.
    ModelNotLoaded,
    /// The underlying compute backend (Burn / CPU / wgpu) returned an error.
    Backend(String),
    /// The provider was constructed with invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::InputTooLong { length, limit } => {
                write!(f, "input too long: {length} > {limit}")
            }
            EmbedError::ModelNotLoaded => write!(f, "model not loaded"),
            EmbedError::Backend(msg) => write!(f, "backend: {msg}"),
            EmbedError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

/// Provider that converts text into fixed-dimension vectors.
///
/// Implementations must be `Send + Sync` so the same provider instance can be
/// shared across threads. `embed` operates on a batch: callers that have one
/// text wrap it in a single-element slice; batched calls let GPU-backed
/// providers amortise dispatch overhead.
pub trait EmbeddingProvider: Send + Sync {
    /// Output vector dimension. Constant for the lifetime of this provider.
    fn dimensions(&self) -> usize;

    /// Similarity metric paired with this provider's output space.
    fn metric(&self) -> SimilarityMetric;

    /// Embed a batch of texts. Returns one vector per input, in input order,
    /// each of length [`dimensions`](Self::dimensions).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Embed a single text. Default implementation wraps `embed`.
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut out = self.embed(&[text])?;
        out.pop().ok_or_else(|| {
            EmbedError::Backend("provider returned no vectors for one input".to_string())
        })
    }
}

/// A boxed provider is a provider.
///
/// Without this, a consumer that picks its backend at runtime has to satisfy a
/// generic bound with a concrete Burn type — which is precisely the accidental
/// dependency the ESP-owned constructors exist to remove. With it, `Box<dyn
/// EmbeddingProvider>` flows into everything generic over `P: EmbeddingProvider`.
impl EmbeddingProvider for Box<dyn EmbeddingProvider> {
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }

    fn metric(&self) -> SimilarityMetric {
        (**self).metric()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        (**self).embed(texts)
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        (**self).embed_one(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::LexicalEmbeddingProvider;

    #[test]
    fn a_boxed_provider_satisfies_the_seam() {
        let boxed: Box<dyn EmbeddingProvider> =
            Box::new(LexicalEmbeddingProvider::new(64).unwrap());
        // Both the trait's own methods and its defaulted one delegate.
        assert_eq!(boxed.dimensions(), 64);
        assert_eq!(boxed.metric(), SimilarityMetric::Cosine);
        assert_eq!(boxed.embed_one("rust").unwrap().len(), 64);
        assert_eq!(boxed.embed(&["rust", "async"]).unwrap().len(), 2);

        // And it composes where a generic bound is expected, which is the
        // whole point: no consumer needs to name a backend type to get here.
        fn takes_provider<P: EmbeddingProvider>(p: P) -> usize {
            p.dimensions()
        }
        assert_eq!(takes_provider(boxed), 64);
    }

    #[test]
    fn metric_higher_is_better_classification() {
        assert!(SimilarityMetric::Cosine.higher_is_better());
        assert!(SimilarityMetric::DotProduct.higher_is_better());
        assert!(!SimilarityMetric::Euclidean.higher_is_better());
    }

    #[test]
    fn metric_serde_roundtrip() {
        for m in [
            SimilarityMetric::Cosine,
            SimilarityMetric::Euclidean,
            SimilarityMetric::DotProduct,
        ] {
            let s = serde_json::to_string(&m).unwrap();
            let back: SimilarityMetric = serde_json::from_str(&s).unwrap();
            assert_eq!(m, back);
        }
    }

    #[test]
    fn embed_error_is_clone_eq() {
        let e = EmbedError::InputTooLong {
            length: 10000,
            limit: 512,
        };
        assert_eq!(e.clone(), e);
    }
}
