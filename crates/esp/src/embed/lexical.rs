// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Lexical feature-hashing [`EmbeddingProvider`] — the "hashing trick".
//!
//! Tokenizes text and hashes configured token n-gram orders into a
//! fixed-dimension vector (signed, to cancel some hash collisions), then
//! L2-normalizes so cosine similarity is well-defined. Pure Rust, no model.
//!
//! The distinction from [`StubEmbeddingProvider`](crate::embed::StubEmbeddingProvider)
//! is the whole point: that one hashes the *entire string* to a seed, so two
//! texts that differ by a single character get unrelated vectors (deterministic
//! but **semantically meaningless** — good only as a test double). This one
//! hashes *per token feature*, so texts that **share tokens** get correlated vectors:
//! "rust async runtime" and "async rust internals" land near each other in
//! cosine space. That is a real, cheap, burn-free *lexical* similarity signal —
//! useful for clustering nodes by shared vocabulary without loading a model.
//! Unigrams are the compatibility default. Callers can add bigrams or trigrams
//! when phrase order matters; changing the orders changes the vector space, so
//! any derived index must be re-minted under the same setting.
//!
//! It is still lexical, not semantic: it clusters on shared surface words, so
//! paraphrases with no shared tokens ("car" vs "automobile") stay apart. True
//! semantic similarity needs the Burn-backed BERT provider (the `bert` feature).

use crate::embed::provider::{EmbedError, EmbeddingProvider, SimilarityMetric};

/// Compatibility/default token n-gram orders: unigrams only.
pub const DEFAULT_TOKEN_NGRAM_ORDERS: [usize; 1] = [1];

/// Lexical feature-hashing embedding provider.
///
/// Construction takes the output dimension (the number of hash buckets). Larger
/// dimensions reduce token collisions at the cost of sparser vectors; 256–512 is
/// a reasonable range for short texts like titles and tag sets.
#[derive(Debug, Clone)]
pub struct LexicalEmbeddingProvider {
    dimensions: usize,
    token_ngram_orders: Vec<usize>,
}

impl LexicalEmbeddingProvider {
    /// Construct a unigram provider with the desired output dimension.
    ///
    /// This preserves the original provider's byte-stable feature hashes and
    /// vectors. Use [`Self::with_token_ngram_orders`] to opt into phrase features.
    pub fn new(dimensions: usize) -> Result<Self, EmbedError> {
        Self::with_token_ngram_orders(dimensions, DEFAULT_TOKEN_NGRAM_ORDERS)
    }

    /// Construct with explicit token n-gram orders.
    ///
    /// Orders must be non-empty and positive. They are sorted and deduplicated
    /// so equivalent settings produce the same vector space. Every feature has
    /// equal weight before the final L2 normalization.
    pub fn with_token_ngram_orders(
        dimensions: usize,
        token_ngram_orders: impl AsRef<[usize]>,
    ) -> Result<Self, EmbedError> {
        if dimensions == 0 {
            return Err(EmbedError::InvalidConfig(
                "dimensions must be > 0".to_string(),
            ));
        }
        let mut token_ngram_orders = token_ngram_orders.as_ref().to_vec();
        if token_ngram_orders.is_empty() {
            return Err(EmbedError::InvalidConfig(
                "token n-gram orders must not be empty".to_string(),
            ));
        }
        if token_ngram_orders.contains(&0) {
            return Err(EmbedError::InvalidConfig(
                "token n-gram orders must be > 0".to_string(),
            ));
        }
        token_ngram_orders.sort_unstable();
        token_ngram_orders.dedup();
        Ok(Self {
            dimensions,
            token_ngram_orders,
        })
    }

    /// Canonical token n-gram orders used by this provider.
    pub fn token_ngram_orders(&self) -> &[usize] {
        &self.token_ngram_orders
    }

    /// Feature-hash one text into an L2-normalized token n-gram vector.
    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dimensions];
        let tokens: Vec<String> = tokenize(text).collect();
        for &order in &self.token_ngram_orders {
            for ngram in tokens.windows(order) {
                let h = hash_token_ngram(ngram);
                let idx = (h % self.dimensions as u64) as usize;
                // Signed hashing (Weinberger et al.): a second hash bit picks the sign,
                // so colliding features partially cancel instead of always reinforcing —
                // it keeps the dot product an unbiased estimate of the true overlap.
                let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
                vec[idx] += sign;
            }
        }
        l2_normalize(&mut vec);
        vec
    }
}

impl EmbeddingProvider for LexicalEmbeddingProvider {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn metric(&self) -> SimilarityMetric {
        SimilarityMetric::Cosine
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }
}

/// Split text into lowercased alphanumeric tokens. Punctuation and whitespace are
/// separators; empty tokens are dropped. Lowercasing folds case so "Rust" and
/// "rust" hash to the same bucket.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64-bit hash. Deterministic, allocation-free, byte-stable — the same
/// hash the whole-string [`StubEmbeddingProvider`](crate::embed::StubEmbeddingProvider)
/// uses, applied here per token feature.
fn fnv1a_64(text: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for byte in text.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash a token n-gram without allocating a joined string. Unigrams delegate
/// to the original token hash exactly. Higher orders use an ASCII unit
/// separator, which cannot appear inside an alphanumeric token, as the boundary.
fn hash_token_ngram(tokens: &[String]) -> u64 {
    if let [token] = tokens {
        return fnv1a_64(token);
    }

    let mut h = FNV_OFFSET;
    for (position, token) in tokens.iter().enumerate() {
        if position > 0 {
            h ^= 0x1f;
            h = h.wrapping_mul(FNV_PRIME);
        }
        for byte in token.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Scale a vector to unit L2 length in place (a no-op on the zero vector, which a
/// token-free text produces — its cosine similarity to anything is then 0).
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cosine similarity of two same-length vectors (they are already L2-normalized,
    /// so this is just their dot product).
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dimensions_and_orders_must_be_valid() {
        assert!(LexicalEmbeddingProvider::new(0).is_err());
        assert!(LexicalEmbeddingProvider::new(256).is_ok());
        assert!(LexicalEmbeddingProvider::with_token_ngram_orders(256, []).is_err());
        assert!(LexicalEmbeddingProvider::with_token_ngram_orders(256, [0, 1]).is_err());
    }

    #[test]
    fn unigram_is_the_canonical_compatibility_default() {
        let default = LexicalEmbeddingProvider::new(64).unwrap();
        let explicit = LexicalEmbeddingProvider::with_token_ngram_orders(64, [1]).unwrap();
        assert_eq!(default.token_ngram_orders(), DEFAULT_TOKEN_NGRAM_ORDERS);
        assert_eq!(
            default.embed_one("Rust, async runtime!").unwrap(),
            explicit.embed_one("Rust, async runtime!").unwrap()
        );

        // Golden bins from the original unigram implementation: keeping these
        // fixes compatibility more strongly than comparing two new constructors.
        let vector = default.embed_one("Rust, async runtime!").unwrap();
        let expected = 1.0f32 / 3.0f32.sqrt();
        assert!((vector[39] + expected).abs() < 1.0e-6);
        assert!((vector[43] - expected).abs() < 1.0e-6);
        assert!((vector[47] - expected).abs() < 1.0e-6);
        assert_eq!(vector.iter().filter(|&&value| value != 0.0).count(), 3);
    }

    #[test]
    fn token_ngram_orders_are_canonicalized() {
        let provider = LexicalEmbeddingProvider::with_token_ngram_orders(64, [3, 1, 2, 2]).unwrap();
        assert_eq!(provider.token_ngram_orders(), [1, 2, 3]);
    }

    #[test]
    fn higher_orders_distinguish_phrase_order() {
        let unigram = LexicalEmbeddingProvider::new(4096).unwrap();
        let bigram = LexicalEmbeddingProvider::with_token_ngram_orders(4096, [1, 2]).unwrap();
        let phrase = "open downloads folder";
        let reordered = "folder downloads open";

        assert_eq!(
            unigram.embed_one(phrase).unwrap(),
            unigram.embed_one(reordered).unwrap()
        );
        let phrase_vector = bigram.embed_one(phrase).unwrap();
        let reordered_vector = bigram.embed_one(reordered).unwrap();
        assert!(cosine(&phrase_vector, &reordered_vector) < 0.75);
    }

    #[test]
    fn metric_is_cosine_and_output_is_normalized() {
        let p = LexicalEmbeddingProvider::new(256).unwrap();
        assert_eq!(p.metric(), SimilarityMetric::Cosine);
        let v = p.embed_one("the quick brown fox").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn deterministic_same_input() {
        let p = LexicalEmbeddingProvider::new(128).unwrap();
        assert_eq!(
            p.embed_one("rust async runtime").unwrap(),
            p.embed_one("rust async runtime").unwrap()
        );
    }

    #[test]
    fn shared_tokens_are_more_similar_than_disjoint_ones() {
        // The core property the StubEmbeddingProvider lacks: shared vocabulary →
        // higher cosine. "async rust" overlaps "rust runtime" (share "rust") more
        // than it overlaps "italian dinner recipes" (no shared token).
        let p = LexicalEmbeddingProvider::new(512).unwrap();
        let a = p.embed_one("async rust programming").unwrap();
        let related = p.embed_one("rust runtime internals").unwrap();
        let unrelated = p.embed_one("italian dinner recipes").unwrap();
        let sim_related = cosine(&a, &related);
        let sim_unrelated = cosine(&a, &unrelated);
        assert!(
            sim_related > sim_unrelated,
            "shared-token pair ({sim_related}) should out-score the disjoint pair ({sim_unrelated})"
        );
    }

    #[test]
    fn case_and_punctuation_are_folded() {
        // "Rust!" and "rust" tokenize to the same token, so identical up to case /
        // punctuation gives cosine ~1.
        let p = LexicalEmbeddingProvider::new(256).unwrap();
        let a = p.embed_one("Rust, Async!").unwrap();
        let b = p.embed_one("rust async").unwrap();
        assert!(
            cosine(&a, &b) > 0.999,
            "case/punct should fold to the same tokens"
        );
    }

    #[test]
    fn empty_and_tokenless_text_is_the_zero_vector() {
        let p = LexicalEmbeddingProvider::new(64).unwrap();
        for text in ["", "   ", "!!! ---"] {
            let v = p.embed_one(text).unwrap();
            assert_eq!(v.len(), 64);
            assert!(
                v.iter().all(|&x| x == 0.0),
                "tokenless {text:?} → zero vector"
            );
        }
    }

    #[test]
    fn batch_preserves_order() {
        let p = LexicalEmbeddingProvider::new(64).unwrap();
        let texts = ["alpha beta", "gamma", "delta epsilon zeta"];
        let batch = p.embed(&texts).unwrap();
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(batch[i], p.embed_one(t).unwrap(), "mismatch at {i}");
        }
    }
}
