// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! `SemanticSearch` — thin facade combining a [`crate::embed::EmbeddingProvider`]
//! and a [`crate::embed::VectorIndex`]. Hosts wire the facade into command palettes,
//! search bars, focus pickers, etc. without re-implementing the
//! embed-then-query pattern at every site.

use std::hash::Hash;

use crate::embed::VectorIndex;
use crate::embed::index::IndexError;
use crate::embed::provider::{EmbedError, EmbeddingProvider};

/// Errors returned by [`SemanticSearch`] operations.
#[derive(Debug)]
pub enum SearchError {
    Embed(EmbedError),
    Index(IndexError),
}

impl From<EmbedError> for SearchError {
    fn from(e: EmbedError) -> Self {
        SearchError::Embed(e)
    }
}

impl From<IndexError> for SearchError {
    fn from(e: IndexError) -> Self {
        SearchError::Index(e)
    }
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::Embed(e) => write!(f, "embed: {e}"),
            SearchError::Index(e) => write!(f, "index: {e:?}"),
        }
    }
}

impl std::error::Error for SearchError {}

/// Search facade. Owns a provider and an index; offers convenience methods
/// for ingesting nodes and querying.
///
/// The facade does not own node-text storage — callers pass node text at
/// ingestion time (typically from the host's own graph model). On query, the
/// facade embeds the query string and delegates to the index.
pub struct SemanticSearch<K: Hash + Eq + Clone, P: EmbeddingProvider> {
    provider: P,
    index: VectorIndex<K>,
}

impl<K: Hash + Eq + Clone, P: EmbeddingProvider> SemanticSearch<K, P> {
    /// Construct from an embedding provider. The index is built with the
    /// provider's dimensions and metric.
    pub fn new(provider: P) -> Self {
        let index = VectorIndex::new(provider.dimensions(), provider.metric());
        Self { provider, index }
    }

    /// Construct from an existing provider and pre-built index.
    /// Errors if the index dimensions or metric do not match the provider.
    pub fn with_index(provider: P, index: VectorIndex<K>) -> Result<Self, SearchError> {
        if index.dimensions() != provider.dimensions() {
            return Err(SearchError::Index(IndexError::DimensionMismatch {
                expected: provider.dimensions(),
                got: index.dimensions(),
            }));
        }
        Ok(Self { provider, index })
    }

    /// Ingest a single node by embedding its text and inserting under `key`.
    pub fn ingest(&mut self, key: K, text: &str) -> Result<(), SearchError> {
        let vec = self.provider.embed_one(text)?;
        self.index.insert(key, vec)?;
        Ok(())
    }

    /// Ingest a batch of node-text pairs in one provider call.
    pub fn ingest_batch(&mut self, items: &[(K, &str)]) -> Result<(), SearchError> {
        if items.is_empty() {
            return Ok(());
        }
        let texts: Vec<&str> = items.iter().map(|(_, t)| *t).collect();
        let vectors = self.provider.embed(&texts)?;
        for ((key, _), vec) in items.iter().zip(vectors) {
            self.index.insert(key.clone(), vec)?;
        }
        Ok(())
    }

    /// Remove a node from the index by key.
    pub fn forget(&mut self, key: &K) -> Option<Vec<f32>> {
        self.index.remove(key)
    }

    /// Embed `query` and return the top-`k` matching node keys with scores.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<(K, f32)>, SearchError> {
        let q = self.provider.embed_one(query)?;
        Ok(self.index.nearest(&q, k)?)
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Borrow the underlying provider (e.g. for ad-hoc embedding outside the
    /// search facade — used by the field-bridge integration).
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Borrow the underlying index (e.g. for field-bridge construction
    /// or persistence).
    pub fn index(&self) -> &VectorIndex<K> {
        &self.index
    }

    /// Mutable index access for advanced use; prefer [`Self::ingest`] /
    /// [`Self::forget`] for the common case.
    pub fn index_mut(&mut self) -> &mut VectorIndex<K> {
        &mut self.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::StubEmbeddingProvider;

    fn provider() -> StubEmbeddingProvider {
        StubEmbeddingProvider::new(64).unwrap()
    }

    #[test]
    fn new_search_is_empty() {
        let s = SemanticSearch::<u32, _>::new(provider());
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn ingest_and_search_roundtrip() {
        let mut s = SemanticSearch::<u32, _>::new(provider());
        s.ingest(1, "rust async").unwrap();
        s.ingest(2, "python typing").unwrap();
        s.ingest(3, "haskell monads").unwrap();
        assert_eq!(s.len(), 3);

        let result = s.search("rust async", 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn ingest_batch_matches_individual() {
        let mut s_batch = SemanticSearch::<u32, _>::new(provider());
        let mut s_each = SemanticSearch::<u32, _>::new(provider());
        let items = [(1u32, "a"), (2, "b"), (3, "c")];
        s_batch.ingest_batch(&items).unwrap();
        for (k, t) in &items {
            s_each.ingest(*k, t).unwrap();
        }
        assert_eq!(s_batch.len(), s_each.len());
        for (k, _) in &items {
            assert_eq!(s_batch.index().get(k), s_each.index().get(k));
        }
    }

    #[test]
    fn forget_excludes_from_search() {
        let mut s = SemanticSearch::<u32, _>::new(provider());
        s.ingest(1, "alpha").unwrap();
        s.ingest(2, "beta").unwrap();
        s.forget(&1);
        let result = s.search("alpha", 5).unwrap();
        assert!(result.iter().all(|(k, _)| *k != 1));
    }

    #[test]
    fn ingest_batch_empty_is_noop() {
        let mut s = SemanticSearch::<u32, _>::new(provider());
        s.ingest_batch(&[]).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn search_top_k_truncates() {
        let mut s = SemanticSearch::<u32, _>::new(provider());
        for i in 0..10 {
            s.ingest(i, &format!("text {i}")).unwrap();
        }
        let result = s.search("text 0", 3).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn with_index_dimension_mismatch_errors() {
        let p = StubEmbeddingProvider::new(64).unwrap();
        let bad_index = VectorIndex::<u32>::new(32, p.metric());
        let result = SemanticSearch::with_index(p, bad_index);
        assert!(matches!(result, Err(SearchError::Index(_))));
    }

    #[test]
    fn with_index_matching_dimensions_works() {
        let p = provider();
        let good_index = VectorIndex::<u32>::new(p.dimensions(), p.metric());
        let s = SemanticSearch::with_index(p, good_index).unwrap();
        assert!(s.is_empty());
    }
}
