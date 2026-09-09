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
use crate::embed::index::{IndexError, IndexVector};
use crate::embed::provider::{EmbedError, EmbeddingProvider, SimilarityMetric};
use crate::embed::sparse::{SparseIndex, SparseVector};

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
///
/// ## Which store
///
/// A provider offering [`EmbeddingProvider::embed_sparse`] is stored sparsely;
/// every other provider is stored densely. The methods below read the same
/// either way — the choice shows only in [`Self::is_sparse`] and in which of
/// [`Self::index`] / [`Self::sparse_index`] holds the entries. Scores are
/// unaffected: a sparse vector omits only terms worth exactly zero.
pub struct SemanticSearch<K: Hash + Eq + Clone, P: EmbeddingProvider> {
    provider: P,
    index: VectorIndex<K>,
    /// `Some` when the provider has a sparse path, and then the only store
    /// holding entries; `index` stays empty.
    sparse: Option<SparseIndex<K>>,
}

impl<K: Hash + Eq + Clone, P: EmbeddingProvider> SemanticSearch<K, P> {
    /// Construct from an embedding provider. The store is built with the
    /// provider's dimensions and metric, sparse if the provider offers it.
    pub fn new(provider: P) -> Self {
        let (dimensions, metric) = (provider.dimensions(), provider.metric());
        // The empty batch answers "is there a sparse path" without embedding.
        let sparse = provider
            .embed_sparse(&[])
            .is_some()
            .then(|| SparseIndex::new(dimensions, metric));
        Self {
            provider,
            index: VectorIndex::new(dimensions, metric),
            sparse,
        }
    }

    /// Construct from an existing provider and pre-built dense index. Forces
    /// the dense store even for a sparse-capable provider: the caller supplied
    /// the entries, so the caller picked the representation.
    /// Errors if the index dimensions do not match the provider.
    pub fn with_index(provider: P, index: VectorIndex<K>) -> Result<Self, SearchError> {
        check_dimensions(&provider, index.dimensions())?;
        Ok(Self {
            provider,
            index,
            sparse: None,
        })
    }

    /// [`Self::with_index`] over a pre-built sparse index.
    pub fn with_sparse_index(provider: P, index: SparseIndex<K>) -> Result<Self, SearchError> {
        check_dimensions(&provider, index.dimensions())?;
        let (dimensions, metric) = (provider.dimensions(), provider.metric());
        Ok(Self {
            provider,
            index: VectorIndex::new(dimensions, metric),
            sparse: Some(index),
        })
    }

    /// Embed through the provider's sparse path, which construction
    /// established exists.
    fn embed_sparse(&self, texts: &[&str]) -> Result<Vec<SparseVector>, SearchError> {
        match self.provider.embed_sparse(texts) {
            Some(result) => Ok(result?),
            None => Err(SearchError::Embed(EmbedError::Backend(
                "provider withdrew its sparse path after construction".to_string(),
            ))),
        }
    }

    /// Ingest a single node by embedding its text and inserting under `key`.
    pub fn ingest(&mut self, key: K, text: &str) -> Result<(), SearchError> {
        if self.sparse.is_some() {
            let vector = self.embed_sparse(&[text])?.pop().ok_or_else(empty_batch)?;
            self.sparse
                .as_mut()
                .expect("sparse store present")
                .insert(key, vector)?;
        } else {
            let vector = self.provider.embed_one(text)?;
            self.index.insert(key, vector)?;
        }
        Ok(())
    }

    /// Ingest a batch of node-text pairs in one provider call.
    pub fn ingest_batch(&mut self, items: &[(K, &str)]) -> Result<(), SearchError> {
        if items.is_empty() {
            return Ok(());
        }
        let texts: Vec<&str> = items.iter().map(|(_, t)| *t).collect();
        if self.sparse.is_some() {
            let vectors = self.embed_sparse(&texts)?;
            let sparse = self.sparse.as_mut().expect("sparse store present");
            for ((key, _), vector) in items.iter().zip(vectors) {
                sparse.insert(key.clone(), vector)?;
            }
        } else {
            let vectors = self.provider.embed(&texts)?;
            for ((key, _), vector) in items.iter().zip(vectors) {
                self.index.insert(key.clone(), vector)?;
            }
        }
        Ok(())
    }

    /// Remove a node from the store by key. Reports whether it was there.
    pub fn forget(&mut self, key: &K) -> bool {
        match self.sparse.as_mut() {
            Some(sparse) => sparse.remove(key).is_some(),
            None => self.index.remove(key).is_some(),
        }
    }

    /// Embed `query` and return the top-`k` matching node keys with scores.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<(K, f32)>, SearchError>
    where
        K: Ord,
    {
        if let Some(sparse) = self.sparse.as_ref() {
            let vector = self.embed_sparse(&[query])?.pop().ok_or_else(empty_batch)?;
            return Ok(sparse.nearest(&vector, k)?);
        }
        let vector = self.provider.embed_one(query)?;
        Ok(self.index.nearest(&vector, k)?)
    }

    /// Top-`k` for an already-embedded dense query, whichever store is in use.
    /// A sparse store sees the query's non-zeros, which scores identically.
    pub fn nearest(&self, query: &[f32], k: usize) -> Result<Vec<(K, f32)>, SearchError>
    where
        K: Ord,
    {
        Ok(match self.sparse.as_ref() {
            Some(sparse) => sparse.nearest(&SparseVector::from_dense(query), k)?,
            None => self.index.nearest(query, k)?,
        })
    }

    /// Raw metric score of a dense query against *every* entry, unranked.
    ///
    /// The whole-store form callers need when the scores drive something other
    /// than a ranking — a scalar field, a heatmap — and the reason such a
    /// caller need not know which store is in use.
    pub fn scores(&self, query: &[f32]) -> Vec<(K, f32)> {
        let metric = self.metric();
        match self.sparse.as_ref() {
            Some(sparse) => {
                let query = SparseVector::from_dense(query);
                sparse
                    .iter()
                    .map(|(key, v)| (key.clone(), SparseVector::score(metric, &query, v)))
                    .collect()
            },
            None => self
                .index
                .iter()
                .map(|(key, v)| {
                    (
                        key.clone(),
                        <Vec<f32> as IndexVector>::score(metric, query, v),
                    )
                })
                .collect(),
        }
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        match self.sparse.as_ref() {
            Some(sparse) => sparse.len(),
            None => self.index.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether entries are held sparsely — i.e. in [`Self::sparse_index`]
    /// rather than [`Self::index`].
    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }

    /// The metric both stores score under (the provider's).
    pub fn metric(&self) -> SimilarityMetric {
        self.provider.metric()
    }

    /// Borrow the underlying provider (e.g. for ad-hoc embedding outside the
    /// search facade — used by the field-bridge integration).
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Borrow the dense index. Empty when [`Self::is_sparse`] — the entries
    /// are in [`Self::sparse_index`] then.
    pub fn index(&self) -> &VectorIndex<K> {
        &self.index
    }

    /// Mutable dense index access for advanced use; prefer [`Self::ingest`] /
    /// [`Self::forget`] for the common case.
    pub fn index_mut(&mut self) -> &mut VectorIndex<K> {
        &mut self.index
    }

    /// Borrow the sparse index, present exactly when [`Self::is_sparse`].
    pub fn sparse_index(&self) -> Option<&SparseIndex<K>> {
        self.sparse.as_ref()
    }

    /// Mutable counterpart to [`Self::sparse_index`].
    pub fn sparse_index_mut(&mut self) -> Option<&mut SparseIndex<K>> {
        self.sparse.as_mut()
    }
}

fn check_dimensions<P: EmbeddingProvider>(provider: &P, got: usize) -> Result<(), SearchError> {
    if got != provider.dimensions() {
        return Err(SearchError::Index(IndexError::DimensionMismatch {
            expected: provider.dimensions(),
            got,
        }));
    }
    Ok(())
}

fn empty_batch() -> SearchError {
    SearchError::Embed(EmbedError::Backend(
        "provider returned no vectors for one input".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{LexicalEmbeddingProvider, StubEmbeddingProvider};

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
        assert!(s.forget(&1));
        assert!(!s.forget(&1));
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
        assert!(!s.is_sparse());
    }

    // --- the sparse lane ------------------------------------------------

    fn lexical() -> LexicalEmbeddingProvider {
        LexicalEmbeddingProvider::new(256).unwrap()
    }

    const DOCS: [(u32, &str); 5] = [
        (1, "rust async runtime"),
        (2, "python typing generics"),
        (3, "async rust internals"),
        (4, "italian dinner recipes"),
        (5, "open downloads folder"),
    ];

    #[test]
    fn a_sparse_capable_provider_selects_the_sparse_store() {
        let mut s = SemanticSearch::<u32, _>::new(lexical());
        assert!(s.is_sparse());
        s.ingest_batch(&DOCS).unwrap();
        assert_eq!(s.len(), DOCS.len());
        assert_eq!(s.sparse_index().unwrap().len(), DOCS.len());
        // The dense side is the empty one, not a duplicate copy.
        assert!(s.index().is_empty());
        assert!(s.forget(&1));
        assert_eq!(s.len(), DOCS.len() - 1);
    }

    #[test]
    fn a_dense_only_provider_selects_the_dense_store() {
        let s = SemanticSearch::<u32, _>::new(provider());
        assert!(!s.is_sparse());
        assert!(s.sparse_index().is_none());
    }

    /// The facade's public reads must not depend on which store it picked.
    #[test]
    fn sparse_and_dense_stores_rank_identically() {
        let mut sparse = SemanticSearch::<u32, _>::new(lexical());
        let mut dense =
            SemanticSearch::with_index(lexical(), VectorIndex::new(256, lexical().metric()))
                .unwrap();
        sparse.ingest_batch(&DOCS).unwrap();
        dense.ingest_batch(&DOCS).unwrap();
        assert!(sparse.is_sparse() && !dense.is_sparse());

        for query in ["rust runtime", "dinner", "nothing at all"] {
            assert_eq!(
                sparse.search(query, DOCS.len()).unwrap(),
                dense.search(query, DOCS.len()).unwrap()
            );
            let embedded = lexical().embed_one(query).unwrap();
            assert_eq!(
                sparse.nearest(&embedded, 3).unwrap(),
                dense.nearest(&embedded, 3).unwrap()
            );
            let (mut a, mut b) = (sparse.scores(&embedded), dense.scores(&embedded));
            a.sort_by_key(|(k, _)| *k);
            b.sort_by_key(|(k, _)| *k);
            assert_eq!(a, b, "whole-store scores for {query:?}");
        }
    }

    #[test]
    fn with_sparse_index_takes_a_prebuilt_store() {
        let p = lexical();
        let mut index = SparseIndex::<u32>::new(p.dimensions(), p.metric());
        index.insert(7, p.embed_sparse_one("rust async")).unwrap();
        let s = SemanticSearch::with_sparse_index(p, index).unwrap();
        assert!(s.is_sparse());
        assert_eq!(s.search("rust", 1).unwrap()[0].0, 7);

        let p = lexical();
        let wrong = SparseIndex::<u32>::new(8, p.metric());
        assert!(matches!(
            SemanticSearch::with_sparse_index(p, wrong),
            Err(SearchError::Index(_))
        ));
    }
}
