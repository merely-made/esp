// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! Provider-bound persistence. Legacy numerical indexes remain independent;
//! this path refuses missing identities, foreign spaces and changed corpora.
use super::{EmbeddingProvider, SemanticSearch, SimilarityMetric, SparseIndex, VectorIndex};
use eidetic::models::embedding::{EmbeddingCorpusV1, EmbeddingEnvelopeV1, EmbeddingMetric};
use eidetic::{ManifestId, SchemaRef, TypedPayload};
use serde::{Deserialize, Serialize};
use std::hash::Hash;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    tag = "representation",
    content = "index",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum CompatibleVectors<K: Hash + Eq + Clone> {
    Dense(VectorIndex<K>),
    Sparse(SparseIndex<K>),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibleIndex<K: Hash + Eq + Clone> {
    pub envelope: EmbeddingEnvelopeV1,
    pub vectors: CompatibleVectors<K>,
}
impl<K> TypedPayload for CompatibleIndex<K>
where
    K: Hash + Eq + Clone + Serialize + serde::de::DeserializeOwned,
{
    fn schema_ref() -> SchemaRef {
        SchemaRef::from_id(ManifestId::of_blob(br#"{"format":"mere-native","schema_id":"esp.CompatibleIndex/v1","body":{"version":1,"description":"Provider-bound embedding envelope and explicit dense/sparse numerical index."}}"#))
    }
}

pub struct CompatibleSearch<K: Hash + Eq + Clone, P: EmbeddingProvider> {
    search: SemanticSearch<K, P>,
    envelope: EmbeddingEnvelopeV1,
}
impl<K: Hash + Eq + Clone, P: EmbeddingProvider> CompatibleSearch<K, P> {
    pub fn build(
        provider: P,
        corpus: EmbeddingCorpusV1,
        items: &[(K, &str)],
    ) -> Result<Self, String> {
        let envelope = envelope(&provider, corpus)?;
        let mut search = SemanticSearch::new(provider);
        // Validate batch cardinality before inserting: a partial provider batch
        // must not silently publish an incomplete derived projection.
        let prefixed: Vec<_> = items
            .iter()
            .map(|(_, t)| format!("{}{t}", envelope.space.document_prefix))
            .collect();
        let texts: Vec<_> = prefixed.iter().map(String::as_str).collect();
        if search.is_sparse() {
            let vectors = search
                .provider()
                .embed_sparse(&texts)
                .ok_or("provider withdrew sparse capability")?
                .map_err(|e| e.to_string())?;
            if vectors.len() != items.len() {
                return Err("embedding batch cardinality differs".into());
            }
            let index = search.sparse_index_mut().unwrap();
            for ((key, _), vector) in items.iter().zip(vectors) {
                validate_sparse(&vector, index.dimensions())?;
                index
                    .insert(key.clone(), vector)
                    .map_err(|e| format!("embedding index: {e:?}"))?;
            }
        } else {
            let vectors = search.provider().embed(&texts).map_err(|e| e.to_string())?;
            if vectors.len() != items.len() {
                return Err("embedding batch cardinality differs".into());
            }
            for ((key, _), vector) in items.iter().zip(vectors) {
                if vector.len() != search.index().dimensions()
                    || vector.iter().any(|v| !v.is_finite())
                {
                    return Err("invalid dense embedding vector".into());
                }
                search
                    .index_mut()
                    .insert(key.clone(), vector)
                    .map_err(|e| format!("embedding index: {e:?}"))?;
            }
        }
        if search.len() != items.len() {
            return Err("embedding keys repeat".into());
        }
        Ok(Self { search, envelope })
    }
    pub fn reopen(
        provider: P,
        corpus: EmbeddingCorpusV1,
        saved: CompatibleIndex<K>,
    ) -> Result<Self, String> {
        let expected = envelope(&provider, corpus)?;
        saved.envelope.ensure_compatible(&expected)?;
        let dimensions = expected.space.dimensions;
        let metric = metric(expected.space.metric);
        let search = match saved.vectors {
            CompatibleVectors::Dense(index) => {
                if index.dimensions() != dimensions
                    || index.metric() != metric
                    || index
                        .iter()
                        .any(|(_, v)| v.len() != dimensions || v.iter().any(|x| !x.is_finite()))
                {
                    return Err("stored dense index differs from its embedding envelope".into());
                }
                SemanticSearch::with_index(provider, index).map_err(|e| e.to_string())?
            }
            CompatibleVectors::Sparse(index) => {
                if provider.embed_sparse(&[]).is_none() {
                    return Err(
                        "stored sparse index requires the provider's sparse capability".into(),
                    );
                }
                if index.dimensions() != dimensions || index.metric() != metric {
                    return Err("stored sparse index differs from its embedding envelope".into());
                }
                for (_, vector) in index.iter() {
                    validate_sparse(vector, dimensions)?;
                }
                SemanticSearch::with_sparse_index(provider, index).map_err(|e| e.to_string())?
            }
        };
        Ok(Self {
            search,
            envelope: expected,
        })
    }
    pub fn snapshot(&self) -> CompatibleIndex<K> {
        CompatibleIndex {
            envelope: self.envelope.clone(),
            vectors: match self.search.sparse_index() {
                Some(index) => CompatibleVectors::Sparse(index.clone()),
                None => CompatibleVectors::Dense(self.search.index().clone()),
            },
        }
    }
    pub fn envelope(&self) -> &EmbeddingEnvelopeV1 {
        &self.envelope
    }
    pub fn is_sparse(&self) -> bool {
        self.search.is_sparse()
    }
    pub fn len(&self) -> usize {
        self.search.len()
    }
    pub fn is_empty(&self) -> bool {
        self.search.is_empty()
    }
    pub fn keys(&self) -> Vec<K> {
        match self.search.sparse_index() {
            Some(index) => index.iter().map(|(k, _)| k.clone()).collect(),
            None => self.search.index().iter().map(|(k, _)| k.clone()).collect(),
        }
    }
    /// Numerical vector storage cost, excluding entry keys, provider weights
    /// and allocator/map overhead. Used alongside the resident's byte ceiling.
    pub fn vector_bytes(&self) -> usize {
        match self.search.sparse_index() {
            Some(index) => index.iter().map(|(_, v)| v.heap_bytes()).sum(),
            None => self
                .search
                .index()
                .iter()
                .map(|(_, v)| v.capacity() * std::mem::size_of::<f32>())
                .sum(),
        }
    }
    pub fn search(&self, question: &str, k: usize) -> Result<Vec<(K, f32)>, String>
    where
        K: Ord,
    {
        let input = format!("{}{question}", self.envelope.space.query_prefix);
        if let Some(index) = self.search.sparse_index() {
            let values = self
                .search
                .provider()
                .embed_sparse(&[&input])
                .ok_or("provider withdrew sparse capability")?
                .map_err(|e| e.to_string())?;
            if values.len() != 1 {
                return Err("embedding query batch cardinality differs".into());
            }
            validate_sparse(&values[0], self.envelope.space.dimensions)?;
            index
                .nearest(&values[0], k)
                .map_err(|e| format!("embedding query: {e:?}"))
        } else {
            let values = self
                .search
                .provider()
                .embed(&[&input])
                .map_err(|e| e.to_string())?;
            if values.len() != 1
                || values[0].len() != self.envelope.space.dimensions
                || values[0].iter().any(|v| !v.is_finite())
            {
                return Err("invalid dense embedding query or batch cardinality".into());
            }
            self.search
                .index()
                .nearest(&values[0], k)
                .map_err(|e| format!("embedding query: {e:?}"))
        }
    }
}

pub fn envelope(
    provider: &impl EmbeddingProvider,
    corpus: EmbeddingCorpusV1,
) -> Result<EmbeddingEnvelopeV1, String> {
    let space = provider
        .compatibility_space()
        .ok_or("embedding provider has no compatibility identity")?;
    if space.dimensions != provider.dimensions() || metric(space.metric) != provider.metric() {
        return Err("embedding provider differs from its declared space".into());
    }
    let value = EmbeddingEnvelopeV1 {
        version: 1,
        space,
        corpus,
    };
    value.validate()?;
    Ok(value)
}
fn metric(value: EmbeddingMetric) -> SimilarityMetric {
    match value {
        EmbeddingMetric::Cosine => SimilarityMetric::Cosine,
        EmbeddingMetric::Euclidean => SimilarityMetric::Euclidean,
        EmbeddingMetric::DotProduct => SimilarityMetric::DotProduct,
    }
}
fn validate_sparse(vector: &super::SparseVector, dimensions: usize) -> Result<(), String> {
    if vector.dimensions() != dimensions
        || vector
            .entries()
            .iter()
            .any(|&(i, v)| i as usize >= dimensions || !v.is_finite() || v == 0.0)
        || vector
            .entries()
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err("invalid sparse embedding vector".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{LexicalEmbeddingProvider, StubEmbeddingProvider};
    fn corpus() -> EmbeddingCorpusV1 {
        EmbeddingCorpusV1 {
            chunker: "unicode/v1/512".into(),
            sources: vec![],
        }
    }
    #[test]
    fn a_same_dimension_foreign_algorithm_is_refused_before_search() {
        let first = CompatibleSearch::build(
            LexicalEmbeddingProvider::new(384).unwrap(),
            corpus(),
            &[("a", "retained browser history")],
        )
        .unwrap();
        let bytes = serde_json::to_vec(&first.snapshot()).unwrap();
        let decoded: CompatibleIndex<&str> = serde_json::from_slice(&bytes).unwrap();
        let reopened = CompatibleSearch::reopen(
            LexicalEmbeddingProvider::new(384).unwrap(),
            corpus(),
            decoded,
        )
        .unwrap();
        assert_eq!(
            first.search("browser", 1).unwrap(),
            reopened.search("browser", 1).unwrap()
        );
        let foreign = LexicalEmbeddingProvider::with_token_ngram_orders(384, [1, 2]).unwrap();
        assert_eq!(foreign.dimensions(), first.envelope().space.dimensions);
        assert!(
            CompatibleSearch::reopen(foreign, corpus(), first.snapshot())
                .err()
                .unwrap()
                .contains("foreign embedding space")
        );
        let mut changed = corpus();
        changed.chunker.push('2');
        assert!(
            CompatibleSearch::reopen(
                LexicalEmbeddingProvider::new(384).unwrap(),
                changed,
                first.snapshot()
            )
            .err()
            .unwrap()
            .contains("corpus or chunker changed")
        );
        assert!(
            CompatibleSearch::<&str, _>::build(
                StubEmbeddingProvider::new(384).unwrap(),
                corpus(),
                &[]
            )
            .err()
            .unwrap()
            .contains("no compatibility identity")
        );
    }
    #[test]
    fn malformed_persisted_vectors_are_not_trusted_after_deserialization() {
        let first = CompatibleSearch::build(
            LexicalEmbeddingProvider::new(384).unwrap(),
            corpus(),
            &[("a".to_string(), "retained text")],
        )
        .unwrap();
        let mut record = serde_json::to_value(first.snapshot()).unwrap();
        record["vectors"]["index"]["entries"]["a"]["entries"] = serde_json::json!([[384, 1.0]]);
        let decoded: CompatibleIndex<String> = serde_json::from_value(record).unwrap();
        assert!(
            CompatibleSearch::reopen(
                LexicalEmbeddingProvider::new(384).unwrap(),
                corpus(),
                decoded
            )
            .is_err()
        );
        let mut saved = first.snapshot();
        saved.vectors =
            CompatibleVectors::Dense(VectorIndex::new(384, SimilarityMetric::DotProduct));
        assert!(
            CompatibleSearch::reopen(LexicalEmbeddingProvider::new(384).unwrap(), corpus(), saved)
                .is_err()
        );
    }
    #[test]
    fn malformed_query_batches_and_withdrawn_sparse_capability_are_refused() {
        struct Dense;
        impl EmbeddingProvider for Dense {
            fn compatibility_space(&self) -> Option<eidetic::models::embedding::EmbeddingSpaceV1> {
                LexicalEmbeddingProvider::new(384)
                    .unwrap()
                    .compatibility_space()
            }
            fn dimensions(&self) -> usize {
                384
            }
            fn metric(&self) -> SimilarityMetric {
                SimilarityMetric::Cosine
            }
            fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, super::super::EmbedError> {
                Ok(texts
                    .iter()
                    .map(|text| match *text {
                        "missing" => Vec::new(),
                        "nan" => vec![f32::NAN; 384],
                        _ => vec![1.0; 384],
                    })
                    .collect())
            }
        }
        let sparse = CompatibleSearch::build(
            LexicalEmbeddingProvider::new(384).unwrap(),
            corpus(),
            &[("a", "text")],
        )
        .unwrap();
        assert!(
            CompatibleSearch::reopen(Dense, corpus(), sparse.snapshot())
                .err()
                .unwrap()
                .contains("sparse capability")
        );
        let dense = CompatibleSearch::build(Dense, corpus(), &[("a", "text")]).unwrap();
        assert!(!dense.is_sparse());
        assert!(dense.search("missing", 1).is_err());
        assert!(dense.search("nan", 1).is_err());
        assert_eq!(dense.search("valid", 1).unwrap().len(), 1);
    }
}
