//! Flat (dense) vector index — `O(N)` per query.
//!
//! Suitable for graphs up to ~10k nodes; an HNSW-backed implementation will
//! follow once scale becomes a real constraint. The flat shape keeps the
//! crate dependency-free and trivially correct.

use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::embed::provider::SimilarityMetric;

/// Reasons an index operation could fail.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexError {
    /// Inserted vector dimension did not match the index dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// Empty input where a non-empty was required (e.g. `nearest` with `k=0`).
    EmptyInput,
}

/// Flat vector index keyed by `K`.
///
/// Insertion overwrites any prior entry at the same key. Removal returns the
/// previous vector if present.
///
/// Serializable when `K: Serialize + Deserialize`, so a caller can persist the
/// index through its own store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndex<K: Hash + Eq + Clone> {
    dimensions: usize,
    metric: SimilarityMetric,
    entries: HashMap<K, Vec<f32>>,
}

impl<K: Hash + Eq + Clone> VectorIndex<K> {
    pub fn new(dimensions: usize, metric: SimilarityMetric) -> Self {
        Self {
            dimensions,
            metric,
            entries: HashMap::new(),
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn metric(&self) -> SimilarityMetric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn insert(&mut self, key: K, vector: Vec<f32>) -> Result<(), IndexError> {
        if vector.len() != self.dimensions {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }
        self.entries.insert(key, vector);
        Ok(())
    }

    pub fn remove(&mut self, key: &K) -> Option<Vec<f32>> {
        self.entries.remove(key)
    }

    pub fn get(&self, key: &K) -> Option<&Vec<f32>> {
        self.entries.get(key)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Find the `k` keys whose vectors are most similar to `query`.
    /// Returns `(key, score)` pairs sorted best-first per the index metric.
    /// `score` is the raw metric value (cosine ∈ [-1, 1], dot product
    /// unbounded, euclidean ≥ 0).
    pub fn nearest(&self, query: &[f32], k: usize) -> Result<Vec<(K, f32)>, IndexError> {
        if query.len() != self.dimensions {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(IndexError::EmptyInput);
        }
        let mut scored: Vec<(K, f32)> = self
            .entries
            .iter()
            .map(|(key, vec)| (key.clone(), score(self.metric, query, vec)))
            .collect();
        sort_by_metric(&mut scored, self.metric);
        scored.truncate(k);
        Ok(scored)
    }

    /// Iterate `(key, vector)` pairs in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &Vec<f32>)> {
        self.entries.iter()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

fn score(metric: SimilarityMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(a, b),
        SimilarityMetric::DotProduct => dot(a, b),
        SimilarityMetric::Euclidean => euclidean(a, b),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn sort_by_metric<K>(scored: &mut [(K, f32)], metric: SimilarityMetric) {
    if metric.higher_is_better() {
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_index() -> VectorIndex<u32> {
        VectorIndex::new(3, SimilarityMetric::Cosine)
    }

    #[test]
    fn new_index_is_empty() {
        let idx = cosine_index();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.dimensions(), 3);
        assert_eq!(idx.metric(), SimilarityMetric::Cosine);
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        let v = idx.get(&1).unwrap();
        assert_eq!(*v, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn insert_dimension_mismatch_errors() {
        let mut idx = cosine_index();
        let err = idx.insert(1, vec![1.0, 0.0]).unwrap_err();
        assert_eq!(
            err,
            IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn insert_overwrites_existing_key() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(1, vec![0.0, 1.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(*idx.get(&1).unwrap(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn remove_returns_prior_value() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        let removed = idx.remove(&1);
        assert_eq!(removed, Some(vec![1.0, 0.0, 0.0]));
        assert!(idx.is_empty());
    }

    #[test]
    fn remove_missing_returns_none() {
        let mut idx = cosine_index();
        assert!(idx.remove(&42).is_none());
    }

    #[test]
    fn contains_reflects_state() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        assert!(idx.contains(&1));
        assert!(!idx.contains(&2));
    }

    #[test]
    fn nearest_cosine_returns_aligned_first() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.0, 0.0, 1.0]).unwrap();
        let result = idx.nearest(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn nearest_truncates_to_k() {
        let mut idx = cosine_index();
        for i in 0..10 {
            let mut v = vec![0.0; 3];
            v[i % 3] = 1.0;
            idx.insert(i as u32, v).unwrap();
        }
        let result = idx.nearest(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn nearest_dimension_mismatch_errors() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        let err = idx.nearest(&[1.0, 0.0], 1).unwrap_err();
        assert_eq!(
            err,
            IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn nearest_with_k_zero_errors() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        let err = idx.nearest(&[1.0, 0.0, 0.0], 0).unwrap_err();
        assert_eq!(err, IndexError::EmptyInput);
    }

    #[test]
    fn nearest_on_empty_index_returns_empty_vec() {
        let idx = cosine_index();
        let result = idx.nearest(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn euclidean_metric_lower_is_better() {
        let mut idx = VectorIndex::<u32>::new(2, SimilarityMetric::Euclidean);
        idx.insert(1, vec![0.0, 0.0]).unwrap();
        idx.insert(2, vec![10.0, 10.0]).unwrap();
        let result = idx.nearest(&[0.0, 0.1], 2).unwrap();
        assert_eq!(result[0].0, 1); // closer to origin
        assert_eq!(result[1].0, 2);
        assert!(result[0].1 < result[1].1); // smaller distance is better
    }

    #[test]
    fn dot_product_metric_higher_is_better() {
        let mut idx = VectorIndex::<u32>::new(2, SimilarityMetric::DotProduct);
        idx.insert(1, vec![1.0, 0.0]).unwrap();
        idx.insert(2, vec![0.5, 0.0]).unwrap();
        let result = idx.nearest(&[1.0, 0.0], 2).unwrap();
        assert_eq!(result[0].0, 1);
        assert_eq!(result[1].0, 2);
    }

    #[test]
    fn iter_yields_all_entries() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        let collected: Vec<_> = idx.iter().map(|(k, _)| *k).collect();
        assert_eq!(collected.len(), 2);
        assert!(collected.contains(&1));
        assert!(collected.contains(&2));
    }

    #[test]
    fn clear_empties_index() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.clear();
        assert!(idx.is_empty());
    }

    #[test]
    fn cosine_zero_vector_query_returns_zero_scores() {
        let mut idx = cosine_index();
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        let result = idx.nearest(&[0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(result[0].1, 0.0);
    }
}
