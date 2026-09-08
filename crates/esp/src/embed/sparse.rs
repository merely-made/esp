// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Sparse vectors — the same space as the dense one, stored by its non-zeros.
//!
//! A feature-hashed title has as many non-zero buckets as it has distinct
//! token n-grams (about ten), whatever the dimension; dense pays for every
//! bucket. [`SparseVector`] pays 8 bytes per non-zero instead, and
//! [`SparseIndex`] is [`VectorIndex`] over it — one `nearest`, one
//! [`SimilarityMetric`].
//!
//! Scores are *bit-identical* to the dense ones, not merely close: every
//! omitted term contributes exactly `0.0` to a sum whose non-zero terms are
//! visited in the same ascending-index order. Dense stays for the batched
//! matmul (`index-burn`) and for model-backed providers.

use serde::{Deserialize, Serialize};

use crate::embed::index::{IndexError, IndexVector, VectorIndex};
use crate::embed::provider::SimilarityMetric;

/// A [`VectorIndex`] over sparse vectors.
pub type SparseIndex<K> = VectorIndex<K, SparseVector>;

/// Bytes one entry costs on the heap, before any `HashMap` overhead.
const ENTRY_BYTES: usize = std::mem::size_of::<(u32, f32)>();

/// A vector held as its non-zero `(index, weight)` pairs, ascending by index.
///
/// `dimensions` is carried so the vector still knows its space (the index's
/// mismatch check, and [`Self::to_dense`]). The invariants the scoring code
/// relies on: indices are strictly ascending and all `< dimensions`, and no
/// stored weight is exactly `0.0`, so one vector has one sparse form. The
/// public constructors establish all three.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    dimensions: usize,
    entries: Vec<(u32, f32)>,
}

impl SparseVector {
    /// Build from arbitrary `(index, weight)` pairs: sorted, duplicate indices
    /// summed, entries that sum to zero dropped, out-of-range indices rejected.
    /// Surviving weights are taken as given — see [`Self::l2_normalize`].
    pub fn from_pairs(dimensions: usize, mut pairs: Vec<(u32, f32)>) -> Result<Self, IndexError> {
        if pairs.iter().any(|&(i, _)| i as usize >= dimensions) {
            return Err(IndexError::DimensionMismatch {
                expected: dimensions,
                got: pairs
                    .iter()
                    .map(|&(i, _)| i as usize + 1)
                    .max()
                    .unwrap_or(0),
            });
        }
        // Stable, so pairs at one index keep their arrival order and the sum
        // below reproduces a dense `+=` accumulation exactly.
        pairs.sort_by_key(|&(i, _)| i);
        coalesce(&mut pairs);
        pairs.retain(|&(_, w)| w != 0.0);
        Ok(Self {
            dimensions,
            entries: pairs,
        })
    }

    /// Build from pairs already sorted, coalesced and stripped of zeros. The
    /// hot construction path; the invariants are the caller's to hold.
    pub(crate) fn from_sorted(dimensions: usize, entries: Vec<(u32, f32)>) -> Self {
        Self {
            dimensions,
            entries,
        }
    }

    /// Take the non-zeros of a dense vector. Exact `0.0` entries are dropped.
    pub fn from_dense(dense: &[f32]) -> Self {
        Self {
            dimensions: dense.len(),
            entries: dense
                .iter()
                .enumerate()
                .filter(|&(_, &w)| w != 0.0)
                .map(|(i, &w)| (i as u32, w))
                .collect(),
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Number of stored (non-zero) entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The stored pairs, ascending by index.
    pub fn entries(&self) -> &[(u32, f32)] {
        &self.entries
    }

    /// Heap bytes this vector holds: `8 · entries` (a `(u32, f32)` each), plus
    /// the 32-byte struct itself inline. Dense costs `4 · dimensions`.
    pub fn heap_bytes(&self) -> usize {
        self.entries.capacity() * ENTRY_BYTES
    }

    /// Materialize the dense form.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.dimensions];
        for &(i, w) in &self.entries {
            dense[i as usize] = w;
        }
        dense
    }

    /// L2 length, summed over the stored entries in ascending index order —
    /// the same order and the same terms the dense sum sees.
    pub fn norm(&self) -> f32 {
        self.entries.iter().map(|&(_, w)| w * w).sum::<f32>().sqrt()
    }

    /// Scale to unit L2 length in place. A no-op on the zero vector, matching
    /// the dense normalizer.
    pub fn l2_normalize(&mut self) {
        let norm = self.norm();
        if norm > f32::EPSILON {
            for (_, w) in self.entries.iter_mut() {
                *w /= norm;
            }
        }
    }

    /// Dot product by sorted merge.
    pub fn dot(&self, other: &Self) -> f32 {
        let (mut i, mut j) = (0usize, 0usize);
        let mut acc = 0.0f32;
        while i < self.entries.len() && j < other.entries.len() {
            let (ai, aw) = self.entries[i];
            let (bi, bw) = other.entries[j];
            match ai.cmp(&bi) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    acc += aw * bw;
                    i += 1;
                    j += 1;
                },
            }
        }
        acc
    }

    /// Cosine similarity. `0.0` when either side is degenerate, as dense does.
    pub fn cosine(&self, other: &Self) -> f32 {
        let (na, nb) = (self.norm(), other.norm());
        if na <= f32::EPSILON || nb <= f32::EPSILON {
            return 0.0;
        }
        self.dot(other) / (na * nb)
    }

    /// Euclidean distance by sorted merge: unmatched indices contribute their
    /// own square, matched ones the squared difference.
    pub fn euclidean(&self, other: &Self) -> f32 {
        let (mut i, mut j) = (0usize, 0usize);
        let mut acc = 0.0f32;
        while i < self.entries.len() || j < other.entries.len() {
            let a = self.entries.get(i).copied();
            let b = other.entries.get(j).copied();
            match (a, b) {
                (Some((ai, aw)), Some((bi, bw))) if ai == bi => {
                    acc += (aw - bw).powi(2);
                    i += 1;
                    j += 1;
                },
                (Some((ai, aw)), Some((bi, _))) if ai < bi => {
                    acc += aw.powi(2);
                    i += 1;
                },
                (Some(_), Some((_, bw))) => {
                    acc += bw.powi(2);
                    j += 1;
                },
                (Some((_, aw)), None) => {
                    acc += aw.powi(2);
                    i += 1;
                },
                (None, Some((_, bw))) => {
                    acc += bw.powi(2);
                    j += 1;
                },
                (None, None) => break,
            }
        }
        acc.sqrt()
    }
}

impl IndexVector for SparseVector {
    type Query = SparseVector;

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn query_dimensions(query: &Self) -> usize {
        query.dimensions
    }

    fn score(metric: SimilarityMetric, query: &Self, stored: &Self) -> f32 {
        match metric {
            SimilarityMetric::Cosine => query.cosine(stored),
            SimilarityMetric::DotProduct => query.dot(stored),
            SimilarityMetric::Euclidean => query.euclidean(stored),
        }
    }
}

/// Sum runs of equal indices in a sorted pair list, in place.
fn coalesce(pairs: &mut Vec<(u32, f32)>) {
    let mut write = 0usize;
    let mut read = 0usize;
    while read < pairs.len() {
        let index = pairs[read].0;
        let mut weight = 0.0f32;
        while read < pairs.len() && pairs[read].0 == index {
            weight += pairs[read].1;
            read += 1;
        }
        pairs[write] = (index, weight);
        write += 1;
    }
    pairs.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(v: &[f32]) -> Vec<f32> {
        v.to_vec()
    }

    #[test]
    fn from_pairs_sorts_coalesces_and_range_checks() {
        let v = SparseVector::from_pairs(8, vec![(5, 1.0), (1, 2.0), (5, -0.5)]).unwrap();
        assert_eq!(v.entries(), [(1, 2.0), (5, 0.5)]);
        // A bucket that cancels exactly is not stored at all.
        let cancelled = SparseVector::from_pairs(8, vec![(3, 1.0), (3, -1.0), (6, 1.0)]).unwrap();
        assert_eq!(cancelled.entries(), [(6, 1.0)]);
        assert!(SparseVector::from_pairs(4, vec![(4, 1.0)]).is_err());
    }

    #[test]
    fn dense_roundtrip_preserves_every_bit() {
        let d = dense(&[0.0, -0.25, 0.0, 0.5, 0.0, 0.0, 0.125, 0.0]);
        let s = SparseVector::from_dense(&d);
        assert_eq!(s.len(), 3);
        assert_eq!(s.to_dense(), d);
    }

    #[test]
    fn normalize_matches_the_dense_normalizer() {
        let d = dense(&[0.0, 1.0, 0.0, -1.0, 1.0]);
        let mut s = SparseVector::from_dense(&d);
        s.l2_normalize();
        let expected = 1.0f32 / 3.0f32.sqrt();
        assert_eq!(s.to_dense(), vec![0.0, expected, 0.0, -expected, expected]);
    }

    #[test]
    fn zero_vector_normalizes_to_itself() {
        let mut s = SparseVector::from_dense(&[0.0, 0.0, 0.0]);
        s.l2_normalize();
        assert!(s.is_empty());
        assert_eq!(s.norm(), 0.0);
    }

    #[test]
    fn dot_cosine_and_euclidean_match_the_dense_arithmetic() {
        let a = dense(&[0.0, 0.6, 0.0, 0.8, 0.0]);
        let b = dense(&[0.5, 0.5, 0.0, 0.5, 0.5]);
        let (sa, sb) = (SparseVector::from_dense(&a), SparseVector::from_dense(&b));

        let dense_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert_eq!(sa.dot(&sb), dense_dot);

        let dense_dist: f32 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        assert_eq!(sa.euclidean(&sb), dense_dist);

        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_eq!(sa.cosine(&sb), dense_dot / (na * nb));
    }

    #[test]
    fn degenerate_cosine_is_zero() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let z = SparseVector::from_dense(&[0.0, 0.0]);
        assert_eq!(a.cosine(&z), 0.0);
        assert_eq!(z.cosine(&a), 0.0);
    }

    #[test]
    fn index_over_sparse_shares_the_nearest_surface() {
        let mut idx = SparseIndex::<u32>::new(4, SimilarityMetric::Cosine);
        idx.insert(1, SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        idx.insert(2, SparseVector::from_dense(&[0.0, 1.0, 0.0, 0.0]))
            .unwrap();
        let query = SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]);
        let hits = idx.nearest(&query, 2).unwrap();
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, 1.0);

        let wrong = SparseVector::from_dense(&[1.0, 0.0]);
        assert!(matches!(
            idx.nearest(&wrong, 1),
            Err(IndexError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            idx.insert(3, SparseVector::from_dense(&[1.0])),
            Err(IndexError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            idx.nearest(&query, 0),
            Err(IndexError::EmptyInput)
        ));
    }

    #[test]
    fn heap_bytes_is_eight_per_entry() {
        let v = SparseVector::from_pairs(4096, vec![(3, 1.0), (900, 1.0)]).unwrap();
        assert_eq!(v.heap_bytes(), 2 * ENTRY_BYTES);
        assert_eq!(ENTRY_BYTES, 8);
    }

    #[test]
    fn serde_roundtrip() {
        let v = SparseVector::from_pairs(64, vec![(1, 0.5), (9, -0.5)]).unwrap();
        let back: SparseVector = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(v, back);
    }
}
