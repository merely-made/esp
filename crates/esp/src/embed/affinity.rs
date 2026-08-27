// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Embedding similarity as a **clustering affinity signal**.
//!
//! [`affinity_pairs`] renders item-to-item similarity as a sparse list of
//! `(a, b, weight)` affinity pairs: each entry's top-K nearest neighbours above a
//! threshold, weighted by (clamped) cosine similarity, with symmetric duplicates
//! emitted once. That is the shape a force-directed layout's attract-only pairwise
//! spring consumes, so a set of items carrying embeddings clusters by meaning —
//! semantically-similar items draw together even with no explicit link between
//! them.
//!
//! Pure over a [`VectorIndex`]; no layout or graph dependency. The caller maps the
//! returned key `K` to its own node handle and builds the spring.

use std::collections::HashSet;
use std::hash::Hash;

use crate::embed::index::{IndexError, VectorIndex};

/// Build affinity pairs from an embedding index: for each entry, its `top_k`
/// nearest neighbours with similarity `>= min_similarity`, weight = similarity
/// clamped to `0..=1`. Symmetric duplicates (`a,b` and `b,a`) are emitted once.
///
/// The index's metric decides "nearest" — cosine is the intended one (a
/// bounded `0..=1`-ish score that maps straight to an affinity weight); with an
/// unbounded metric (dot product) the caller should pick `min_similarity`
/// accordingly.
pub fn affinity_pairs<K: Hash + Eq + Clone>(
    index: &VectorIndex<K>,
    top_k: usize,
    min_similarity: f32,
) -> Result<Vec<(K, K, f32)>, IndexError> {
    // Snapshot entries so the per-node `nearest` queries don't hold the iter borrow.
    let entries: Vec<(K, Vec<f32>)> = index
        .iter()
        .map(|(key, vec)| (key.clone(), vec.clone()))
        .collect();

    let mut seen: HashSet<(K, K)> = HashSet::new();
    let mut pairs: Vec<(K, K, f32)> = Vec::new();
    for (key, vec) in &entries {
        // `+1` because the nearest set includes the node itself (similarity to
        // its own vector) at the top.
        let neighbours = index.nearest(vec, top_k + 1)?;
        for (neighbour, similarity) in neighbours {
            if &neighbour == key || similarity < min_similarity {
                continue;
            }
            // Emit each unordered pair once: skip if the reverse already landed.
            if seen.contains(&(neighbour.clone(), key.clone())) {
                continue;
            }
            if seen.insert((key.clone(), neighbour.clone())) {
                pairs.push((key.clone(), neighbour, similarity.clamp(0.0, 1.0)));
            }
        }
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::provider::SimilarityMetric;

    /// Two clean clusters in a 4-D space: A points along axis 0, B along axis 1.
    fn two_cluster_index() -> VectorIndex<u32> {
        let mut index = VectorIndex::new(4, SimilarityMetric::Cosine);
        // Cluster A (near [1,0,0,0]).
        index.insert(1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, vec![0.95, 0.05, 0.0, 0.0]).unwrap();
        index.insert(3, vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        // Cluster B (near [0,1,0,0]).
        index.insert(4, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(5, vec![0.05, 0.95, 0.0, 0.0]).unwrap();
        index.insert(6, vec![0.1, 0.9, 0.0, 0.0]).unwrap();
        index
    }

    fn has_pair(pairs: &[(u32, u32, f32)], a: u32, b: u32) -> bool {
        pairs
            .iter()
            .any(|&(x, y, _)| (x == a && y == b) || (x == b && y == a))
    }

    #[test]
    fn pairs_connect_within_clusters_not_across() {
        let index = two_cluster_index();
        // Threshold above the cross-cluster cosine (~0.05-0.14) but below the
        // intra-cluster one (~0.99).
        let pairs = affinity_pairs(&index, 3, 0.8).unwrap();
        // Every intra-A and intra-B pair present.
        for &(a, b) in &[(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)] {
            assert!(has_pair(&pairs, a, b), "missing intra-cluster pair {a},{b}");
        }
        // No cross-cluster pair (an A node with a B node) leaks in.
        for a in [1, 2, 3] {
            for b in [4, 5, 6] {
                assert!(
                    !has_pair(&pairs, a, b),
                    "cross-cluster pair {a},{b} should be below threshold"
                );
            }
        }
    }

    #[test]
    fn symmetric_pairs_are_deduped() {
        let index = two_cluster_index();
        let pairs = affinity_pairs(&index, 5, 0.8).unwrap();
        let mut seen = HashSet::new();
        for &(a, b, _) in &pairs {
            let canon = if a < b { (a, b) } else { (b, a) };
            assert!(seen.insert(canon), "duplicate pair {a},{b}");
        }
    }

    #[test]
    fn weights_are_clamped_and_similarity_ordered() {
        let index = two_cluster_index();
        let pairs = affinity_pairs(&index, 3, 0.8).unwrap();
        for &(_, _, w) in &pairs {
            assert!((0.0..=1.0).contains(&w), "weight out of range: {w}");
        }
        // The tightest intra-cluster pair (1,2 at ~0.999) outweighs a looser one
        // (1,3 at ~0.995).
        let w = |a: u32, b: u32| {
            pairs
                .iter()
                .find(|&&(x, y, _)| (x == a && y == b) || (x == b && y == a))
                .map(|&(_, _, w)| w)
                .unwrap()
        };
        assert!(
            w(1, 2) >= w(1, 3),
            "closer pair should weigh at least as much"
        );
    }

    #[test]
    fn empty_index_yields_no_pairs() {
        let index = VectorIndex::<u32>::new(4, SimilarityMetric::Cosine);
        assert!(affinity_pairs(&index, 3, 0.8).unwrap().is_empty());
    }
}
