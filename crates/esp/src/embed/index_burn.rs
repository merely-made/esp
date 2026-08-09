//! Batched-cosine index kernel (the `index-burn` feature).
//!
//! [`VectorIndex::nearest`](crate::embed::VectorIndex::nearest) scores a query against
//! every entry on the CPU, `O(N·d)` per query. For a batch of `Q` queries that is
//! `O(Q·N·d)` — and that is a **matmul**: normalize both sides to unit rows and
//! the cosine similarity matrix is `queries · corpusᵀ`. So [`cosine_top_k`] runs
//! that one matmul on burn (ndarray, or wgpu under `index-burn-wgpu`), where
//! `O(Q·N·d)` matmul throughput is exactly what the hardware is built for, then
//! takes each query's top-k on the CPU over the read-back scores.
//!
//! This is the same tensor-program shape as a tensorized N-body force pass: a
//! dense pairwise interaction over `[Q,d]` and `[N,d]`, reduced. The GPU wins from
//! roughly a thousand entries up (dispatch + the `[Q,N]` readback dominate below
//! that); the crossover is measured, not assumed.
//!
//! **Honest bounds.** The result is exact (no approximation — this is brute force
//! on faster hardware, not HNSW). The top-k runs on the CPU over the full `[Q,N]`
//! score matrix, so the readback is `O(Q·N)`; the heavy `O(Q·N·d)` matmul is what
//! moves to the GPU. For very large `Q·N` a GPU top-k (returning only `[Q,k]`)
//! would shrink the readback; that is a refinement, not this cut.

use std::collections::HashSet;
use std::hash::Hash;

use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::embed::index::VectorIndex;

/// Entry count at or above which the GPU batched kernel beats the CPU flat scan
/// for **all-pairs** work (arrangement / affinity), from the P2 crossover
/// measurement. Below it the dispatch + `[N,N]` readback dominate the small
/// `O(N²)`, so [`crate::embed::affinity_pairs`] (CPU) wins; a consumer routes on this.
pub const AFFINITY_GPU_MIN_ENTRIES: usize = 1024;

/// Entry count at or above which the GPU kernel beats the CPU flat scan for
/// **few-query** work (recall / search), from the P2 crossover measurement.
pub const SEARCH_GPU_MIN_ENTRIES: usize = 4096;

/// For each query row, the `k` corpus entries most similar by **cosine**, as
/// `(corpus_index, score)` pairs sorted best-first. Rows of `queries` and `corpus`
/// must share the same dimension `d`.
///
/// The `O(Q·N·d)` normalize-and-matmul runs on backend `B`; the per-query top-k
/// runs on the CPU over the read-back `[Q,N]` scores. Zero-length rows (a
/// token-free embedding) score 0 against everything, matching
/// [`VectorIndex`](crate::embed::VectorIndex)'s cosine handling.
pub fn cosine_top_k<B: Backend>(
    queries: &[Vec<f32>],
    corpus: &[Vec<f32>],
    k: usize,
    device: &B::Device,
) -> Vec<Vec<(usize, f32)>> {
    let q = queries.len();
    let n = corpus.len();
    if q == 0 {
        return Vec::new();
    }
    if n == 0 || k == 0 {
        return vec![Vec::new(); q];
    }
    let dim = corpus[0].len();
    let k = k.min(n);

    // [Q, N] cosine similarity as one matmul on unit-normalized rows.
    let qn = l2_normalize_rows(rows_to_tensor::<B>(queries, dim, device));
    let cn = l2_normalize_rows(rows_to_tensor::<B>(corpus, dim, device));
    let sim = qn.matmul(cn.swap_dims(0, 1)); // [Q, d] · [d, N] = [Q, N]
    let scores = sim
        .into_data()
        .to_vec::<f32>()
        .expect("similarity readback is f32");

    // Per-query top-k over the read-back row (cheap O(N) vs the matmul above).
    let mut out = Vec::with_capacity(q);
    for qi in 0..q {
        let row = &scores[qi * n..(qi + 1) * n];
        let mut scored: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        out.push(scored);
    }
    out
}

/// Stack `rows` (each length `dim`) into a `[rows.len(), dim]` tensor on `B`.
fn rows_to_tensor<B: Backend>(rows: &[Vec<f32>], dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut flat = Vec::with_capacity(rows.len() * dim);
    for row in rows {
        flat.extend_from_slice(row);
    }
    Tensor::from_data(TensorData::new(flat, [rows.len(), dim]), device)
}

/// Scale each row of a `[R, d]` tensor to unit L2 length. A zero row (norm ~0)
/// stays ~zero (the `1e-12` guard divides it by epsilon, not by zero), so its
/// cosine against anything is 0 — matching [`VectorIndex`](crate::embed::VectorIndex).
fn l2_normalize_rows<B: Backend>(t: Tensor<B, 2>) -> Tensor<B, 2> {
    let sq_sum = (t.clone() * t.clone()).sum_dim(1); // [R, 1]
    let inv_norm = sq_sum.sqrt().add_scalar(1e-12).recip(); // [R, 1]
    t * inv_norm // [R, d] * [R, 1] broadcasts over the row
}

/// [`cosine_top_k`] over a [`VectorIndex`]'s entries, returning results keyed by
/// the index's own `K` (not corpus positions). The GPU path for recall / search;
/// a consumer routes to it when the index has at least [`SEARCH_GPU_MIN_ENTRIES`].
/// Cosine only.
pub fn nearest_over_index<K, B>(
    index: &VectorIndex<K>,
    queries: &[Vec<f32>],
    k: usize,
    device: &B::Device,
) -> Vec<Vec<(K, f32)>>
where
    K: Hash + Eq + Clone,
    B: Backend,
{
    let entries: Vec<(K, Vec<f32>)> = index
        .iter()
        .map(|(key, v)| (key.clone(), v.clone()))
        .collect();
    let corpus: Vec<Vec<f32>> = entries.iter().map(|(_, v)| v.clone()).collect();
    cosine_top_k::<B>(queries, &corpus, k, device)
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(pos, score)| (entries[pos].0.clone(), score))
                .collect()
        })
        .collect()
}

/// The GPU all-pairs analog of [`affinity_pairs`](crate::embed::affinity_pairs): each
/// entry's `top_k` nearest above `min_similarity`, symmetric-deduped, weight =
/// clamped cosine. Drop-in equivalent to the CPU version (same pairs, weights
/// within float tolerance) but with the all-pairs step on burn. The GPU path for
/// arrangement / affinity; a consumer routes to it when the index has at least
/// [`AFFINITY_GPU_MIN_ENTRIES`]. Cosine only — the index metric is assumed
/// [`Cosine`](crate::embed::SimilarityMetric::Cosine), as `affinity_pairs` intends.
pub fn affinity_pairs_over_index<K, B>(
    index: &VectorIndex<K>,
    top_k: usize,
    min_similarity: f32,
    device: &B::Device,
) -> Vec<(K, K, f32)>
where
    K: Hash + Eq + Clone,
    B: Backend,
{
    let entries: Vec<(K, Vec<f32>)> = index
        .iter()
        .map(|(key, v)| (key.clone(), v.clone()))
        .collect();
    if entries.len() < 2 {
        return Vec::new();
    }
    let vectors: Vec<Vec<f32>> = entries.iter().map(|(_, v)| v.clone()).collect();
    // `+1` because each entry's nearest set includes itself at the top.
    let neighbours = cosine_top_k::<B>(&vectors, &vectors, top_k + 1, device);

    // Identical filter/dedup to `affinity_pairs`, over the GPU-computed neighbours.
    let mut seen: HashSet<(K, K)> = HashSet::new();
    let mut pairs: Vec<(K, K, f32)> = Vec::new();
    for (i, (key, _)) in entries.iter().enumerate() {
        for &(pos, similarity) in &neighbours[i] {
            let neighbour = &entries[pos].0;
            if neighbour == key || similarity < min_similarity {
                continue;
            }
            if seen.contains(&(neighbour.clone(), key.clone())) {
                continue;
            }
            if seen.insert((key.clone(), neighbour.clone())) {
                pairs.push((key.clone(), neighbour.clone(), similarity.clamp(0.0, 1.0)));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::index::VectorIndex;
    use crate::embed::provider::SimilarityMetric;

    /// A small corpus with strictly-distinct cosines to a query (no tie
    /// ambiguity), plus the query.
    fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let corpus = vec![
            vec![1.0, 0.0, 0.0], // 0: cosine 1.000 to the query
            vec![0.9, 0.1, 0.0], // 1: cosine ~0.994
            vec![0.5, 0.5, 0.0], // 2: cosine ~0.707
            vec![0.0, 1.0, 0.0], // 3: cosine 0
            vec![0.0, 0.0, 1.0], // 4: cosine 0
        ];
        let queries = vec![vec![1.0, 0.0, 0.0]];
        (queries, corpus)
    }

    #[test]
    fn matches_vector_index_nearest() {
        type B = burn::backend::NdArray<f32>;
        let (queries, corpus) = fixture();
        let k = 3;
        let got = cosine_top_k::<B>(&queries, &corpus, k, &Default::default());

        // Reference: the CPU flat index this kernel is meant to accelerate.
        let mut idx = VectorIndex::<usize>::new(3, SimilarityMetric::Cosine);
        for (i, v) in corpus.iter().enumerate() {
            idx.insert(i, v.clone()).unwrap();
        }
        let want = idx.nearest(&queries[0], k).unwrap();

        assert_eq!(got[0].len(), want.len(), "same k");
        for ((gi, gs), (wi, ws)) in got[0].iter().zip(&want) {
            assert_eq!(gi, wi, "same ranked index");
            assert!((gs - ws).abs() < 1e-5, "score {gs} vs reference {ws}");
        }
    }

    #[test]
    fn all_pairs_ranks_self_first() {
        // Queries == corpus: each item's nearest is itself at cosine ~1.
        type B = burn::backend::NdArray<f32>;
        let (_, corpus) = fixture();
        let got = cosine_top_k::<B>(&corpus, &corpus, 1, &Default::default());
        for (i, row) in got.iter().enumerate() {
            assert_eq!(row[0].0, i, "item {i} should be its own nearest");
            assert!(
                (row[0].1 - 1.0).abs() < 1e-5,
                "self-cosine ~1, got {}",
                row[0].1
            );
        }
    }

    #[test]
    fn edges_are_handled() {
        type B = burn::backend::NdArray<f32>;
        let dev = Default::default();
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // No queries → no rows.
        assert!(cosine_top_k::<B>(&[], &corpus, 2, &dev).is_empty());
        // Empty corpus / k=0 → one empty result per query.
        assert_eq!(
            cosine_top_k::<B>(&[vec![1.0, 0.0]], &[], 2, &dev),
            vec![Vec::<(usize, f32)>::new()]
        );
        assert_eq!(
            cosine_top_k::<B>(&[vec![1.0, 0.0]], &corpus, 0, &dev),
            vec![Vec::<(usize, f32)>::new()]
        );
        // k clamps to the corpus size.
        assert_eq!(
            cosine_top_k::<B>(&[vec![1.0, 0.0]], &corpus, 99, &dev)[0].len(),
            2
        );
    }

    #[test]
    fn zero_query_scores_zero_everywhere() {
        // A token-free (zero) query vector: cosine 0 to all, like the flat index.
        type B = burn::backend::NdArray<f32>;
        let corpus = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let got = cosine_top_k::<B>(&[vec![0.0, 0.0, 0.0]], &corpus, 2, &Default::default());
        for (_, s) in &got[0] {
            assert!(s.abs() < 1e-6, "zero query scores ~0, got {s}");
        }
    }

    #[test]
    fn nearest_over_index_matches_cpu_nearest() {
        type B = burn::backend::NdArray<f32>;
        let mut index = VectorIndex::<u32>::new(3, SimilarityMetric::Cosine);
        index.insert(10, vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(20, vec![0.9, 0.1, 0.0]).unwrap();
        index.insert(30, vec![0.0, 1.0, 0.0]).unwrap();
        let queries = vec![vec![1.0, 0.0, 0.0]];

        let got = nearest_over_index::<u32, B>(&index, &queries, 2, &Default::default());
        let want = index.nearest(&queries[0], 2).unwrap(); // the CPU path it accelerates
        assert_eq!(got[0].len(), want.len());
        for ((gk, gs), (wk, ws)) in got[0].iter().zip(&want) {
            assert_eq!(gk, wk, "same key ranked");
            assert!((gs - ws).abs() < 1e-5, "score {gs} vs reference {ws}");
        }
    }

    #[test]
    fn affinity_pairs_over_index_matches_cpu() {
        use crate::embed::affinity::affinity_pairs;
        use std::collections::HashMap;
        type B = burn::backend::NdArray<f32>;
        // Two clean clusters (A near axis 0, B near axis 1) — the affinity fixture.
        let mut index = VectorIndex::<u32>::new(4, SimilarityMetric::Cosine);
        for (key, v) in [
            (1u32, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.95, 0.05, 0.0, 0.0]),
            (3, vec![0.9, 0.1, 0.0, 0.0]),
            (4, vec![0.0, 1.0, 0.0, 0.0]),
            (5, vec![0.05, 0.95, 0.0, 0.0]),
            (6, vec![0.1, 0.9, 0.0, 0.0]),
        ] {
            index.insert(key, v).unwrap();
        }
        let cpu = affinity_pairs(&index, 3, 0.8).unwrap();
        let gpu = affinity_pairs_over_index::<u32, B>(&index, 3, 0.8, &Default::default());

        // Compare as canonical (min,max) → weight maps (pair order is not defined).
        let canon = |v: &[(u32, u32, f32)]| -> HashMap<(u32, u32), f32> {
            v.iter()
                .map(|&(a, b, w)| (if a < b { (a, b) } else { (b, a) }, w))
                .collect()
        };
        let (cm, gm) = (canon(&cpu), canon(&gpu));
        assert_eq!(
            cm.len(),
            gm.len(),
            "same pair count: cpu {} gpu {}",
            cm.len(),
            gm.len()
        );
        for (pair, cw) in &cm {
            let gw = gm
                .get(pair)
                .unwrap_or_else(|| panic!("gpu missing pair {pair:?}"));
            assert!((cw - gw).abs() < 1e-5, "weight {cw} vs {gw} for {pair:?}");
        }
    }

    /// ndarray↔wgpu parity: the kernel is backend-generic, so the GPU path must
    /// rank the same and score within float tolerance. Gated on the wgpu backend
    /// (a real GPU); run with `--features index-burn-wgpu`.
    #[cfg(feature = "index-burn-wgpu")]
    #[test]
    fn parity_ndarray_wgpu() {
        type Cpu = burn::backend::NdArray<f32>;
        type Gpu = burn::backend::Wgpu<f32, i32>;
        let (queries, corpus) = fixture();
        let cpu = cosine_top_k::<Cpu>(&queries, &corpus, 4, &Default::default());
        let gpu = cosine_top_k::<Gpu>(&queries, &corpus, 4, &Default::default());
        assert_eq!(cpu.len(), gpu.len());
        for (rc, rg) in cpu.iter().zip(&gpu) {
            assert_eq!(rc.len(), rg.len());
            for ((ci, cs), (gi, gs)) in rc.iter().zip(rg) {
                assert_eq!(ci, gi, "same ranked index across backends");
                assert!((cs - gs).abs() < 1e-4, "score parity {cs} vs {gs}");
            }
        }
    }

    /// P2 crossover: the CPU flat scan (`VectorIndex::nearest` per query) vs the
    /// GPU batched kernel, across `N` and query shape. Ignored + wgpu-gated; run in
    /// **release** (debug burn is far too slow to time honestly):
    ///
    /// ```text
    /// cargo test --release --features index-burn-wgpu -- --ignored crossover --nocapture
    /// ```
    #[cfg(feature = "index-burn-wgpu")]
    #[test]
    #[ignore = "timing sweep; run in release with --ignored --nocapture (see the doc comment)"]
    fn crossover_cpu_flat_vs_gpu_batched() {
        use std::time::Instant;
        type Gpu = burn::backend::Wgpu<f32, i32>;
        let device = Default::default();
        let d = 384; // a realistic embedding dimension (MiniLM-class)
        let k = 10;

        // Deterministic pseudo-random vectors in [-1, 1] (splitmix64).
        fn corpus(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
            let mut s = seed;
            let mut next = move || {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
            };
            (0..n).map(|_| (0..d).map(|_| next()).collect()).collect()
        }

        // Warm the GPU: the first dispatch pays device/pipeline init.
        let w = corpus(64, d, 1);
        let _ = cosine_top_k::<Gpu>(&w, &w, k, &device);

        let run = |label: &str, corpus_v: &[Vec<f32>], queries: &[Vec<f32>]| {
            let n = corpus_v.len();
            let mut idx = VectorIndex::<usize>::new(d, SimilarityMetric::Cosine);
            for (i, v) in corpus_v.iter().enumerate() {
                idx.insert(i, v.clone()).unwrap();
            }
            let t = Instant::now();
            for q in queries {
                let _ = idx.nearest(q, k).unwrap();
            }
            let cpu = t.elapsed().as_secs_f64() * 1000.0;

            let t = Instant::now();
            let _ = cosine_top_k::<Gpu>(queries, corpus_v, k, &device);
            let gpu = t.elapsed().as_secs_f64() * 1000.0;

            println!(
                "{label},{n},{},{d},{cpu:.2},{gpu:.2},{:.2}x",
                queries.len(),
                cpu / gpu
            );
        };

        println!("shape,N,Q,d,cpu_flat_ms,gpu_batched_ms,speedup");
        // All-pairs (Q = N): the arrangement / affinity shape. Capped at 4096 —
        // the CPU side is O(N²·d) and the GPU readback is O(N²).
        for &n in &[256usize, 1024, 4096] {
            let c = corpus(n, d, 7);
            run("all-pairs", &c, &c);
        }
        // Few-query (Q = 8): the recall / search shape.
        for &n in &[1024usize, 4096, 16384] {
            let c = corpus(n, d, 9);
            let q: Vec<Vec<f32>> = c.iter().take(8).cloned().collect();
            run("few-query", &c, &q);
        }
    }
}
