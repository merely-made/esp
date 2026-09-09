// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Sparse-vs-dense parity for the lexical lane, plus the `#[ignore]`d scale
//! receipt behind it.
//!
//! The claim under test is stronger than "close": the sparse path omits only
//! terms worth exactly `0.0`, and visits the rest in the same ascending-index
//! order as the dense path, so every float it produces is bit-identical.
//! Tolerance is therefore **zero** — the assertions compare `f32::to_bits`.
//!
//! Ranking is identical too, and for a separate reason: `nearest` breaks equal
//! scores by key, so neither index's result depends on hash-map iteration
//! order. The assertions below compare whole ranked lists, keys included.

use esp::embed::{
    EmbeddingProvider, LexicalEmbeddingProvider, RECOMMENDED_DIMENSIONS_SHORT_TEXT,
    SimilarityMetric, SparseIndex, SparseVector, VectorIndex,
};

/// A mixed corpus: ASCII, Unicode, repeated tokens, punctuation, empty and
/// token-free texts, long and one-word texts.
fn corpus() -> Vec<&'static str> {
    vec![
        "rust async runtime",
        "Rust, Async!",
        "async rust internals and the rust runtime rust rust",
        "",
        "   ",
        "!!! --- ???",
        "naïve café façade",
        "ЖУРНАЛ передач — Москва",
        "日本語 の トークン",
        "emoji 🌱 seedling 🌱 seedling",
        "one",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "MiXeD CaSe TOKENS mixed case tokens",
        "hyphen-separated.dotted:colon/slashed?query=1",
        "the quick brown fox jumps over the lazy dog the fox",
        "italian dinner recipes",
        "open downloads folder",
        "folder downloads open",
        "0123 4567 89 numbers 0123",
        "trailing whitespace and punctuation ...   ",
    ]
}

fn providers() -> Vec<(String, LexicalEmbeddingProvider)> {
    let mut out = Vec::new();
    for dims in [64usize, 256, 512, 4096] {
        for orders in [vec![1usize], vec![1, 2], vec![1, 2, 3], vec![2, 3]] {
            let name = format!("dims={dims} orders={orders:?}");
            out.push((
                name,
                LexicalEmbeddingProvider::with_token_ngram_orders(dims, &orders).unwrap(),
            ));
        }
    }
    out
}

/// Bit-for-bit equality of two float vectors.
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn sparse_to_dense_is_bit_identical_over_a_mixed_corpus() {
    for (setting, provider) in providers() {
        for text in corpus() {
            let dense = provider.embed_one(text).unwrap();
            let sparse = provider.embed_sparse_one(text);
            assert_eq!(sparse.dimensions(), provider.dimensions());
            assert!(
                bits_equal(&sparse.to_dense(), &dense),
                "dense/sparse divergence at {setting} on {text:?}"
            );
            // And the trait's batch form agrees with the singular one.
            let batch = provider.embed_sparse(&[text]).unwrap().unwrap();
            assert_eq!(batch[0], sparse);
        }
    }
}

#[test]
fn sparse_holds_only_the_occupied_buckets() {
    let provider = LexicalEmbeddingProvider::new(4096).unwrap();
    let sparse = provider.embed_sparse_one("rust async runtime");
    let dense = provider.embed_one("rust async runtime").unwrap();
    assert_eq!(sparse.len(), 3);
    assert_eq!(dense.iter().filter(|&&x| x != 0.0).count(), 3);
    // 8 bytes per entry against 4 bytes per dimension.
    assert_eq!(sparse.heap_bytes(), 24);
    assert_eq!(dense.len() * 4, 16_384);
}

#[test]
fn tokenless_text_is_the_empty_sparse_vector() {
    let provider = LexicalEmbeddingProvider::new(256).unwrap();
    for text in ["", "   ", "!!! ---"] {
        let sparse = provider.embed_sparse_one(text);
        assert!(sparse.is_empty(), "{text:?} should hash to nothing");
        assert!(sparse.to_dense().iter().all(|&x| x == 0.0));
    }
}

#[test]
fn sparse_and_dense_scores_are_bit_identical() {
    for (setting, provider) in providers() {
        let texts = corpus();
        for (i, a) in texts.iter().enumerate() {
            for b in texts.iter().skip(i) {
                let (da, db) = (
                    provider.embed_one(a).unwrap(),
                    provider.embed_one(b).unwrap(),
                );
                let (sa, sb) = (provider.embed_sparse_one(a), provider.embed_sparse_one(b));

                let dense_dot: f32 = da.iter().zip(&db).map(|(x, y)| x * y).sum();
                assert_eq!(
                    sa.dot(&sb).to_bits(),
                    dense_dot.to_bits(),
                    "dot at {setting} on {a:?}/{b:?}"
                );

                let na: f32 = da.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = db.iter().map(|x| x * x).sum::<f32>().sqrt();
                let dense_cos = if na <= f32::EPSILON || nb <= f32::EPSILON {
                    0.0
                } else {
                    dense_dot / (na * nb)
                };
                assert_eq!(
                    sa.cosine(&sb).to_bits(),
                    dense_cos.to_bits(),
                    "cosine at {setting} on {a:?}/{b:?}"
                );

                let dense_dist: f32 = da
                    .iter()
                    .zip(&db)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert_eq!(
                    sa.euclidean(&sb).to_bits(),
                    dense_dist.to_bits(),
                    "euclidean at {setting} on {a:?}/{b:?}"
                );
            }
        }
    }
}

#[test]
fn nearest_agrees_between_the_dense_and_sparse_indexes() {
    for (setting, provider) in providers() {
        let texts = corpus();
        let mut dense_index = VectorIndex::<usize>::new(provider.dimensions(), provider.metric());
        let mut sparse_index = SparseIndex::<usize>::new(provider.dimensions(), provider.metric());
        for (id, text) in texts.iter().enumerate() {
            dense_index
                .insert(id, provider.embed_one(text).unwrap())
                .unwrap();
            sparse_index
                .insert(id, provider.embed_sparse_one(text))
                .unwrap();
        }
        assert_eq!(dense_index.len(), sparse_index.len());

        for query in [
            "rust runtime",
            "downloads folder",
            "café",
            "日本語",
            "nothing here",
        ] {
            let k = texts.len();
            let dense_hits = dense_index
                .nearest(&provider.embed_one(query).unwrap(), k)
                .unwrap();
            let sparse_hits = sparse_index
                .nearest(&provider.embed_sparse_one(query), k)
                .unwrap();
            assert_eq!(dense_hits.len(), sparse_hits.len());

            // Key *and* score at every rank, ties included — no tolerance and
            // no set comparison to hide a swap behind.
            for (rank, (d, s)) in dense_hits.iter().zip(&sparse_hits).enumerate() {
                assert_eq!(d.0, s.0, "key at rank {rank} for {query:?} at {setting}");
                assert_eq!(
                    d.1.to_bits(),
                    s.1.to_bits(),
                    "score at rank {rank} for {query:?} at {setting}"
                );
            }
            // And the top-10 prefix, which is what a caller reads.
            let cut = 10.min(dense_hits.len());
            assert_eq!(
                dense_hits[..cut],
                sparse_hits[..cut],
                "top-{cut} for {query:?} at {setting}"
            );
        }
    }
}

#[test]
fn sparse_index_serializes_like_the_dense_one() {
    let provider = LexicalEmbeddingProvider::new(512).unwrap();
    let mut index = SparseIndex::<String>::new(provider.dimensions(), provider.metric());
    for text in corpus() {
        index
            .insert(text.to_string(), provider.embed_sparse_one(text))
            .unwrap();
    }
    let json = serde_json::to_string(&index).unwrap();
    let back: SparseIndex<String> = serde_json::from_str(&json).unwrap();
    assert_eq!(back.len(), index.len());
    assert_eq!(back.metric(), SimilarityMetric::Cosine);
    for text in corpus() {
        assert_eq!(back.get(&text.to_string()), index.get(&text.to_string()));
    }
}

#[test]
fn from_dense_recovers_the_provider_vector() {
    let provider = LexicalEmbeddingProvider::with_token_ngram_orders(512, [1, 2]).unwrap();
    for text in corpus() {
        let dense = provider.embed_one(text).unwrap();
        assert_eq!(
            SparseVector::from_dense(&dense),
            provider.embed_sparse_one(text),
            "from_dense should agree with the direct sparse path on {text:?}"
        );
    }
}

#[test]
fn collision_rate_is_measurable_and_falls_as_dimensions_rise() {
    let texts = corpus();
    let mut previous = f32::INFINITY;
    for dims in [16usize, 64, 256, 512, 4096] {
        let provider = LexicalEmbeddingProvider::new(dims).unwrap();
        let stats = provider.hashing_stats(texts.iter().copied());
        assert_eq!(stats.dimensions, dims);
        assert_eq!(stats.texts, texts.len());
        assert!(stats.distinct_features > 0 && stats.features >= stats.distinct_features);
        assert!(stats.occupied_buckets <= stats.distinct_features.min(dims));
        assert!((0.0..=1.0).contains(&stats.collision_rate));
        assert!(
            stats.collision_rate <= previous,
            "collision rate rose from {previous} to {} at {dims} dims",
            stats.collision_rate
        );
        previous = stats.collision_rate;
    }
    // The documented range is a constant a caller can read, not folklore.
    assert_eq!(*RECOMMENDED_DIMENSIONS_SHORT_TEXT.start(), 256);
    assert_eq!(*RECOMMENDED_DIMENSIONS_SHORT_TEXT.end(), 512);
    let recommended =
        LexicalEmbeddingProvider::new(*RECOMMENDED_DIMENSIONS_SHORT_TEXT.start()).unwrap();
    assert!(
        recommended
            .hashing_stats(texts.iter().copied())
            .collision_rate
            < 0.25
    );
}

#[test]
fn hashing_stats_on_an_empty_corpus_is_zero() {
    let provider = LexicalEmbeddingProvider::new(256).unwrap();
    let stats = provider.hashing_stats(std::iter::empty());
    assert_eq!(stats.texts, 0);
    assert_eq!(stats.distinct_features, 0);
    assert_eq!(stats.collision_rate, 0.0);
}

// --- the scale receipt --------------------------------------------------

/// 42k synthetic titles, dense against sparse, at 4,096 and at 512 dimensions.
///
/// `cargo test -p esp --release --test sparse_dense_parity -- --ignored --nocapture`
#[test]
#[ignore = "measurement, not a correctness gate; run with --release --nocapture"]
fn sparse_vs_dense_scale_receipt() {
    use std::time::Instant;

    const PAGES: usize = 42_000;
    const QUERIES: usize = 100;
    const TOP_K: usize = 10;
    // One rank past the cut, so a tie straddling rank 10 can be recognised
    // rather than blamed on the representation.
    const PROBE_K: usize = TOP_K + 1;

    let titles = synthetic_titles(PAGES);
    let queries: Vec<String> = (0..QUERIES)
        .map(|i| {
            let t = &titles[(i * 397) % titles.len()];
            t.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
        })
        .collect();

    println!(
        "\n{:>6} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "dims", "storage", "bytes/vec", "total MB", "ingest ms", "query ms/100", "top10 equal"
    );

    for dims in [4096usize, 512] {
        let provider = LexicalEmbeddingProvider::new(dims).unwrap();
        let stats = provider.hashing_stats(titles.iter().map(String::as_str));

        // Dense.
        let mut dense_index = VectorIndex::<u32>::new(dims, provider.metric());
        let start = Instant::now();
        for (id, title) in titles.iter().enumerate() {
            dense_index
                .insert(id as u32, provider.embed_one(title).unwrap())
                .unwrap();
        }
        let dense_ingest = start.elapsed().as_secs_f64() * 1e3;
        let dense_queries: Vec<Vec<f32>> = queries
            .iter()
            .map(|q| provider.embed_one(q).unwrap())
            .collect();
        let start = Instant::now();
        let mut dense_tops: Vec<Vec<(u32, f32)>> = Vec::with_capacity(QUERIES);
        for q in &dense_queries {
            dense_tops.push(dense_index.nearest(q, PROBE_K).unwrap());
        }
        let dense_query = start.elapsed().as_secs_f64() * 1e3;
        let dense_bytes = dims * 4;

        // Sparse.
        let mut sparse_index = SparseIndex::<u32>::new(dims, provider.metric());
        let start = Instant::now();
        for (id, title) in titles.iter().enumerate() {
            sparse_index
                .insert(id as u32, provider.embed_sparse_one(title))
                .unwrap();
        }
        let sparse_ingest = start.elapsed().as_secs_f64() * 1e3;
        let sparse_queries: Vec<SparseVector> = queries
            .iter()
            .map(|q| provider.embed_sparse_one(q))
            .collect();
        let start = Instant::now();
        let mut sparse_tops: Vec<Vec<(u32, f32)>> = Vec::with_capacity(QUERIES);
        for q in &sparse_queries {
            sparse_tops.push(sparse_index.nearest(q, PROBE_K).unwrap());
        }
        let sparse_query = start.elapsed().as_secs_f64() * 1e3;
        let sparse_heap: usize = sparse_index.iter().map(|(_, v)| v.heap_bytes()).sum();
        let sparse_bytes = sparse_heap as f64 / PAGES as f64 + 32.0;

        // Key and score at every rank must agree exactly. A tie straddling the
        // cut used to let the top-10 diverge; the key tiebreak in `nearest`
        // closed that, so `cut_ties` is now reported, not excused.
        let mut identical = 0usize;
        let mut cut_ties = 0usize;
        for (d, s) in dense_tops.iter().zip(&sparse_tops) {
            for (rank, (a, b)) in d.iter().zip(s).enumerate() {
                assert_eq!(a.0, b.0, "key at rank {rank} diverged at {dims} dims");
                assert_eq!(
                    a.1.to_bits(),
                    b.1.to_bits(),
                    "score at rank {rank} diverged at {dims} dims"
                );
            }
            cut_ties += usize::from(d.len() > TOP_K && d[TOP_K - 1].1 == d[TOP_K].1);
            assert_eq!(
                d[..TOP_K],
                s[..TOP_K],
                "top-{TOP_K} differed at {dims} dims"
            );
            identical += 1;
        }

        for (label, bytes, total, ingest, query) in [
            (
                "dense",
                dense_bytes as f64,
                (dense_bytes * PAGES) as f64 / 1e6,
                dense_ingest,
                dense_query,
            ),
            (
                "sparse",
                sparse_bytes,
                sparse_bytes * PAGES as f64 / 1e6,
                sparse_ingest,
                sparse_query,
            ),
        ] {
            println!(
                "{dims:>6} {label:>8} {bytes:>12.1} {total:>12.1} {ingest:>12.1} {query:>12.1} {:>12}",
                format!("{identical}/{QUERIES}")
            );
        }
        println!(
            "       collision rate {:.4} over {} distinct n-grams in {} buckets; \
             every score bit-identical; {cut_ties}/{QUERIES} queries tie across \
             the rank-{TOP_K} cut",
            stats.collision_rate, stats.distinct_features, stats.occupied_buckets
        );
    }
}

/// Deterministic, title-shaped synthetic corpus: a handful of words drawn from
/// a small vocabulary, in the shape of browser page titles.
fn synthetic_titles(count: usize) -> Vec<String> {
    const WORDS: [&str; 32] = [
        "rust", "async", "runtime", "docs", "guide", "issue", "pull", "request", "search",
        "results", "index", "vector", "sparse", "dense", "hash", "bucket", "browser", "history",
        "profile", "release", "notes", "crate", "module", "trait", "impl", "test", "bench",
        "corpus", "token", "ngram", "cosine", "recall",
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..count)
        .map(|i| {
            let len = 4 + (next() % 7) as usize;
            let mut words: Vec<&str> = (0..len)
                .map(|_| WORDS[(next() % WORDS.len() as u64) as usize])
                .collect();
            // A per-page unique token, as a real title's distinguishing noun.
            let unique = format!("page{i}");
            words.push(&unique);
            words.join(" ")
        })
        .collect()
}
