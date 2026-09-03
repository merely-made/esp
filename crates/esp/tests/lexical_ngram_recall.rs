// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Fixed-corpus receipt for ESP token n-gram lexical recall and the existing
//! Eidetic BM25/vector fusion seam.
//!
//! The target records and held-out queries model five application surfaces:
//! browsing actions, page titles, URLs, named entities, and commands. Each
//! phrase-sensitive target competes with a shorter record containing the same
//! unigrams in a different order. Five controls cover ordinary unigram or
//! order-insensitive recall. Nothing is trained; every configuration sees the
//! same documents and queries through the existing `SemanticSearch` and dense
//! `VectorIndex` derived-index boundary. A second receipt projects those same
//! records into a real `TrailIndex`, then compares BM25, each vector ranking,
//! and reciprocal-rank fusion. The cross-crate dependency is test-only: the
//! caller continues to bring both rankings to Eidetic's engine-agnostic seam.

#[cfg(feature = "bert")]
use std::fmt::Write as _;
use std::hint::black_box;
#[cfg(feature = "bert")]
use std::io::Read;
use std::mem::size_of;
#[cfg(feature = "bert")]
use std::path::{Path, PathBuf};
use std::time::Instant;

use eidetic::browsing::{BrowsingTrace, PageRef, TraceEvent, TraceTransition};
use eidetic_search::{TrailIndex, fuse};
use esp::embed::{EmbeddingProvider, LexicalEmbeddingProvider, SemanticSearch};
#[cfg(feature = "bert")]
use sha2::{Digest, Sha256};

const DIMENSIONS: usize = 4096;

const DOCUMENTS: &[(&str, &str)] = &[
    ("browse-storage", "Browser Storage Quota settings"),
    (
        "browse-storage-decoy",
        "Quota browser report: storage totals by site",
    ),
    ("browse-tab", "Restore Closed Tab history"),
    ("browse-tab-decoy", "Tab restore: closed"),
    ("title-local-first", "Local First Software essay"),
    ("title-local-first-decoy", "Software first: local"),
    ("title-device-policy", "Shared Device Policy notes"),
    ("title-device-policy-decoy", "Device policy: shared"),
    (
        "url-rust-async",
        "https://example.test/rust/async/book/getting-started",
    ),
    ("url-rust-async-decoy", "https://rust.book/async"),
    (
        "url-graph-query",
        "https://api.example.test/graph/query/nodes/by-label",
    ),
    ("url-graph-query-decoy", "https://graph.nodes/query"),
    ("entity-nyt", "New York Times profile"),
    ("entity-nyt-decoy", "Times: New York"),
    ("entity-wapo", "Washington Post Company profile"),
    ("entity-wapo-decoy", "Company post: Washington"),
    ("command-downloads", "Open Downloads Folder command"),
    ("command-downloads-decoy", "Folder Downloads: Open"),
    ("command-tabs", "Close Other Tabs command"),
    ("command-tabs-decoy", "Tabs menu: other close actions"),
    ("control-wgpu", "WGPU render pipeline diagnostics"),
    (
        "control-offline-sync",
        "Offline browsing sync across owned devices",
    ),
    (
        "control-entity-aliases",
        "Entity aliases and provenance records",
    ),
    ("control-pin-page", "Pin Current Page to workspace"),
    ("control-reader", "Reader mode typography settings"),
    (
        "background-radio",
        "Reticulum packet radio link budget notes",
    ),
    ("background-music", "Fingerstyle guitar practice journal"),
    ("background-cooking", "Weeknight lentil soup recipe"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Browsing,
    Title,
    Url,
    Entity,
    Command,
    Control,
}

#[derive(Clone, Copy)]
struct EvalCase {
    surface: Surface,
    query: &'static str,
    expected: &'static str,
}

const EVAL_CASES: &[EvalCase] = &[
    EvalCase {
        surface: Surface::Browsing,
        query: "browser storage quota",
        expected: "browse-storage",
    },
    EvalCase {
        surface: Surface::Browsing,
        query: "restore closed tab",
        expected: "browse-tab",
    },
    EvalCase {
        surface: Surface::Title,
        query: "local first software",
        expected: "title-local-first",
    },
    EvalCase {
        surface: Surface::Title,
        query: "shared device policy",
        expected: "title-device-policy",
    },
    EvalCase {
        surface: Surface::Url,
        query: "rust async book",
        expected: "url-rust-async",
    },
    EvalCase {
        surface: Surface::Url,
        query: "graph query nodes",
        expected: "url-graph-query",
    },
    EvalCase {
        surface: Surface::Entity,
        query: "new york times",
        expected: "entity-nyt",
    },
    EvalCase {
        surface: Surface::Entity,
        query: "washington post company",
        expected: "entity-wapo",
    },
    EvalCase {
        surface: Surface::Command,
        query: "open downloads folder",
        expected: "command-downloads",
    },
    EvalCase {
        surface: Surface::Command,
        query: "close other tabs",
        expected: "command-tabs",
    },
    EvalCase {
        surface: Surface::Control,
        query: "wgpu",
        expected: "control-wgpu",
    },
    EvalCase {
        surface: Surface::Control,
        query: "offline sync",
        expected: "control-offline-sync",
    },
    EvalCase {
        surface: Surface::Control,
        query: "provenance aliases",
        expected: "control-entity-aliases",
    },
    EvalCase {
        surface: Surface::Control,
        query: "pin page",
        expected: "control-pin-page",
    },
    EvalCase {
        surface: Surface::Control,
        query: "typography settings",
        expected: "control-reader",
    },
];

type Search = SemanticSearch<&'static str, LexicalEmbeddingProvider>;
#[cfg(feature = "bert")]
type DenseSearch = SemanticSearch<&'static str, Box<dyn EmbeddingProvider>>;
type RankedCases = Vec<(Surface, &'static str, usize)>;
type OptionalRankedCases = Vec<(Surface, &'static str, Option<usize>)>;

fn build_search(orders: &[usize]) -> Search {
    let provider = LexicalEmbeddingProvider::with_token_ngram_orders(DIMENSIONS, orders).unwrap();
    let mut search = SemanticSearch::new(provider);
    search.ingest_batch(DOCUMENTS).unwrap();
    search
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RankingTally {
    at_1: usize,
    at_3: usize,
    phrase_at_1: usize,
    controls_at_1: usize,
}

fn evaluate(search: &Search) -> (RankingTally, RankedCases) {
    let mut tally = RankingTally::default();
    let mut ranks = Vec::with_capacity(EVAL_CASES.len());
    for case in EVAL_CASES {
        let ranking = search.search(case.query, DOCUMENTS.len()).unwrap();
        let rank = ranking
            .iter()
            .position(|(key, _)| key == &case.expected)
            .map(|zero_based| zero_based + 1)
            .expect("every expected record is indexed");
        tally.at_1 += usize::from(rank <= 1);
        tally.at_3 += usize::from(rank <= 3);
        if case.surface == Surface::Control {
            tally.controls_at_1 += usize::from(rank <= 1);
        } else {
            tally.phrase_at_1 += usize::from(rank <= 1);
        }
        ranks.push((case.surface, case.query, rank));
    }
    (tally, ranks)
}

fn record_url(id: &str) -> String {
    let (_, text) = DOCUMENTS
        .iter()
        .find(|(document_id, _)| *document_id == id)
        .expect("record id belongs to the fixed corpus");
    if id.starts_with("url-") {
        (*text).to_string()
    } else {
        format!("https://fixture.test/{id}")
    }
}

fn browsing_trace() -> BrowsingTrace {
    let events = DOCUMENTS
        .iter()
        .enumerate()
        .map(|(index, (id, text))| TraceEvent {
            from: None,
            to: PageRef {
                url: record_url(id),
                title: (!id.starts_with("url-")).then(|| (*text).to_string()),
            },
            transition: TraceTransition::Imported,
            at_ms: index as u64 + 1,
            dwell_ms: None,
            candidates: Vec::new(),
        })
        .collect();
    BrowsingTrace::from_events("held-out", events)
}

fn lexical_ranking(trail: &TrailIndex, query: &str) -> Vec<String> {
    trail
        .search(query, DOCUMENTS.len())
        .unwrap()
        .into_iter()
        .map(|hit| hit.url)
        .collect()
}

fn vector_ranking<P: EmbeddingProvider>(
    search: &SemanticSearch<&'static str, P>,
    query: &str,
) -> Vec<String> {
    search
        .search(query, DOCUMENTS.len())
        .unwrap()
        .into_iter()
        .map(|(id, _)| record_url(id))
        .collect()
}

fn evaluate_ranker(
    ranker: impl Fn(&EvalCase) -> Vec<String>,
) -> (RankingTally, OptionalRankedCases) {
    let mut tally = RankingTally::default();
    let mut ranks = Vec::with_capacity(EVAL_CASES.len());
    for case in EVAL_CASES {
        let ranking = ranker(case);
        let expected = record_url(case.expected);
        let rank = ranking
            .iter()
            .position(|candidate| candidate == &expected)
            .map(|zero_based| zero_based + 1);
        tally.at_1 += usize::from(rank.is_some_and(|rank| rank <= 1));
        tally.at_3 += usize::from(rank.is_some_and(|rank| rank <= 3));
        if case.surface == Surface::Control {
            tally.controls_at_1 += usize::from(rank.is_some_and(|rank| rank <= 1));
        } else {
            tally.phrase_at_1 += usize::from(rank.is_some_and(|rank| rank <= 1));
        }
        ranks.push((case.surface, case.query, rank));
    }
    (tally, ranks)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FusionTally {
    deterministic: RankingTally,
    unique_at_1: usize,
    phrase_unique_at_1: usize,
    controls_unique_at_1: usize,
    expected_top_ties: usize,
}

fn evaluate_fusion<P: EmbeddingProvider>(
    trail: &TrailIndex,
    vector: &SemanticSearch<&'static str, P>,
    weights: (f64, f64),
) -> (FusionTally, RankedCases) {
    let mut tally = FusionTally::default();
    let mut ranks = Vec::with_capacity(EVAL_CASES.len());
    for case in EVAL_CASES {
        let fused = fuse(
            &lexical_ranking(trail, case.query),
            &vector_ranking(vector, case.query),
            60.0,
            weights,
        );
        let expected = record_url(case.expected);
        let rank = fused
            .iter()
            .position(|hit| hit.url == expected)
            .map(|zero_based| zero_based + 1)
            .expect("every expected record is returned by fusion");
        tally.deterministic.at_1 += usize::from(rank <= 1);
        tally.deterministic.at_3 += usize::from(rank <= 3);
        if case.surface == Surface::Control {
            tally.deterministic.controls_at_1 += usize::from(rank <= 1);
        } else {
            tally.deterministic.phrase_at_1 += usize::from(rank <= 1);
        }

        let top_score = fused.first().unwrap().score;
        let expected_score = fused.iter().find(|hit| hit.url == expected).unwrap().score;
        let expected_is_top = (expected_score - top_score).abs() < f64::EPSILON;
        let top_count = fused
            .iter()
            .take_while(|hit| (hit.score - top_score).abs() < f64::EPSILON)
            .count();
        let unique_at_1 = expected_is_top && top_count == 1;
        tally.unique_at_1 += usize::from(unique_at_1);
        tally.expected_top_ties += usize::from(expected_is_top && top_count > 1);
        if case.surface == Surface::Control {
            tally.controls_unique_at_1 += usize::from(unique_at_1);
        } else {
            tally.phrase_unique_at_1 += usize::from(unique_at_1);
        }
        ranks.push((case.surface, case.query, rank));
    }
    (tally, ranks)
}

#[test]
fn held_out_phrase_ranking_improves_without_control_regressions() {
    let unigram = build_search(&[1]);
    let bigram = build_search(&[1, 2]);
    let trigram = build_search(&[1, 2, 3]);

    let (unigram_tally, unigram_ranks) = evaluate(&unigram);
    let (bigram_tally, bigram_ranks) = evaluate(&bigram);
    let (trigram_tally, trigram_ranks) = evaluate(&trigram);

    assert_eq!(
        unigram_tally,
        RankingTally {
            at_1: 7,
            at_3: 15,
            phrase_at_1: 2,
            controls_at_1: 5,
        },
        "unigram baseline drifted: {unigram_ranks:?}"
    );
    for (label, tally, ranks) in [
        ("unigram+bigram", bigram_tally, bigram_ranks),
        ("unigram+bigram+trigram", trigram_tally, trigram_ranks),
    ] {
        assert_eq!(
            tally,
            RankingTally {
                at_1: 15,
                at_3: 15,
                phrase_at_1: 10,
                controls_at_1: 5,
            },
            "{label} receipt drifted: {ranks:?}"
        );
    }

    // Trigrams are exercised but earn no ranking win over bigrams on this
    // corpus. That distinction belongs in the receipt rather than a default.
    assert_eq!(bigram_tally, trigram_tally);
}

#[test]
fn held_out_bm25_and_fusion_baselines_are_explicit() {
    let directory = tempfile::tempdir().unwrap();
    let trace = browsing_trace();
    let trail = TrailIndex::rebuild(directory.path().join("trail"), [&trace]).unwrap();
    let unigram = build_search(&[1]);
    let bigram = build_search(&[1, 2]);

    let bm25_baseline = RankingTally {
        at_1: 7,
        at_3: 15,
        phrase_at_1: 2,
        controls_at_1: 5,
    };
    let vector_baseline = RankingTally {
        at_1: 7,
        at_3: 15,
        phrase_at_1: 2,
        controls_at_1: 5,
    };
    let phrase_sensitive = RankingTally {
        at_1: 15,
        at_3: 15,
        phrase_at_1: 10,
        controls_at_1: 5,
    };

    let (bm25, bm25_ranks) = evaluate_ranker(|case| lexical_ranking(&trail, case.query));
    let (unigram_vector, unigram_ranks) =
        evaluate_ranker(|case| vector_ranking(&unigram, case.query));
    let (bigram_vector, bigram_ranks) = evaluate_ranker(|case| vector_ranking(&bigram, case.query));
    assert_eq!(bm25, bm25_baseline, "BM25 baseline drifted: {bm25_ranks:?}");
    assert!(
        bm25_ranks.iter().all(|(_, _, rank)| rank.is_some()),
        "tokenized URL components must keep every expected record recallable: {bm25_ranks:?}"
    );
    assert_eq!(
        unigram_vector, vector_baseline,
        "unigram baseline drifted: {unigram_ranks:?}"
    );
    assert_eq!(
        bigram_vector, phrase_sensitive,
        "1+2 vector receipt drifted: {bigram_ranks:?}"
    );

    let (equal_unigram, equal_unigram_ranks) = evaluate_fusion(&trail, &unigram, (1.0, 1.0));
    assert_eq!(
        equal_unigram,
        FusionTally {
            deterministic: vector_baseline,
            unique_at_1: 7,
            phrase_unique_at_1: 2,
            controls_unique_at_1: 5,
            expected_top_ties: 0,
        },
        "equal-weight BM25 + unigram drifted: {equal_unigram_ranks:?}"
    );

    let (equal_bigram, equal_bigram_ranks) = evaluate_fusion(&trail, &bigram, (1.0, 1.0));
    assert_eq!(
        equal_bigram,
        FusionTally {
            // The deterministic URL tie-break happens to put every expected
            // target first. Eight phrase cases have the same RRF score as the
            // decoy, so only seven results are unique winners.
            deterministic: phrase_sensitive,
            unique_at_1: 7,
            phrase_unique_at_1: 2,
            controls_unique_at_1: 5,
            expected_top_ties: 8,
        },
        "equal-weight BM25 + 1+2 drifted: {equal_bigram_ranks:?}"
    );

    let (vector_heavy_bigram, vector_heavy_ranks) = evaluate_fusion(&trail, &bigram, (1.0, 2.0));
    assert_eq!(
        vector_heavy_bigram,
        FusionTally {
            deterministic: phrase_sensitive,
            unique_at_1: 15,
            phrase_unique_at_1: 10,
            controls_unique_at_1: 5,
            expected_top_ties: 0,
        },
        "vector-heavy BM25 + 1+2 probe drifted: {vector_heavy_ranks:?}"
    );
}

#[cfg(feature = "bert")]
const MINILM_ARTIFACTS: &[(&str, &str)] = &[
    (
        "config.json",
        "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    ),
    (
        "tokenizer.json",
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    ),
    (
        "model.safetensors",
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    ),
];

#[cfg(feature = "bert")]
fn minilm_dir() -> PathBuf {
    std::env::var_os("SIBYLLA_MINILM_DIR")
        .map(PathBuf::from)
        .expect("SIBYLLA_MINILM_DIR must point at the fixed all-MiniLM-L6-v2 artifact")
}

#[cfg(feature = "bert")]
fn sha256(path: &Path) -> String {
    let mut file = std::fs::File::open(path).unwrap_or_else(|error| {
        panic!("open model artifact {}: {error}", path.display());
    });
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).unwrap_or_else(|error| {
            panic!("hash model artifact {}: {error}", path.display());
        });
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    hex
}

#[cfg(feature = "bert")]
fn verify_minilm_artifacts(model_dir: &Path) -> u64 {
    let mut total_bytes = 0;
    for (name, expected_hash) in MINILM_ARTIFACTS {
        let path = model_dir.join(name);
        let metadata = std::fs::metadata(&path).unwrap_or_else(|error| {
            panic!("stat model artifact {}: {error}", path.display());
        });
        let actual_hash = sha256(&path);
        assert_eq!(
            actual_hash,
            *expected_hash,
            "model artifact digest drifted: {}",
            path.display()
        );
        total_bytes += metadata.len();
    }
    total_bytes
}

#[cfg(feature = "bert")]
fn build_minilm_search(model_dir: &Path) -> (DenseSearch, u128, u128) {
    let load_started = Instant::now();
    let provider = esp::embed::bert::load_cpu(model_dir).unwrap_or_else(|error| {
        panic!("load MiniLM from {}: {error}", model_dir.display());
    });
    let load_ms = load_started.elapsed().as_millis();
    assert_eq!(provider.dimensions(), 384);
    let mut search = SemanticSearch::new(provider);
    let ingest_started = Instant::now();
    search.ingest_batch(DOCUMENTS).unwrap();
    let ingest_ms = ingest_started.elapsed().as_millis();
    (search, load_ms, ingest_ms)
}

/// Explicit learned-vector baseline for the lexical forcing fixture.
///
/// This stays ignored because the checkpoint is a 90 MB local artifact rather
/// than a repository or CI fixture. Run on the portable CPU backend:
///
/// `SIBYLLA_MINILM_DIR=/path/to/all-MiniLM-L6-v2 cargo test -p esp --features bert --test lexical_ngram_recall -- learned_minilm_baseline --ignored --nocapture --test-threads=1`
#[cfg(feature = "bert")]
#[test]
#[ignore = "requires the digest-pinned all-MiniLM-L6-v2 artifact"]
fn learned_minilm_baseline() {
    let model_dir = minilm_dir();
    let artifact_bytes = verify_minilm_artifacts(&model_dir);

    let (dense, load_ms, ingest_ms) = build_minilm_search(&model_dir);

    let directory = tempfile::tempdir().unwrap();
    let trace = browsing_trace();
    let trail = TrailIndex::rebuild(directory.path().join("trail"), [&trace]).unwrap();

    let query_started = Instant::now();
    let (dense_tally, dense_ranks) = evaluate_ranker(|case| vector_ranking(&dense, case.query));
    let dense_query_ms = query_started.elapsed().as_millis();
    let (equal_fusion, equal_fusion_ranks) = evaluate_fusion(&trail, &dense, (1.0, 1.0));
    let (vector_heavy_fusion, vector_heavy_fusion_ranks) =
        evaluate_fusion(&trail, &dense, (1.0, 2.0));

    let vector_index_bytes = dense.len() * dense.provider().dimensions() * size_of::<f32>();
    println!(
        "MiniLM receipt: backend=cpu, dimensions={}, documents={}, held_out_cases={}, artifact_bytes={artifact_bytes}, vector_index_bytes={vector_index_bytes}, load_ms={load_ms}, ingest_ms={ingest_ms}, dense_query_sweep_ms={dense_query_ms}",
        dense.provider().dimensions(),
        DOCUMENTS.len(),
        EVAL_CASES.len(),
    );
    println!("dense={dense_tally:?} ranks={dense_ranks:?}");
    println!("bm25+dense (1,1)={equal_fusion:?} ranks={equal_fusion_ranks:?}");
    println!("bm25+dense (1,2)={vector_heavy_fusion:?} ranks={vector_heavy_fusion_ranks:?}");

    // The fixed artifact makes these rankings a reproducible comparison, not
    // a machine-local smoke test.
    let dense_expected = RankingTally {
        at_1: 12,
        at_3: 15,
        phrase_at_1: 7,
        controls_at_1: 5,
    };
    assert_eq!(
        dense_tally, dense_expected,
        "MiniLM ranking drifted: {dense_ranks:?}"
    );
    assert_eq!(
        equal_fusion,
        FusionTally {
            deterministic: dense_expected,
            unique_at_1: 7,
            phrase_unique_at_1: 2,
            controls_unique_at_1: 5,
            expected_top_ties: 5,
        },
        "equal-weight BM25 + MiniLM drifted: {equal_fusion_ranks:?}"
    );
    assert_eq!(
        vector_heavy_fusion,
        FusionTally {
            deterministic: dense_expected,
            unique_at_1: 12,
            phrase_unique_at_1: 7,
            controls_unique_at_1: 5,
            expected_top_ties: 0,
        },
        "dense-heavy BM25 + MiniLM drifted: {vector_heavy_fusion_ranks:?}"
    );
}

fn median(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Explicit cost harness. Run optimized and single-threaded:
///
/// `cargo test -p esp --release --test lexical_ngram_recall -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "measurement harness; run explicitly in release mode"]
fn measure_lexical_ngram_costs() {
    const SAMPLE_COUNT: usize = 11;
    const BUILD_REPEATS: usize = 50;
    const QUERY_SWEEPS: usize = 10;

    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!(
        "lexical n-gram receipt: profile={profile}, os={}, arch={}, dimensions={DIMENSIONS}, documents={}, held_out_cases={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        DOCUMENTS.len(),
        EVAL_CASES.len()
    );
    println!(
        "orders,ranking_at_1,phrase_ranking_at_1,controls_ranking_at_1,median_build_ns_per_document,median_query_ns,index_dense_vector_bytes,index_json_bytes,nonzero_slots"
    );

    let configs = [
        ("1", &[1][..]),
        ("1+2", &[1, 2][..]),
        ("1+2+3", &[1, 2, 3][..]),
    ];
    let searches: Vec<Search> = configs
        .iter()
        .map(|(_, orders)| build_search(orders))
        .collect();

    // Warm every configuration before measuring, then rotate the order in each
    // sample so CPU ramp-up cannot systematically favour the later settings.
    for (_, orders) in configs {
        for _ in 0..10 {
            black_box(build_search(black_box(orders)));
        }
    }
    for search in &searches {
        for _ in 0..2 {
            for case in EVAL_CASES {
                black_box(search.search(black_box(case.query), 5).unwrap());
            }
        }
    }

    let mut build_samples: [Vec<u128>; 3] = std::array::from_fn(|_| Vec::new());
    let mut query_samples: [Vec<u128>; 3] = std::array::from_fn(|_| Vec::new());
    for sample in 0..SAMPLE_COUNT {
        for position in 0..configs.len() {
            let index = (sample + position) % configs.len();
            let orders = configs[index].1;
            let start = Instant::now();
            for _ in 0..BUILD_REPEATS {
                black_box(build_search(black_box(orders)));
            }
            build_samples[index]
                .push(start.elapsed().as_nanos() / (BUILD_REPEATS * DOCUMENTS.len()) as u128);
        }
        for position in 0..configs.len() {
            let index = (sample * 2 + position) % configs.len();
            let search = &searches[index];
            let start = Instant::now();
            for _ in 0..QUERY_SWEEPS {
                for case in EVAL_CASES {
                    black_box(search.search(black_box(case.query), 5).unwrap());
                }
            }
            query_samples[index]
                .push(start.elapsed().as_nanos() / (QUERY_SWEEPS * EVAL_CASES.len()) as u128);
        }
    }

    for (index, (label, _)) in configs.into_iter().enumerate() {
        let search = &searches[index];
        let (tally, _) = evaluate(search);
        let dense_vector_bytes =
            search.index().len() * search.provider().dimensions() * size_of::<f32>();
        let index_json_bytes = serde_json::to_vec(search.index()).unwrap().len();
        let nonzero_slots: usize = search
            .index()
            .iter()
            .map(|(_, vector)| vector.iter().filter(|&&value| value != 0.0).count())
            .sum();
        let build_ns = median(&mut build_samples[index]);
        let query_ns = median(&mut query_samples[index]);

        println!(
            "{label},{}/{},{}/{},{}/{},{build_ns},{query_ns},{dense_vector_bytes},{index_json_bytes},{nonzero_slots}",
            tally.at_1,
            EVAL_CASES.len(),
            tally.phrase_at_1,
            EVAL_CASES.len() - 5,
            tally.controls_at_1,
            5,
        );
    }
}
