# sibylla

A local-embedding and semantic-retrieval seam: one `EmbeddingProvider` trait
that turns text into fixed-dimension vectors, a `SimilarityMetric` each provider
declares for its output space, and a pure-Rust retrieval core over those
vectors. The default build depends only on `serde` and is wasm-clean; burn-backed
compute rides features.

```rust
use sibylla::{LexicalEmbeddingProvider, SemanticSearch};

let mut search = SemanticSearch::<u32, _>::new(LexicalEmbeddingProvider::new(512)?);
search.ingest(1, "rust async runtime internals")?;
search.ingest(2, "tokio scheduler and futures")?;
search.ingest(3, "italian pasta recipe")?;

// Top-k node keys with scores, ranked by the provider's metric.
let hits: Vec<(u32, f32)> = search.search("async rust programming", 2)?;
```

## Modules

| Module | Contents |
| --- | --- |
| `provider` | `EmbeddingProvider` (`dimensions`, `metric`, `embed`, `embed_one`), `SimilarityMetric` (`Cosine`, `Euclidean`, `DotProduct`, plus `higher_is_better`), `EmbedError`. |
| `index` | `VectorIndex<K>`, a flat dense index, `O(N)` per query: `insert`, `remove`, `get`, `contains`, `iter`, `nearest`, `clear`. Serde-serializable. `IndexError`. |
| `search` | `SemanticSearch<K, P>`: `new`, `with_index`, `ingest`, `ingest_batch`, `forget`, `search`, `provider`, `index`, `index_mut`. `SearchError`. |
| `lexical` | `LexicalEmbeddingProvider::new(dimensions)`, feature hashing. Texts sharing vocabulary get correlated vectors without loading a model. |
| `stub` | `StubEmbeddingProvider::new(dimensions)`, a deterministic test double. Equal text gives equal vectors; unrelated text gives uncorrelated vectors, so it is for exercising the pipeline rather than for recall. |
| `affinity` | `affinity_pairs(index, top_k, min_similarity) -> Vec<(K, K, f32)>`, each entry's nearest neighbours as clustering pairs, symmetric duplicates emitted once. |
| `index_burn` (`index-burn`) | `cosine_top_k`, `nearest_over_index`, `affinity_pairs_over_index`, and the `SEARCH_GPU_MIN_ENTRIES` / `AFFINITY_GPU_MIN_ENTRIES` thresholds. Batched cosine as one burn matmul, top-k on CPU over the readback. |
| `bert` (`bert`) | `BertEmbeddingProvider<B>` with `load(model_dir, device)` over the HuggingFace triple (`config.json`, `tokenizer.json`, `model.safetensors`); `BertConfig` and the `MINILM_L6_V2`, `BGE_MICRO_V2`, `SNOWFLAKE_ARCTIC_EMBED_XS` presets; `BertModel`, `BertEncoder`, `BertLayer`, `BertAttention`, `BertEmbeddings`, `Pooling`, `BertTokenizer`, `ModelArtifacts`, `LoaderError`. An unloaded provider returns `EmbedError::ModelNotLoaded`. |

## Features

| Feature | Effect |
| --- | --- |
| default | The seam plus the retrieval core and the two burn-free embedders. |
| `index-burn` | The batched-cosine kernel on burn's ndarray backend. |
| `index-burn-wgpu` | The same kernel on burn's wgpu backend. |
| `bert` | The BERT provider. Pulls `burn`, `tokenizers`, `safetensors`, `serde_json`. |
| `bert-wgpu` | GPU execution for the BERT provider. |
| `bert-validation` | Slot for continuous validation against a reference engine. The test currently panics with a wire-up message. |

## Validation

`bert::validation::FIXTURES` holds three reference texts with the first eight
floats of their L2-normalized MiniLM embeddings, captured from
`sentence-transformers/all-MiniLM-L6-v2` and compared within `TOLERANCE`
(`1e-4`). The test is `#[ignore]`d because it needs the weights:

```bash
export SIBYLLA_MINILM_DIR=/path/to/all-MiniLM-L6-v2
cargo test -p sibylla --features bert -- --ignored bert::validation
```

## Next

- Founding split and the kernel plan: `design_docs/`.
- Sibling crate `vates` (generation), at `crates/intel/vates`.
- Mere's persistence and canvas glue over this crate: `crates/intel/embed`.

License: dual MIT OR Apache-2.0, at your option.
