# Index burn-lift: batched cosine as a matmul

> **Historical home.** The landed implementation moved unchanged to
> `esp::embed::index_burn` on 2026-08-09.

**Date:** 2026-07-08
**Status:** P1 (kernel + parity) + P2 (crossover measured, see Findings) + P3
(keyed GPU accelerators over `VectorIndex` + routing thresholds) landed. The
burn-lift is complete for the flat index; HNSW is the separate algorithmic path.
**Related:** the founding proposal (`2026-07-07_sibylla_founding_proposal.md`) for
the crate shape; re-homes the scoping first written mere-side as
`intel_vector_index_burn_lift_plan` now that the `VectorIndex` lives here.

## The insight

`VectorIndex::nearest` scores a query against every entry on the CPU, `O(N·d)` per
query — `O(Q·N·d)` for a batch of `Q`. That is a **matmul**: normalize both sides
to unit rows and the cosine similarity matrix is `queries · corpusᵀ`, `[Q,N]`.

Two reasons this earns a kernel rather than a footnote:

1. **It is the same tensor-program shape as a tensorized N-body force pass** — a
   dense pairwise interaction over `[Q,d]` and `[N,d]`, reduced. That program is
   measured to beat naive CPU on burn-wgpu from ~1k entries up (the mere Lane-5
   force-pass timings). All-pairs cosine is that program with a dot-product
   reduction instead of inverse-square.
2. **The flat index is a shared ceiling.** The same `VectorIndex` backs
   `SemanticSearch` (recall) and `affinity_pairs` (clustering). One kernel lifts
   both, plus any consumer downstream in mere / Isometry.

## The kernel (P1, this commit)

`index_burn::cosine_top_k<B: Backend>(queries, corpus, k, device) -> Vec<Vec<(usize, f32)>>`:

- Normalize rows of `queries` and `corpus` to unit L2 on `B`.
- `sim = queries · corpusᵀ` — one matmul, `[Q,N]`, the heavy `O(Q·N·d)` work.
- Read back `[Q,N]` and take each query's top-k on the CPU (cheap `O(N)` per row).

Deliberately uses only proven burn idioms (`matmul`, `swap_dims`, element-wise
square, `sum_dim`, `sqrt`, `recip`, f32 readback) — **no `topk`, no Int-tensor
readback** — so it is identical across the ndarray and wgpu backends without the
per-backend int-element hazard. Gated `index-burn` (ndarray) / `index-burn-wgpu`
(adds burn's wgpu backend); the default build stays serde-only and wasm-clean.

**Tested:** correctness against `VectorIndex::nearest` (the CPU path this
accelerates — same ranked indices, scores within 1e-5), all-pairs ranks self
first, zero-vector query scores zero, edge cases (empty / k=0 / k>N), and an
ndarray↔wgpu parity test gated on `index-burn-wgpu`.

## Roadmap

- **P2 — crossover measurement.** Time CPU-flat vs GPU-batched across `N` and `Q`
  (readback included, GPU warmed), for both all-pairs (`Q=N`, arrangement) and
  few-query (recall) shapes — different crossovers, both matter. Record where the
  matmul path overtakes the flat scan.
- **P3 — accelerators over the index (landed 2026-07-08).** The extraction
  discipline shaped this: `search.rs` and `affinity.rs` stay **pure and
  serde-only** (sibylla is the source of truth the mere copies reconcile onto), so
  the burn fast-paths live entirely in the feature-gated `index_burn` module as
  *keyed* entry points over the portable `VectorIndex`:
  - `nearest_over_index<K, B>` — `cosine_top_k` over an index, results keyed by the
    index's own `K` (the search / recall accelerator).
  - `affinity_pairs_over_index<K, B>` — the GPU all-pairs analog of
    `affinity_pairs`, drop-in equivalent (same pairs, weights within 1e-5, tested).
  - `AFFINITY_GPU_MIN_ENTRIES` (1024) and `SEARCH_GPU_MIN_ENTRIES` (4096) — the
    measured crossovers a consumer routes on (`if index.len() >= … { gpu } else
    { cpu }`). Routing is the caller's one-line check, not hidden fallback — single
    -responsibility functions compose cleanly, and the caller can override the
    threshold. When mere / Isometry adopt sibylla they enable `index-burn(-wgpu)`
    and route large graphs to these; nothing couples the pure facade to burn.

## Findings

**2026-07-08 — P2 crossover** (release, real GPU, `d=384`, `k=10`, readback
included, GPU warmed; CPU = `VectorIndex::nearest` per query, GPU =
`cosine_top_k::<Wgpu>`):

| shape | N | Q | CPU flat | GPU batched | speedup |
| --- | --- | --- | --- | --- | --- |
| all-pairs | 256 | 256 | 45 ms | 170 ms | 0.27× (CPU wins) |
| all-pairs | 1024 | 1024 | 738 ms | 320 ms | 2.31× |
| all-pairs | 4096 | 4096 | 13,144 ms | 646 ms | **20.3×** |
| few-query | 1024 | 8 | 6.4 ms | 186 ms\* | 0.03× (CPU wins) |
| few-query | 4096 | 8 | 26 ms | 8.3 ms | 3.14× |
| few-query | 16384 | 8 | 114 ms | 25 ms | 4.52× |

- **All-pairs (arrangement / affinity): crossover ~500-1000.** Below it the GPU
  dispatch + the `[N,N]` readback dominate the small `O(N²)` and the CPU scan wins;
  above it the GPU pulls away hard — at 4k the CPU is an unusable 13 s against the
  GPU's 0.65 s. This matches the mere Lane-5 **force-pass** crossover almost
  exactly: the same `O(N²)` shape crosses at the same ~1k node count. This is the
  number that matters for routing `affinity_pairs`.
- **Few-query (recall / search): GPU pays from a few thousand corpus entries.** 8
  queries is little CPU work, so the flat scan wins at N=1024; by 4k the GPU is 3×,
  by 16k 4.5×.
- **\*The N=1024 few-query GPU time (186 ms) is a one-time artifact, not steady
  state.** cubecl compiles a kernel per matmul *shape* on first dispatch, and the
  few-query shape (`[8,d]·[d,N]`) was cold — the warmup ran the all-pairs *square*
  shape. That ~180 ms is compilation; it amortizes once the shape recurs. The
  honest warm few-query crossover is ~2-4k corpus. A per-shape warm-on-first-use
  pass is the mitigation, and a real production consideration: the first query of a
  new shape pays compilation.

**P3 routing thresholds (conservative):** GPU above ~1000 for all-pairs
(affinity), above ~4000 corpus for few-query (search); the flat CPU scan stays the
default below those and when the feature is off.

## Honest bounds

- **Exact, not approximate.** This is brute force on faster hardware, not HNSW; it
  defers the `O(N²)` asymptotics, it does not solve them. An HNSW index is the
  separate algorithmic path for the large tail (and the browser floor, where GPU
  compute is least certain).
- **The `[Q,N]` readback.** Top-k runs CPU-side over the full score matrix, so the
  readback is `O(Q·N)`; the `O(Q·N·d)` matmul is what moves to the GPU (for
  384-d embeddings the matmul dominates, so the readback is a fraction). A GPU
  top-k returning only `[Q,k]` would shrink it — a refinement, not P1.
- **Small N is fine on CPU.** This is a scaling lever; sequence P3 when a consumer
  actually pushes `N` up, not speculatively.
