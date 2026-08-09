# sibylla Founding Proposal

> **Superseded 2026-08-09.** The implementation now lives under `esp::embed` in
> `crates/intel/esp`. The `sibylla` package is a deprecated compatibility shim.
> This document is retained as the historical boundary argument.

**Date:** 2026-07-07
**Status:** founding proposal + **P1 landed (2026-07-08)**. Scaffolds sibylla as a
standalone crate by promoting mere's `intel/embed` seam, and plans the porting of
its retrieval and model-backed pieces. P0 (the seam + the deterministic stub) and
**P1 (the portable retrieval core: `index`, `search`, `lexical`, `affinity`)** are
ported and green; the burn-wgpu BERT backend (P2) is the remaining roadmap. See
the Progress section.

## 1. What sibylla is

sibylla is a local-embedding and semantic-retrieval seam: a small,
backend-agnostic contract for turning text into vectors, plus the pure-Rust
retrieval machinery that runs over those vectors and a set of pluggable
embedders behind the trait.

- **The trait.** `EmbeddingProvider` turns a batch of texts into fixed-dimension
  vectors, declares its output `dimensions`, and names the `SimilarityMetric`
  (Cosine, Euclidean, DotProduct) its output space wants. Implementations are
  `Send + Sync` so one loaded model is shared across threads. `embed_one` is the
  single-text convenience over the batch call.
- **The floor.** `StubEmbeddingProvider` is a deterministic test double
  (FNV-1a hash to seed, xorshift64 to a vector, L2-normalized). Same text yields
  the same vector, so the embed-to-index-to-search pipeline tests with no GPU and
  no model, but different texts yield unrelated vectors, so it is explicitly not
  a similarity signal. It is the default build: `serde` only, wasm-clean.
- **The retrieval half (roadmap).** Unlike vates, embed is not just a provider
  seam: it carries a pure-Rust flat vector index (`VectorIndex`), a
  `SemanticSearch` facade (ingest ids and text, query top-k by cosine), a
  burn-free lexical embedder that is a real similarity signal, and an affinity
  helper. These are portable and are the next port after the seam.

The posture, inherited from `intel/embed`: model-backed embedders land behind
features, never bound as universal. The burn-wgpu BERT provider (MiniLM,
BGE-micro, arctic-xs) is the strategic in-process target; the lexical embedder
is the burn-free real signal that ships without a model.

## 2. Why a standalone crate

`intel/embed` already has the right shape (a clean trait, a stub, a pure-Rust
index and search facade, feature-gated model backends), but it lives inside mere
and is `publish = false`. Two consumers now want it independently:

- **mere** itself (its statistical-intelligence tier), the current owner.
- **Isometry** (a standalone Merely-stack app), for the RAG and
  semantic-recall half of the optional DM-loaded-model lane: retrieving relevant
  world and entity context to ground an NPC line or a recap. See isometry
  `design_docs/2026-07-07_optional_intelligence_vision.md`; the generation half
  is vates, the retrieval half is this.

Isometry consumes serval and netrender, not mere; it must not depend on mere to
get retrieval. So the seam belongs in a standalone crate that both consume, the
same one-way pattern as wgpu-graft/weld/scry and the reason those are their own
repos rather than mere modules. sibylla is that crate. It is the retrieval
sibling of vates: where vates voices and foretells (generation), sibylla is the
consulted corpus (embedding and retrieval). The name is the Sibyl of classical
prophecy, the consulted oracle, feminine sibling to the masculine *vates*.

## 3. Origin and what is ported

Promoted from `mere/crates/intel/embed`. Ported in this founding commit, MPL
headers intact, mere-internal doc references genericized:

- `provider.rs`: the `EmbeddingProvider` trait, `SimilarityMetric`, `EmbedError`.
  Portable already (`serde` + `std`), tests included.
- `stub.rs`: `StubEmbeddingProvider`, tests included. Its doc links to the
  not-yet-ported lexical provider were reworded to plain prose.

The mere embed crate has ten modules. They fall in two groups.

**Portable core** (belongs in sibylla; ported or next to port):

- `provider`, `stub` — ported (P0).
- `lexical` — `LexicalEmbeddingProvider`, a burn-free real similarity signal
  (shared vocabulary yields nearby vectors). Portable. Next (P1).
- `index` — `VectorIndex` / `IndexError`, a pure-Rust flat vector index.
  Portable. Next (P1).
- `search` — `SemanticSearch`, the ingest-and-query facade over a provider plus
  an index. Portable. Next (P1).
- `affinity` — `affinity_pairs`, a small similarity-pair helper. Portable.
- `bert` — the burn-wgpu BERT provider, feature-gated (`bert`, `bert-wgpu`).
  Heavy deps (burn, tokenizers, safetensors, half). Later (P2).

**Mere glue** (stays mere-side or re-homes; carries a mere coupling):

- `persistence` — `load_from_eidetic` / `save_to_eidetic`. Rides `eidetic`,
  mere's typed store. A mere coupling (section 5).
- `field_bridge` — `build_query_similarity_field` /
  `register_query_similarity_field`. Rides `aether`, mere's field algebra. A
  mere coupling that projects query similarity into the graph-canvas field
  model; app-specific, not part of a portable retrieval crate.
- `canvas_search` — `CanvasSearchSurface`. Graph-canvas-specific glue.

## 4. Porting roadmap

Done-conditions, not time estimates.

- **P0 (this commit): the seam.** `provider` + `stub` ported; the crate compiles
  and its tests pass with `serde` only. **Done when** `cargo test` is green
  (it is, 13 tests).
- **P1: the portable retrieval core.** Port `lexical`, `index`, `search`, and
  `affinity`. All burn-free and portable; this is what makes sibylla a retrieval
  crate rather than a bare trait. **Done when** a caller ingests text through
  `SemanticSearch` over the lexical embedder and gets sensible top-k results with
  `serde` only, no model. This is the near-term target and the fastest path to a
  useful crate.
- **P2: the burn-wgpu BERT backend.** Port `bert` behind `bert` and `bert-wgpu`
  features (burn, tokenizers, safetensors, half). **Done when** a real small
  model (MiniLM first) loads and embeds through the trait on wgpu, and
  `SemanticSearch` over it beats the lexical baseline on a semantic query.
  Mirrors vates's decoder/decoder-wgpu split exactly.
- **P3: mere reconciliation.** mere switches its `intel/embed` consumers to
  sibylla for the portable core, and keeps `persistence` / `field_bridge` /
  `canvas_search` mere-side as a thin glue layer over sibylla (or re-homes them
  against portable stores per section 5). **Done when** mere builds against
  sibylla and its canvas-search and field-bridge features work over the
  promoted core.
- **P4: Isometry adoption.** Isometry adds sibylla behind its own retrieval seam
  once the schema/Lua-ABI keystone and the generators lane land (sibylla is a
  post-keystone horizon there, not a blocker, exactly like vates). **Done when**
  Isometry's recap or dialog spike retrieves real world context through sibylla
  to ground a vates-generated line.

P1 and P2 are independent and can land in either order; P1 is the faster path to
a usable retrieval crate, P2 is the strategic semantic target. P1 first is
recommended (it needs no model and no heavy deps).

## 5. The mere-glue decision

Three modules carry a mere coupling: `persistence` (eidetic), `field_bridge`
(aether), and `canvas_search` (graph-canvas). Unlike vates's single `armillary`
coupling, these are genuinely app-facing, not infrastructure sibylla should
own. Three options, per module or as a group:

1. **Keep them mere-side (recommended).** sibylla ships the portable core
   (provider, stub, lexical, index, search, affinity, bert); mere keeps
   `persistence` / `field_bridge` / `canvas_search` as a thin glue layer that
   depends on sibylla and wires the portable pieces into eidetic, aether, and
   the canvas. Cost: none to sibylla; mere owns its own integration, which is
   correct. Benefit: sibylla stays dependency-clean and app-neutral.
2. **Re-home against portable abstractions.** If a portable store or field trait
   emerges, `persistence` and `field_bridge` could be ported against it. Only
   worth it if a second consumer wants persistence or field projection; today
   neither does.
3. **Promote the couplings too.** Extract eidetic or aether as standalone
   crates so sibylla can depend on them. Out of scope for this founding; those
   are mere's own extraction decisions (bundle-when-lockstep says do this only
   if the coupling is genuinely shared).

**Recommendation: option 1.** These three modules are mere's integration of a
generic retrieval crate into its specific store, field algebra, and canvas.
That integration belongs to the app, not the library. sibylla owns embedding and
retrieval; mere owns wiring them into eidetic/aether/canvas.

## 6. Backend selection

One trait, many embedders, chosen by capability and deps:

| Embedder | Feature | Deps | Role |
| --- | --- | --- | --- |
| `stub` | (default) | serde | The test/dev floor; no GPU, no model, not a real signal |
| `lexical` | (default, P1) | serde | Burn-free real similarity signal (shared vocabulary) |
| `bert` (BERT) | `bert`, `bert-wgpu` | burn, tokenizers, safetensors, half | The own in-process semantic embedder (strategic target) |

The default build is the seam plus the deterministic stub, wasm-clean; the
lexical signal joins the default build at P1 (it is burn-free). BERT and its
wgpu execution land behind features, mirroring vates.

## 7. Consumers, scope, licensing

- **Consumers and direction.** mere and Isometry consume sibylla; the flow is
  one-way (apps depend on sibylla, sibylla depends on neither). This mirrors the
  wgpu-sibling libs and vates. sibylla itself may reach ML weights and runtimes
  but names no app.
- **Scope: embedding and retrieval only.** sibylla is the embed-index-search
  half. Generation (the LLM decoder, the streaming actor, NPC voice) is vates,
  its sibling crate. They share a shape (provider trait + capability + stub) but
  not a purpose, and stay separate crates: a consumer that wants only retrieval
  (semantic recall, RAG grounding) pulls sibylla without a decoder, and a
  consumer that wants only generation pulls vates without an index. A RAG
  consumer that wants both depends on both, which is the honest dependency.
- **Licensing.** The ported files are MPL-2.0 (mere's license), so sibylla is
  MPL-2.0 for now. The Merely app workspaces (isometry, serval) use
  `MIT OR Apache-2.0`; whether sibylla relicenses to match the crates.io norm is
  Mark's call before first publish, and should be decided together with vates so
  the sibling crates match. Until then MPL is the safe default for promoted code.

## 8. Open questions

1. **mere glue** (section 5): keep `persistence` / `field_bridge` /
   `canvas_search` mere-side (recommended), re-home against portable
   abstractions, or promote the eidetic/aether couplings.
2. **License** (section 7): MPL-2.0 or relicense to MIT/Apache before publish;
   decide alongside vates.
3. **Publish vs git-dep:** publish to crates.io (like wgpu-scry) or consume as a
   git dep first. `publish` stays off until this and the license settle. Decide
   alongside vates.
4. **Lexical in the default build:** ship `lexical` in the default feature set
   (it is burn-free and a real signal, so it improves the no-model experience)
   or gate it behind a `lexical` feature for a minimal default. Recommended:
   default, since it costs only serde.
5. **Repo home for the BERT port:** port P2 into this repo directly, or develop
   it in mere's `intel/embed` and move the finished body over. Same question
   vates faces for its decoder; answer both the same way. The former keeps
   sibylla the source of truth.

## Progress

- **2026-07-08 — P2 BERT backend landed** (sibylla `c06932b`, mere reconciled
  `fc5b8db`). The full Burn-backed BERT embedder moved from mere's intel/embed into
  sibylla behind `bert` / `bert-wgpu` — 16 modules (config, embeddings, attention,
  feed-forward, layer, encoder, model, word-piece tokenizer, safetensors loader,
  provider, validation). Ported verbatim but for stripped MPL headers (MIT/Apache),
  genericized eidetic doc refs ("a host store"), a rewritten mod doc (it is a real
  working embedder, not the stale "scaffold"), and a `bert`-gated `serde_json` for
  config parsing. sibylla `--features bert` is green (113 tests: 55 core + the bert
  unit suite on random weights; 1 ignored dir-load validation via
  `SIBYLLA_MINILM_DIR`). mere re-exports it (`pub use sibylla::bert`) and keeps its
  eidetic↔BERT integration test mere-side (that one is app-coupled). This resolves
  open-Q5 (develop-in-mere-then-move, same as vates's decoder). Remaining: P4
  (Isometry adoption).
- **2026-07-08 — P3 mere reconciliation landed** (mere `1fe6a82`). mere's
  `intel/embed` now consumes sibylla for the portable core: it re-exports sibylla's
  modules (`provider` / `stub` / `index` / `search` / `lexical` / `affinity`) at the
  same paths and deletes its six duplicated files, keeping only the glue (`bert` /
  `persistence` / `field_bridge` / `canvas_search`). The one real edge was §5's
  persistence coupling: `impl eidetic::TypedPayload for VectorIndex` became an
  orphan-rule violation once `VectorIndex` was sibylla's, resolved with a local
  `#[serde(transparent)]` newtype (byte-identical persisted format). This is mere's
  first promoted-crate git-dep reconciliation. Verified: embed default + `bert` +
  the eidetic-search consumer. The founding-P3 done-condition (mere builds against
  sibylla, glue works over the promoted core) is met. P4 (Isometry) and sibylla's
  own P2 (BERT — mere still holds `bert` until then) remain.
- **2026-07-08 — index burn-lift (P1 kernel).** Added `index_burn::cosine_top_k`
  behind `index-burn` / `index-burn-wgpu`: batched cosine as one matmul on burn
  (`queries · corpusᵀ`), CPU top-k over the readback — the same tensor-program
  shape as a tensorized N-body force pass, lifting the flat index's `O(N·d)`
  per-query scan onto the GPU. Verified against `VectorIndex::nearest` and
  ndarray↔wgpu parity green on the real GPU (54 tests). Exact, not HNSW. Its own
  doc + roadmap (P2 crossover, P3 route search/affinity) is
  `2026-07-08_index_burn_lift_plan.md`. This is orthogonal to the founding roadmap
  below (it accelerates the ported `index`, independent of the BERT P2).
- **2026-07-08 — P1 landed: the portable retrieval core.** Ported `index`
  (`VectorIndex` / `IndexError`), `search` (`SemanticSearch` / `SearchError`),
  `lexical` (`LexicalEmbeddingProvider`), and `affinity` (`affinity_pairs`) from
  mere's `intel/embed`, verbatim but for genericized doc references (the eidetic
  persistence pointer in `index`, and `affinity`'s gyre/canvas/force-directed
  framing reworded to a layout-neutral description). All four are serde-only, so
  they ship in the **default build** rather than behind features — the Cargo
  comment's earlier "index/search/lexical behind features" note is superseded by
  §6 / open-Q4 (the burn-free core is default; only BERT rides features), which is
  what makes the P1 done-condition — a working `SemanticSearch` over the lexical
  embedder in the base build — actually hold. `cargo test` green, **49 tests** (up
  from 13). The done-condition is met: a caller ingests text through
  `SemanticSearch` over `LexicalEmbeddingProvider` and gets sensible top-k with
  serde only, no model (see the README example). Next: P2 (BERT backend) or P3
  (mere reconciliation); P1 and P2 are independent.

## Provenance

Grounded in a read of `mere/crates/intel/embed` (lib, provider, stub, Cargo.toml,
module map) 2026-07-07. The sibling crate vates was founded the same day
(`repos/vates/design_docs/2026-07-07_vates_founding_proposal.md`); its section 7
scope note names embed as this separate sibling promotion. The name and the
consumer-side vision are recorded in the isometry
`design_docs/2026-07-07_optional_intelligence_vision.md` and the workspace
memory.
