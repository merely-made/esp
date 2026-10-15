# esp

ESP is Mere's portable model-execution boundary — the inference engine's
execution lane in the ruled consolidation map (mere
`design_docs/mere_docs/implementation_strategy/2026-08-22_conatus_engine_plan.md`);
the semantic *realm* is the corpora esp executes models against, not esp
itself. It contains two deliberately separate namespaces:

- `esp::infer`: generation providers, capability matching, streaming, the
  Armillary actor, and the Burn llama-family decoder.
- `esp::embed`: embedding providers, exact vector retrieval, lexical and stub
  providers, affinity helpers, the Burn cosine kernel, and the Burn BERT model.

The default build keeps both contracts dependency-light. Model execution is
selected explicitly through features; ESP does not own model artifacts, agent
identity, job authorization, transport, or global device policy.

`LexicalEmbeddingProvider::new(dimensions)` keeps the original unigram vector
space. Phrase-sensitive callers can opt into explicit token n-gram orders with
`LexicalEmbeddingProvider::with_token_ngram_orders(dimensions, [1, 2])`.
Changing the orders changes the vector space, so callers must re-mint any
derived `VectorIndex` under the same setting. The fixed-corpus receipt and cost
numbers, including the real BM25/RRF integration probe and digest-pinned
MiniLM CPU baseline, live in the
[search surface wiring plan](../../../design_docs/mere_docs/implementation_strategy/2026-08-12_search_surface_wiring_plan.md).

## Features

| Namespace | CPU | WGPU | Other |
| --- | --- | --- | --- |
| `infer` | `decoder` | `decoder-wgpu` | `actor`, `model-session` |
| `embed` | `index-burn`, `bert` | `index-burn-wgpu`, `bert-wgpu` | `bert-validation` |

Native tokenizer builds use Oniguruma. `wasm32-unknown-unknown` uses
tokenizers' pure-Rust `unstable_wasm` path, and every Burn feature activates
the target's JavaScript entropy backend.

With `decoder,model-session`, `infer::library` resolves a model/template row
from a caller-owned Eidetic store without a network fetcher. `infer::prompt`
renders strict `esp.prompt-template/v1` data and binds the exact artifact hash
to the immutable model session. The local import example, pinned SmolLM2 row,
independent-reference reproduction and resident qualification are recorded in
the [resident inference plan](../../../design_docs/mere_docs/implementation_strategy/2026-10-15_resident_inference_and_recall_plan.md).

The `actor` feature compiles for wasm, but its Armillary thread actor requires
a host executor at runtime. See the
[feature and target matrix](design_docs/2026-08-09_feature_target_matrix.md)
for the precise compile, execution, and headed-browser boundaries.

The historical Vates and Sibylla documents are retained under `design_docs/`
with supersession notes. The `vates` and `sibylla` compatibility packages were
removed on 2026-09-23; `esp` is the only package.
