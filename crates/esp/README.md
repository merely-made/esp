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

Native imports can pass a sixth argument to `import_inference_model`, a JSON
file with `query_chunk`, `max_layer_bytes` and `max_shared_bytes`. For example:

```json
{"query_chunk":32,"max_layer_bytes":268435456,"max_shared_bytes":1073741824}
```

That selects `esp.layer-weights/v1`: one shared embedding/norm/head artifact
and an ordered list of exact layer artifacts. The importer reads bounded
tensor ranges from a single safetensors file or an HF sharded weight index.
`infer::layer_library::import_layered_model` exposes the same operation to
hosts. It neither acquires model bytes over the network nor changes their
precision. A resolved layered row needs `load_with_layer_source`; the source
borrows its existing library owner for each verified read. The loader's
`/layer-stream-v1` suffix and the weight index enter the exact session identity.

The streamed decoder retains one layer's weights at a time, with the shared
weights and KV cache resident. Native execution drains each query chunk and
layer before proceeding; generation applies the output head only to the final
position. `streaming_stats` counts logical artifact bytes and layer reads,
including OS-cache hits. These counters are not physical disk traffic or a
process memory measurement. The real-model fixture is
`tests/layer_streaming_real.rs`. I6 qualifies seven-shard OpenLlama 3B v2
(13.71 GB of f32 weights on an 8 GB GPU), exact reference tokens and resumed
tokens, plus a 4,126-token SmolLM2 prefill. The plan records memory, timing,
logical I/O and the measurement boundaries.

Providers expose cooperative cancellation separately from text fragments, so
a host can cancel a device wait before generation starts. The LoRA trainer
arms also expose controlled entry points that check cancellation between
bounded tensor operations; the original entry points retain their behavior.

`InferenceProvider::generate_checkpointed` adds token-level durable generation
for the decoder and session provider. It persists an actual seed before the
first sampled token, resumes splitmix64 at the committed draw, and rebuilds KV
state from the recorded prompt and generated IDs. A session provider refuses a
checkpoint from another exact session before executing it. The actor accepts
checkpointed requests and pause commands, with a synchronous host commit barrier
before each visible checkpoint. Providers without this capability refuse it.
The native SmolLM2 GPU receipt compares resumed IDs with uninterrupted decoding
and the independent greedy reference; browser durable execution is unqualified.

The `actor` feature compiles for wasm, but its Armillary thread actor requires
a host executor at runtime. See the
[feature and target matrix](design_docs/2026-08-09_feature_target_matrix.md)
for the precise compile, execution, and headed-browser boundaries.

The historical Vates and Sibylla documents are retained under `design_docs/`
with supersession notes. The `vates` and `sibylla` compatibility packages were
removed on 2026-09-23; `esp` is the only package.
