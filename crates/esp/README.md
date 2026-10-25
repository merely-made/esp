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
[search surface wiring plan](https://github.com/merely-made/mere/blob/main/design_docs/mere_docs/implementation_strategy/2026-08-12_search_surface_wiring_plan.md).

Downstream integration tests can depend on `esp-testkit` for supplier-owned
tiny decoder fixtures and templates without repinning the ESP runtime package.

## Features

| Namespace | CPU | WGPU | Other |
| --- | --- | --- | --- |
| `infer` | `decoder`, `decoder-flex` | `decoder-wgpu` | `actor`, `model-session` |
| `embed` | `index-burn`, `bert` | `index-burn-wgpu`, `bert-wgpu` | `bert-validation` |

Native tokenizer builds use Oniguruma. `wasm32-unknown-unknown` uses
tokenizers' pure-Rust `unstable_wasm` path, and every Burn feature activates
the target's JavaScript entropy backend.

With `decoder,model-session`, `infer::library` resolves a model/template row
from a caller-owned Eidetic store without a network fetcher. `infer::prompt`
renders strict `esp.prompt-template/v1` data and binds the exact artifact hash
to the immutable model session. The local import example, pinned SmolLM2 row,
independent-reference reproduction and resident qualification are recorded in
the [resident inference plan](https://github.com/merely-made/mere/blob/main/design_docs/mere_docs/implementation_strategy/2026-10-15_resident_inference_and_recall_plan.md).

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

The decoder also admits the declared Mistral, Qwen2 and Qwen3 dense blocks:
independent head widths, full or per-layer sliding attention, q/k/v biases,
Qwen3 per-head QK norms, and static linear/Llama3 RoPE scaling. Templates for
Qwen2 instruct and Qwen3 nonthinking turns are literal data. A template's exact
end-of-turn token augments config EOS IDs before token commitment. Other RoPE
variants, activations, norms and architecture families still fail by name.
`tests/reference_architectures.py` generates independent Transformers fixtures;
the native real-row checks are in `tests/architecture_rows.rs`.

Qwen2 MoE imports use a separate `esp.expert-weights/v1` index. It binds core
layers and each routed expert independently. A host calls
`ResolvedInferenceRow::artifact_source` with its existing library-owner reader;
the decoder never acquires network access. Equal routing scores prefer lower
expert IDs. Only selected experts materialize,
one at a time, with synchronization before release. `query_chunk` also bounds
MoE prefill work. Shared embeddings and heads use exact f32 row shards bounded
by `max_layer_bytes`, so a large vocabulary matrix need not be one device
buffer. `max_shared_bytes` bounds the complete encoded shared artifact.
The exact index/options and `/expert-stream-v1/f32` loader bind the session.
Expert read counters are logical artifact reads, including cache hits.
Expert formats other than f32 and MoE adapters/training remain refused pending
their own qualification. `tests/architecture_moe_real.rs` is the explicit
large-model gate; adding primitives alone does not close the plan's I9 phase.
Its pinned Qwen2-57B-A14B-Instruct row has completed a streamed-expert ask on
an 8 GB RTX 4060 Laptop GPU with sixteen exact independent reference tokens.
Its weights occupy 229,634,635,776 bytes when expanded to f32. The
[first-ask receipt](https://github.com/merely-made/mere/blob/main/support/receipts/resident_inference_architecture_first_ask.json)
records timing, selected reads and the observed process memory peak for that
first ask. The plan records further answers, checkpoint/provider-reload and
consumer checks separately; this receipt covers one physical computer's
library-backed provider row.
The [bounded I9 qualification receipt](https://github.com/merely-made/mere/blob/main/support/receipts/resident_inference_architecture_qualification.json)
adds both exact 57B answers and serialized checkpoint/provider reload, trained
Mistral 7B CPU/GPU rows with natural EOS and checkpoint parity, final small-Qwen
rows and resident consumer regressions. All use one physical computer; the
plan keeps the largest model in each family and further architectures open.
The [main integration receipt](https://github.com/merely-made/mere/blob/main/support/receipts/resident_inference_architecture_integration.json)
records the fresh release build, native CPU/GPU consumer checks, and the full
Djinn suite after the later upstream intake. Earlier model receipts bind the
unchanged inference source; the remaining I9 scale and family gates stay open.

Explicit weight formats use `esp.layer-weights/v2`. Add a seventh import
argument, `f16`, `bf16`, `q8-block32` or `q4-block32`, after the streaming
options file. The result is a candidate row. `import_formatted_model` performs
the same import for hosts. Half tensors are constructed with an explicit dtype;
Q8 and Q4 projections are quantized once in HF output-row order, with signed
integers packed into little-endian u32 words and one f32 scale per 32 input
weights within each output row.
Their embeddings and norms remain f16, including a tied output head. Projection
input widths must be divisible by 32; output widths need not be. The runtime reads these stored buffers
directly, transposes their metadata and multiplies packed weights in the GPU kernel.

`decoder-flex` enables CPU f16/bf16. The wgpu row supports f16 and the packed
formats when its device reports support. Unsupported formats return an error.
The current Flex backend rejects these packed schemes; bf16 GPU execution
requires a backend reporting that dtype and has not been qualified on this
Windows Vulkan device.

A candidate is accessible through the explicit diagnostic
`resolve_library_candidate` path. Resident selection uses
`resolve_library_session` and requires `FormatQualification`: a comparison
against the same source checkpoint's bf16 row on exact rendered prompts,
next-token logit RMSE and maximum error, generated token ids, and memory and
streaming costs. `qualify_formatted_model` publishes the measured envelope;
`qualify_inference_model LIBRARY_REDB CANDIDATE_ROW QUALIFICATION_JSON` exposes
that operation as a host command. The current GPU qualification uses Mere's
root `cubek-matmul` patch; embedding hosts must carry that patch in their own
Cargo root to reproduce the Q4 workgroup-memory row. Qualification is specific to its measured
loader. `/layer-stream-v2/<format>` and the format field enter session identity;
a format change refuses an existing checkpoint. Payload byte counts exclude
allocator overhead, temporary compute buffers, activations and KV caches.
`tests/weight_formats_real.rs` and `support/run_resident_weight_formats.ps1`
record real-model comparisons and separate host memory measurements.

Providers expose cooperative cancellation separately from text fragments, so
a host can cancel a device wait before generation starts. The LoRA trainer
arms also expose controlled entry points that check cancellation between
bounded tensor operations; the original entry points retain their behavior.

The decoder's `stage` module loads a contiguous range without retaining the
full-model source. The final range also owns its norm/output head. Each stage keeps only its own KV
and requires matching absolute positions, resetting on a complete prefill.
`assign_ranges` uses host-provided resident payload budgets after shared and
backend reservations; layer costs include KV at the declared context limit.
The coordinator retains its embedding, and the final range returns a last-position
f32 logits row for the coordinator to sample.
`ResolvedInferenceRow::load_with_stages` checks the chain's exact session and
preserves the local streamed loader identity, so placement can change without
silently changing model, format or backend class. Distillery owns the iroh
protocol, lease admission, private-data lock and recovery. ESP does not acquire
remote weights or grant stage work. Checkpoints retain placement changes as
optional provenance alongside the existing immutable session and sampling state.

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
[feature and target matrix](https://github.com/merely-made/esp/blob/main/crates/esp/design_docs/2026-08-09_feature_target_matrix.md)
for the precise compile, execution, and headed-browser boundaries.

The historical Vates and Sibylla documents are retained under `design_docs/`
with supersession notes. The `vates` and `sibylla` compatibility packages were
removed on 2026-09-23; `esp` is the only package.

Indexed decoder rows can retain all verified layers through
`load_resident_with_layer_source` when a host admits the full runtime weight
and declared context-KV payload. `resident_payload_bytes` excludes backend,
RoPE, activation, artifact-transfer and allocator reserves. Resident placement
retains the streamed row's format, session and query-chunk prefill arithmetic;
placement changes are provenance rather than a new model session.


## Guarded embedding projections

With `persistence`, `embed::compatible::CompatibleSearch` binds a dense or sparse
index to Eidetic's exact embedding envelope. Providers report the algorithm,
loaded artifact digests, tokenizer, preprocessing, pooling, role prefixes,
normalization, shape and metric. The host supplies the versioned chunker and
exact retained source revisions. Reopen validates the entire envelope and
numerical representation before search; unlabeled legacy indexes require a
rebuild. Generic `VectorIndex` and `SemanticSearch` remain available independently.

Loaded BERT buffers and the deterministic lexical algorithm supply identities;
unknown/manual component assemblies refuse guarded persistence. The core
contract and resident acceptance are linked from
`design_docs/2026-10-20_embedding_compatibility_plan.md`.
