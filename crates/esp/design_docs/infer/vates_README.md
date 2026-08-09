# vates

A local-model inference seam: one streaming-first `InferenceProvider` trait,
capability descriptors so a caller matches a model to a runtime, and a
deterministic `CannedProvider` that exercises the whole pipeline without a GPU
or model weights. Model-backed backends sit behind features and are selected by
capability.

```rust
use vates::{CannedProvider, GenerationRequest, InferenceProvider};

let p = CannedProvider::new().with_response("ping", "pong");
let out = p.generate(&GenerationRequest { prompt: "ping".into(), ..Default::default() })?;
assert_eq!(out, "pong");
```

## Modules

| Module | Contents |
| --- | --- |
| `provider` | `InferenceProvider` (`capability`, `generate_streaming`, `generate`), `ModelCapability` (`model_id`, `context_window`, `quantization`, `loader`, `streaming`, plus `satisfies`), `CapabilityQuery`, `GenerationRequest` (`prompt`, `max_tokens`, `temperature`, `top_p`, `seed`, `stop`), `InferError`. |
| `canned` | `CannedProvider`: `new`, `with_response`, `with_context_window`. Exact-match prompts return their canned text, anything else returns `echo: <prompt>`. Output streams word by word and honours `max_tokens` and `stop`. |
| `decoder` (`decoder`) | The own llama-family decoder on burn. `DecoderConfig` parses a HuggingFace `config.json`; `DecoderModel`, `LoadedDecoder`, `DecoderLayer`, `DecoderAttention` (GQA), `KvCache`, `LayerKvCache`, `generate_ids` / `generate_ids_with` / `TokenPicker`, `Sampler`, `load_decoder_from_bytes`, `DecoderProvider`. Under `decoder-wgpu` it also exposes `WgpuDecoderProvider` and `load_wgpu_provider`. |
| `actor` (`actor`) | `spawn_inference_actor`, `InferCommand`, `InferUpdate`. Holds a loaded provider on an armillary thread, built on that thread, and streams fragments back as typed updates. Cancellation is drained from inside the streaming callback; cancelled requests emit `InferUpdate::Cancelled`. |

Streaming is the primary call: `generate_streaming` pushes each fragment through
an `FnMut(&str) -> ControlFlow<()>` callback, and returning `Break` stops
generation. `generate` is the provided wrapper that returns only the final text.

## Features

| Feature | Effect |
| --- | --- |
| default | The seam plus `CannedProvider`. Serde only, thread-free, compiles for `wasm32-unknown-unknown`. |
| `decoder` | The burn decoder. Pulls `burn` (ndarray), `serde_json`, `safetensors`, `half`, `tokenizers`, `tracing`. |
| `decoder-wgpu` | The decoder on burn's wgpu backend. |
| `actor` | The threaded streaming actor. Pulls `armillary`. |

An external OpenAI-compatible endpoint backend (Ollama, llama.cpp) is planned in
`design_docs/`; there is no `endpoint` feature yet.

## Tests

`tests/tinyllama_real.rs` validates the decoder against a real checkpoint. All
of it is `#[ignore]`d and needs `VATES_TINYLLAMA_DIR` pointing at a
TinyLlama-1.1B-Chat-v1.0 directory in HuggingFace layout:

```bash
VATES_TINYLLAMA_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  cargo test -p vates --features decoder-wgpu --release \
  --test tinyllama_real -- --ignored --nocapture --test-threads=1
```

## Next

- Founding proposal and backend roadmap: `design_docs/`.
- Sibling crate `sibylla` (embedding and retrieval), at `crates/intel/sibylla`.
- Actor harness: `crates/armillary`.

License: dual MIT OR Apache-2.0, at your option.
