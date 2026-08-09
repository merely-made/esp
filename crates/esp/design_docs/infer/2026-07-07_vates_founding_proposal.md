# vates Founding Proposal

> **Superseded 2026-08-09.** The implementation now lives under `esp::infer` in
> `crates/intel/esp`. The `vates` package is a deprecated compatibility shim.
> This document is retained as the historical boundary argument.

**Date:** 2026-07-07
**Status:** founding proposal. This repo's first doc. Scaffolds vates as a
standalone crate by promoting mere's `intel/infer` seam, and plans the porting
of its model-backed backends. The seam plus the canned stub are ported and
green in this commit; everything below the seam is roadmap.

## 1. What vates is

vates is a local-model inference seam: a small, backend-agnostic contract for
generating text from a prompt, plus a set of pluggable backends behind it.

- **The trait.** `InferenceProvider` is streaming-first. `generate_streaming`
  pushes each text fragment through a callback as it is produced and returns
  the assembled text; the callback's `ControlFlow::Break` is the cancellation
  channel. `generate` is the collect-it-all wrapper. Implementations are
  `Send + Sync` so one loaded model is shared across threads.
- **Capability match.** `ModelCapability` (identity, context window,
  quantization, loader id, streaming flag) plus `CapabilityQuery` let a caller
  pick a provider by what it needs, so a runtime is selected, never bound as
  universal.
- **The floor.** `CannedProvider` is a deterministic, dependency-light stub
  (exact-match responses, else `echo:`), streaming word by word and honouring
  `max_tokens` / `stop`. It lets the whole pipeline (seam, streaming, prompt
  plumbing, a consumer's dialog or RAG loop) test with no GPU, no model, and no
  nondeterminism. It is the default build: `serde` + `std` only, wasm-clean.

The posture, inherited verbatim from `intel/infer`: model-backed backends land
behind features, selected by capability, never bound as universal. The own
burn-wgpu decoder is the first; external endpoints (Ollama, llama.cpp) and
native runtimes (mistral.rs) are future backends behind the same trait.

## 2. Why a standalone crate

`intel/infer` already has the right shape (a clean trait, a stub, feature-gated
backends), but it lives inside mere and is `publish = false`. Two consumers now
want it independently:

- **mere** itself (its statistical-intelligence tier), the current owner.
- **Isometry** (a standalone Merely-stack app), for the optional DM-loaded
  model lane: the DM-in-the-loop dialog system, recap, RAG, generation. See
  isometry `design_docs/2026-07-07_optional_intelligence_vision.md`, section
  3.5 (the recommended inference architecture is exactly this seam: DM-only,
  external-endpoint-first, burn as the eventual swap).

Isometry consumes serval and netrender, not mere; it must not depend on mere to
get inference. So the seam belongs in a standalone crate that both consume, the
same one-way pattern as wgpu-graft/weld/scry and the reason those are their own
repos rather than mere modules. vates is that crate. The name is the poet-
prophet (Latin *vates*): it both voices (speaks as characters) and foretells
(inference), which is precisely a crate that voices NPCs and runs inference.

## 3. Origin and what is ported

Promoted from `mere/crates/intel/infer`. Ported in this founding commit, MPL
headers intact, mere-internal doc references genericized:

- `provider.rs`: the `InferenceProvider` trait, `ModelCapability` /
  `CapabilityQuery`, `GenerationRequest`, `InferError`. Portable already
  (`serde` + `std`), tests included.
- `canned.rs`: `CannedProvider`, tests included.

Not yet ported (the roadmap in section 4), because each carries either heavy
deps or a mere coupling:

- The **decoder** (`intel/infer/src/decoder/`): an own llama-family body on
  burn-nn primitives (RmsNorm, RotaryEncoding, SwiGlu, autoregressive KV-cache),
  an ndarray baseline and a wgpu backend, with safetensors + HF-tokenizers +
  half (bf16/f16 to f32) loading. All external deps; a clean port.
- The **actor** (`intel/infer/src/actor.rs`): a threaded streaming actor. It
  rides `armillary`, a mere-internal thread-actor harness. The one coupling to
  resolve (section 5).
- The **corridor test** (`tests/eidetic_corridor.rs`): rides `eidetic`, mere's
  typed store. Test-only; either dropped or re-homed against a portable store.

## 4. Porting roadmap

Done-conditions, not time estimates.

- **P0 (this commit): the seam.** `provider` + `canned` ported; the crate
  compiles and its tests pass with `serde` only. **Done when** `cargo test`
  is green (it is).
- **P1: the burn-wgpu decoder.** Port `decoder/` behind `decoder` and
  `decoder-wgpu` features (burn, safetensors, tokenizers, half). **Done when**
  a real small model (TinyLlama first, then a 3B tool-caller) loads and streams
  through the trait on wgpu, matching the ndarray baseline. Mirrors embed's
  bert/bert-wgpu split.
- **P2: the streaming actor.** Resolve the `armillary` decision (section 5),
  then port the actor. **Done when** a provider streams fragments as actor
  messages with no mere dependency (or with a deliberately kept, documented
  one).
- **P3: an external-endpoint backend.** An OpenAI-compatible `endpoint`
  provider (Ollama, llama.cpp server) behind an `endpoint` feature. This is the
  isometry vision's external-first recommendation: it meets reliable multi-turn
  tool-calling on all desktop targets today, while the burn decoder matures.
  **Done when** a caller drives a local Ollama model through the same trait and
  capability path as the canned stub.
- **P4: reconciliation and adoption.** mere switches its `intel/infer`
  consumers to vates (intel/infer becomes a thin re-export, or is deleted per
  no-legacy-friction); Isometry adds vates behind its own provider seam once
  the schema/Lua-ABI keystone and the generators lane land (vates is a
  post-keystone horizon there, not a blocker). **Done when** mere builds
  against vates and Isometry's dialog spike streams a real NPC line.

P1 and P3 are independent and can land in either order; P3 is the faster path
to a usable model for consumers, P1 is the strategic in-process target.

## 5. The armillary decision

The actor is the only mere coupling. Three options:

1. **Portable actor primitive.** Reimplement the thin thread-actor loop
   (spawn, command/update channels) inside vates with `std::thread` +
   `std::sync::mpsc`, no `armillary`. Cost: a small duplication of a simple
   pattern. Benefit: vates stays dependency-clean and its actor ships with it.
2. **Promote armillary too.** If armillary is itself a clean, reusable
   thread-actor harness, promote it to a standalone crate (its own extraction)
   and depend on it. Cost: a second extraction; only worth it if other crates
   want armillary. Bundle-when-lockstep says do this only if the coupling is
   genuinely shared.
3. **Keep the actor mere-side.** Ship vates as the seam + backends; let mere
   keep its own actor wrapping a vates provider. Cost: consumers that want a
   threaded actor (a browser-off host, Isometry's DM host) re-implement it.

**Recommendation: option 1.** A streaming inference actor is a few dozen lines
over the trait, and keeping it in vates means consumers get streaming off the
main thread for free without pulling a mere harness. Revisit option 2 only if
armillary proves broadly wanted across the ecosystem.

**Update (2026-07-07): resolved as option 2.** armillary was read (732 LOC, one
external dep `tracing`, host-neutral) and promoted to a standalone crate the same
day at `repos/armillary`. Three consumers cleared the bar: mere/meerkat, this
crate's actor (P2), and Isometry's serval host. So vates's actor depends on
armillary rather than reimplementing a primitive; consumers still get streaming
off the main thread for free, now from a shared crate. See
`repos/armillary/design_docs/2026-07-07_armillary_founding_proposal.md`.

## 6. Backend selection

One trait, many loaders, chosen by `CapabilityQuery`:

| Loader | Feature | Deps | Role |
| --- | --- | --- | --- |
| `canned` | (default) | serde | The test/dev floor; no GPU, no model |
| `burn-wgpu` | `decoder`, `decoder-wgpu` | burn, tokenizers, safetensors, half | The own in-process decoder (strategic target) |
| `endpoint` | `endpoint` | an http client | External Ollama / llama.cpp server (fast path to a real model) |
| `mistral.rs` | (future) | mistral.rs | A native in-process runtime where its GPU targets are guaranteed |

A caller asks for the capability it needs (context window, streaming, a
specific loader) and vates hands back a satisfying provider; the caller never
hardcodes a runtime.

## 7. Consumers, scope, licensing

- **Consumers and direction.** mere and Isometry consume vates; the flow is
  one-way (apps depend on vates, vates depends on neither). This mirrors the
  wgpu-sibling libs. vates itself may reach ML weights and runtimes but names
  no app.
- **Scope: inference (LLM) only.** `intel/infer` has a sibling, `intel/embed`
  (a BERT `EmbeddingProvider` for RAG/semantic search). They share a shape
  (provider trait + capability + stub) but not a purpose. vates is the
  inference/decoder crate. embed is a candidate for its own sibling promotion
  (the RAG/search half of the isometry vision's opportunity catalog); keep them
  separate crates unless a consumer proves they want one dependency.
- **Licensing.** The ported files are MPL-2.0 (mere's license), so vates is
  MPL-2.0 for now. The Merely app workspaces (isometry, serval) use
  `MIT OR Apache-2.0`; the other standalone libs' license should be confirmed.
  Whether vates relicenses to match the crates.io ecosystem norm is Mark's call
  before first publish; until then MPL is the safe default for promoted code.

## 8. Open questions

1. **armillary** (section 5): RESOLVED 2026-07-07 as promote. armillary is now a
   standalone crate (`repos/armillary`); vates's actor depends on it. (Was:
   portable primitive vs promote vs keep mere-side.)
2. **License** (section 7): MPL-2.0 or relicense to MIT/Apache before publish.
3. **embed promotion:** a separate `vates`-sibling crate for the embedder, or
   fold embeddings into vates as a second provider family.
4. **Publish vs git-dep:** publish to crates.io (like wgpu-scry) or consume as
   a git dep first, mirroring the woodshed pattern. `publish` stays off until
   this and the license settle.
5. **Repo home for the decoder port:** port P1 into this repo directly, or
   develop it in mere's `intel/infer` and move the finished body over. The
   former keeps vates the source of truth; the latter reuses mere's existing
   decoder tests until they migrate.

## Provenance

Grounded in a read of `mere/crates/intel/infer` (lib, provider, canned,
Cargo.toml) 2026-07-07. The name and the consumer-side vision are recorded in
the isometry `design_docs/2026-07-07_optional_intelligence_vision.md` and the
workspace memory `project_vates_inference_crate`.
