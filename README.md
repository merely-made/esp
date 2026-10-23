# Esp

Portable model execution, inference and embeddings with optional actor and persistence adapters.

This repository preserves the implementation history promoted from Mere on 2026-10-19. Crates retain their names, APIs, licenses and selectable features.

Canonical migration: `mere/design_docs/mere_docs/implementation_strategy/2026-10-19_family_promotion_plan.md`.

| Crate | Source |
|---|---|
| `esp` | [crates/esp](crates/esp) |

Run `cargo test --workspace` for default library coverage. Optional network, GPU, external model and host acceptance have separate gates. No registry release is implied by this source promotion.

Constrained decoder requests can carry `OutputGrammar`: literals, bounded JSON
strings, sequences and choices. The current implementation requires a ByteLevel
decoder with round-tripping token bytes. It masks invalid tokens at model decision
points and canonically tokenizes forced spans, appending and checkpointing every
token before evaluating the span as a causal block. Forced syntax is part of the
decoding policy, not a model decision or repaired output. For seeded sampling it
consumes recorded sampling positions, including forced tokens, so replay agrees.

Acceptance ends a constrained document. EOS, stop strings, token exhaustion and
unsupported tokenizers cannot turn an incomplete grammar into successful output.
Template text stops are omitted for constrained requests; explicit caller stops
are refused. The async decoder and stub provider refuse constrained requests.
The grammar travels with durable requests and is replayed over recorded token
IDs; private literal buffers are cleared with the request. Grammar validity alone
does not establish factual correctness, relevance, or agreement between sources.
