# esp

ESP is Mere's portable model-execution boundary. It contains two deliberately
separate namespaces:

- `esp::infer`: generation providers, capability matching, streaming, the
  Armillary actor, and the Burn llama-family decoder.
- `esp::embed`: embedding providers, exact vector retrieval, lexical and stub
  providers, affinity helpers, the Burn cosine kernel, and the Burn BERT model.

The default build keeps both contracts dependency-light. Model execution is
selected explicitly through features; ESP does not own model artifacts, agent
identity, job authorization, transport, or global device policy.

## Features

| Namespace | CPU | WGPU | Other |
| --- | --- | --- | --- |
| `infer` | `decoder` | `decoder-wgpu` | `actor` |
| `embed` | `index-burn`, `bert` | `index-burn-wgpu`, `bert-wgpu` | `bert-validation` |

Native tokenizer builds use Oniguruma. `wasm32-unknown-unknown` uses
tokenizers' pure-Rust `unstable_wasm` path, and every Burn feature activates
the target's JavaScript entropy backend.

The `actor` feature compiles for wasm, but its Armillary thread actor requires
a host executor at runtime. See the
[feature and target matrix](design_docs/2026-08-09_feature_target_matrix.md)
for the precise compile, execution, and headed-browser boundaries.

The historical Vates and Sibylla documents are retained under `design_docs/`
with supersession notes. The `vates` and `sibylla` packages are compatibility
shims; new code should depend on `esp` directly.
