# ESP feature and target matrix

**Date:** 2026-08-09; Burn 0.22 prerelease matrix re-run 2026-08-20

**Toolchain:** the workspace's current Rust toolchain

**Targets:** native Windows and `wasm32-unknown-unknown`

This is the E0 portability receipt for the Vates and Sibylla consolidation.
It records compilation and execution separately. A target check does not claim
that a browser or actor runtime was exercised.

## Dependency floor

`cargo tree -p esp --no-default-features -e normal` contains only ESP and
Serde's derive tree. Neither Burn nor tokenizers is present.

Native tokenizer builds select Oniguruma. Browser-wasm builds select
tokenizers' `unstable_wasm` regex path. Every Burn-bearing ESP feature enables
getrandom 0.4's `wasm_js` backend because the dependency is reached by the CPU
feature trees as well as their WGPU extensions.

## Compile matrix

| Configuration | Native | `wasm32-unknown-unknown` | Meaning |
| --- | --- | --- | --- |
| empty/default | pass | pass | Serde-only public contracts |
| `actor` | pass | pass | Compiles on wasm; Armillary's thread actor is supported at runtime only where its executor is available |
| `decoder` | pass | pass | NdArray inference model and wasm tokenizer compile |
| `decoder-wgpu` | pass | pass | Same model body with Burn WGPU |
| `index-burn` | pass | pass | NdArray exact cosine kernel |
| `index-burn-wgpu` | pass | pass | Exact cosine kernel with Burn WGPU |
| `bert` | pass | pass | NdArray BERT and wasm tokenizer compile |
| `bert-wgpu` | pass | pass | BERT with Burn WGPU |
| `bert-validation` | pass | pass | Validation utilities compile with BERT |
| all CPU model features | pass | pass | `decoder,index-burn,bert,bert-validation` coexist |
| all browser WGPU features | pass | pass | `decoder-wgpu,index-burn-wgpu,bert-wgpu,bert-validation` coexist |

The wasm receipts are compile receipts. Headed browser loading, persistence,
first-token latency, throughput, memory, cancellation, frame impact, and worker
restart remain the separate D2 measurement lane named by the consolidation
plan.

The entire table was re-run after the Burn `0.22.0-pre.2` migration, one row at
a time. The WGPU rows initially exposed an upstream `cubecl-runtime` wasm
manifest defect; the narrow workspace patch recorded in the
[Burn closure receipt](../../../../design_docs/mere_docs/testing/2026-08-20_burn_0_22_prerelease_closure.md)
applies upstream's published corrections without changing CubeCL source. All
eleven ESP rows then passed. This remains a compile receipt, not a headed
browser execution claim.

The workspace ignores its generated `Cargo.lock`. Ordinary and offline wasm
checks pass. `--locked` is not used as a portability receipt because concurrent
workspace Cargo processes can regenerate the ignored target-specific lockfile.

## Execution receipts

- Merged native CPU unit suite: 171 passed, 0 failed, 1 ignored.
- Eidetic inference corridor: 1 passed.
- Real-device WGPU parity: 3 passed, covering decoder layer, index kernel, and
  BERT sentence parity on the machine's available AMD Radeon 780M and NVIDIA
  RTX 4060 Laptop adapters.
- `mere-eidetic-search`: 12 library tests and 4 example tests passed.
- `mere-embed` with `bert,bert-wgpu`: test targets compiled.
- ESP 0.1.0: packaged 54 files, verified the extracted crate, and published to
  crates.io from clean commit `1283b4a8`.
- Vates 0.1.2 and Sibylla 0.1.2: all-feature workspace checks passed; after ESP
  became available, both packages resolved the registry release, verified
  their extracted crates, and published to crates.io.

Knot's ESP consumer compiled during E1. Its later full library-test run met a
concurrent, unrelated borrow error in the new publication-client test at
`ports/knot/src/publish_host.rs`; that work is outside this consolidation and
was left untouched.

## Done boundary

E0 through E4 are complete. The registry sequence completed on 2026-08-09:
ESP 0.1.0, Vates 0.1.2, then Sibylla 0.1.2.
