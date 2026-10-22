# Esp documentation

Preserve host-owned authority, existing formats and independent crate feature boundaries. Read [documentation policy](DOC_POLICY.md) before editing.

Canonical migration and current receipts: `mere/design_docs/mere_docs/implementation_strategy/2026-10-19_family_promotion_plan.md`.

Execution/shutdown receiving (2026-10-19): Armillary now exposes fallible actor completion; ESP keeps model execution and validates clean actor shutdown in its existing provider tests. Canonical source and consumer receipts remain in `mere/design_docs/mere_docs/implementation_strategy/2026-10-19_family_promotion_plan.md`.

Pool lifetime receiving (2026-10-21): the workspace and generated lock now pin
Armillary `92d5b15cf39aaac40a3f2d863ffee69c9be11bfc` for Q112 admission and
Q113 retirement. `cargo +1.99.1 test --workspace --locked -j2` passes 94 cases
with one existing benchmark ignored; `cargo +1.99.1 test -p esp --features actor
--lib --locked -j2` passes 88 cases, including all six inference actor cases;
the same actor library sweep passes on the supported Rust 1.97.1 minimum.
ESP continues to use dedicated `spawn_named` actors. Its actor normal dependency
tree resolves one canonical Armillary source without Tokio; its default normal
tree omits Armillary. Pool behavior belongs to the supplier and standalone
consumer qualification; model, GPU, native and product acceptance remain their
existing gates. Coordinated ESP → Knot → Mere/product receipt:
`mere/support/family-promotion/pool-lifetime-20261021.json`.

- [Embedding compatibility](2026-10-20_embedding_compatibility_plan.md) — landed and resident-qualified; provider identities and guarded dense/sparse persistence, with Eidetic P4 and resident I10 as the acceptance owners.
