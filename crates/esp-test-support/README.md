# ESP test support

This development-only package provides the tiny deterministic decoder helpers,
SmolLM2 instruction template and Qwen2 MoE qualification assets used by ESP
and its integration-test consumers. It depends on Safetensors, without the
ESP runtime, a decoder backend, a GPU backend or a model download.

`decoder` is shared source. The bundled instruction template and MoE assets
are qualification snapshots from ESP's templates and independent Transformers
fixtures. ESP checks their equality to its corresponding source assets, so a
fixture refresh must update the bundle in the same pass.
