//! ESP is Mere's portable model-execution boundary.
//!
//! [`infer`] owns text generation, streaming, and model capability matching.
//! [`embed`] owns embeddings, vector retrieval, and affinity computation.
//! Heavy model backends remain feature-gated; the default build is the two
//! dependency-light contracts and their deterministic test providers.

pub mod embed;
pub mod infer;
