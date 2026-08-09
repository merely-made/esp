//! Text-generation and streaming inference seam.
//!
//! One trait ([`InferenceProvider`]), a capability descriptor
//! ([`ModelCapability`]) so a caller matches a model to a runtime, and a
//! deterministic dependency-light stub ([`StubInferenceProvider`]) so the whole
//! pipeline (seam, streaming, prompt plumbing) tests on machines with no GPU
//! and no model. Model-backed backends land behind features: the own
//! burn-wgpu decoder first, then external endpoints (Ollama, llama.cpp) and
//! native runtimes (mistral.rs), each behind this same trait, selected by
//! capability, never bound as universal.
//!
//! Streaming is the primary call:
//! [`InferenceProvider::generate_streaming`] pushes each text fragment through
//! a callback as it is produced; [`InferenceProvider::generate`] is the
//! provided collect-it-all wrapper.
//!
//! The decoder (own Burn llama-family decoder, `decoder` / `decoder-wgpu`) and
//! the streaming actor (`actor`, on Armillary threads) moved here from Vates.
//! External OpenAI-compatible endpoints remain a separate roadmap lane.

#[cfg(feature = "actor")]
pub mod actor;
#[cfg(feature = "decoder")]
pub mod decoder;
pub mod provider;
pub mod stub;

#[cfg(feature = "actor")]
pub use actor::{InferCommand, InferUpdate, spawn_inference_actor};
pub use provider::{
    CapabilityQuery, GenerationRequest, InferError, InferenceProvider, ModelCapability,
};
pub use stub::StubInferenceProvider;
