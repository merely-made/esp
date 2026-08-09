//! Inference provider trait + capability vocabulary.

use serde::{Deserialize, Serialize};

/// What a loaded model/runtime pair can do: the matching surface a caller uses
/// to pick a provider, and a status surface a UI can show.
///
/// Deliberately small: identity, context window, quantization, the
/// runtime-loader id, and the streaming flag. A richer LoRA-adapter
/// compatibility envelope is a separate future contract on an `AdapterLoader`
/// seam, not this descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapability {
    /// Model identity, HF-style where applicable
    /// (e.g. `meta-llama/Llama-3.2-1B-Instruct`), or a stub id.
    pub model_id: String,
    /// Context window in tokens.
    pub context_window: usize,
    /// Quantization scheme (e.g. `q4_0`); `None` = unquantized.
    pub quantization: Option<String>,
    /// Runtime-loader id: which engine executes this model
    /// (e.g. `stub`, `burn-wgpu`, `endpoint`, `mistral.rs`).
    pub loader: String,
    /// Whether fragments stream as they are produced, or arrive only as
    /// one final text.
    pub streaming: bool,
}

/// A caller's requirements when picking a provider by capability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityQuery {
    /// Minimum context window the caller needs, in tokens.
    pub min_context_window: usize,
    /// Require a specific runtime loader (`None` = any).
    pub loader: Option<String>,
    /// Require streaming delivery.
    pub require_streaming: bool,
}

impl ModelCapability {
    /// Whether this capability satisfies the caller's requirements.
    pub fn satisfies(&self, query: &CapabilityQuery) -> bool {
        self.context_window >= query.min_context_window
            && query.loader.as_ref().is_none_or(|l| l == &self.loader)
            && (!query.require_streaming || self.streaming)
    }
}

/// One generation call's inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// The full prompt (any chat templating happens above this seam).
    pub prompt: String,
    /// Hard cap on generated fragments/tokens.
    pub max_tokens: usize,
    /// Sampling temperature; `0.0` = deterministic/greedy. Stub providers
    /// ignore it (they are deterministic by construction).
    pub temperature: f32,
    /// Nucleus (top-p) cutoff for sampled generation; `None` = no cut.
    /// Ignored when `temperature` is `0.0`.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// RNG seed for sampled generation: `Some` = bit-reproducible token
    /// stream, `None` = the provider draws a fresh seed. Ignored when
    /// `temperature` is `0.0`.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Generation stops before emitting any of these sequences.
    pub stop: Vec<String>,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 256,
            temperature: 0.0,
            top_p: None,
            seed: None,
            stop: Vec::new(),
        }
    }
}

/// Reasons an inference call could fail.
#[derive(Debug, Clone, PartialEq)]
pub enum InferError {
    /// Prompt exceeded the model's context window.
    PromptTooLong { length: usize, limit: usize },
    /// The provider type can hold a model but no weights are loaded yet.
    ModelNotLoaded,
    /// The underlying runtime (Burn / CPU / wgpu / native engine) failed.
    Backend(String),
    /// The request itself is invalid (empty prompt, zero max_tokens, ...).
    InvalidRequest(String),
    /// A model config could not be parsed or is internally inconsistent.
    InvalidConfig(String),
    /// Model weights were missing, malformed, or shape-mismatched.
    InvalidWeights(String),
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::PromptTooLong { length, limit } => {
                write!(f, "prompt too long: {length} > {limit}")
            }
            InferError::ModelNotLoaded => write!(f, "model not loaded"),
            InferError::Backend(msg) => write!(f, "backend: {msg}"),
            InferError::InvalidRequest(msg) => write!(f, "invalid request: {msg}"),
            InferError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            InferError::InvalidWeights(msg) => write!(f, "invalid weights: {msg}"),
        }
    }
}

impl std::error::Error for InferError {}

/// Provider that generates text from a prompt.
///
/// Implementations must be `Send + Sync` so one loaded instance can be shared
/// across threads (a threaded inference actor holds one). Streaming is
/// primary: fragments go through `on_token` in production order, and the
/// assembled full text is returned at the end. The callback's `ControlFlow`
/// return is the cancellation channel: `Break` stops generation after the
/// current fragment (the fragment has been delivered; whether the caller
/// surfaces it is the caller's choice). Callers that only want the final text
/// use [`generate`](Self::generate).
pub trait InferenceProvider: Send + Sync {
    /// What this provider's loaded model/runtime pair can do.
    fn capability(&self) -> &ModelCapability;

    /// Generate, pushing each fragment through `on_token` as it is
    /// produced; stop early when the callback returns
    /// [`ControlFlow::Break`]. Returns the full generated text (the
    /// concatenation of every delivered fragment).
    ///
    /// [`ControlFlow::Break`]: std::ops::ControlFlow::Break
    fn generate_streaming(
        &self,
        request: &GenerationRequest,
        on_token: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> Result<String, InferError>;

    /// Generate and return only the final text.
    fn generate(&self, request: &GenerationRequest) -> Result<String, InferError> {
        self.generate_streaming(request, &mut |_| std::ops::ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(loader: &str, window: usize, streaming: bool) -> ModelCapability {
        ModelCapability {
            model_id: "test/model".to_string(),
            context_window: window,
            quantization: None,
            loader: loader.to_string(),
            streaming,
        }
    }

    #[test]
    fn capability_satisfies_matching_query() {
        let c = cap("burn-wgpu", 4096, true);
        assert!(c.satisfies(&CapabilityQuery {
            min_context_window: 2048,
            loader: Some("burn-wgpu".to_string()),
            require_streaming: true,
        }));
    }

    #[test]
    fn capability_rejects_small_window_wrong_loader_no_streaming() {
        let c = cap("stub", 512, false);
        assert!(!c.satisfies(&CapabilityQuery {
            min_context_window: 1024,
            ..Default::default()
        }));
        assert!(!c.satisfies(&CapabilityQuery {
            loader: Some("burn-wgpu".to_string()),
            ..Default::default()
        }));
        assert!(!c.satisfies(&CapabilityQuery {
            require_streaming: true,
            ..Default::default()
        }));
    }

    #[test]
    fn capability_serde_roundtrip() {
        let c = cap("burn-wgpu", 8192, true);
        let s = serde_json::to_string(&c).unwrap();
        let back: ModelCapability = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn infer_error_display_is_stable() {
        assert_eq!(
            InferError::PromptTooLong {
                length: 9000,
                limit: 4096
            }
            .to_string(),
            "prompt too long: 9000 > 4096"
        );
        assert_eq!(InferError::ModelNotLoaded.to_string(), "model not loaded");
    }
}
