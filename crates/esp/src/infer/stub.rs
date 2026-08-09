//! Deterministic stub provider: the seam's dependency-light floor.
//!
//! Exact-match prompts return their stub response; anything else echoes
//! the prompt (`echo: <prompt>`). Output streams word by word (with leading
//! spaces on continuation fragments), honouring `max_tokens` and `stop`, so
//! callers exercise real streaming/truncation logic without a model, a GPU,
//! or nondeterminism.

use std::collections::HashMap;

use crate::infer::provider::{GenerationRequest, InferError, InferenceProvider, ModelCapability};

/// Deterministic stub/echo inference provider for tests and dev harnesses.
#[derive(Debug, Clone)]
pub struct StubInferenceProvider {
    capability: ModelCapability,
    responses: HashMap<String, String>,
}

impl StubInferenceProvider {
    /// A provider that echoes every prompt.
    pub fn new() -> Self {
        Self {
            capability: ModelCapability {
                model_id: "stub/echo".to_string(),
                context_window: 4096,
                quantization: None,
                loader: "stub".to_string(),
                streaming: true,
            },
            responses: HashMap::new(),
        }
    }

    /// Add an exact-match stub response.
    pub fn with_response(mut self, prompt: impl Into<String>, response: impl Into<String>) -> Self {
        self.responses.insert(prompt.into(), response.into());
        self
    }

    /// Override the advertised context window (tests exercising
    /// `PromptTooLong` paths shrink it).
    pub fn with_context_window(mut self, tokens: usize) -> Self {
        self.capability.context_window = tokens;
        self
    }

    /// Whitespace-token count, the stub's stand-in for real tokenization.
    fn token_len(text: &str) -> usize {
        text.split_whitespace().count()
    }
}

impl Default for StubInferenceProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceProvider for StubInferenceProvider {
    fn capability(&self) -> &ModelCapability {
        &self.capability
    }

    fn generate_streaming(
        &self,
        request: &GenerationRequest,
        on_token: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> Result<String, InferError> {
        if request.prompt.is_empty() {
            return Err(InferError::InvalidRequest("empty prompt".to_string()));
        }
        if request.max_tokens == 0 {
            return Err(InferError::InvalidRequest("max_tokens is 0".to_string()));
        }
        let prompt_tokens = Self::token_len(&request.prompt);
        if prompt_tokens > self.capability.context_window {
            return Err(InferError::PromptTooLong {
                length: prompt_tokens,
                limit: self.capability.context_window,
            });
        }

        let response = self
            .responses
            .get(&request.prompt)
            .cloned()
            .unwrap_or_else(|| format!("echo: {}", request.prompt));

        let mut out = String::new();
        for (i, word) in response.split_whitespace().enumerate() {
            if i >= request.max_tokens {
                break;
            }
            let fragment = if i == 0 {
                word.to_string()
            } else {
                format!(" {word}")
            };
            let candidate = format!("{out}{fragment}");
            if request.stop.iter().any(|s| candidate.contains(s.as_str())) {
                // Stop before emitting the fragment that completes the
                // stop sequence, matching real-runtime stop semantics.
                break;
            }
            let flow = on_token(&fragment);
            out = candidate;
            if flow.is_break() {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echoes_unknown_prompts_and_streams_in_order() {
        let p = StubInferenceProvider::new();
        let mut fragments = Vec::new();
        let full = p
            .generate_streaming(
                &GenerationRequest {
                    prompt: "hello world".to_string(),
                    ..Default::default()
                },
                &mut |t| {
                    fragments.push(t.to_string());
                    std::ops::ControlFlow::Continue(())
                },
            )
            .unwrap();
        assert_eq!(full, "echo: hello world");
        assert_eq!(fragments, vec!["echo:", " hello", " world"]);
        assert_eq!(fragments.concat(), full);
    }

    #[test]
    fn stub_response_wins_over_echo() {
        let p = StubInferenceProvider::new().with_response("ping", "pong");
        let out = p
            .generate(&GenerationRequest {
                prompt: "ping".to_string(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(out, "pong");
    }

    #[test]
    fn max_tokens_truncates() {
        let p = StubInferenceProvider::new().with_response("q", "one two three four");
        let out = p
            .generate(&GenerationRequest {
                prompt: "q".to_string(),
                max_tokens: 2,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(out, "one two");
    }

    #[test]
    fn stop_sequence_halts_before_emitting() {
        let p = StubInferenceProvider::new().with_response("q", "alpha beta STOP gamma");
        let mut fragments = Vec::new();
        let out = p
            .generate_streaming(
                &GenerationRequest {
                    prompt: "q".to_string(),
                    stop: vec!["STOP".to_string()],
                    ..Default::default()
                },
                &mut |t| {
                    fragments.push(t.to_string());
                    std::ops::ControlFlow::Continue(())
                },
            )
            .unwrap();
        assert_eq!(out, "alpha beta");
        assert!(!fragments.iter().any(|f| f.contains("STOP")));
    }

    #[test]
    fn callback_break_stops_after_current_fragment() {
        let p = StubInferenceProvider::new().with_response("q", "one two three four");
        let mut fragments = Vec::new();
        let out = p
            .generate_streaming(
                &GenerationRequest {
                    prompt: "q".to_string(),
                    ..Default::default()
                },
                &mut |t| {
                    fragments.push(t.to_string());
                    if fragments.len() == 2 {
                        std::ops::ControlFlow::Break(())
                    } else {
                        std::ops::ControlFlow::Continue(())
                    }
                },
            )
            .unwrap();
        assert_eq!(out, "one two", "generation stops after the Break fragment");
        assert_eq!(fragments.len(), 2);
    }

    #[test]
    fn prompt_too_long_reports_window() {
        let p = StubInferenceProvider::new().with_context_window(3);
        let err = p
            .generate(&GenerationRequest {
                prompt: "one two three four five".to_string(),
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(
            err,
            InferError::PromptTooLong {
                length: 5,
                limit: 3
            }
        );
    }

    #[test]
    fn invalid_requests_rejected() {
        let p = StubInferenceProvider::new();
        assert!(matches!(
            p.generate(&GenerationRequest::default()).unwrap_err(),
            InferError::InvalidRequest(_)
        ));
        assert!(matches!(
            p.generate(&GenerationRequest {
                prompt: "x".to_string(),
                max_tokens: 0,
                ..Default::default()
            })
            .unwrap_err(),
            InferError::InvalidRequest(_)
        ));
    }

    #[test]
    fn provider_is_object_safe_and_shared() {
        let p: Box<dyn InferenceProvider> = Box::new(StubInferenceProvider::new());
        assert_eq!(p.capability().loader, "stub");
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StubInferenceProvider>();
    }
}
