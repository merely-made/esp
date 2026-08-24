//! Immutable model sessions and adapter-loader contracts.
//!
//! A session binds content identities and an ordered adapter set before a
//! request reaches an [`InferenceProvider`]. The provider remains the portable
//! streaming execution seam; loaders build session-bound providers without
//! mutating adapter state shared by other residents.

use std::collections::HashSet;
use std::ops::ControlFlow;

use eidetic::{Hash, ManifestId, ModelAdapterManifest};
use serde::{Deserialize, Serialize};

use super::{GenerationRequest, InferError, InferenceProvider};

/// One adapter selected for a session, in application order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdapterSelection {
    /// Content-addressed adapter manifest.
    pub manifest_ref: ManifestId,
    /// User-selected multiplier applied after the adapter's own alpha/rank scale.
    pub scale: f32,
}

/// Immutable compatibility envelope for one loaded model session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSession {
    /// Content-addressed base-model manifest.
    pub base_model_ref: ManifestId,
    /// Runtime-visible model id expected from the provider.
    pub model_id: String,
    /// Exact tokenizer blob referenced by the base and every adapter.
    pub tokenizer_ref: ManifestId,
    /// Hash of the prompt/chat template bytes used above the provider seam.
    pub prompt_template_hash: Hash,
    /// Quantization contract; `None` means unquantized resident weights.
    pub quantization: Option<String>,
    /// Exact runtime loader id.
    pub loader: String,
    /// Ordered adapter composition selected by the user or caller.
    pub adapters: Vec<AdapterSelection>,
}

/// A request prepared against one exact [`ModelSession`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedGenerationRequest {
    /// Content identity of the session that prepared this request.
    pub session_id: ManifestId,
    /// Full rendered prompt and sampling controls for the provider.
    pub request: GenerationRequest,
}

/// A resolved adapter manifest plus the opaque runtime artifacts it names.
pub struct AdapterArtifact<'a> {
    /// Manifest identity expected by the session's ordered selection.
    pub manifest_ref: ManifestId,
    /// Loaded, schema-checked adapter manifest.
    pub manifest: &'a ModelAdapterManifest,
    /// Runtime-specific adapter configuration bytes.
    pub config_bytes: &'a [u8],
    /// Opaque adapter weight bytes.
    pub weight_bytes: &'a [u8],
}

/// Validation failures while constructing or using an immutable session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSessionError(String);

impl ModelSessionError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for ModelSessionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ModelSessionError {}

impl From<ModelSessionError> for InferError {
    fn from(error: ModelSessionError) -> Self {
        InferError::InvalidConfig(format!("model session: {error}"))
    }
}

impl ModelSession {
    /// Validate the session's own unambiguous identity fields.
    pub fn validate(&self) -> Result<(), ModelSessionError> {
        if self.model_id.trim().is_empty() {
            return Err(ModelSessionError::new("model_id is empty"));
        }
        if self.loader.trim().is_empty() {
            return Err(ModelSessionError::new("loader is empty"));
        }
        let mut seen = HashSet::with_capacity(self.adapters.len());
        for selection in &self.adapters {
            if !selection.scale.is_finite() || selection.scale <= 0.0 {
                return Err(ModelSessionError::new(format!(
                    "adapter {} scale must be finite and greater than zero, got {}",
                    selection.manifest_ref, selection.scale
                )));
            }
            if !seen.insert(selection.manifest_ref) {
                return Err(ModelSessionError::new(format!(
                    "adapter {} appears more than once",
                    selection.manifest_ref
                )));
            }
        }
        Ok(())
    }

    /// Validate exact, ordered adapter manifests against this session.
    pub fn validate_adapters(
        &self,
        artifacts: &[AdapterArtifact<'_>],
    ) -> Result<(), ModelSessionError> {
        self.validate()?;
        if artifacts.len() != self.adapters.len() {
            return Err(ModelSessionError::new(format!(
                "adapter count mismatch: session selects {}, loader received {}",
                self.adapters.len(),
                artifacts.len()
            )));
        }
        for (index, (selection, artifact)) in self.adapters.iter().zip(artifacts.iter()).enumerate()
        {
            if selection.manifest_ref != artifact.manifest_ref {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} order mismatch: session selects {}, loader received {}",
                    selection.manifest_ref, artifact.manifest_ref
                )));
            }
            artifact.manifest.validate().map_err(|message| {
                ModelSessionError::new(format!("adapter {index} manifest invalid: {message}"))
            })?;
            if ManifestId::of_blob(artifact.config_bytes) != artifact.manifest.adapter_config_blob {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} config bytes do not match the manifest"
                )));
            }
            if ManifestId::of_blob(artifact.weight_bytes) != artifact.manifest.adapter_blob {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} weight bytes do not match the manifest"
                )));
            }
            if artifact.manifest.base_model_ref != self.base_model_ref {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} base model mismatch"
                )));
            }
            if artifact.manifest.tokenizer_ref != self.tokenizer_ref {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} tokenizer mismatch"
                )));
            }
            if artifact.manifest.prompt_template_hash != self.prompt_template_hash {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} prompt template mismatch"
                )));
            }
            if artifact.manifest.quantization_assumption != self.quantization {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} quantization mismatch"
                )));
            }
            let loaders = &artifact.manifest.runtime_compat.known_loaders;
            if !loaders.iter().any(|loader| loader == &self.loader) {
                return Err(ModelSessionError::new(format!(
                    "adapter {index} was not verified for loader {}",
                    self.loader
                )));
            }
        }
        Ok(())
    }

    /// Content identity of the complete ordered session envelope.
    pub fn id(&self) -> Result<ManifestId, ModelSessionError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| {
            ModelSessionError::new(format!("serialize identity envelope: {error}"))
        })?;
        Ok(ManifestId::of_blob(&bytes))
    }
}

/// A provider bound to one immutable session envelope.
pub struct BoundModelSession<P> {
    session: ModelSession,
    session_id: ManifestId,
    provider: P,
}

impl<P: InferenceProvider> BoundModelSession<P> {
    /// Bind a loaded provider after adapter compatibility has been validated.
    pub fn bind(session: ModelSession, provider: P) -> Result<Self, ModelSessionError> {
        let capability = provider.capability();
        if capability.model_id != session.model_id {
            return Err(ModelSessionError::new(format!(
                "provider model {} does not match session model {}",
                capability.model_id, session.model_id
            )));
        }
        if capability.loader != session.loader {
            return Err(ModelSessionError::new(format!(
                "provider loader {} does not match session loader {}",
                capability.loader, session.loader
            )));
        }
        if capability.quantization != session.quantization {
            return Err(ModelSessionError::new(
                "provider quantization does not match session quantization",
            ));
        }
        let session_id = session.id()?;
        Ok(Self {
            session,
            session_id,
            provider,
        })
    }

    /// Immutable session envelope.
    pub fn session(&self) -> &ModelSession {
        &self.session
    }

    /// Content identity of [`Self::session`].
    pub fn session_id(&self) -> ManifestId {
        self.session_id
    }

    /// Underlying runtime provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Bind an already-rendered request to the exact template bytes used above
    /// the provider seam.
    pub fn prepare(
        &self,
        request: GenerationRequest,
        prompt_template: &[u8],
    ) -> Result<PreparedGenerationRequest, ModelSessionError> {
        let actual = Hash::of(prompt_template);
        if actual != self.session.prompt_template_hash {
            return Err(ModelSessionError::new(format!(
                "prompt template mismatch: session expects {}, caller supplied {}",
                self.session.prompt_template_hash, actual
            )));
        }
        Ok(PreparedGenerationRequest {
            session_id: self.session_id,
            request,
        })
    }

    /// Execute a request only when it was prepared for this exact session.
    pub fn generate_prepared(
        &self,
        prepared: &PreparedGenerationRequest,
    ) -> Result<String, InferError> {
        self.generate_prepared_streaming(prepared, &mut |_| ControlFlow::Continue(()))
    }

    /// Streaming counterpart to [`Self::generate_prepared`].
    pub fn generate_prepared_streaming(
        &self,
        prepared: &PreparedGenerationRequest,
        on_token: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<String, InferError> {
        if prepared.session_id != self.session_id {
            return Err(InferError::InvalidRequest(format!(
                "prepared request belongs to session {}, active session is {}",
                prepared.session_id, self.session_id
            )));
        }
        self.provider
            .generate_streaming(&prepared.request, on_token)
    }
}

/// Runtime-specific adapter loader producing a session-bound provider.
pub trait AdapterLoader {
    /// Provider implementation produced by this loader.
    type Provider: InferenceProvider;

    /// Load the exact ordered adapter artifacts without mutating any other
    /// loaded session.
    fn load_session(
        &self,
        session: ModelSession,
        artifacts: &[AdapterArtifact<'_>],
    ) -> Result<BoundModelSession<Self::Provider>, InferError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::StubInferenceProvider;
    use eidetic::AdapterRuntimeCompat;

    fn adapter_manifest(session: &ModelSession, name: &str) -> ModelAdapterManifest {
        ModelAdapterManifest {
            name: name.into(),
            base_model_ref: session.base_model_ref,
            adapter_blob: ManifestId::of_blob(name.as_bytes()),
            adapter_config_blob: ManifestId::of_blob(b"{}"),
            adapter_format: "peft-lora".into(),
            adapter_format_version: "1".into(),
            runtime_compat: AdapterRuntimeCompat {
                minimum_capabilities: vec!["peft-lora".into()],
                known_loaders: vec![session.loader.clone()],
                converter_lineage: vec![],
            },
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".into()],
            tokenizer_ref: session.tokenizer_ref,
            prompt_template_hash: session.prompt_template_hash,
            quantization_assumption: session.quantization.clone(),
            training_corpus_root: None,
            training_method: serde_json::Value::Null,
            eval_results: None,
        }
    }

    fn session(adapter_refs: Vec<ManifestId>) -> ModelSession {
        ModelSession {
            base_model_ref: ManifestId::of_blob(b"base"),
            model_id: "stub/echo".into(),
            tokenizer_ref: ManifestId::of_blob(b"tokenizer"),
            prompt_template_hash: Hash::of(b"{{ prompt }}"),
            quantization: None,
            loader: "stub".into(),
            adapters: adapter_refs
                .into_iter()
                .map(|manifest_ref| AdapterSelection {
                    manifest_ref,
                    scale: 1.0,
                })
                .collect(),
        }
    }

    #[test]
    fn order_changes_session_identity() {
        let a = ManifestId::of_blob(b"adapter-a");
        let b = ManifestId::of_blob(b"adapter-b");
        assert_ne!(
            session(vec![a, b]).id().unwrap(),
            session(vec![b, a]).id().unwrap()
        );
    }

    #[test]
    fn exact_adapter_envelope_passes_and_mismatch_rejects() {
        let adapter_ref = ManifestId::of_blob(b"adapter-a");
        let session = session(vec![adapter_ref]);
        let manifest = adapter_manifest(&session, "a");
        let artifact = AdapterArtifact {
            manifest_ref: adapter_ref,
            manifest: &manifest,
            config_bytes: b"{}",
            weight_bytes: b"a",
        };
        session.validate_adapters(&[artifact]).unwrap();

        let mut mismatched = manifest;
        mismatched.tokenizer_ref = ManifestId::of_blob(b"other tokenizer");
        let artifact = AdapterArtifact {
            manifest_ref: adapter_ref,
            manifest: &mismatched,
            config_bytes: b"{}",
            weight_bytes: b"a",
        };
        assert!(
            session
                .validate_adapters(&[artifact])
                .unwrap_err()
                .to_string()
                .contains("tokenizer")
        );
    }

    #[test]
    fn prepared_request_cannot_cross_session_or_template() {
        let session = session(vec![]);
        let provider = StubInferenceProvider::new();
        let bound = BoundModelSession::bind(session, provider).unwrap();
        assert!(
            bound
                .prepare(GenerationRequest::default(), b"wrong")
                .unwrap_err()
                .to_string()
                .contains("template")
        );
        let mut prepared = bound
            .prepare(GenerationRequest::default(), b"{{ prompt }}")
            .unwrap();
        prepared.session_id = ManifestId::of_blob(b"other session");
        assert!(matches!(
            bound.generate_prepared(&prepared),
            Err(InferError::InvalidRequest(_))
        ));
    }
}
