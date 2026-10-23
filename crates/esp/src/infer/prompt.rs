// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Declared, inert chat templates. Strings are literal data, never programs.

use super::{BoundModelSession, GenerationRequest, InferError, InferenceProvider, ModelCapability};
use eidetic::typed::TypedPayload;
use eidetic::{Hash, ManifestId, SchemaRef};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// The only template grammar understood by this renderer.
pub const PROMPT_TEMPLATE_SCHEMA: &str = "esp.prompt-template/v1";
const PROMPT_TEMPLATE_SCHEMA_PAYLOAD: &[u8] = br#"{"format":"mere-native","schema_id":"esp.prompt-template/v1","body":{"version":1,"description":"Literal role delimiters, beginning and end of sequence, context delimiters and generation stops for a single answering turn.","required":["schema","bos","eos","system","user","assistant","system_prompt","context","stop"],"additional_properties":false}}"#;

/// Literal delimiters around a role's content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleTemplate {
    pub prefix: String,
    pub suffix: String,
}

/// A total template for a system instruction, optional context and one ask.
/// BOS is inserted once at the start. EOS is a generation stop, never appended
/// to the unfinished assistant turn. Tokenizer postprocessing must not add BOS
/// again; the library loader disables it when it binds the template.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptTemplate {
    pub schema: String,
    pub bos: String,
    pub eos: String,
    pub system: RoleTemplate,
    pub user: RoleTemplate,
    pub assistant: RoleTemplate,
    pub system_prompt: String,
    pub context: RoleTemplate,
    pub stop: Vec<String>,
}

impl TypedPayload for PromptTemplate {
    fn schema_ref() -> SchemaRef {
        SchemaRef::from_id(ManifestId::of_blob(PROMPT_TEMPLATE_SCHEMA_PAYLOAD))
    }
}

impl PromptTemplate {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, InferError> {
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|error| InferError::InvalidConfig(format!("prompt template: {error}")))?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), InferError> {
        if self.schema != PROMPT_TEMPLATE_SCHEMA {
            return Err(InferError::InvalidConfig(format!(
                "prompt template schema {:?}",
                self.schema
            )));
        }
        if self.stop.iter().any(String::is_empty) {
            return Err(InferError::InvalidConfig(
                "prompt template has an empty stop string".into(),
            ));
        }
        Ok(())
    }

    /// No interpolation occurs inside content, including delimiter-looking text.
    pub fn render(&self, question: &str, context: &str) -> Result<String, InferError> {
        self.validate()?;
        if question.trim().is_empty() {
            return Err(InferError::InvalidRequest("empty question".into()));
        }
        let mut out = self.bos.clone();
        if !self.system_prompt.is_empty() {
            out.push_str(&self.system.prefix);
            out.push_str(&self.system_prompt);
            out.push_str(&self.system.suffix);
        }
        out.push_str(&self.user.prefix);
        if !context.is_empty() {
            out.push_str(&self.context.prefix);
            out.push_str(context);
            out.push_str(&self.context.suffix);
        }
        out.push_str(question);
        out.push_str(&self.user.suffix);
        out.push_str(&self.assistant.prefix);
        Ok(out)
    }

    pub fn stops(&self) -> Vec<String> {
        let mut stops = self.stop.clone();
        for stop in [&self.eos, &self.assistant.suffix] {
            if !stop.is_empty() && !stops.contains(stop) {
                stops.push(stop.clone());
            }
        }
        stops
    }
}

/// Actor-compatible wrapper: renders a raw question and executes only against
/// the exact bound session and the exact template bytes loaded from Eidetic.
pub struct SessionProvider<P> {
    bound: BoundModelSession<P>,
    template: PromptTemplate,
    template_bytes: Vec<u8>,
    max_tokens: usize,
}

impl<P: InferenceProvider> SessionProvider<P> {
    pub fn new(
        bound: BoundModelSession<P>,
        template_bytes: Vec<u8>,
        max_tokens: usize,
    ) -> Result<Self, InferError> {
        let template = PromptTemplate::from_bytes(&template_bytes)?;
        if max_tokens == 0 {
            return Err(InferError::InvalidConfig(
                "max_tokens must be greater than zero".into(),
            ));
        }
        // Validate the exact hash before announcing a ready provider.
        bound.prepare(GenerationRequest::default(), &template_bytes)?;
        Ok(Self {
            bound,
            template,
            template_bytes,
            max_tokens,
        })
    }

    pub fn bound(&self) -> &BoundModelSession<P> {
        &self.bound
    }

    pub fn prepare(
        &self,
        request: &GenerationRequest,
        context: &str,
    ) -> Result<super::PreparedGenerationRequest, InferError> {
        let mut rendered = request.clone();
        rendered.prompt = self.template.render(&request.prompt, context)?;
        zeroize::Zeroize::zeroize(&mut rendered.context);
        rendered.max_tokens = request.max_tokens.min(self.max_tokens);
        // Grammar acceptance terminates constrained documents. Template stops
        // must not truncate their JSON strings. Explicit caller stops remain
        // in the request and are refused by the constrained decoder.
        if rendered.output_grammar.is_none() {
            for stop in self.template.stops() {
                if !rendered.stop.contains(&stop) {
                    rendered.stop.push(stop);
                }
            }
        }
        self.bound
            .prepare(rendered, &self.template_bytes)
            .map_err(Into::into)
    }
}

impl<P: InferenceProvider> InferenceProvider for SessionProvider<P> {
    fn generate_checkpointed(
        &self,
        request: &GenerationRequest,
        resume: Option<&super::provider::GenerationCheckpoint>,
        commit: &mut dyn FnMut(&super::provider::GenerationCheckpoint) -> Result<(), InferError>,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<super::provider::GenerationCheckpoint, InferError> {
        let identity = self.session_identity();
        if resume.is_some_and(|checkpoint| checkpoint.session != identity) {
            return Err(InferError::InvalidRequest(
                "checkpoint belongs to a different exact session".into(),
            ));
        }
        // Render only at creation. A resume uses the exact persisted bytes,
        // including the originally chosen seed and stop strings.
        let prepared = match resume {
            Some(checkpoint) => checkpoint.request.clone(),
            None => self.prepare(request, &request.context)?.request,
        };
        let mut checkpoint = self.bound.provider().generate_checkpointed(
            &prepared,
            resume,
            &mut |checkpoint| {
                let mut checkpoint = checkpoint.clone();
                checkpoint.session = identity.clone();
                commit(&checkpoint)
            },
            stop,
        )?;
        checkpoint.session = identity;
        Ok(checkpoint)
    }

    fn capability(&self) -> &ModelCapability {
        self.bound.provider().capability()
    }

    fn session_identity(&self) -> Option<super::provider::SessionIdentity> {
        Some(super::provider::SessionIdentity {
            session_id: self.bound.session_id().to_string(),
            prompt_template_hash: Hash::of(&self.template_bytes).to_string(),
        })
    }

    fn generate_streaming(
        &self,
        request: &GenerationRequest,
        on_token: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<String, InferError> {
        let prepared = self.prepare(request, &request.context)?;
        self.bound.generate_prepared_streaming(&prepared, on_token)
    }
}
