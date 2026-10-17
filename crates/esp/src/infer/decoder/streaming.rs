// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! One-layer-at-a-time execution. Sources return content-verified local
//! artifacts, never an entire checkpoint. No network acquisition lives here.

use super::super::InferError;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A host-owned source of standalone safetensors layer artifacts.
/// Implementations must check their declared size against `limit` before
/// allocating, and verify content identity before returning bytes.
pub trait LayerSource: Send + Sync {
    fn read_layer(&self, index: usize, limit: usize) -> Result<Vec<u8>, InferError>;
    fn read_expert(
        &self,
        _layer: usize,
        _expert: usize,
        _limit: usize,
    ) -> Result<Vec<u8>, InferError> {
        Err(InferError::InvalidWeights(
            "layer source has no expert reader".into(),
        ))
    }
}

/// Compose host-owned verified artifact readers without acquiring any network
/// or durable-store authority inside the decoder.
pub fn with_expert_reader<L, E>(layers: L, experts: E) -> impl LayerSource
where
    L: Fn(usize, usize) -> Result<Vec<u8>, InferError> + Send + Sync,
    E: Fn(usize, usize, usize) -> Result<Vec<u8>, InferError> + Send + Sync,
{
    struct Readers<L, E>(L, E);
    impl<L, E> LayerSource for Readers<L, E>
    where
        L: Fn(usize, usize) -> Result<Vec<u8>, InferError> + Send + Sync,
        E: Fn(usize, usize, usize) -> Result<Vec<u8>, InferError> + Send + Sync,
    {
        fn read_layer(&self, index: usize, limit: usize) -> Result<Vec<u8>, InferError> {
            self.0(index, limit)
        }
        fn read_expert(
            &self,
            layer: usize,
            expert: usize,
            limit: usize,
        ) -> Result<Vec<u8>, InferError> {
            self.1(layer, expert, limit)
        }
    }
    Readers(layers, experts)
}

impl LayerSource for Arc<dyn LayerSource> {
    fn read_layer(&self, index: usize, limit: usize) -> Result<Vec<u8>, InferError> {
        (**self).read_layer(index, limit)
    }
    fn read_expert(
        &self,
        layer: usize,
        expert: usize,
        limit: usize,
    ) -> Result<Vec<u8>, InferError> {
        (**self).read_expert(layer, expert, limit)
    }
}

impl<F> LayerSource for F
where
    F: Fn(usize, usize) -> Result<Vec<u8>, InferError> + Send + Sync,
{
    fn read_layer(&self, index: usize, limit: usize) -> Result<Vec<u8>, InferError> {
        self(index, limit)
    }
}

/// Explicit per-provider limits, retained in the library's weight index.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingOptions {
    pub query_chunk: usize,
    pub max_layer_bytes: usize,
    pub max_shared_bytes: usize,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            query_chunk: 32,
            max_layer_bytes: 256 * 1024 * 1024,
            max_shared_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl StreamingOptions {
    pub fn validate(&self) -> Result<(), InferError> {
        if self.query_chunk == 0 || self.max_layer_bytes == 0 || self.max_shared_bytes == 0 {
            return Err(InferError::InvalidConfig(
                "streaming limits must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// Logical verified artifact reads. OS cache hits still count as reads;
/// these are not physical disk-traffic measurements or a process RSS bound.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StreamingReadStats {
    pub layer_reads: u64,
    pub encoded_bytes_read: u64,
    pub largest_encoded_layer: u64,
    pub expert_reads: u64,
    pub expert_encoded_bytes_read: u64,
    pub largest_encoded_expert: u64,
}

#[derive(Default, Debug)]
pub(crate) struct ExpertReads {
    pub count: AtomicU64,
    pub bytes: AtomicU64,
    pub largest: AtomicU64,
}

pub(crate) struct StreamingLayers {
    pub source: Arc<dyn LayerSource>,
    pub expert_reads: Arc<ExpertReads>,
    pub options: StreamingOptions,
    reads: AtomicU64,
    bytes: AtomicU64,
    largest: AtomicU64,
}

impl std::fmt::Debug for StreamingLayers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingLayers")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl StreamingLayers {
    pub fn new(source: Box<dyn LayerSource>, options: StreamingOptions) -> Self {
        Self {
            source: Arc::from(source),
            expert_reads: Arc::new(ExpertReads::default()),
            options,
            reads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            largest: AtomicU64::new(0),
        }
    }

    pub fn read(&self, index: usize) -> Result<Vec<u8>, InferError> {
        let bytes = self
            .source
            .read_layer(index, self.options.max_layer_bytes)?;
        if bytes.len() > self.options.max_layer_bytes {
            return Err(InferError::InvalidWeights(
                "layer source exceeded its byte limit".into(),
            ));
        }
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.largest
            .fetch_max(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    pub fn stats(&self) -> StreamingReadStats {
        StreamingReadStats {
            layer_reads: self.reads.load(Ordering::Relaxed),
            encoded_bytes_read: self.bytes.load(Ordering::Relaxed),
            largest_encoded_layer: self.largest.load(Ordering::Relaxed),
            expert_reads: self.expert_reads.count.load(Ordering::Relaxed),
            expert_encoded_bytes_read: self.expert_reads.bytes.load(Ordering::Relaxed),
            largest_encoded_expert: self.expert_reads.largest.load(Ordering::Relaxed),
        }
    }
}
