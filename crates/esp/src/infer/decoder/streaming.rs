// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! One-layer-at-a-time execution. Sources return content-verified local
//! artifacts, never an entire checkpoint. No network acquisition lives here.

use super::super::InferError;
use std::sync::atomic::{AtomicU64, Ordering};

/// A host-owned source of standalone safetensors layer artifacts.
/// Implementations must check their declared size against `limit` before
/// allocating, and verify content identity before returning bytes.
pub trait LayerSource: Send + Sync {
    fn read_layer(&self, index: usize, limit: usize) -> Result<Vec<u8>, InferError>;
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
}

pub(crate) struct StreamingLayers {
    pub source: Box<dyn LayerSource>,
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
            source,
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
        }
    }
}
