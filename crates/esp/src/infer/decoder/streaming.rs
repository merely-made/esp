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
    /// Shared immutable bytes let an owner retain a verified artifact without
    /// copying it back into a fresh buffer on every decoding step.
    fn read_layer_shared(&self, index: usize, limit: usize) -> Result<Arc<Vec<u8>>, InferError> {
        self.read_layer(index, limit).map(Arc::new)
    }
    fn cache_stats(&self) -> Option<LayerCacheStats> {
        None
    }
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
    fn read_layer_shared(&self, index: usize, limit: usize) -> Result<Arc<Vec<u8>>, InferError> {
        (**self).read_layer_shared(index, limit)
    }
    fn cache_stats(&self) -> Option<LayerCacheStats> {
        (**self).cache_stats()
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

/// Host allocation accounting, separate from logical decoder reads and GPU
/// residence. Container/allocator overhead is not a process RSS measurement.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct LayerCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub retained_capacity_bytes: usize,
    pub entries: usize,
    pub capacity_budget_bytes: usize,
}

/// Cache verified *immutable* artifacts from a fixed model index. The caller
/// must supply a reader that verifies the indexed content on a miss. Hits use
/// the same immutable allocation, rather than trusting mutable store bytes.
/// Only encoded host bytes are retained. This wrapper adds no device tensors;
/// the decoder's device placement is a separate owner choice. Experts retain
/// their existing uncached read path.
///
/// Admission is first-fit, capped at 128 entries and `budget` bytes of Vec
/// capacity. A sequential sweep larger than the budget cannot evict the whole
/// previous sweep before it reaches a reusable entry. Errors are never cached.
pub fn cache_verified_layers(source: Arc<dyn LayerSource>, budget: usize) -> Arc<dyn LayerSource> {
    if budget == 0 {
        return source;
    }
    struct Cache {
        entries: Vec<(usize, Arc<Vec<u8>>)>,
        stats: LayerCacheStats,
    }
    struct Cached {
        source: Arc<dyn LayerSource>,
        cache: std::sync::Mutex<Cache>,
    }
    impl LayerSource for Cached {
        fn read_layer(&self, index: usize, limit: usize) -> Result<Vec<u8>, InferError> {
            self.read_layer_shared(index, limit)
                .map(|bytes| (*bytes).clone())
        }
        fn read_layer_shared(
            &self,
            index: usize,
            limit: usize,
        ) -> Result<Arc<Vec<u8>>, InferError> {
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| InferError::Backend("layer cache poisoned".into()))?;
            if let Some((_, bytes)) = cache.entries.iter().find(|(key, _)| *key == index) {
                if bytes.len() > limit {
                    return Err(InferError::InvalidWeights(
                        "cached layer exceeds its byte limit".into(),
                    ));
                }
                let bytes = bytes.clone();
                cache.stats.hits += 1;
                return Ok(bytes);
            }
            cache.stats.misses += 1;
            let bytes = self.source.read_layer_shared(index, limit)?;
            if bytes.len() > limit {
                return Err(InferError::InvalidWeights(
                    "layer source exceeded its byte limit".into(),
                ));
            }
            if cache.entries.len() < 128
                && bytes.capacity()
                    <= cache.stats.capacity_budget_bytes - cache.stats.retained_capacity_bytes
            {
                cache.stats.retained_capacity_bytes += bytes.capacity();
                cache.entries.push((index, bytes.clone()));
                cache.stats.entries = cache.entries.len();
            }
            Ok(bytes)
        }
        fn read_expert(
            &self,
            layer: usize,
            expert: usize,
            limit: usize,
        ) -> Result<Vec<u8>, InferError> {
            self.source.read_expert(layer, expert, limit)
        }
        fn cache_stats(&self) -> Option<LayerCacheStats> {
            self.cache.lock().ok().map(|cache| cache.stats.clone())
        }
    }
    Arc::new(Cached {
        source,
        cache: std::sync::Mutex::new(Cache {
            entries: vec![],
            stats: LayerCacheStats {
                capacity_budget_bytes: budget,
                ..Default::default()
            },
        }),
    })
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

/// Logical tensor payload retained on the execution device. Shared weights,
/// KV, workspace, allocator overhead and the transient next layer are excluded.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct DeviceLayerCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub retained_payload_bytes: u64,
    pub payload_budget_bytes: u64,
    pub layers: usize,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified_cache: Option<LayerCacheStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_cache: Option<DeviceLayerCacheStats>,
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

    pub fn read(&self, index: usize) -> Result<Arc<Vec<u8>>, InferError> {
        let bytes = self
            .source
            .read_layer_shared(index, self.options.max_layer_bytes)?;
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
            verified_cache: self.source.cache_stats(),
            device_cache: None,
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn immutable_hits_share_the_verified_allocation_and_recheck_limits() {
        let reads = Arc::new(AtomicUsize::new(0));
        let count = reads.clone();
        let source = cache_verified_layers(
            Arc::new(move |index, _| {
                count.fetch_add(1, Ordering::Relaxed);
                Ok(vec![index as u8; 32])
            }),
            64,
        );
        let first = source.read_layer_shared(3, 32).unwrap();
        let second = source.clone().read_layer_shared(3, 32).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(source.read_layer(3, 32).unwrap(), vec![3; 32]);
        assert!(source.read_layer_shared(3, 31).is_err());
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        let stats = source.cache_stats().unwrap();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.retained_capacity_bytes, 32);
    }

    #[test]
    fn budget_counts_allocation_capacity_and_does_not_thrash_on_a_sweep() {
        let source = cache_verified_layers(
            Arc::new(|index, _| {
                let mut value = Vec::with_capacity(if index == 2 { 128 } else { 32 });
                value.resize(32, index as u8);
                Ok(value)
            }),
            64,
        );
        for _ in 0..3 {
            for index in 0..4 {
                source.read_layer_shared(index, 32).unwrap();
            }
        }
        let stats = source.cache_stats().unwrap();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.retained_capacity_bytes, 64);
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 8);
    }

    #[test]
    fn failed_or_oversized_reads_never_enter_the_cache() {
        let count = Arc::new(AtomicUsize::new(0));
        let reads = count.clone();
        let source = cache_verified_layers(
            Arc::new(move |_, _| match reads.fetch_add(1, Ordering::Relaxed) {
                0 => Err(InferError::InvalidWeights("content hash mismatch".into())),
                1 => Ok(vec![0; 33]),
                _ => Ok(vec![1; 32]),
            }),
            64,
        );
        assert!(source.read_layer_shared(0, 32).is_err());
        assert!(source.read_layer_shared(0, 32).is_err());
        assert_eq!(source.cache_stats().unwrap().entries, 0);
        assert_eq!(
            source.read_layer_shared(0, 32).unwrap().as_slice(),
            &[1; 32]
        );
        source.read_layer_shared(0, 32).unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }
}
