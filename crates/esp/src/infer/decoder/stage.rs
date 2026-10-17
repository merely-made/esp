// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Contiguous resident decoder stages. Transport and lease authority belong
//! to the host; a stage owns only its layer weights and transient KV cache.

use std::sync::Mutex;

use burn::tensor::{Device, Tensor};
use serde::{Deserialize, Serialize};

use super::streaming::{LayerSource, StreamingOptions};
use super::{DecoderConfig, DecoderLayer, LayerKvCache, LlamaRotaryEncoding, WeightFormat};
use crate::infer::InferError;

/// Half-open absolute layer range. Adjacent stages cover the model once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRange {
    pub first: usize,
    pub end: usize,
}

impl LayerRange {
    pub fn validate(self, layers: usize) -> Result<(), InferError> {
        if self.first >= self.end || self.end > layers {
            return Err(InferError::InvalidConfig(
                "empty or out-of-model stage range".into(),
            ));
        }
        Ok(())
    }
}

/// Available layer memory after the host reserves shared weights, KV for the
/// chosen context, activation buffers and backend overhead. Device ids must
/// be distinct authenticated personal-device identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageCapacity {
    pub device: [u8; 32],
    pub layer_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageAssignment {
    pub device: [u8; 32],
    pub range: LayerRange,
}

/// Runtime weights plus the complete context's KV payload for one layer.
/// Hosts must separately reserve shared weights, activations, allocator and
/// backend overhead. These are tensor payload counts, not process RSS limits.
pub fn layer_resident_bytes(
    config: &DecoderConfig,
    format: WeightFormat,
) -> Result<u64, InferError> {
    if config.num_attention_heads == 0 || config.kv_heads() == 0 || config.hidden_size == 0 {
        return Err(InferError::InvalidConfig(
            "invalid stage head dimensions".into(),
        ));
    }
    let h = config.hidden_size as u128;
    let kv = config.kv_heads() as u128 * config.head_dim() as u128;
    let inter = config.intermediate_size as u128;
    let mlp = if let Some(moe) = &config.moe {
        if format != WeightFormat::F32 {
            return Err(InferError::Unsupported {
                feature: "MoE stage format".into(),
                detail: "only f32 expert execution is qualified".into(),
            });
        }
        // Reserve the complete bank, and the larger dense MLP for a hybrid
        // configuration when its dense width exceeds that bank.
        let bank = h
            .saturating_mul(3)
            .saturating_mul(
                (moe.num_experts as u128)
                    .saturating_mul(moe.moe_intermediate_size as u128)
                    .saturating_add(moe.shared_expert_intermediate_size as u128),
            )
            .saturating_add(h.saturating_mul(moe.num_experts as u128 + 1));
        if (0..config.num_hidden_layers).any(|index| !config.layer_is_moe(index)) {
            bank.max(h.saturating_mul(inter).saturating_mul(3))
        } else {
            bank
        }
    } else {
        h.saturating_mul(inter).saturating_mul(3)
    };
    let projections = h
        .saturating_mul(config.query_width() as u128)
        .saturating_mul(2)
        .saturating_add(h.saturating_mul(kv).saturating_mul(2))
        .saturating_add(mlp);
    let scalar = if format == WeightFormat::F32 { 4 } else { 2 };
    let extra = (if config.qkv_bias() {
        config.query_width() as u128 + 2 * kv
    } else {
        0
    }) + (if config.output_bias() { h } else { 0 })
        + (if config.qk_norm() {
            2 * config.head_dim() as u128
        } else {
            0
        });
    let weights = match format {
        WeightFormat::Q8Block32 => projections
            .saturating_add(projections / 32 * 4)
            .saturating_add((2 * h + extra) * scalar),
        WeightFormat::Q4Block32 => (projections / 2)
            .saturating_add(projections / 32 * 4)
            .saturating_add((2 * h + extra) * scalar),
        _ => projections
            .saturating_add(2 * h + extra)
            .saturating_mul(scalar),
    };
    u64::try_from(
        weights.saturating_add(
            kv.saturating_mul(config.max_position_embeddings as u128)
                .saturating_mul(2 * scalar),
        ),
    )
    .map_err(|_| InferError::InvalidConfig("stage resident payload size overflow".into()))
}

/// Assign contiguous ranges without oversubscribing a device. Costs are
/// runtime resident weight sizes, not compressed disk artifact sizes.
pub fn assign_ranges(
    costs: &[u64],
    devices: &[StageCapacity],
) -> Result<Vec<StageAssignment>, InferError> {
    if costs.is_empty()
        || costs.contains(&0)
        || devices.is_empty()
        || devices
            .iter()
            .enumerate()
            .any(|(i, d)| devices[..i].iter().any(|p| p.device == d.device))
    {
        return Err(InferError::InvalidConfig(
            "invalid stage costs or duplicate devices".into(),
        ));
    }
    let mut first = 0;
    let mut ranges = Vec::new();
    for device in devices {
        let mut end = first;
        let mut left = device.layer_bytes;
        while end < costs.len() && costs[end] <= left {
            left -= costs[end];
            end += 1;
        }
        if end > first {
            ranges.push(StageAssignment {
                device: device.device,
                range: LayerRange { first, end },
            });
            first = end;
        }
        if first == costs.len() {
            return Ok(ranges);
        }
    }
    Err(InferError::InvalidConfig(
        "available stage memory cannot hold all layers".into(),
    ))
}

/// Synchronous actor-thread seam. A host transports bounded hidden states;
/// every fresh generation/resume starts at zero and rebuilds all stage KV.
pub trait StageExecutor: std::fmt::Debug + Send + Sync {
    /// Hosts supplying remote stages must bind their complete chain to the
    /// exact session being loaded. Bare decoder stages have no session owner.
    fn session_id(&self) -> Option<&str> {
        None
    }
    /// Complete model chains return last-position f32 logits. Individual
    /// layer stages used below the host protocol return hidden states.
    fn returns_logits(&self) -> bool {
        false
    }
    fn forward(&self, hidden: Tensor<3>, start: usize) -> Result<Tensor<3>, InferError>;
}

/// Tensor type exposed to host stage adapters without a separate Burn edge.
pub type StageHidden = Tensor<3>;

/// Lossless f32 wire values (half values widen exactly), little-endian, with
/// u64 sequence and width first. This v1 deliberately spends extra bytes for
/// a common transport representation; it never requantizes hidden states.
pub fn encode_hidden(hidden: Tensor<3>, limit: usize) -> Result<Vec<u8>, InferError> {
    let [batch, seq, width] = hidden.dims();
    let bytes = seq
        .checked_mul(width)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(16));
    if batch != 1 || seq == 0 || width == 0 || bytes.is_none_or(|n| n > limit) {
        return Err(InferError::InvalidRequest(
            "hidden wire budget or shape".into(),
        ));
    }
    let values = hidden
        .cast(burn::tensor::DType::F32)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| InferError::Backend(format!("stage readback: {e}")))?;
    let mut out = Vec::with_capacity(bytes.unwrap());
    out.extend_from_slice(&(seq as u64).to_le_bytes());
    out.extend_from_slice(&(width as u64).to_le_bytes());
    for value in values {
        if !value.is_finite() {
            return Err(InferError::Backend("nonfinite stage hidden".into()));
        }
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

pub fn decode_hidden(
    bytes: &[u8],
    width: usize,
    max_seq: usize,
    format: WeightFormat,
    device: &Device,
) -> Result<Tensor<3>, InferError> {
    if bytes.len() < 16 {
        return Err(InferError::InvalidRequest("truncated hidden header".into()));
    }
    let seq =
        usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap_or(usize::MAX);
    let wire_width = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    if seq == 0
        || seq > max_seq
        || wire_width != width as u64
        || seq
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(16))
            != Some(bytes.len())
    {
        return Err(InferError::InvalidRequest(
            "hidden wire dimensions or length".into(),
        ));
    }
    let values: Vec<f32> = bytes[16..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(InferError::InvalidRequest("nonfinite hidden values".into()));
    }
    if values.iter().any(|&value| match format {
        WeightFormat::F32 => false,
        WeightFormat::Bf16 => half::bf16::from_f32(value).to_f32() != value,
        _ => half::f16::from_f32(value).to_f32() != value,
    }) {
        return Err(InferError::InvalidRequest(
            "hidden values are not exact in the declared activation dtype".into(),
        ));
    }
    Ok(Tensor::<3>::from_data(
        burn::tensor::TensorData::new(values, [1, seq, width]),
        device,
    )
    .cast(format.activation_dtype()))
}

struct Cache {
    layers: Vec<LayerKvCache>,
    position: usize,
    failed: bool,
}

/// Resident weights for exactly one contiguous layer range. No embeddings,
/// tokenizer, full-model source or durable persona data. The final range may
/// additionally own its normalization and output head.
pub struct DecoderStage {
    layers: Vec<DecoderLayer>,
    rope: LlamaRotaryEncoding,
    config: DecoderConfig,
    range: LayerRange,
    format: WeightFormat,
    options: StreamingOptions,
    device: Device,
    cache: Mutex<Cache>,
    output: Option<(burn::nn::RmsNorm, super::model::OutputHead)>,
}

impl std::fmt::Debug for DecoderStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderStage")
            .field("range", &self.range)
            .finish_non_exhaustive()
    }
}

impl DecoderStage {
    pub fn from_source(
        config: DecoderConfig,
        range: LayerRange,
        source: &dyn LayerSource,
        options: StreamingOptions,
        format: WeightFormat,
        device: &Device,
    ) -> Result<Self, InferError> {
        range.validate(config.num_hidden_layers)?;
        options.validate()?;
        format.validate_device(device)?;
        if cfg!(target_family = "wasm") {
            return Err(InferError::Unsupported {
                feature: "native layer stages".into(),
                detail: "stage draining requires synchronous device execution".into(),
            });
        }
        let mut layers = Vec::with_capacity(range.end - range.first);
        for index in range.first..range.end {
            let bytes = source.read_layer(index, options.max_layer_bytes)?;
            if bytes.len() > options.max_layer_bytes {
                return Err(InferError::InvalidWeights(
                    "stage layer exceeds artifact budget".into(),
                ));
            }
            let tensors = safetensors::SafeTensors::deserialize(&bytes)
                .map_err(|e| InferError::InvalidWeights(e.to_string()))?;
            super::loader::validate_layer_names(&config, &tensors, index)?;
            let loaded = super::loader::load_decoder_layer_with_mode(
                &config,
                &tensors,
                index,
                device,
                format,
                super::loader::ExpertMode::Resident(source, options.max_layer_bytes),
            )?;
            layers.push(DecoderLayer::from_loaded(&config, loaded, device));
        }
        Self::from_layers(config, range, layers, options, format, device)
    }

    fn from_layers(
        config: DecoderConfig,
        range: LayerRange,
        layers: Vec<DecoderLayer>,
        options: StreamingOptions,
        format: WeightFormat,
        device: &Device,
    ) -> Result<Self, InferError> {
        range.validate(config.num_hidden_layers)?;
        if layers.len() != range.end - range.first {
            return Err(InferError::InvalidWeights(
                "stage layer count disagrees with range".into(),
            ));
        }
        let rope = LlamaRotaryEncoding::for_config(&config, device);
        Ok(Self {
            cache: Mutex::new(Cache {
                layers: (0..layers.len()).map(|_| LayerKvCache::default()).collect(),
                position: 0,
                failed: false,
            }),
            layers,
            rope,
            config,
            range,
            format,
            options,
            device: device.clone(),
            output: None,
        })
    }

    pub fn range(&self) -> LayerRange {
        self.range
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn format(&self) -> WeightFormat {
        self.format
    }

    /// The final range owns normalization and the output projection. Tied
    /// heads reuse the embedding's tensor; an untied embedding is dropped.
    pub fn with_output(mut self, shared: super::model::LoadedDecoder) -> Result<Self, InferError> {
        if self.range.end != self.config.num_hidden_layers || !shared.layers.is_empty() {
            return Err(InferError::InvalidWeights(
                "only the final stage may own the output head".into(),
            ));
        }
        let head = shared
            .lm_head_w
            .unwrap_or_else(|| shared.embed_w.transpose());
        self.output = Some((
            super::layer::rms_norm_from_loaded(
                shared.final_norm_gamma,
                self.config.rms_norm_eps,
                &self.device,
            ),
            super::model::OutputHead(vec![super::attention::linear_no_bias_from_loaded(
                head,
                &self.device,
            )]),
        ));
        Ok(self)
    }

    pub(crate) fn with_output_shards(
        mut self,
        shared: super::model::LoadedSharedShards,
    ) -> Result<Self, InferError> {
        if self.format != WeightFormat::F32 {
            return Err(InferError::Unsupported {
                feature: "shared-row stage format".into(),
                detail: "shared-row shards are qualified for f32 execution".into(),
            });
        }
        if self.range.end != self.config.num_hidden_layers {
            return Err(InferError::InvalidWeights(
                "only the final stage may own the output head".into(),
            ));
        }
        let head = shared.head.unwrap_or_else(|| {
            shared
                .embed
                .into_iter()
                .map(|weight| weight.transpose())
                .collect()
        });
        self.output = Some((
            super::layer::rms_norm_from_loaded(shared.norm, self.config.rms_norm_eps, &self.device),
            super::model::OutputHead(
                head.into_iter()
                    .map(|weight| {
                        super::attention::linear_no_bias_from_loaded(weight, &self.device)
                    })
                    .collect(),
            ),
        ));
        Ok(self)
    }

    pub fn forward_bytes(
        &self,
        bytes: &[u8],
        start: usize,
        limit: usize,
    ) -> Result<Vec<u8>, InferError> {
        if bytes.len() > limit {
            return Err(InferError::InvalidRequest("hidden wire budget".into()));
        }
        let hidden = decode_hidden(
            bytes,
            self.config.hidden_size,
            self.config.max_position_embeddings,
            self.format,
            &self.device,
        )?;
        encode_hidden(self.forward(hidden, start)?, limit)
    }

    /// Synchronize before the lease releases its hardware guard.
    pub fn drain(&self) -> Result<(), InferError> {
        self.device
            .sync()
            .map_err(|e| InferError::Backend(format!("drain stage: {e}")))
    }
}

impl StageExecutor for DecoderStage {
    fn forward(&self, mut hidden: Tensor<3>, start: usize) -> Result<Tensor<3>, InferError> {
        let [batch, seq, width] = hidden.dims();
        if batch != 1
            || seq == 0
            || width != self.config.hidden_size
            || hidden.dtype() != self.format.activation_dtype()
            || start
                .checked_add(seq)
                .is_none_or(|end| end > self.config.max_position_embeddings)
        {
            return Err(InferError::InvalidRequest(
                "stage hidden shape, dtype or context".into(),
            ));
        }
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| InferError::Backend("stage cache poisoned".into()))?;
        if start == 0 {
            cache.layers = (0..self.layers.len())
                .map(|_| LayerKvCache::default())
                .collect();
            cache.position = 0;
            cache.failed = false;
        }
        if cache.failed || cache.position != start || (start != 0 && seq != 1) {
            return Err(InferError::InvalidRequest(
                "stage KV position mismatch; restart from checkpoint".into(),
            ));
        }
        // Any partial layer failure invalidates this cache until a full prefill.
        cache.failed = true;
        for (layer, kv) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward_cached_chunked(
                hidden,
                &self.rope,
                kv,
                start,
                self.options.query_chunk,
            )?;
        }
        if let Some((norm, head)) = &self.output {
            let [batch, seq, width] = hidden.dims();
            hidden = head
                .forward(norm.forward(hidden.slice([0..batch, seq - 1..seq, 0..width])))
                .cast(burn::tensor::DType::F32);
        }
        self.drain()?;
        cache.position += seq;
        cache.failed = false;
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{model::tests::det_loaded, test_support::tiny_config};
    use super::*;

    #[test]
    fn disjoint_stages_match_full_layers_and_enforce_positions() {
        let config = tiny_config();
        let device = Device::ndarray();
        let loaded = det_loaded(&config, &device);
        let rope = LlamaRotaryEncoding::new(
            config.max_position_embeddings,
            config.head_dim(),
            config.rope_theta,
            &device,
        );
        let layers: Vec<_> = loaded
            .layers
            .into_iter()
            .map(|l| DecoderLayer::from_loaded(&config, l, &device))
            .collect();
        let mut full_cache = [LayerKvCache::default(), LayerKvCache::default()];
        let mut separate = det_loaded(&config, &device).layers.into_iter();
        let first = DecoderStage::from_layers(
            config.clone(),
            LayerRange { first: 0, end: 1 },
            vec![DecoderLayer::from_loaded(
                &config,
                separate.next().unwrap(),
                &device,
            )],
            StreamingOptions::default(),
            WeightFormat::F32,
            &device,
        )
        .unwrap();
        let last = DecoderStage::from_layers(
            config.clone(),
            LayerRange { first: 1, end: 2 },
            vec![DecoderLayer::from_loaded(
                &config,
                separate.next().unwrap(),
                &device,
            )],
            StreamingOptions::default(),
            WeightFormat::F32,
            &device,
        )
        .unwrap();
        for (start, seq) in [(0, 3), (3, 1), (4, 1)] {
            let input = Tensor::<3>::ones([1, seq, config.hidden_size], &device);
            let mut expected = input.clone();
            for (layer, kv) in layers.iter().zip(&mut full_cache) {
                expected = layer.forward_cached(expected, &rope, kv, start);
            }
            let actual = last
                .forward(first.forward(input, start).unwrap(), start)
                .unwrap();
            assert_eq!(actual.into_data(), expected.into_data());
        }
        assert!(
            first
                .forward(Tensor::ones([1, 1, config.hidden_size], &device), 4)
                .is_err()
        );
        assert!(
            first
                .forward(Tensor::ones([1, 3, config.hidden_size], &device), 0)
                .is_ok()
        );
    }

    #[test]
    fn ranges_honor_runtime_budgets_and_distinct_devices() {
        let devices = [
            StageCapacity {
                device: [1; 32],
                layer_bytes: 21,
            },
            StageCapacity {
                device: [2; 32],
                layer_bytes: 30,
            },
        ];
        assert_eq!(
            assign_ranges(&[10, 11, 12, 13], &devices).unwrap(),
            vec![
                StageAssignment {
                    device: [1; 32],
                    range: LayerRange { first: 0, end: 2 }
                },
                StageAssignment {
                    device: [2; 32],
                    range: LayerRange { first: 2, end: 4 }
                }
            ]
        );
        assert!(assign_ranges(&[22, 31], &devices).is_err());
        assert!(assign_ranges(&[1], &[devices[0].clone(), devices[0].clone()]).is_err());
        assert_eq!(
            layer_resident_bytes(&tiny_config(), WeightFormat::F32).unwrap(),
            2880
        );
        let devices = [
            StageCapacity {
                device: [1; 32],
                layer_bytes: 1,
            },
            StageCapacity {
                device: [2; 32],
                layer_bytes: 20,
            },
        ];
        assert_eq!(
            assign_ranges(&[10, 10], &devices).unwrap()[0].device,
            [2; 32]
        );
    }

    #[test]
    fn hidden_wire_refuses_overflow_nonfinite_and_dtype_changes() {
        let device = Device::ndarray();
        let mut bytes = encode_hidden(Tensor::ones([1, 3, 8], &device), 112).unwrap();
        assert_eq!(
            decode_hidden(&bytes, 8, 16, WeightFormat::F32, &device)
                .unwrap()
                .dims(),
            [1, 3, 8]
        );
        assert!(encode_hidden(Tensor::ones([1, 3, 8], &device), 111).is_err());
        assert!(decode_hidden(&bytes, 4, 16, WeightFormat::F32, &device).is_err());
        assert!(decode_hidden(&bytes, 8, 2, WeightFormat::F32, &device).is_err());
        bytes[16..20].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_hidden(&bytes, 8, 16, WeightFormat::F32, &device).is_err());
        bytes[16..20].copy_from_slice(&0.1234567f32.to_le_bytes());
        assert!(decode_hidden(&bytes, 8, 16, WeightFormat::F16, &device).is_err());
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode_hidden(&bytes, 8, usize::MAX, WeightFormat::F32, &device).is_err());
    }
}
#[test]
fn hybrid_admission_covers_a_dense_mlp_larger_than_the_expert_bank() {
    let mut dense = tiny_config();
    dense.intermediate_size = 4096;
    let baseline = layer_resident_bytes(&dense, WeightFormat::F32).unwrap();
    let mut hybrid = dense;
    hybrid.moe = Some(super::super::config::MoeConfig {
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 1,
        shared_expert_intermediate_size: 1,
        norm_topk_prob: false,
        decoder_sparse_step: 2,
        mlp_only_layers: vec![],
    });
    assert!(layer_resident_bytes(&hybrid, WeightFormat::F32).unwrap() >= baseline);
}
