// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Qwen2 routed and shared experts. A streamed forward reads only experts
//! selected by this query block and drains each before releasing its weights.
use super::attention::linear_no_bias_from_loaded;
use super::{
    WeightFormat,
    config::{DecoderConfig, MoeConfig},
    streaming::{ExpertReads, LayerSource},
};
use crate::infer::InferError;
use burn::{
    nn::{Linear, SwiGlu},
    tensor::{
        Device, Int, Tensor, TensorData,
        activation::{sigmoid, softmax},
    },
};
use std::sync::{Arc, atomic::Ordering};

#[derive(Clone)]
pub struct MlpWeights {
    pub gate_w: Tensor<2>,
    pub up_w: Tensor<2>,
    pub down_w: Tensor<2>,
}
impl MlpWeights {
    pub(crate) fn detach(&mut self) {
        self.gate_w = self.gate_w.clone().detach();
        self.up_w = self.up_w.clone().detach();
        self.down_w = self.down_w.clone().detach();
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct ExpertStreaming {
    pub(crate) source: Arc<dyn LayerSource>,
    pub(crate) reads: Arc<ExpertReads>,
    pub(crate) layer: usize,
    pub(crate) limit: usize,
    pub(crate) format: WeightFormat,
}
impl std::fmt::Debug for ExpertStreaming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertStreaming")
            .field("layer", &self.layer)
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub enum LoadedExperts {
    Resident(Vec<MlpWeights>),
    #[doc(hidden)]
    Streamed(ExpertStreaming),
}

#[derive(Clone)]
pub enum LoadedMlp {
    Dense(MlpWeights),
    Moe {
        router: Tensor<2>,
        shared: MlpWeights,
        shared_gate: Tensor<2>,
        experts: LoadedExperts,
    },
}

#[derive(Debug)]
pub(crate) struct Expert {
    gated: SwiGlu,
    down: Linear,
}
impl Expert {
    fn new(weights: MlpWeights, device: &Device) -> Self {
        Self {
            gated: super::layer::swiglu_from_loaded(weights.gate_w, weights.up_w, device),
            down: linear_no_bias_from_loaded(weights.down_w, device),
        }
    }
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.down.forward(self.gated.forward(x))
    }
}

#[derive(Debug)]
pub(crate) enum Experts {
    Resident(Vec<Expert>),
    Streamed(ExpertStreaming),
}
#[derive(Debug)]
pub(crate) enum DecoderMlp {
    Dense(Expert),
    Moe {
        config: MoeConfig,
        router: Linear,
        shared: Expert,
        shared_gate: Linear,
        experts: Experts,
    },
}
impl DecoderMlp {
    pub fn new(loaded: LoadedMlp, config: &DecoderConfig, device: &Device) -> Self {
        match loaded {
            LoadedMlp::Dense(weights) => Self::Dense(Expert::new(weights, device)),
            LoadedMlp::Moe {
                router,
                shared,
                shared_gate,
                experts,
            } => Self::Moe {
                config: config.moe.clone().expect("declared MoE configuration"),
                router: linear_no_bias_from_loaded(router, device),
                shared: Expert::new(shared, device),
                shared_gate: linear_no_bias_from_loaded(shared_gate, device),
                experts: match experts {
                    LoadedExperts::Resident(weights) => Experts::Resident(
                        weights
                            .into_iter()
                            .map(|w| Expert::new(w, device))
                            .collect(),
                    ),
                    LoadedExperts::Streamed(source) => Experts::Streamed(source),
                },
            },
        }
    }

    pub fn forward(&self, x: Tensor<3>, decoder: &DecoderConfig) -> Result<Tensor<3>, InferError> {
        let Self::Moe {
            config,
            router,
            shared,
            shared_gate,
            experts,
        } = self
        else {
            let Self::Dense(expert) = self else {
                unreachable!()
            };
            return Ok(expert.forward(x));
        };
        let [batch, sequence, hidden] = x.dims();
        let count = batch * sequence;
        let dtype = x.dtype();
        let device = x.device();
        let x = x.reshape([count, hidden]);
        // Routing is fp32 even when activations are half precision. The CPU
        // selection is transient private material; nothing is logged or stored.
        let probabilities = softmax(router.forward(x.clone()).cast(burn::tensor::DType::F32), 1)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| InferError::Backend(format!("router read: {e}")))?;
        let mut selected = vec![Vec::<(i32, f32)>::new(); config.num_experts];
        for (token, row) in probabilities.chunks_exact(config.num_experts).enumerate() {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(InferError::Backend("non-finite expert router".into()));
            }
            let mut order: Vec<_> = (0..config.num_experts).collect();
            order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then_with(|| a.cmp(b)));
            let sum = if config.norm_topk_prob {
                order[..config.num_experts_per_tok]
                    .iter()
                    .map(|i| row[*i])
                    .sum()
            } else {
                1.0
            };
            if !sum.is_finite() || sum <= 0.0 {
                return Err(InferError::Backend("invalid top-k routing mass".into()));
            }
            for &expert in &order[..config.num_experts_per_tok] {
                selected[expert].push((token as i32, row[expert] / sum));
            }
        }
        let mut output = Tensor::<2>::zeros([count, hidden], &device).cast(dtype);
        for (index, tokens) in selected
            .iter()
            .enumerate()
            .filter(|(_, tokens)| !tokens.is_empty())
        {
            let ids = Tensor::<1, Int>::from_data(
                TensorData::new(
                    tokens.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                    [tokens.len()],
                ),
                &device,
            );
            let weights = Tensor::<2>::from_data(
                TensorData::new(
                    tokens.iter().map(|(_, weight)| *weight).collect::<Vec<_>>(),
                    [tokens.len(), 1],
                ),
                &device,
            )
            .cast(dtype);
            let input = x.clone().select(0, ids.clone());
            let evaluate = |expert: &Expert| expert.forward(input.clone()) * weights.clone();
            let delta = match experts {
                Experts::Resident(experts) => evaluate(&experts[index]),
                Experts::Streamed(stream) => {
                    let bytes = stream
                        .source
                        .read_expert(stream.layer, index, stream.limit)?;
                    if bytes.len() > stream.limit {
                        return Err(InferError::InvalidWeights(
                            "expert source exceeded artifact budget".into(),
                        ));
                    }
                    stream.reads.count.fetch_add(1, Ordering::Relaxed);
                    stream
                        .reads
                        .bytes
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                    stream
                        .reads
                        .largest
                        .fetch_max(bytes.len() as u64, Ordering::Relaxed);
                    let tensors = safetensors::SafeTensors::deserialize(&bytes)
                        .map_err(|e| InferError::InvalidWeights(e.to_string()))?;
                    let weights = super::loader::load_expert(
                        decoder,
                        &tensors,
                        stream.layer,
                        index,
                        &device,
                        stream.format,
                    )?;
                    let expert = Expert::new(weights, &device);
                    let delta = evaluate(&expert);
                    // Execute while this expert is held. The next selected
                    // expert cannot retain this one's weights through laziness.
                    device
                        .sync()
                        .map_err(|e| InferError::Backend(format!("drain streamed expert: {e}")))?;
                    delta
                },
            };
            output = output.select_assign(0, ids, delta, burn::tensor::IndexingUpdateOp::Add);
            if matches!(experts, Experts::Streamed(_)) {
                device
                    .sync()
                    .map_err(|e| InferError::Backend(format!("drain expert accumulation: {e}")))?;
            }
        }
        Ok(
            (output + sigmoid(shared_gate.forward(x.clone())) * shared.forward(x))
                .reshape([batch, sequence, hidden]),
        )
    }
}
