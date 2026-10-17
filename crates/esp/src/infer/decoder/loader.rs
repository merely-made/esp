// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! HF safetensors → [`DecoderModel`].
//!
//! Maps the llama-family checkpoint names (`model.embed_tokens.weight`,
//! `model.layers.{i}.self_attn.q_proj.weight`, …) onto the decoder's
//! loaded-tensor bundles, transposing every linear from HF `[out, in]`
//! to Burn `[in, out]` at this boundary. Byte-buffer in, model out — the
//! eidetic corridor's `ModelComponents.weight_bytes` feeds this directly
//! (P2), and a file path is just `std::fs::read` above it.

use std::collections::HashSet;

use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use burn::tensor::Device;

use super::config::DecoderConfig;
use super::layer::LoadedDecoderLayer;
use super::model::{DecoderModel, LoadedDecoder};
use crate::infer::provider::InferError;

fn view<'a>(tensors: &'a SafeTensors<'_>, name: &str) -> Result<TensorView<'a>, InferError> {
    tensors
        .tensor(name)
        .map_err(|_| InferError::InvalidWeights(format!("missing tensor: {name}")))
}

/// The per-layer tensors the Llama block consumes, under
/// `model.layers.{i}.`; the loader below reads exactly these.
pub(crate) const LAYER_TENSORS: [&str; 9] = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

pub(crate) fn layer_tensor_names(config: &DecoderConfig) -> Vec<&'static str> {
    let mut names = LAYER_TENSORS.to_vec();
    if config.qkv_bias() {
        names.extend([
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]);
    }
    if config.output_bias() {
        names.push("self_attn.o_proj.bias");
    }
    if config.qk_norm() {
        names.extend(["self_attn.q_norm.weight", "self_attn.k_norm.weight"]);
    }
    names
}

pub(crate) fn layer_core_tensor_names(config: &DecoderConfig, index: usize) -> Vec<String> {
    let mut names: Vec<_> = layer_tensor_names(config)
        .into_iter()
        .map(str::to_string)
        .collect();
    if config.layer_is_moe(index) {
        names.retain(|name| !name.starts_with("mlp."));
        names.extend(
            [
                "mlp.gate.weight",
                "mlp.shared_expert_gate.weight",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert.down_proj.weight",
            ]
            .map(str::to_string),
        );
    }
    names
}

pub(crate) fn expert_tensor_names(layer: usize, expert: usize) -> Vec<String> {
    ["gate_proj", "up_proj", "down_proj"]
        .map(|name| format!("model.layers.{layer}.mlp.experts.{expert}.{name}.weight"))
        .into()
}

pub(crate) fn layer_all_tensor_names(config: &DecoderConfig, index: usize) -> Vec<String> {
    let mut names = layer_core_tensor_names(config, index);
    if config.layer_is_moe(index) {
        for expert in 0..config.moe.as_ref().unwrap().num_experts {
            names.extend(expert_tensor_names(index, expert).into_iter().map(|name| {
                name.strip_prefix(&format!("model.layers.{index}."))
                    .unwrap()
                    .to_string()
            }));
        }
    }
    names
}

pub(crate) enum ExpertMode<'a> {
    InTensors,
    Resident(&'a dyn super::streaming::LayerSource, usize),
    Streamed(super::moe::ExpertStreaming),
}

fn mlp_weights(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    h: usize,
    inter: usize,
    device: &Device,
    format: super::WeightFormat,
) -> Result<super::moe::MlpWeights, InferError> {
    let read = |name: &str, shape| {
        super::weight_format::extract(
            &view(tensors, &format!("{prefix}.{name}.weight"))?,
            shape,
            device,
            format,
            true,
        )
    };
    Ok(super::moe::MlpWeights {
        gate_w: read("gate_proj", [inter, h])?,
        up_w: read("up_proj", [inter, h])?,
        down_w: read("down_proj", [h, inter])?,
    })
}

pub(crate) fn load_expert(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    layer: usize,
    expert: usize,
    device: &Device,
    format: super::WeightFormat,
) -> Result<super::moe::MlpWeights, InferError> {
    let moe = config
        .moe
        .as_ref()
        .ok_or_else(|| InferError::InvalidConfig("expert read for a dense model".into()))?;
    if !config.layer_is_moe(layer)
        || expert >= moe.num_experts
        || tensors.names().into_iter().cloned().collect::<HashSet<_>>()
            != expert_tensor_names(layer, expert)
                .into_iter()
                .collect::<HashSet<_>>()
    {
        return Err(InferError::InvalidWeights(
            "expert artifact names disagree with layer/expert selection".into(),
        ));
    }
    mlp_weights(
        tensors,
        &format!("model.layers.{layer}.mlp.experts.{expert}"),
        config.hidden_size,
        moe.moe_intermediate_size,
        device,
        format,
    )
}

fn load_mlp(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    i: usize,
    device: &Device,
    format: super::WeightFormat,
    mode: ExpertMode<'_>,
) -> Result<super::moe::LoadedMlp, InferError> {
    use super::moe::{LoadedExperts, LoadedMlp};
    let prefix = format!("model.layers.{i}.mlp");
    let h = config.hidden_size;
    if !config.layer_is_moe(i) {
        return Ok(LoadedMlp::Dense(mlp_weights(
            tensors,
            &prefix,
            h,
            config.intermediate_size,
            device,
            format,
        )?));
    }
    if format != super::WeightFormat::F32 {
        return Err(InferError::Unsupported { feature:"MoE weight format".into(),detail:"routed expert formats require separate qualification; only f32 execution is admitted".into() });
    }
    let moe = config.moe.as_ref().unwrap();
    let experts = match mode {
        ExpertMode::Streamed(source) => LoadedExperts::Streamed(source),
        mode => {
            let mut experts = Vec::with_capacity(moe.num_experts);
            for expert in 0..moe.num_experts {
                experts.push(match mode {
                    ExpertMode::Resident(source, limit) => {
                        let bytes = source.read_expert(i, expert, limit)?;
                        if bytes.len() > limit {
                            return Err(InferError::InvalidWeights(
                                "resident expert source exceeded its budget".into(),
                            ));
                        }
                        let tensors = SafeTensors::deserialize(&bytes)
                            .map_err(|e| InferError::InvalidWeights(e.to_string()))?;
                        load_expert(config, &tensors, i, expert, device, format)?
                    },
                    ExpertMode::InTensors => mlp_weights(
                        tensors,
                        &format!("{prefix}.experts.{expert}"),
                        h,
                        moe.moe_intermediate_size,
                        device,
                        format,
                    )?,
                    ExpertMode::Streamed(_) => unreachable!(),
                });
            }
            LoadedExperts::Resident(experts)
        },
    };
    let matrix = |name: &str, rows| {
        super::weight_format::extract(
            &view(tensors, &format!("{prefix}.{name}.weight"))?,
            [rows, h],
            device,
            format,
            true,
        )
    };
    Ok(LoadedMlp::Moe {
        router: matrix("gate", moe.num_experts)?,
        shared: mlp_weights(
            tensors,
            &format!("{prefix}.shared_expert"),
            h,
            moe.shared_expert_intermediate_size,
            device,
            format,
        )?,
        shared_gate: matrix("shared_expert_gate", 1)?,
        experts,
    })
}

/// Older checkpoints store RoPE's inverse frequencies as a buffer. They are
/// derived from `rope_theta`, which the config already supplies, and a
/// scaled variant is refused at the config.
fn ignorable(name: &str) -> bool {
    name.ends_with(".rotary_emb.inv_freq")
}

fn why_unconsumed(name: &str) -> &'static str {
    if name.ends_with(".bias") {
        "this bias is not declared by the decoder configuration"
    } else if name.contains("q_norm") || name.contains("k_norm") {
        "this head norm is not declared by the decoder configuration"
    } else if name.contains("experts")
        || name.contains("block_sparse_moe")
        || name.ends_with(".mlp.gate.weight")
    {
        "this expert or router is not declared by the decoder configuration"
    } else if name.contains("norm") {
        "normalization layers beyond the Llama block's two per layer are not implemented"
    } else {
        "the Llama-block decoder does not consume this tensor"
    }
}

/// Refuse a checkpoint carrying tensors the decoder would not read. An
/// unread bias, norm or expert weight means the checkpoint computes
/// something this decoder does not, so loading it would answer wrongly.
fn refuse_unconsumed(config: &DecoderConfig, tensors: &SafeTensors<'_>) -> Result<(), InferError> {
    let mut consumed: HashSet<String> = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    for i in 0..config.num_hidden_layers {
        for suffix in layer_all_tensor_names(config, i) {
            consumed.insert(format!("model.layers.{i}.{suffix}"));
        }
    }
    let mut unconsumed: Vec<&String> = tensors
        .names()
        .into_iter()
        .filter(|name| !consumed.contains(name.as_str()) && !ignorable(name))
        .collect();
    unconsumed.sort();
    match unconsumed.first() {
        None => Ok(()),
        Some(first) => Err(InferError::Unsupported {
            feature: format!("tensor {first}"),
            detail: format!(
                "{}; {} unconsumed tensor(s) in this checkpoint",
                why_unconsumed(first),
                unconsumed.len()
            ),
        }),
    }
}

/// Decode HF `model.safetensors` bytes into resident decoder tensors.
///
/// `lm_head.weight` may be absent when `config.tie_word_embeddings` is
/// set (Llama 3.2 1B, SmolLM); a checkpoint that omits it *without* the
/// config flag is rejected rather than silently tied.
pub(crate) fn load_decoder_tensors_from_bytes(
    config: &DecoderConfig,
    weights: &[u8],
    device: &Device,
) -> Result<LoadedDecoder, InferError> {
    let tensors = SafeTensors::deserialize(weights)
        .map_err(|e| InferError::InvalidWeights(format!("safetensors parse: {e}")))?;
    refuse_unconsumed(config, &tensors)?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        layers.push(load_decoder_layer(config, &tensors, i, device)?);
    }

    let mut shared = load_decoder_shared_tensors(config, &tensors, device)?;
    shared.layers = layers;
    Ok(shared)
}

pub(crate) fn load_decoder_shared_tensors(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    device: &Device,
) -> Result<LoadedDecoder, InferError> {
    load_decoder_shared_tensors_with_format(
        config,
        tensors,
        device,
        super::weight_format::WeightFormat::F32,
    )
}

pub(crate) fn load_shared_shards(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    device: &Device,
    max_matrix_bytes: usize,
) -> Result<super::model::LoadedSharedShards, InferError> {
    if tensors.names().iter().any(|name| {
        !matches!(
            name.as_str(),
            "model.embed_tokens.weight" | "model.norm.weight" | "lm_head.weight"
        )
    }) {
        return Err(InferError::InvalidWeights(
            "unexpected tensor in shared-row artifact".into(),
        ));
    }
    let h = config.hidden_size;
    let row_bytes = h
        .checked_mul(4)
        .ok_or_else(|| InferError::InvalidWeights("shared row byte overflow".into()))?;
    let rows = max_matrix_bytes / row_bytes;
    if rows == 0 {
        return Err(InferError::InvalidWeights(
            "shared tensor bound cannot hold one f32 row".into(),
        ));
    }
    let read = |name: &str, transpose: bool| -> Result<Vec<burn::tensor::Tensor<2>>, InferError> {
        let tensor = view(tensors, name)?;
        if tensor.shape() != [config.vocab_size, h] {
            return Err(InferError::InvalidWeights(format!(
                "shared matrix shape {name}"
            )));
        }
        let encoded_row = h
            .checked_mul(tensor.dtype().size())
            .ok_or_else(|| InferError::InvalidWeights("encoded row byte overflow".into()))?;
        (0..config.vocab_size)
            .step_by(rows)
            .map(|first| {
                let end = first.saturating_add(rows).min(config.vocab_size);
                let part = TensorView::new(
                    tensor.dtype(),
                    vec![end - first, h],
                    &tensor.data()[first * encoded_row..end * encoded_row],
                )
                .map_err(|e| InferError::InvalidWeights(e.to_string()))?;
                super::weight_format::extract(
                    &part,
                    [end - first, h],
                    device,
                    super::WeightFormat::F32,
                    transpose,
                )
            })
            .collect()
    };
    let head = match tensors.tensor("lm_head.weight") {
        Ok(_) => Some(read("lm_head.weight", true)?),
        Err(_) if config.tie_word_embeddings => None,
        Err(_) => {
            return Err(InferError::InvalidWeights(
                "untied output head missing".into(),
            ));
        },
    };
    Ok(super::model::LoadedSharedShards {
        embed: read("model.embed_tokens.weight", false)?,
        norm: super::weight_format::extract(
            &view(tensors, "model.norm.weight")?,
            [h],
            device,
            super::WeightFormat::F32,
            false,
        )?,
        head,
    })
}

pub(crate) fn load_decoder_shared_tensors_with_format(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    device: &Device,
    format: super::weight_format::WeightFormat,
) -> Result<LoadedDecoder, InferError> {
    let extract_1d = |v: &TensorView<'_>, len, device: &Device| {
        super::weight_format::extract(v, [len], device, format, false)
    };
    let extract_2d = |v: &TensorView<'_>, rows, cols, device: &Device| {
        super::weight_format::extract(v, [rows, cols], device, format, false)
    };
    let extract_2d_transposed = |v: &TensorView<'_>, rows, cols, device: &Device| {
        super::weight_format::extract(v, [rows, cols], device, format, true)
    };
    let h = config.hidden_size;
    let lm_head_w = match tensors.tensor("lm_head.weight") {
        Ok(v) => Some(extract_2d_transposed(&v, config.vocab_size, h, device)?),
        Err(_) if config.tie_word_embeddings => None,
        Err(_) => {
            return Err(InferError::InvalidWeights(
                "lm_head.weight missing but tie_word_embeddings is false".to_string(),
            ));
        },
    };

    Ok(LoadedDecoder {
        embed_w: extract_2d(
            &view(&tensors, "model.embed_tokens.weight")?,
            config.vocab_size,
            h,
            device,
        )?,
        layers: Vec::new(),
        final_norm_gamma: extract_1d(&view(&tensors, "model.norm.weight")?, h, device)?,
        lm_head_w,
    })
}

/// Decode one layer without constructing tensors for any other layer.
pub(crate) fn load_decoder_layer(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    i: usize,
    device: &Device,
) -> Result<LoadedDecoderLayer, InferError> {
    load_decoder_layer_with_format(
        config,
        tensors,
        i,
        device,
        super::weight_format::WeightFormat::F32,
    )
}

pub(crate) fn load_decoder_layer_with_format(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    i: usize,
    device: &Device,
    format: super::weight_format::WeightFormat,
) -> Result<LoadedDecoderLayer, InferError> {
    load_decoder_layer_with_mode(config, tensors, i, device, format, ExpertMode::InTensors)
}

pub(crate) fn load_decoder_layer_with_mode(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    i: usize,
    device: &Device,
    format: super::WeightFormat,
    mode: ExpertMode<'_>,
) -> Result<LoadedDecoderLayer, InferError> {
    let extract_1d = |v: &TensorView<'_>, len, device: &Device| {
        super::weight_format::extract(v, [len], device, format, false)
    };
    let extract_2d_transposed = |v: &TensorView<'_>, rows, cols, device: &Device| {
        super::weight_format::extract(v, [rows, cols], device, format, true)
    };
    let h = config.hidden_size;
    let kv = config.kv_heads() * config.head_dim();
    let p = format!("model.layers.{i}");
    let optional = |required: bool, suffix: &str, len: usize| {
        required
            .then(|| extract_1d(&view(tensors, &format!("{p}.{suffix}"))?, len, device))
            .transpose()
    };
    Ok(LoadedDecoderLayer {
        attention: super::attention::LoadedAttentionOptions {
            q_bias: optional(
                config.qkv_bias(),
                "self_attn.q_proj.bias",
                config.query_width(),
            )?,
            k_bias: optional(config.qkv_bias(), "self_attn.k_proj.bias", kv)?,
            v_bias: optional(config.qkv_bias(), "self_attn.v_proj.bias", kv)?,
            o_bias: optional(config.output_bias(), "self_attn.o_proj.bias", h)?,
            q_norm: optional(
                config.qk_norm(),
                "self_attn.q_norm.weight",
                config.head_dim(),
            )?,
            k_norm: optional(
                config.qk_norm(),
                "self_attn.k_norm.weight",
                config.head_dim(),
            )?,
            layer_index: i,
        },
        input_norm_gamma: extract_1d(
            &view(&tensors, &format!("{p}.input_layernorm.weight"))?,
            h,
            device,
        )?,
        q_w: extract_2d_transposed(
            &view(&tensors, &format!("{p}.self_attn.q_proj.weight"))?,
            config.query_width(),
            h,
            device,
        )?,
        k_w: extract_2d_transposed(
            &view(&tensors, &format!("{p}.self_attn.k_proj.weight"))?,
            kv,
            h,
            device,
        )?,
        v_w: extract_2d_transposed(
            &view(&tensors, &format!("{p}.self_attn.v_proj.weight"))?,
            kv,
            h,
            device,
        )?,
        o_w: extract_2d_transposed(
            &view(&tensors, &format!("{p}.self_attn.o_proj.weight"))?,
            h,
            config.query_width(),
            device,
        )?,
        post_norm_gamma: extract_1d(
            &view(&tensors, &format!("{p}.post_attention_layernorm.weight"))?,
            h,
            device,
        )?,
        mlp: load_mlp(config, tensors, i, device, format, mode)?,
    })
}

/// Build a [`DecoderModel`] from HF `model.safetensors` bytes.
pub fn load_decoder_from_bytes(
    config: &DecoderConfig,
    weights: &[u8],
    device: &Device,
) -> Result<DecoderModel, InferError> {
    let loaded = load_decoder_tensors_from_bytes(config, weights, device)?;
    Ok(DecoderModel::from_loaded(config.clone(), loaded, device))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::test_support::{det_vec, tiny_config};
    use super::*;
    use burn::tensor::{Int, Tensor};
    use safetensors::tensor::Dtype;

    // backend chosen per call site via Device

    /// The full name → (shape, values) table for `tiny_config`, values
    /// deterministic per tensor.
    pub(crate) fn tiny_tensor_table(
        config: &DecoderConfig,
        with_lm_head: bool,
    ) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let h = config.hidden_size;
        let kv = config.kv_heads() * config.head_dim();
        let inter = config.intermediate_size;
        let mut out: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut push = |name: String, shape: Vec<usize>, salt: usize| {
            let n: usize = shape.iter().product();
            out.push((name, shape, det_vec(n, salt)));
        };
        push(
            "model.embed_tokens.weight".into(),
            vec![config.vocab_size, h],
            7,
        );
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let s = 100 * (i + 1);
            push(format!("{p}.input_layernorm.weight"), vec![h], s);
            push(format!("{p}.self_attn.q_proj.weight"), vec![h, h], s + 1);
            push(format!("{p}.self_attn.k_proj.weight"), vec![kv, h], s + 2);
            push(format!("{p}.self_attn.v_proj.weight"), vec![kv, h], s + 3);
            push(format!("{p}.self_attn.o_proj.weight"), vec![h, h], s + 4);
            push(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                s + 8,
            );
            push(format!("{p}.mlp.gate_proj.weight"), vec![inter, h], s + 5);
            push(format!("{p}.mlp.up_proj.weight"), vec![inter, h], s + 6);
            push(format!("{p}.mlp.down_proj.weight"), vec![h, inter], s + 7);
        }
        push("model.norm.weight".into(), vec![h], 9);
        if with_lm_head {
            push("lm_head.weight".into(), vec![config.vocab_size, h], 8);
        }
        out
    }

    pub(crate) fn serialize_f32(table: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let buffers: Vec<(String, Vec<usize>, Vec<u8>)> = table
            .iter()
            .map(|(n, s, v)| {
                (
                    n.clone(),
                    s.clone(),
                    v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                )
            })
            .collect();
        let views: Vec<(&str, TensorView<'_>)> = buffers
            .iter()
            .map(|(n, s, b)| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
                )
            })
            .collect();
        safetensors::serialize(views, &None).unwrap()
    }

    fn serialize_bf16(table: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let buffers: Vec<(String, Vec<usize>, Vec<u8>)> = table
            .iter()
            .map(|(n, s, v)| {
                (
                    n.clone(),
                    s.clone(),
                    v.iter()
                        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
                        .collect(),
                )
            })
            .collect();
        let views: Vec<(&str, TensorView<'_>)> = buffers
            .iter()
            .map(|(n, s, b)| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::BF16, s.clone(), b).unwrap(),
                )
            })
            .collect();
        safetensors::serialize(views, &None).unwrap()
    }

    fn ids() -> Tensor<2, Int> {
        Tensor::from_data([[1, 5, 9, 2, 7]], &Device::ndarray())
    }

    #[test]
    fn loads_synthetic_checkpoint_and_produces_logits() {
        let config = tiny_config();
        let bytes = serialize_f32(&tiny_tensor_table(&config, true));
        let model = load_decoder_from_bytes(&config, &bytes, &Device::ndarray()).unwrap();
        let logits = model.logits(ids(), 0);
        assert_eq!(logits.dims(), [1, 5, config.vocab_size]);
        let v = logits.into_data().try_to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn bf16_checkpoint_loads_and_stays_close_to_f32() {
        let config = tiny_config();
        let table = tiny_tensor_table(&config, true);
        let f32_model =
            load_decoder_from_bytes(&config, &serialize_f32(&table), &Device::ndarray()).unwrap();
        let bf16_model =
            load_decoder_from_bytes(&config, &serialize_bf16(&table), &Device::ndarray()).unwrap();
        let a = f32_model
            .logits(ids(), 0)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let b = bf16_model
            .logits(ids(), 0)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.05, "bf16 load diverged from f32: {max_diff}");
    }

    #[test]
    fn tied_checkpoint_without_lm_head_loads_when_config_says_tied() {
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let bytes = serialize_f32(&tiny_tensor_table(&config, false));
        let model = load_decoder_from_bytes(&config, &bytes, &Device::ndarray()).unwrap();
        assert_eq!(model.logits(ids(), 0).dims(), [1, 5, config.vocab_size]);
    }

    #[test]
    fn missing_lm_head_without_tie_flag_rejected() {
        let config = tiny_config(); // tie_word_embeddings = false
        let bytes = serialize_f32(&tiny_tensor_table(&config, false));
        let err = load_decoder_from_bytes(&config, &bytes, &Device::ndarray()).unwrap_err();
        assert!(matches!(err, InferError::InvalidWeights(_)));
    }

    /// Load the tiny checkpoint with one extra tensor and return the refused
    /// feature.
    fn refused_with_extra(name: &str, shape: Vec<usize>) -> String {
        let config = tiny_config();
        let mut table = tiny_tensor_table(&config, true);
        let n = shape.iter().product();
        table.push((name.to_string(), shape, det_vec(n, 3)));
        match load_decoder_from_bytes(&config, &serialize_f32(&table), &Device::ndarray()) {
            Err(InferError::Unsupported { feature, .. }) => feature,
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("expected Unsupported, got a model"),
        }
    }

    #[test]
    fn attention_bias_tensor_refused() {
        // Qwen2 and 2.5 carry these; ignoring them answers wrongly.
        let name = "model.layers.0.self_attn.q_proj.bias";
        assert_eq!(refused_with_extra(name, vec![8]), format!("tensor {name}"));
    }

    #[test]
    fn mlp_bias_tensor_refused() {
        let name = "model.layers.1.mlp.down_proj.bias";
        assert_eq!(refused_with_extra(name, vec![8]), format!("tensor {name}"));
    }

    #[test]
    fn qk_norm_tensor_refused() {
        // Qwen3 8B and 14B pass the shape checks but carry these.
        let name = "model.layers.0.self_attn.q_norm.weight";
        assert_eq!(refused_with_extra(name, vec![2]), format!("tensor {name}"));
    }

    #[test]
    fn expert_tensor_refused() {
        let name = "model.layers.0.block_sparse_moe.experts.0.w1.weight";
        assert_eq!(
            refused_with_extra(name, vec![16, 8]),
            format!("tensor {name}")
        );
    }

    #[test]
    fn extra_norm_tensor_refused() {
        // Gemma 2 and 3 add norms around the MLP.
        let name = "model.layers.0.pre_feedforward_layernorm.weight";
        assert_eq!(refused_with_extra(name, vec![8]), format!("tensor {name}"));
    }

    #[test]
    fn layer_beyond_config_refused() {
        // A checkpoint deeper than its config would silently drop layers.
        let name = "model.layers.2.mlp.up_proj.weight";
        assert_eq!(
            refused_with_extra(name, vec![16, 8]),
            format!("tensor {name}")
        );
    }

    #[test]
    fn rope_inv_freq_buffer_is_tolerated() {
        let config = tiny_config();
        let mut table = tiny_tensor_table(&config, true);
        table.push((
            "model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
            vec![1],
            vec![1.0],
        ));
        load_decoder_from_bytes(&config, &serialize_f32(&table), &Device::ndarray()).unwrap();
    }

    #[test]
    fn missing_layer_tensor_names_the_tensor() {
        let config = tiny_config();
        let mut table = tiny_tensor_table(&config, true);
        table.retain(|(n, _, _)| n != "model.layers.1.mlp.up_proj.weight");
        let err = load_decoder_from_bytes(&config, &serialize_f32(&table), &Device::ndarray())
            .unwrap_err();
        match err {
            InferError::InvalidWeights(msg) => {
                assert!(msg.contains("model.layers.1.mlp.up_proj.weight"), "{msg}")
            },
            other => panic!("wrong error: {other:?}"),
        }
    }
}

/// A core artifact contains exactly this layer's declared tensors. Routed
/// experts live in separate artifacts and cannot be smuggled into the core.
pub(crate) fn validate_layer_names(
    config: &DecoderConfig,
    tensors: &SafeTensors<'_>,
    index: usize,
) -> Result<(), InferError> {
    let expected: HashSet<String> = layer_core_tensor_names(config, index)
        .iter()
        .map(|suffix| format!("model.layers.{index}.{suffix}"))
        .collect();
    let actual: HashSet<String> = tensors.names().into_iter().cloned().collect();
    if actual != expected {
        return Err(InferError::InvalidWeights(format!(
            "layer {index} tensor names do not match the declared Llama block"
        )));
    }
    Ok(())
}
