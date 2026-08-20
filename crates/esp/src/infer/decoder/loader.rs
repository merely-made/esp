//! HF safetensors → [`DecoderModel`].
//!
//! Maps the llama-family checkpoint names (`model.embed_tokens.weight`,
//! `model.layers.{i}.self_attn.q_proj.weight`, …) onto the decoder's
//! loaded-tensor bundles, transposing every linear from HF `[out, in]`
//! to Burn `[in, out]` at this boundary. Byte-buffer in, model out — the
//! eidetic corridor's `ModelComponents.weight_bytes` feeds this directly
//! (P2), and a file path is just `std::fs::read` above it.

use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use burn::tensor::Device;

use super::config::DecoderConfig;
use super::layer::LoadedDecoderLayer;
use super::model::{DecoderModel, LoadedDecoder};
use super::tensors::{extract_1d, extract_2d, extract_2d_transposed};
use crate::infer::provider::InferError;

fn view<'a>(tensors: &'a SafeTensors<'_>, name: &str) -> Result<TensorView<'a>, InferError> {
    tensors
        .tensor(name)
        .map_err(|_| InferError::InvalidWeights(format!("missing tensor: {name}")))
}

/// Build a [`DecoderModel`] from HF `model.safetensors` bytes.
///
/// `lm_head.weight` may be absent when `config.tie_word_embeddings` is
/// set (Llama 3.2 1B, SmolLM); a checkpoint that omits it *without* the
/// config flag is rejected rather than silently tied.
pub fn load_decoder_from_bytes(
    config: &DecoderConfig,
    weights: &[u8],
    device: &Device,
) -> Result<DecoderModel, InferError> {
    let tensors = SafeTensors::deserialize(weights)
        .map_err(|e| InferError::InvalidWeights(format!("safetensors parse: {e}")))?;

    let h = config.hidden_size;
    let kv = config.kv_heads() * config.head_dim();
    let inter = config.intermediate_size;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");
        layers.push(LoadedDecoderLayer {
            input_norm_gamma: extract_1d(
                &view(&tensors, &format!("{p}.input_layernorm.weight"))?,
                h,
                device,
            )?,
            q_w: extract_2d_transposed(
                &view(&tensors, &format!("{p}.self_attn.q_proj.weight"))?,
                h,
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
                h,
                device,
            )?,
            post_norm_gamma: extract_1d(
                &view(&tensors, &format!("{p}.post_attention_layernorm.weight"))?,
                h,
                device,
            )?,
            gate_w: extract_2d_transposed(
                &view(&tensors, &format!("{p}.mlp.gate_proj.weight"))?,
                inter,
                h,
                device,
            )?,
            up_w: extract_2d_transposed(
                &view(&tensors, &format!("{p}.mlp.up_proj.weight"))?,
                inter,
                h,
                device,
            )?,
            down_w: extract_2d_transposed(
                &view(&tensors, &format!("{p}.mlp.down_proj.weight"))?,
                h,
                inter,
                device,
            )?,
        });
    }

    let lm_head_w = match tensors.tensor("lm_head.weight") {
        Ok(v) => Some(extract_2d_transposed(&v, config.vocab_size, h, device)?),
        Err(_) if config.tie_word_embeddings => None,
        Err(_) => {
            return Err(InferError::InvalidWeights(
                "lm_head.weight missing but tie_word_embeddings is false".to_string(),
            ));
        }
    };

    let loaded = LoadedDecoder {
        embed_w: extract_2d(
            &view(&tensors, "model.embed_tokens.weight")?,
            config.vocab_size,
            h,
            device,
        )?,
        layers,
        final_norm_gamma: extract_1d(&view(&tensors, "model.norm.weight")?, h, device)?,
        lm_head_w,
    };
    Ok(DecoderModel::from_loaded(config.clone(), loaded, device))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{det_vec, tiny_config};
    use super::*;
        use burn::tensor::{Int, Tensor};
    use safetensors::tensor::Dtype;

    // backend chosen per call site via Device

    /// The full name → (shape, values) table for `tiny_config`, values
    /// deterministic per tensor.
    fn tiny_tensor_table(
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

    fn serialize_f32(table: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
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
        let v = logits.into_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn bf16_checkpoint_loads_and_stays_close_to_f32() {
        let config = tiny_config();
        let table = tiny_tensor_table(&config, true);
        let f32_model =
            load_decoder_from_bytes(&config, &serialize_f32(&table), &Device::ndarray())
                .unwrap();
        let bf16_model =
            load_decoder_from_bytes(&config, &serialize_bf16(&table), &Device::ndarray())
                .unwrap();
        let a = f32_model
            .logits(ids(), 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = bf16_model
            .logits(ids(), 0)
            .into_data()
            .to_vec::<f32>()
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

    #[test]
    fn missing_layer_tensor_names_the_tensor() {
        let config = tiny_config();
        let mut table = tiny_tensor_table(&config, true);
        table.retain(|(n, _, _)| n != "model.layers.1.mlp.up_proj.weight");
        let err =
            load_decoder_from_bytes(&config, &serialize_f32(&table), &Device::ndarray())
                .unwrap_err();
        match err {
            InferError::InvalidWeights(msg) => {
                assert!(msg.contains("model.layers.1.mlp.up_proj.weight"), "{msg}")
            }
            other => panic!("wrong error: {other:?}"),
        }
    }
}
