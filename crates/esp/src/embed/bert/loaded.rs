//! Typed bundle of all tensors loaded from a HF BERT safetensors file.
//!
//! This is the structured intermediate between bytes-on-disk and a
//! constructed `BertModel<B>`. After extraction, every weight/bias is on
//! the chosen Burn backend with the correct rank and shape; assembly into
//! the model happens via [`crate::embed::bert::loader::load_into_model`].
//!
//! Organisation mirrors the BERT module hierarchy so each sub-bundle
//! maps 1:1 to one constructed sub-module.

use burn::tensor::{Tensor, backend::Backend};
use safetensors::SafeTensors;

use super::config::BertConfig;
use super::loader::{LoaderError, ModelArtifacts};
use super::safetensors_io::{extract_1d, extract_2d};

/// Loaded tensors for the embedding stage:
/// word + position + token-type lookup tables, plus LayerNorm gamma/beta.
#[derive(Debug)]
pub struct LoadedEmbeddings<B: Backend> {
    pub word: Tensor<B, 2>,
    pub position: Tensor<B, 2>,
    pub token_type: Tensor<B, 2>,
    pub ln_gamma: Tensor<B, 1>,
    pub ln_beta: Tensor<B, 1>,
}

/// Loaded tensors for one transformer layer.
///
/// Field naming follows the HF prefix structure within each layer:
/// `attention.self.{q,k,v}` → `q_*/k_*/v_*`; `attention.output.dense` →
/// `attn_out_*`; `attention.output.LayerNorm` → `attn_ln_*`;
/// `intermediate.dense` → `inter_*`; `output.dense` → `out_*`;
/// `output.LayerNorm` → `out_ln_*`.
///
/// **Linear-weight shape convention**: Burn `[in_features, out_features]`,
/// **already transposed** from HF's `[out, in]` at the safetensors-extraction
/// boundary. So `inter_w` is `[hidden, intermediate]` (not HF's
/// `[intermediate, hidden]`), and `out_w` is `[intermediate, hidden]`
/// (not HF's `[hidden, intermediate]`). Q/K/V and attention-output
/// weights are all `[hidden, hidden]` so the transpose is shape-invariant
/// but is still applied to keep numerics correct.
#[derive(Debug)]
pub struct LoadedBertLayer<B: Backend> {
    pub q_w: Tensor<B, 2>,
    pub q_b: Tensor<B, 1>,
    pub k_w: Tensor<B, 2>,
    pub k_b: Tensor<B, 1>,
    pub v_w: Tensor<B, 2>,
    pub v_b: Tensor<B, 1>,
    pub attn_out_w: Tensor<B, 2>,
    pub attn_out_b: Tensor<B, 1>,
    pub attn_ln_gamma: Tensor<B, 1>,
    pub attn_ln_beta: Tensor<B, 1>,
    pub inter_w: Tensor<B, 2>,
    pub inter_b: Tensor<B, 1>,
    pub out_w: Tensor<B, 2>,
    pub out_b: Tensor<B, 1>,
    pub out_ln_gamma: Tensor<B, 1>,
    pub out_ln_beta: Tensor<B, 1>,
}

/// Complete loaded BERT model: embedding stage + one entry per transformer layer.
#[derive(Debug)]
pub struct LoadedBert<B: Backend> {
    pub config: BertConfig,
    pub embeddings: LoadedEmbeddings<B>,
    pub layers: Vec<LoadedBertLayer<B>>,
}

impl<B: Backend> LoadedBert<B> {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Build a `BertModel<B>` from the loaded tensors. This is the
    /// canonical "tensors → model" entry point.
    pub fn into_model(&self, device: &B::Device) -> super::model::BertModel<B> {
        super::model::BertModel::from_loaded(self, device)
    }
}

/// Extract every expected tensor from a loaded safetensors file into a
/// typed [`LoadedBert<B>`]. Validates dtype and shape per tensor; returns
/// the first error encountered.
///
/// Path-based wrapper: reads `artifacts.weights_path` from disk, then
/// delegates to [`extract_all_tensors_from_bytes`].
pub fn extract_all_tensors<B: Backend>(
    artifacts: &ModelArtifacts,
    device: &B::Device,
) -> Result<LoadedBert<B>, LoaderError> {
    let bytes = std::fs::read(&artifacts.weights_path)?;
    extract_all_tensors_from_bytes::<B>(&artifacts.config, &bytes, device)
}

/// Extract every expected tensor directly from in-memory weights bytes.
/// Used when the model artifact flowed through a host store rather than the
/// filesystem.
pub fn extract_all_tensors_from_bytes<B: Backend>(
    config: &super::config::BertConfig,
    weights_bytes: &[u8],
    device: &B::Device,
) -> Result<LoadedBert<B>, LoaderError> {
    let st = SafeTensors::deserialize(weights_bytes)
        .map_err(|e| LoaderError::InvalidWeights(format!("{e}")))?;
    let cfg = config;
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let view = |name: &str| -> Result<safetensors::tensor::TensorView<'_>, LoaderError> {
        st.tensor(name)
            .map_err(|_| LoaderError::InvalidWeights(format!("missing tensor: {name}")))
    };

    let embeddings = LoadedEmbeddings {
        word: extract_2d(
            &view("embeddings.word_embeddings.weight")?,
            cfg.vocab_size,
            h,
            device,
        )?,
        position: extract_2d(
            &view("embeddings.position_embeddings.weight")?,
            cfg.max_position_embeddings,
            h,
            device,
        )?,
        token_type: extract_2d(
            &view("embeddings.token_type_embeddings.weight")?,
            cfg.type_vocab_size,
            h,
            device,
        )?,
        ln_gamma: extract_1d(&view("embeddings.LayerNorm.weight")?, h, device)?,
        ln_beta: extract_1d(&view("embeddings.LayerNorm.bias")?, h, device)?,
    };

    // PyTorch `Linear.weight` is stored as `[out, in]`; Burn expects
    // `[in, out]`. Transpose every `*_w` Linear tensor at this boundary
    // so `LoadedBertLayer` always holds Burn-convention shapes.
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("encoder.layer.{i}");
        let l = LoadedBertLayer {
            q_w: extract_2d(
                &view(&format!("{p}.attention.self.query.weight"))?,
                h,
                h,
                device,
            )?
            .swap_dims(0, 1),
            q_b: extract_1d(&view(&format!("{p}.attention.self.query.bias"))?, h, device)?,
            k_w: extract_2d(
                &view(&format!("{p}.attention.self.key.weight"))?,
                h,
                h,
                device,
            )?
            .swap_dims(0, 1),
            k_b: extract_1d(&view(&format!("{p}.attention.self.key.bias"))?, h, device)?,
            v_w: extract_2d(
                &view(&format!("{p}.attention.self.value.weight"))?,
                h,
                h,
                device,
            )?
            .swap_dims(0, 1),
            v_b: extract_1d(&view(&format!("{p}.attention.self.value.bias"))?, h, device)?,
            attn_out_w: extract_2d(
                &view(&format!("{p}.attention.output.dense.weight"))?,
                h,
                h,
                device,
            )?
            .swap_dims(0, 1),
            attn_out_b: extract_1d(
                &view(&format!("{p}.attention.output.dense.bias"))?,
                h,
                device,
            )?,
            attn_ln_gamma: extract_1d(
                &view(&format!("{p}.attention.output.LayerNorm.weight"))?,
                h,
                device,
            )?,
            attn_ln_beta: extract_1d(
                &view(&format!("{p}.attention.output.LayerNorm.bias"))?,
                h,
                device,
            )?,
            inter_w: extract_2d(
                &view(&format!("{p}.intermediate.dense.weight"))?,
                inter,
                h,
                device,
            )?
            .swap_dims(0, 1),
            inter_b: extract_1d(
                &view(&format!("{p}.intermediate.dense.bias"))?,
                inter,
                device,
            )?,
            out_w: extract_2d(
                &view(&format!("{p}.output.dense.weight"))?,
                h,
                inter,
                device,
            )?
            .swap_dims(0, 1),
            out_b: extract_1d(&view(&format!("{p}.output.dense.bias"))?, h, device)?,
            out_ln_gamma: extract_1d(&view(&format!("{p}.output.LayerNorm.weight"))?, h, device)?,
            out_ln_beta: extract_1d(&view(&format!("{p}.output.LayerNorm.bias"))?, h, device)?,
        };
        layers.push(l);
    }

    Ok(LoadedBert {
        config: cfg.clone(),
        embeddings,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn loaded_bert_layer_has_sixteen_tensor_fields() {
        // Sanity: per-layer tensor count matches the expected_tensors
        // contribution of 16 per layer (Q/K/V each {weight, bias} = 6;
        // attn output dense {w,b} = 2; attn output LN {γ,β} = 2;
        // intermediate {w,b} = 2; output dense {w,b} = 2; output LN {γ,β} = 2;
        // total 16).
        // This test compiles only if the struct really has 16 named tensors.
        // We construct a LoadedBertLayer with placeholder tensors to enforce.
        let device = Default::default();
        let _layer: LoadedBertLayer<B> = LoadedBertLayer {
            q_w: burn::tensor::Tensor::zeros([2, 2], &device),
            q_b: burn::tensor::Tensor::zeros([2], &device),
            k_w: burn::tensor::Tensor::zeros([2, 2], &device),
            k_b: burn::tensor::Tensor::zeros([2], &device),
            v_w: burn::tensor::Tensor::zeros([2, 2], &device),
            v_b: burn::tensor::Tensor::zeros([2], &device),
            attn_out_w: burn::tensor::Tensor::zeros([2, 2], &device),
            attn_out_b: burn::tensor::Tensor::zeros([2], &device),
            attn_ln_gamma: burn::tensor::Tensor::zeros([2], &device),
            attn_ln_beta: burn::tensor::Tensor::zeros([2], &device),
            inter_w: burn::tensor::Tensor::zeros([4, 2], &device),
            inter_b: burn::tensor::Tensor::zeros([4], &device),
            out_w: burn::tensor::Tensor::zeros([2, 4], &device),
            out_b: burn::tensor::Tensor::zeros([2], &device),
            out_ln_gamma: burn::tensor::Tensor::zeros([2], &device),
            out_ln_beta: burn::tensor::Tensor::zeros([2], &device),
        };
    }

    #[test]
    fn loaded_embeddings_has_five_tensor_fields() {
        let device = Default::default();
        let _emb: LoadedEmbeddings<B> = LoadedEmbeddings {
            word: burn::tensor::Tensor::zeros([10, 4], &device),
            position: burn::tensor::Tensor::zeros([8, 4], &device),
            token_type: burn::tensor::Tensor::zeros([2, 4], &device),
            ln_gamma: burn::tensor::Tensor::zeros([4], &device),
            ln_beta: burn::tensor::Tensor::zeros([4], &device),
        };
    }

    /// Tiny BERT config used for end-to-end load-and-run tests.
    /// Keeps tensor counts small so test time stays fast.
    fn tiny_config() -> BertConfig {
        BertConfig {
            vocab_size: 32,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            layer_norm_eps: 1.0e-12,
            hidden_act: "gelu".to_string(),
            pad_token_id: 0,
        }
    }

    fn synth_loaded(config: &BertConfig) -> LoadedBert<B> {
        use burn::tensor::{Distribution, Tensor};
        let device = Default::default();
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let dist = Distribution::Normal(0.0, 0.02);
        let r1 = |n: usize| Tensor::<B, 1>::random([n], dist, &device);
        let r2 = |a: usize, b: usize| Tensor::<B, 2>::random([a, b], dist, &device);

        let embeddings = LoadedEmbeddings {
            word: r2(config.vocab_size, h),
            position: r2(config.max_position_embeddings, h),
            token_type: r2(config.type_vocab_size, h),
            ln_gamma: Tensor::<B, 1>::ones([h], &device),
            ln_beta: Tensor::<B, 1>::zeros([h], &device),
        };
        // Synthesised tensors are in Burn convention `[in, out]`,
        // matching what `extract_all_tensors` produces after the
        // PyTorch-to-Burn transpose. See LoadedBertLayer doc.
        let layers = (0..config.num_hidden_layers)
            .map(|_| LoadedBertLayer {
                q_w: r2(h, h),
                q_b: r1(h),
                k_w: r2(h, h),
                k_b: r1(h),
                v_w: r2(h, h),
                v_b: r1(h),
                attn_out_w: r2(h, h),
                attn_out_b: r1(h),
                attn_ln_gamma: Tensor::<B, 1>::ones([h], &device),
                attn_ln_beta: Tensor::<B, 1>::zeros([h], &device),
                inter_w: r2(h, inter),
                inter_b: r1(inter),
                out_w: r2(inter, h),
                out_b: r1(h),
                out_ln_gamma: Tensor::<B, 1>::ones([h], &device),
                out_ln_beta: Tensor::<B, 1>::zeros([h], &device),
            })
            .collect();
        LoadedBert {
            config: config.clone(),
            embeddings,
            layers,
        }
    }

    #[test]
    fn synthesized_loaded_into_model_runs_forward_no_nans() {
        use burn::tensor::{Int, Tensor};
        let device = Default::default();
        let cfg = tiny_config();
        let loaded = synth_loaded(&cfg);
        let model = loaded.into_model(&device);
        let input_ids: Tensor<B, 2, Int> = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let out = model.forward_tokens(input_ids);
        assert_eq!(out.dims(), [1, 5, cfg.hidden_size]);
        let v = out.into_data().to_vec::<f32>().unwrap();
        assert!(
            v.iter().all(|x| !x.is_nan() && !x.is_infinite()),
            "BertModel built from LoadedBert produced NaN/Inf — \
             likely a layer-construction bug"
        );
    }

    #[test]
    fn synthesized_loaded_sentence_forward_returns_normalized_vec() {
        use burn::tensor::{Int, Tensor};
        let device = Default::default();
        let cfg = tiny_config();
        let loaded = synth_loaded(&cfg);
        let model = loaded.into_model(&device);
        let input_ids: Tensor<B, 2, Int> = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let out = model.forward_sentence(input_ids, crate::embed::bert::model::Pooling::Mean, true);
        assert_eq!(out.dims(), [1, cfg.hidden_size]);
        let v = out.into_data().to_vec::<f32>().unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1.0e-4,
            "sentence vector not L2-normalized: norm = {norm}"
        );
    }

    #[test]
    fn synthesized_loaded_is_deterministic_per_input() {
        use burn::tensor::{Int, Tensor};
        let device = Default::default();
        let cfg = tiny_config();
        let loaded = synth_loaded(&cfg);
        let model = loaded.into_model(&device);
        let input_ids: Tensor<B, 2, Int> = Tensor::from_data([[1, 2, 3]], &device);
        let a = model
            .forward_sentence(
                input_ids.clone(),
                crate::embed::bert::model::Pooling::Mean,
                true,
            )
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = model
            .forward_sentence(input_ids, crate::embed::bert::model::Pooling::Mean, true)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(a, b, "same input must produce identical output");
    }
}
