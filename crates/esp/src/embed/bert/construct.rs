//! Construct Burn nn modules (`Linear`, `Embedding`, `LayerNorm`) with
//! pre-loaded weight tensors.
//!
//! BERT weight loading boils down to: build each Burn nn primitive with a
//! tensor we already have, rather than via random initialisation. This
//! module provides the small bridge functions that do exactly that.
//!
//! These functions are the primitives the future `LoadedBert::into_model`
//! method composes into a fully-loaded `BertModel`.

use burn::module::Param;
use burn::nn::{Embedding, LayerNorm, LayerNormConfig, Linear};
use burn::tensor::{Device, Tensor};

/// Build a `Linear` whose weight and bias are the supplied tensors.
///
/// `weight` shape is `[in_features, out_features]` (Burn convention; HF
/// `Linear.weight` is stored as `[out, in]` so callers may need to
/// transpose at the safetensors-extraction boundary).
pub fn linear_from_loaded(weight: Tensor<2>, bias: Tensor<1>, _device: &Device) -> Linear {
    Linear {
        weight: Param::from_tensor(weight),
        bias: Some(Param::from_tensor(bias)),
    }
}

/// Build an `Embedding` whose weight is the supplied lookup table.
/// Shape is `[n_embeddings, d_embedding]`.
pub fn embedding_from_loaded(weight: Tensor<2>, _device: &Device) -> Embedding {
    Embedding {
        weight: Param::from_tensor(weight),
    }
}

/// Build a `LayerNorm` whose gamma and beta are the supplied tensors.
///
/// HF stores `LayerNorm.weight`/`LayerNorm.bias`; Burn calls these
/// `gamma`/`beta`. Same semantics, different field names — we map at this
/// boundary so callers above use the HF naming.
pub fn layer_norm_from_loaded(
    gamma: Tensor<1>,
    beta: Tensor<1>,
    epsilon: f64,
    device: &Device,
) -> LayerNorm {
    let [size] = gamma.dims();
    let mut ln = LayerNormConfig::new(size)
        .with_epsilon(epsilon)
        .init(device);
    ln.gamma = Param::from_tensor(gamma);
    ln.beta = Some(Param::from_tensor(beta));
    ln
}

#[cfg(test)]
mod tests {
    use super::*;

    // backend chosen per call site via Device

    #[test]
    fn linear_from_loaded_round_trips_a_known_weight() {
        let device = Device::ndarray();
        // 2-input, 3-output linear. Weight [in=2, out=3].
        let w = Tensor::<2>::from_data([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &device);
        let b = Tensor::<1>::from_data([0.5, 0.5, 0.5], &device);
        let linear = linear_from_loaded(w, b, &device);
        // Forward y = x @ W + b. x = [1.0, 2.0] → y = [1+0.5, 2+0.5, 0+0.5]
        let x = Tensor::<2>::from_data([[1.0, 2.0]], &device);
        let y = linear.forward(x);
        let v = y.into_data().to_vec::<f32>().unwrap();
        let approx = |a: f32, b: f32| (a - b).abs() < 1.0e-5;
        assert!(approx(v[0], 1.5), "v[0] = {}", v[0]);
        assert!(approx(v[1], 2.5), "v[1] = {}", v[1]);
        assert!(approx(v[2], 0.5), "v[2] = {}", v[2]);
    }

    #[test]
    fn embedding_from_loaded_round_trips_lookups() {
        let device = Device::ndarray();
        // 4-token vocabulary, 2-dim embeddings.
        let w = Tensor::<2>::from_data(
            [
                [1.0, 0.0],  // token 0
                [0.0, 1.0],  // token 1
                [-1.0, 0.0], // token 2
                [0.0, -1.0], // token 3
            ],
            &device,
        );
        let emb = embedding_from_loaded(w, &device);
        // Look up tokens [0, 2]. Should return [[1, 0], [-1, 0]].
        let ids = Tensor::<2, burn::tensor::Int>::from_data([[0, 2]], &device);
        let out = emb.forward(ids);
        let v = out.into_data().to_vec::<f32>().unwrap();
        let approx = |a: f32, b: f32| (a - b).abs() < 1.0e-5;
        assert!(approx(v[0], 1.0));
        assert!(approx(v[1], 0.0));
        assert!(approx(v[2], -1.0));
        assert!(approx(v[3], 0.0));
    }

    #[test]
    fn layer_norm_from_loaded_applies_gamma_and_beta() {
        let device = Device::ndarray();
        // Identity LN: gamma=1, beta=0. Should normalize input to ~zero mean, unit variance.
        let gamma = Tensor::<1>::from_data([1.0, 1.0, 1.0, 1.0], &device);
        let beta = Tensor::<1>::from_data([0.0, 0.0, 0.0, 0.0], &device);
        let ln = layer_norm_from_loaded(gamma, beta, 1.0e-5, &device);
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device);
        let y = ln.forward(x);
        let v = y.into_data().to_vec::<f32>().unwrap();
        // Output should have mean ~0, variance ~1.
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 1.0e-4, "mean = {}", mean);
    }

    #[test]
    fn layer_norm_from_loaded_with_nontrivial_gamma_scales_output() {
        let device = Device::ndarray();
        let gamma = Tensor::<1>::from_data([2.0, 2.0, 2.0, 2.0], &device);
        let beta = Tensor::<1>::from_data([0.0, 0.0, 0.0, 0.0], &device);
        let ln = layer_norm_from_loaded(gamma, beta, 1.0e-5, &device);
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device);
        let y = ln.forward(x);
        let v = y.into_data().to_vec::<f32>().unwrap();
        // With gamma=2, output norm doubles relative to identity LN.
        // Sanity: largest abs value should be ~2x the largest of identity-LN output.
        assert!(v.iter().any(|x| x.abs() > 1.5));
    }

    #[test]
    fn linear_from_loaded_no_nans() {
        let device = Device::ndarray();
        let w = Tensor::<2>::random(
            [16, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let b = Tensor::<1>::random([32], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let linear = linear_from_loaded(w, b, &device);
        let x = Tensor::<2>::random(
            [4, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let y = linear.forward(x);
        let v = y.into_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| !x.is_nan() && !x.is_infinite()));
    }
}
