//! Safetensors bytes → Burn tensors, with dtype decode.
//!
//! The `embed::bert::safetensors_io` pattern (validate rank/shape/bytes,
//! decode little-endian, hand to Burn), widened for the llama-family
//! reality: HF checkpoints in this class ship **bfloat16** (TinyLlama,
//! Llama 3.2) or f16, so BF16/F16 decode to f32 here at the boundary.
//! Weights live as f32 in memory regardless of the on-disk dtype;
//! quantized formats are a later, deliberate step, not an auto-cast.

use burn::tensor::{Tensor, TensorData, backend::Backend};
use safetensors::tensor::{Dtype, TensorView};

use crate::infer::provider::InferError;

fn decode_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>, InferError> {
    match dtype {
        Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => Err(InferError::InvalidWeights(format!(
            "unsupported dtype {other:?} (f32/bf16/f16 decode; quantized formats are a later step)"
        ))),
    }
}

/// Read a tensor of known rank `D`, validating shape and decoding dtype.
pub fn extract<B: Backend, const D: usize>(
    view: &TensorView<'_>,
    expected_shape: [usize; D],
    device: &B::Device,
) -> Result<Tensor<B, D>, InferError> {
    let actual = view.shape();
    if actual.len() != D || actual != expected_shape {
        return Err(InferError::InvalidWeights(format!(
            "shape mismatch: expected {expected_shape:?}, got {actual:?}"
        )));
    }
    let data = decode_f32(view.dtype(), view.data())?;
    let expected_len: usize = expected_shape.iter().product();
    if data.len() != expected_len {
        return Err(InferError::InvalidWeights(format!(
            "element-count mismatch: shape {expected_shape:?} implies {expected_len}, got {}",
            data.len()
        )));
    }
    Ok(Tensor::from_data(
        TensorData::new(data, expected_shape),
        device,
    ))
}

/// Read a 1-D tensor (an RmsNorm gamma).
pub fn extract_1d<B: Backend>(
    view: &TensorView<'_>,
    len: usize,
    device: &B::Device,
) -> Result<Tensor<B, 1>, InferError> {
    extract::<B, 1>(view, [len], device)
}

/// Read a 2-D tensor in HF `[out, in]` orientation and transpose to
/// Burn's `[in, out]` — the same transpose-at-the-boundary rule
/// `embed::bert` documents.
pub fn extract_2d_transposed<B: Backend>(
    view: &TensorView<'_>,
    hf_rows_out: usize,
    hf_cols_in: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>, InferError> {
    Ok(extract::<B, 2>(view, [hf_rows_out, hf_cols_in], device)?.transpose())
}

/// Read a 2-D tensor as stored (an embedding lookup table, which stays
/// `[vocab, hidden]`).
pub fn extract_2d<B: Backend>(
    view: &TensorView<'_>,
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>, InferError> {
    extract::<B, 2>(view, [rows, cols], device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn bf16_decodes_close_to_f32() {
        let device = Default::default();
        let values = [0.05_f32, -0.31, 1.5, -2.25];
        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let view = TensorView::new(Dtype::BF16, vec![4], &bf16_bytes).unwrap();
        let t: Tensor<B, 1> = extract_1d(&view, 4, &device).unwrap();
        let out = t.into_data().to_vec::<f32>().unwrap();
        for (a, b) in out.iter().zip(&values) {
            assert!(
                (a - b).abs() < 0.02,
                "bf16 round-trip too lossy: {a} vs {b}"
            );
        }
    }

    #[test]
    fn transposed_extraction_flips_hf_orientation() {
        let device = Default::default();
        // HF [out=2, in=3], row-major [1 2 3 | 4 5 6].
        let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let view = TensorView::new(Dtype::F32, vec![2, 3], &bytes).unwrap();
        let t: Tensor<B, 2> = extract_2d_transposed(&view, 2, 3, &device).unwrap();
        assert_eq!(t.dims(), [3, 2]);
        let out = t.into_data().to_vec::<f32>().unwrap();
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn unsupported_dtype_names_the_policy() {
        let device = Default::default();
        let bytes = vec![0u8; 8];
        let view = TensorView::new(Dtype::I64, vec![1], &bytes).unwrap();
        let err = extract_1d::<B>(&view, 1, &device).unwrap_err();
        assert!(matches!(err, InferError::InvalidWeights(_)));
    }

    #[test]
    fn shape_mismatch_rejected() {
        let device = Default::default();
        let bytes = f32_bytes(&[1.0, 2.0, 3.0]);
        let view = TensorView::new(Dtype::F32, vec![3], &bytes).unwrap();
        assert!(extract_1d::<B>(&view, 4, &device).is_err());
    }
}
