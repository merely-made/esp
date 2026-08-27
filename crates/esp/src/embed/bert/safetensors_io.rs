// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Convert raw safetensors tensor bytes into Burn `Tensor<D>` values.
//!
//! `safetensors` stores tensors as raw little-endian bytes plus a JSON
//! header describing dtype + shape. This module is the bytes → tensor
//! bridge: validate dtype + shape, decode F32 or F16 into Burn's f32 tensors,
//! and reject every other published dtype explicitly.

use burn::tensor::{Device, Tensor, TensorData};
use half::f16;
use safetensors::tensor::{Dtype, TensorView};

use super::loader::LoaderError;

/// Read a tensor of known rank `D` from a safetensors view, validating
/// dtype and shape, and constructing a Burn `Tensor<D>` on the given
/// device.
pub fn extract<const D: usize>(
    view: &TensorView<'_>,
    expected_shape: [usize; D],
    device: &Device,
) -> Result<Tensor<D>, LoaderError> {
    let actual_shape = view.shape();
    if actual_shape.len() != D {
        return Err(LoaderError::InvalidWeights(format!(
            "expected rank-{D} tensor, got rank-{} (shape {:?})",
            actual_shape.len(),
            actual_shape
        )));
    }
    let mut shape_array = [0usize; D];
    for (i, &dim) in actual_shape.iter().enumerate() {
        shape_array[i] = dim;
    }
    if shape_array != expected_shape {
        return Err(LoaderError::InvalidWeights(format!(
            "shape mismatch: expected {expected_shape:?}, got {shape_array:?}"
        )));
    }
    let bytes = view.data();
    let elem_count: usize = expected_shape.iter().product();
    let bytes_per_element = match view.dtype() {
        Dtype::F32 => 4,
        Dtype::F16 => 2,
        dtype => {
            return Err(LoaderError::InvalidWeights(format!(
                "expected F32 or F16, got {dtype:?}"
            )));
        }
    };
    if bytes.len() != elem_count * bytes_per_element {
        return Err(LoaderError::InvalidWeights(format!(
            "byte-count mismatch: shape {expected_shape:?} implies {} bytes, got {}",
            elem_count * bytes_per_element,
            bytes.len()
        )));
    }
    let data: Vec<f32> = match view.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect(),
        _ => unreachable!("unsupported dtypes returned above"),
    };
    let tensor_data = TensorData::new(data, expected_shape);
    Ok(Tensor::from_data(tensor_data, device))
}

/// Read a 1-D tensor (bias / LayerNorm gamma / LayerNorm beta).
pub fn extract_1d(
    view: &TensorView<'_>,
    expected_len: usize,
    device: &Device,
) -> Result<Tensor<1>, LoaderError> {
    extract::<1>(view, [expected_len], device)
}

/// Read a 2-D tensor (Linear weight, Embedding weight).
pub fn extract_2d(
    view: &TensorView<'_>,
    expected_rows: usize,
    expected_cols: usize,
    device: &Device,
) -> Result<Tensor<2>, LoaderError> {
    extract::<2>(view, [expected_rows, expected_cols], device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    // backend chosen per call site via Device

    fn make_view<'a>(dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> TensorView<'a> {
        TensorView::new(dtype, shape, data).expect("valid TensorView")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn extract_1d_round_trips_f32() {
        let device = Device::ndarray();
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes = f32_bytes(&values);
        let view = make_view(Dtype::F32, vec![4], &bytes);
        let t: Tensor<1> = extract_1d(&view, 4, &device).unwrap();
        let out = t.into_data().to_vec::<f32>().unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn extract_2d_round_trips_f32() {
        let device = Device::ndarray();
        // 2x3: row-major [1, 2, 3 | 4, 5, 6]
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = f32_bytes(&values);
        let view = make_view(Dtype::F32, vec![2, 3], &bytes);
        let t: Tensor<2> = extract_2d(&view, 2, 3, &device).unwrap();
        let out = t.into_data().to_vec::<f32>().unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn rank_mismatch_errors() {
        let device = Device::ndarray();
        let bytes = f32_bytes(&[1.0, 2.0]);
        let view = make_view(Dtype::F32, vec![2], &bytes);
        let result = extract::<2>(&view, [1, 2], &device);
        assert!(matches!(result, Err(LoaderError::InvalidWeights(_))));
    }

    #[test]
    fn shape_mismatch_errors() {
        let device = Device::ndarray();
        let bytes = f32_bytes(&[1.0, 2.0, 3.0]);
        let view = make_view(Dtype::F32, vec![3], &bytes);
        let result = extract_1d(&view, 4, &device);
        assert!(matches!(result, Err(LoaderError::InvalidWeights(_))));
    }

    #[test]
    fn extract_1d_decodes_f16_to_f32() {
        let device = Device::ndarray();
        let expected = [1.0_f32, -2.0, 0.5, 65_504.0];
        let raw_bytes = expected
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect::<Vec<_>>();
        let view = make_view(Dtype::F16, vec![4], &raw_bytes);
        let tensor = extract_1d(&view, 4, &device).unwrap();
        assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn unsupported_dtype_errors() {
        let device = Device::ndarray();
        let raw_bytes = vec![0; 8];
        let view = make_view(Dtype::BF16, vec![4], &raw_bytes);
        let result = extract_1d(&view, 4, &device);
        assert!(matches!(result, Err(LoaderError::InvalidWeights(_))));
    }

    #[test]
    fn extract_2d_with_realistic_dimensions() {
        // MiniLM word embedding shape: [vocab_size, hidden_size]. Try a
        // smaller analog to verify multi-dim layouts round-trip cleanly.
        let device = Device::ndarray();
        let mut values = Vec::with_capacity(16 * 8);
        for i in 0..(16 * 8) {
            values.push(i as f32 * 0.01);
        }
        let bytes = f32_bytes(&values);
        let view = make_view(Dtype::F32, vec![16, 8], &bytes);
        let t: Tensor<2> = extract_2d(&view, 16, 8, &device).unwrap();
        assert_eq!(t.dims(), [16, 8]);
        let back = t.into_data().to_vec::<f32>().unwrap();
        assert_eq!(back, values);
    }

    // Note: a "buggy view" (shape inconsistent with byte-count) is
    // unconstructable — `TensorView::new` validates that itself and
    // returns `InvalidTensorView` before our code sees it. So our
    // byte-count check inside `extract` is defense-in-depth that's not
    // separately testable here. Documented for future maintainers.
}
