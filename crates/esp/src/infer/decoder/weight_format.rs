// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Explicit, versioned weight representations. Packed projections are stored
//! in HF [out, in] order and transposed by metadata at load. Embeddings and
//! norms retain their HF order.

use crate::infer::InferError;
use burn::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{DType, Device, Tensor, TensorData};
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightFormat {
    /// The original loader, including its deliberate widening of HF halves.
    #[default]
    F32,
    F16,
    Bf16,
    #[serde(rename = "q8-k32-f32-scale-f16-shared-v1")]
    Q8Block32,
    #[serde(rename = "q4-k32-f32-scale-f16-shared-v1")]
    Q4Block32,
}

impl std::str::FromStr for WeightFormat {
    type Err = InferError;
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            "q8-block32" | "q8-k32-f32-scale-f16-shared-v1" => Ok(Self::Q8Block32),
            "q4-block32" | "q4-k32-f32-scale-f16-shared-v1" => Ok(Self::Q4Block32),
            _ => Err(InferError::InvalidConfig(format!(
                "unknown weight format {name}"
            ))),
        }
    }
}

impl WeightFormat {
    pub fn id(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8Block32 => "q8-k32-f32-scale-f16-shared-v1",
            Self::Q4Block32 => "q4-k32-f32-scale-f16-shared-v1",
        }
    }

    pub fn activation_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::BF16,
            _ => DType::F16,
        }
    }

    pub fn scheme(self) -> Option<QuantScheme> {
        let value = match self {
            Self::Q8Block32 => QuantValue::Q8S,
            Self::Q4Block32 => QuantValue::Q4S,
            _ => return None,
        };
        Some(
            QuantScheme::default()
                .with_value(value)
                .with_store(QuantStore::PackedU32(0))
                .per_block([32], ScaleDtype::F32),
        )
    }

    pub fn execution_scheme(self) -> Option<QuantScheme> {
        self.scheme().map(|scheme| {
            scheme
                .with_store(QuantStore::PackedU32(1))
                .per_block([32, 1], ScaleDtype::F32)
        })
    }

    pub fn validate_device(self, device: &Device) -> Result<(), InferError> {
        for dtype in
            std::iter::once(self.activation_dtype()).chain(self.scheme().map(DType::QFloat))
        {
            if !device.supports_dtype(dtype) {
                return Err(InferError::Unsupported {
                    feature: format!("weight format {}", self.id()),
                    detail: format!("selected device does not support {dtype:?}"),
                });
            }
        }
        Ok(())
    }
}

fn invalid(error: impl std::fmt::Display) -> InferError {
    InferError::InvalidWeights(format!("weight format: {error}"))
}

pub(crate) fn values(view: &TensorView<'_>) -> Result<Vec<f32>, InferError> {
    let values: Vec<f32> = match view.dtype() {
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect(),
        other => return Err(invalid(format!("cannot import {other:?}"))),
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite weight"));
    }
    Ok(values)
}

/// Convert one bounded HF artifact at import, never on a generation pass.
/// Quantized buffers contain little-endian packed signed integers followed
/// by f32 block scales. Their U8 safetensors shape is the encoded byte count.
pub fn convert_artifact(bytes: &[u8], format: WeightFormat) -> Result<Vec<u8>, InferError> {
    let tensors = SafeTensors::deserialize(bytes).map_err(invalid)?;
    let mut buffers = Vec::new();
    for (name, view) in tensors.tensors() {
        let data = values(&view)?;
        let projection = view.shape().len() == 2 && name != "model.embed_tokens.weight";
        if let Some(scheme) = format.scheme().filter(|_| projection) {
            let [rows, cols]: [usize; 2] = view.shape().try_into().map_err(invalid)?;
            // Pack along K within each HF output row. Transposition at load
            // changes metadata, keeping packing and scales on the same axis.
            if !cols.is_multiple_of(32) {
                return Err(invalid(format!(
                    "{name}: projection input width {cols} must be divisible by 32"
                )));
            }
            let bits = scheme.value.size_bits();
            let maximum = ((1 << (bits - 1)) - 1) as f32;
            let mut packed = vec![0u8; data.len() * bits / 8];
            let mut scales = vec![0u8; data.len() / 32 * 4];
            for row in 0..rows {
                for block in (0..cols).step_by(32) {
                    let peak = data[row * cols + block..row * cols + block + 32]
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0f32, f32::max);
                    let scale = if peak == 0.0 {
                        1.0
                    } else {
                        (peak / maximum).max(f32::MIN_POSITIVE)
                    };
                    let offset = (row * (cols / 32) + block / 32) * 4;
                    scales[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
                    for column in block..block + 32 {
                        let q = (data[row * cols + column] / scale)
                            .round()
                            .clamp(-maximum, maximum) as i8;
                        let index = row * cols + column;
                        if bits == 8 {
                            packed[index] = q as u8;
                        } else {
                            packed[index / 2] |= ((q as u8) & 15) << ((index % 2) * 4);
                        }
                    }
                }
            }
            packed.extend(scales);
            buffers.push((name, Dtype::U8, vec![packed.len()], packed));
        } else {
            let dtype = match format {
                WeightFormat::F32 => Dtype::F32,
                WeightFormat::Bf16 => Dtype::BF16,
                _ => Dtype::F16,
            };
            let mut encoded =
                Vec::with_capacity(data.len() * if format == WeightFormat::F32 { 4 } else { 2 });
            for value in data {
                match format {
                    WeightFormat::F32 => encoded.extend_from_slice(&value.to_le_bytes()),
                    WeightFormat::Bf16 => {
                        let half = half::bf16::from_f32(value);
                        if !half.is_finite() {
                            return Err(invalid(format!("{name}: bf16 conversion overflow")));
                        }
                        encoded.extend_from_slice(&half.to_le_bytes());
                    },
                    _ => {
                        let half = half::f16::from_f32(value);
                        if !half.is_finite() {
                            return Err(invalid(format!("{name}: f16 conversion overflow")));
                        }
                        encoded.extend_from_slice(&half.to_le_bytes());
                    },
                }
            }
            let data = encoded;
            buffers.push((name, dtype, view.shape().to_vec(), data));
        }
    }
    let views: Result<Vec<_>, _> = buffers
        .iter()
        .map(|(name, dtype, shape, data)| {
            TensorView::new(*dtype, shape.clone(), data).map(|view| (name.clone(), view))
        })
        .collect();
    safetensors::serialize(views.map_err(invalid)?, &None).map_err(invalid)
}

pub(crate) fn extract<const D: usize>(
    view: &TensorView<'_>,
    shape: [usize; D],
    device: &Device,
    format: WeightFormat,
    transpose: bool,
) -> Result<Tensor<D>, InferError> {
    if view.dtype() == Dtype::U8 {
        if view.shape() != [view.data().len()] {
            return Err(invalid(
                "packed buffer must have its encoded byte-count shape",
            ));
        }
        let scheme = format
            .scheme()
            .ok_or_else(|| invalid("packed tensor in a floating-point row"))?;
        if D != 2 || !transpose {
            return Err(invalid("packed tensor must be a projection"));
        }
        if !shape[1].is_multiple_of(32) {
            return Err(invalid(
                "packed projection input width must be divisible by 32",
            ));
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("projection shape overflow"))?;
        let value_bytes = elements
            .checked_mul(scheme.value.size_bits())
            .ok_or_else(|| invalid("projection byte count overflow"))?
            / 8;
        let scale_bytes = elements / 32 * 4;
        if value_bytes.checked_add(scale_bytes) != Some(view.data().len())
            || view.data()[value_bytes..].chunks_exact(4).any(|b| {
                let scale = f32::from_le_bytes(b.try_into().unwrap());
                !scale.is_finite() || scale <= 0.0
            })
        {
            return Err(invalid(
                "packed projection byte count or block scales are invalid",
            ));
        }
        let data =
            TensorData::try_from_bytes_vec(view.data().to_vec(), shape, DType::QFloat(scheme))
                .map_err(invalid)?;
        let dtype = data.dtype();
        return Ok(Tensor::from_data(data, (device, dtype)).transpose());
    }
    if view.shape() != shape {
        return Err(invalid(format!(
            "expected {shape:?}, got {:?}",
            view.shape()
        )));
    }
    let expected = match format {
        WeightFormat::F32 => None,
        WeightFormat::Bf16 => Some(Dtype::BF16),
        _ => Some(Dtype::F16),
    };
    if expected.is_some_and(|dtype| dtype != view.dtype()) {
        return Err(invalid("artifact dtype disagrees with declared format"));
    }
    if format.scheme().is_some() && transpose {
        return Err(invalid("unpacked projection in a quantized row"));
    }
    let data = match format {
        WeightFormat::F32 => TensorData::new(values(view)?, shape),
        WeightFormat::Bf16 => TensorData::new(
            view.data()
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            shape,
        ),
        _ => TensorData::new(
            view.data()
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            shape,
        ),
    };
    let dtype = data.dtype();
    let tensor: Tensor<D> = Tensor::from_data(data, (device, dtype));
    Ok(if transpose {
        tensor.transpose()
    } else {
        tensor
    })
}

#[cfg(all(test, feature = "decoder-flex"))]
mod tests {
    use super::*;

    fn projection(rows: usize, cols: usize) -> Vec<u8> {
        let data: Vec<u8> = (0..rows * cols)
            .flat_map(|i| (((i % 103) as f32 - 51.0) / 40.0).to_le_bytes())
            .collect();
        let view = TensorView::new(Dtype::F32, vec![rows, cols], &data).unwrap();
        safetensors::serialize([("projection.weight", view)], &None).unwrap()
    }

    fn check(device: Device, format: WeightFormat, width: usize, sequence: usize) {
        format.validate_device(&device).unwrap();
        let source = projection(width, width);
        let encoded = convert_artifact(&source, format).unwrap();
        let tensors = SafeTensors::deserialize(&encoded).unwrap();
        let weights = extract::<2>(
            &tensors.tensor("projection.weight").unwrap(),
            [width, width],
            &device,
            format,
            true,
        )
        .unwrap();
        if format.scheme().is_none() {
            assert_eq!(weights.dtype(), format.activation_dtype());
        } else {
            assert!(matches!(weights.dtype(), DType::QFloat(_)));
        }
        // Decode the stored representation on the host, independently of the
        // GPU matmul kernel. This separates quantization loss from arithmetic
        // correctness, including the full-size workgroup-memory regression.
        let reference: Vec<f32> = if let Some(scheme) = format.scheme() {
            let view = tensors.tensor("projection.weight").unwrap();
            let data = TensorData::try_from_bytes_vec(
                view.data().to_vec(),
                [width, width],
                DType::QFloat(scheme),
            )
            .unwrap();
            let shape = data.shape().clone();
            let packed = burn::tensor::quantization::QuantizedBytes {
                bytes: data.into_bytes(),
                shape,
                scheme,
            };
            let (values, scales) = packed.into_vec_i8();
            (0..width * width)
                .map(|i| {
                    let (k, n) = (i / width, i % width);
                    values[n * width + k] as f32 * scales.block[n * (width / 32) + k / 32]
                })
                .collect()
        } else {
            weights
                .clone()
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap()
        };
        let weight_tolerance = match format {
            WeightFormat::Q4Block32 => 0.1,
            WeightFormat::Q8Block32 => 0.006,
            _ => 0.01,
        };
        for (index, value) in reference.iter().enumerate() {
            let original = ((((index % width) * width + index / width) % 103) as f32 - 51.0) / 40.0;
            assert!(
                (value - original).abs() < weight_tolerance,
                "weight error for {format:?} at {index}"
            );
        }
        let input = Tensor::<2>::ones([sequence, width], &device).cast(format.activation_dtype());
        let result = input
            .matmul(weights)
            .cast(DType::F32)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for column in 0..width {
            let expected: f32 = (0..width).map(|i| reference[i * width + column]).sum();
            let tolerance = 0.1;
            assert!(
                (result[column] - expected).abs() < tolerance,
                "{format:?} at {column}: {} vs {expected}",
                result[column]
            );
        }
    }

    #[test]
    fn cpu_preserves_halves_and_refuses_unsupported_packed_schemes() {
        for format in [WeightFormat::F16, WeightFormat::Bf16] {
            check(Device::flex(), format, 64, 1);
        }
        for format in [WeightFormat::Q8Block32, WeightFormat::Q4Block32] {
            assert!(format.validate_device(&Device::flex()).is_err());
        }
    }

    #[cfg(feature = "decoder-wgpu")]
    #[test]
    fn gpu_preserves_half_and_multiplies_imported_blocks() {
        let device = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0));
        for format in [
            WeightFormat::F16,
            WeightFormat::Q8Block32,
            WeightFormat::Q4Block32,
        ] {
            check(device.clone(), format, 64, 1);
        }
    }

    #[cfg(feature = "decoder-wgpu")]
    #[test]
    fn gpu_q4_real_projection_shape_fits_workgroup_memory() {
        let device = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0));
        check(device.clone(), WeightFormat::Q4Block32, 960, 30);
        check(device, WeightFormat::Q4Block32, 960, 1);
    }

    #[cfg(feature = "decoder-wgpu")]
    #[test]
    fn gpu_streamed_formats_cover_untied_heads_and_warm_decode() {
        use super::super::loader::tests::{serialize_f32, tiny_tensor_table};
        use super::super::{loader, model::DecoderModel, streaming::StreamingOptions};
        use burn::tensor::Int;
        let mut config = super::super::test_support::tiny_config();
        config.hidden_size = 64;
        config.intermediate_size = 128;
        config.vocab_size = 79;
        let table = tiny_tensor_table(&config, true);
        let shared: Vec<_> = table
            .iter()
            .filter(|(name, _, _)| !name.starts_with("model.layers."))
            .cloned()
            .collect();
        let device = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0));
        let mut baseline: Option<Vec<f32>> = None;
        for format in [
            WeightFormat::F16,
            WeightFormat::Q8Block32,
            WeightFormat::Q4Block32,
        ] {
            let bytes = convert_artifact(&serialize_f32(&shared), format).unwrap();
            let tensors = SafeTensors::deserialize(&bytes).unwrap();
            let loaded =
                loader::load_decoder_shared_tensors_with_format(&config, &tensors, &device, format)
                    .unwrap();
            let layers: Vec<_> = (0..config.num_hidden_layers)
                .map(|i| {
                    let prefix = format!("model.layers.{i}.");
                    let layer: Vec<_> = table
                        .iter()
                        .filter(|(name, _, _)| name.starts_with(&prefix))
                        .cloned()
                        .collect();
                    convert_artifact(&serialize_f32(&layer), format).unwrap()
                })
                .collect();
            let model = DecoderModel::from_layer_source_with_format(
                config.clone(),
                loaded,
                Box::new(move |i: usize, _| Ok(layers[i].clone())),
                StreamingOptions::default(),
                &device,
                format,
            )
            .unwrap();
            let mut cache = model.new_cache();
            let mut logits = Vec::new();
            for ids in [vec![3i32, 7], vec![9i32]] {
                let len = ids.len();
                let output = model
                    .try_forward_cached_last(
                        Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), &device),
                        &mut cache,
                    )
                    .unwrap();
                assert_eq!(output.dims(), [1, 1, 79]);
                assert_eq!(output.dtype(), DType::F32);
                logits.extend(output.into_data().try_to_vec::<f32>().unwrap());
            }
            assert!(logits.iter().all(|v| v.is_finite()));
            if let Some(reference) = &baseline {
                assert!(
                    logits
                        .iter()
                        .zip(reference)
                        .all(|(a, b)| (a - b).abs() < 0.05)
                );
            } else {
                baseline = Some(logits);
            }
        }
    }

    #[test]
    fn refuses_mixed_format_and_non_finite_imports() {
        let source = projection(64, 64);
        let quantized = convert_artifact(&source, WeightFormat::Q4Block32).unwrap();
        let table = SafeTensors::deserialize(&quantized).unwrap();
        assert!(
            extract::<2>(
                &table.tensor("projection.weight").unwrap(),
                [64, 64],
                &Device::flex(),
                WeightFormat::F16,
                true
            )
            .is_err()
        );
        let data = f32::NAN.to_le_bytes();
        let view = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let bytes = safetensors::serialize([("norm.weight", view)], &None).unwrap();
        assert!(convert_artifact(&bytes, WeightFormat::F16).is_err());
        let data = f32::MAX.to_le_bytes();
        let view = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let bytes = safetensors::serialize([("norm.weight", view)], &None).unwrap();
        assert!(convert_artifact(&bytes, WeightFormat::F16).is_err());
    }

    #[test]
    fn refuses_invalid_scales_before_device_upload() {
        let mut bytes = vec![0; 64 * 64 / 2 + 64 * 64 / 32 * 4];
        let view = TensorView::new(Dtype::U8, vec![bytes.len()], &bytes).unwrap();
        assert!(
            extract::<2>(
                &view,
                [64, 64],
                &Device::flex(),
                WeightFormat::Q4Block32,
                true
            )
            .is_err()
        );
        for scale in bytes[64 * 64 / 2..].chunks_exact_mut(4) {
            scale.copy_from_slice(&f32::NAN.to_le_bytes());
        }
        let view = TensorView::new(Dtype::U8, vec![bytes.len()], &bytes).unwrap();
        assert!(
            extract::<2>(
                &view,
                [64, 64],
                &Device::flex(),
                WeightFormat::Q4Block32,
                true
            )
            .is_err()
        );
    }
}
