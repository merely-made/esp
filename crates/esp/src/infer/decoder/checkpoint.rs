// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Bounded host-side safetensors import. Headers are read separately, then
//! exact tensor ranges; the source checkpoint is never read into one buffer.
//! The resident uses the resulting immutable library artifacts, not paths.

use super::{DecoderConfig, loader::LAYER_TENSORS};
use crate::infer::InferError;
use safetensors::tensor::{Dtype, Metadata, TensorInfo, TensorView};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const HEADER_LIMIT: u64 = 16 * 1024 * 1024;

fn invalid(error: impl std::fmt::Display) -> InferError {
    InferError::InvalidWeights(format!("checkpoint: {error}"))
}

struct Shard {
    file: File,
    data_start: u64,
}
struct Location {
    shard: usize,
    info: TensorInfo,
}

/// Open files and their validated tensor directory. An HF weight-map may
/// distribute one layer's tensors over several shards.
pub struct CheckpointReader {
    shards: Vec<Shard>,
    locations: BTreeMap<String, Location>,
    bytes_read: u64,
}

impl CheckpointReader {
    pub fn open(directory: &Path, config: &DecoderConfig) -> Result<Self, InferError> {
        let index_path = directory.join("model.safetensors.index.json");
        let weight_map: Option<BTreeMap<String, String>> = match File::open(index_path) {
            Ok(file) => {
                let mut bytes = Vec::new();
                file.take(HEADER_LIMIT + 1)
                    .read_to_end(&mut bytes)
                    .map_err(invalid)?;
                if bytes.len() as u64 > HEADER_LIMIT {
                    return Err(invalid("weight index exceeds header limit"));
                }
                #[derive(serde::Deserialize)]
                struct Index {
                    weight_map: BTreeMap<String, String>,
                }
                Some(
                    serde_json::from_slice::<Index>(&bytes)
                        .map_err(invalid)?
                        .weight_map,
                )
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(invalid(error)),
        };
        let files: HashSet<String> = weight_map
            .as_ref()
            .map(|map| map.values().cloned().collect())
            .unwrap_or_else(|| HashSet::from(["model.safetensors".into()]));
        if files.is_empty() {
            return Err(invalid("empty weight map"));
        }
        let mut files: Vec<_> = files.into_iter().collect();
        files.sort();
        let mut reader = Self {
            shards: Vec::new(),
            locations: BTreeMap::new(),
            bytes_read: 0,
        };
        for name in files {
            // Index names are checkpoint basenames, never arbitrary paths or
            // Windows drive/alternate-stream references.
            if name.contains(['/', '\\', ':']) || !name.ends_with(".safetensors") {
                return Err(invalid(format!("invalid shard basename {name:?}")));
            }
            let mut file = File::open(directory.join(&name)).map_err(invalid)?;
            let length = file.metadata().map_err(invalid)?.len();
            let mut prefix = [0u8; 8];
            file.read_exact(&mut prefix).map_err(invalid)?;
            let header_length = u64::from_le_bytes(prefix);
            if header_length > HEADER_LIMIT
                || header_length.checked_add(8).is_none_or(|n| n > length)
            {
                return Err(invalid("invalid safetensors header length"));
            }
            let mut header = vec![0; header_length as usize];
            file.read_exact(&mut header).map_err(invalid)?;
            let metadata: Metadata = serde_json::from_slice(&header).map_err(invalid)?;
            let mut infos: Vec<_> = metadata.tensors().into_iter().collect();
            infos.sort_by_key(|(_, info)| info.data_offsets);
            let mut next = 0usize;
            let shard = reader.shards.len();
            for (tensor, info) in infos {
                let elements = info.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
                let bytes = elements.and_then(|n| n.checked_mul(info.dtype.size()));
                if info.data_offsets.0 != next
                    || info.data_offsets.1 < next
                    || bytes != Some(info.data_offsets.1 - next)
                {
                    return Err(invalid(format!("invalid tensor offsets: {tensor}")));
                }
                if !matches!(info.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
                    return Err(invalid(format!("unsupported tensor dtype: {tensor}")));
                }
                next = info.data_offsets.1;
                if weight_map
                    .as_ref()
                    .is_some_and(|map| map.get(&tensor) != Some(&name))
                {
                    return Err(invalid(format!(
                        "weight map disagrees with shard for {tensor}"
                    )));
                }
                if reader
                    .locations
                    .insert(
                        tensor.clone(),
                        Location {
                            shard,
                            info: info.clone(),
                        },
                    )
                    .is_some()
                {
                    return Err(invalid(format!("duplicate tensor: {tensor}")));
                }
            }
            if next as u64 != length - 8 - header_length {
                return Err(invalid("shard data size disagrees with header"));
            }
            reader.bytes_read += 8 + header_length;
            reader.shards.push(Shard {
                file,
                data_start: 8 + header_length,
            });
        }
        if weight_map
            .as_ref()
            .is_some_and(|map| map.len() != reader.locations.len())
        {
            return Err(invalid("weight map names missing tensors"));
        }
        let mut expected: HashSet<String> = ["model.embed_tokens.weight", "model.norm.weight"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for i in 0..config.num_hidden_layers {
            for suffix in LAYER_TENSORS {
                expected.insert(format!("model.layers.{i}.{suffix}"));
            }
        }
        if !config.tie_word_embeddings || reader.locations.contains_key("lm_head.weight") {
            expected.insert("lm_head.weight".into());
        }
        let actual: HashSet<_> = reader
            .locations
            .keys()
            .filter(|name| !name.ends_with(".rotary_emb.inv_freq"))
            .cloned()
            .collect();
        if expected != actual {
            return Err(invalid(
                "tensor names do not match the declared Llama block",
            ));
        }
        Ok(reader)
    }

    /// Bytes requested from source files, including headers. Page-cache hits
    /// are not distinguished from physical disk reads.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    pub fn shared_bytes(&mut self, limit: usize) -> Result<Vec<u8>, InferError> {
        let names = self
            .locations
            .keys()
            .filter(|n| {
                matches!(
                    n.as_str(),
                    "model.embed_tokens.weight" | "model.norm.weight" | "lm_head.weight"
                )
            })
            .cloned()
            .collect();
        self.pack(names, limit)
    }

    pub fn layer_bytes(&mut self, index: usize, limit: usize) -> Result<Vec<u8>, InferError> {
        let names = LAYER_TENSORS
            .iter()
            .map(|suffix| format!("model.layers.{index}.{suffix}"))
            .collect();
        self.pack(names, limit)
    }

    fn pack(&mut self, names: Vec<String>, limit: usize) -> Result<Vec<u8>, InferError> {
        let total = names.iter().try_fold(0usize, |total, name| {
            let location = self
                .locations
                .get(name)
                .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
            total
                .checked_add(location.info.data_offsets.1 - location.info.data_offsets.0)
                .ok_or_else(|| invalid("tensor byte count overflow"))
        })?;
        if total > limit {
            return Err(invalid(
                "artifact tensor data exceeds configured byte limit",
            ));
        }
        let mut buffers = Vec::new();
        for name in &names {
            let location = &self.locations[name];
            let info = &location.info;
            let shard = &mut self.shards[location.shard];
            shard
                .file
                .seek(SeekFrom::Start(
                    shard.data_start + info.data_offsets.0 as u64,
                ))
                .map_err(invalid)?;
            let mut bytes = vec![0; info.data_offsets.1 - info.data_offsets.0];
            shard.file.read_exact(&mut bytes).map_err(invalid)?;
            self.bytes_read += bytes.len() as u64;
            buffers.push(bytes);
        }
        let views = names
            .iter()
            .zip(&buffers)
            .map(|(name, bytes)| {
                let info = &self.locations[name].info;
                TensorView::new(info.dtype, info.shape.clone(), bytes)
                    .map(|view| (name.as_str(), view))
                    .map_err(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let packed = safetensors::serialize(views, &None).map_err(invalid)?;
        if packed.len() > limit {
            return Err(invalid("encoded artifact exceeds configured byte limit"));
        }
        Ok(packed)
    }
}
