// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Wrapper around HuggingFace's `tokenizers` crate, scoped to BERT
//! word-piece encoding for the embedding-provider use case.
//!
//! Loads from a tokenizer.json (the HF format) and exposes a small batch
//! interface returning padded `[batch, seq_len]` token ids and attention
//! masks ready for [`BertModel::forward_sentence`](super::model::BertModel).

use std::path::Path;

use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

use crate::embed::provider::EmbedError;

/// BERT word-piece tokenizer wrapper.
///
/// Configures the underlying `Tokenizer` with batch-aware padding (longest
/// item in the batch) and truncation at `max_seq_len`. Suitable for
/// sentence-transformers MiniLM (max 512).
#[derive(Debug)]
pub struct BertTokenizer {
    inner: Tokenizer,
}

/// One tokenized batch.
#[derive(Debug, Clone)]
pub struct EncodedBatch {
    /// `[batch, seq_len]` flattened row-major.
    pub input_ids: Vec<i64>,
    /// `[batch, seq_len]` flattened row-major. 1 = real token, 0 = padding.
    pub attention_mask: Vec<i64>,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl BertTokenizer {
    /// Load from a HuggingFace `tokenizer.json` file. Configures batch
    /// padding and truncation up to `max_seq_len`.
    pub fn from_file(
        path: impl AsRef<Path>,
        max_seq_len: usize,
        pad_token_id: u32,
    ) -> Result<Self, EmbedError> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| EmbedError::InvalidConfig(format!("tokenizer load failed: {e}")))?;
        Self::configure(inner, max_seq_len, pad_token_id)
    }

    /// Load from a HuggingFace `tokenizer.json` byte buffer. Useful when
    /// the tokenizer artifact is held in memory (e.g. resolved through
    /// a host store) rather than on the local filesystem. Configures the same
    /// batch padding and truncation as [`Self::from_file`].
    pub fn from_bytes(
        bytes: &[u8],
        max_seq_len: usize,
        pad_token_id: u32,
    ) -> Result<Self, EmbedError> {
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(|e| EmbedError::InvalidConfig(format!("tokenizer parse failed: {e}")))?;
        Self::configure(inner, max_seq_len, pad_token_id)
    }

    fn configure(
        mut inner: Tokenizer,
        max_seq_len: usize,
        pad_token_id: u32,
    ) -> Result<Self, EmbedError> {
        inner.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: pad_token_id,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));
        inner
            .with_truncation(Some(TruncationParams {
                max_length: max_seq_len,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| EmbedError::InvalidConfig(format!("tokenizer truncation: {e}")))?;
        Ok(Self { inner })
    }

    /// Encode a batch of texts. Adds special tokens ([CLS], [SEP]); pads
    /// to the longest item in the batch; truncates at the configured
    /// max_seq_len.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<EncodedBatch, EmbedError> {
        if texts.is_empty() {
            return Ok(EncodedBatch {
                input_ids: Vec::new(),
                attention_mask: Vec::new(),
                batch_size: 0,
                seq_len: 0,
            });
        }
        let inputs: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let encodings = self
            .inner
            .encode_batch(inputs, true)
            .map_err(|e| EmbedError::Backend(format!("tokenizer encode_batch: {e}")))?;

        let batch_size = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            // BatchLongest already padded to seq_len, so ids.len() == seq_len.
            for &id in ids {
                input_ids.push(id as i64);
            }
            for &m in mask {
                attention_mask.push(m as i64);
            }
        }
        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            batch_size,
            seq_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure the type compiles and exposes the expected surface — without
    /// loading a real tokenizer.json (which would need a fixture file). The
    /// runtime test of `from_file` + `encode_batch` belongs in the
    /// validation slice that ships a real MiniLM tokenizer.json.
    #[test]
    fn encoded_batch_default_shape() {
        let batch = EncodedBatch {
            input_ids: vec![],
            attention_mask: vec![],
            batch_size: 0,
            seq_len: 0,
        };
        assert_eq!(batch.batch_size, 0);
        assert_eq!(batch.seq_len, 0);
        assert!(batch.input_ids.is_empty());
    }

    #[test]
    fn from_file_errors_on_missing_path() {
        let result = BertTokenizer::from_file("this_file_does_not_exist_anywhere.json", 512, 0);
        assert!(matches!(result, Err(EmbedError::InvalidConfig(_))));
    }
}
