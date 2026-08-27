// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! ESP is Mere's portable model-execution boundary.
//!
//! [`infer`] owns text generation, streaming, and model capability matching.
//! [`embed`] owns embeddings, vector retrieval, and affinity computation.
//! Heavy model backends remain feature-gated; the default build is the two
//! dependency-light contracts and their deterministic test providers.

pub mod embed;
pub mod infer;
