// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Save/load [`VectorIndex`] through eidetic's Layer 3 typed-payload API.
//!
//! The serialization format is JSON via serde (human-inspectable, stable
//! across schema additions). Migration to a binary format (rkyv,
//! bincode/postcard) is a follow-up if size or speed become real
//! constraints — the per-schema serializer is chosen by the
//! [`TypedPayload`] impl for `VectorIndex<K>` below.
//!
//! ## Phase 6 — Layer 3 migration
//!
//! Earlier versions used `Store::save_blob` / `Store::load_blob` directly
//! with a hand-picked string key. This module now goes through
//! [`eidetic::save_typed`] / [`eidetic::load_typed`], producing manifests
//! with content hash, schema reference, and three-axis classification.
//! Callers receive a [`ManifestId`] (BLAKE3 of the serialized index) and
//! resolve back through the typed-payload API.

use std::hash::Hash;

use eidetic::{
    BlobFetcher, BlobManifest, BlobSource, Hash as ContentHash, ManifestId, PrivacyClass,
    ProvenanceRecord, SchemaRef, Timestamp, TrustEnvelope, TypedPayload, list_typed, load_typed,
    save_typed,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{SparseIndex, SparseVector, VectorIndex};

/// Canonical bytes of the `VectorIndex` schema codicil payload.
///
/// Mere-native schema declaring the persisted vector-index shape. Stable
/// across instances; its BLAKE3 hash is [`VECTOR_INDEX_SCHEMA_REF`].
const VECTOR_INDEX_SCHEMA_PAYLOAD: &[u8] = br#"{"format":"mere-native","schema_id":"embed.VectorIndex/v1","body":{"version":1,"description":"Persisted vector index over embeddings: dimensions, metric, and key->vector entries.","required":["dimensions","metric","entries"],"fields":{"dimensions":{"type":"u64"},"metric":{"type":"string"},"entries":{"type":"object"}}}}"#;

/// Schema reference for `VectorIndex` codicils.
///
/// Computed lazily from [`VECTOR_INDEX_SCHEMA_PAYLOAD`] at first use.
pub fn vector_index_schema_ref() -> SchemaRef {
    SchemaRef::from_id(ManifestId::from_hash(ContentHash::of(
        VECTOR_INDEX_SCHEMA_PAYLOAD,
    )))
}

/// The well-known schema reference for persisted vector indices.
///
/// All `VectorIndex<K>` codicils use this schema regardless of `K` — the
/// key type is data-level (varies per use case), not schema-level.
pub static VECTOR_INDEX_SCHEMA_REF: std::sync::LazyLock<SchemaRef> =
    std::sync::LazyLock::new(vector_index_schema_ref);

/// Canonical bytes of the `SparseIndex` schema codicil payload.
///
/// A separate schema, not a version bump on the dense one: the entry shape
/// differs (a list of `(bucket, weight)` pairs plus the vector's dimension,
/// not a dense array), so a reader that resolves this reference knows what it
/// is holding without inspecting the data.
const SPARSE_INDEX_SCHEMA_PAYLOAD: &[u8] = br#"{"format":"mere-native","schema_id":"embed.SparseIndex/v1","body":{"version":1,"description":"Persisted sparse vector index over embeddings: dimensions, metric, and key->sparse-vector entries, each vector holding its non-zero (index, weight) pairs ascending by index.","required":["dimensions","metric","entries"],"fields":{"dimensions":{"type":"u64"},"metric":{"type":"string"},"entries":{"type":"object"}}}}"#;

/// Schema reference for `SparseIndex` codicils, from
/// [`SPARSE_INDEX_SCHEMA_PAYLOAD`].
pub fn sparse_index_schema_ref() -> SchemaRef {
    SchemaRef::from_id(ManifestId::from_hash(ContentHash::of(
        SPARSE_INDEX_SCHEMA_PAYLOAD,
    )))
}

/// The well-known schema reference for persisted sparse vector indices.
pub static SPARSE_INDEX_SCHEMA_REF: std::sync::LazyLock<SchemaRef> =
    std::sync::LazyLock::new(sparse_index_schema_ref);

/// The schema a stored representation declares. One per `V` in
/// `VectorIndex<K, V>`; the key type is data-level and does not vary it.
pub trait IndexSchema {
    fn schema_ref() -> SchemaRef;
}

impl IndexSchema for Vec<f32> {
    fn schema_ref() -> SchemaRef {
        *VECTOR_INDEX_SCHEMA_REF
    }
}

impl IndexSchema for SparseVector {
    fn schema_ref() -> SchemaRef {
        *SPARSE_INDEX_SCHEMA_REF
    }
}

/// A local newtype carrying the [`TypedPayload`] schema for a [`VectorIndex`].
///
/// `VectorIndex` (sibylla) and `TypedPayload` (eidetic) are both foreign to this
/// crate, so the impl cannot sit on `VectorIndex` directly — the orphan rule
/// forbids it. `TypedPayload` requires `Serialize + DeserializeOwned`, so the
/// wrapper owns its index (a borrowing form cannot deserialize). It is
/// `#[serde(transparent)]`, so it serializes byte-identically to the raw index —
/// the persisted format and its schema are unchanged. Save clones into it (saves
/// are infrequent; the bytes are identical).
#[derive(Serialize, serde::Deserialize)]
#[serde(transparent)]
struct PersistedIndex<K: Hash + Eq + Clone, V>(VectorIndex<K, V>);

impl<K, V> TypedPayload for PersistedIndex<K, V>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
    V: IndexSchema + Serialize + DeserializeOwned,
{
    fn schema_ref() -> SchemaRef {
        V::schema_ref()
    }
}

/// Save a vector index through Layer 3, returning the manifest id of the
/// produced codicil. Callers that need to recover the index later persist
/// the returned id (typically alongside the graph view it indexes).
pub async fn save_to_eidetic<K>(
    store: &mut dyn eidetic::Store,
    index: &VectorIndex<K>,
    privacy: PrivacyClass,
    provenance: ProvenanceRecord,
    trust: TrustEnvelope,
    created_at: Timestamp,
) -> eidetic::Result<ManifestId>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
{
    save_index(store, index, privacy, provenance, trust, created_at).await
}

/// [`save_to_eidetic`] for a [`SparseIndex`]. Its own codicil schema
/// ([`SPARSE_INDEX_SCHEMA_REF`]), so a mistyped read is caught by the same
/// schema check that guards the dense one.
pub async fn save_sparse_to_eidetic<K>(
    store: &mut dyn eidetic::Store,
    index: &SparseIndex<K>,
    privacy: PrivacyClass,
    provenance: ProvenanceRecord,
    trust: TrustEnvelope,
    created_at: Timestamp,
) -> eidetic::Result<ManifestId>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
{
    save_index(store, index, privacy, provenance, trust, created_at).await
}

async fn save_index<K, V>(
    store: &mut dyn eidetic::Store,
    index: &VectorIndex<K, V>,
    privacy: PrivacyClass,
    provenance: ProvenanceRecord,
    trust: TrustEnvelope,
    created_at: Timestamp,
) -> eidetic::Result<ManifestId>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
    V: IndexSchema + Clone + Serialize + DeserializeOwned,
{
    save_typed(
        store,
        &PersistedIndex(index.clone()),
        Vec::<BlobSource>::new(),
        privacy,
        provenance,
        trust,
        created_at,
    )
    .await
}

/// List the manifests of every stored vector index. Ordering is the store's;
/// callers pick (typically the newest by `created_at`).
///
/// This is the listing counterpart to [`save_to_eidetic`] / [`load_from_eidetic`],
/// and it has to live here: `eidetic::list_typed` is generic over the
/// [`TypedPayload`] that carries the schema, and the only type carrying
/// `VectorIndex`'s schema is the private `PersistedIndex` newtype above. A caller
/// outside this module cannot name it, so it cannot list without this.
pub async fn list_from_eidetic<K>(
    store: &mut dyn eidetic::Store,
) -> eidetic::Result<Vec<BlobManifest>>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
{
    list_typed::<PersistedIndex<K, Vec<f32>>>(store).await
}

/// [`list_from_eidetic`] over the sparse schema.
pub async fn list_sparse_from_eidetic<K>(
    store: &mut dyn eidetic::Store,
) -> eidetic::Result<Vec<BlobManifest>>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
{
    list_typed::<PersistedIndex<K, SparseVector>>(store).await
}

/// Load a vector index by its manifest id. Returns `Ok(None)` if no
/// manifest is stored under that id; returns an error if the manifest's
/// schema doesn't match `VectorIndex` (mistyped read) or if the resolved
/// blob fails BLAKE3 verification.
pub async fn load_from_eidetic<K>(
    store: &mut dyn eidetic::Store,
    fetcher: &mut dyn BlobFetcher,
    id: ManifestId,
) -> eidetic::Result<Option<VectorIndex<K>>>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
{
    Ok(load_typed::<PersistedIndex<K, Vec<f32>>>(store, fetcher, id)
        .await?
        .map(|persisted| persisted.0))
}

/// [`load_from_eidetic`] for a [`SparseIndex`]. A dense manifest id fails the
/// schema check here, and a sparse one fails it there.
pub async fn load_sparse_from_eidetic<K>(
    store: &mut dyn eidetic::Store,
    fetcher: &mut dyn BlobFetcher,
    id: ManifestId,
) -> eidetic::Result<Option<SparseIndex<K>>>
where
    K: Hash + Eq + Clone + Serialize + DeserializeOwned,
{
    Ok(
        load_typed::<PersistedIndex<K, SparseVector>>(store, fetcher, id)
            .await?
            .map(|persisted| persisted.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::SimilarityMetric;
    use eidetic::{ModerationState, ProvenanceOrigin, TrustLevel};

    // The in-memory store is muniment's (2026-07-12): the hand-rolled
    // one was the same map behind the same seam.
    use muniment::Backend as _;
    use muniment::MemoryBackend as InMemoryStore;

    fn make_index() -> VectorIndex<u32> {
        let mut idx = VectorIndex::<u32>::new(3, SimilarityMetric::Cosine);
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.5, 0.5, 0.0]).unwrap();
        idx
    }

    fn test_provenance() -> ProvenanceRecord {
        ProvenanceRecord {
            origin: ProvenanceOrigin::Generated,
            upstream: Vec::new(),
            tooling: Some("embed-test".to_string()),
            generated_at: Timestamp(0),
        }
    }

    fn test_trust() -> TrustEnvelope {
        TrustEnvelope {
            level: TrustLevel::SelfAsserted,
            signatures: Vec::new(),
            moderation_state: ModerationState::Unreviewed,
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        pollster::block_on(async {
            let mut store = InMemoryStore::default();
            let mut fetcher = eidetic::NoFetcher;
            let original = make_index();

            let id = save_to_eidetic(
                &mut store,
                &original,
                PrivacyClass::LocalOnly,
                test_provenance(),
                test_trust(),
                Timestamp(0),
            )
            .await
            .unwrap();

            let loaded: VectorIndex<u32> = load_from_eidetic(&mut store, &mut fetcher, id)
                .await
                .unwrap()
                .expect("manifest present after save");

            assert_eq!(loaded.dimensions(), original.dimensions());
            assert_eq!(loaded.metric(), original.metric());
            assert_eq!(loaded.len(), original.len());
            for (k, v) in original.iter() {
                assert_eq!(loaded.get(k), Some(v));
            }
        });
    }

    #[test]
    fn load_missing_returns_none() {
        pollster::block_on(async {
            let mut store = InMemoryStore::default();
            let mut fetcher = eidetic::NoFetcher;
            let unknown = ManifestId::of_blob(b"never-saved-vector-index");
            let loaded: Option<VectorIndex<u32>> =
                load_from_eidetic(&mut store, &mut fetcher, unknown)
                    .await
                    .unwrap();
            assert!(loaded.is_none());
        });
    }

    #[test]
    fn save_overwrites_via_content_addressing() {
        pollster::block_on(async {
            // With content-addressed ids, "overwrite" is automatic — saving
            // the same index twice produces the same id; saving a different
            // index produces a different id and the original remains.
            let mut store = InMemoryStore::default();
            let mut fetcher = eidetic::NoFetcher;

            let first = make_index();
            let id_first = save_to_eidetic(
                &mut store,
                &first,
                PrivacyClass::LocalOnly,
                test_provenance(),
                test_trust(),
                Timestamp(0),
            )
            .await
            .unwrap();

            let mut second = first.clone();
            second.insert(99, vec![0.0, 0.0, 1.0]).unwrap();
            let id_second = save_to_eidetic(
                &mut store,
                &second,
                PrivacyClass::LocalOnly,
                test_provenance(),
                test_trust(),
                Timestamp(0),
            )
            .await
            .unwrap();

            assert_ne!(id_first, id_second, "different content -> different ids");

            let loaded_second: VectorIndex<u32> =
                load_from_eidetic(&mut store, &mut fetcher, id_second)
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(loaded_second.len(), 4);
            assert!(loaded_second.contains(&99));

            // Original is still loadable — it wasn't deleted.
            let loaded_first: VectorIndex<u32> =
                load_from_eidetic(&mut store, &mut fetcher, id_first)
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(loaded_first.len(), 3);
        });
    }

    #[test]
    fn empty_index_roundtrips() {
        pollster::block_on(async {
            let mut store = InMemoryStore::default();
            let mut fetcher = eidetic::NoFetcher;
            let empty: VectorIndex<u32> = VectorIndex::new(8, SimilarityMetric::DotProduct);
            let id = save_to_eidetic(
                &mut store,
                &empty,
                PrivacyClass::LocalOnly,
                test_provenance(),
                test_trust(),
                Timestamp(0),
            )
            .await
            .unwrap();
            let loaded: VectorIndex<u32> = load_from_eidetic(&mut store, &mut fetcher, id)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(loaded.dimensions(), 8);
            assert_eq!(loaded.metric(), SimilarityMetric::DotProduct);
            assert!(loaded.is_empty());
        });
    }

    #[test]
    fn schema_ref_is_stable() {
        // Two derivations agree (no environmental input).
        assert_eq!(vector_index_schema_ref(), *VECTOR_INDEX_SCHEMA_REF);
        assert_eq!(sparse_index_schema_ref(), *SPARSE_INDEX_SCHEMA_REF);
        // And the two representations are not the same schema.
        assert_ne!(*VECTOR_INDEX_SCHEMA_REF, *SPARSE_INDEX_SCHEMA_REF);
    }

    fn make_sparse_index() -> SparseIndex<u32> {
        let provider = crate::embed::LexicalEmbeddingProvider::new(256).unwrap();
        let mut idx = SparseIndex::<u32>::new(256, SimilarityMetric::Cosine);
        for (key, text) in [(1u32, "rust async runtime"), (2, "italian dinner"), (3, "")] {
            idx.insert(key, provider.embed_sparse_one(text)).unwrap();
        }
        idx
    }

    #[test]
    fn sparse_save_and_load_roundtrip() {
        pollster::block_on(async {
            let mut store = InMemoryStore::default();
            let mut fetcher = eidetic::NoFetcher;
            let original = make_sparse_index();

            let id = save_sparse_to_eidetic(
                &mut store,
                &original,
                PrivacyClass::LocalOnly,
                test_provenance(),
                test_trust(),
                Timestamp(0),
            )
            .await
            .unwrap();

            let listed = list_sparse_from_eidetic::<u32>(&mut store).await.unwrap();
            assert!(listed.iter().any(|m| m.id == id));

            let loaded: SparseIndex<u32> = load_sparse_from_eidetic(&mut store, &mut fetcher, id)
                .await
                .unwrap()
                .expect("manifest present after save");
            assert_eq!(loaded.dimensions(), original.dimensions());
            assert_eq!(loaded.metric(), original.metric());
            assert_eq!(loaded.len(), original.len());
            for (k, v) in original.iter() {
                assert_eq!(loaded.get(k), Some(v));
            }
            // Ranking survives the round trip, which is the point of storing it.
            let query = crate::embed::LexicalEmbeddingProvider::new(256)
                .unwrap()
                .embed_sparse_one("rust runtime");
            assert_eq!(
                loaded.nearest(&query, 3).unwrap(),
                original.nearest(&query, 3).unwrap()
            );
        });
    }

    #[test]
    fn a_sparse_manifest_is_not_readable_as_a_dense_one() {
        pollster::block_on(async {
            let mut store = InMemoryStore::default();
            let mut fetcher = eidetic::NoFetcher;
            let id = save_sparse_to_eidetic(
                &mut store,
                &make_sparse_index(),
                PrivacyClass::LocalOnly,
                test_provenance(),
                test_trust(),
                Timestamp(0),
            )
            .await
            .unwrap();
            let mistyped: eidetic::Result<Option<VectorIndex<u32>>> =
                load_from_eidetic(&mut store, &mut fetcher, id).await;
            assert!(mistyped.is_err(), "separate schemas must catch the mix-up");
            // And the listings do not bleed into one another.
            assert!(list_from_eidetic::<u32>(&mut store).await.unwrap().is_empty());
        });
    }

    #[test]
    fn corrupted_blob_returns_error() {
        pollster::block_on(async {
            // Save a valid index, then tamper with the underlying blob
            // bytes. The Layer-2 BLAKE3 check rejects the tampered bytes
            // before deserialization.
            let mut store = InMemoryStore::default();
            let mut fetcher = eidetic::NoFetcher;
            let index = make_index();
            let id = save_to_eidetic(
                &mut store,
                &index,
                PrivacyClass::LocalOnly,
                test_provenance(),
                test_trust(),
                Timestamp(0),
            )
            .await
            .unwrap();

            let blob_key = format!("blob:{}", id.0.to_hex());
            store
                .put(&blob_key, b"not valid index bytes")
                .await
                .unwrap();

            let result: eidetic::Result<Option<VectorIndex<u32>>> =
                load_from_eidetic(&mut store, &mut fetcher, id).await;
            assert!(result.is_err());
        });
    }
}
