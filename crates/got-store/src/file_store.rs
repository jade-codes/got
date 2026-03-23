// ---------------------------------------------------------------------------
// FileStore — file-system-backed attestation store (PoC persistence)
// ---------------------------------------------------------------------------
//
// Layout:
//   <root>/attestations/<sha256_hex>.json   — one file per attestation
//   <root>/meta/<sha256_hex>.json           — signer hash for each attestation
//   <root>/index.json                       — model_id → [content_hash, ...] mapping
//
// The store is append-only: files are written, never modified or deleted.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ed25519_dalek::VerifyingKey;
use got_attest::verify;
use got_core::GeometricAttestation;
use serde::{Deserialize, Serialize};

use crate::audit::{build_audit_report, AuditReport};
use crate::store::{attestation_store_id, AttestationStore, StoreError, StoreFilter, StoreId};

/// Metadata stored alongside each attestation (signer info).
#[derive(Serialize, Deserialize)]
struct MetaEntry {
    signer_hash: String, // hex-encoded [u8; 32]
}

/// Index mapping model_id → list of content hashes (hex-encoded).
#[derive(Serialize, Deserialize, Default)]
struct Index {
    models: HashMap<String, Vec<String>>,
}

/// File-system-backed attestation store.
pub struct FileStore {
    root: PathBuf,
    /// In-memory index (also persisted to disk).
    index: Index,
    /// In-memory cache of loaded attestations.
    cache: HashMap<StoreId, (GeometricAttestation, [u8; 32])>,
    /// Insertion order (content hashes).
    order: Vec<StoreId>,
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(s: &str) -> Result<[u8; 32], StoreError> {
    if s.len() != 64 {
        return Err(StoreError::Internal(format!(
            "expected 64 hex chars, got {}",
            s.len()
        )));
    }
    let bytes: Vec<u8> = (0..32)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| StoreError::Internal(e.to_string()))?;
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Atomic write: write to a temporary file then rename into place.
/// This avoids leaving a truncated file if the process crashes mid-write.
fn atomic_write(path: &Path, data: &[u8]) -> Result<(), StoreError> {
    let parent = path.parent().unwrap_or(Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .map_err(|e| StoreError::Internal(format!("tmpfile: {e}")))?;
    tmp.write_all(data)
        .map_err(|e| StoreError::Internal(format!("tmpfile write: {e}")))?;
    tmp.flush()
        .map_err(|e| StoreError::Internal(format!("tmpfile flush: {e}")))?;
    tmp.persist(path)
        .map_err(|e| StoreError::Internal(format!("tmpfile persist: {e}")))?;
    Ok(())
}

impl FileStore {
    /// Open or create a file store at the given directory path.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("attestations"))?;
        fs::create_dir_all(root.join("meta"))?;

        // Load or create index.
        let index_path = root.join("index.json");
        let index: Index = if index_path.exists() {
            let data = fs::read_to_string(&index_path)?;
            serde_json::from_str(&data)
                .map_err(|e| StoreError::Internal(format!("index parse error: {e}")))?
        } else {
            Index::default()
        };

        // Rebuild in-memory cache from disk.
        let mut cache = HashMap::new();
        let mut order = Vec::new();

        // Collect all content hashes from the index (preserving order).
        let mut all_hashes: Vec<String> = Vec::new();
        // Sort model keys for deterministic load order.
        let mut model_ids: Vec<&String> = index.models.keys().collect();
        model_ids.sort();
        for mid in &model_ids {
            for h in index.models.get(*mid).unwrap() {
                if !all_hashes.contains(h) {
                    all_hashes.push(h.clone());
                }
            }
        }

        for hex_hash in &all_hashes {
            let id = hex_decode(hex_hash)?;
            let att_path = root.join("attestations").join(format!("{hex_hash}.json"));
            let meta_path = root.join("meta").join(format!("{hex_hash}.json"));

            if att_path.exists() && meta_path.exists() {
                let att_data = fs::read_to_string(&att_path)?;
                let att: GeometricAttestation = serde_json::from_str(&att_data)
                    .map_err(|e| StoreError::Internal(format!("attestation parse: {e}")))?;
                let meta_data = fs::read_to_string(&meta_path)?;
                let meta: MetaEntry = serde_json::from_str(&meta_data)
                    .map_err(|e| StoreError::Internal(format!("meta parse: {e}")))?;
                let signer_hash = hex_decode(&meta.signer_hash)?;

                // S-16: Integrity check — verify that the content hash of the
                // loaded attestation matches the expected store ID (filename).
                // If someone tampered with the JSON, the hash will differ.
                let computed_id = attestation_store_id(&att)?;
                if computed_id != id {
                    return Err(StoreError::Internal(format!(
                        "integrity check failed for {hex_hash}: content hash mismatch (file was tampered)"
                    )));
                }

                cache.insert(id, (att, signer_hash));
                order.push(id);
            }
        }

        Ok(Self {
            root,
            index,
            cache,
            order,
        })
    }

    /// Persist the index to disk (atomic write).
    fn write_index(&self) -> Result<(), StoreError> {
        let data = serde_json::to_string_pretty(&self.index)
            .map_err(|e| StoreError::Serialisation(e.to_string()))?;
        atomic_write(&self.root.join("index.json"), data.as_bytes())
    }

    /// Write an attestation and its metadata to disk (atomic writes).
    fn write_attestation(
        &self,
        id: &StoreId,
        attestation: &GeometricAttestation,
        signer_hash: &[u8; 32],
    ) -> Result<(), StoreError> {
        let hex = hex_encode(id);

        let att_data = serde_json::to_string_pretty(attestation)
            .map_err(|e| StoreError::Serialisation(e.to_string()))?;
        atomic_write(
            &self.root.join("attestations").join(format!("{hex}.json")),
            att_data.as_bytes(),
        )?;

        let meta = MetaEntry {
            signer_hash: hex_encode(signer_hash),
        };
        let meta_data = serde_json::to_string_pretty(&meta)
            .map_err(|e| StoreError::Serialisation(e.to_string()))?;
        atomic_write(
            &self.root.join("meta").join(format!("{hex}.json")),
            meta_data.as_bytes(),
        )
    }
}

impl AttestationStore for FileStore {
    fn append(
        &mut self,
        attestation: &GeometricAttestation,
        verifying_key: &VerifyingKey,
    ) -> Result<StoreId, StoreError> {
        // 1. Verify signature.
        verify(attestation, verifying_key).map_err(|e| match e {
            got_attest::AttestationError::SignatureInvalid => StoreError::InvalidSignature,
            other => StoreError::Serialisation(other.to_string()),
        })?;

        // 2. Content-addressed ID.
        let id = attestation_store_id(attestation)?;

        // 3. Idempotent.
        if self.cache.contains_key(&id) {
            return Ok(id);
        }

        // 4. Chain validation.
        if let Some(parent_hash) = attestation.parent_attestation_hash {
            if !self.cache.contains_key(&parent_hash) {
                let hex = hex_encode(&parent_hash);
                return Err(StoreError::OrphanedAttestation(hex));
            }
        }

        // 5. Store to disk.
        let signer_hash = got_core::sha256(verifying_key.as_bytes());
        self.write_attestation(&id, attestation, &signer_hash)?;

        // 6. Update index.
        let hex = hex_encode(&id);
        self.index
            .models
            .entry(attestation.model_id.clone())
            .or_default()
            .push(hex);
        self.write_index()?;

        // 7. Update in-memory cache.
        self.cache.insert(id, (attestation.clone(), signer_hash));
        self.order.push(id);

        Ok(id)
    }

    fn get(&self, id: &StoreId) -> Option<&GeometricAttestation> {
        self.cache.get(id).map(|(a, _)| a)
    }

    fn chain(&self, model_id: &str) -> Vec<&GeometricAttestation> {
        let mut chain: Vec<&GeometricAttestation> = self
            .order
            .iter()
            .filter_map(|id| self.cache.get(id))
            .filter(|(a, _)| a.model_id == model_id)
            .map(|(a, _)| a)
            .collect();
        chain.sort_by_key(|a| a.timestamp);
        chain
    }

    fn query(&self, filter: &StoreFilter) -> Vec<&GeometricAttestation> {
        self.order
            .iter()
            .filter_map(|id| self.cache.get(id))
            .filter(|(a, sh)| filter.matches(a, sh))
            .map(|(a, _)| a)
            .collect()
    }

    fn audit(&self, model_id: &str) -> AuditReport {
        let chain = self.chain(model_id);
        let signer_hashes: Vec<[u8; 32]> = self
            .order
            .iter()
            .filter_map(|id| self.cache.get(id))
            .filter(|(a, _)| a.model_id == model_id)
            .map(|(_, sh)| *sh)
            .collect();
        build_audit_report(model_id, &chain, &signer_hashes)
    }

    fn len(&self) -> usize {
        self.cache.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoreFilter;
    use ed25519_dalek::{Signer, SigningKey};
    use got_attest::serialise_for_signing;
    use got_core::{
        CausalScoreRecord, GeometricAttestation, InnerProduct, Precision, SCHEMA_VERSION_3,
    };

    fn test_key() -> SigningKey {
        SigningKey::from_bytes(&[0x42; 32])
    }

    fn make_attestation(
        key: &SigningKey,
        model_id: &str,
        timestamp: u64,
        parent: Option<[u8; 32]>,
        drift: Option<f32>,
        causal_flag: Option<bool>,
        causal_scores: Vec<CausalScoreRecord>,
    ) -> GeometricAttestation {
        let mut a = GeometricAttestation {
            schema_version: SCHEMA_VERSION_3,
            model_id: model_id.to_string(),
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xBB; 32],
            timestamp,
            corpus_version: "test-v1".to_string(),
            probe_version: "probe-v1".to_string(),
            layer_readings: vec![vec![0.5, 0.6]],
            confidence: vec![0.9, 0.8],
            coverage_flags: vec![false, false],
            divergence_flag: false,
            parent_attestation_hash: parent,
            geometry_hash: Some([0xCC; 32]),
            geometry_drift: drift,
            causal_scores,
            intervention_delta: Some(0.1),
            causal_flag,
            sequence_number: 0,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            signature: [0u8; 64],
        };
        let payload = serialise_for_signing(&a).unwrap();
        let sig = key.sign(&payload);
        a.signature = sig.to_bytes();
        a
    }

    fn content_hash(a: &GeometricAttestation) -> [u8; 32] {
        attestation_store_id(a).unwrap()
    }

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("got-store-test-{}-{}", std::process::id(), id));
        // Clean up from prior runs.
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn file_store_append_and_retrieve() {
        let root = temp_dir().join("append_retrieve");
        let key = test_key();
        let a = make_attestation(&key, "model-a", 1000, None, None, None, vec![]);
        {
            let mut store = FileStore::open(&root).unwrap();
            let id = store.append(&a, &key.verifying_key()).unwrap();
            let got = store.get(&id).unwrap();
            assert_eq!(got.model_id, "model-a");
        }
        // Reopen — data should still be there.
        {
            let store = FileStore::open(&root).unwrap();
            assert_eq!(store.len(), 1);
            let chain = store.chain("model-a");
            assert_eq!(chain.len(), 1);
            assert_eq!(chain[0].model_id, "model-a");
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_store_duplicate_idempotent() {
        let root = temp_dir().join("dup");
        let key = test_key();
        let a = make_attestation(&key, "m", 1000, None, None, None, vec![]);
        let mut store = FileStore::open(&root).unwrap();
        let id1 = store.append(&a, &key.verifying_key()).unwrap();
        let id2 = store.append(&a, &key.verifying_key()).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_store_rejects_invalid_signature() {
        let root = temp_dir().join("invalid_sig");
        let key = test_key();
        let mut a = make_attestation(&key, "m", 1000, None, None, None, vec![]);
        a.timestamp = 9999;
        let mut store = FileStore::open(&root).unwrap();
        assert!(matches!(
            store.append(&a, &key.verifying_key()),
            Err(StoreError::InvalidSignature)
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_store_rejects_orphan() {
        let root = temp_dir().join("orphan");
        let key = test_key();
        let a = make_attestation(&key, "m", 1000, Some([0xFF; 32]), None, None, vec![]);
        let mut store = FileStore::open(&root).unwrap();
        assert!(matches!(
            store.append(&a, &key.verifying_key()),
            Err(StoreError::OrphanedAttestation(_))
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_store_chain_across_reopen() {
        let root = temp_dir().join("chain_reopen");
        let key = test_key();
        let a0 = make_attestation(&key, "m", 1000, None, Some(0.01), None, vec![]);
        let a0_hash = content_hash(&a0);
        let a1 = make_attestation(&key, "m", 2000, Some(a0_hash), Some(0.02), None, vec![]);

        {
            let mut store = FileStore::open(&root).unwrap();
            store.append(&a0, &key.verifying_key()).unwrap();
            store.append(&a1, &key.verifying_key()).unwrap();
        }
        {
            let store = FileStore::open(&root).unwrap();
            assert_eq!(store.len(), 2);
            let chain = store.chain("m");
            assert_eq!(chain.len(), 2);
            assert_eq!(chain[0].timestamp, 1000);
            assert_eq!(chain[1].timestamp, 2000);

            let report = store.audit("m");
            assert_eq!(report.total_attestations, 2);
            assert_eq!(report.chain_length, 2);
            assert!(report.chain_valid);
            assert_eq!(report.drift_summary.readings_with_drift, 2);
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_store_query_by_model_and_time() {
        let root = temp_dir().join("query");
        let key = test_key();
        let a = make_attestation(&key, "model-a", 1000, None, None, None, vec![]);
        let b = make_attestation(&key, "model-b", 2000, None, None, None, vec![]);
        let c = make_attestation(&key, "model-a", 3000, None, None, None, vec![]);

        let mut store = FileStore::open(&root).unwrap();
        store.append(&a, &key.verifying_key()).unwrap();
        store.append(&b, &key.verifying_key()).unwrap();
        store.append(&c, &key.verifying_key()).unwrap();

        let results = store.query(&StoreFilter::new().model_id("model-a"));
        assert_eq!(results.len(), 2);

        let results = store.query(&StoreFilter::new().after(1500));
        assert_eq!(results.len(), 2);

        let results = store.query(&StoreFilter::new().model_id("model-a").after(1500));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, 3000);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn file_store_matches_memory_store() {
        use crate::MemoryStore;

        let root = temp_dir().join("cross");
        let key = test_key();
        let a0 = make_attestation(
            &key,
            "m",
            1000,
            None,
            Some(0.01),
            Some(true),
            vec![CausalScoreRecord {
                delta_plus: 0.5,
                delta_minus: 0.4,
                consistency: 0.9,
                is_causal: true,
            }],
        );
        let a0_hash = content_hash(&a0);
        let a1 = make_attestation(
            &key,
            "m",
            2000,
            Some(a0_hash),
            Some(0.02),
            Some(true),
            vec![CausalScoreRecord {
                delta_plus: 0.6,
                delta_minus: 0.3,
                consistency: 0.8,
                is_causal: true,
            }],
        );

        let mut mem = MemoryStore::new();
        let mut file = FileStore::open(&root).unwrap();

        let vk = key.verifying_key();
        mem.append(&a0, &vk).unwrap();
        mem.append(&a1, &vk).unwrap();
        file.append(&a0, &vk).unwrap();
        file.append(&a1, &vk).unwrap();

        // Same chain.
        let mc = mem.chain("m");
        let fc = file.chain("m");
        assert_eq!(mc.len(), fc.len());
        for (m, f) in mc.iter().zip(fc.iter()) {
            assert_eq!(m.timestamp, f.timestamp);
            assert_eq!(m.model_id, f.model_id);
        }

        // Same audit.
        let mr = mem.audit("m");
        let fr = file.audit("m");
        assert_eq!(mr.total_attestations, fr.total_attestations);
        assert_eq!(mr.chain_length, fr.chain_length);
        assert_eq!(mr.chain_valid, fr.chain_valid);
        assert_eq!(
            mr.drift_summary.readings_with_drift,
            fr.drift_summary.readings_with_drift
        );
        assert_eq!(
            mr.causal_summary.causal_pass_count,
            fr.causal_summary.causal_pass_count
        );

        let _ = fs::remove_dir_all(root);
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issues 27, 37)
    // -----------------------------------------------------------------------

    /// Issue #27 (S-6): FileStore uses atomic writes (write-to-temp + rename).
    /// This test writes an attestation and verifies the file is valid JSON.
    #[test]
    fn sec_file_store_writes_valid_json() {
        let root = temp_dir().join("atomic_write");
        let key = test_key();
        let a = make_attestation(&key, "m", 1000, None, None, None, vec![]);

        {
            let mut store = FileStore::open(&root).unwrap();
            store.append(&a, &key.verifying_key()).unwrap();
        }

        // Verify all .json files in the attestation directory are valid JSON.
        let att_dir = root.join("attestations");
        for entry in fs::read_dir(&att_dir).unwrap() {
            let entry = entry.unwrap();
            let data = fs::read_to_string(entry.path()).unwrap();
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&data);
            assert!(
                parsed.is_ok(),
                "attestation JSON at {:?} is not valid: {:?}",
                entry.path(),
                parsed.err()
            );
        }

        let _ = fs::remove_dir_all(root);
    }

    /// Issue #37 (S-16): FileStore verifies content-hash integrity on load.
    /// A tampered JSON file must be rejected.
    #[test]
    fn sec_file_store_rejects_tampered_json_on_reload() {
        let root = temp_dir().join("tamper_reload");
        let key = test_key();
        let a = make_attestation(&key, "m", 1000, None, None, None, vec![]);

        // Write the attestation to disk.
        {
            let mut store = FileStore::open(&root).unwrap();
            store.append(&a, &key.verifying_key()).unwrap();
        }

        // Tamper with the stored attestation JSON: change the timestamp.
        let att_dir = root.join("attestations");
        for entry in fs::read_dir(&att_dir).unwrap() {
            let path = entry.unwrap().path();
            let data = fs::read_to_string(&path).unwrap();
            let tampered = data.replace("1000", "9999");
            fs::write(&path, tampered).unwrap();
        }

        // Reopen the store: integrity check should reject the tampered file.
        let result = FileStore::open(&root);
        assert!(result.is_err(), "FileStore must reject tampered attestation on reload");

        let _ = fs::remove_dir_all(root);
    }
}
