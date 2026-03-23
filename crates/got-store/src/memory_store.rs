// ---------------------------------------------------------------------------
// MemoryStore — in-memory attestation store (for testing)
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use ed25519_dalek::VerifyingKey;
use got_attest::verify;
use got_core::GeometricAttestation;

use crate::audit::{build_audit_report, AuditReport};
use crate::store::{attestation_store_id, AttestationStore, StoreError, StoreFilter, StoreId};

/// Entry in the memory store: attestation + signer key hash.
struct Entry {
    attestation: GeometricAttestation,
    signer_hash: [u8; 32],
}

/// In-memory implementation of `AttestationStore`.
///
/// Append-only. No persistence. Suitable for tests and short-lived pipelines.
pub struct MemoryStore {
    /// Content-hash → entry.
    entries: HashMap<StoreId, Entry>,
    /// Insertion order for deterministic iteration.
    order: Vec<StoreId>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AttestationStore for MemoryStore {
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

        // 2. Compute content-addressed ID.
        let id = attestation_store_id(attestation)?;

        // 3. Idempotent: if already present, return existing ID.
        if self.entries.contains_key(&id) {
            return Ok(id);
        }

        // 4. Chain validation: parent must already be in store.
        if let Some(parent_hash) = attestation.parent_attestation_hash {
            if !self.entries.contains_key(&parent_hash) {
                let hex: String = parent_hash.iter().map(|b| format!("{b:02x}")).collect();
                return Err(StoreError::OrphanedAttestation(hex));
            }
        }

        // 5. Store.
        let signer_hash = got_core::sha256(verifying_key.as_bytes());
        self.entries.insert(
            id,
            Entry {
                attestation: attestation.clone(),
                signer_hash,
            },
        );
        self.order.push(id);

        Ok(id)
    }

    fn get(&self, id: &StoreId) -> Option<&GeometricAttestation> {
        self.entries.get(id).map(|e| &e.attestation)
    }

    fn chain(&self, model_id: &str) -> Vec<&GeometricAttestation> {
        let mut chain: Vec<&GeometricAttestation> = self
            .order
            .iter()
            .filter_map(|id| self.entries.get(id))
            .filter(|e| e.attestation.model_id == model_id)
            .map(|e| &e.attestation)
            .collect();
        chain.sort_by_key(|a| a.timestamp);
        chain
    }

    fn query(&self, filter: &StoreFilter) -> Vec<&GeometricAttestation> {
        self.order
            .iter()
            .filter_map(|id| self.entries.get(id))
            .filter(|e| filter.matches(&e.attestation, &e.signer_hash))
            .map(|e| &e.attestation)
            .collect()
    }

    fn audit(&self, model_id: &str) -> AuditReport {
        let chain = self.chain(model_id);
        let signer_hashes: Vec<[u8; 32]> = self
            .order
            .iter()
            .filter_map(|id| self.entries.get(id))
            .filter(|e| e.attestation.model_id == model_id)
            .map(|e| e.signer_hash)
            .collect();
        build_audit_report(model_id, &chain, &signer_hashes)
    }

    fn len(&self) -> usize {
        self.entries.len()
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

    // --- append & get ---

    #[test]
    fn append_and_retrieve() {
        let key = test_key();
        let a = make_attestation(&key, "model-a", 1000, None, None, None, vec![]);
        let mut store = MemoryStore::new();
        let id = store.append(&a, &key.verifying_key()).unwrap();
        let got = store.get(&id).unwrap();
        assert_eq!(got.model_id, "model-a");
        assert_eq!(got.timestamp, 1000);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn duplicate_append_is_idempotent() {
        let key = test_key();
        let a = make_attestation(&key, "model-a", 1000, None, None, None, vec![]);
        let mut store = MemoryStore::new();
        let id1 = store.append(&a, &key.verifying_key()).unwrap();
        let id2 = store.append(&a, &key.verifying_key()).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn rejects_invalid_signature() {
        let key = test_key();
        let mut a = make_attestation(&key, "model-a", 1000, None, None, None, vec![]);
        a.timestamp = 9999; // modify after signing → invalid
        let mut store = MemoryStore::new();
        let err = store.append(&a, &key.verifying_key());
        assert!(matches!(err, Err(StoreError::InvalidSignature)));
    }

    // --- chain validation ---

    #[test]
    fn chain_valid_parent_accepted() {
        let key = test_key();
        let a0 = make_attestation(&key, "model-a", 1000, None, None, None, vec![]);
        let a0_hash = content_hash(&a0);
        let a1 = make_attestation(&key, "model-a", 2000, Some(a0_hash), None, None, vec![]);

        let mut store = MemoryStore::new();
        store.append(&a0, &key.verifying_key()).unwrap();
        store.append(&a1, &key.verifying_key()).unwrap();
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn chain_rejects_orphaned_parent() {
        let key = test_key();
        let fake_parent = [0xFF; 32];
        let a = make_attestation(&key, "model-a", 1000, Some(fake_parent), None, None, vec![]);
        let mut store = MemoryStore::new();
        let err = store.append(&a, &key.verifying_key());
        assert!(matches!(err, Err(StoreError::OrphanedAttestation(_))));
    }

    // --- query ---

    #[test]
    fn query_by_model_id() {
        let key = test_key();
        let a = make_attestation(&key, "model-a", 1000, None, None, None, vec![]);
        let b = make_attestation(&key, "model-b", 2000, None, None, None, vec![]);

        let mut store = MemoryStore::new();
        store.append(&a, &key.verifying_key()).unwrap();
        store.append(&b, &key.verifying_key()).unwrap();

        let results = store.query(&StoreFilter::new().model_id("model-a"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model_id, "model-a");
    }

    #[test]
    fn query_by_time_range() {
        let key = test_key();
        let a = make_attestation(&key, "m", 1000, None, None, None, vec![]);
        let b = make_attestation(&key, "m", 2000, None, None, None, vec![]);
        let c = make_attestation(&key, "m", 3000, None, None, None, vec![]);

        let mut store = MemoryStore::new();
        store.append(&a, &key.verifying_key()).unwrap();
        store.append(&b, &key.verifying_key()).unwrap();
        store.append(&c, &key.verifying_key()).unwrap();

        let results = store.query(&StoreFilter::new().after(1500).before(2500));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, 2000);
    }

    #[test]
    fn query_by_causal_flag() {
        let key = test_key();
        let a = make_attestation(&key, "m", 1000, None, None, Some(true), vec![]);
        let b = make_attestation(&key, "m", 2000, None, None, Some(false), vec![]);

        let mut store = MemoryStore::new();
        store.append(&a, &key.verifying_key()).unwrap();
        store.append(&b, &key.verifying_key()).unwrap();

        let passing = store.query(&StoreFilter::new().causal_flag(true));
        assert_eq!(passing.len(), 1);
        assert_eq!(passing[0].timestamp, 1000);

        let failing = store.query(&StoreFilter::new().causal_flag(false));
        assert_eq!(failing.len(), 1);
        assert_eq!(failing[0].timestamp, 2000);
    }

    #[test]
    fn query_by_signer() {
        let key_a = SigningKey::from_bytes(&[0xAA; 32]);
        let key_b = SigningKey::from_bytes(&[0xBB; 32]);
        let a = make_attestation(&key_a, "m", 1000, None, None, None, vec![]);
        let b = make_attestation(&key_b, "m", 2000, None, None, None, vec![]);

        let mut store = MemoryStore::new();
        store.append(&a, &key_a.verifying_key()).unwrap();
        store.append(&b, &key_b.verifying_key()).unwrap();

        let results = store.query(&StoreFilter::new().signer(&key_a.verifying_key()));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, 1000);
    }

    // --- chain ---

    #[test]
    fn chain_returns_sorted_by_timestamp() {
        let key = test_key();
        // Insert out of order.
        let a2 = make_attestation(&key, "m", 3000, None, None, None, vec![]);
        let a0 = make_attestation(&key, "m", 1000, None, None, None, vec![]);
        let a1 = make_attestation(&key, "m", 2000, None, None, None, vec![]);

        let mut store = MemoryStore::new();
        store.append(&a2, &key.verifying_key()).unwrap();
        store.append(&a0, &key.verifying_key()).unwrap();
        store.append(&a1, &key.verifying_key()).unwrap();

        let chain = store.chain("m");
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].timestamp, 1000);
        assert_eq!(chain[1].timestamp, 2000);
        assert_eq!(chain[2].timestamp, 3000);
    }

    // --- audit ---

    #[test]
    fn audit_report_empty_model() {
        let store = MemoryStore::new();
        let report = store.audit("nonexistent");
        assert_eq!(report.total_attestations, 0);
        assert_eq!(report.chain_length, 0);
        assert!(report.chain_valid);
    }

    #[test]
    fn audit_report_with_chain_and_drift() {
        let key = test_key();
        let a0 = make_attestation(
            &key,
            "model-x",
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
            "model-x",
            2000,
            Some(a0_hash),
            Some(0.03),
            Some(true),
            vec![CausalScoreRecord {
                delta_plus: 0.6,
                delta_minus: 0.3,
                consistency: 0.8,
                is_causal: true,
            }],
        );

        let mut store = MemoryStore::new();
        store.append(&a0, &key.verifying_key()).unwrap();
        store.append(&a1, &key.verifying_key()).unwrap();

        let report = store.audit("model-x");
        assert_eq!(report.total_attestations, 2);
        assert_eq!(report.chain_length, 2);
        assert!(report.chain_valid);
        assert_eq!(report.first_timestamp, Some(1000));
        assert_eq!(report.last_timestamp, Some(2000));
        assert_eq!(report.schema_versions_seen, vec![SCHEMA_VERSION_3]);
        assert_eq!(report.drift_summary.readings_with_drift, 2);
        assert!((report.drift_summary.max_drift.unwrap() - 0.03).abs() < 1e-6);
        assert!((report.drift_summary.mean_drift.unwrap() - 0.02).abs() < 1e-6);
        assert_eq!(report.causal_summary.attestations_with_causal, 2);
        assert_eq!(report.causal_summary.causal_pass_count, 2);
        assert_eq!(report.causal_summary.causal_fail_count, 0);
        let mean_c = report.causal_summary.mean_consistency.unwrap();
        assert!((mean_c - 0.85).abs() < 1e-6);
        assert_eq!(report.signers.len(), 1);
    }

    #[test]
    fn audit_detects_broken_chain() {
        let key = test_key();
        // a0 and a1 are both roots (no parent). Neither is linked to the other.
        let a0 = make_attestation(&key, "m", 1000, None, None, None, vec![]);
        let a1 = make_attestation(&key, "m", 2000, None, None, None, vec![]);

        let mut store = MemoryStore::new();
        store.append(&a0, &key.verifying_key()).unwrap();
        store.append(&a1, &key.verifying_key()).unwrap();

        let report = store.audit("m");
        assert_eq!(report.total_attestations, 2);
        // Chain length from tip: a1 has no parent → length 1.
        assert_eq!(report.chain_length, 1);
    }
}
