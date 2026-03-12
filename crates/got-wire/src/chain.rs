// ---------------------------------------------------------------------------
// Chain Verification — Phase 10, §10.11.
//
// Walk a chain of attestations verifying:
//   1. Anchor has no parent_attestation_hash
//   2. Model identity consistency across all links
//   3. Each link's signature is valid
//   4. Each link's parent hash matches the previous entry
//   5. Drift is within LOCAL bounds
//   6. Sequence numbers are monotonic (Phase 13)
// ---------------------------------------------------------------------------

use ed25519_dalek::VerifyingKey;

use got_core::GeometricAttestation;

use crate::WireError;

/// Outcome of a successful chain verification.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainVerdict {
    /// Number of attestations in the chain (including current).
    pub length: usize,
    /// Maximum drift observed across all links.
    pub max_drift_observed: f32,
}

/// Compute the SHA-256 hash of an attestation's canonical bytes.
///
/// Delegates to [`got_attest::attestation_hash`] for a single canonical
/// implementation, converting the error type for wire-protocol callers.
pub fn attestation_hash(a: &GeometricAttestation) -> Result<[u8; 32], WireError> {
    got_attest::attestation_hash(a).map_err(WireError::from)
}

/// Verify a chain of attestations: signatures, linkage, and drift bounds.
///
/// `chain` is the ancestry from oldest (chain[0] = anchor) to newest.
/// `current` is the latest attestation (the one being exchanged).
/// `signer_pks` is one or more Ed25519 verifying keys — each attestation in
///   the chain must verify against **at least one** key in this set.  This
///   supports key rotation: pass `&[old_key, new_key]` to accept chains that
///   span a rotation boundary.
/// `max_drift` is the LOCAL policy drift threshold — never from the wire.
pub fn verify_chain(
    chain: &[GeometricAttestation],
    current: &GeometricAttestation,
    signer_pks: &[VerifyingKey],
    max_drift: f32,
) -> Result<ChainVerdict, WireError> {
    if signer_pks.is_empty() {
        return Err(WireError::Chain("no signer keys provided".to_string()));
    }

    // Build the combined list: chain ++ [current]
    let mut all: Vec<&GeometricAttestation> = chain.iter().collect();
    all.push(current);

    if all.is_empty() {
        return Err(WireError::Chain("empty chain".to_string()));
    }

    // 1. Anchor check: first attestation must have no parent.
    if all[0].parent_attestation_hash.is_some() {
        return Err(WireError::Chain(
            "anchor (chain[0]) must have no parent_attestation_hash".to_string(),
        ));
    }

    let mut max_drift_observed: f32 = 0.0;

    // 2. Model identity consistency — every attestation must refer to the same model.
    let anchor_model_id = &all[0].model_id;
    for (i, att) in all.iter().enumerate().skip(1) {
        if att.model_id != *anchor_model_id {
            return Err(WireError::Chain(format!(
                "model_id mismatch at chain index {i}: expected \"{anchor_model_id}\", got \"{}\"",
                att.model_id
            )));
        }
    }

    // 3. Walk each link.
    for i in 0..all.len() {
        // Signature check — must verify against at least one trusted key.
        let sig_ok = signer_pks
            .iter()
            .any(|pk| got_attest::verify(all[i], pk).is_ok());
        if !sig_ok {
            return Err(WireError::Chain(format!(
                "invalid signature at chain index {i}: no trusted key verified"
            )));
        }

        // Linkage check — every element after the anchor must reference its predecessor.
        if i > 0 {
            let expected = attestation_hash(all[i - 1])?;
            match &all[i].parent_attestation_hash {
                None => {
                    return Err(WireError::Chain(format!(
                        "chain[{i}] has no parent_attestation_hash but follows chain[{}]",
                        i - 1
                    )));
                }
                Some(actual) => {
                    if *actual != expected {
                        return Err(WireError::Chain(format!(
                            "broken link at chain index {i}: parent hash mismatch"
                        )));
                    }
                }
            }
        }

        // Drift check — LOCAL max_drift, not from wire.
        if let Some(drift) = all[i].geometry_drift {
            if drift > max_drift {
                return Err(WireError::Chain(format!(
                    "drift exceeded at chain index {i}: drift={drift}, max={max_drift}"
                )));
            }
            if drift > max_drift_observed {
                max_drift_observed = drift;
            }
        }

        // Sequence number check — Phase 13: monotonic, no gaps, no duplicates.
        if i > 0 {
            let expected_seq = all[i - 1].sequence_number + 1;
            if all[i].sequence_number != expected_seq {
                return Err(WireError::Chain(format!(
                    "sequence gap at chain index {i}: expected {expected_seq}, got {}",
                    all[i].sequence_number
                )));
            }
        }
    }

    Ok(ChainVerdict {
        length: all.len(),
        max_drift_observed,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use got_attest::assemble_and_sign;
    use got_core::{InnerProduct, Precision, SCHEMA_VERSION, SCHEMA_VERSION_2};

    fn test_key() -> SigningKey {
        SigningKey::from_bytes(&[0x42; 32])
    }

    fn make_v1_attestation(key: &SigningKey) -> GeometricAttestation {
        let a = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: "test-model".to_string(),
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xBB; 32],
            timestamp: 1700000000,
            corpus_version: "v1".to_string(),
            probe_version: "v1".to_string(),
            layer_readings: vec![vec![1.0, 2.0]],
            confidence: vec![0.95],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: None,
            geometry_hash: None,
            geometry_drift: None,
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 0,
            directional_drifts: vec![],
            probe_commitment: None,
            signature: [0u8; 64],
        };
        assemble_and_sign(a, key).unwrap()
    }

    fn make_v2_child(
        key: &SigningKey,
        parent: &GeometricAttestation,
        drift: f32,
    ) -> GeometricAttestation {
        let parent_hash = attestation_hash(parent).unwrap();
        let a = GeometricAttestation {
            schema_version: SCHEMA_VERSION_2,
            model_id: "test-model".to_string(),
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xCC; 32],
            timestamp: parent.timestamp + 100,
            corpus_version: "v2".to_string(),
            probe_version: "v2".to_string(),
            layer_readings: vec![vec![1.1, 2.1]],
            confidence: vec![0.93],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: Some(parent_hash),
            geometry_hash: Some([0xDD; 32]),
            geometry_drift: Some(drift),
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: parent.sequence_number + 1,
            directional_drifts: vec![],
            probe_commitment: None,
            signature: [0u8; 64],
        };
        assemble_and_sign(a, key).unwrap()
    }

    #[test]
    fn valid_single_attestation() {
        let key = test_key();
        let a = make_v1_attestation(&key);
        let verdict = verify_chain(&[], &a, &[key.verifying_key()], 1.0).unwrap();
        assert_eq!(verdict.length, 1);
        assert!((verdict.max_drift_observed - 0.0).abs() < 1e-6);
    }

    #[test]
    fn valid_two_link_chain() {
        let key = test_key();
        let anchor = make_v1_attestation(&key);
        let child = make_v2_child(&key, &anchor, 0.03);
        let verdict = verify_chain(&[anchor], &child, &[key.verifying_key()], 0.05).unwrap();
        assert_eq!(verdict.length, 2);
        assert!((verdict.max_drift_observed - 0.03).abs() < 1e-6);
    }

    #[test]
    fn valid_three_link_chain() {
        let key = test_key();
        let a0 = make_v1_attestation(&key);
        let a1 = make_v2_child(&key, &a0, 0.02);
        let a2 = make_v2_child(&key, &a1, 0.04);
        let verdict = verify_chain(&[a0, a1], &a2, &[key.verifying_key()], 0.05).unwrap();
        assert_eq!(verdict.length, 3);
        assert!((verdict.max_drift_observed - 0.04).abs() < 1e-6);
    }

    #[test]
    fn broken_anchor_has_parent() {
        let key = test_key();
        // Build an attestation with a parent hash set (invalid anchor).
        let mut bad_anchor = GeometricAttestation {
            schema_version: SCHEMA_VERSION_2,
            model_id: "test-model".to_string(),
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xBB; 32],
            timestamp: 1700000000,
            corpus_version: "v1".to_string(),
            probe_version: "v1".to_string(),
            layer_readings: vec![vec![1.0]],
            confidence: vec![0.9],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: Some([0xFF; 32]),
            geometry_hash: Some([0xEE; 32]),
            geometry_drift: Some(0.01),
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 0,
            directional_drifts: vec![],
            probe_commitment: None,
            signature: [0u8; 64],
        };
        bad_anchor = assemble_and_sign(bad_anchor, &key).unwrap();
        let err = verify_chain(&[], &bad_anchor, &[key.verifying_key()], 1.0);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("anchor"), "error should mention anchor: {msg}");
    }

    #[test]
    fn broken_link_wrong_parent_hash() {
        let key = test_key();
        let anchor = make_v1_attestation(&key);
        // Build a child that points to a different parent hash.
        let mut bad_child = GeometricAttestation {
            schema_version: SCHEMA_VERSION_2,
            model_id: "test-model".to_string(),
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xCC; 32],
            timestamp: anchor.timestamp + 100,
            corpus_version: "v2".to_string(),
            probe_version: "v2".to_string(),
            layer_readings: vec![vec![1.1]],
            confidence: vec![0.9],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: Some([0x00; 32]), // wrong!
            geometry_hash: Some([0xDD; 32]),
            geometry_drift: Some(0.01),
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 1,
            directional_drifts: vec![],
            probe_commitment: None,
            signature: [0u8; 64],
        };
        bad_child = assemble_and_sign(bad_child, &key).unwrap();
        let err = verify_chain(&[anchor], &bad_child, &[key.verifying_key()], 1.0);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("broken link"),
            "error should mention broken link: {msg}"
        );
    }

    #[test]
    fn drift_exceeds_local_max() {
        let key = test_key();
        let anchor = make_v1_attestation(&key);
        let child = make_v2_child(&key, &anchor, 0.08); // drift = 0.08
        let err = verify_chain(&[anchor], &child, &[key.verifying_key()], 0.05); // max = 0.05
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("drift exceeded"),
            "error should mention drift: {msg}"
        );
    }

    #[test]
    fn invalid_signature_detected() {
        let key = test_key();
        let other_key = SigningKey::from_bytes(&[0x99; 32]);
        let a = make_v1_attestation(&key);
        // Verify with wrong key.
        let err = verify_chain(&[], &a, &[other_key.verifying_key()], 1.0);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("invalid signature"),
            "error should mention signature: {msg}"
        );
    }

    #[test]
    fn attestation_hash_deterministic() {
        let key = test_key();
        let a = make_v1_attestation(&key);
        let h1 = attestation_hash(&a).unwrap();
        let h2 = attestation_hash(&a).unwrap();
        assert_eq!(h1, h2, "hash must be deterministic");
    }

    #[test]
    fn model_id_mismatch_rejected() {
        let key = test_key();
        let anchor = make_v1_attestation(&key);
        let parent_hash = attestation_hash(&anchor).unwrap();

        // Create a child with a DIFFERENT model_id.
        let mut bad = GeometricAttestation {
            schema_version: SCHEMA_VERSION_2,
            model_id: "other-model".to_string(), // mismatch!
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xCC; 32],
            timestamp: anchor.timestamp + 100,
            corpus_version: "v2".to_string(),
            probe_version: "v2".to_string(),
            layer_readings: vec![vec![1.1]],
            confidence: vec![0.9],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: Some(parent_hash),
            geometry_hash: Some([0xDD; 32]),
            geometry_drift: Some(0.01),
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 1,
            directional_drifts: vec![],
            probe_commitment: None,
            signature: [0u8; 64],
        };
        bad = assemble_and_sign(bad, &key).unwrap();
        let err = verify_chain(&[anchor], &bad, &[key.verifying_key()], 1.0);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("model_id mismatch"),
            "error should mention model_id mismatch: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issue 29)
    // -----------------------------------------------------------------------

    /// Issue #29 (S-8): verify_chain requires the same signer for all
    /// attestations, but doesn't support key rotation.
    ///
    /// After fix: verify_chain either looks up per-link signers or accepts
    /// a key-rotation attestation in the chain.  For now we verify the
    /// current behaviour: a chain signed by two different keys is rejected.
    #[test]
    fn sec_verify_chain_key_rotation() {
        let key_a = SigningKey::from_bytes(&[0xAA; 32]);
        let key_b = SigningKey::from_bytes(&[0xBB; 32]);

        // Anchor signed by key_a.
        let anchor = make_v1_attestation(&key_a);

        // Child signed by key_b — simulates post-rotation attestation.
        let parent_hash = crate::chain::attestation_hash(&anchor).unwrap();
        let child = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: "test-model".to_string(),
            model_hash: Some([0x11; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xCC; 32],
            timestamp: anchor.timestamp + 100,
            corpus_version: "v2".to_string(),
            probe_version: "v2".to_string(),
            layer_readings: vec![vec![1.1]],
            confidence: vec![0.9],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: Some(parent_hash),
            geometry_hash: Some([0xDD; 32]),
            geometry_drift: Some(0.01),
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 1,
            directional_drifts: vec![],
            probe_commitment: None,
            signature: [0u8; 64],
        };
        let child = assemble_and_sign(child, &key_b).unwrap();

        // With only key_a → child signature fails.
        let err = verify_chain(&[anchor.clone()], &child, &[key_a.verifying_key()], 1.0);
        assert!(
            err.is_err(),
            "single old key should reject post-rotation link"
        );

        // With only key_b → anchor signature fails.
        let err = verify_chain(&[anchor.clone()], &child, &[key_b.verifying_key()], 1.0);
        assert!(
            err.is_err(),
            "single new key should reject pre-rotation link"
        );

        // With both keys → chain validates (key rotation supported).
        let verdict = verify_chain(
            &[anchor],
            &child,
            &[key_a.verifying_key(), key_b.verifying_key()],
            1.0,
        )
        .expect("both keys provided: chain should verify after key rotation");
        assert_eq!(verdict.length, 2);

        // Empty key slice → rejected.
        let anchor2 = make_v1_attestation(&key_a);
        let err = verify_chain(&[], &anchor2, &[], 1.0);
        assert!(err.is_err(), "empty key slice must be rejected");
    }
}
