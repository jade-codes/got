// ---------------------------------------------------------------------------
// Attestation assembly, signing, verification, and Merkle tree.
//
// The critical contract: `serialise_for_signing` must be a pure function.
// Same GeometricAttestation → identical bytes. Always.
// ---------------------------------------------------------------------------

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use got_core::GeometricAttestation;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AttestationError {
    #[error("NaN detected in attestation field: {field}")]
    NaN { field: &'static str },
    #[error("infinity detected in attestation field: {field}")]
    Infinity { field: &'static str },
    #[error("signature verification failed")]
    SignatureInvalid,
    #[error("unknown schema version: {0}")]
    UnknownSchemaVersion(u16),
    #[error("field too large: {field} ({size} bytes, max {max})")]
    FieldTooLarge {
        field: &'static str,
        size: usize,
        max: usize,
    },
    #[error("timestamp too far in the future ({delta}s > max {max}s)")]
    TimestampFuture { delta: u64, max: u64 },
}

/// Supported schema versions (v1 = original, v2 = chained attestation, v3 = causal intervention).
const SUPPORTED_SCHEMA_V1: u16 = 1;
const SUPPORTED_SCHEMA_V2: u16 = 2;
const SUPPORTED_SCHEMA_V3: u16 = 3;

fn is_supported_schema(v: u16) -> bool {
    v == SUPPORTED_SCHEMA_V1 || v == SUPPORTED_SCHEMA_V2 || v == SUPPORTED_SCHEMA_V3
}

/// Sign an attestation. Serialises all fields except `signature` to canonical
/// bytes, signs with Ed25519, writes signature into the struct.
/// Maximum allowed clock skew (seconds) for attestation timestamps.
const MAX_FUTURE_SECS: u64 = 300;
/// Maximum length of string fields (model_id, corpus_version, probe_version).
const MAX_STRING_LEN: usize = 256;
/// Maximum number of layers in layer_readings.
const MAX_LAYERS: usize = 1024;
/// Maximum total readings across all layers.
const MAX_TOTAL_READINGS: usize = 65536;

pub fn assemble_and_sign(
    mut attestation: GeometricAttestation,
    signing_key: &SigningKey,
) -> Result<GeometricAttestation, AttestationError> {
    if !is_supported_schema(attestation.schema_version) {
        return Err(AttestationError::UnknownSchemaVersion(
            attestation.schema_version,
        ));
    }

    // S-7: Reject far-future timestamps.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if attestation.timestamp > now + MAX_FUTURE_SECS {
        return Err(AttestationError::TimestampFuture {
            delta: attestation.timestamp - now,
            max: MAX_FUTURE_SECS,
        });
    }

    // S-13: Reject oversized string fields.
    validate_string_len(attestation.model_id.len(), "model_id")?;
    validate_string_len(attestation.corpus_version.len(), "corpus_version")?;
    validate_string_len(attestation.probe_version.len(), "probe_version")?;

    // S-20: Reject oversized arrays.
    if attestation.layer_readings.len() > MAX_LAYERS {
        return Err(AttestationError::FieldTooLarge {
            field: "layer_readings",
            size: attestation.layer_readings.len(),
            max: MAX_LAYERS,
        });
    }
    let total_readings: usize = attestation.layer_readings.iter().map(|l| l.len()).sum();
    if total_readings > MAX_TOTAL_READINGS {
        return Err(AttestationError::FieldTooLarge {
            field: "total_readings",
            size: total_readings,
            max: MAX_TOTAL_READINGS,
        });
    }

    let payload = serialise_for_signing(&attestation)?;
    let sig = signing_key.sign(&payload);
    attestation.signature = sig.to_bytes();
    Ok(attestation)
}

fn validate_string_len(len: usize, field: &'static str) -> Result<(), AttestationError> {
    if len > MAX_STRING_LEN {
        return Err(AttestationError::FieldTooLarge {
            field,
            size: len,
            max: MAX_STRING_LEN,
        });
    }
    Ok(())
}

/// Type-safe variant of [`assemble_and_sign`] that accepts an
/// [`UnsignedAttestation`], enforcing at the type level that only
/// unsigned attestations go through signing.
pub fn sign(
    unsigned: got_core::UnsignedAttestation,
    signing_key: &SigningKey,
) -> Result<GeometricAttestation, AttestationError> {
    assemble_and_sign(unsigned.into_inner(), signing_key)
}

/// Verify an attestation's signature.
///
/// Returns `Ok(())` if the signature is valid, or `Err(SignatureInvalid)` if not.
/// This API makes it impossible to accidentally ignore a bad signature via `?`.
pub fn verify(
    attestation: &GeometricAttestation,
    verifying_key: &VerifyingKey,
) -> Result<(), AttestationError> {
    if !is_supported_schema(attestation.schema_version) {
        return Err(AttestationError::UnknownSchemaVersion(
            attestation.schema_version,
        ));
    }

    let payload = serialise_for_signing(attestation)?;
    let sig = ed25519_dalek::Signature::from_bytes(&attestation.signature);
    verifying_key
        .verify(&payload, &sig)
        .map_err(|_| AttestationError::SignatureInvalid)
}

/// Deterministic canonical serialisation of all attestation fields except `signature`.
///
/// RULES:
///  - All integers: little-endian fixed width
///  - Strings: u32 LE length prefix + UTF-8 bytes
///  - Floats: canonicalise (-0.0 → 0.0, reject NaN), then f32::to_le_bytes()
///  - Booleans: 0x00 or 0x01
///  - Variable-length fields: u32 LE count prefix + elements
///  - Field order: exactly as listed in the struct definition
pub fn serialise_for_signing(a: &GeometricAttestation) -> Result<Vec<u8>, AttestationError> {
    if !is_supported_schema(a.schema_version) {
        return Err(AttestationError::UnknownSchemaVersion(a.schema_version));
    }

    let mut buf = Vec::with_capacity(4096);

    // === Common fields (v1 + v2) ===

    // schema_version
    write_u16(&mut buf, a.schema_version);

    // model_id
    write_string(&mut buf, &a.model_id);

    // model_hash (tagged: 0x00 = None, 0x01 + 32 bytes = Some)
    match &a.model_hash {
        Some(h) => {
            buf.push(0x01);
            buf.extend_from_slice(h);
        }
        None => {
            buf.push(0x00);
        }
    }

    // precision
    buf.push(a.precision.tag());

    // inner_product
    buf.push(a.inner_product.tag());
    if let got_core::InnerProduct::CausalRegularised { epsilon } = a.inner_product {
        write_f32(&mut buf, epsilon, "inner_product.epsilon")?;
    }

    // input_hash
    buf.extend_from_slice(&a.input_hash);

    // timestamp
    write_u64(&mut buf, a.timestamp);

    // corpus_version
    write_string(&mut buf, &a.corpus_version);

    // probe_version
    write_string(&mut buf, &a.probe_version);

    // layer_readings: Vec<Vec<f32>>
    write_u32(&mut buf, a.layer_readings.len() as u32);
    for layer in &a.layer_readings {
        write_f32_vec(&mut buf, layer, "layer_readings")?;
    }

    // confidence
    write_f32_vec(&mut buf, &a.confidence, "confidence")?;

    // coverage_flags
    write_bool_vec(&mut buf, &a.coverage_flags);

    // divergence_flag
    buf.push(if a.divergence_flag { 0x01 } else { 0x00 });

    // === v2 extension fields (chained attestation) ===
    if a.schema_version >= SUPPORTED_SCHEMA_V2 {
        // parent_attestation_hash: Option<[u8; 32]>
        match &a.parent_attestation_hash {
            None => buf.push(0x00),
            Some(hash) => {
                buf.push(0x01);
                buf.extend_from_slice(hash);
            }
        }
        // geometry_hash: Option<[u8; 32]>
        match &a.geometry_hash {
            None => buf.push(0x00),
            Some(hash) => {
                buf.push(0x01);
                buf.extend_from_slice(hash);
            }
        }
        // geometry_drift: Option<f32>
        match a.geometry_drift {
            None => buf.push(0x00),
            Some(drift) => {
                buf.push(0x01);
                write_f32(&mut buf, drift, "geometry_drift")?;
            }
        }
    }

    // === v3 extension fields (causal intervention) ===
    if a.schema_version >= SUPPORTED_SCHEMA_V3 {
        // causal_scores: Vec<CausalScoreRecord>
        write_u32(&mut buf, a.causal_scores.len() as u32);
        for cs in &a.causal_scores {
            write_f32(&mut buf, cs.delta_plus, "causal_scores.delta_plus")?;
            write_f32(&mut buf, cs.delta_minus, "causal_scores.delta_minus")?;
            write_f32(&mut buf, cs.consistency, "causal_scores.consistency")?;
            buf.push(if cs.is_causal { 0x01 } else { 0x00 });
        }
        // intervention_delta: Option<f32>
        match a.intervention_delta {
            None => buf.push(0x00),
            Some(d) => {
                buf.push(0x01);
                write_f32(&mut buf, d, "intervention_delta")?;
            }
        }
        // causal_flag: Option<bool>
        match a.causal_flag {
            None => buf.push(0x00),
            Some(f) => {
                buf.push(0x01);
                buf.push(if f { 0x01 } else { 0x00 });
            }
        }
    }

    // === Phase 13 fields (adversarial hardening) ===
    // Gated to v2+ to preserve backward-compatible serialisation for v1 attestations.
    // v1 attestations never participate in chains so they have no meaningful
    // sequence_number or directional_drifts.
    if a.schema_version >= SUPPORTED_SCHEMA_V2 {
        // sequence_number: u64
        write_u64(&mut buf, a.sequence_number);
        // directional_drifts: Vec<DirectionalDrift>
        write_u32(&mut buf, a.directional_drifts.len() as u32);
        for dd in &a.directional_drifts {
            write_string(&mut buf, &dd.probe_name);
            write_f32(&mut buf, dd.drift, "directional_drift")?;
        }
        // probe_commitment: Option<[u8; 32]>
        match &a.probe_commitment {
            None => buf.push(0x00),
            Some(hash) => {
                buf.push(0x01);
                buf.extend_from_slice(hash);
            }
        }
    }

    // NOTE: signature is excluded.

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Merkle tree over weight shards
// ---------------------------------------------------------------------------

/// Compute the Merkle root over sorted weight shards.
///
/// Uses domain separation (RFC 6962) to prevent second-preimage attacks:
///   - Leaf hash:     H(0x00 || data)
///   - Internal hash: H(0x01 || left || right)
///
/// Odd nodes at any level are duplicated before hashing.
///
/// Each shard is an opaque byte slice (the caller is responsible for canonical
/// shard serialisation — see PLAN.md §3.4 for the shard format).
pub fn merkle_root(shards: &[Vec<u8>]) -> [u8; 32] {
    if shards.is_empty() {
        // Domain-separated empty sentinel so "zero shards" is distinguishable
        // from all-zeroes or the CLI "no --shards" sentinel of [0xFF; 32].
        return got_core::sha256(b"merkle-empty");
    }

    // Hash each leaf with 0x00 domain separator
    let mut nodes: Vec<[u8; 32]> = shards
        .iter()
        .map(|shard| {
            let mut hasher = Sha256::new();
            hasher.update([0x00]); // leaf prefix
            hasher.update(shard);
            hasher.finalize().into()
        })
        .collect();

    // Build tree bottom-up with 0x01 domain separator for internal nodes
    while nodes.len() > 1 {
        // Duplicate odd node to make length even
        if nodes.len() % 2 != 0 {
            nodes.push(*nodes.last().unwrap());
        }
        nodes = nodes
            .chunks(2)
            .map(|pair| {
                let mut hasher = Sha256::new();
                hasher.update([0x01]); // internal node prefix
                hasher.update(pair[0]);
                hasher.update(pair[1]);
                hasher.finalize().into()
            })
            .collect();
    }

    nodes[0]
}

/// SHA-256 of the canonical serialised payload of an attestation.
/// Used to compute `parent_attestation_hash` for chained attestations.
pub fn attestation_hash(a: &GeometricAttestation) -> Result<[u8; 32], AttestationError> {
    let payload = serialise_for_signing(a)?;
    Ok(got_core::sha256(&payload))
}

/// Validate structural consistency of causal intervention fields.
///
/// Returns `true` if:
///   - `causal_flag` is `None` and `causal_scores` is empty (pre-intervention attestation), OR
///   - `causal_flag` matches `causal_scores.iter().all(|s| s.is_causal)`, AND
///   - `intervention_delta` is `Some` when `causal_scores` is non-empty.
///
/// A hand-crafted attestation claiming `causal_flag: Some(true)` while having
/// failing causal scores would return `false`.
pub fn validate_causal_consistency(a: &GeometricAttestation) -> bool {
    if a.causal_scores.is_empty() {
        // No causal intervention was run. causal_flag should be None.
        // intervention_delta should also be None.
        return a.causal_flag.is_none() && a.intervention_delta.is_none();
    }

    // causal_scores is non-empty — intervention_delta must be present.
    if a.intervention_delta.is_none() {
        return false;
    }

    // causal_flag must match the actual scores.
    match a.causal_flag {
        None => false, // scores present but no flag
        Some(flag) => flag == a.causal_scores.iter().all(|s| s.is_causal),
    }
}

// ---------------------------------------------------------------------------
// Serialisation helpers — all little-endian, all deterministic
// ---------------------------------------------------------------------------

fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

/// Canonicalise and write a single f32.
fn write_f32(buf: &mut Vec<u8>, v: f32, field: &'static str) -> Result<(), AttestationError> {
    if v.is_nan() {
        return Err(AttestationError::NaN { field });
    }
    if v.is_infinite() {
        return Err(AttestationError::Infinity { field });
    }
    // Canonicalise: -0.0 → 0.0
    let canonical = if v == 0.0 { 0.0f32 } else { v };
    buf.extend_from_slice(&canonical.to_le_bytes());
    Ok(())
}

fn write_f32_vec(
    buf: &mut Vec<u8>,
    vs: &[f32],
    field: &'static str,
) -> Result<(), AttestationError> {
    write_u32(buf, vs.len() as u32);
    for &v in vs {
        write_f32(buf, v, field)?;
    }
    Ok(())
}

fn write_bool_vec(buf: &mut Vec<u8>, vs: &[bool]) {
    write_u32(buf, vs.len() as u32);
    for &v in vs {
        buf.push(if v { 0x01 } else { 0x00 });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use got_core::{InnerProduct, Precision, SCHEMA_VERSION};
    use sha2::{Digest, Sha256};

    fn make_test_attestation() -> GeometricAttestation {
        GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: "test-model".to_string(),
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0xBB; 32],
            timestamp: 1709568000,
            corpus_version: "test-corpus-v1".to_string(),
            probe_version: "test-probe-v1".to_string(),
            layer_readings: vec![vec![1.0, 2.0, 3.0]],
            confidence: vec![0.9, 0.8, 0.7],
            coverage_flags: vec![false, false, true],
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
        }
    }

    fn test_signing_key() -> SigningKey {
        // Deterministic key from fixed seed for testing
        let seed: [u8; 32] = [42u8; 32];
        SigningKey::from_bytes(&seed)
    }

    #[test]
    fn sign_and_verify_roundtrip() {
        let attestation = make_test_attestation();
        let key = test_signing_key();
        let verifying_key = key.verifying_key();

        let signed = assemble_and_sign(attestation, &key).unwrap();
        assert_ne!(signed.signature, [0u8; 64], "signature should be non-zero");

        verify(&signed, &verifying_key).expect("signature should verify");
    }

    #[test]
    fn tampered_attestation_fails() {
        let attestation = make_test_attestation();
        let key = test_signing_key();

        let mut signed = assemble_and_sign(attestation, &key).unwrap();

        // Tamper with a field
        signed.timestamp += 1;

        assert!(
            verify(&signed, &key.verifying_key()).is_err(),
            "tampered attestation should fail verification"
        );
    }

    #[test]
    fn serialisation_is_deterministic() {
        let a = make_test_attestation();
        let bytes1 = serialise_for_signing(&a).unwrap();
        let bytes2 = serialise_for_signing(&a).unwrap();
        assert_eq!(bytes1, bytes2, "serialisation must be deterministic");

        // Do it 100 more times for confidence
        for _ in 0..100 {
            let bytes_n = serialise_for_signing(&a).unwrap();
            assert_eq!(bytes1, bytes_n);
        }
    }

    #[test]
    fn nan_in_readings_rejected() {
        let mut a = make_test_attestation();
        a.layer_readings = vec![vec![1.0, f32::NAN, 3.0]];

        let result = serialise_for_signing(&a);
        assert!(result.is_err());
    }

    #[test]
    fn infinity_in_readings_rejected() {
        let mut a = make_test_attestation();
        a.layer_readings = vec![vec![1.0, f32::INFINITY, 3.0]];
        assert!(serialise_for_signing(&a).is_err());

        let mut a2 = make_test_attestation();
        a2.confidence = vec![f32::NEG_INFINITY, 0.8, 0.7];
        assert!(serialise_for_signing(&a2).is_err());
    }

    #[test]
    fn negative_zero_canonicalised() {
        let mut a1 = make_test_attestation();
        let mut a2 = make_test_attestation();

        a1.layer_readings = vec![vec![0.0]];
        a2.layer_readings = vec![vec![-0.0]];

        a1.confidence = vec![0.0];
        a2.confidence = vec![-0.0];

        a1.coverage_flags = vec![false];
        a2.coverage_flags = vec![false];

        let bytes1 = serialise_for_signing(&a1).unwrap();
        let bytes2 = serialise_for_signing(&a2).unwrap();
        assert_eq!(bytes1, bytes2, "-0.0 and 0.0 must serialise identically");
    }

    #[test]
    fn merkle_root_single_shard() {
        let shard = vec![1u8, 2, 3, 4];
        let root = merkle_root(&[shard.clone()]);
        // Single leaf: H(0x00 || data)
        let mut hasher = Sha256::new();
        hasher.update([0x00]);
        hasher.update(&shard);
        let expected: [u8; 32] = hasher.finalize().into();
        assert_eq!(root, expected);
    }

    #[test]
    fn merkle_root_four_leaves() {
        let shards: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 16]).collect();
        let root = merkle_root(&shards);

        // Hand-compute with domain separation:
        //   Leaf: H(0x00 || data)
        //   Internal: H(0x01 || left || right)
        let leaf_hash = |data: &[u8]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x00]);
            h.update(data);
            h.finalize().into()
        };
        let node_hash = |left: [u8; 32], right: [u8; 32]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x01]);
            h.update(left);
            h.update(right);
            h.finalize().into()
        };

        let h0 = leaf_hash(&shards[0]);
        let h1 = leaf_hash(&shards[1]);
        let h2 = leaf_hash(&shards[2]);
        let h3 = leaf_hash(&shards[3]);

        let h01 = node_hash(h0, h1);
        let h23 = node_hash(h2, h3);
        let expected = node_hash(h01, h23);

        assert_eq!(root, expected);
    }

    #[test]
    fn merkle_root_empty() {
        assert_eq!(merkle_root(&[]), got_core::sha256(b"merkle-empty"));
    }

    #[test]
    fn unknown_schema_version_rejected() {
        let mut a = make_test_attestation();
        a.schema_version = 999;
        let key = test_signing_key();

        // serialise_for_signing itself should reject unknown schema
        assert!(
            serialise_for_signing(&a).is_err(),
            "serialise_for_signing should reject unknown schema"
        );

        // assemble_and_sign should also reject
        let a2 = make_test_attestation();
        let mut a2_bad = a2;
        a2_bad.schema_version = 999;
        assert!(
            assemble_and_sign(a2_bad, &key).is_err(),
            "assemble_and_sign should reject unknown schema"
        );

        // verify should reject even if we manually craft a signature
        // (we can't serialise to sign, so just set a dummy signature)
        a.signature = [0xCC; 64];
        let result = verify(&a, &key.verifying_key());
        assert!(result.is_err(), "verify should reject unknown schema");
    }

    #[test]
    fn regularised_inner_product_serialisation() {
        let mut a = make_test_attestation();
        a.inner_product = InnerProduct::CausalRegularised { epsilon: 1e-6 };

        let bytes = serialise_for_signing(&a).unwrap();
        // Should be longer than the Causal variant (extra 4 bytes for epsilon)
        let baseline = {
            let mut a2 = make_test_attestation();
            a2.inner_product = InnerProduct::Causal;
            serialise_for_signing(&a2).unwrap()
        };
        assert_eq!(bytes.len(), baseline.len() + 4);
    }

    // --- Tests added during review ---

    #[test]
    fn wrong_key_verification_fails() {
        let attestation = make_test_attestation();
        let key_a = test_signing_key(); // seed [42; 32]

        let signed = assemble_and_sign(attestation, &key_a).unwrap();

        // Different key
        let key_b = SigningKey::from_bytes(&[99u8; 32]);
        assert!(
            verify(&signed, &key_b.verifying_key()).is_err(),
            "wrong key should not verify"
        );
    }

    #[test]
    fn tamper_each_field_detected() {
        let key = test_signing_key();

        // Tamper model_id
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.model_id = "evil-model".to_string();
        assert!(verify(&a, &key.verifying_key()).is_err(), "model_id tamper");

        // Tamper model_hash
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.model_hash.as_mut().unwrap()[0] ^= 0xFF;
        assert!(
            verify(&a, &key.verifying_key()).is_err(),
            "model_hash tamper"
        );

        // Tamper layer_readings
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.layer_readings[0][0] += 0.001;
        assert!(verify(&a, &key.verifying_key()).is_err(), "readings tamper");

        // Tamper confidence
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.confidence[0] += 0.001;
        assert!(
            verify(&a, &key.verifying_key()).is_err(),
            "confidence tamper"
        );

        // Tamper coverage_flags
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.coverage_flags[0] = !a.coverage_flags[0];
        assert!(
            verify(&a, &key.verifying_key()).is_err(),
            "coverage_flags tamper"
        );

        // Tamper divergence_flag
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.divergence_flag = !a.divergence_flag;
        assert!(
            verify(&a, &key.verifying_key()).is_err(),
            "divergence_flag tamper"
        );

        // Tamper input_hash
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.input_hash[0] ^= 0xFF;
        assert!(
            verify(&a, &key.verifying_key()).is_err(),
            "input_hash tamper"
        );

        // Tamper corpus_version
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.corpus_version = "tampered".to_string();
        assert!(
            verify(&a, &key.verifying_key()).is_err(),
            "corpus_version tamper"
        );

        // Tamper probe_version
        let mut a = assemble_and_sign(make_test_attestation(), &key).unwrap();
        a.probe_version = "tampered".to_string();
        assert!(
            verify(&a, &key.verifying_key()).is_err(),
            "probe_version tamper"
        );
    }

    #[test]
    fn nan_in_confidence_rejected() {
        let mut a = make_test_attestation();
        a.confidence = vec![0.9, f32::NAN, 0.7];
        assert!(serialise_for_signing(&a).is_err());
    }

    #[test]
    fn nan_in_epsilon_rejected() {
        let mut a = make_test_attestation();
        a.inner_product = InnerProduct::CausalRegularised { epsilon: f32::NAN };
        assert!(serialise_for_signing(&a).is_err());
    }

    #[test]
    fn merkle_root_two_leaves() {
        let shards = vec![vec![1u8; 8], vec![2u8; 8]];
        let root = merkle_root(&shards);

        let leaf_hash = |data: &[u8]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x00]);
            h.update(data);
            h.finalize().into()
        };
        let node_hash = |left: [u8; 32], right: [u8; 32]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x01]);
            h.update(left);
            h.update(right);
            h.finalize().into()
        };

        let h0 = leaf_hash(&shards[0]);
        let h1 = leaf_hash(&shards[1]);
        let expected = node_hash(h0, h1);
        assert_eq!(root, expected);
    }

    #[test]
    fn merkle_root_three_leaves_odd_duplication() {
        // 3 leaves exercises the odd-node duplication path
        let shards = vec![vec![1u8; 8], vec![2u8; 8], vec![3u8; 8]];
        let root = merkle_root(&shards);

        let leaf_hash = |data: &[u8]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x00]);
            h.update(data);
            h.finalize().into()
        };
        let node_hash = |left: [u8; 32], right: [u8; 32]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x01]);
            h.update(left);
            h.update(right);
            h.finalize().into()
        };

        let h0 = leaf_hash(&shards[0]);
        let h1 = leaf_hash(&shards[1]);
        let h2 = leaf_hash(&shards[2]);

        // Round 1: [h0, h1, h2] → odd, duplicate h2 → [h0, h1, h2, h2]
        let h01 = node_hash(h0, h1);
        let h22 = node_hash(h2, h2);
        // Round 2: [h01, h22] → even
        let expected = node_hash(h01, h22);

        assert_eq!(root, expected);
    }

    #[test]
    fn merkle_root_five_leaves_odd_duplication() {
        // 5 leaves: two rounds of odd-node handling
        let shards: Vec<Vec<u8>> = (0..5).map(|i| vec![i as u8; 8]).collect();
        let root = merkle_root(&shards);

        let leaf_hash = |data: &[u8]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x00]);
            h.update(data);
            h.finalize().into()
        };
        let node_hash = |left: [u8; 32], right: [u8; 32]| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update([0x01]);
            h.update(left);
            h.update(right);
            h.finalize().into()
        };

        let h0 = leaf_hash(&shards[0]);
        let h1 = leaf_hash(&shards[1]);
        let h2 = leaf_hash(&shards[2]);
        let h3 = leaf_hash(&shards[3]);
        let h4 = leaf_hash(&shards[4]);

        // Round 1: 5 nodes → odd → [h0,h1,h2,h3,h4,h4]
        let h01 = node_hash(h0, h1);
        let h23 = node_hash(h2, h3);
        let h44 = node_hash(h4, h4);

        // Round 2: 3 nodes → odd → [h01, h23, h44, h44]
        let h0123 = node_hash(h01, h23);
        let h4444 = node_hash(h44, h44);

        // Round 3: 2 nodes → even
        let expected = node_hash(h0123, h4444);

        assert_eq!(root, expected);
    }

    #[test]
    fn attestation_json_roundtrip() {
        // Tests the real-world path: serde_json serialize → deserialize → verify
        let key = test_signing_key();
        let signed = assemble_and_sign(make_test_attestation(), &key).unwrap();

        // Serialize to JSON (as the CLI does)
        let json = serde_json::to_string_pretty(&signed).unwrap();

        // Deserialize (as a verifier would)
        let deserialized: got_core::GeometricAttestation = serde_json::from_str(&json).unwrap();

        // Verify the deserialized attestation
        verify(&deserialized, &key.verifying_key())
            .expect("JSON round-tripped attestation should verify");

        // All fields should match
        assert_eq!(signed.model_id, deserialized.model_id);
        assert_eq!(signed.model_hash, deserialized.model_hash);
        assert_eq!(signed.input_hash, deserialized.input_hash);
        assert_eq!(signed.signature, deserialized.signature);
        assert_eq!(signed.layer_readings, deserialized.layer_readings);
        assert_eq!(signed.confidence, deserialized.confidence);
        assert_eq!(signed.coverage_flags, deserialized.coverage_flags);
        assert_eq!(signed.timestamp, deserialized.timestamp);
    }

    #[test]
    fn assemble_rejects_unknown_schema_version() {
        let mut a = make_test_attestation();
        a.schema_version = 999;
        let key = test_signing_key();
        let result = assemble_and_sign(a, &key);
        assert!(
            result.is_err(),
            "assemble_and_sign should reject unknown schema"
        );
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issues 22, 28, 34, 41)
    // -----------------------------------------------------------------------

    /// Issue #22 (S-1): verify() must return Err on invalid signature, not Ok(false).
    #[test]
    fn sec_verify_returns_err_on_bad_signature() {
        let key = test_signing_key();
        let mut signed = assemble_and_sign(make_test_attestation(), &key).unwrap();
        signed.timestamp += 1; // tamper

        let result = verify(&signed, &key.verifying_key());
        assert!(
            matches!(result, Err(AttestationError::SignatureInvalid)),
            "tampered attestation must return Err(SignatureInvalid), got: {result:?}"
        );
    }

    /// Issue #28 (S-7): assemble_and_sign should reject far-future timestamps.
    ///
    /// After the fix, timestamps > now + MAX_FUTURE_SECS should be rejected.
    #[test]
    fn sec_assemble_rejects_future_timestamp() {
        let key = test_signing_key();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut a = make_test_attestation();
        a.timestamp = now + 3600; // 1 hour in the future

        let result = assemble_and_sign(a, &key);
        assert!(result.is_err(), "future timestamps should be rejected");
        assert!(
            matches!(result, Err(AttestationError::TimestampFuture { .. })),
            "expected TimestampFuture, got: {result:?}"
        );
    }

    /// Issue #34 (S-13): assemble_and_sign should reject oversized string fields.
    #[test]
    fn sec_assemble_rejects_oversized_model_id() {
        let key = test_signing_key();
        let mut a = make_test_attestation();
        a.model_id = "x".repeat(1_000_000); // 1 MB model_id

        let result = assemble_and_sign(a, &key);
        assert!(
            matches!(
                result,
                Err(AttestationError::FieldTooLarge {
                    field: "model_id",
                    ..
                })
            ),
            "1 MB model_id should be rejected, got: {result:?}"
        );
    }

    /// Issue #41 (S-20): assemble_and_sign should reject oversized arrays.
    #[test]
    fn sec_assemble_rejects_oversized_layer_readings() {
        let key = test_signing_key();
        let mut a = make_test_attestation();
        // 2000 layers, each with one reading
        a.layer_readings = (0..2000).map(|i| vec![i as f32]).collect();
        a.confidence = vec![0.9; 2000];
        a.coverage_flags = vec![false; 2000];

        let result = assemble_and_sign(a, &key);
        assert!(
            matches!(
                result,
                Err(AttestationError::FieldTooLarge {
                    field: "layer_readings",
                    ..
                })
            ),
            "2000 layers should be rejected, got: {result:?}"
        );
    }
}
