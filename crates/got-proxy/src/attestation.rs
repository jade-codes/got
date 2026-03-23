// ---------------------------------------------------------------------------
// BehavioralAttestation — cryptographically signed snapshot of a proxy
// session's value space state.
//
// Structurally distinct from GeometricAttestation.  Schema prefix "B1"
// prevents confusion between the two attestation types.
// ---------------------------------------------------------------------------

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

use got_core::manifold::{CurvatureReading, DensityReading};

use crate::deviation::{DeviationReport, DeviationVerdict};
use crate::ProxyError;

/// Schema version for behavioral attestations.
pub const BEHAVIORAL_SCHEMA_VERSION: &str = "B1";

/// Maximum allowed clock skew (seconds) for attestation timestamps.
const MAX_FUTURE_SECS: u64 = 300;

/// Type of behavioral attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationType {
    /// Initial baseline establishment.
    Baseline,
    /// Deviation alert triggered during monitoring.
    Alert,
    /// Session start marker.
    SessionStart,
    /// Periodic snapshot (no deviation).
    Snapshot,
}

/// Summary statistics embedded in the attestation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationSummary {
    /// Top value terms by EWMA score, sorted descending.
    pub top_values: Vec<(String, f64)>,
    /// Overall coherence score from the latest analysis.
    pub coherence_score: f64,
    /// Cumulative profile drift from the initial baseline.
    pub cumulative_drift: f64,
}

/// A cryptographically signed behavioral attestation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralAttestation {
    /// Always "B1" for behavioral attestations.
    pub schema_version: String,
    /// Identifier of the closed-source model being monitored.
    pub target_model_id: String,
    /// SHA-256 of the reference geometry's Gram matrix.
    #[serde(with = "got_core::hex32")]
    pub reference_geometry_hash: [u8; 32],
    /// Type of this attestation.
    pub attestation_type: AttestationType,
    /// Number of observations processed at time of attestation.
    pub observation_count: u64,
    /// Monotonic sequence number within this session.
    pub sequence_number: u64,
    /// Unix UTC seconds.
    pub timestamp: u64,
    /// SHA-256 of the value space snapshot.
    #[serde(with = "got_core::hex32")]
    pub value_space_hash: [u8; 32],
    /// SHA-256 of the previous attestation in this chain (None for first).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "got_core::optional_hex32"
    )]
    pub parent_hash: Option<[u8; 32]>,
    /// Summary statistics.
    pub summary: AttestationSummary,
    /// Deviation report (present only for Alert attestations).
    pub deviation: Option<DeviationReport>,
    /// Manifold density reading at snapshot time. None if insufficient activations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density_reading: Option<DensityReading>,
    /// Manifold curvature reading at snapshot time. None if insufficient activations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curvature_reading: Option<CurvatureReading>,
    /// Ed25519 signature over canonical serialisation of all preceding fields.
    #[serde(with = "got_core::hex64")]
    pub signature: [u8; 64],
}

/// Canonical serialisation for signing.
///
/// Deterministic: same attestation → identical bytes. Always.
/// Includes all fields except `signature`.
pub fn serialise_for_signing(a: &BehavioralAttestation) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);

    // Schema version
    buf.extend_from_slice(a.schema_version.as_bytes());
    buf.push(0); // null separator

    // Target model ID
    buf.extend_from_slice(a.target_model_id.as_bytes());
    buf.push(0);

    // Reference geometry hash
    buf.extend_from_slice(&a.reference_geometry_hash);

    // Attestation type tag
    buf.push(match a.attestation_type {
        AttestationType::Baseline => 0,
        AttestationType::Alert => 1,
        AttestationType::SessionStart => 2,
        AttestationType::Snapshot => 3,
    });

    // Observation count, sequence number, timestamp
    buf.extend_from_slice(&a.observation_count.to_le_bytes());
    buf.extend_from_slice(&a.sequence_number.to_le_bytes());
    buf.extend_from_slice(&a.timestamp.to_le_bytes());

    // Value space hash
    buf.extend_from_slice(&a.value_space_hash);

    // Parent hash (32 zero bytes if None)
    match &a.parent_hash {
        Some(h) => {
            buf.push(1);
            buf.extend_from_slice(h);
        }
        None => {
            buf.push(0);
        }
    }

    // Summary: top values
    buf.extend_from_slice(&(a.summary.top_values.len() as u32).to_le_bytes());
    for (term, score) in &a.summary.top_values {
        buf.extend_from_slice(&(term.len() as u32).to_le_bytes());
        buf.extend_from_slice(term.as_bytes());
        buf.extend_from_slice(&score.to_le_bytes());
    }
    buf.extend_from_slice(&a.summary.coherence_score.to_le_bytes());
    buf.extend_from_slice(&a.summary.cumulative_drift.to_le_bytes());

    // Deviation report presence
    match &a.deviation {
        Some(dev) => {
            buf.push(1);
            buf.extend_from_slice(&dev.term_score.to_le_bytes());
            buf.extend_from_slice(&dev.profile_drift.to_le_bytes());
            buf.extend_from_slice(&dev.relationship_score.to_le_bytes());
            buf.extend_from_slice(&dev.combined_score.to_le_bytes());
            buf.push(match dev.verdict {
                DeviationVerdict::WithinBaseline => 0,
                DeviationVerdict::Drifting => 1,
                DeviationVerdict::Deviated => 2,
            });
        }
        None => {
            buf.push(0);
        }
    }

    // Density reading presence
    match &a.density_reading {
        Some(dr) => {
            buf.push(1);
            buf.extend_from_slice(&(dr.points.len() as u32).to_le_bytes());
            for p in &dr.points {
                buf.extend_from_slice(&p.log_density.to_le_bytes());
                buf.extend_from_slice(&p.intrinsic_dim.to_le_bytes());
            }
            buf.extend_from_slice(&dr.mean_intrinsic_dim.to_le_bytes());
            buf.extend_from_slice(&dr.std_intrinsic_dim.to_le_bytes());
            buf.extend_from_slice(&dr.mean_log_density.to_le_bytes());
            buf.extend_from_slice(&dr.k.to_le_bytes());
            buf.extend_from_slice(&dr.num_degenerate.to_le_bytes());
        }
        None => buf.push(0),
    }

    // Curvature reading presence
    match &a.curvature_reading {
        Some(cr) => {
            buf.push(1);
            buf.extend_from_slice(&(cr.points.len() as u32).to_le_bytes());
            for p in &cr.points {
                buf.extend_from_slice(&p.sectional_curvature.to_le_bytes());
                buf.extend_from_slice(&p.num_triangles.to_le_bytes());
            }
            buf.extend_from_slice(&cr.mean_curvature.to_le_bytes());
            buf.extend_from_slice(&cr.std_curvature.to_le_bytes());
            match cr.curvature_uncertainty_correlation {
                Some(r) => {
                    buf.push(1);
                    buf.extend_from_slice(&r.to_le_bytes());
                }
                None => buf.push(0),
            }
            buf.extend_from_slice(&cr.k.to_le_bytes());
            buf.extend_from_slice(&cr.num_degenerate.to_le_bytes());
        }
        None => buf.push(0),
    }

    buf
}

/// Compute SHA-256 hash of an attestation's canonical serialisation.
pub fn attestation_hash(a: &BehavioralAttestation) -> [u8; 32] {
    let bytes = serialise_for_signing(a);
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Sign a behavioral attestation. Writes the Ed25519 signature into the struct.
pub fn sign_attestation(
    mut attestation: BehavioralAttestation,
    signing_key: &SigningKey,
) -> Result<BehavioralAttestation, ProxyError> {
    // Validate schema version
    if attestation.schema_version != BEHAVIORAL_SCHEMA_VERSION {
        return Err(ProxyError::InvalidSchemaVersion(
            attestation.schema_version.clone(),
        ));
    }

    // Reject far-future timestamps
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if attestation.timestamp > now + MAX_FUTURE_SECS {
        return Err(ProxyError::TimestampFuture {
            delta: attestation.timestamp - now,
            max: MAX_FUTURE_SECS,
        });
    }

    let bytes = serialise_for_signing(&attestation);
    let sig = signing_key.sign(&bytes);
    attestation.signature = sig.to_bytes();
    Ok(attestation)
}

/// Verify a behavioral attestation's signature.
pub fn verify_attestation(
    attestation: &BehavioralAttestation,
    verifying_key: &VerifyingKey,
) -> Result<(), ProxyError> {
    let bytes = serialise_for_signing(attestation);
    let sig = ed25519_dalek::Signature::from_bytes(&attestation.signature);
    verifying_key
        .verify(&bytes, &sig)
        .map_err(|_| ProxyError::SignatureInvalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn make_test_attestation() -> BehavioralAttestation {
        BehavioralAttestation {
            schema_version: BEHAVIORAL_SCHEMA_VERSION.into(),
            target_model_id: "gpt-4".into(),
            reference_geometry_hash: [0xAA; 32],
            attestation_type: AttestationType::Baseline,
            observation_count: 50,
            sequence_number: 1,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            value_space_hash: [0xBB; 32],
            parent_hash: None,
            summary: AttestationSummary {
                top_values: vec![("honesty".into(), 0.8), ("courage".into(), 0.5)],
                coherence_score: 0.9,
                cumulative_drift: 0.05,
            },
            deviation: None,
            density_reading: None,
            curvature_reading: None,
            signature: [0; 64],
        }
    }

    #[test]
    fn serialisation_determinism() {
        let a = make_test_attestation();
        let s1 = serialise_for_signing(&a);
        let s2 = serialise_for_signing(&a);
        assert_eq!(s1, s2);
    }

    #[test]
    fn sign_and_verify() {
        let sk = SigningKey::from_bytes(&[42u8; 32]);
        let vk = sk.verifying_key();

        let a = make_test_attestation();
        let signed = sign_attestation(a, &sk).unwrap();
        assert!(verify_attestation(&signed, &vk).is_ok());
    }

    #[test]
    fn tampered_attestation_fails_verify() {
        let sk = SigningKey::from_bytes(&[42u8; 32]);
        let vk = sk.verifying_key();

        let a = make_test_attestation();
        let mut signed = sign_attestation(a, &sk).unwrap();
        signed.observation_count = 999; // tamper
        assert!(verify_attestation(&signed, &vk).is_err());
    }

    #[test]
    fn wrong_key_fails_verify() {
        let sk1 = SigningKey::from_bytes(&[42u8; 32]);
        let sk2 = SigningKey::from_bytes(&[99u8; 32]);
        let vk2 = sk2.verifying_key();

        let a = make_test_attestation();
        let signed = sign_attestation(a, &sk1).unwrap();
        assert!(verify_attestation(&signed, &vk2).is_err());
    }
}
