// ---------------------------------------------------------------------------
// Agent Certificates — Issue #48 (F-3).
//
// Binds an Ed25519 public key to an identity (name, roles, policy) with
// an expiry window and an issuer signature.  This is the building block
// for a minimal PKI over the trust registry.
//
// The certificate is signed using the same deterministic canonical
// serialisation approach as `serialise_for_signing` — length-prefixed LE
// fields, no serde dependency in the signing path.
// ---------------------------------------------------------------------------

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::WireError;

// ---------------------------------------------------------------------------
// AgentCertificate
// ---------------------------------------------------------------------------

/// A signed certificate binding a public key to an agent identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCertificate {
    /// Human-readable subject name.
    pub subject_name: String,
    /// The agent's Ed25519 public key (32 bytes).
    #[serde(with = "hex_key")]
    pub subject_public_key: VerifyingKey,
    /// The issuing CA's Ed25519 public key (32 bytes).
    #[serde(with = "hex_key")]
    pub issuer_public_key: VerifyingKey,
    /// Certificate is not valid before this Unix timestamp.
    pub not_before: u64,
    /// Certificate is not valid after this Unix timestamp.
    pub not_after: u64,
    /// Roles authorised for this agent.
    pub roles: Vec<String>,
    /// Maximum geometry drift this agent is permitted.
    pub max_drift_accepted: f32,
    /// If set, the agent MUST attest for this model hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_model_hash: Option<[u8; 32]>,
    /// Ed25519 signature by the issuer over canonical fields.
    #[serde(with = "hex_sig")]
    pub signature: [u8; 64],
}

/// Canonical bytes for signing — deterministic, length-prefixed, LE.
fn canonical_bytes(cert: &AgentCertificate) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);

    // subject_name
    let name = cert.subject_name.as_bytes();
    buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
    buf.extend_from_slice(name);

    // subject_public_key
    buf.extend_from_slice(cert.subject_public_key.as_bytes());

    // issuer_public_key
    buf.extend_from_slice(cert.issuer_public_key.as_bytes());

    // not_before, not_after
    buf.extend_from_slice(&cert.not_before.to_le_bytes());
    buf.extend_from_slice(&cert.not_after.to_le_bytes());

    // roles
    buf.extend_from_slice(&(cert.roles.len() as u32).to_le_bytes());
    for role in &cert.roles {
        let r = role.as_bytes();
        buf.extend_from_slice(&(r.len() as u32).to_le_bytes());
        buf.extend_from_slice(r);
    }

    // max_drift_accepted
    let drift = if cert.max_drift_accepted == 0.0 {
        0.0f32
    } else {
        cert.max_drift_accepted
    };
    buf.extend_from_slice(&drift.to_le_bytes());

    // expected_model_hash: tag byte + optional 32 bytes
    match cert.expected_model_hash {
        Some(hash) => {
            buf.push(0x01);
            buf.extend_from_slice(&hash);
        }
        None => {
            buf.push(0x00);
        }
    }

    buf
}

/// Issue (sign) a certificate for a subject using the issuer's signing key.
///
/// The certificate's `signature` field is populated by this function.
pub fn sign_certificate(
    subject_name: &str,
    subject_public_key: &VerifyingKey,
    roles: Vec<String>,
    max_drift_accepted: f32,
    expected_model_hash: Option<[u8; 32]>,
    not_before: u64,
    not_after: u64,
    issuer_signing_key: &SigningKey,
) -> AgentCertificate {
    let mut cert = AgentCertificate {
        subject_name: subject_name.to_string(),
        subject_public_key: *subject_public_key,
        issuer_public_key: issuer_signing_key.verifying_key(),
        not_before,
        not_after,
        roles,
        max_drift_accepted,
        expected_model_hash,
        signature: [0u8; 64],
    };

    let payload = canonical_bytes(&cert);
    let sig: Signature = issuer_signing_key.sign(&payload);
    cert.signature = sig.to_bytes();
    cert
}

/// Verify that a certificate's signature is valid against its issuer key.
pub fn verify_certificate(cert: &AgentCertificate) -> Result<(), WireError> {
    let payload = canonical_bytes(cert);
    let sig = Signature::from_bytes(&cert.signature);
    cert.issuer_public_key
        .verify(&payload, &sig)
        .map_err(|_| WireError::CertificateSignatureInvalid)
}

/// Check if a certificate is valid at a given Unix timestamp.
pub fn is_valid_at(cert: &AgentCertificate, now_unix: u64) -> bool {
    now_unix >= cert.not_before && now_unix <= cert.not_after
}

/// Compute a unique certificate fingerprint (SHA-256 of canonical bytes + signature).
pub fn certificate_fingerprint(cert: &AgentCertificate) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(canonical_bytes(cert));
    hasher.update(cert.signature);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

// ---------------------------------------------------------------------------
// Key Rotation — Issue #50 (F-5).
// ---------------------------------------------------------------------------

/// A key rotation record: old key ↔ new key with mutual cross-signatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotation {
    /// The agent's old (outgoing) public key.
    #[serde(with = "hex_key")]
    pub old_public_key: VerifyingKey,
    /// The agent's new (incoming) public key.
    #[serde(with = "hex_key")]
    pub new_public_key: VerifyingKey,
    /// Certificate for the new key, issued by a CA.
    pub new_certificate: AgentCertificate,
    /// Unix timestamp of the rotation.
    pub timestamp: u64,
    /// Old key signs: canonical_rotation_payload.
    #[serde(with = "hex_sig")]
    pub old_key_signature: [u8; 64],
    /// New key signs: canonical_rotation_payload.
    #[serde(with = "hex_sig")]
    pub new_key_signature: [u8; 64],
}

/// Canonical bytes for the rotation record (excluding signatures).
fn rotation_canonical_bytes(
    old_pk: &VerifyingKey,
    new_pk: &VerifyingKey,
    cert_fingerprint: &[u8; 32],
    timestamp: u64,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(b"GOT-KEY-ROTATION-V1");
    buf.extend_from_slice(old_pk.as_bytes());
    buf.extend_from_slice(new_pk.as_bytes());
    buf.extend_from_slice(cert_fingerprint);
    buf.extend_from_slice(&timestamp.to_le_bytes());
    buf
}

/// Create a key rotation record with mutual cross-signatures.
///
/// Both the old and new signing keys sign the same canonical payload,
/// proving possession of both keys. The `new_certificate` must already
/// be issued by a CA.
pub fn create_rotation(
    old_signing_key: &SigningKey,
    new_signing_key: &SigningKey,
    new_certificate: AgentCertificate,
    timestamp: u64,
) -> KeyRotation {
    let old_pk = old_signing_key.verifying_key();
    let new_pk = new_signing_key.verifying_key();
    let cert_fp = certificate_fingerprint(&new_certificate);
    let payload = rotation_canonical_bytes(&old_pk, &new_pk, &cert_fp, timestamp);

    use ed25519_dalek::Signer;
    let old_sig: Signature = old_signing_key.sign(&payload);
    let new_sig: Signature = new_signing_key.sign(&payload);

    KeyRotation {
        old_public_key: old_pk,
        new_public_key: new_pk,
        new_certificate,
        timestamp,
        old_key_signature: old_sig.to_bytes(),
        new_key_signature: new_sig.to_bytes(),
    }
}

/// Verify a key rotation record.
///
/// Checks:
///   1. Both cross-signatures are valid.
///   2. The new certificate is valid (signature from issuer).
///   3. The new certificate's subject key matches `new_public_key`.
pub fn verify_rotation(rotation: &KeyRotation) -> Result<(), WireError> {
    // Verify the new certificate first.
    verify_certificate(&rotation.new_certificate)?;

    // Certificate subject must match the new key.
    if rotation.new_certificate.subject_public_key != rotation.new_public_key {
        return Err(WireError::CertificateSubjectMismatch);
    }

    // Verify cross-signatures.
    let cert_fp = certificate_fingerprint(&rotation.new_certificate);
    let payload = rotation_canonical_bytes(
        &rotation.old_public_key,
        &rotation.new_public_key,
        &cert_fp,
        rotation.timestamp,
    );

    let old_sig = Signature::from_bytes(&rotation.old_key_signature);
    rotation
        .old_public_key
        .verify(&payload, &old_sig)
        .map_err(|_| WireError::RotationOldSignatureInvalid)?;

    let new_sig = Signature::from_bytes(&rotation.new_key_signature);
    rotation
        .new_public_key
        .verify(&payload, &new_sig)
        .map_err(|_| WireError::RotationNewSignatureInvalid)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Certificate Revocation List — Issue #52 (F-7).
// ---------------------------------------------------------------------------

/// A single entry in a certificate revocation list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevokedEntry {
    /// Fingerprint of the revoked certificate.
    pub certificate_fingerprint: [u8; 32],
    /// Unix timestamp of revocation.
    pub revocation_time: u64,
    /// Human-readable reason.
    pub reason: String,
}

/// A signed certificate revocation list (CRL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateRevocationList {
    /// Issuing CA's public key.
    #[serde(with = "hex_key")]
    pub issuer: VerifyingKey,
    /// Revoked certificate entries.
    pub entries: Vec<RevokedEntry>,
    /// When this CRL was issued.
    pub issued_at: u64,
    /// When the next CRL update is expected.
    pub next_update: u64,
    /// Ed25519 signature over canonical CRL bytes.
    #[serde(with = "hex_sig")]
    pub signature: [u8; 64],
}

/// Canonical bytes for CRL signing.
fn crl_canonical_bytes(crl: &CertificateRevocationList) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(b"GOT-CRL-V1");
    buf.extend_from_slice(crl.issuer.as_bytes());
    buf.extend_from_slice(&crl.issued_at.to_le_bytes());
    buf.extend_from_slice(&crl.next_update.to_le_bytes());
    buf.extend_from_slice(&(crl.entries.len() as u32).to_le_bytes());
    for entry in &crl.entries {
        buf.extend_from_slice(&entry.certificate_fingerprint);
        buf.extend_from_slice(&entry.revocation_time.to_le_bytes());
        let reason = entry.reason.as_bytes();
        buf.extend_from_slice(&(reason.len() as u32).to_le_bytes());
        buf.extend_from_slice(reason);
    }
    buf
}

/// Sign a CRL using the CA's signing key.
pub fn sign_crl(
    entries: Vec<RevokedEntry>,
    issued_at: u64,
    next_update: u64,
    ca_signing_key: &SigningKey,
) -> CertificateRevocationList {
    let mut crl = CertificateRevocationList {
        issuer: ca_signing_key.verifying_key(),
        entries,
        issued_at,
        next_update,
        signature: [0u8; 64],
    };

    let payload = crl_canonical_bytes(&crl);
    use ed25519_dalek::Signer;
    let sig: Signature = ca_signing_key.sign(&payload);
    crl.signature = sig.to_bytes();
    crl
}

/// Verify a CRL's signature against a CA public key.
pub fn verify_crl(crl: &CertificateRevocationList, ca_key: &VerifyingKey) -> Result<(), WireError> {
    if crl.issuer != *ca_key {
        return Err(WireError::CrlIssuerMismatch);
    }
    let payload = crl_canonical_bytes(crl);
    let sig = Signature::from_bytes(&crl.signature);
    ca_key
        .verify(&payload, &sig)
        .map_err(|_| WireError::CrlSignatureInvalid)
}

/// Check if a certificate fingerprint appears in the CRL.
pub fn is_revoked(crl: &CertificateRevocationList, cert_fingerprint: &[u8; 32]) -> bool {
    crl.entries
        .iter()
        .any(|e| e.certificate_fingerprint == *cert_fingerprint)
}

// ---------------------------------------------------------------------------
// Serde helpers for hex-encoded keys and signatures
// ---------------------------------------------------------------------------

mod hex_key {
    use ed25519_dalek::VerifyingKey;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(key: &VerifyingKey, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = key.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<VerifyingKey, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 64 {
            return Err(serde::de::Error::custom(format!(
                "expected 64 hex chars for key, got {}",
                s.len()
            )));
        }
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::de::Error::custom)?;
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        VerifyingKey::from_bytes(&arr).map_err(serde::de::Error::custom)
    }
}

mod hex_sig {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(sig: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex: String = sig.iter().map(|b| format!("{b:02x}")).collect();
        serializer.serialize_str(&hex)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() != 128 {
            return Err(serde::de::Error::custom(format!(
                "expected 128 hex chars for signature, got {}",
                s.len()
            )));
        }
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::de::Error::custom)?;
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&bytes);
        Ok(arr)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn ca_key() -> SigningKey {
        SigningKey::from_bytes(&[0xCA; 32])
    }

    fn agent_key() -> SigningKey {
        SigningKey::from_bytes(&[0xAA; 32])
    }

    fn agent_key_2() -> SigningKey {
        SigningKey::from_bytes(&[0xBB; 32])
    }

    // -----------------------------------------------------------------------
    // Certificate tests (Issue #48)
    // -----------------------------------------------------------------------

    #[test]
    fn sign_verify_roundtrip() {
        let ca = ca_key();
        let agent = agent_key();
        let cert = sign_certificate(
            "alice",
            &agent.verifying_key(),
            vec!["producer".into(), "verifier".into()],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        assert_eq!(cert.subject_name, "alice");
        assert_eq!(cert.subject_public_key, agent.verifying_key());
        assert_eq!(cert.issuer_public_key, ca.verifying_key());

        verify_certificate(&cert).expect("valid cert should verify");
    }

    #[test]
    fn tampered_cert_rejected() {
        let ca = ca_key();
        let agent = agent_key();
        let mut cert = sign_certificate(
            "alice",
            &agent.verifying_key(),
            vec!["producer".into()],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        // Tamper: change subject name
        cert.subject_name = "mallory".into();
        let result = verify_certificate(&cert);
        assert!(result.is_err(), "tampered cert must be rejected");
    }

    #[test]
    fn wrong_issuer_rejected() {
        let ca = ca_key();
        let agent = agent_key();
        let cert = sign_certificate(
            "alice",
            &agent.verifying_key(),
            vec![],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        // Pretend a different CA issued it
        let mut fake = cert.clone();
        fake.issuer_public_key = agent.verifying_key(); // wrong issuer
        let result = verify_certificate(&fake);
        assert!(result.is_err(), "wrong issuer key must fail verification");
    }

    #[test]
    fn validity_window() {
        let ca = ca_key();
        let agent = agent_key();
        let cert = sign_certificate(
            "alice",
            &agent.verifying_key(),
            vec![],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        assert!(!is_valid_at(&cert, 999), "before not_before");
        assert!(is_valid_at(&cert, 1000), "at not_before");
        assert!(is_valid_at(&cert, 1500), "within window");
        assert!(is_valid_at(&cert, 2000), "at not_after");
        assert!(!is_valid_at(&cert, 2001), "after not_after");
    }

    #[test]
    fn cert_json_roundtrip() {
        let ca = ca_key();
        let agent = agent_key();
        let cert = sign_certificate(
            "alice",
            &agent.verifying_key(),
            vec!["producer".into()],
            0.05,
            Some([0x42; 32]),
            1000,
            2000,
            &ca,
        );

        let json = serde_json::to_string_pretty(&cert).unwrap();
        let cert2: AgentCertificate = serde_json::from_str(&json).unwrap();

        assert_eq!(cert.subject_name, cert2.subject_name);
        assert_eq!(cert.subject_public_key, cert2.subject_public_key);
        assert_eq!(cert.issuer_public_key, cert2.issuer_public_key);
        assert_eq!(cert.not_before, cert2.not_before);
        assert_eq!(cert.not_after, cert2.not_after);
        assert_eq!(cert.roles, cert2.roles);
        assert_eq!(cert.signature, cert2.signature);
        assert_eq!(cert.expected_model_hash, cert2.expected_model_hash);

        // Must still verify after round-trip
        verify_certificate(&cert2).expect("deserialized cert should still verify");
    }

    #[test]
    fn deterministic_canonical_bytes() {
        let ca = ca_key();
        let agent = agent_key();
        let cert = sign_certificate(
            "alice",
            &agent.verifying_key(),
            vec!["producer".into()],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        let b1 = canonical_bytes(&cert);
        let b2 = canonical_bytes(&cert);
        assert_eq!(b1, b2, "canonical bytes must be deterministic");
    }

    #[test]
    fn cert_with_model_hash() {
        let ca = ca_key();
        let agent = agent_key();
        let model_hash = [0xDE; 32];
        let cert = sign_certificate(
            "alice",
            &agent.verifying_key(),
            vec![],
            0.05,
            Some(model_hash),
            1000,
            2000,
            &ca,
        );

        verify_certificate(&cert).unwrap();
        assert_eq!(cert.expected_model_hash, Some(model_hash));
    }

    // -----------------------------------------------------------------------
    // Key Rotation tests (Issue #50)
    // -----------------------------------------------------------------------

    #[test]
    fn rotation_roundtrip() {
        let ca = ca_key();
        let old_key = agent_key();
        let new_key = agent_key_2();

        let new_cert = sign_certificate(
            "alice",
            &new_key.verifying_key(),
            vec!["producer".into()],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        let rotation = create_rotation(&old_key, &new_key, new_cert, 1500);
        verify_rotation(&rotation).expect("valid rotation should verify");
    }

    #[test]
    fn rotation_bad_old_signature_rejected() {
        let ca = ca_key();
        let old_key = agent_key();
        let new_key = agent_key_2();
        let fake_key = SigningKey::from_bytes(&[0xCC; 32]);

        let new_cert = sign_certificate(
            "alice",
            &new_key.verifying_key(),
            vec![],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        // Create rotation with wrong old key
        let rotation = create_rotation(&fake_key, &new_key, new_cert, 1500);
        // Manually patch old_public_key to be the real old key (mismatched sig)
        let mut tampered = rotation;
        tampered.old_public_key = old_key.verifying_key();
        let result = verify_rotation(&tampered);
        assert!(result.is_err(), "mismatched old key signature must fail");
    }

    #[test]
    fn rotation_bad_cert_subject_rejected() {
        let ca = ca_key();
        let old_key = agent_key();
        let new_key = agent_key_2();
        let other_key = SigningKey::from_bytes(&[0xDD; 32]);

        // Issue cert for OTHER key, not new key
        let cert_for_other = sign_certificate(
            "alice",
            &other_key.verifying_key(),
            vec![],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        let rotation = create_rotation(&old_key, &new_key, cert_for_other, 1500);
        let result = verify_rotation(&rotation);
        assert!(
            result.is_err(),
            "cert subject mismatch with new key must fail"
        );
    }

    #[test]
    fn rotation_json_roundtrip() {
        let ca = ca_key();
        let old_key = agent_key();
        let new_key = agent_key_2();

        let new_cert = sign_certificate(
            "alice",
            &new_key.verifying_key(),
            vec!["producer".into()],
            0.05,
            None,
            1000,
            2000,
            &ca,
        );

        let rotation = create_rotation(&old_key, &new_key, new_cert, 1500);
        let json = serde_json::to_string_pretty(&rotation).unwrap();
        let rotation2: KeyRotation = serde_json::from_str(&json).unwrap();

        verify_rotation(&rotation2).expect("deserialized rotation should verify");
    }

    // -----------------------------------------------------------------------
    // CRL tests (Issue #52)
    // -----------------------------------------------------------------------

    #[test]
    fn crl_sign_verify_roundtrip() {
        let ca = ca_key();

        let entries = vec![RevokedEntry {
            certificate_fingerprint: [0xDE; 32],
            revocation_time: 1500,
            reason: "key-compromise".into(),
        }];

        let crl = sign_crl(entries, 1500, 2500, &ca);
        verify_crl(&crl, &ca.verifying_key()).expect("valid CRL should verify");
    }

    #[test]
    fn crl_wrong_ca_rejected() {
        let ca = ca_key();
        let other = agent_key();

        let entries = vec![RevokedEntry {
            certificate_fingerprint: [0xDE; 32],
            revocation_time: 1500,
            reason: "test".into(),
        }];

        let crl = sign_crl(entries, 1500, 2500, &ca);
        let result = verify_crl(&crl, &other.verifying_key());
        assert!(result.is_err(), "wrong CA key must fail CRL verification");
    }

    #[test]
    fn crl_tampered_rejected() {
        let ca = ca_key();

        let entries = vec![RevokedEntry {
            certificate_fingerprint: [0xDE; 32],
            revocation_time: 1500,
            reason: "test".into(),
        }];

        let mut crl = sign_crl(entries, 1500, 2500, &ca);
        // Tamper: add another entry
        crl.entries.push(RevokedEntry {
            certificate_fingerprint: [0xBE; 32],
            revocation_time: 1600,
            reason: "injected".into(),
        });
        let result = verify_crl(&crl, &ca.verifying_key());
        assert!(result.is_err(), "tampered CRL must fail verification");
    }

    #[test]
    fn is_revoked_check() {
        let ca = ca_key();
        let fp_revoked = [0xDE; 32];
        let fp_clean = [0xAB; 32];

        let entries = vec![RevokedEntry {
            certificate_fingerprint: fp_revoked,
            revocation_time: 1500,
            reason: "key-compromise".into(),
        }];

        let crl = sign_crl(entries, 1500, 2500, &ca);
        assert!(is_revoked(&crl, &fp_revoked));
        assert!(!is_revoked(&crl, &fp_clean));
    }

    #[test]
    fn crl_json_roundtrip() {
        let ca = ca_key();
        let entries = vec![
            RevokedEntry {
                certificate_fingerprint: [0xDE; 32],
                revocation_time: 1500,
                reason: "test".into(),
            },
            RevokedEntry {
                certificate_fingerprint: [0xBE; 32],
                revocation_time: 1600,
                reason: "test2".into(),
            },
        ];

        let crl = sign_crl(entries, 1500, 2500, &ca);
        let json = serde_json::to_string_pretty(&crl).unwrap();
        let crl2: CertificateRevocationList = serde_json::from_str(&json).unwrap();
        verify_crl(&crl2, &ca.verifying_key()).expect("deserialized CRL should verify");
        assert_eq!(crl2.entries.len(), 2);
    }

    #[test]
    fn crl_empty_is_valid() {
        let ca = ca_key();
        let crl = sign_crl(vec![], 1500, 2500, &ca);
        verify_crl(&crl, &ca.verifying_key()).expect("empty CRL should verify");
        assert!(!is_revoked(&crl, &[0xDE; 32]));
    }
}
