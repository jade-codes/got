// ---------------------------------------------------------------------------
// Trust Registry — Phase 10, §10.8.
//
// TOML configuration file mapping agent identities to public keys and policy.
// Drift thresholds are LOCAL policy — never sent on the wire.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::Path;

use ed25519_dalek::VerifyingKey;

use crate::certificate::{
    certificate_fingerprint, is_revoked, is_valid_at, verify_certificate, AgentCertificate,
    CertificateRevocationList,
};
use crate::domain::{Domain, DomainPattern, DomainScope, InteractionMode, PermittedDomain};
use crate::governance::{GovernanceEntryToml, GovernanceTable};
use crate::WireError;

/// A trust registry mapping agent IDs to public keys and local policy.
#[derive(Debug, Clone)]
pub struct TrustRegistry {
    /// Agent entries keyed by agent_id (SHA-256 of public key).
    pub agents: HashMap<[u8; 32], AgentEntry>,
    /// Maximum chain length we'll accept from any agent.
    pub max_chain_length: usize,
    /// Maximum age of an exchange envelope timestamp (seconds).
    pub max_envelope_age_secs: u64,
    /// Maximum allowed age of an attestation's own timestamp (seconds).
    /// Defence-in-depth: even if the envelope is fresh, an old attestation
    /// should not be accepted indefinitely. Defaults to 3600 (1 hour).
    pub max_attestation_age_secs: u64,
    /// Trusted CA public keys. When present, certificates are validated
    /// against these keys. Empty = PoC mode (no certificate enforcement).
    pub ca_public_keys: Vec<VerifyingKey>,
    /// Loaded certificate revocation lists.
    pub crls: Vec<CertificateRevocationList>,
}

/// An agent entry in the trust registry.
#[derive(Debug, Clone)]
pub struct AgentEntry {
    /// Human-readable name.
    pub name: String,
    /// Ed25519 verifying (public) key.
    pub public_key: VerifyingKey,
    /// Agent ID = SHA-256(public_key bytes).
    pub agent_id: [u8; 32],
    /// Maximum geometry drift we'll accept from this agent (LOCAL policy).
    pub max_drift_accepted: f32,
    /// Roles this agent is authorised for.
    pub roles: Vec<String>,
    /// If set, the agent's attestations MUST carry this model_hash.
    /// Prevents an agent from attesting for an unexpected model.
    /// None = any model_hash accepted (PoC default).
    pub expected_model_hash: Option<[u8; 32]>,
    /// Optional certificate binding this key to an identity.
    /// When present and the registry has CA keys, the certificate is validated
    /// on add and checked for expiry during exchange.
    pub certificate: Option<AgentCertificate>,
    /// Optional domain scope (§4). When set on both peers, the exchange
    /// runs the bidirectional compatibility check at Phase 0 — *before*
    /// any cryptographic or geometric verification.  When `None`, the
    /// agent is treated as domain-unscoped (PoC default, backwards
    /// compatible).
    pub domain_scope: Option<DomainScope>,
    /// Per-domain governance thresholds (§7.3 / §8.2).  Consulted in
    /// validate_request / validate_response: when the peer's primary
    /// domain matches one of these entries, the most-specific policy
    /// overrides the flat `max_drift_accepted` and additionally enforces
    /// min_confidence, min_causal_score, require_chain, and
    /// require_causal_validation.  Empty table = fall back to the flat
    /// per-agent defaults.
    pub governance_table: GovernanceTable,
}

/// Compute the canonical agent ID for a public key.
pub fn compute_agent_id(pk: &VerifyingKey) -> [u8; 32] {
    got_core::sha256(pk.as_bytes())
}

impl TrustRegistry {
    /// Build an empty registry.
    pub fn empty() -> Self {
        Self {
            agents: HashMap::new(),
            max_chain_length: 100,
            max_envelope_age_secs: 300,
            max_attestation_age_secs: 3600,
            ca_public_keys: Vec::new(),
            crls: Vec::new(),
        }
    }

    /// Add an agent to the registry.
    pub fn add_agent(&mut self, entry: AgentEntry) {
        self.agents.insert(entry.agent_id, entry);
    }

    /// Add an agent with certificate validation.
    ///
    /// If the registry has CA keys, the certificate must:
    ///   1. Be present.
    ///   2. Have a valid signature from one of the CA keys.
    ///   3. Have a matching subject key.
    ///   4. Not be revoked.
    ///
    /// If no CA keys are configured, this behaves like `add_agent`.
    pub fn add_agent_verified(&mut self, entry: AgentEntry) -> Result<(), WireError> {
        if !self.ca_public_keys.is_empty() {
            let cert = entry
                .certificate
                .as_ref()
                .ok_or(WireError::CertificateUnknownIssuer)?;

            // Verify certificate signature.
            verify_certificate(cert)?;

            // Check issuer is a known CA.
            if !self.ca_public_keys.contains(&cert.issuer_public_key) {
                return Err(WireError::CertificateUnknownIssuer);
            }

            // Check certificate subject matches the entry public key.
            if cert.subject_public_key != entry.public_key {
                return Err(WireError::CertificateSubjectMismatch);
            }

            // Check revocation.
            let fp = certificate_fingerprint(cert);
            for crl in &self.crls {
                if is_revoked(crl, &fp) {
                    return Err(WireError::CertificateRevoked);
                }
            }
        }
        self.agents.insert(entry.agent_id, entry);
        Ok(())
    }

    /// Validate that an agent's certificate is still valid at the given timestamp.
    ///
    /// Returns `Ok(())` if no certificate is present (PoC mode), or if the
    /// certificate is valid and not revoked.
    pub fn validate_agent_certificate(
        &self,
        agent_id: &[u8; 32],
        now_unix: u64,
    ) -> Result<(), WireError> {
        let entry = match self.lookup(agent_id) {
            Some(e) => e,
            None => {
                let hex: String = agent_id.iter().map(|b| format!("{b:02x}")).collect();
                return Err(WireError::UnknownAgent(hex));
            }
        };

        if let Some(ref cert) = entry.certificate {
            if !is_valid_at(cert, now_unix) {
                return Err(WireError::CertificateExpired {
                    now: now_unix,
                    not_before: cert.not_before,
                    not_after: cert.not_after,
                });
            }
            // Check revocation at validation time.
            let fp = certificate_fingerprint(cert);
            for crl in &self.crls {
                if is_revoked(crl, &fp) {
                    return Err(WireError::CertificateRevoked);
                }
            }
        }

        Ok(())
    }

    /// Load a CRL into the registry.
    ///
    /// Verifies the CRL signature against one of the CA keys, then stores it.
    pub fn load_crl(&mut self, crl: CertificateRevocationList) -> Result<(), WireError> {
        // Verify CRL signature against a known CA.
        let mut verified = false;
        for ca_pk in &self.ca_public_keys {
            if crate::certificate::verify_crl(&crl, ca_pk).is_ok() {
                verified = true;
                break;
            }
        }
        if !verified && !self.ca_public_keys.is_empty() {
            return Err(WireError::CrlSignatureInvalid);
        }
        self.crls.push(crl);
        Ok(())
    }

    /// Apply a key rotation: verify it, update the agent entry, and
    /// retain the old key in `previous_keys` for chain verification.
    pub fn apply_rotation(
        &mut self,
        rotation: &crate::certificate::KeyRotation,
    ) -> Result<(), WireError> {
        crate::certificate::verify_rotation(rotation)?;

        let old_agent_id = compute_agent_id(&rotation.old_public_key);
        let new_agent_id = compute_agent_id(&rotation.new_public_key);

        // Find the existing entry.
        let old_entry = self.agents.remove(&old_agent_id).ok_or_else(|| {
            let hex: String = old_agent_id.iter().map(|b| format!("{b:02x}")).collect();
            WireError::UnknownAgent(hex)
        })?;

        // Build the updated entry from the new certificate.
        let cert = &rotation.new_certificate;
        let new_entry = AgentEntry {
            name: old_entry.name,
            public_key: rotation.new_public_key,
            agent_id: new_agent_id,
            max_drift_accepted: cert.max_drift_accepted,
            roles: cert.roles.clone(),
            expected_model_hash: cert.expected_model_hash,
            certificate: Some(cert.clone()),
            domain_scope: old_entry.domain_scope,
            governance_table: old_entry.governance_table,
        };

        self.agents.insert(new_agent_id, new_entry);
        Ok(())
    }

    /// Lookup an agent by their ID.
    pub fn lookup(&self, agent_id: &[u8; 32]) -> Option<&AgentEntry> {
        self.agents.get(agent_id)
    }

    /// Load a trust registry from a TOML string.
    ///
    /// Expected format:
    /// ```toml
    /// [registry]
    /// max_chain_length = 100
    /// max_envelope_age_secs = 300
    ///
    /// [[agents]]
    /// id = "alice"
    /// public_key = "64 hex chars for 32-byte Ed25519 verifying key"
    /// max_drift_accepted = 0.05
    /// roles = ["producer", "verifier"]
    /// ```
    pub fn from_toml(toml_str: &str) -> Result<Self, WireError> {
        let parsed: TomlFile =
            toml::from_str(toml_str).map_err(|e| WireError::RegistryParse(e.to_string()))?;

        let max_chain_length = parsed
            .registry
            .as_ref()
            .and_then(|r| r.max_chain_length)
            .unwrap_or(100);
        let max_envelope_age_secs = parsed
            .registry
            .as_ref()
            .and_then(|r| r.max_envelope_age_secs)
            .unwrap_or(300);
        let max_attestation_age_secs = parsed
            .registry
            .as_ref()
            .and_then(|r| r.max_attestation_age_secs)
            .unwrap_or(3600);

        let mut registry = Self {
            agents: HashMap::new(),
            max_chain_length,
            max_envelope_age_secs,
            max_attestation_age_secs,
            ca_public_keys: Vec::new(),
            crls: Vec::new(),
        };

        if let Some(agents) = parsed.agents {
            for mut a in agents {
                let pk_bytes = parse_hex_32(&a.public_key)
                    .map_err(|e| WireError::RegistryParse(format!("agent {}: {}", a.id, e)))?;
                let pk = VerifyingKey::from_bytes(&pk_bytes).map_err(|e| {
                    WireError::RegistryParse(format!("agent {}: invalid Ed25519 key: {}", a.id, e))
                })?;
                let agent_id = compute_agent_id(&pk);

                let expected_model_hash = match a.expected_model_hash {
                    Some(ref hex) => Some(parse_hex_32(hex).map_err(|e| {
                        WireError::RegistryParse(format!(
                            "agent {}: expected_model_hash: {}",
                            a.id, e
                        ))
                    })?),
                    None => None,
                };

                let domain_scope = parse_domain_scope(&a)?;

                let mut governance_entries = Vec::new();
                if let Some(rows) = a.governance_thresholds.take() {
                    for row in rows {
                        governance_entries.push(row.into_entry(&a.id)?);
                    }
                }
                let governance_table = GovernanceTable {
                    entries: governance_entries,
                };

                registry.add_agent(AgentEntry {
                    name: a.id,
                    public_key: pk,
                    agent_id,
                    max_drift_accepted: a.max_drift_accepted.unwrap_or(0.05),
                    roles: a.roles.unwrap_or_default(),
                    expected_model_hash,
                    certificate: None,
                    domain_scope,
                    governance_table,
                });
            }
        }

        Ok(registry)
    }

    /// Load from a TOML file on disk **with** integrity verification.
    ///
    /// `expected_sha256` is the SHA-256 digest of the raw file bytes.  The caller
    /// should obtain this out-of-band (e.g. pinned in source, distributed via a
    /// secure channel, or committed into a Merkle tree).  If the file's digest
    /// does not match, an `Err(WireError::RegistryIntegrity { .. })` is returned
    /// *before* any parsing takes place.
    pub fn load(path: &Path, expected_sha256: &[u8; 32]) -> Result<Self, WireError> {
        let raw = std::fs::read(path).map_err(|e| WireError::Io(e.to_string()))?;
        let actual = got_core::sha256(&raw);
        if actual != *expected_sha256 {
            return Err(WireError::RegistryIntegrity {
                expected: hex_encode(expected_sha256),
                actual: hex_encode(&actual),
            });
        }
        // Safe to interpret as UTF-8 — the SHA-256 matched the pinned digest.
        let contents = String::from_utf8(raw)
            .map_err(|e| WireError::RegistryParse(format!("invalid UTF-8: {e}")))?;
        Self::from_toml(&contents)
    }

    /// Load from a TOML file **without** integrity verification.
    ///
    /// # Security
    ///
    /// This bypasses the SHA-256 digest check.  Use [`load`] in production to
    /// protect against filesystem-level tampering of the trust registry.
    pub fn load_unverified(path: &Path) -> Result<Self, WireError> {
        let contents = std::fs::read_to_string(path).map_err(|e| WireError::Io(e.to_string()))?;
        Self::from_toml(&contents)
    }
}

/// Encode 32 bytes as a lowercase hex string.
fn hex_encode(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// TOML structures (serde)
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct TomlFile {
    registry: Option<RegistrySection>,
    agents: Option<Vec<AgentToml>>,
}

#[derive(Debug, serde::Deserialize)]
struct RegistrySection {
    max_chain_length: Option<usize>,
    max_envelope_age_secs: Option<u64>,
    max_attestation_age_secs: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct AgentToml {
    id: String,
    public_key: String,
    max_drift_accepted: Option<f32>,
    roles: Option<Vec<String>>,
    expected_model_hash: Option<String>,
    primary_domain: Option<String>,
    permitted_domains: Option<Vec<PermittedDomainToml>>,
    exclusion_domains: Option<Vec<String>>,
    governance_thresholds: Option<Vec<GovernanceEntryToml>>,
}

#[derive(Debug, serde::Deserialize)]
struct PermittedDomainToml {
    pattern: String,
    mode: InteractionMode,
}

fn parse_domain_scope(a: &AgentToml) -> Result<Option<DomainScope>, WireError> {
    let primary = match a.primary_domain.as_ref() {
        Some(p) => p,
        None => {
            // No primary declared: only valid if no permitted/exclusion lists either.
            if a.permitted_domains.is_some() || a.exclusion_domains.is_some() {
                return Err(WireError::RegistryParse(format!(
                    "agent {}: permitted_domains/exclusion_domains require primary_domain",
                    a.id
                )));
            }
            return Ok(None);
        }
    };
    let primary = Domain::parse(primary)
        .map_err(|e| WireError::RegistryParse(format!("agent {}: primary_domain: {e}", a.id)))?;

    let mut permitted = Vec::new();
    if let Some(list) = &a.permitted_domains {
        for entry in list {
            let pattern = DomainPattern::parse(&entry.pattern).map_err(|e| {
                WireError::RegistryParse(format!("agent {}: permitted_domains: {e}", a.id))
            })?;
            permitted.push(PermittedDomain {
                pattern,
                mode: entry.mode,
            });
        }
    }

    let mut exclusions = Vec::new();
    if let Some(list) = &a.exclusion_domains {
        for s in list {
            let pat = DomainPattern::parse(s).map_err(|e| {
                WireError::RegistryParse(format!("agent {}: exclusion_domains: {e}", a.id))
            })?;
            exclusions.push(pat);
        }
    }

    Ok(Some(DomainScope {
        primary,
        permitted,
        exclusions,
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a 64-character hex string into 32 bytes.
fn parse_hex_32(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", hex.len()));
    }
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("non-hex characters in key".to_string());
    }
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn test_key_alice() -> SigningKey {
        SigningKey::from_bytes(&[0xAA; 32])
    }

    fn test_key_bob() -> SigningKey {
        SigningKey::from_bytes(&[0xBB; 32])
    }

    fn pk_hex(key: &SigningKey) -> String {
        key.verifying_key()
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    #[test]
    fn agent_id_matches_sha256_of_pk() {
        let key = test_key_alice();
        let pk = key.verifying_key();
        let id = compute_agent_id(&pk);
        let expected = got_core::sha256(pk.as_bytes());
        assert_eq!(id, expected);
    }

    #[test]
    fn registry_from_toml_roundtrip() {
        let alice = test_key_alice();
        let bob = test_key_bob();

        let toml_str = format!(
            r#"
[registry]
max_chain_length = 50
max_envelope_age_secs = 600

[[agents]]
id = "alice"
public_key = "{}"
max_drift_accepted = 0.05
roles = ["producer", "verifier"]

[[agents]]
id = "bob"
public_key = "{}"
max_drift_accepted = 0.10
roles = ["verifier"]
"#,
            pk_hex(&alice),
            pk_hex(&bob),
        );

        let registry = TrustRegistry::from_toml(&toml_str).unwrap();
        assert_eq!(registry.max_chain_length, 50);
        assert_eq!(registry.max_envelope_age_secs, 600);
        assert_eq!(registry.agents.len(), 2);

        let alice_id = compute_agent_id(&alice.verifying_key());
        let alice_entry = registry.lookup(&alice_id).unwrap();
        assert_eq!(alice_entry.name, "alice");
        assert!((alice_entry.max_drift_accepted - 0.05).abs() < 1e-6);
        assert_eq!(alice_entry.roles, vec!["producer", "verifier"]);

        let bob_id = compute_agent_id(&bob.verifying_key());
        let bob_entry = registry.lookup(&bob_id).unwrap();
        assert_eq!(bob_entry.name, "bob");
        assert!((bob_entry.max_drift_accepted - 0.10).abs() < 1e-6);
    }

    #[test]
    fn registry_unknown_agent_returns_none() {
        let registry = TrustRegistry::empty();
        assert!(registry.lookup(&[0xFF; 32]).is_none());
    }

    #[test]
    fn registry_defaults() {
        let toml_str = "";
        let registry = TrustRegistry::from_toml(toml_str).unwrap();
        assert_eq!(registry.max_chain_length, 100);
        assert_eq!(registry.max_envelope_age_secs, 300);
        assert!(registry.agents.is_empty());
    }

    #[test]
    fn registry_toml_parses_domain_scope() {
        let alice = test_key_alice();
        let toml_str = format!(
            r#"
[[agents]]
id = "alice"
public_key = "{}"
primary_domain = "agriculture.crop-management"
exclusion_domains = ["transport.*"]
permitted_domains = [
    {{ pattern = "agriculture.*", mode = "cooperative" }},
    {{ pattern = "meteorology.*", mode = "advisory" }},
]
"#,
            pk_hex(&alice),
        );

        let reg = TrustRegistry::from_toml(&toml_str).unwrap();
        let id = compute_agent_id(&alice.verifying_key());
        let entry = reg.lookup(&id).unwrap();
        let scope = entry.domain_scope.as_ref().expect("domain scope present");
        assert_eq!(scope.primary.as_str(), "agriculture.crop-management");
        assert_eq!(scope.permitted.len(), 2);
        assert_eq!(scope.exclusions.len(), 1);
    }

    #[test]
    fn registry_toml_rejects_permitted_without_primary() {
        let alice = test_key_alice();
        let toml_str = format!(
            r#"
[[agents]]
id = "alice"
public_key = "{}"
permitted_domains = [{{ pattern = "agriculture.*", mode = "cooperative" }}]
"#,
            pk_hex(&alice),
        );
        let err = TrustRegistry::from_toml(&toml_str).unwrap_err();
        assert!(matches!(err, WireError::RegistryParse(_)));
    }

    #[test]
    fn registry_bad_hex_rejected() {
        let toml_str = r#"
[[agents]]
id = "bad"
public_key = "ZZZZ"
"#;
        let err = TrustRegistry::from_toml(toml_str).unwrap_err();
        assert!(matches!(err, WireError::RegistryParse(_)));
    }

    #[test]
    fn registry_add_agent_programmatic() {
        let key = test_key_alice();
        let pk = key.verifying_key();
        let id = compute_agent_id(&pk);

        let mut registry = TrustRegistry::empty();
        registry.add_agent(AgentEntry {
            name: "alice".to_string(),
            public_key: pk,
            agent_id: id,
            max_drift_accepted: 0.05,
            roles: vec!["producer".to_string()],
            expected_model_hash: None,
            certificate: None,
            domain_scope: None,
            governance_table: GovernanceTable::default(),
        });

        assert!(registry.lookup(&id).is_some());
        assert_eq!(registry.lookup(&id).unwrap().name, "alice");
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issue 23)
    // -----------------------------------------------------------------------

    /// Issue #23 (S-2): TrustRegistry loaded from TOML validates
    /// public keys on parse — a tampered key is rejected.
    #[test]
    fn sec_registry_load_rejects_tampered_toml() {
        let alice = test_key_alice();

        let good_toml = format!(
            r#"
[registry]
max_chain_length = 50

[[agents]]
id = "alice"
public_key = "{}"
"#,
            pk_hex(&alice),
        );

        // Registry loads fine with correct TOML.
        let reg = TrustRegistry::from_toml(&good_toml).unwrap();
        assert!(!reg.agents.is_empty());

        // Tamper: flip a byte in the public key to make it an invalid curve point.
        let pk_str = pk_hex(&alice);
        // Replace the first two hex chars of the key with a value that
        // (with overwhelming probability) yields an invalid Ed25519 point.
        let tampered_pk = format!("00{}", &pk_str[2..]);
        let tampered = good_toml.replace(&pk_str, &tampered_pk);
        let result = TrustRegistry::from_toml(&tampered);
        assert!(
            result.is_err(),
            "tampered registry TOML must be rejected (invalid Ed25519 key)"
        );
    }

    /// Issue #23 (S-2): `load()` verifies SHA-256 digest of the raw file.
    /// A valid-but-tampered registry (attacker substituted their own key)
    /// must be rejected if the digest doesn't match.
    #[test]
    fn sec_registry_load_rejects_integrity_mismatch() {
        use std::io::Write;
        let alice = test_key_alice();

        let good_toml = format!(
            r#"
[registry]
max_chain_length = 50

[[agents]]
id = "alice"
public_key = "{}"
"#,
            pk_hex(&alice),
        );

        // Write the good TOML to a temp file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.toml");
        std::fs::write(&path, &good_toml).unwrap();

        // Compute the correct digest.
        let correct_digest = got_core::sha256(good_toml.as_bytes());

        // load() with correct digest succeeds.
        let reg = TrustRegistry::load(&path, &correct_digest).unwrap();
        assert!(!reg.agents.is_empty());

        // Now tamper: attacker writes a valid TOML with their own key.
        let attacker = SigningKey::from_bytes(&[0xEE; 32]);
        let tampered_toml = format!(
            r#"
[registry]
max_chain_length = 50

[[agents]]
id = "attacker"
public_key = "{}"
"#,
            pk_hex(&attacker),
        );
        std::fs::write(&path, &tampered_toml).unwrap();

        // load() with the ORIGINAL digest must fail — the file was tampered.
        let err = TrustRegistry::load(&path, &correct_digest).unwrap_err();
        assert!(
            matches!(err, WireError::RegistryIntegrity { .. }),
            "tampered file must fail integrity check, got: {err:?}"
        );

        // load_unverified() still works (no integrity check).
        let reg2 = TrustRegistry::load_unverified(&path).unwrap();
        assert_eq!(reg2.agents.values().next().unwrap().name, "attacker");
    }
}
