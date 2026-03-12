// ---------------------------------------------------------------------------
// AttestationStore trait & supporting types
// ---------------------------------------------------------------------------

use ed25519_dalek::VerifyingKey;
use got_core::GeometricAttestation;
use thiserror::Error;

use crate::audit::AuditReport;

/// Content-addressed store ID: SHA-256 of the attestation's canonical serialisation.
pub type StoreId = [u8; 32];

/// Errors arising from store operations.
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("attestation signature invalid")]
    InvalidSignature,

    #[error("parent attestation {0} not found in store")]
    OrphanedAttestation(String),

    #[error("verifying key required for signature check")]
    MissingVerifyingKey,

    #[error("serialisation error: {0}")]
    Serialisation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("store error: {0}")]
    Internal(String),
}

/// Conjunctive query filter: all specified fields must match.
#[derive(Debug, Clone, Default)]
pub struct StoreFilter {
    pub model_id: Option<String>,
    pub signer: Option<[u8; 32]>,
    pub after: Option<u64>,
    pub before: Option<u64>,
    pub schema_version: Option<u16>,
    pub causal_flag: Option<bool>,
}

impl StoreFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    pub fn signer(mut self, key: &VerifyingKey) -> Self {
        self.signer = Some(got_core::sha256(key.as_bytes()));
        self
    }

    pub fn after(mut self, ts: u64) -> Self {
        self.after = Some(ts);
        self
    }

    pub fn before(mut self, ts: u64) -> Self {
        self.before = Some(ts);
        self
    }

    pub fn schema_version(mut self, v: u16) -> Self {
        self.schema_version = Some(v);
        self
    }

    pub fn causal_flag(mut self, flag: bool) -> Self {
        self.causal_flag = Some(flag);
        self
    }

    /// Returns true if the attestation matches all specified filter fields.
    pub fn matches(&self, a: &GeometricAttestation, signer_hash: &[u8; 32]) -> bool {
        if let Some(ref mid) = self.model_id {
            if a.model_id != *mid {
                return false;
            }
        }
        if let Some(ref s) = self.signer {
            if signer_hash != s {
                return false;
            }
        }
        if let Some(after) = self.after {
            if a.timestamp < after {
                return false;
            }
        }
        if let Some(before) = self.before {
            if a.timestamp > before {
                return false;
            }
        }
        if let Some(sv) = self.schema_version {
            if a.schema_version != sv {
                return false;
            }
        }
        if let Some(cf) = self.causal_flag {
            if a.causal_flag != Some(cf) {
                return false;
            }
        }
        true
    }
}

/// Compute the content-addressed store ID for an attestation.
///
/// Delegates to `got_attest::attestation_hash` for a single canonical
/// implementation of "SHA-256 of serialise_for_signing()".
pub fn attestation_store_id(a: &GeometricAttestation) -> Result<StoreId, StoreError> {
    got_attest::attestation_hash(a).map_err(|e| StoreError::Serialisation(e.to_string()))
}

/// Abstract attestation store.
///
/// Implementations must enforce:
/// - Append-only: no mutation or deletion.
/// - Signature verification on insert.
/// - Chain validation (parent_hash must exist unless None).
/// - Idempotent duplicate inserts (same content hash).
pub trait AttestationStore {
    /// Insert an attestation. The verifying key is needed to check the
    /// signature. Returns the content-addressed StoreId.
    ///
    /// If the attestation's `parent_attestation_hash` is `Some(h)`,
    /// `h` must already be present in the store (or the insert fails
    /// with `OrphanedAttestation`).
    ///
    /// Duplicate inserts (same content hash) are idempotent — the
    /// existing StoreId is returned.
    fn append(
        &mut self,
        attestation: &GeometricAttestation,
        verifying_key: &VerifyingKey,
    ) -> Result<StoreId, StoreError>;

    /// Retrieve an attestation by content hash.
    fn get(&self, id: &StoreId) -> Option<&GeometricAttestation>;

    /// Retrieve the full attestation chain for a model (ordered by timestamp).
    fn chain(&self, model_id: &str) -> Vec<&GeometricAttestation>;

    /// Query attestations matching all specified filter fields.
    fn query(&self, filter: &StoreFilter) -> Vec<&GeometricAttestation>;

    /// Produce an audit report for a model.
    fn audit(&self, model_id: &str) -> AuditReport;

    /// Total number of attestations in the store.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
