// ---------------------------------------------------------------------------
// Federated registry composer + signed vouching — Protocol §14.5.
//
// This module provides two layers that compose together:
//
//   1. Composition (`FederatedRegistry`): an ordered list of named
//      `TrustRegistry` instances with explicit priority, multi-
//      registry lookup, policy conflict reporting, and a `resolve()`
//      method that merges into a single flat `TrustRegistry` the rest
//      of the exchange pipeline consumes unchanged.  No protocol
//      changes — `validate_request` etc. operate on the resolved
//      registry exactly as on a stand-alone one.
//
//   2. Cryptographic vouching (`FederationVoucher`): a signed
//      attestation in which the operator of one member registry
//      vouches for the operator (and the on-disk digest) of another.
//      `FederatedRegistry::verify_vouchers` walks the federation and
//      reports any non-lead member that is not vouched for by at
//      least one higher-priority member.  The lead registry
//      (priority 0) is the root of trust; everything below it must
//      be reachable through a chain of vouchers from the lead.
//
// Conflict semantics: because `agent_id = SHA-256(public_key)`, two
// registries with the same `agent_id` are by definition talking about
// the same key.  We never have to arbitrate cryptographic identity —
// only the *policy* attached to that identity (drift bounds, scope,
// governance thresholds, expected_model_hash, certificate).  Conflicts
// of policy are reported as `FederationWarning`s; the highest-
// priority registry wins on resolve.  Lower priority is "better" —
// priority `0` beats priority `1`, matching how Linux process nice
// values work.
//
// What this module still does NOT cover:
//   - Async sync between member registries (the §14.5 paper design's
//     "live federation" — pull/push refresh, eventual consistency).
//   - Revocation propagation across the federation (CRLs are unioned
//     statically at resolve time, but a fresh CRL on registry A does
//     not propagate to a verifier holding registry B until both are
//     reloaded).
//   - Arbitration policy ("when EU and US disagree, who wins") beyond
//     priority ordering — that is institutional, not code.
// ---------------------------------------------------------------------------

use std::collections::{BTreeMap, HashSet};

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};

use crate::registry::{compute_agent_id, AgentEntry, TrustRegistry};
use crate::WireError;

/// Default maximum depth of a multi-hop voucher chain.  Each fixed-
/// point iteration of `verify_vouchers` adds at most one hop, so a
/// chain of length N (lead → A1 → A2 → … → AN) requires N iterations
/// to fully verify.  10 covers any realistic federation topology and
/// caps worst-case work at O(depth × members² × vouchers_per_member),
/// which is trivial for sub-100-member federations.
pub const DEFAULT_MAX_VOUCHER_CHAIN_DEPTH: usize = 10;

/// One member registry of a federation.
///
/// `priority` is the resolution rank: lower numbers win on conflict.
/// `name` is a human label that appears in `FederationWarning`s and
/// in the merged registry's audit trail.
///
/// `digest` is the SHA-256 of the registry file's bytes as loaded —
/// the same value the operator passes to `TrustRegistry::load`.  It
/// is used to match against vouchers (a voucher binds to a specific
/// digest, so silently swapping the file would invalidate the chain).
/// `None` means "no integrity pin" and the member cannot participate
/// in voucher verification — only useful for unverified-load PoC
/// federations.
///
/// `operator_key` is the verifying key of the operator that publishes
/// this registry (the EU healthcare authority's signing identity, for
/// example).  It is distinct from any agent key inside the registry —
/// the operator signs vouchers, agents sign attestations, and the two
/// roles must not overlap.  `None` means the member's operator does
/// not issue or accept vouchers.
///
/// `vouchers` is the set of cross-registry vouchers received for this
/// member from other federation operators.  An empty list is fine for
/// the lead (priority 0) member; non-lead members need at least one
/// valid voucher from a higher-priority member to be considered
/// "vouched" by `verify_vouchers`.
#[derive(Debug, Clone)]
pub struct NamedRegistry {
    pub name: String,
    pub priority: u32,
    pub registry: TrustRegistry,
    pub digest: Option<[u8; 32]>,
    pub operator_key: Option<VerifyingKey>,
    pub vouchers: Vec<FederationVoucher>,
}

impl NamedRegistry {
    /// Convenience constructor for the common case: a member with
    /// no integrity pin, no operator key, and no vouchers (matches
    /// the pre-vouching API).
    pub fn unverified(name: impl Into<String>, priority: u32, registry: TrustRegistry) -> Self {
        Self {
            name: name.into(),
            priority,
            registry,
            digest: None,
            operator_key: None,
            vouchers: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// FederationVoucher — signed cross-registry vouching.
// ---------------------------------------------------------------------------

/// Current voucher format version.  Bump when the canonical layout
/// changes in a way that breaks signature compatibility.
pub const VOUCHER_VERSION: u16 = 1;

/// Maximum length (in bytes) of the human-readable string fields
/// inside a voucher.  Bounds the canonical size and prevents pathological
/// allocations during deserialisation.
pub const VOUCHER_MAX_STRING_LEN: usize = 256;

/// A signed cross-registry voucher.
///
/// Issued by the operator of one federation member to vouch for the
/// existence and integrity of another member.  The signature binds:
///
///   - the issuing operator's identity (`issuer_id` = SHA-256 of the
///     issuer's verifying key);
///   - the subject registry's *file digest* (the same SHA-256 the
///     operator would pass to `TrustRegistry::load` for integrity
///     pinning);
///   - the subject's human-readable name;
///   - the voucher's creation timestamp;
///   - an expiry deadline (`not_after`, or `0` for no expiry);
///   - an optional rationale string for governance audit.
///
/// A holder of the voucher and the issuer's verifying key can verify
/// the signature, check expiry, and confirm that the subject registry
/// file they have on disk matches the digest the issuer signed over.
/// `FederatedRegistry::verify_vouchers` automates this walk for an
/// entire federation.
///
/// Vouchers are *one-hop*: the verifier checks that each non-lead
/// member is signed by some higher-priority member.  Multi-hop chains
/// (A vouches for B vouches for C) are not aggregated automatically;
/// each member must carry a voucher from a higher-priority neighbour.
#[derive(Debug, Clone)]
pub struct FederationVoucher {
    pub voucher_version: u16,
    pub issuer_id: [u8; 32],
    pub subject_digest: [u8; 32],
    pub subject_name: String,
    pub timestamp: u64,
    pub not_after: u64,
    pub rationale: String,
    pub signature: [u8; 64],
}

impl FederationVoucher {
    /// Build the canonical signable byte sequence.
    ///
    /// Layout:
    ///   u16 LE voucher_version
    ///   [u8; 32] issuer_id
    ///   [u8; 32] subject_digest
    ///   u32 LE subject_name length
    ///   subject_name UTF-8 bytes
    ///   u64 LE timestamp
    ///   u64 LE not_after
    ///   u32 LE rationale length
    ///   rationale UTF-8 bytes
    ///
    /// `signature` is excluded.
    pub fn signable_bytes(&self) -> Result<Vec<u8>, WireError> {
        if self.subject_name.len() > VOUCHER_MAX_STRING_LEN {
            return Err(WireError::Protocol(format!(
                "voucher subject_name too large: {} bytes",
                self.subject_name.len()
            )));
        }
        if self.rationale.len() > VOUCHER_MAX_STRING_LEN {
            return Err(WireError::Protocol(format!(
                "voucher rationale too large: {} bytes",
                self.rationale.len()
            )));
        }
        let mut buf = Vec::with_capacity(96 + self.subject_name.len() + self.rationale.len());
        buf.extend_from_slice(&self.voucher_version.to_le_bytes());
        buf.extend_from_slice(&self.issuer_id);
        buf.extend_from_slice(&self.subject_digest);
        buf.extend_from_slice(&(self.subject_name.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.subject_name.as_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.not_after.to_le_bytes());
        buf.extend_from_slice(&(self.rationale.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.rationale.as_bytes());
        Ok(buf)
    }

    /// Create and sign a fresh voucher.
    ///
    /// `issuer_id` is the SHA-256 of `signing_key.verifying_key()`.
    /// The constructor does not compute it for you so the caller can
    /// reuse a pre-computed agent_id from a registry lookup.
    pub fn create(
        issuer_id: [u8; 32],
        subject_digest: [u8; 32],
        subject_name: &str,
        timestamp: u64,
        not_after: u64,
        rationale: &str,
        signing_key: &SigningKey,
    ) -> Result<Self, WireError> {
        let mut v = Self {
            voucher_version: VOUCHER_VERSION,
            issuer_id,
            subject_digest,
            subject_name: subject_name.to_string(),
            timestamp,
            not_after,
            rationale: rationale.to_string(),
            signature: [0u8; 64],
        };
        let payload = v.signable_bytes()?;
        v.signature = signing_key.sign(&payload).to_bytes();
        Ok(v)
    }

    /// Verify the voucher signature against `issuer_vk` and check
    /// that the issuer key actually corresponds to `issuer_id`.
    ///
    /// Returns `Ok(())` only if the signature verifies *and* the
    /// SHA-256 of `issuer_vk` matches `issuer_id` — i.e. the voucher
    /// names the same key that signed it.  Mismatches indicate either
    /// a wrong-key verification attempt or a tampered `issuer_id`.
    pub fn verify(&self, issuer_vk: &VerifyingKey) -> Result<(), WireError> {
        if self.voucher_version != VOUCHER_VERSION {
            return Err(WireError::Protocol(format!(
                "unsupported voucher version {}",
                self.voucher_version
            )));
        }
        let derived_id = compute_agent_id(issuer_vk);
        if derived_id != self.issuer_id {
            return Err(WireError::Protocol(
                "voucher issuer_id does not match the verifying key".into(),
            ));
        }
        let payload = self.signable_bytes()?;
        let sig = ed25519_dalek::Signature::from_bytes(&self.signature);
        issuer_vk
            .verify(&payload, &sig)
            .map_err(|_| WireError::Protocol("voucher signature invalid".into()))
    }

    /// True if the voucher is currently valid at `now_unix`.  A
    /// `not_after` of `0` is treated as "no expiry".  Future-dated
    /// `timestamp`s (issuer clock ahead of verifier) are accepted by
    /// up to 300 seconds of skew, mirroring `assemble_and_sign`.
    pub fn is_valid_at(&self, now_unix: u64) -> bool {
        const MAX_FUTURE_SKEW: u64 = 300;
        if self.timestamp > now_unix + MAX_FUTURE_SKEW {
            return false;
        }
        if self.not_after == 0 {
            return true;
        }
        now_unix <= self.not_after
    }
}

// ---------------------------------------------------------------------------
// OperatorKeyRotation — long-lived operator identity across key rotations.
// ---------------------------------------------------------------------------

/// Current rotation format version.  Bump if the canonical layout
/// changes in a way that breaks signature compatibility.
pub const OPERATOR_ROTATION_VERSION: u16 = 1;

/// Cross-signed record establishing that two operator keys belong to
/// the same long-lived identity.
///
/// `old_public_key` signs the canonical bytes (proving the rotation
/// was authorised by the holder of the old key) AND `new_public_key`
/// signs the same canonical bytes (proving the new key exists and is
/// controllable).  Both signatures must verify for the rotation to
/// be accepted by `verify()`.
///
/// `timestamp` records when the rotation took effect.  After this
/// instant, the holder of the old key SHOULD NOT issue any new
/// vouchers — the federation enforces this by accepting vouchers
/// signed with the old key only if their `timestamp` is strictly
/// less than the rotation's `timestamp`.  Existing vouchers issued
/// before the rotation remain valid until their own `not_after`
/// expiry.
///
/// Multi-step rotations form a chain (A → B → C → …).  The
/// federation walks the chain greedily — each step's old key must
/// match the previous step's new key.  Cycles are rejected.
#[derive(Debug, Clone)]
pub struct OperatorKeyRotation {
    pub rotation_version: u16,
    pub old_public_key: VerifyingKey,
    pub new_public_key: VerifyingKey,
    pub timestamp: u64,
    pub old_key_signature: [u8; 64],
    pub new_key_signature: [u8; 64],
}

impl OperatorKeyRotation {
    /// Build the canonical signable bytes shared by both signatures.
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2 + 32 + 32 + 8);
        buf.extend_from_slice(&self.rotation_version.to_le_bytes());
        buf.extend_from_slice(self.old_public_key.as_bytes());
        buf.extend_from_slice(self.new_public_key.as_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf
    }

    /// Create and double-sign an operator key rotation.
    pub fn create(
        old_signing_key: &SigningKey,
        new_signing_key: &SigningKey,
        timestamp: u64,
    ) -> Self {
        let mut r = Self {
            rotation_version: OPERATOR_ROTATION_VERSION,
            old_public_key: old_signing_key.verifying_key(),
            new_public_key: new_signing_key.verifying_key(),
            timestamp,
            old_key_signature: [0u8; 64],
            new_key_signature: [0u8; 64],
        };
        let payload = r.signable_bytes();
        r.old_key_signature = old_signing_key.sign(&payload).to_bytes();
        r.new_key_signature = new_signing_key.sign(&payload).to_bytes();
        r
    }

    /// Verify both cross-signatures.
    pub fn verify(&self) -> Result<(), WireError> {
        if self.rotation_version != OPERATOR_ROTATION_VERSION {
            return Err(WireError::Protocol(format!(
                "unsupported operator rotation version {}",
                self.rotation_version
            )));
        }
        let payload = self.signable_bytes();
        let old_sig = ed25519_dalek::Signature::from_bytes(&self.old_key_signature);
        self.old_public_key
            .verify(&payload, &old_sig)
            .map_err(|_| WireError::Protocol("operator rotation: old-key signature invalid".into()))?;
        let new_sig = ed25519_dalek::Signature::from_bytes(&self.new_key_signature);
        self.new_public_key
            .verify(&payload, &new_sig)
            .map_err(|_| WireError::Protocol("operator rotation: new-key signature invalid".into()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FederationSyncSource — pluggable transport for refreshing member registries.
// ---------------------------------------------------------------------------

/// A registry file fetched from a `FederationSyncSource`.
///
/// Carries the raw bytes plus a SHA-256 digest the caller can pass to
/// `TrustRegistry::load(path, &digest)` for integrity-pinned loading.
/// `fetched_at` is a unix timestamp for staleness tracking.
#[derive(Debug, Clone)]
pub struct SyncedRegistry {
    pub digest: [u8; 32],
    pub bytes: Vec<u8>,
    pub fetched_at: u64,
}

impl SyncedRegistry {
    /// Construct a `SyncedRegistry` from raw bytes, computing the
    /// digest with `got_core::sha256`.
    pub fn from_bytes(bytes: Vec<u8>, fetched_at: u64) -> Self {
        let digest = got_core::sha256(&bytes);
        Self {
            digest,
            bytes,
            fetched_at,
        }
    }
}

/// Pluggable transport for refreshing a federation member's registry
/// file from its source of truth.
///
/// Implementations are responsible for the *what* (fetching bytes
/// from a file, an HTTP endpoint, an object store, a git repo) and
/// the *when-to-skip* (returning `Ok(None)` if nothing has changed
/// since `since`).  The polling cadence and retry logic live in the
/// caller — typically `FederationSyncManager` in `got-net`.
///
/// The trait is sync.  Implementations that need async I/O should
/// either run their fetch on a blocking thread (the manager wraps
/// every poll in `tokio::task::spawn_blocking` for exactly this
/// reason) or expose an async version higher up in the stack.
pub trait FederationSyncSource: Send + Sync + std::fmt::Debug {
    /// Fetch the latest version of the registry file.
    ///
    /// `since` is the digest the caller already holds (e.g. from a
    /// previous successful fetch).  Implementations may return
    /// `Ok(None)` if they can cheaply confirm that nothing has
    /// changed (e.g. via HTTP `If-None-Match`/etag, file `mtime`,
    /// git rev compare).  Returning `Ok(Some(_))` always indicates
    /// fresh content; the manager treats it as a successful refresh
    /// and resets the staleness counter.
    fn fetch(&self, since: Option<[u8; 32]>) -> Result<Option<SyncedRegistry>, WireError>;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;
}

/// In-memory `FederationSyncSource` that always returns the same
/// fixed bytes.  Useful for tests and for "I want the federation API
/// but not the live sync semantics" cases.
#[derive(Debug, Clone)]
pub struct StaticSyncSource {
    name: String,
    snapshot: SyncedRegistry,
}

impl StaticSyncSource {
    pub fn new(name: impl Into<String>, bytes: Vec<u8>, fetched_at: u64) -> Self {
        Self {
            name: name.into(),
            snapshot: SyncedRegistry::from_bytes(bytes, fetched_at),
        }
    }

    pub fn snapshot(&self) -> &SyncedRegistry {
        &self.snapshot
    }
}

impl FederationSyncSource for StaticSyncSource {
    fn fetch(&self, since: Option<[u8; 32]>) -> Result<Option<SyncedRegistry>, WireError> {
        if since == Some(self.snapshot.digest) {
            return Ok(None);
        }
        Ok(Some(self.snapshot.clone()))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// `FederationSyncSource` backed by a local file.
///
/// On each `fetch`, reads the file's bytes and computes the SHA-256.
/// If the digest matches `since`, returns `Ok(None)` to signal "no
/// change".  Otherwise returns the new content.
///
/// `mtime` is **not** consulted: a file edited within the same second
/// as the previous fetch would otherwise be missed.  Reading the
/// bytes is cheap on local disk and the digest comparison is the
/// authoritative change signal.
#[derive(Debug, Clone)]
pub struct FileSyncSource {
    name: String,
    path: std::path::PathBuf,
}

impl FileSyncSource {
    pub fn new(name: impl Into<String>, path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
        }
    }
}

impl FederationSyncSource for FileSyncSource {
    fn fetch(&self, since: Option<[u8; 32]>) -> Result<Option<SyncedRegistry>, WireError> {
        let bytes = std::fs::read(&self.path).map_err(|e| WireError::Io(e.to_string()))?;
        let digest = got_core::sha256(&bytes);
        if since == Some(digest) {
            return Ok(None);
        }
        let fetched_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(Some(SyncedRegistry {
            digest,
            bytes,
            fetched_at,
        }))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// FederationRevocationList — signed list of revoked voucher fingerprints.
// ---------------------------------------------------------------------------

/// Current FRL format version.  Bump if the canonical layout changes
/// in a way that breaks signature compatibility.
pub const FRL_VERSION: u16 = 1;

/// Canonical fingerprint of a `FederationVoucher` — SHA-256 of the
/// voucher's `signable_bytes` plus its 64-byte signature.  Two
/// vouchers with bit-identical contents have the same fingerprint;
/// any modification (including a re-sign with a different timestamp)
/// produces a different fingerprint.
pub fn voucher_fingerprint(voucher: &FederationVoucher) -> Result<[u8; 32], WireError> {
    let mut bytes = voucher.signable_bytes()?;
    bytes.extend_from_slice(&voucher.signature);
    Ok(got_core::sha256(&bytes))
}

/// One revoked voucher in a federation revocation list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevokedVoucher {
    pub voucher_fingerprint: [u8; 32],
    pub revocation_time: u64,
    pub reason: String,
}

/// A signed federation revocation list.
///
/// Issued by a federation operator (using the same Ed25519 key that
/// signs vouchers) and consulted by `FederatedRegistry::verify_vouchers`
/// to reject vouchers whose fingerprint appears in any FRL signed by
/// an in-chain operator.
///
/// Revocation semantics:
///   - The fingerprint binds to the *exact* voucher bytes, so a
///     re-issued voucher with the same content but a different
///     timestamp is a *different* voucher with a different
///     fingerprint and is not affected by the revocation.
///   - The FRL is signed by an operator key.  Only FRLs signed by an
///     operator that is in the verified chain (current `operator_key`
///     of some federation member) are honoured — an outsider's FRL
///     cannot revoke vouchers in your federation.
///   - Multi-FRL: a federation can hold many FRLs (one per operator).
///     A voucher fingerprint appearing in *any* honoured FRL is
///     revoked.
#[derive(Debug, Clone)]
pub struct FederationRevocationList {
    pub frl_version: u16,
    pub issuer_id: [u8; 32],
    pub timestamp: u64,
    pub entries: Vec<RevokedVoucher>,
    pub signature: [u8; 64],
}

/// Maximum length (in bytes) of a `RevokedVoucher::reason` string.
pub const FRL_MAX_REASON_LEN: usize = 256;

impl FederationRevocationList {
    /// Build the canonical signable byte sequence.
    ///
    /// Layout:
    ///   u16 LE frl_version
    ///   [u8; 32] issuer_id
    ///   u64 LE timestamp
    ///   u32 LE entry_count
    ///   for each entry:
    ///     [u8; 32] voucher_fingerprint
    ///     u64 LE revocation_time
    ///     u32 LE reason_length
    ///     reason UTF-8 bytes
    ///
    /// `signature` is excluded.
    pub fn signable_bytes(&self) -> Result<Vec<u8>, WireError> {
        let mut buf = Vec::with_capacity(48 + self.entries.len() * 80);
        buf.extend_from_slice(&self.frl_version.to_le_bytes());
        buf.extend_from_slice(&self.issuer_id);
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            if e.reason.len() > FRL_MAX_REASON_LEN {
                return Err(WireError::Protocol(format!(
                    "FRL reason too large: {} bytes",
                    e.reason.len()
                )));
            }
            buf.extend_from_slice(&e.voucher_fingerprint);
            buf.extend_from_slice(&e.revocation_time.to_le_bytes());
            buf.extend_from_slice(&(e.reason.len() as u32).to_le_bytes());
            buf.extend_from_slice(e.reason.as_bytes());
        }
        Ok(buf)
    }

    /// Sign a fresh FRL.  `issuer_id` is the SHA-256 of the issuer's
    /// verifying key — same convention as `FederationVoucher::create`.
    pub fn create(
        issuer_id: [u8; 32],
        timestamp: u64,
        entries: Vec<RevokedVoucher>,
        signing_key: &SigningKey,
    ) -> Result<Self, WireError> {
        let mut frl = Self {
            frl_version: FRL_VERSION,
            issuer_id,
            timestamp,
            entries,
            signature: [0u8; 64],
        };
        let payload = frl.signable_bytes()?;
        frl.signature = signing_key.sign(&payload).to_bytes();
        Ok(frl)
    }

    /// Verify the FRL signature against `issuer_vk` and check that
    /// the issuer key actually corresponds to `issuer_id`.
    pub fn verify(&self, issuer_vk: &VerifyingKey) -> Result<(), WireError> {
        if self.frl_version != FRL_VERSION {
            return Err(WireError::Protocol(format!(
                "unsupported FRL version {}",
                self.frl_version
            )));
        }
        let derived_id = compute_agent_id(issuer_vk);
        if derived_id != self.issuer_id {
            return Err(WireError::Protocol(
                "FRL issuer_id does not match the verifying key".into(),
            ));
        }
        let payload = self.signable_bytes()?;
        let sig = ed25519_dalek::Signature::from_bytes(&self.signature);
        issuer_vk
            .verify(&payload, &sig)
            .map_err(|_| WireError::Protocol("FRL signature invalid".into()))
    }

    /// True if `fingerprint` appears in this FRL's entries.
    pub fn contains(&self, fingerprint: &[u8; 32]) -> bool {
        self.entries
            .iter()
            .any(|e| &e.voucher_fingerprint == fingerprint)
    }
}

// ---------------------------------------------------------------------------
// VoucherWarning — non-fatal divergences from verify_vouchers.
// ---------------------------------------------------------------------------

/// A non-fatal divergence found by `FederatedRegistry::verify_vouchers`.
#[derive(Debug, Clone, PartialEq)]
pub enum VoucherWarning {
    /// A non-lead member has no valid voucher from any higher-priority
    /// member.  `expected_issuer_keys` lists the member names whose
    /// operators *could* have signed a voucher.
    Missing {
        member: String,
        expected_issuer_names: Vec<String>,
    },
    /// A voucher was found but its expiry has passed.
    Expired {
        member: String,
        issuer_name: String,
        not_after: u64,
    },
    /// A voucher was found but the signature did not verify against
    /// the named issuer's public key.
    SignatureInvalid {
        member: String,
        issuer_name: String,
    },
    /// A voucher was found but the digest it signed over does not
    /// match the on-disk digest of the member registry.  Either the
    /// member file was tampered with or the voucher is stale.
    DigestMismatch {
        member: String,
        issuer_name: String,
    },
    /// A voucher was found but its `issuer_id` does not match any
    /// higher-priority member's operator key.
    UnknownIssuer { member: String, issuer_hex: String },
    /// A non-lead member has no `digest` set, so vouchers cannot bind
    /// to it.  Set `NamedRegistry::digest` from the same SHA-256 you
    /// pass to `TrustRegistry::load`.
    NoDigestPin { member: String },
    /// A voucher was signed by a key that has been rotated out, and
    /// the voucher's `timestamp` is at or after the rotation's
    /// timestamp.  Vouchers issued *before* the rotation remain
    /// valid until their own `not_after` expiry; vouchers issued
    /// *after* the rotation must use the new key.
    IssuerRotatedOut {
        member: String,
        rotated_to: String,
        rotation_timestamp: u64,
    },
    /// A voucher's fingerprint appears in a loaded
    /// `FederationRevocationList` signed by an in-chain operator.
    Revoked {
        member: String,
        revoking_issuer: String,
        revocation_time: u64,
        reason: String,
    },
}

impl std::fmt::Display for VoucherWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoucherWarning::Missing {
                member,
                expected_issuer_names,
            } => write!(
                f,
                "member {member:?} has no valid voucher from any higher-priority member \
                 (expected from one of: {expected_issuer_names:?})"
            ),
            VoucherWarning::Expired {
                member,
                issuer_name,
                not_after,
            } => write!(
                f,
                "member {member:?} voucher from {issuer_name:?} expired at unix {not_after}"
            ),
            VoucherWarning::SignatureInvalid {
                member,
                issuer_name,
            } => write!(
                f,
                "member {member:?} voucher from {issuer_name:?} has an invalid signature"
            ),
            VoucherWarning::DigestMismatch {
                member,
                issuer_name,
            } => write!(
                f,
                "member {member:?} voucher from {issuer_name:?} signed over a different file digest"
            ),
            VoucherWarning::UnknownIssuer { member, issuer_hex } => write!(
                f,
                "member {member:?} voucher signed by unknown issuer {issuer_hex} \
                 (no matching operator_key in any higher-priority member)"
            ),
            VoucherWarning::NoDigestPin { member } => write!(
                f,
                "member {member:?} has no digest pin so vouchers cannot bind to it"
            ),
            VoucherWarning::IssuerRotatedOut {
                member,
                rotated_to,
                rotation_timestamp,
            } => write!(
                f,
                "member {member:?} voucher signed by a key that was rotated to {rotated_to:?} \
                 at unix {rotation_timestamp}; voucher post-dates the rotation"
            ),
            VoucherWarning::Revoked {
                member,
                revoking_issuer,
                revocation_time,
                reason,
            } => write!(
                f,
                "member {member:?} voucher revoked by {revoking_issuer:?} at unix \
                 {revocation_time}: {reason}"
            ),
        }
    }
}

/// Composes multiple `TrustRegistry` instances into a single resolution
/// surface with explicit precedence and conflict reporting.
///
/// Use [`FederatedRegistry::resolve`] to produce a flat `TrustRegistry`
/// that the rest of the exchange pipeline can consume unchanged.  Use
/// [`FederatedRegistry::validate_consistency`] before resolving to see
/// any policy conflicts that the resolution will silently override.
#[derive(Debug, Clone)]
pub struct FederatedRegistry {
    members: Vec<NamedRegistry>,
    /// Operator key rotations accepted by this federation.  See
    /// `OperatorKeyRotation` and `add_key_rotation`.  Vouchers signed
    /// with a rotated-out key are accepted by `verify_vouchers` only
    /// if (a) there is a valid rotation chain from the signing key to
    /// some current `operator_key` and (b) the voucher's timestamp is
    /// strictly less than the timestamp of the first rotation away
    /// from the signing key.
    key_rotations: Vec<OperatorKeyRotation>,
    /// Federation revocation lists loaded into this federation.  Each
    /// FRL is signed by an operator; only FRLs whose issuer key is
    /// also in the verified chain are honoured by `verify_vouchers`.
    /// A voucher whose fingerprint appears in any honoured FRL is
    /// rejected with `VoucherWarning::Revoked`.
    frls: Vec<FederationRevocationList>,
}

impl FederatedRegistry {
    /// Build an empty federation.  Add member registries with [`add`].
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            key_rotations: Vec::new(),
            frls: Vec::new(),
        }
    }

    /// Build a federation from an ordered list of member registries.
    /// The order of `members` does not matter — they are sorted by
    /// `priority` (ascending) on insertion, so the resolution order is
    /// always priority-driven, never source-order-driven.
    pub fn from_members(mut members: Vec<NamedRegistry>) -> Self {
        members.sort_by_key(|m| m.priority);
        Self {
            members,
            key_rotations: Vec::new(),
            frls: Vec::new(),
        }
    }

    /// Add a member registry.  Re-sorts the federation by priority.
    pub fn add(&mut self, member: NamedRegistry) {
        self.members.push(member);
        self.members.sort_by_key(|m| m.priority);
    }

    /// Add an operator key rotation.  The rotation's cross-signatures
    /// are verified before insertion — invalid rotations are rejected
    /// with `WireError::Protocol`.
    pub fn add_key_rotation(&mut self, rotation: OperatorKeyRotation) -> Result<(), WireError> {
        rotation.verify()?;
        self.key_rotations.push(rotation);
        Ok(())
    }

    /// All accepted operator key rotations, in insertion order.
    pub fn key_rotations(&self) -> &[OperatorKeyRotation] {
        &self.key_rotations
    }

    /// Load a federation revocation list.  The FRL's signature is
    /// verified against the issuer key found via
    /// `vk_for_issuer_id` — i.e. the FRL must be signed by an
    /// operator whose key is currently or historically in the
    /// federation.  Whether the FRL is *honoured* by `verify_vouchers`
    /// is a separate decision based on the verified chain.
    pub fn add_frl(&mut self, frl: FederationRevocationList) -> Result<(), WireError> {
        let issuer_vk = self
            .vk_for_issuer_id(&frl.issuer_id)
            .ok_or_else(|| WireError::Protocol("FRL issuer not in federation".into()))?;
        frl.verify(&issuer_vk)?;
        self.frls.push(frl);
        Ok(())
    }

    /// All loaded federation revocation lists, in insertion order.
    pub fn frls(&self) -> &[FederationRevocationList] {
        &self.frls
    }

    /// All members in priority order (lowest priority first).
    pub fn members(&self) -> &[NamedRegistry] {
        &self.members
    }

    /// Total count of distinct member registries.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Look up an agent by ID, returning the first (highest-priority)
    /// member that knows about it.  Returns `(member_name, entry)` so
    /// the caller can tell which registry the entry came from.
    pub fn lookup_with_source(&self, agent_id: &[u8; 32]) -> Option<(&str, &AgentEntry)> {
        for m in &self.members {
            if let Some(entry) = m.registry.lookup(agent_id) {
                return Some((&m.name, entry));
            }
        }
        None
    }

    /// All entries for a given `agent_id` across the federation, in
    /// priority order.  Length 0 = unknown agent; length 1 = a single
    /// authoritative entry; length > 1 = the same key registered in
    /// multiple registries (potentially with different policy).
    pub fn entries_for(&self, agent_id: &[u8; 32]) -> Vec<(&str, &AgentEntry)> {
        let mut out = Vec::new();
        for m in &self.members {
            if let Some(entry) = m.registry.lookup(agent_id) {
                out.push((m.name.as_str(), entry));
            }
        }
        out
    }

    /// Walk the federation and collect every policy conflict.  A
    /// conflict is the same `agent_id` registered in two or more
    /// member registries with materially different policy fields:
    /// `max_drift_accepted`, `expected_model_hash`, `domain_scope`, or
    /// the set of governance pattern keys.
    ///
    /// Use this **before** [`resolve`] in deployments where conflicts
    /// should be reviewed.  Empty `Vec` means the federation is
    /// internally consistent.
    /// Verify the cross-registry voucher chain at `now_unix` using
    /// the default maximum chain depth ([`DEFAULT_MAX_VOUCHER_CHAIN_DEPTH`]).
    ///
    /// The lead member (priority 0) is the root of trust — it does
    /// not need a voucher.  Every other member is **multi-hop
    /// verified** if there exists a chain of valid vouchers leading
    /// back to the lead: the member must hold a valid voucher from
    /// some other member, who must in turn either be the lead or
    /// hold a valid voucher from another verified member, and so on.
    ///
    /// "Valid" means the voucher's `issuer_id` corresponds to some
    /// member's `operator_key`, the signature verifies against that
    /// key, the `subject_digest` matches the member's on-disk
    /// `digest`, and the voucher has not expired at `now_unix`.
    ///
    /// Returns one warning per non-verified member.  Empty `Vec`
    /// means the federation's voucher graph is fully connected back
    /// to the lead.  Note: this is structural verification only —
    /// the lead's own authority is taken on faith (configured into
    /// the deployment alongside the pinned registry digest).
    pub fn verify_vouchers(&self, now_unix: u64) -> Vec<VoucherWarning> {
        self.verify_vouchers_with_depth(now_unix, DEFAULT_MAX_VOUCHER_CHAIN_DEPTH)
    }

    /// Same as [`verify_vouchers`] with a configurable maximum chain
    /// depth.  Useful when you want to enforce a stricter limit than
    /// the default — e.g. healthcare federations might insist that
    /// every member be at most 2 hops from the lead.
    pub fn verify_vouchers_with_depth(
        &self,
        now_unix: u64,
        max_depth: usize,
    ) -> Vec<VoucherWarning> {
        let mut warnings = Vec::new();
        if self.members.len() < 2 {
            return warnings;
        }

        // ----- Pass 1: fixed-point iteration to find verified members.
        //
        // Start with `verified = {lead}`.  Each iteration takes a
        // snapshot of the current verified set, then walks every
        // non-verified member and adds it to a *new* working set if
        // any of its vouchers passes (issuer in the **snapshot**,
        // signature valid, digest matches, not expired).  Snapshotting
        // ensures each iteration adds *exactly one hop* to the set,
        // so `max_depth` is a real bound on chain length: max_depth
        // iterations can verify chains of length up to max_depth
        // hops from the lead.  Stop when no new member is added
        // (chain converged) or when `max_depth` iterations have run.
        // Cycles are handled implicitly — a cycle that never touches
        // an already-verified member never grows the set.
        let mut verified: HashSet<usize> = HashSet::new();
        verified.insert(0);

        for _ in 0..max_depth {
            let snapshot = verified.clone();
            let mut changed = false;
            for idx in 1..self.members.len() {
                if verified.contains(&idx) {
                    continue;
                }
                if self.member_has_valid_voucher_in(idx, &snapshot, now_unix) {
                    verified.insert(idx);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // ----- Pass 2: diagnose every non-verified member.
        //
        // For members that fixed-point could not verify, walk their
        // direct vouchers and emit specific failure warnings.  This
        // reproduces the per-failure granularity of the previous
        // single-hop API while still benefiting from multi-hop
        // semantics in pass 1 — a member that fails the direct
        // checks here would also have failed in any deeper traversal.
        for idx in 1..self.members.len() {
            if verified.contains(&idx) {
                continue;
            }
            self.diagnose_member_voucher_failures(idx, &verified, now_unix, &mut warnings);
        }

        warnings
    }

    /// Returns true if `member` has at least one voucher that passes
    /// every check (issuer in `verified_set`, signature valid, digest
    /// matches, not expired, and — for rotated-out keys — the
    /// voucher predates the first rotation away from the signing key).
    /// No warnings emitted — used for the fixed-point iteration where
    /// we only care about the boolean.
    fn member_has_valid_voucher_in(
        &self,
        member_idx: usize,
        verified_set: &HashSet<usize>,
        now_unix: u64,
    ) -> bool {
        let member = &self.members[member_idx];
        let member_digest = match member.digest {
            Some(d) => d,
            None => return false,
        };
        for voucher in &member.vouchers {
            // Resolve the voucher's issuer to a member, allowing the
            // signing key to be a rotated-out predecessor of a current
            // operator_key.  In that case the voucher is only valid if
            // its timestamp predates the first rotation away from the
            // signing key.
            let resolved = match self.resolve_voucher_issuer(voucher) {
                Some(r) => r,
                None => continue,
            };
            if !verified_set.contains(&resolved.member_idx) {
                continue;
            }
            // The verifying key is the *issuer's* key (which may be a
            // historical rotated-out key), not the current operator_key.
            let issuer_vk = match self.vk_for_issuer_id(&voucher.issuer_id) {
                Some(vk) => vk,
                None => continue,
            };
            if voucher.verify(&issuer_vk).is_err() {
                continue;
            }
            if voucher.subject_digest != member_digest {
                continue;
            }
            if !voucher.is_valid_at(now_unix) {
                continue;
            }
            // Revocation check: if any honoured FRL contains this
            // voucher's fingerprint, treat it as invalid.
            if self.voucher_revocation(voucher, verified_set).is_some() {
                continue;
            }
            return true;
        }
        false
    }

    /// Resolve a voucher's `issuer_id` to a current federation member.
    /// Returns the member index, the verifying key the signature was
    /// produced with, the rotation status, and (if rotated) the
    /// timestamp of the first rotation away from the issuer key.
    /// Includes the temporal check: vouchers signed with a rotated-out
    /// key must predate the rotation, otherwise the resolution returns
    /// `None`.
    fn resolve_voucher_issuer(&self, voucher: &FederationVoucher) -> Option<VoucherIssuerInfo> {
        // Direct hit: voucher was signed by a current operator_key.
        if let Some(idx) = self.member_idx_for_issuer(&voucher.issuer_id) {
            return Some(VoucherIssuerInfo {
                member_idx: idx,
                rotated: false,
            });
        }
        // Rotated: walk the rotation chain forward.
        let resolution = self.resolve_rotated_issuer(&voucher.issuer_id)?;
        if voucher.timestamp >= resolution.first_rotation_timestamp {
            // Voucher post-dates the rotation away from this key.
            return None;
        }
        Some(VoucherIssuerInfo {
            member_idx: resolution.current_member_idx,
            rotated: true,
        })
    }

    /// Check whether `voucher` has been revoked by any loaded FRL
    /// whose issuer is reachable in the verified set.  Returns
    /// `Some(revocation info)` if revoked, `None` otherwise.
    fn voucher_revocation(
        &self,
        voucher: &FederationVoucher,
        verified_set: &HashSet<usize>,
    ) -> Option<RevocationHit> {
        let fingerprint = voucher_fingerprint(voucher).ok()?;
        for frl in &self.frls {
            if !frl.contains(&fingerprint) {
                continue;
            }
            // FRL must be signed by an operator that is in the verified
            // chain.  An outsider's FRL cannot revoke our vouchers.
            let frl_issuer_idx = self.member_idx_for_issuer(&frl.issuer_id)?;
            if !verified_set.contains(&frl_issuer_idx) {
                continue;
            }
            let entry = frl
                .entries
                .iter()
                .find(|e| e.voucher_fingerprint == fingerprint)?;
            return Some(RevocationHit {
                revoking_issuer: self.members[frl_issuer_idx].name.clone(),
                revocation_time: entry.revocation_time,
                reason: entry.reason.clone(),
            });
        }
        None
    }

    /// Find the verifying key (current or historical) corresponding
    /// to a voucher's `issuer_id`.  Walks both current `operator_key`s
    /// and the `old_public_key` / `new_public_key` of every accepted
    /// rotation.
    fn vk_for_issuer_id(&self, issuer_id: &[u8; 32]) -> Option<VerifyingKey> {
        for m in &self.members {
            if let Some(vk) = m.operator_key {
                if compute_agent_id(&vk) == *issuer_id {
                    return Some(vk);
                }
            }
        }
        for r in &self.key_rotations {
            if compute_agent_id(&r.old_public_key) == *issuer_id {
                return Some(r.old_public_key);
            }
            if compute_agent_id(&r.new_public_key) == *issuer_id {
                return Some(r.new_public_key);
            }
        }
        None
    }

    /// Walk the direct vouchers on a non-verified member and emit a
    /// `VoucherWarning` for each specific failure.  Falls back to
    /// `Missing` if there were no vouchers at all.
    fn diagnose_member_voucher_failures(
        &self,
        member_idx: usize,
        verified_set: &HashSet<usize>,
        now_unix: u64,
        out: &mut Vec<VoucherWarning>,
    ) {
        let member = &self.members[member_idx];
        let member_digest = match member.digest {
            Some(d) => d,
            None => {
                out.push(VoucherWarning::NoDigestPin {
                    member: member.name.clone(),
                });
                return;
            }
        };

        if member.vouchers.is_empty() {
            // Best diagnostic: list the verified members the operator
            // could plausibly have asked for a voucher from.
            let expected: Vec<String> = (0..member_idx)
                .filter(|i| verified_set.contains(i))
                .filter(|i| self.members[*i].operator_key.is_some())
                .map(|i| self.members[i].name.clone())
                .collect();
            out.push(VoucherWarning::Missing {
                member: member.name.clone(),
                expected_issuer_names: expected,
            });
            return;
        }

        for voucher in &member.vouchers {
            // First: try direct issuer lookup.
            let direct_idx = self.member_idx_for_issuer(&voucher.issuer_id);
            // Second: try rotation chain (with timestamp check).
            let rotation = self.resolve_rotated_issuer(&voucher.issuer_id);

            let issuer_idx = match (direct_idx, &rotation) {
                (Some(i), _) => i,
                (None, Some(r)) => {
                    // Rotated-out key. Check the temporal constraint:
                    // voucher must predate the rotation, otherwise the
                    // operator should have re-signed with the new key.
                    if voucher.timestamp >= r.first_rotation_timestamp {
                        out.push(VoucherWarning::IssuerRotatedOut {
                            member: member.name.clone(),
                            rotated_to: r.current_member_name.clone(),
                            rotation_timestamp: r.first_rotation_timestamp,
                        });
                        continue;
                    }
                    r.current_member_idx
                }
                (None, None) => {
                    let issuer_hex: String = voucher
                        .issuer_id
                        .iter()
                        .take(8)
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                        + "…";
                    out.push(VoucherWarning::UnknownIssuer {
                        member: member.name.clone(),
                        issuer_hex,
                    });
                    continue;
                }
            };
            let issuer_name = self.members[issuer_idx].name.clone();
            // The issuer might exist but not be reachable through the
            // verified set — surface that as UnknownIssuer with a
            // qualifying note.
            if !verified_set.contains(&issuer_idx) {
                out.push(VoucherWarning::UnknownIssuer {
                    member: member.name.clone(),
                    issuer_hex: format!("{issuer_name} (not in verified chain to lead)"),
                });
                continue;
            }
            let issuer_vk = match self.vk_for_issuer_id(&voucher.issuer_id) {
                Some(vk) => vk,
                None => continue, // shouldn't happen if direct/rotation lookup succeeded
            };
            if voucher.verify(&issuer_vk).is_err() {
                out.push(VoucherWarning::SignatureInvalid {
                    member: member.name.clone(),
                    issuer_name,
                });
                continue;
            }
            if voucher.subject_digest != member_digest {
                out.push(VoucherWarning::DigestMismatch {
                    member: member.name.clone(),
                    issuer_name,
                });
                continue;
            }
            if !voucher.is_valid_at(now_unix) {
                out.push(VoucherWarning::Expired {
                    member: member.name.clone(),
                    issuer_name,
                    not_after: voucher.not_after,
                });
                continue;
            }
            if let Some(hit) = self.voucher_revocation(voucher, verified_set) {
                out.push(VoucherWarning::Revoked {
                    member: member.name.clone(),
                    revoking_issuer: hit.revoking_issuer,
                    revocation_time: hit.revocation_time,
                    reason: hit.reason,
                });
                continue;
            }
            // If we get here the voucher actually passed but the
            // member still wasn't marked verified — this shouldn't
            // happen unless max_depth was exceeded.  Emit Missing
            // to flag the depth-limit case.
            out.push(VoucherWarning::Missing {
                member: member.name.clone(),
                expected_issuer_names: vec![format!(
                    "{issuer_name} (chain exceeds max depth)"
                )],
            });
        }
    }

    /// Find the member index whose `operator_key` hashes to `issuer_id`.
    fn member_idx_for_issuer(&self, issuer_id: &[u8; 32]) -> Option<usize> {
        self.members.iter().position(|m| {
            m.operator_key
                .map(|vk| compute_agent_id(&vk) == *issuer_id)
                .unwrap_or(false)
        })
    }

    /// Walk the operator-key rotation graph forward from `issuer_id`,
    /// looking for a current member operator_key.  Returns the
    /// matching member index AND the timestamp of the first rotation
    /// away from `issuer_id` — the caller MUST check that the
    /// voucher's signing time is strictly less than this timestamp,
    /// otherwise the voucher post-dates a rotation and is invalid.
    ///
    /// Returns `None` if no chain reaches a current operator_key,
    /// or if the chain contains a cycle.
    fn resolve_rotated_issuer(&self, issuer_id: &[u8; 32]) -> Option<RotationResolution> {
        let mut current_id = *issuer_id;
        let mut visited = std::collections::HashSet::new();
        visited.insert(current_id);
        let mut first_rotation_ts: Option<u64> = None;
        let mut current_target_name: Option<String> = None;

        // Bound the walk by the number of rotations — defends against
        // pathological input even though `visited` already breaks
        // cycles.
        for _ in 0..self.key_rotations.len() + 1 {
            if let Some(idx) = self.member_idx_for_issuer(&current_id) {
                return Some(RotationResolution {
                    current_member_idx: idx,
                    first_rotation_timestamp: first_rotation_ts.unwrap_or(u64::MAX),
                    current_member_name: current_target_name
                        .unwrap_or_else(|| self.members[idx].name.clone()),
                });
            }
            // Find a verified rotation FROM current_id.
            let next = self.key_rotations.iter().find(|r| {
                compute_agent_id(&r.old_public_key) == current_id && r.verify().is_ok()
            });
            let r = match next {
                Some(r) => r,
                None => return None,
            };
            let new_id = compute_agent_id(&r.new_public_key);
            if !visited.insert(new_id) {
                return None;
            }
            if first_rotation_ts.is_none() {
                first_rotation_ts = Some(r.timestamp);
            }
            current_target_name = self
                .members
                .iter()
                .find(|m| {
                    m.operator_key
                        .map(|vk| compute_agent_id(&vk) == new_id)
                        .unwrap_or(false)
                })
                .map(|m| m.name.clone());
            current_id = new_id;
        }
        None
    }
}

/// Result of walking the operator key rotation chain to resolve a
/// voucher's issuer.
#[derive(Debug, Clone)]
struct RotationResolution {
    current_member_idx: usize,
    first_rotation_timestamp: u64,
    current_member_name: String,
}

/// Per-voucher issuer resolution result, used to bridge between the
/// voucher-walking helpers and the rotation graph.
#[derive(Debug, Clone)]
struct VoucherIssuerInfo {
    member_idx: usize,
    /// True if the voucher was signed with a rotated-out key whose
    /// chain we walked through.
    rotated: bool,
}

/// Result of looking up a voucher's fingerprint in the federation's
/// loaded FRLs.
#[derive(Debug, Clone)]
struct RevocationHit {
    revoking_issuer: String,
    revocation_time: u64,
    reason: String,
}

impl FederatedRegistry {

    pub fn validate_consistency(&self) -> Vec<FederationWarning> {
        let mut warnings = Vec::new();
        // Group entries by agent_id across all members.
        let mut by_id: BTreeMap<[u8; 32], Vec<(&str, &AgentEntry)>> = BTreeMap::new();
        for m in &self.members {
            for entry in m.registry.agents.values() {
                by_id
                    .entry(entry.agent_id)
                    .or_default()
                    .push((m.name.as_str(), entry));
            }
        }

        for (agent_id, entries) in by_id.iter() {
            if entries.len() < 2 {
                continue;
            }
            // Walk every pair (i, j) with i < j and report any field
            // that differs.  The pairwise loop produces redundant
            // warnings when N >= 3 but is the clearest output for N = 2,
            // which is by far the common case.  N <= small in practice.
            for i in 0..entries.len() {
                for j in (i + 1)..entries.len() {
                    let (a_name, a) = entries[i];
                    let (b_name, b) = entries[j];

                    if (a.max_drift_accepted - b.max_drift_accepted).abs() > f32::EPSILON {
                        warnings.push(FederationWarning::MaxDriftMismatch {
                            agent_id: *agent_id,
                            registry_a: a_name.into(),
                            registry_b: b_name.into(),
                            value_a: a.max_drift_accepted,
                            value_b: b.max_drift_accepted,
                        });
                    }

                    if a.expected_model_hash != b.expected_model_hash {
                        warnings.push(FederationWarning::ExpectedModelHashMismatch {
                            agent_id: *agent_id,
                            registry_a: a_name.into(),
                            registry_b: b_name.into(),
                        });
                    }

                    if scope_signature(a) != scope_signature(b) {
                        warnings.push(FederationWarning::DomainScopeMismatch {
                            agent_id: *agent_id,
                            registry_a: a_name.into(),
                            registry_b: b_name.into(),
                        });
                    }

                    if governance_signature(a) != governance_signature(b) {
                        warnings.push(FederationWarning::GovernanceTableMismatch {
                            agent_id: *agent_id,
                            registry_a: a_name.into(),
                            registry_b: b_name.into(),
                        });
                    }
                }
            }
        }

        warnings
    }

    /// Resolve the federation into a single flat `TrustRegistry`.
    ///
    /// Resolution order: members are walked in priority order; the
    /// first member to claim an `agent_id` wins, and lower-priority
    /// members are silently overridden for that ID.  Run
    /// [`validate_consistency`] first if you want to know what got
    /// overridden.
    ///
    /// The returned registry inherits its global `[registry]`-section
    /// fields (`max_chain_length`, `max_envelope_age_secs`,
    /// `max_attestation_age_secs`) from the highest-priority member.
    /// CA keys and CRLs are unioned across all members so the
    /// resolved registry trusts every CA any member trusts.
    pub fn resolve(&self) -> Result<TrustRegistry, WireError> {
        if self.members.is_empty() {
            return Ok(TrustRegistry::empty());
        }
        // Take the highest-priority member's globals as the baseline.
        let head = &self.members[0].registry;
        let mut out = TrustRegistry {
            agents: Default::default(),
            max_chain_length: head.max_chain_length,
            max_envelope_age_secs: head.max_envelope_age_secs,
            max_attestation_age_secs: head.max_attestation_age_secs,
            ca_public_keys: Vec::new(),
            crls: Vec::new(),
        };

        // Walk in priority order; first claim wins.
        for m in &self.members {
            for entry in m.registry.agents.values() {
                out.agents.entry(entry.agent_id).or_insert_with(|| entry.clone());
            }
            // Union CAs and CRLs so the resolved registry trusts every
            // certificate authority any member trusts.
            for ca in &m.registry.ca_public_keys {
                if !out.ca_public_keys.contains(ca) {
                    out.ca_public_keys.push(*ca);
                }
            }
            for crl in &m.registry.crls {
                out.crls.push(crl.clone());
            }
        }
        Ok(out)
    }
}

impl Default for FederatedRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A non-fatal divergence between two member registries in a federation.
#[derive(Debug, Clone, PartialEq)]
pub enum FederationWarning {
    MaxDriftMismatch {
        agent_id: [u8; 32],
        registry_a: String,
        registry_b: String,
        value_a: f32,
        value_b: f32,
    },
    ExpectedModelHashMismatch {
        agent_id: [u8; 32],
        registry_a: String,
        registry_b: String,
    },
    DomainScopeMismatch {
        agent_id: [u8; 32],
        registry_a: String,
        registry_b: String,
    },
    GovernanceTableMismatch {
        agent_id: [u8; 32],
        registry_a: String,
        registry_b: String,
    },
}

impl std::fmt::Display for FederationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id_hex = |id: &[u8; 32]| -> String {
            id.iter().take(8).map(|b| format!("{b:02x}")).collect::<String>() + "…"
        };
        match self {
            FederationWarning::MaxDriftMismatch {
                agent_id,
                registry_a,
                registry_b,
                value_a,
                value_b,
            } => write!(
                f,
                "agent {} max_drift_accepted differs: {registry_a:?}={value_a} vs {registry_b:?}={value_b}",
                id_hex(agent_id)
            ),
            FederationWarning::ExpectedModelHashMismatch {
                agent_id,
                registry_a,
                registry_b,
            } => write!(
                f,
                "agent {} expected_model_hash differs between {registry_a:?} and {registry_b:?}",
                id_hex(agent_id)
            ),
            FederationWarning::DomainScopeMismatch {
                agent_id,
                registry_a,
                registry_b,
            } => write!(
                f,
                "agent {} domain_scope differs between {registry_a:?} and {registry_b:?}",
                id_hex(agent_id)
            ),
            FederationWarning::GovernanceTableMismatch {
                agent_id,
                registry_a,
                registry_b,
            } => write!(
                f,
                "agent {} governance_table differs between {registry_a:?} and {registry_b:?}",
                id_hex(agent_id)
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Cheap structural fingerprint of an agent's `domain_scope` for
/// equality testing across federation members.  Sorts the permitted
/// and exclusion lists into canonical form so reordering is not
/// flagged as a conflict.
fn scope_signature(entry: &AgentEntry) -> Option<String> {
    let scope = entry.domain_scope.as_ref()?;
    let mut perm: Vec<String> = scope
        .permitted
        .iter()
        .map(|p| format!("{}:{:?}", p.pattern.canonical(), p.mode))
        .collect();
    perm.sort();
    let mut excl: Vec<String> = scope.exclusions.iter().map(|p| p.canonical()).collect();
    excl.sort();
    Some(format!(
        "primary={};permitted=[{}];exclusions=[{}]",
        scope.primary.as_str(),
        perm.join(","),
        excl.join(",")
    ))
}

/// Cheap structural fingerprint of an agent's `governance_table`.
fn governance_signature(entry: &AgentEntry) -> String {
    let mut rows: Vec<String> = entry
        .governance_table
        .entries
        .iter()
        .map(|e| {
            format!(
                "{}:max_drift={};min_conf={};min_causal={:?};req_chain={};req_causal={}",
                e.pattern.canonical(),
                e.thresholds.max_drift,
                e.thresholds.min_confidence,
                e.thresholds.min_causal_score,
                e.thresholds.require_chain,
                e.thresholds.require_causal_validation
            )
        })
        .collect();
    rows.sort();
    rows.join("|")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Domain, DomainPattern, DomainScope, InteractionMode, PermittedDomain};
    use crate::governance::{GovernanceEntry, GovernanceTable, GovernanceThresholds};
    use crate::registry::{compute_agent_id, AgentEntry};
    use ed25519_dalek::SigningKey;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn entry(name: &str, key: &SigningKey, max_drift: f32) -> AgentEntry {
        let pk = key.verifying_key();
        AgentEntry {
            name: name.into(),
            public_key: pk,
            agent_id: compute_agent_id(&pk),
            max_drift_accepted: max_drift,
            roles: vec![],
            expected_model_hash: None,
            certificate: None,
            domain_scope: None,
            governance_table: GovernanceTable::default(),
        }
    }

    fn registry_with(entries: Vec<AgentEntry>) -> TrustRegistry {
        let mut r = TrustRegistry::empty();
        for e in entries {
            r.add_agent(e);
        }
        r
    }

    fn member(name: &str, priority: u32, registry: TrustRegistry) -> NamedRegistry {
        NamedRegistry::unverified(name, priority, registry)
    }

    #[test]
    fn members_sort_by_priority() {
        let alice = key(0xAA);
        let fed = FederatedRegistry::from_members(vec![
            member("low", 5, registry_with(vec![entry("alice", &alice, 0.05)])),
            member("high", 0, registry_with(vec![entry("alice", &alice, 0.10)])),
            member("mid", 2, registry_with(vec![entry("alice", &alice, 0.07)])),
        ]);
        let names: Vec<&str> = fed.members().iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["high", "mid", "low"]);
    }

    #[test]
    fn lookup_returns_highest_priority_match() {
        let alice = key(0xAA);
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![entry("alice", &alice, 0.02)])),
            member("us", 1, registry_with(vec![entry("alice", &alice, 0.10)])),
        ]);
        let agent_id = compute_agent_id(&alice.verifying_key());
        let (source, e) = fed.lookup_with_source(&agent_id).unwrap();
        assert_eq!(source, "eu");
        assert!((e.max_drift_accepted - 0.02).abs() < f32::EPSILON);
    }

    #[test]
    fn lookup_returns_none_for_unknown() {
        let fed = FederatedRegistry::new();
        assert!(fed.lookup_with_source(&[0xFF; 32]).is_none());
    }

    #[test]
    fn entries_for_returns_all_matching_in_priority_order() {
        let alice = key(0xAA);
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![entry("alice-eu", &alice, 0.02)])),
            member("us", 1, registry_with(vec![entry("alice-us", &alice, 0.10)])),
            member("uk", 2, registry_with(vec![entry("alice-uk", &alice, 0.05)])),
        ]);
        let agent_id = compute_agent_id(&alice.verifying_key());
        let entries = fed.entries_for(&agent_id);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, "eu");
        assert_eq!(entries[1].0, "us");
        assert_eq!(entries[2].0, "uk");
    }

    #[test]
    fn validate_consistency_clean_federation() {
        let alice = key(0xAA);
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![entry("alice", &alice, 0.05)])),
            member("us", 1, registry_with(vec![entry("alice", &alice, 0.05)])),
        ]);
        let warnings = fed.validate_consistency();
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn validate_consistency_max_drift_mismatch() {
        let alice = key(0xAA);
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![entry("alice", &alice, 0.02)])),
            member("us", 1, registry_with(vec![entry("alice", &alice, 0.10)])),
        ]);
        let warnings = fed.validate_consistency();
        assert_eq!(warnings.len(), 1);
        assert!(matches!(
            warnings[0],
            FederationWarning::MaxDriftMismatch { .. }
        ));
    }

    #[test]
    fn validate_consistency_domain_scope_mismatch() {
        let alice = key(0xAA);
        let mut e1 = entry("alice", &alice, 0.05);
        e1.domain_scope = Some(DomainScope {
            primary: Domain::parse("agriculture.crop-management").unwrap(),
            permitted: vec![],
            exclusions: vec![],
        });
        let mut e2 = entry("alice", &alice, 0.05);
        e2.domain_scope = Some(DomainScope {
            primary: Domain::parse("agriculture.supply-chain").unwrap(),
            permitted: vec![],
            exclusions: vec![],
        });
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![e1])),
            member("us", 1, registry_with(vec![e2])),
        ]);
        let warnings = fed.validate_consistency();
        assert!(warnings.iter().any(|w| matches!(
            w,
            FederationWarning::DomainScopeMismatch { .. }
        )));
    }

    #[test]
    fn validate_consistency_ignores_pattern_reordering() {
        // Same scope, just permitted-list reordered.  Must NOT warn.
        let alice = key(0xAA);
        let mut e1 = entry("alice", &alice, 0.05);
        e1.domain_scope = Some(DomainScope {
            primary: Domain::parse("agriculture.crop-management").unwrap(),
            permitted: vec![
                PermittedDomain {
                    pattern: DomainPattern::parse("agriculture.*").unwrap(),
                    mode: InteractionMode::Cooperative,
                },
                PermittedDomain {
                    pattern: DomainPattern::parse("meteorology.*").unwrap(),
                    mode: InteractionMode::Advisory,
                },
            ],
            exclusions: vec![],
        });
        let mut e2 = entry("alice", &alice, 0.05);
        e2.domain_scope = Some(DomainScope {
            primary: Domain::parse("agriculture.crop-management").unwrap(),
            permitted: vec![
                PermittedDomain {
                    pattern: DomainPattern::parse("meteorology.*").unwrap(),
                    mode: InteractionMode::Advisory,
                },
                PermittedDomain {
                    pattern: DomainPattern::parse("agriculture.*").unwrap(),
                    mode: InteractionMode::Cooperative,
                },
            ],
            exclusions: vec![],
        });
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![e1])),
            member("us", 1, registry_with(vec![e2])),
        ]);
        let warnings = fed.validate_consistency();
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn validate_consistency_governance_mismatch() {
        let alice = key(0xAA);
        let mut e1 = entry("alice", &alice, 0.05);
        e1.governance_table = GovernanceTable {
            entries: vec![GovernanceEntry {
                pattern: DomainPattern::parse("healthcare.*").unwrap(),
                thresholds: GovernanceThresholds {
                    max_drift: 0.03,
                    min_confidence: 0.0,
                    min_causal_score: None,
                    require_chain: true,
                    require_causal_validation: true,
                },
            }],
        };
        let mut e2 = entry("alice", &alice, 0.05);
        e2.governance_table = GovernanceTable {
            entries: vec![GovernanceEntry {
                pattern: DomainPattern::parse("healthcare.*").unwrap(),
                thresholds: GovernanceThresholds {
                    max_drift: 0.10,
                    min_confidence: 0.0,
                    min_causal_score: None,
                    require_chain: false,
                    require_causal_validation: false,
                },
            }],
        };
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![e1])),
            member("us", 1, registry_with(vec![e2])),
        ]);
        let warnings = fed.validate_consistency();
        assert!(warnings.iter().any(|w| matches!(
            w,
            FederationWarning::GovernanceTableMismatch { .. }
        )));
    }

    #[test]
    fn resolve_picks_highest_priority_entry() {
        let alice = key(0xAA);
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![entry("alice", &alice, 0.02)])),
            member("us", 1, registry_with(vec![entry("alice", &alice, 0.10)])),
        ]);
        let resolved = fed.resolve().unwrap();
        let agent_id = compute_agent_id(&alice.verifying_key());
        let e = resolved.lookup(&agent_id).unwrap();
        assert!((e.max_drift_accepted - 0.02).abs() < f32::EPSILON);
    }

    #[test]
    fn resolve_unions_distinct_agents_across_members() {
        let alice = key(0xAA);
        let bob = key(0xBB);
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, registry_with(vec![entry("alice", &alice, 0.02)])),
            member("us", 1, registry_with(vec![entry("bob", &bob, 0.10)])),
        ]);
        let resolved = fed.resolve().unwrap();
        assert_eq!(resolved.agents.len(), 2);
        assert!(resolved
            .lookup(&compute_agent_id(&alice.verifying_key()))
            .is_some());
        assert!(resolved
            .lookup(&compute_agent_id(&bob.verifying_key()))
            .is_some());
    }

    #[test]
    fn resolve_inherits_globals_from_highest_priority() {
        let alice = key(0xAA);
        let mut eu = registry_with(vec![entry("alice", &alice, 0.02)]);
        eu.max_envelope_age_secs = 60;
        eu.max_chain_length = 50;
        let mut us = registry_with(vec![]);
        us.max_envelope_age_secs = 600;
        us.max_chain_length = 200;
        let fed = FederatedRegistry::from_members(vec![
            member("eu", 0, eu),
            member("us", 1, us),
        ]);
        let resolved = fed.resolve().unwrap();
        assert_eq!(resolved.max_envelope_age_secs, 60);
        assert_eq!(resolved.max_chain_length, 50);
    }

    #[test]
    fn empty_federation_resolves_to_empty_registry() {
        let fed = FederatedRegistry::new();
        let resolved = fed.resolve().unwrap();
        assert!(resolved.agents.is_empty());
    }

    // -----------------------------------------------------------------------
    // FederationVoucher unit tests
    // -----------------------------------------------------------------------

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    #[test]
    fn voucher_create_verify_roundtrip() {
        let issuer = key(0xEE);
        let issuer_id = compute_agent_id(&issuer.verifying_key());
        let v = FederationVoucher::create(
            issuer_id,
            [0xCC; 32],
            "us-fda",
            now(),
            now() + 86_400,
            "ratified by joint working group",
            &issuer,
        )
        .unwrap();
        v.verify(&issuer.verifying_key()).unwrap();
    }

    #[test]
    fn voucher_verify_fails_for_wrong_key() {
        let issuer = key(0xEE);
        let other = key(0xFF);
        let issuer_id = compute_agent_id(&issuer.verifying_key());
        let v = FederationVoucher::create(
            issuer_id,
            [0xCC; 32],
            "us-fda",
            now(),
            0,
            "",
            &issuer,
        )
        .unwrap();
        let err = v.verify(&other.verifying_key()).unwrap_err();
        assert!(matches!(err, WireError::Protocol(_)));
    }

    #[test]
    fn voucher_verify_fails_when_issuer_id_tampered() {
        let issuer = key(0xEE);
        let issuer_id = compute_agent_id(&issuer.verifying_key());
        let mut v = FederationVoucher::create(
            issuer_id,
            [0xCC; 32],
            "us-fda",
            now(),
            0,
            "",
            &issuer,
        )
        .unwrap();
        // Flip a byte in issuer_id; verify must reject because the
        // derived id from the key no longer matches.
        v.issuer_id[0] ^= 0xFF;
        let err = v.verify(&issuer.verifying_key()).unwrap_err();
        assert!(matches!(err, WireError::Protocol(_)));
    }

    #[test]
    fn voucher_expiry_check() {
        let issuer = key(0xEE);
        let issuer_id = compute_agent_id(&issuer.verifying_key());
        let v = FederationVoucher::create(
            issuer_id,
            [0xCC; 32],
            "us-fda",
            1_000_000,
            2_000_000,
            "",
            &issuer,
        )
        .unwrap();
        assert!(v.is_valid_at(1_500_000));
        assert!(!v.is_valid_at(2_000_001));
        // not_after = 0 means no expiry.
        let v2 = FederationVoucher::create(
            issuer_id,
            [0xCC; 32],
            "us-fda",
            1_000_000,
            0,
            "",
            &issuer,
        )
        .unwrap();
        assert!(v2.is_valid_at(u64::MAX / 2));
    }

    #[test]
    fn voucher_signable_bytes_rejects_oversized_strings() {
        let huge = "x".repeat(VOUCHER_MAX_STRING_LEN + 1);
        let v = FederationVoucher {
            voucher_version: VOUCHER_VERSION,
            issuer_id: [0; 32],
            subject_digest: [0; 32],
            subject_name: huge.clone(),
            timestamp: 0,
            not_after: 0,
            rationale: "".into(),
            signature: [0; 64],
        };
        assert!(v.signable_bytes().is_err());
        let v2 = FederationVoucher {
            subject_name: "ok".into(),
            rationale: huge,
            ..v
        };
        assert!(v2.signable_bytes().is_err());
    }

    // -----------------------------------------------------------------------
    // FederatedRegistry::verify_vouchers integration tests
    // -----------------------------------------------------------------------

    /// Build a NamedRegistry with a specific digest + operator key.
    fn member_with_pin(
        name: &str,
        priority: u32,
        registry: TrustRegistry,
        digest: [u8; 32],
        operator: &SigningKey,
    ) -> NamedRegistry {
        NamedRegistry {
            name: name.into(),
            priority,
            registry,
            digest: Some(digest),
            operator_key: Some(operator.verifying_key()),
            vouchers: Vec::new(),
        }
    }

    #[test]
    fn verify_vouchers_lead_only_passes() {
        let alice = key(0xAA);
        let eu_op = key(0xE1);
        let fed = FederatedRegistry::from_members(vec![member_with_pin(
            "eu",
            0,
            registry_with(vec![entry("alice", &alice, 0.02)]),
            [0x01; 32],
            &eu_op,
        )]);
        // Single member = trivially the lead, no vouchers required.
        let warnings = fed.verify_vouchers(now());
        assert!(warnings.is_empty());
    }

    #[test]
    fn verify_vouchers_missing_voucher_warns() {
        let alice = key(0xAA);
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        let fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![entry("alice", &alice, 0.02)]),
                [0x01; 32],
                &eu_op,
            ),
            member_with_pin(
                "us",
                1,
                registry_with(vec![entry("bob", &key(0xBB), 0.05)]),
                [0x02; 32],
                &us_op,
            ),
        ]);
        let warnings = fed.verify_vouchers(now());
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0], VoucherWarning::Missing { .. }));
    }

    #[test]
    fn verify_vouchers_valid_voucher_passes() {
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        let us_digest = [0x02; 32];
        let voucher = FederationVoucher::create(
            compute_agent_id(&eu_op.verifying_key()),
            us_digest,
            "us",
            now(),
            now() + 86_400,
            "vouched",
            &eu_op,
        )
        .unwrap();
        let mut us = member_with_pin(
            "us",
            1,
            registry_with(vec![entry("bob", &key(0xBB), 0.05)]),
            us_digest,
            &us_op,
        );
        us.vouchers.push(voucher);
        let fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![entry("alice", &key(0xAA), 0.02)]),
                [0x01; 32],
                &eu_op,
            ),
            us,
        ]);
        let warnings = fed.verify_vouchers(now());
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn verify_vouchers_digest_mismatch_warns() {
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        // EU vouches for us_digest = [0x02; 32]...
        let voucher = FederationVoucher::create(
            compute_agent_id(&eu_op.verifying_key()),
            [0x02; 32],
            "us",
            now(),
            now() + 86_400,
            "",
            &eu_op,
        )
        .unwrap();
        // ...but the on-disk member uses a different digest.
        let mut us = member_with_pin(
            "us",
            1,
            registry_with(vec![entry("bob", &key(0xBB), 0.05)]),
            [0xAB; 32],
            &us_op,
        );
        us.vouchers.push(voucher);
        let fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![entry("alice", &key(0xAA), 0.02)]),
                [0x01; 32],
                &eu_op,
            ),
            us,
        ]);
        let warnings = fed.verify_vouchers(now());
        assert!(warnings.iter().any(|w| matches!(w, VoucherWarning::DigestMismatch { .. })));
        // The new multi-hop diagnose pass emits one specific warning
        // per failed voucher and reserves Missing for the no-vouchers-
        // at-all case, so we should NOT also see Missing here.
        assert!(!warnings.iter().any(|w| matches!(w, VoucherWarning::Missing { .. })));
    }

    #[test]
    fn verify_vouchers_expired_warns() {
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        let us_digest = [0x02; 32];
        let voucher = FederationVoucher::create(
            compute_agent_id(&eu_op.verifying_key()),
            us_digest,
            "us",
            1_000_000,
            1_500_000, // expired before any realistic `now`
            "",
            &eu_op,
        )
        .unwrap();
        let mut us = member_with_pin(
            "us",
            1,
            registry_with(vec![entry("bob", &key(0xBB), 0.05)]),
            us_digest,
            &us_op,
        );
        us.vouchers.push(voucher);
        let fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![entry("alice", &key(0xAA), 0.02)]),
                [0x01; 32],
                &eu_op,
            ),
            us,
        ]);
        let warnings = fed.verify_vouchers(2_000_000);
        assert!(warnings.iter().any(|w| matches!(w, VoucherWarning::Expired { .. })));
    }

    #[test]
    fn verify_vouchers_unknown_issuer_warns() {
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        // Unknown issuer = not in the federation at all.
        let stranger = key(0xFE);
        let us_digest = [0x02; 32];
        let voucher = FederationVoucher::create(
            compute_agent_id(&stranger.verifying_key()),
            us_digest,
            "us",
            now(),
            now() + 86_400,
            "",
            &stranger,
        )
        .unwrap();
        let mut us = member_with_pin(
            "us",
            1,
            registry_with(vec![entry("bob", &key(0xBB), 0.05)]),
            us_digest,
            &us_op,
        );
        us.vouchers.push(voucher);
        let fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![entry("alice", &key(0xAA), 0.02)]),
                [0x01; 32],
                &eu_op,
            ),
            us,
        ]);
        let warnings = fed.verify_vouchers(now());
        assert!(warnings.iter().any(|w| matches!(w, VoucherWarning::UnknownIssuer { .. })));
    }

    // -----------------------------------------------------------------------
    // Multi-hop voucher chain tests
    // -----------------------------------------------------------------------

    /// Helper: create a voucher from `issuer` for `subject_digest`.
    fn voucher_for(
        issuer_op: &SigningKey,
        subject_digest: [u8; 32],
        subject_name: &str,
        not_after: u64,
    ) -> FederationVoucher {
        FederationVoucher::create(
            compute_agent_id(&issuer_op.verifying_key()),
            subject_digest,
            subject_name,
            now(),
            not_after,
            "test",
            issuer_op,
        )
        .unwrap()
    }

    #[test]
    fn verify_vouchers_two_hop_chain_passes() {
        // A → B → C: A is the lead, A vouches for B, B vouches for C.
        // Single-hop verify_vouchers would reject C; multi-hop must
        // accept it.
        let a_op = key(0xA1);
        let b_op = key(0xB1);
        let c_op = key(0xC1);
        let a_digest = [0xA0; 32];
        let b_digest = [0xB0; 32];
        let c_digest = [0xC0; 32];
        let exp = now() + 86_400;

        let mut b = member_with_pin(
            "b",
            1,
            registry_with(vec![entry("agent-b", &key(0xBB), 0.05)]),
            b_digest,
            &b_op,
        );
        b.vouchers.push(voucher_for(&a_op, b_digest, "b", exp));

        let mut c = member_with_pin(
            "c",
            2,
            registry_with(vec![entry("agent-c", &key(0xCC), 0.05)]),
            c_digest,
            &c_op,
        );
        c.vouchers.push(voucher_for(&b_op, c_digest, "c", exp));

        let fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "a",
                0,
                registry_with(vec![entry("agent-a", &key(0xAA), 0.02)]),
                a_digest,
                &a_op,
            ),
            b,
            c,
        ]);

        let warnings = fed.verify_vouchers(now());
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn verify_vouchers_three_hop_chain_passes() {
        // A → B → C → D
        let a_op = key(0xA1);
        let b_op = key(0xB1);
        let c_op = key(0xC1);
        let d_op = key(0xD1);
        let a_digest = [0xA0; 32];
        let b_digest = [0xB0; 32];
        let c_digest = [0xC0; 32];
        let d_digest = [0xD0; 32];
        let exp = now() + 86_400;

        let mut b = member_with_pin(
            "b",
            1,
            registry_with(vec![]),
            b_digest,
            &b_op,
        );
        b.vouchers.push(voucher_for(&a_op, b_digest, "b", exp));

        let mut c = member_with_pin(
            "c",
            2,
            registry_with(vec![]),
            c_digest,
            &c_op,
        );
        c.vouchers.push(voucher_for(&b_op, c_digest, "c", exp));

        let mut d = member_with_pin(
            "d",
            3,
            registry_with(vec![]),
            d_digest,
            &d_op,
        );
        d.vouchers.push(voucher_for(&c_op, d_digest, "d", exp));

        let fed = FederatedRegistry::from_members(vec![
            member_with_pin("a", 0, registry_with(vec![]), a_digest, &a_op),
            b,
            c,
            d,
        ]);

        let warnings = fed.verify_vouchers(now());
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn verify_vouchers_chain_exceeds_max_depth() {
        // Build a 5-hop chain (A → B → C → D → E → F) and verify
        // with max_depth=3.  The first three hops verify; F should
        // remain unverified and emit Missing with a "exceeds max
        // depth" hint.
        let ops: Vec<SigningKey> = (0..6).map(|i| key(0x10 + i)).collect();
        let digests: Vec<[u8; 32]> = (0..6).map(|i| [0x20 + i; 32]).collect();
        let exp = now() + 86_400;

        let mut members = Vec::new();
        members.push(member_with_pin(
            "a",
            0,
            registry_with(vec![]),
            digests[0],
            &ops[0],
        ));
        for i in 1..6 {
            let mut m = member_with_pin(
                ["b", "c", "d", "e", "f"][i - 1],
                i as u32,
                registry_with(vec![]),
                digests[i],
                &ops[i],
            );
            m.vouchers.push(voucher_for(
                &ops[i - 1],
                digests[i],
                ["b", "c", "d", "e", "f"][i - 1],
                exp,
            ));
            members.push(m);
        }

        let fed = FederatedRegistry::from_members(members);
        let warnings = fed.verify_vouchers_with_depth(now(), 3);
        // Members b, c, d are reachable in <= 3 hops; e and f are not.
        let unverified: Vec<&str> = warnings
            .iter()
            .filter_map(|w| match w {
                VoucherWarning::Missing { member, .. } => Some(member.as_str()),
                _ => None,
            })
            .collect();
        assert!(unverified.contains(&"e") || unverified.contains(&"f"));
    }

    #[test]
    fn verify_vouchers_cycle_does_not_loop() {
        // B vouches for C, C vouches for B.  Neither has a path to
        // the lead.  Both should fail without the algorithm hanging.
        let a_op = key(0xA1);
        let b_op = key(0xB1);
        let c_op = key(0xC1);
        let a_digest = [0xA0; 32];
        let b_digest = [0xB0; 32];
        let c_digest = [0xC0; 32];
        let exp = now() + 86_400;

        let mut b = member_with_pin(
            "b",
            1,
            registry_with(vec![]),
            b_digest,
            &b_op,
        );
        b.vouchers.push(voucher_for(&c_op, b_digest, "b", exp));

        let mut c = member_with_pin(
            "c",
            2,
            registry_with(vec![]),
            c_digest,
            &c_op,
        );
        c.vouchers.push(voucher_for(&b_op, c_digest, "c", exp));

        let fed = FederatedRegistry::from_members(vec![
            member_with_pin("a", 0, registry_with(vec![]), a_digest, &a_op),
            b,
            c,
        ]);

        let warnings = fed.verify_vouchers(now());
        // Neither B nor C is verified — both should appear in the
        // warning list.  The exact warning kind is the diagnose
        // pass's call (will be UnknownIssuer or Missing depending on
        // chain reachability).
        let mentioned_members: HashSet<&str> = warnings
            .iter()
            .filter_map(|w| match w {
                VoucherWarning::Missing { member, .. }
                | VoucherWarning::Expired { member, .. }
                | VoucherWarning::SignatureInvalid { member, .. }
                | VoucherWarning::DigestMismatch { member, .. }
                | VoucherWarning::UnknownIssuer { member, .. }
                | VoucherWarning::NoDigestPin { member }
                | VoucherWarning::IssuerRotatedOut { member, .. }
                | VoucherWarning::Revoked { member, .. } => Some(member.as_str()),
            })
            .collect();
        assert!(mentioned_members.contains("b"));
        assert!(mentioned_members.contains("c"));
    }

    // -----------------------------------------------------------------------
    // OperatorKeyRotation tests
    // -----------------------------------------------------------------------

    #[test]
    fn operator_rotation_create_verify_roundtrip() {
        let old = key(0xA1);
        let new_key = key(0xA2);
        let r = OperatorKeyRotation::create(&old, &new_key, 1_000_000);
        r.verify().unwrap();
        assert_eq!(r.old_public_key, old.verifying_key());
        assert_eq!(r.new_public_key, new_key.verifying_key());
        assert_eq!(r.timestamp, 1_000_000);
    }

    #[test]
    fn operator_rotation_tampered_signature_rejected() {
        let old = key(0xA1);
        let new_key = key(0xA2);
        let mut r = OperatorKeyRotation::create(&old, &new_key, 1_000_000);
        r.old_key_signature[0] ^= 0xFF;
        assert!(r.verify().is_err());
    }

    #[test]
    fn operator_rotation_tampered_timestamp_rejected() {
        let old = key(0xA1);
        let new_key = key(0xA2);
        let mut r = OperatorKeyRotation::create(&old, &new_key, 1_000_000);
        // Tampering the timestamp invalidates both signatures since
        // they cover the canonical bytes that include the timestamp.
        r.timestamp = 2_000_000;
        assert!(r.verify().is_err());
    }

    #[test]
    fn add_key_rotation_rejects_invalid() {
        let old = key(0xA1);
        let new_key = key(0xA2);
        let mut r = OperatorKeyRotation::create(&old, &new_key, 1_000_000);
        r.new_key_signature[0] ^= 0xFF;
        let mut fed = FederatedRegistry::new();
        assert!(fed.add_key_rotation(r).is_err());
        assert_eq!(fed.key_rotations().len(), 0);
    }

    #[test]
    fn voucher_signed_by_rotated_key_before_rotation_passes() {
        // Setup: A vouches for B at time T0; later A rotates to A'.
        // The original voucher (signed by A at T0) must still verify.
        let a_old = key(0xA1);
        let a_new = key(0xA2);
        let b_op = key(0xB1);
        let b_digest = [0xB0; 32];

        // Voucher signed at T0 = 500_000 by the OLD A key.
        let voucher = FederationVoucher::create(
            compute_agent_id(&a_old.verifying_key()),
            b_digest,
            "b",
            500_000,
            5_000_000,
            "issued before rotation",
            &a_old,
        )
        .unwrap();

        // Rotation happens at T1 = 1_000_000 (after the voucher).
        let rotation = OperatorKeyRotation::create(&a_old, &a_new, 1_000_000);

        let mut b = member_with_pin(
            "b",
            1,
            registry_with(vec![]),
            b_digest,
            &b_op,
        );
        b.vouchers.push(voucher);

        // The federation now uses the NEW A key as the operator.
        let mut fed = FederatedRegistry::from_members(vec![
            NamedRegistry {
                name: "a".into(),
                priority: 0,
                registry: registry_with(vec![]),
                digest: Some([0xA0; 32]),
                operator_key: Some(a_new.verifying_key()),
                vouchers: vec![],
            },
            b,
        ]);
        fed.add_key_rotation(rotation).unwrap();

        let warnings = fed.verify_vouchers(2_000_000);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn voucher_signed_by_rotated_key_after_rotation_warns() {
        // Same setup as above but the voucher is signed AFTER the
        // rotation — must be rejected.
        let a_old = key(0xA1);
        let a_new = key(0xA2);
        let b_op = key(0xB1);
        let b_digest = [0xB0; 32];

        let rotation = OperatorKeyRotation::create(&a_old, &a_new, 1_000_000);

        // Voucher signed by old A at T = 1_500_000 (AFTER the rotation).
        let voucher = FederationVoucher::create(
            compute_agent_id(&a_old.verifying_key()),
            b_digest,
            "b",
            1_500_000,
            5_000_000,
            "issued after rotation - should be rejected",
            &a_old,
        )
        .unwrap();

        let mut b = member_with_pin(
            "b",
            1,
            registry_with(vec![]),
            b_digest,
            &b_op,
        );
        b.vouchers.push(voucher);

        let mut fed = FederatedRegistry::from_members(vec![
            NamedRegistry {
                name: "a".into(),
                priority: 0,
                registry: registry_with(vec![]),
                digest: Some([0xA0; 32]),
                operator_key: Some(a_new.verifying_key()),
                vouchers: vec![],
            },
            b,
        ]);
        fed.add_key_rotation(rotation).unwrap();

        let warnings = fed.verify_vouchers(2_000_000);
        assert!(warnings
            .iter()
            .any(|w| matches!(w, VoucherWarning::IssuerRotatedOut { .. })));
    }

    #[test]
    fn voucher_signed_by_two_step_rotation_chain_passes() {
        // A → A' → A''.  Voucher signed by A at T0 (before any
        // rotation) must verify even though A is two hops away from
        // the current operator key.
        let a0 = key(0xA1);
        let a1 = key(0xA2);
        let a2 = key(0xA3);
        let b_op = key(0xB1);
        let b_digest = [0xB0; 32];

        let voucher = FederationVoucher::create(
            compute_agent_id(&a0.verifying_key()),
            b_digest,
            "b",
            100_000,
            5_000_000,
            "issued by oldest key",
            &a0,
        )
        .unwrap();

        let rot1 = OperatorKeyRotation::create(&a0, &a1, 500_000);
        let rot2 = OperatorKeyRotation::create(&a1, &a2, 1_000_000);

        let mut b = member_with_pin(
            "b",
            1,
            registry_with(vec![]),
            b_digest,
            &b_op,
        );
        b.vouchers.push(voucher);

        let mut fed = FederatedRegistry::from_members(vec![
            NamedRegistry {
                name: "a".into(),
                priority: 0,
                registry: registry_with(vec![]),
                digest: Some([0xA0; 32]),
                operator_key: Some(a2.verifying_key()),
                vouchers: vec![],
            },
            b,
        ]);
        fed.add_key_rotation(rot1).unwrap();
        fed.add_key_rotation(rot2).unwrap();

        let warnings = fed.verify_vouchers(2_000_000);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    // -----------------------------------------------------------------------
    // FederationSyncSource tests
    // -----------------------------------------------------------------------

    #[test]
    fn static_sync_source_returns_snapshot_then_none() {
        let bytes = b"hello federation".to_vec();
        let src = StaticSyncSource::new("test", bytes.clone(), 1_000_000);
        let first = src.fetch(None).unwrap().expect("first fetch returns content");
        assert_eq!(first.bytes, bytes);
        assert_eq!(first.fetched_at, 1_000_000);
        // Repeating the fetch with the same digest returns None.
        let second = src.fetch(Some(first.digest)).unwrap();
        assert!(second.is_none());
        // A different `since` returns the snapshot again.
        let third = src.fetch(Some([0xFF; 32])).unwrap();
        assert!(third.is_some());
    }

    #[test]
    fn file_sync_source_detects_change_via_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.toml");
        std::fs::write(&path, b"version = 1").unwrap();

        let src = FileSyncSource::new("test", &path);
        let first = src.fetch(None).unwrap().expect("first fetch returns content");
        assert_eq!(first.bytes, b"version = 1");

        // Re-fetch with the same digest: should return None.
        let same = src.fetch(Some(first.digest)).unwrap();
        assert!(same.is_none());

        // Modify the file: should return new content.
        std::fs::write(&path, b"version = 2").unwrap();
        let updated = src
            .fetch(Some(first.digest))
            .unwrap()
            .expect("updated content");
        assert_eq!(updated.bytes, b"version = 2");
        assert_ne!(updated.digest, first.digest);
    }

    #[test]
    fn file_sync_source_propagates_io_errors() {
        let src = FileSyncSource::new("missing", "/this/path/does/not/exist.toml");
        let err = src.fetch(None).unwrap_err();
        assert!(matches!(err, WireError::Io(_)));
    }

    // -----------------------------------------------------------------------
    // FederationRevocationList tests
    // -----------------------------------------------------------------------

    #[test]
    fn frl_create_verify_roundtrip() {
        let signer = key(0xF1);
        let issuer_id = compute_agent_id(&signer.verifying_key());
        let frl = FederationRevocationList::create(
            issuer_id,
            1_000_000,
            vec![RevokedVoucher {
                voucher_fingerprint: [0xDE; 32],
                revocation_time: 999_999,
                reason: "key-compromise".into(),
            }],
            &signer,
        )
        .unwrap();
        frl.verify(&signer.verifying_key()).unwrap();
        assert!(frl.contains(&[0xDE; 32]));
        assert!(!frl.contains(&[0xEF; 32]));
    }

    #[test]
    fn frl_tampered_signature_rejected() {
        let signer = key(0xF1);
        let issuer_id = compute_agent_id(&signer.verifying_key());
        let mut frl = FederationRevocationList::create(
            issuer_id,
            0,
            vec![],
            &signer,
        )
        .unwrap();
        frl.signature[0] ^= 0xFF;
        assert!(frl.verify(&signer.verifying_key()).is_err());
    }

    #[test]
    fn voucher_fingerprint_is_deterministic_and_distinct() {
        let issuer = key(0xF1);
        let issuer_id = compute_agent_id(&issuer.verifying_key());
        let v1 = FederationVoucher::create(
            issuer_id, [0xCC; 32], "us", 1000, 9999, "first", &issuer,
        )
        .unwrap();
        let v2 = FederationVoucher::create(
            issuer_id, [0xCC; 32], "us", 1001, 9999, "first", &issuer,
        )
        .unwrap();
        let f1 = voucher_fingerprint(&v1).unwrap();
        let f2 = voucher_fingerprint(&v2).unwrap();
        assert_eq!(f1, voucher_fingerprint(&v1).unwrap());
        assert_ne!(f1, f2, "different timestamp ⇒ different fingerprint");
    }

    #[test]
    fn add_frl_rejects_outsider_issuer() {
        let mut fed = FederatedRegistry::new();
        let outsider = key(0xFE);
        let outsider_id = compute_agent_id(&outsider.verifying_key());
        let frl = FederationRevocationList::create(outsider_id, 0, vec![], &outsider).unwrap();
        // No member knows the outsider, so add_frl rejects it.
        assert!(fed.add_frl(frl).is_err());
    }

    #[test]
    fn revoked_voucher_is_rejected() {
        // EU vouches for US.  EU then revokes the voucher.
        // verify_vouchers must return Revoked, not Accepted.
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        let us_digest = [0x02; 32];

        let voucher = FederationVoucher::create(
            compute_agent_id(&eu_op.verifying_key()),
            us_digest,
            "us",
            now(),
            now() + 86_400,
            "before revocation",
            &eu_op,
        )
        .unwrap();
        let voucher_fp = voucher_fingerprint(&voucher).unwrap();

        let mut us = member_with_pin(
            "us",
            1,
            registry_with(vec![]),
            us_digest,
            &us_op,
        );
        us.vouchers.push(voucher);

        let mut fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![]),
                [0x01; 32],
                &eu_op,
            ),
            us,
        ]);

        // Sanity check: without the FRL, the voucher passes.
        assert!(fed.verify_vouchers(now()).is_empty());

        // Now publish the FRL.
        let frl = FederationRevocationList::create(
            compute_agent_id(&eu_op.verifying_key()),
            now(),
            vec![RevokedVoucher {
                voucher_fingerprint: voucher_fp,
                revocation_time: now(),
                reason: "key-compromise".into(),
            }],
            &eu_op,
        )
        .unwrap();
        fed.add_frl(frl).unwrap();

        let warnings = fed.verify_vouchers(now());
        assert!(warnings.iter().any(|w| matches!(w, VoucherWarning::Revoked { .. })));
    }

    #[test]
    fn revocation_only_honoured_for_in_chain_issuer() {
        // EU vouches for US.  An *outsider* (not in the federation
        // and not in the chain) signs an FRL listing the US voucher's
        // fingerprint.  The FRL is loaded as if the outsider were a
        // member, but verify_vouchers should NOT honour it because
        // the outsider is not in the verified chain to the lead.
        //
        // We model this by adding a low-priority member whose
        // operator key publishes the FRL — but the member has no
        // voucher chain back to the lead, so it is itself unverified
        // and its FRL is ignored.
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        let outsider = key(0xFE);
        let us_digest = [0x02; 32];
        let outsider_digest = [0x99; 32];

        let voucher = FederationVoucher::create(
            compute_agent_id(&eu_op.verifying_key()),
            us_digest,
            "us",
            now(),
            now() + 86_400,
            "",
            &eu_op,
        )
        .unwrap();
        let voucher_fp = voucher_fingerprint(&voucher).unwrap();

        let mut us = member_with_pin(
            "us",
            1,
            registry_with(vec![]),
            us_digest,
            &us_op,
        );
        us.vouchers.push(voucher);

        // Add the outsider as a federation member but with no
        // voucher chain to the lead.  The outsider's FRL will load
        // (it's a known operator key) but won't be honoured.
        let outsider_member = member_with_pin(
            "outsider",
            2,
            registry_with(vec![]),
            outsider_digest,
            &outsider,
        );

        let mut fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![]),
                [0x01; 32],
                &eu_op,
            ),
            us,
            outsider_member,
        ]);

        let frl = FederationRevocationList::create(
            compute_agent_id(&outsider.verifying_key()),
            now(),
            vec![RevokedVoucher {
                voucher_fingerprint: voucher_fp,
                revocation_time: now(),
                reason: "trying to revoke from outside".into(),
            }],
            &outsider,
        )
        .unwrap();
        fed.add_frl(frl).unwrap();

        let warnings = fed.verify_vouchers(now());
        // US must NOT have a Revoked warning (the FRL came from an
        // unverified issuer).  US is itself accepted by EU's voucher.
        // The outsider IS unverified, so we expect a Missing warning
        // for the outsider, but NOT a Revoked warning anywhere.
        assert!(!warnings
            .iter()
            .any(|w| matches!(w, VoucherWarning::Revoked { .. })));
        assert!(warnings.iter().any(|w| matches!(w, VoucherWarning::Missing { member, .. } if member == "outsider")));
    }

    #[test]
    fn rotation_chain_temporal_check_uses_first_hop() {
        // A0 rotates to A1 at T=500.  A1 rotates to A2 at T=1000.
        // A voucher signed by A0 at T=600 (after rot1 but before
        // rot2) is INVALID because the relevant rotation is the
        // first one away from A0 (T=500), not the latest one (T=1000).
        let a0 = key(0xA1);
        let a1 = key(0xA2);
        let a2 = key(0xA3);
        let b_op = key(0xB1);
        let b_digest = [0xB0; 32];

        let voucher = FederationVoucher::create(
            compute_agent_id(&a0.verifying_key()),
            b_digest,
            "b",
            600, // after rot1 (T=500), before rot2 (T=1000)
            10_000,
            "should be rejected: post-dates rot1",
            &a0,
        )
        .unwrap();

        let rot1 = OperatorKeyRotation::create(&a0, &a1, 500);
        let rot2 = OperatorKeyRotation::create(&a1, &a2, 1_000);

        let mut b = member_with_pin(
            "b",
            1,
            registry_with(vec![]),
            b_digest,
            &b_op,
        );
        b.vouchers.push(voucher);

        let mut fed = FederatedRegistry::from_members(vec![
            NamedRegistry {
                name: "a".into(),
                priority: 0,
                registry: registry_with(vec![]),
                digest: Some([0xA0; 32]),
                operator_key: Some(a2.verifying_key()),
                vouchers: vec![],
            },
            b,
        ]);
        fed.add_key_rotation(rot1).unwrap();
        fed.add_key_rotation(rot2).unwrap();

        let warnings = fed.verify_vouchers(2_000);
        assert!(warnings
            .iter()
            .any(|w| matches!(w, VoucherWarning::IssuerRotatedOut { rotation_timestamp, .. } if *rotation_timestamp == 500)));
    }

    #[test]
    fn verify_vouchers_three_hop_chain_fails_at_depth_2() {
        // A → B → C → D, max_depth = 2.  D is unreachable.
        let ops: Vec<SigningKey> = (0..4).map(|i| key(0x40 + i)).collect();
        let digests: Vec<[u8; 32]> = (0..4).map(|i| [0x50 + i; 32]).collect();
        let exp = now() + 86_400;

        let mut members = vec![member_with_pin(
            "a",
            0,
            registry_with(vec![]),
            digests[0],
            &ops[0],
        )];
        let names = ["b", "c", "d"];
        for i in 1..4 {
            let mut m = member_with_pin(
                names[i - 1],
                i as u32,
                registry_with(vec![]),
                digests[i],
                &ops[i],
            );
            m.vouchers
                .push(voucher_for(&ops[i - 1], digests[i], names[i - 1], exp));
            members.push(m);
        }

        let fed = FederatedRegistry::from_members(members);

        // depth 3 → all four members verified (lead + b + c + d).
        assert!(fed.verify_vouchers_with_depth(now(), 3).is_empty());
        // depth 2 → b and c verified, d not.
        let warnings_at_2 = fed.verify_vouchers_with_depth(now(), 2);
        assert_eq!(warnings_at_2.len(), 1);
        let only = warnings_at_2.iter().find_map(|w| match w {
            VoucherWarning::Missing { member, .. } => Some(member.as_str()),
            _ => None,
        });
        assert_eq!(only, Some("d"));
    }

    #[test]
    fn verify_vouchers_no_digest_pin_warns() {
        let eu_op = key(0xE1);
        let us_op = key(0xE2);
        let mut us = member_with_pin(
            "us",
            1,
            registry_with(vec![entry("bob", &key(0xBB), 0.05)]),
            [0x02; 32],
            &us_op,
        );
        us.digest = None; // no pin
        let fed = FederatedRegistry::from_members(vec![
            member_with_pin(
                "eu",
                0,
                registry_with(vec![entry("alice", &key(0xAA), 0.02)]),
                [0x01; 32],
                &eu_op,
            ),
            us,
        ]);
        let warnings = fed.verify_vouchers(now());
        assert!(warnings.iter().any(|w| matches!(w, VoucherWarning::NoDigestPin { .. })));
    }
}
