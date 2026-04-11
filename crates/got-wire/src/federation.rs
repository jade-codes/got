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

use std::collections::BTreeMap;

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};

use crate::registry::{compute_agent_id, AgentEntry, TrustRegistry};
use crate::WireError;

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
}

impl FederatedRegistry {
    /// Build an empty federation.  Add member registries with [`add`].
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
        }
    }

    /// Build a federation from an ordered list of member registries.
    /// The order of `members` does not matter — they are sorted by
    /// `priority` (ascending) on insertion, so the resolution order is
    /// always priority-driven, never source-order-driven.
    pub fn from_members(mut members: Vec<NamedRegistry>) -> Self {
        members.sort_by_key(|m| m.priority);
        Self { members }
    }

    /// Add a member registry.  Re-sorts the federation by priority.
    pub fn add(&mut self, member: NamedRegistry) {
        self.members.push(member);
        self.members.sort_by_key(|m| m.priority);
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
    /// Verify the cross-registry voucher chain at `now_unix`.
    ///
    /// The lead member (lowest priority number) is the root of trust
    /// — it does not need a voucher.  Every other member must carry
    /// at least one valid voucher from a higher-priority member that
    /// has its `operator_key` set, signs over the member's `digest`,
    /// and has not expired.
    ///
    /// Returns one warning per failure.  Empty `Vec` means the
    /// federation's voucher chain is fully verified at `now_unix`.
    ///
    /// Note: this is structural verification only.  The lead's own
    /// authority is taken on faith — establishing the root operator
    /// key is an out-of-band governance decision (typically the
    /// member is configured into a deployment by the same process
    /// that pins the registry digest).
    pub fn verify_vouchers(&self, now_unix: u64) -> Vec<VoucherWarning> {
        let mut warnings = Vec::new();
        if self.members.len() < 2 {
            return warnings;
        }

        // Build a quick lookup: issuer_id → (member_name, vk).
        // Only members with an operator_key participate as issuers.
        let mut issuer_index: BTreeMap<[u8; 32], (&str, VerifyingKey)> = BTreeMap::new();
        for m in &self.members {
            if let Some(vk) = m.operator_key {
                issuer_index.insert(compute_agent_id(&vk), (m.name.as_str(), vk));
            }
        }

        // Walk every non-lead member.
        for (idx, member) in self.members.iter().enumerate() {
            if idx == 0 {
                // Lead is the root of trust.
                continue;
            }

            let member_digest = match member.digest {
                Some(d) => d,
                None => {
                    warnings.push(VoucherWarning::NoDigestPin {
                        member: member.name.clone(),
                    });
                    continue;
                }
            };

            // Higher-priority members are everything before this index
            // in the priority-sorted list.
            let higher_priority: Vec<&str> = self.members[..idx]
                .iter()
                .filter(|m| m.operator_key.is_some())
                .map(|m| m.name.as_str())
                .collect();

            let mut found_valid = false;
            for voucher in &member.vouchers {
                // 1. Issuer must be known.
                let (issuer_name, issuer_vk) = match issuer_index.get(&voucher.issuer_id) {
                    Some(entry) => *entry,
                    None => {
                        let issuer_hex: String = voucher
                            .issuer_id
                            .iter()
                            .take(8)
                            .map(|b| format!("{b:02x}"))
                            .collect::<String>()
                            + "…";
                        warnings.push(VoucherWarning::UnknownIssuer {
                            member: member.name.clone(),
                            issuer_hex,
                        });
                        continue;
                    }
                };

                // 2. Issuer must be a *higher* priority member.
                let issuer_is_higher_priority = self.members[..idx]
                    .iter()
                    .any(|m| m.name == issuer_name);
                if !issuer_is_higher_priority {
                    warnings.push(VoucherWarning::UnknownIssuer {
                        member: member.name.clone(),
                        issuer_hex: format!("{issuer_name} (not higher priority)"),
                    });
                    continue;
                }

                // 3. Signature must verify.
                if voucher.verify(&issuer_vk).is_err() {
                    warnings.push(VoucherWarning::SignatureInvalid {
                        member: member.name.clone(),
                        issuer_name: issuer_name.into(),
                    });
                    continue;
                }

                // 4. Digest must match the on-disk file.
                if voucher.subject_digest != member_digest {
                    warnings.push(VoucherWarning::DigestMismatch {
                        member: member.name.clone(),
                        issuer_name: issuer_name.into(),
                    });
                    continue;
                }

                // 5. Must not be expired.
                if !voucher.is_valid_at(now_unix) {
                    warnings.push(VoucherWarning::Expired {
                        member: member.name.clone(),
                        issuer_name: issuer_name.into(),
                        not_after: voucher.not_after,
                    });
                    continue;
                }

                // All checks passed.
                found_valid = true;
                break;
            }

            if !found_valid {
                warnings.push(VoucherWarning::Missing {
                    member: member.name.clone(),
                    expected_issuer_names: higher_priority
                        .into_iter()
                        .map(String::from)
                        .collect(),
                });
            }
        }

        warnings
    }

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
        // ...and the lookup fails to find a *passing* voucher, so it
        // also reports Missing.
        assert!(warnings.iter().any(|w| matches!(w, VoucherWarning::Missing { .. })));
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
