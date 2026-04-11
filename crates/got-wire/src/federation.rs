// ---------------------------------------------------------------------------
// Federated registry composer — Protocol §14.5 (scoped).
//
// The full federation design from the protocol paper involves signed
// cross-registry vouching, async sync between authorities, revocation
// propagation, and arbitration policies for jurisdictional conflicts.
// That is multi-week work and out of scope for the PoC.
//
// What this module provides instead is the *composition* layer: a
// `FederatedRegistry` that wraps an ordered list of named
// `TrustRegistry` instances with explicit priority, lets the
// application code resolve any agent across all of them, reports
// policy conflicts (the same agent_id with different domain_scope or
// governance bounds in two registries), and produces a single flat
// `TrustRegistry` that the rest of the exchange code consumes
// unchanged.  The federation layer is *outside* the existing
// validation pipeline — no signature changes to `validate_request`,
// no new generic parameters, no new wire format.
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
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;

use crate::registry::{AgentEntry, TrustRegistry};
use crate::WireError;

/// One member registry of a federation.
///
/// `priority` is the resolution rank: lower numbers win on conflict.
/// `name` is a human label that appears in `FederationWarning`s and
/// in the merged registry's audit trail.
#[derive(Debug, Clone)]
pub struct NamedRegistry {
    pub name: String,
    pub priority: u32,
    pub registry: TrustRegistry,
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
        NamedRegistry {
            name: name.into(),
            priority,
            registry,
        }
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
}
