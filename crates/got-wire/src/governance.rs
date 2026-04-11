// ---------------------------------------------------------------------------
// Governance Thresholds — Protocol §7.3 / §8.2.
//
// Per-domain quantitative bounds that a verifier applies to an incoming
// attestation, keyed by the *peer's* primary domain.  The paper calls out
// that a healthcare agent should be held to stricter drift and confidence
// bounds than a commercial supply-chain agent, and that critical-
// infrastructure deployments may additionally mandate Tier 3 causal
// attestations.  This module provides the structured policy container and
// the most-specific-pattern lookup that validate_request / validate_response
// consult after domain compatibility passes.
//
// Backwards compatibility: if an AgentEntry declares no governance table,
// the verifier falls back to the flat `max_drift_accepted` field, which is
// how every pre-§4 registry worked.
// ---------------------------------------------------------------------------

use std::collections::HashSet;

use serde::Deserialize;

use crate::domain::{Domain, DomainPattern};
use crate::WireError;

/// Quantitative thresholds a verifier enforces against an attestation.
///
/// Trust tiers in the protocol paper (Tier 1/2/3) are *content-based* —
/// they are derived from which fields the attestation actually populates.
/// Tier 2 requires `parent_attestation_hash` + `geometry_drift` (a chain);
/// Tier 3 requires non-empty `causal_scores` with every record causal.
/// Governance policy is therefore expressed as `require_chain` and
/// `require_causal_validation` rather than a numeric version gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernanceThresholds {
    /// Maximum acceptable Frobenius drift on the chain (§7.3).
    pub max_drift: f32,
    /// Minimum per-reading confidence (§8.2 "probe reading thresholds").
    /// 0.0 = no check.
    pub min_confidence: f32,
    /// Minimum causal consistency score for Tier-3 attestations.
    /// `None` = no check.  §5.4 mandates ≥ 0.85 for critical infra.
    pub min_causal_score: Option<f32>,
    /// §8.2: require the attestation to belong to a chain (Tier 2+).
    /// Enforced by checking that `parent_attestation_hash.is_some()`.
    pub require_chain: bool,
    /// §8.2: require the attestation to carry Tier-3 causal validation
    /// (non-empty `causal_scores` with every record's `is_causal` true).
    pub require_causal_validation: bool,
}

impl GovernanceThresholds {
    /// A permissive default equivalent to the pre-§8.2 PoC behaviour:
    /// only the flat per-agent drift bound is enforced, no probe
    /// threshold, no causal gate, no tier requirement.
    pub fn permissive(max_drift: f32) -> Self {
        Self {
            max_drift,
            min_confidence: 0.0,
            min_causal_score: None,
            require_chain: false,
            require_causal_validation: false,
        }
    }
}

/// A governance table entry: which peer domain this policy applies to,
/// and the thresholds to enforce.
#[derive(Debug, Clone)]
pub struct GovernanceEntry {
    pub pattern: DomainPattern,
    pub thresholds: GovernanceThresholds,
}

/// Per-agent governance table.  Empty = fall back to flat per-agent
/// defaults in the verifier.
#[derive(Debug, Clone, Default)]
pub struct GovernanceTable {
    pub entries: Vec<GovernanceEntry>,
}

impl GovernanceTable {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Validate that the table has no exact-duplicate patterns.  Two
    /// entries with the same canonical pattern are ambiguous (which
    /// thresholds win?) and almost always a config typo.  Overlapping
    /// patterns with *different* specificity are fine — the most
    /// specific one wins in `lookup`.
    pub fn validate(&self, agent_id: &str) -> Result<(), WireError> {
        let mut seen = HashSet::with_capacity(self.entries.len());
        for entry in &self.entries {
            if !seen.insert(&entry.pattern) {
                return Err(WireError::DomainScopeInvalid(format!(
                    "agent {agent_id}: duplicate governance_thresholds pattern {:?}",
                    entry.pattern.canonical()
                )));
            }
        }
        Ok(())
    }

    /// Find the most-specific policy that applies to `peer_domain`.
    /// Exact patterns beat wildcards; longer wildcards beat shorter ones.
    pub fn lookup(&self, peer_domain: &Domain) -> Option<&GovernanceThresholds> {
        let mut best: Option<(usize, &GovernanceThresholds)> = None;
        for entry in &self.entries {
            if entry.pattern.matches(peer_domain) {
                let s = entry.pattern.specificity();
                if best.map(|(b, _)| s > b).unwrap_or(true) {
                    best = Some((s, &entry.thresholds));
                }
            }
        }
        best.map(|(_, t)| t)
    }
}

// ---------------------------------------------------------------------------
// TOML deserialisation helpers
// ---------------------------------------------------------------------------

/// TOML row for a single governance entry.  Used by the registry loader.
#[derive(Debug, Deserialize)]
pub struct GovernanceEntryToml {
    pub pattern: String,
    pub max_drift: f32,
    #[serde(default)]
    pub min_confidence: f32,
    #[serde(default)]
    pub min_causal_score: Option<f32>,
    #[serde(default)]
    pub require_chain: bool,
    #[serde(default)]
    pub require_causal_validation: bool,
}

impl GovernanceEntryToml {
    pub fn into_entry(self, agent_id: &str) -> Result<GovernanceEntry, WireError> {
        let pattern = DomainPattern::parse(&self.pattern).map_err(|e| {
            WireError::RegistryParse(format!("agent {agent_id}: governance pattern: {e}"))
        })?;
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(WireError::RegistryParse(format!(
                "agent {agent_id}: min_confidence must be in [0,1], got {}",
                self.min_confidence
            )));
        }
        if self.max_drift.is_nan() || self.max_drift < 0.0 {
            return Err(WireError::RegistryParse(format!(
                "agent {agent_id}: max_drift must be non-negative, got {}",
                self.max_drift
            )));
        }
        if let Some(c) = self.min_causal_score {
            if !(0.0..=1.0).contains(&c) {
                return Err(WireError::RegistryParse(format!(
                    "agent {agent_id}: min_causal_score must be in [0,1], got {c}"
                )));
            }
        }
        Ok(GovernanceEntry {
            pattern,
            thresholds: GovernanceThresholds {
                max_drift: self.max_drift,
                min_confidence: self.min_confidence,
                min_causal_score: self.min_causal_score,
                require_chain: self.require_chain,
                require_causal_validation: self.require_causal_validation,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Domain;

    fn t(max_drift: f32, require_causal: bool) -> GovernanceThresholds {
        GovernanceThresholds {
            max_drift,
            min_confidence: 0.0,
            min_causal_score: None,
            require_chain: false,
            require_causal_validation: require_causal,
        }
    }

    #[test]
    fn lookup_prefers_exact_over_wildcard() {
        let table = GovernanceTable {
            entries: vec![
                GovernanceEntry {
                    pattern: DomainPattern::parse("healthcare.*").unwrap(),
                    thresholds: t(0.10, false),
                },
                GovernanceEntry {
                    pattern: DomainPattern::parse("healthcare.drug-interaction").unwrap(),
                    thresholds: t(0.02, true),
                },
            ],
        };
        let exact = table
            .lookup(&Domain::parse("healthcare.drug-interaction").unwrap())
            .unwrap();
        assert_eq!(exact.max_drift, 0.02);
        assert!(exact.require_causal_validation);

        let fallback = table.lookup(&Domain::parse("healthcare.other").unwrap()).unwrap();
        assert_eq!(fallback.max_drift, 0.10);
        assert!(!fallback.require_causal_validation);
    }

    #[test]
    fn empty_table_returns_none() {
        let table = GovernanceTable::default();
        assert!(table.lookup(&Domain::parse("agriculture").unwrap()).is_none());
    }

    #[test]
    fn toml_row_rejects_out_of_range_confidence() {
        let row = GovernanceEntryToml {
            pattern: "healthcare.*".to_string(),
            max_drift: 0.05,
            min_confidence: 1.5,
            min_causal_score: None,
            require_chain: false,
            require_causal_validation: false,
        };
        assert!(row.into_entry("alice").is_err());
    }

    #[test]
    fn validate_accepts_non_duplicate_patterns() {
        let table = GovernanceTable {
            entries: vec![
                GovernanceEntry {
                    pattern: DomainPattern::parse("healthcare.*").unwrap(),
                    thresholds: t(0.10, false),
                },
                GovernanceEntry {
                    pattern: DomainPattern::parse("healthcare.drug-interaction").unwrap(),
                    thresholds: t(0.02, true),
                },
            ],
        };
        table.validate("alice").unwrap();
    }

    #[test]
    fn validate_rejects_duplicate_governance_patterns() {
        let table = GovernanceTable {
            entries: vec![
                GovernanceEntry {
                    pattern: DomainPattern::parse("healthcare.*").unwrap(),
                    thresholds: t(0.10, false),
                },
                GovernanceEntry {
                    pattern: DomainPattern::parse("healthcare.*").unwrap(),
                    thresholds: t(0.02, true),
                },
            ],
        };
        let err = table.validate("alice").unwrap_err();
        assert!(
            matches!(err, WireError::DomainScopeInvalid(ref m) if m.contains("duplicate") && m.contains("healthcare.*")),
            "{err:?}"
        );
    }

    #[test]
    fn toml_row_rejects_negative_drift() {
        let row = GovernanceEntryToml {
            pattern: "healthcare.*".to_string(),
            max_drift: -0.01,
            min_confidence: 0.0,
            min_causal_score: None,
            require_chain: false,
            require_causal_validation: false,
        };
        assert!(row.into_entry("alice").is_err());
    }
}
