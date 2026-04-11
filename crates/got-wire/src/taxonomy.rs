// ---------------------------------------------------------------------------
// Domain taxonomy — Protocol §14.4 / §4.4 / §10.3.
//
// The protocol's domain hierarchy is illustrative in the paper — the
// actual canonical taxonomy is a *governance* artefact that the
// protocol cannot and should not pick.  This module provides the
// machinery a governance body would use to publish, distribute, and
// consult a taxonomy: a TOML format, a parser, hierarchy queries, and
// an opt-in registry validator hook that warns when a registry uses
// domains that the loaded taxonomy does not recognise.
//
// Crucially, the validator returns *warnings*, not errors.  A
// deployment is free to use domain names that are not in the loaded
// taxonomy — operators sometimes need to ship faster than governance
// can ratify a new domain.  The warning surfaces the divergence so
// review can happen, without blocking the registry from loading.
//
// The reference taxonomy file (`taxonomies/got-reference-v1.toml`)
// captures the paper's §4.4 illustrative hierarchy plus the
// dual-purpose `vehicle.*` subtree from the §4.5 worked example.
// Production deployments fork the reference taxonomy or write their
// own; the protocol just needs *some* taxonomy to validate against.
// ---------------------------------------------------------------------------

use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;

use crate::domain::Domain;
use crate::registry::TrustRegistry;
use crate::WireError;

/// One entry in a taxonomy: a single concrete domain with metadata
/// and suggested governance defaults that a registry author can copy.
#[derive(Debug, Clone)]
pub struct DomainEntry {
    pub description: String,
    pub examples: Vec<String>,
    pub suggested_max_drift: Option<f32>,
    pub suggested_min_confidence: Option<f32>,
    pub suggested_min_causal_score: Option<f32>,
    pub suggested_require_chain: bool,
    pub suggested_require_causal_validation: bool,
}

/// A taxonomy: a curated set of `Domain → DomainEntry` mappings plus
/// publication metadata.  Loaded from TOML, immutable once built.
#[derive(Debug, Clone)]
pub struct Taxonomy {
    pub name: String,
    pub version: String,
    pub maintainer: Option<String>,
    pub last_updated: Option<String>,
    /// Domains in canonical sort order.  `BTreeMap` so iteration and
    /// hashing are deterministic for taxonomy fingerprinting.
    pub domains: BTreeMap<Domain, DomainEntry>,
}

impl Taxonomy {
    /// Build an empty taxonomy with just publication metadata.
    pub fn empty(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            maintainer: None,
            last_updated: None,
            domains: BTreeMap::new(),
        }
    }

    /// Parse a taxonomy from TOML.
    pub fn from_toml(s: &str) -> Result<Self, WireError> {
        let parsed: TaxonomyFile = toml::from_str(s)
            .map_err(|e| WireError::DomainParse(format!("taxonomy parse: {e}")))?;

        let header = parsed.taxonomy.unwrap_or_default();
        let mut domains = BTreeMap::new();

        for entry in parsed.domain.unwrap_or_default() {
            let d = Domain::parse(&entry.name)?;
            if let Some(c) = entry.suggested_min_confidence {
                if !(0.0..=1.0).contains(&c) {
                    return Err(WireError::DomainParse(format!(
                        "taxonomy {}: suggested_min_confidence must be in [0,1], got {c}",
                        entry.name
                    )));
                }
            }
            if let Some(c) = entry.suggested_min_causal_score {
                if !(0.0..=1.0).contains(&c) {
                    return Err(WireError::DomainParse(format!(
                        "taxonomy {}: suggested_min_causal_score must be in [0,1], got {c}",
                        entry.name
                    )));
                }
            }
            if let Some(d_) = entry.suggested_max_drift {
                if d_.is_nan() || d_ < 0.0 {
                    return Err(WireError::DomainParse(format!(
                        "taxonomy {}: suggested_max_drift must be non-negative, got {d_}",
                        entry.name
                    )));
                }
            }
            if domains.contains_key(&d) {
                return Err(WireError::DomainParse(format!(
                    "taxonomy: duplicate domain entry {}",
                    entry.name
                )));
            }
            domains.insert(
                d,
                DomainEntry {
                    description: entry.description,
                    examples: entry.examples.unwrap_or_default(),
                    suggested_max_drift: entry.suggested_max_drift,
                    suggested_min_confidence: entry.suggested_min_confidence,
                    suggested_min_causal_score: entry.suggested_min_causal_score,
                    suggested_require_chain: entry.suggested_require_chain,
                    suggested_require_causal_validation: entry.suggested_require_causal_validation,
                },
            );
        }

        Ok(Self {
            name: header.name.unwrap_or_else(|| "unnamed".into()),
            version: header.version.unwrap_or_else(|| "0".into()),
            maintainer: header.maintainer,
            last_updated: header.last_updated,
            domains,
        })
    }

    /// Load a taxonomy from a TOML file on disk.
    pub fn load(path: &std::path::Path) -> Result<Self, WireError> {
        let contents = std::fs::read_to_string(path).map_err(|e| WireError::Io(e.to_string()))?;
        Self::from_toml(&contents)
    }

    /// Look up a domain in the taxonomy.  Returns `None` if the domain
    /// is not registered.
    pub fn lookup(&self, domain: &Domain) -> Option<&DomainEntry> {
        self.domains.get(domain)
    }

    /// True if the taxonomy contains an entry for `domain`.
    pub fn contains(&self, domain: &Domain) -> bool {
        self.domains.contains_key(domain)
    }

    /// Compute the parent domain by stripping the last dot-separated
    /// segment.  `agriculture.crop-management` → `Some("agriculture")`,
    /// top-level domain → `None`.  Does not check whether the parent
    /// is itself in the taxonomy — use [`Taxonomy::contains`] for that.
    pub fn parent_of(&self, domain: &Domain) -> Option<Domain> {
        let s = domain.as_str();
        let last_dot = s.rfind('.')?;
        Domain::parse(&s[..last_dot]).ok()
    }

    /// All taxonomy entries whose canonical form is `prefix` or starts
    /// with `prefix.`.  Returned in sorted order.
    pub fn descendants_of(&self, prefix: &Domain) -> Vec<&Domain> {
        let prefix_str = prefix.as_str();
        let dotted = format!("{prefix_str}.");
        self.domains
            .keys()
            .filter(|d| d.as_str() == prefix_str || d.as_str().starts_with(&dotted))
            .collect()
    }

    /// Validate a registry against this taxonomy.  Returns one warning
    /// per agent whose `primary_domain` is not in the taxonomy.  Empty
    /// `Vec` means the registry is fully consistent with the taxonomy.
    ///
    /// This is a *warning* path, not an error path: registries are
    /// allowed to use domains not in the taxonomy, but the divergence
    /// is surfaced for review.
    pub fn validate_registry(&self, registry: &TrustRegistry) -> Vec<TaxonomyWarning> {
        let mut warnings = Vec::new();
        let mut seen = HashSet::new();
        for entry in registry.agents.values() {
            if let Some(scope) = entry.domain_scope.as_ref() {
                if !self.contains(&scope.primary) && seen.insert(scope.primary.clone()) {
                    warnings.push(TaxonomyWarning::UnknownPrimaryDomain {
                        agent_name: entry.name.clone(),
                        domain: scope.primary.as_str().to_string(),
                    });
                }
            }
        }
        warnings
    }
}

/// A non-fatal divergence between a taxonomy and a registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaxonomyWarning {
    /// An agent's `primary_domain` is not registered in the taxonomy.
    UnknownPrimaryDomain { agent_name: String, domain: String },
}

impl std::fmt::Display for TaxonomyWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaxonomyWarning::UnknownPrimaryDomain { agent_name, domain } => {
                write!(
                    f,
                    "agent {agent_name:?} declares primary domain {domain:?} which is not in the loaded taxonomy"
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TOML deserialisation
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct TaxonomyFile {
    taxonomy: Option<TaxonomyHeader>,
    #[serde(rename = "domain")]
    domain: Option<Vec<DomainEntryToml>>,
}

#[derive(Debug, Default, Deserialize)]
struct TaxonomyHeader {
    name: Option<String>,
    version: Option<String>,
    maintainer: Option<String>,
    last_updated: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DomainEntryToml {
    name: String,
    description: String,
    #[serde(default)]
    examples: Option<Vec<String>>,
    #[serde(default)]
    suggested_max_drift: Option<f32>,
    #[serde(default)]
    suggested_min_confidence: Option<f32>,
    #[serde(default)]
    suggested_min_causal_score: Option<f32>,
    #[serde(default)]
    suggested_require_chain: bool,
    #[serde(default)]
    suggested_require_causal_validation: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{DomainPattern, DomainScope, InteractionMode, PermittedDomain};
    use crate::registry::AgentEntry;
    use ed25519_dalek::SigningKey;

    fn d(s: &str) -> Domain {
        Domain::parse(s).unwrap()
    }

    fn sample_taxonomy() -> Taxonomy {
        let toml = r#"
[taxonomy]
name = "test-taxonomy"
version = "1.0"
maintainer = "Test Suite"

[[domain]]
name = "agriculture"
description = "Agricultural operations"

[[domain]]
name = "agriculture.crop-management"
description = "Crop management"
suggested_max_drift = 0.10
suggested_require_chain = false

[[domain]]
name = "vehicle.agricultural-tractor"
description = "Self-driving agricultural machinery"
suggested_max_drift = 0.02
suggested_require_chain = true
suggested_require_causal_validation = true

[[domain]]
name = "healthcare.diagnostic-advisory"
description = "Diagnostic advisory"
suggested_max_drift = 0.03
suggested_min_causal_score = 0.85
suggested_require_causal_validation = true
"#;
        Taxonomy::from_toml(toml).unwrap()
    }

    #[test]
    fn parses_taxonomy_with_metadata() {
        let t = sample_taxonomy();
        assert_eq!(t.name, "test-taxonomy");
        assert_eq!(t.version, "1.0");
        assert_eq!(t.maintainer, Some("Test Suite".to_string()));
        assert_eq!(t.domains.len(), 4);
    }

    #[test]
    fn lookup_returns_entry() {
        let t = sample_taxonomy();
        let e = t.lookup(&d("vehicle.agricultural-tractor")).unwrap();
        assert!(e.description.contains("agricultural"));
        assert_eq!(e.suggested_max_drift, Some(0.02));
        assert!(e.suggested_require_chain);
        assert!(e.suggested_require_causal_validation);
    }

    #[test]
    fn lookup_returns_none_for_unknown() {
        let t = sample_taxonomy();
        assert!(t.lookup(&d("transport.autonomous-vehicle")).is_none());
    }

    #[test]
    fn parent_of_strips_last_segment() {
        let t = sample_taxonomy();
        assert_eq!(
            t.parent_of(&d("agriculture.crop-management"))
                .map(|d| d.as_str().to_string()),
            Some("agriculture".into())
        );
        assert_eq!(t.parent_of(&d("agriculture")), None);
    }

    #[test]
    fn descendants_of_returns_subtree() {
        let t = sample_taxonomy();
        let descendants = t.descendants_of(&d("agriculture"));
        assert_eq!(descendants.len(), 2);
        let names: Vec<&str> = descendants.iter().map(|d| d.as_str()).collect();
        assert!(names.contains(&"agriculture"));
        assert!(names.contains(&"agriculture.crop-management"));
    }

    #[test]
    fn descendants_of_does_not_match_substring_prefix() {
        let toml = r#"
[[domain]]
name = "agri"
description = "agri"

[[domain]]
name = "agri-x"
description = "unrelated"

[[domain]]
name = "agri.subtree"
description = "real child"
"#;
        let t = Taxonomy::from_toml(toml).unwrap();
        let descendants = t.descendants_of(&d("agri"));
        let names: Vec<&str> = descendants.iter().map(|d| d.as_str()).collect();
        assert!(names.contains(&"agri"));
        assert!(names.contains(&"agri.subtree"));
        assert!(!names.contains(&"agri-x")); // critical: substring guard
    }

    #[test]
    fn rejects_duplicate_entries() {
        let toml = r#"
[[domain]]
name = "x"
description = "first"

[[domain]]
name = "x"
description = "second"
"#;
        assert!(Taxonomy::from_toml(toml).is_err());
    }

    #[test]
    fn rejects_out_of_range_min_confidence() {
        let toml = r#"
[[domain]]
name = "x"
description = "test"
suggested_min_confidence = 1.5
"#;
        assert!(Taxonomy::from_toml(toml).is_err());
    }

    #[test]
    fn rejects_negative_max_drift() {
        let toml = r#"
[[domain]]
name = "x"
description = "test"
suggested_max_drift = -0.1
"#;
        assert!(Taxonomy::from_toml(toml).is_err());
    }

    fn registry_with_scope(name: &str, primary: &str) -> TrustRegistry {
        let key = SigningKey::from_bytes(&[0xAA; 32]);
        let pk = key.verifying_key();
        let mut r = TrustRegistry::empty();
        r.add_agent(AgentEntry {
            name: name.into(),
            public_key: pk,
            agent_id: crate::registry::compute_agent_id(&pk),
            max_drift_accepted: 0.05,
            roles: vec![],
            expected_model_hash: None,
            certificate: None,
            domain_scope: Some(DomainScope {
                primary: Domain::parse(primary).unwrap(),
                permitted: vec![PermittedDomain {
                    pattern: DomainPattern::parse("agriculture.*").unwrap(),
                    mode: InteractionMode::Cooperative,
                }],
                exclusions: vec![],
            }),
            governance_table: crate::governance::GovernanceTable::default(),
        });
        r
    }

    #[test]
    fn validate_registry_clean_passes() {
        let t = sample_taxonomy();
        let r = registry_with_scope("alice", "agriculture.crop-management");
        let warnings = t.validate_registry(&r);
        assert!(warnings.is_empty());
    }

    #[test]
    fn validate_registry_warns_on_unknown_domain() {
        let t = sample_taxonomy();
        let r = registry_with_scope("alice", "transport.autonomous-vehicle");
        let warnings = t.validate_registry(&r);
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            TaxonomyWarning::UnknownPrimaryDomain { agent_name, domain } => {
                assert_eq!(agent_name, "alice");
                assert_eq!(domain, "transport.autonomous-vehicle");
            }
        }
    }

    /// Smoke test against the actual `taxonomies/got-reference-v1.toml`
    /// file shipped in the repo.  Catches regressions where someone edits
    /// the reference taxonomy in a way that breaks the parser.
    #[test]
    fn loads_reference_taxonomy_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("taxonomies")
            .join("got-reference-v1.toml");
        if !path.exists() {
            // The reference taxonomy lives at the workspace root; skip
            // gracefully if this test crate is being run from a sandbox
            // that does not include the file.
            return;
        }
        let t = Taxonomy::load(&path).expect("reference taxonomy must parse");
        assert_eq!(t.name, "GoT Reference Taxonomy");
        assert!(
            t.domains.contains_key(&Domain::parse("vehicle.agricultural-tractor").unwrap()),
            "reference taxonomy must include the dual-purpose tractor"
        );
        let tractor = t.lookup(&Domain::parse("vehicle.agricultural-tractor").unwrap()).unwrap();
        assert_eq!(tractor.suggested_max_drift, Some(0.02));
        assert!(tractor.suggested_require_chain);
        assert!(tractor.suggested_require_causal_validation);
    }

    #[test]
    fn validate_registry_skips_unscoped_agents() {
        let t = sample_taxonomy();
        let mut r = TrustRegistry::empty();
        let key = SigningKey::from_bytes(&[0xBB; 32]);
        let pk = key.verifying_key();
        r.add_agent(AgentEntry {
            name: "bob".into(),
            public_key: pk,
            agent_id: crate::registry::compute_agent_id(&pk),
            max_drift_accepted: 0.05,
            roles: vec![],
            expected_model_hash: None,
            certificate: None,
            domain_scope: None, // unscoped
            governance_table: crate::governance::GovernanceTable::default(),
        });
        let warnings = t.validate_registry(&r);
        assert!(warnings.is_empty());
    }
}
