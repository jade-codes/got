// ---------------------------------------------------------------------------
// Domain Scoping — Protocol §4 / Appendix B.
//
// Registry-side declarative scope that prevents cross-domain attestation
// exchanges between agents whose value geometries are incommensurable.
// The check runs at Phase 0 — before any cryptographic or geometric
// verification — and is a structural property that cannot be overridden
// by high probe readings or governance dispensation.
// ---------------------------------------------------------------------------

use serde::Deserialize;

use crate::WireError;

/// A dot-separated domain namespace, e.g. "agriculture.crop-management".
///
/// Permitted characters: lowercase ASCII letters, digits, `-`, and `.`
/// as a separator. No leading/trailing dots and no empty segments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Domain(String);

impl Domain {
    pub fn parse(s: &str) -> Result<Self, WireError> {
        if s.is_empty() {
            return Err(WireError::DomainParse("empty domain".into()));
        }
        for ch in s.chars() {
            if !(ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '.' || ch == '-') {
                return Err(WireError::DomainParse(format!(
                    "invalid character {ch:?} in domain {s:?}"
                )));
            }
        }
        if s.starts_with('.') || s.ends_with('.') || s.contains("..") {
            return Err(WireError::DomainParse(format!("malformed domain {s:?}")));
        }
        Ok(Domain(s.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A pattern matching one or more domains.
///
/// Either an exact domain (`agriculture.crop-management`), a sub-tree
/// wildcard (`agriculture.*`), or the global wildcard (`*`). Wildcards
/// are only legal as a trailing `.*` (or the bare `*`).
///
/// `agriculture.*` matches `agriculture`, `agriculture.crop-management`,
/// `agriculture.supply-chain`, etc., but NOT `transport.*` or any
/// unrelated top-level domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DomainPattern {
    prefix: String, // empty for the global wildcard
    wildcard: bool,
}

impl DomainPattern {
    pub fn parse(s: &str) -> Result<Self, WireError> {
        if s == "*" {
            return Ok(DomainPattern {
                prefix: String::new(),
                wildcard: true,
            });
        }
        if let Some(stem) = s.strip_suffix(".*") {
            // Validate the stem as a domain.
            let _ = Domain::parse(stem)?;
            return Ok(DomainPattern {
                prefix: stem.to_string(),
                wildcard: true,
            });
        }
        if s.contains('*') {
            return Err(WireError::DomainParse(format!(
                "wildcard only allowed as trailing .* in {s:?}"
            )));
        }
        let _ = Domain::parse(s)?;
        Ok(DomainPattern {
            prefix: s.to_string(),
            wildcard: false,
        })
    }

    /// Test whether this pattern matches a concrete domain.
    pub fn matches(&self, d: &Domain) -> bool {
        let s = d.as_str();
        if self.wildcard {
            if self.prefix.is_empty() {
                return true;
            }
            s == self.prefix || s.starts_with(&format!("{}.", self.prefix))
        } else {
            s == self.prefix
        }
    }

    /// Specificity score used to break ties when several patterns match.
    /// Exact patterns dominate wildcards; longer prefixes dominate shorter.
    fn specificity(&self) -> usize {
        let base = self.prefix.len();
        if self.wildcard {
            base
        } else {
            base + 1_000_000
        }
    }
}

/// Interaction modes (§4.2).
///
/// `ReadOnly`   — receive information only.
/// `Advisory`   — provide non-binding recommendations.
/// `Cooperative` — joint decision-making.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InteractionMode {
    ReadOnly,
    Advisory,
    Cooperative,
}

/// A permitted-domain entry: which pattern is allowed and at what mode.
#[derive(Debug, Clone)]
pub struct PermittedDomain {
    pub pattern: DomainPattern,
    pub mode: InteractionMode,
}

/// Domain scope declaration for an agent (§4.2).
#[derive(Debug, Clone)]
pub struct DomainScope {
    pub primary: Domain,
    pub permitted: Vec<PermittedDomain>,
    pub exclusions: Vec<DomainPattern>,
}

impl DomainScope {
    /// Find the interaction mode this scope grants for `target`, if any.
    /// Most-specific matching pattern wins.
    pub fn mode_for(&self, target: &Domain) -> Option<InteractionMode> {
        let mut best: Option<(usize, InteractionMode)> = None;
        for p in &self.permitted {
            if p.pattern.matches(target) {
                let s = p.pattern.specificity();
                if best.map(|(b, _)| s > b).unwrap_or(true) {
                    best = Some((s, p.mode));
                }
            }
        }
        best.map(|(_, m)| m)
    }

    pub fn is_excluded(&self, target: &Domain) -> bool {
        self.exclusions.iter().any(|p| p.matches(target))
    }
}

/// Are two declared modes compatible?
///
/// Both ReadOnly is the only structurally empty intersection: neither
/// side is willing to transmit, so no exchange can take place.  Mixed
/// modes (the asymmetric pattern in §5.2) are deliberately allowed.
fn modes_compatible(a: InteractionMode, b: InteractionMode) -> bool {
    !(a == InteractionMode::ReadOnly && b == InteractionMode::ReadOnly)
}

/// Bidirectional domain compatibility check (Appendix B).
///
/// Runs *before* any cryptographic verification.  Returns the structural
/// reason for refusal as a `WireError` if any of the four checks fail.
pub fn check_domain_compatibility(
    a: &DomainScope,
    b: &DomainScope,
) -> Result<(), WireError> {
    // 1. Exclusion lists (hard veto, either direction).
    if a.is_excluded(&b.primary) {
        return Err(WireError::DomainExcluded {
            excluder: a.primary.as_str().to_string(),
            target: b.primary.as_str().to_string(),
        });
    }
    if b.is_excluded(&a.primary) {
        return Err(WireError::DomainExcluded {
            excluder: b.primary.as_str().to_string(),
            target: a.primary.as_str().to_string(),
        });
    }

    // 2. Bidirectional permission.
    let a_mode = a.mode_for(&b.primary).ok_or_else(|| WireError::DomainNotPermitted {
        from: a.primary.as_str().to_string(),
        target: b.primary.as_str().to_string(),
    })?;
    let b_mode = b.mode_for(&a.primary).ok_or_else(|| WireError::DomainNotPermitted {
        from: b.primary.as_str().to_string(),
        target: a.primary.as_str().to_string(),
    })?;

    // 3. Interaction mode intersection.
    if !modes_compatible(a_mode, b_mode) {
        return Err(WireError::DomainModeIncompatible {
            a: format!("{a_mode:?}"),
            b: format!("{b_mode:?}"),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Domain {
        Domain::parse(s).unwrap()
    }
    fn p(s: &str) -> DomainPattern {
        DomainPattern::parse(s).unwrap()
    }

    #[test]
    fn domain_parse_accepts_valid_names() {
        Domain::parse("agriculture").unwrap();
        Domain::parse("agriculture.crop-management").unwrap();
        Domain::parse("healthcare.diagnostic-advisory").unwrap();
    }

    #[test]
    fn domain_parse_rejects_garbage() {
        assert!(Domain::parse("").is_err());
        assert!(Domain::parse(".bad").is_err());
        assert!(Domain::parse("bad.").is_err());
        assert!(Domain::parse("a..b").is_err());
        assert!(Domain::parse("Bad.Case").is_err());
        assert!(Domain::parse("has space").is_err());
        assert!(Domain::parse("has/slash").is_err());
    }

    #[test]
    fn pattern_matches_exact() {
        assert!(p("agriculture").matches(&d("agriculture")));
        assert!(!p("agriculture").matches(&d("agriculture.crop-management")));
    }

    #[test]
    fn pattern_matches_wildcard_subtree() {
        let pat = p("agriculture.*");
        assert!(pat.matches(&d("agriculture")));
        assert!(pat.matches(&d("agriculture.crop-management")));
        assert!(pat.matches(&d("agriculture.supply-chain")));
        assert!(!pat.matches(&d("transport.autonomous-vehicle")));
        // Substring guard: "agriculture-x" must NOT match "agriculture.*".
        assert!(!pat.matches(&d("agriculture-x")));
    }

    #[test]
    fn pattern_global_wildcard() {
        let pat = p("*");
        assert!(pat.matches(&d("anything")));
        assert!(pat.matches(&d("transport.autonomous-vehicle")));
    }

    #[test]
    fn pattern_rejects_internal_wildcard() {
        assert!(DomainPattern::parse("agri*ulture").is_err());
        assert!(DomainPattern::parse("*.agriculture").is_err());
    }

    fn agri_scope() -> DomainScope {
        DomainScope {
            primary: d("agriculture.crop-management"),
            permitted: vec![
                PermittedDomain {
                    pattern: p("agriculture.*"),
                    mode: InteractionMode::Cooperative,
                },
                PermittedDomain {
                    pattern: p("meteorology.*"),
                    mode: InteractionMode::Advisory,
                },
            ],
            exclusions: vec![p("transport.*")],
        }
    }

    fn vehicle_scope() -> DomainScope {
        DomainScope {
            primary: d("transport.autonomous-vehicle"),
            permitted: vec![
                PermittedDomain {
                    pattern: p("transport.*"),
                    mode: InteractionMode::Cooperative,
                },
                PermittedDomain {
                    pattern: p("infrastructure.traffic-management"),
                    mode: InteractionMode::Cooperative,
                },
            ],
            exclusions: vec![],
        }
    }

    #[test]
    fn use_case_5_1_agri_vs_vehicle_rejected_by_exclusion() {
        let err = check_domain_compatibility(&agri_scope(), &vehicle_scope()).unwrap_err();
        match err {
            WireError::DomainExcluded { excluder, target } => {
                assert_eq!(excluder, "agriculture.crop-management");
                assert_eq!(target, "transport.autonomous-vehicle");
            }
            other => panic!("expected DomainExcluded, got {other:?}"),
        }
    }

    #[test]
    fn use_case_5_1_rejected_even_without_exclusion() {
        // Strip the explicit exclusion: should still fail because the
        // vehicle agent is not in the agri agent's permitted patterns.
        let mut a = agri_scope();
        a.exclusions.clear();
        let err = check_domain_compatibility(&a, &vehicle_scope()).unwrap_err();
        assert!(matches!(err, WireError::DomainNotPermitted { .. }));
    }

    fn diag_scope() -> DomainScope {
        DomainScope {
            primary: d("healthcare.diagnostic-advisory"),
            permitted: vec![PermittedDomain {
                pattern: p("healthcare.drug-interaction"),
                mode: InteractionMode::Advisory,
            }],
            exclusions: vec![],
        }
    }

    fn drug_scope() -> DomainScope {
        DomainScope {
            primary: d("healthcare.drug-interaction"),
            permitted: vec![PermittedDomain {
                pattern: p("healthcare.diagnostic-advisory"),
                mode: InteractionMode::ReadOnly,
            }],
            exclusions: vec![],
        }
    }

    #[test]
    fn use_case_5_2_diag_drug_permitted_asymmetric() {
        check_domain_compatibility(&diag_scope(), &drug_scope()).unwrap();
        check_domain_compatibility(&drug_scope(), &diag_scope()).unwrap();
    }

    #[test]
    fn both_read_only_is_incompatible() {
        let mut a = diag_scope();
        a.permitted[0].mode = InteractionMode::ReadOnly;
        let err = check_domain_compatibility(&a, &drug_scope()).unwrap_err();
        assert!(matches!(err, WireError::DomainModeIncompatible { .. }));
    }

    #[test]
    fn most_specific_pattern_wins_for_mode_lookup() {
        let scope = DomainScope {
            primary: d("a"),
            permitted: vec![
                PermittedDomain {
                    pattern: p("healthcare.*"),
                    mode: InteractionMode::ReadOnly,
                },
                PermittedDomain {
                    pattern: p("healthcare.drug-interaction"),
                    mode: InteractionMode::Cooperative,
                },
            ],
            exclusions: vec![],
        };
        assert_eq!(
            scope.mode_for(&d("healthcare.drug-interaction")),
            Some(InteractionMode::Cooperative)
        );
        assert_eq!(
            scope.mode_for(&d("healthcare.diagnostic-advisory")),
            Some(InteractionMode::ReadOnly)
        );
    }
}
