// ---------------------------------------------------------------------------
// Domain Scoping — Protocol §4 / Appendix B.
//
// Registry-side declarative scope that prevents cross-domain attestation
// exchanges between agents whose value geometries are incommensurable.
// The check runs at Phase 0 — before any cryptographic or geometric
// verification — and is a structural property that cannot be overridden
// by high probe readings or governance dispensation.
// ---------------------------------------------------------------------------

use std::collections::HashSet;

use serde::Deserialize;

use got_core::{DomainScopeDeclaration, InteractionModeTag, PermittedDomainDeclaration};

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

    /// Does `self` (treated as an exclusion pattern) subsume `other`
    /// (treated as a permission pattern)?  That is: is every domain
    /// matched by `other` also matched by `self`?
    ///
    /// Used by `DomainScope::validate()` to detect dead permissions,
    /// e.g. `permit transport.*` alongside `exclude transport.*`, or
    /// `permit transport.trucks` alongside `exclude transport.*`.
    /// A narrower exclusion (e.g. `exclude transport.autonomous-vehicle`
    /// against `permit transport.*`) is **not** subsumption — the rest
    /// of the transport sub-tree is still permitted, which is a
    /// legitimate "allow-with-carveout" configuration.
    pub fn subsumes(&self, other: &DomainPattern) -> bool {
        if !self.wildcard {
            // Exact exclusion only subsumes an exact permission of the same value.
            return !other.wildcard && self.prefix == other.prefix;
        }
        if self.prefix.is_empty() {
            // Global wildcard subsumes everything.
            return true;
        }
        // Non-global wildcard `self` = "x.*": subsumes `other` only when every
        // domain other matches is inside the x sub-tree.  A global-wildcard
        // `other` matches domains outside x.* too, so it is never subsumed.
        if other.wildcard && other.prefix.is_empty() {
            return false;
        }
        let other_prefix = &other.prefix;
        other_prefix == &self.prefix
            || other_prefix.starts_with(&format!("{}.", self.prefix))
    }

    /// Canonical string form for serialisation.
    /// Exact: "agriculture.crop-management"; wildcard: "agriculture.*";
    /// global wildcard: "*".
    pub fn canonical(&self) -> String {
        if self.wildcard {
            if self.prefix.is_empty() {
                "*".to_string()
            } else {
                format!("{}.*", self.prefix)
            }
        } else {
            self.prefix.clone()
        }
    }

    /// Specificity score used to break ties when several patterns match.
    /// Exact patterns dominate wildcards; longer prefixes dominate shorter.
    pub fn specificity(&self) -> usize {
        let base = self.prefix.len();
        if self.wildcard {
            base
        } else {
            base + 1_000_000
        }
    }
}

impl InteractionMode {
    /// Convert to the wire-level tag carried in attestations (§2.1).
    pub fn to_tag(self) -> InteractionModeTag {
        match self {
            InteractionMode::ReadOnly => InteractionModeTag::ReadOnly,
            InteractionMode::Advisory => InteractionModeTag::Advisory,
            InteractionMode::Cooperative => InteractionModeTag::Cooperative,
            InteractionMode::Supervised => InteractionModeTag::Supervised,
        }
    }

    pub fn from_tag(tag: InteractionModeTag) -> Self {
        match tag {
            InteractionModeTag::ReadOnly => InteractionMode::ReadOnly,
            InteractionModeTag::Advisory => InteractionMode::Advisory,
            InteractionModeTag::Cooperative => InteractionMode::Cooperative,
            InteractionModeTag::Supervised => InteractionMode::Supervised,
        }
    }
}

/// Interaction modes (§4.2, §5.5).
///
/// `ReadOnly`   — receive information only.
/// `Advisory`   — provide non-binding recommendations.
/// `Cooperative` — joint decision-making.
/// `Supervised` — regulatory oversight with asymmetric disclosure (§5.5):
///     the regulator may demand attestations from the supervised agent
///     without producing one of its own, and the supervised agent must
///     accept the regulator's cooperation refusals without challenge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InteractionMode {
    ReadOnly,
    Advisory,
    Cooperative,
    Supervised,
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
    /// Validate the internal consistency of this scope.  Catches the
    /// configuration mistakes the type system and parser do not:
    ///
    /// 1. **Duplicate permitted patterns** — two entries with the same
    ///    canonical pattern are ambiguous (which mode wins a tie-broken
    ///    lookup?) and almost always a config typo.
    /// 2. **Duplicate exclusion patterns** — redundant and confusing;
    ///    reject to force the author to collapse them.
    /// 3. **Exclusion subsumes permission** — e.g. `permit transport.*`
    ///    alongside `exclude transport.*` (exact match) or
    ///    `permit transport.trucks` alongside `exclude transport.*`
    ///    (the permission is dead code because exclusion runs first in
    ///    `check_domain_compatibility`).  Narrower exclusions that
    ///    carve out part of a broader permission (e.g.
    ///    `permit transport.*` + `exclude transport.autonomous-vehicle`)
    ///    are **not** flagged — they are a legitimate
    ///    "allow-with-carveout" configuration.
    ///
    /// Run this at registry load time (see `parse_domain_scope` in
    /// `got-wire::registry`).  An empty `permitted` list is allowed: a
    /// scope with no permitted peers describes an observer-only agent
    /// that refuses all inbound cooperation.
    pub fn validate(&self) -> Result<(), WireError> {
        // 1. Duplicate permitted patterns — O(n) via HashSet.
        //    DomainPattern derives Hash + Eq so we can hash borrows.
        let mut seen_permit = HashSet::with_capacity(self.permitted.len());
        for entry in &self.permitted {
            if !seen_permit.insert(&entry.pattern) {
                return Err(WireError::DomainScopeInvalid(format!(
                    "duplicate permitted pattern {:?} in scope for {}",
                    entry.pattern.canonical(),
                    self.primary.as_str()
                )));
            }
        }

        // 2. Duplicate exclusion patterns — O(n) via HashSet.
        let mut seen_excl = HashSet::with_capacity(self.exclusions.len());
        for excl in &self.exclusions {
            if !seen_excl.insert(excl) {
                return Err(WireError::DomainScopeInvalid(format!(
                    "duplicate exclusion pattern {:?} in scope for {}",
                    excl.canonical(),
                    self.primary.as_str()
                )));
            }
        }

        // 3. Dead permissions: any exclusion that subsumes a permission.
        //    Subsumption is a structural pairwise relation (not equality),
        //    so hashing doesn't help — this stays O(permits × exclusions).
        //    Scopes are curated by humans and realistically hold a handful
        //    of entries, so O(p × e) is fine.
        for permit in &self.permitted {
            for excl in &self.exclusions {
                if excl.subsumes(&permit.pattern) {
                    return Err(WireError::DomainScopeInvalid(format!(
                        "exclusion {:?} subsumes permitted pattern {:?} in scope for {} \
                         (the permission is dead code because exclusions take precedence)",
                        excl.canonical(),
                        permit.pattern.canonical(),
                        self.primary.as_str()
                    )));
                }
            }
        }

        Ok(())
    }

    /// Serialise this scope into the wire-level declaration that travels
    /// inside a signed attestation (§2.1).  String-based for stability.
    pub fn to_declaration(&self) -> DomainScopeDeclaration {
        DomainScopeDeclaration {
            primary: self.primary.as_str().to_string(),
            permitted: self
                .permitted
                .iter()
                .map(|p| PermittedDomainDeclaration {
                    pattern: p.pattern.canonical(),
                    mode: p.mode.to_tag(),
                })
                .collect(),
            exclusions: self
                .exclusions
                .iter()
                .map(|p| p.canonical())
                .collect(),
        }
    }

    /// Parse a wire-level declaration back into a rich `DomainScope`,
    /// re-validating every string through the domain / pattern parsers.
    pub fn from_declaration(decl: &DomainScopeDeclaration) -> Result<Self, WireError> {
        let primary = Domain::parse(&decl.primary)?;
        let mut permitted = Vec::with_capacity(decl.permitted.len());
        for p in &decl.permitted {
            permitted.push(PermittedDomain {
                pattern: DomainPattern::parse(&p.pattern)?,
                mode: InteractionMode::from_tag(p.mode),
            });
        }
        let mut exclusions = Vec::with_capacity(decl.exclusions.len());
        for e in &decl.exclusions {
            exclusions.push(DomainPattern::parse(e)?);
        }
        Ok(DomainScope {
            primary,
            permitted,
            exclusions,
        })
    }

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
/// modes (the asymmetric patterns in §5.2 and §5.5) are deliberately
/// allowed — including (Supervised, Supervised), which models the
/// regulator ↔ supervised-agent relationship where the regulator
/// demands an attestation without producing one of its own.
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

    // -----------------------------------------------------------------------
    // Validator: DomainPattern::subsumes
    // -----------------------------------------------------------------------

    #[test]
    fn global_wildcard_subsumes_everything() {
        let global = p("*");
        assert!(global.subsumes(&p("transport.*")));
        assert!(global.subsumes(&p("agriculture.crop-management")));
        assert!(global.subsumes(&p("*")));
    }

    #[test]
    fn subtree_wildcard_subsumes_descendants() {
        let trans = p("transport.*");
        assert!(trans.subsumes(&p("transport.*")));
        assert!(trans.subsumes(&p("transport.autonomous-vehicle")));
        assert!(trans.subsumes(&p("transport.logistics.*")));
        assert!(!trans.subsumes(&p("agriculture.*")));
        assert!(!trans.subsumes(&p("transport-adjacent")));
        assert!(!trans.subsumes(&p("*"))); // narrower can't subsume global
    }

    #[test]
    fn exact_pattern_only_subsumes_itself() {
        let exact = p("transport.autonomous-vehicle");
        assert!(exact.subsumes(&p("transport.autonomous-vehicle")));
        assert!(!exact.subsumes(&p("transport.*")));
        assert!(!exact.subsumes(&p("transport.autonomous-vehicle.truck")));
    }

    // -----------------------------------------------------------------------
    // Validator: DomainScope::validate
    // -----------------------------------------------------------------------

    fn scope(
        primary: &str,
        permitted: &[(&str, InteractionMode)],
        exclusions: &[&str],
    ) -> DomainScope {
        DomainScope {
            primary: d(primary),
            permitted: permitted
                .iter()
                .map(|(pat, mode)| PermittedDomain {
                    pattern: p(pat),
                    mode: *mode,
                })
                .collect(),
            exclusions: exclusions.iter().map(|s| p(s)).collect(),
        }
    }

    #[test]
    fn validate_accepts_well_formed_scope() {
        let s = scope(
            "agriculture.crop-management",
            &[
                ("agriculture.*", InteractionMode::Cooperative),
                ("meteorology.*", InteractionMode::Advisory),
            ],
            &["transport.*"],
        );
        s.validate().unwrap();
    }

    #[test]
    fn validate_accepts_empty_permitted_list() {
        // Observer-only agent: refuses all inbound cooperation.
        let s = scope("healthcare.auditor", &[], &[]);
        s.validate().unwrap();
    }

    #[test]
    fn validate_allows_narrower_exclusion_carveout() {
        // Legitimate "allow all transport except autonomous-vehicle".
        let s = scope(
            "transport.dispatcher",
            &[("transport.*", InteractionMode::Cooperative)],
            &["transport.autonomous-vehicle"],
        );
        s.validate().unwrap();
    }

    #[test]
    fn validate_rejects_duplicate_permitted_patterns() {
        let s = scope(
            "agriculture.crop-management",
            &[
                ("agriculture.*", InteractionMode::Cooperative),
                ("agriculture.*", InteractionMode::ReadOnly),
            ],
            &[],
        );
        let err = s.validate().unwrap_err();
        assert!(
            matches!(err, WireError::DomainScopeInvalid(ref m) if m.contains("duplicate permitted")),
            "{err:?}"
        );
    }

    #[test]
    fn validate_rejects_duplicate_exclusion_patterns() {
        let s = scope(
            "agriculture.crop-management",
            &[("agriculture.*", InteractionMode::Cooperative)],
            &["transport.*", "transport.*"],
        );
        let err = s.validate().unwrap_err();
        assert!(
            matches!(err, WireError::DomainScopeInvalid(ref m) if m.contains("duplicate exclusion")),
            "{err:?}"
        );
    }

    #[test]
    fn validate_rejects_exact_exclusion_shadowing_permission() {
        let s = scope(
            "agriculture.crop-management",
            &[("transport.*", InteractionMode::Cooperative)],
            &["transport.*"],
        );
        let err = s.validate().unwrap_err();
        assert!(
            matches!(err, WireError::DomainScopeInvalid(ref m) if m.contains("subsumes")),
            "{err:?}"
        );
    }

    #[test]
    fn validate_rejects_wildcard_exclusion_shadowing_narrower_permission() {
        // permit transport.trucks + exclude transport.* → permission is dead
        let s = scope(
            "agriculture.crop-management",
            &[("transport.trucks", InteractionMode::Cooperative)],
            &["transport.*"],
        );
        let err = s.validate().unwrap_err();
        assert!(
            matches!(err, WireError::DomainScopeInvalid(ref m) if m.contains("subsumes")),
            "{err:?}"
        );
    }

    #[test]
    fn validate_rejects_global_exclusion_with_any_permission() {
        let s = scope(
            "agriculture.crop-management",
            &[("agriculture.*", InteractionMode::Cooperative)],
            &["*"],
        );
        let err = s.validate().unwrap_err();
        assert!(
            matches!(err, WireError::DomainScopeInvalid(ref m) if m.contains("subsumes")),
            "{err:?}"
        );
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
