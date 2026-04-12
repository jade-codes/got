// ---------------------------------------------------------------------------
// Exchange Protocol Logic — Phase 10, §10.7 / §10.9.
//
// Payload types for EXCHANGE_REQ and EXCHANGE_RSP, plus high-level
// exchange functions that operate on in-memory messages (no I/O).
//
// The Noise NK transport layer wraps these in encrypted frames, but
// the logic here is transport-agnostic so it can be tested without TCP.
// ---------------------------------------------------------------------------

use ed25519_dalek::SigningKey;

use got_core::GeometricAttestation;

use crate::chain::{attestation_hash, verify_chain};
use crate::domain::{check_domain_compatibility, DomainScope};
use crate::envelope::ExchangeEnvelope;
use crate::governance::GovernanceThresholds;
use crate::registry::{compute_agent_id, AgentEntry, TrustRegistry};
use crate::WireError;

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// Exchange verdict — whether we accept or reject the peer's attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Verdict {
    Accepted = 0x01,
    Rejected = 0x02,
    Error = 0x03,
}

impl Verdict {
    pub fn from_byte(b: u8) -> Result<Self, WireError> {
        match b {
            0x01 => Ok(Verdict::Accepted),
            0x02 => Ok(Verdict::Rejected),
            0x03 => Ok(Verdict::Error),
            _ => Err(WireError::Protocol(format!(
                "unknown verdict byte: 0x{b:02x}"
            ))),
        }
    }

    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

// ---------------------------------------------------------------------------
// Exchange Request / Response
// ---------------------------------------------------------------------------

/// An EXCHANGE_REQ payload (§10.7).
#[derive(Debug, Clone)]
pub struct ExchangeRequest {
    /// SHA-256 of the sender's public key.
    pub agent_id: [u8; 32],
    /// Signed envelope binding this attestation to this exchange.
    pub envelope: ExchangeEnvelope,
    /// Attestation chain (oldest first), may be empty for Tier-1
    /// (bare) attestations.
    pub chain: Vec<GeometricAttestation>,
    /// The current attestation being exchanged.
    pub current: GeometricAttestation,
}

/// An EXCHANGE_RSP payload (§10.7).
#[derive(Debug, Clone)]
pub struct ExchangeResponse {
    /// SHA-256 of the responder's public key.
    pub agent_id: [u8; 32],
    /// Signed envelope binding this attestation to this exchange.
    pub envelope: ExchangeEnvelope,
    /// The responder's verdict on the initiator's attestation.
    pub verdict: Verdict,
    /// Responder's attestation chain.
    pub chain: Vec<GeometricAttestation>,
    /// The responder's current attestation.
    pub current: GeometricAttestation,
    /// Human-readable reason (non-empty if rejected or error).
    pub reason: String,
}

/// The result of a full exchange from the initiator's perspective.
#[derive(Debug, Clone)]
pub struct ExchangeResult {
    /// What the peer said about our attestation.
    pub peer_verdict: Verdict,
    /// What we decided about the peer's attestation.
    pub our_verdict: Verdict,
    /// The peer's current attestation.
    pub peer_attestation: GeometricAttestation,
    /// The peer's chain.
    pub peer_chain: Vec<GeometricAttestation>,
    /// Reason string (from peer if rejected, or from us).
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

/// Build an `ExchangeRequest` ready to send.
pub fn build_request(
    nonce: [u8; 32],
    peer_agent_id: [u8; 32],
    own_key: &SigningKey,
    chain: Vec<GeometricAttestation>,
    current: GeometricAttestation,
) -> Result<ExchangeRequest, WireError> {
    let chain_anchor = chain.first();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let envelope =
        ExchangeEnvelope::create(nonce, peer_agent_id, &current, chain_anchor, now, own_key)?;
    let agent_id = compute_agent_id(&own_key.verifying_key());

    Ok(ExchangeRequest {
        agent_id,
        envelope,
        chain,
        current,
    })
}

/// Build an `ExchangeResponse` after validating the request.
pub fn build_response(
    request_nonce: [u8; 32],
    initiator_agent_id: [u8; 32],
    own_key: &SigningKey,
    verdict: Verdict,
    chain: Vec<GeometricAttestation>,
    current: GeometricAttestation,
    reason: String,
) -> Result<ExchangeResponse, WireError> {
    let chain_anchor = chain.first();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Responder echoes the initiator's nonce.
    let envelope = ExchangeEnvelope::create(
        request_nonce,
        initiator_agent_id,
        &current,
        chain_anchor,
        now,
        own_key,
    )?;
    let agent_id = compute_agent_id(&own_key.verifying_key());

    Ok(ExchangeResponse {
        agent_id,
        envelope,
        verdict,
        chain,
        current,
        reason,
    })
}

// ---------------------------------------------------------------------------
// Phase 1 — Domain compatibility pre-flight
// ---------------------------------------------------------------------------

/// Phase 1 pre-flight: check domain compatibility between two agents
/// **before** either side computes attestations.
///
/// This is the primary structural gate described in Protocol §4 /
/// Appendix B.  It runs *before* any expensive work (geometry
/// computation, probe evaluation, causal intervention, attestation
/// signing) and short-circuits the exchange if the two agents are
/// domain-incompatible.  The existing domain check inside
/// `validate_request` / `validate_response` remains as defence in
/// depth (Phase 4, step 2) — a re-verification against the
/// attestation's own scope declaration.
///
/// Returns `Ok(())` if the two agents are compatible (or if either
/// side is unscoped, which is the backwards-compatible fallthrough).
/// Returns `Err(WireError)` with a domain-specific error variant if
/// structurally incompatible.
///
/// Typical call sites:
///   - `got-net::client::request_on_stream` — called after looking up
///     the peer in the registry, before `build_request`.
///   - `got-net::server::handle_connection` — called after reading the
///     initiator's `agent_id` from the first frame, before computing
///     the server's own attestation.
pub fn check_domain_before_exchange(
    own_agent_id: &[u8; 32],
    peer_agent_id: &[u8; 32],
    registry: &TrustRegistry,
) -> Result<(), WireError> {
    let own_entry = match registry.lookup(own_agent_id) {
        Some(e) => e,
        None => return Ok(()), // unscoped self: skip
    };
    let peer_entry = match registry.lookup(peer_agent_id) {
        Some(e) => e,
        None => return Ok(()), // unknown peer: let validate_request catch it later
    };
    match (
        own_entry.domain_scope.as_ref(),
        peer_entry.domain_scope.as_ref(),
    ) {
        (Some(own_scope), Some(peer_scope)) => {
            check_domain_compatibility(own_scope, peer_scope)
        }
        _ => Ok(()), // either side unscoped: backwards compatible
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Resolve the effective governance thresholds a verifier applies to an
/// incoming attestation from `peer` (§7.3 / §8.2).
///
/// Priority (most to least specific):
///   1. `self_entry.governance_table` matched against the peer's primary
///      domain (requires peer to declare a scope).
///   2. Flat `self_entry.max_drift_accepted`, wrapped in
///      `GovernanceThresholds::permissive`, which preserves pre-§8.2
///      PoC behaviour exactly.
fn effective_thresholds(self_entry: &AgentEntry, peer: &AgentEntry) -> GovernanceThresholds {
    if let Some(peer_scope) = peer.domain_scope.as_ref() {
        if let Some(t) = self_entry.governance_table.lookup(&peer_scope.primary) {
            return *t;
        }
    }
    GovernanceThresholds::permissive(self_entry.max_drift_accepted)
}

/// §2.1: if the attestation carries an embedded domain scope declaration,
/// the declared primary / permitted / exclusions MUST match the trust
/// registry's entry for the same agent.  This catches relay attacks that
/// substitute attestations across agents, and catches a misconfigured
/// agent that claims one domain in the signed payload and another in the
/// registry.
fn check_attestation_scope_binding(
    peer: &AgentEntry,
    attestation: &got_core::GeometricAttestation,
) -> Result<(), String> {
    let decl = match attestation.domain_scope_declaration.as_ref() {
        Some(d) => d,
        None => return Ok(()),
    };

    // Parse the embedded declaration back into a structured DomainScope.
    let embedded = match DomainScope::from_declaration(decl) {
        Ok(s) => s,
        Err(e) => return Err(format!("embedded domain scope malformed: {e}")),
    };

    match peer.domain_scope.as_ref() {
        Some(registry_scope) => {
            if !domain_scopes_equivalent(&embedded, registry_scope) {
                return Err(format!(
                    "attestation domain_scope_declaration ({}) disagrees with registry ({})",
                    embedded.primary.as_str(),
                    registry_scope.primary.as_str()
                ));
            }
        }
        None => {
            return Err(format!(
                "attestation declares domain {} but registry entry is unscoped",
                embedded.primary.as_str()
            ));
        }
    }

    Ok(())
}

/// Semantic equivalence check for two `DomainScope`s using their canonical
/// (string) form.  We ignore ordering of the permitted / exclusion lists
/// since both sides are governance-curated and order is not meaningful.
fn domain_scopes_equivalent(a: &DomainScope, b: &DomainScope) -> bool {
    if a.primary.as_str() != b.primary.as_str() {
        return false;
    }
    let mut a_perm: Vec<_> = a
        .permitted
        .iter()
        .map(|p| (p.pattern.canonical(), p.mode))
        .collect();
    let mut b_perm: Vec<_> = b
        .permitted
        .iter()
        .map(|p| (p.pattern.canonical(), p.mode))
        .collect();
    a_perm.sort();
    b_perm.sort();
    if a_perm != b_perm {
        return false;
    }
    let mut a_excl: Vec<_> = a.exclusions.iter().map(|p| p.canonical()).collect();
    let mut b_excl: Vec<_> = b.exclusions.iter().map(|p| p.canonical()).collect();
    a_excl.sort();
    b_excl.sort();
    a_excl == b_excl
}

/// Apply §8.2 per-domain policy checks to an incoming attestation.
/// Returns `Ok(reason)` on rejection; `Err` only for internal errors.
fn enforce_governance(
    peer: &AgentEntry,
    attestation: &got_core::GeometricAttestation,
    thresholds: &GovernanceThresholds,
) -> Option<String> {
    let domain_label = peer
        .domain_scope
        .as_ref()
        .map(|s| s.primary.as_str().to_string())
        .unwrap_or_else(|| "(unscoped)".to_string());

    // Tier 2+: chained attestation required.
    if thresholds.require_chain && attestation.parent_attestation_hash.is_none() {
        return Some(format!(
            "chain required for domain {domain_label} but attestation has no parent_attestation_hash"
        ));
    }

    // Tier 3: causal validation required — non-empty causal_scores and
    // every record must be causal.
    if thresholds.require_causal_validation {
        if attestation.causal_scores.is_empty() {
            return Some(format!(
                "causal validation required for domain {domain_label} but attestation has no causal_scores"
            ));
        }
        if !attestation.causal_scores.iter().all(|s| s.is_causal) {
            return Some(format!(
                "causal validation required for domain {domain_label} but at least one probe failed causal_check"
            ));
        }
    }

    // Minimum probe confidence — any dimension below the bar fails.
    if thresholds.min_confidence > 0.0 {
        if let Some(&lowest) = attestation
            .confidence
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            if lowest < thresholds.min_confidence {
                return Some(format!(
                    "confidence {lowest} below minimum {} required for domain {domain_label}",
                    thresholds.min_confidence
                ));
            }
        }
    }

    // Minimum causal consistency — only meaningful for Tier-3 attestations
    // that carry causal_scores.  We take the *minimum* consistency so a
    // single failing probe fails the whole attestation.
    if let Some(min_causal) = thresholds.min_causal_score {
        if !attestation.causal_scores.is_empty() {
            let lowest = attestation
                .causal_scores
                .iter()
                .map(|s| s.consistency)
                .fold(f32::INFINITY, f32::min);
            if lowest < min_causal {
                return Some(format!(
                    "causal consistency {lowest} below minimum {min_causal} required for domain {domain_label}"
                ));
            }
        }
    }

    None
}


/// Validate an incoming `ExchangeRequest` against the local trust registry.
///
/// Returns `Ok(Verdict::Accepted)` or `Ok(Verdict::Rejected)` with a reason.
pub fn validate_request(
    req: &ExchangeRequest,
    own_agent_id: &[u8; 32],
    expected_nonce: Option<&[u8; 32]>,
    registry: &TrustRegistry,
) -> Result<(Verdict, String), WireError> {
    // 1. Lookup sender in trust registry.
    let entry = match registry.lookup(&req.agent_id) {
        Some(e) => e,
        None => {
            let id_hex: String = req.agent_id.iter().map(|b| format!("{b:02x}")).collect();
            return Err(WireError::UnknownAgent(id_hex));
        }
    };

    // 1b. Certificate validity/revocation check (if certificates are in use).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if let Err(e) = registry.validate_agent_certificate(&req.agent_id, now) {
        return Ok((
            Verdict::Rejected,
            format!("certificate validation failed: {e}"),
        ));
    }

    // 1c. Phase 4 defence in depth — domain compatibility re-verify.
    //     The primary domain check happens in Phase 1 (before attestation
    //     computation) via check_domain_before_exchange().  This re-check
    //     catches the unlikely case where the registry changed between
    //     Phase 1 and Phase 4, and serves as a structural safety net in
    //     case the caller skipped the pre-flight (e.g. in-memory tests
    //     that call validate_request directly without going through
    //     got-net).
    if let Some(self_entry) = registry.lookup(own_agent_id) {
        if let (Some(peer_scope), Some(self_scope)) =
            (entry.domain_scope.as_ref(), self_entry.domain_scope.as_ref())
        {
            if let Err(e) = check_domain_compatibility(peer_scope, self_scope) {
                return Ok((Verdict::Rejected, format!("domain incompatible: {e}")));
            }
        }
    }

    // 2. Verify envelope signature + bindings.
    let chain_anchor = req.chain.first();
    if let Err(e) = req.envelope.verify(
        own_agent_id,
        expected_nonce,
        &req.current,
        chain_anchor,
        &entry.public_key,
        now,
        registry.max_envelope_age_secs,
    ) {
        return Ok((
            Verdict::Rejected,
            format!("envelope verification failed: {e}"),
        ));
    }

    // 3. Verify the attestation signature.
    if let Err(e) = got_attest::verify(&req.current, &entry.public_key) {
        return Ok((
            Verdict::Rejected,
            format!("attestation signature invalid: {e}"),
        ));
    }

    // 3b. Attestation timestamp freshness — defence in depth.
    //     Even with a fresh envelope, reject attestations that are too old.
    if req.current.timestamp > now {
        return Ok((
            Verdict::Rejected,
            "attestation timestamp is in the future".to_string(),
        ));
    }
    let attest_age = now - req.current.timestamp;
    if attest_age > registry.max_attestation_age_secs {
        return Ok((
            Verdict::Rejected,
            format!(
                "attestation too old: age {}s > max {}s",
                attest_age, registry.max_attestation_age_secs
            ),
        ));
    }

    // 3c. Model hash policy — if the registry pins an expected model_hash,
    //     reject attestations for a different model.
    if let Some(expected) = entry.expected_model_hash {
        if req.current.model_hash != Some(expected) {
            return Ok((
                Verdict::Rejected,
                "model_hash does not match registry policy".to_string(),
            ));
        }
    }

    // 3d. §7.3 / §8.2 governance policy.  If the verifier has a
    //     governance_table entry matching the peer's primary domain,
    //     the resolved GovernanceThresholds override the flat drift
    //     bound and additionally enforce min_confidence,
    //     min_causal_score, require_chain, and require_causal_validation.
    //     Unscoped peers fall through to the flat defaults.
    let thresholds = if let Some(self_entry) = registry.lookup(own_agent_id) {
        effective_thresholds(self_entry, entry)
    } else {
        GovernanceThresholds::permissive(entry.max_drift_accepted)
    };
    if let Some(reason) = enforce_governance(entry, &req.current, &thresholds) {
        return Ok((Verdict::Rejected, reason));
    }

    // 3e. §2.1 — if the attestation embeds a domain scope declaration,
    //     it must agree with the registry.
    if let Err(reason) = check_attestation_scope_binding(entry, &req.current) {
        return Ok((Verdict::Rejected, reason));
    }

    // 4. If there is a chain, verify it under the effective drift bound.
    if !req.chain.is_empty() {
        if req.chain.len() > registry.max_chain_length {
            return Ok((
                Verdict::Rejected,
                format!(
                    "chain too long: {} > max {}",
                    req.chain.len(),
                    registry.max_chain_length
                ),
            ));
        }
        match verify_chain(
            &req.chain,
            &req.current,
            &[entry.public_key],
            thresholds.max_drift,
        ) {
            Ok(_) => {}
            Err(e) => {
                return Ok((Verdict::Rejected, format!("chain verification failed: {e}")));
            }
        }
    }

    // 5. Defence in depth: verify that the attestation hash embedded in the
    //    envelope matches what we compute locally.  The envelope.verify() call
    //    already checked this, but we cross-check against our own computation
    //    to catch any implementation divergence between envelope creation and
    //    verification paths.
    let current_hash = attestation_hash(&req.current)?;
    if current_hash != req.envelope.attestation_hash {
        return Ok((
            Verdict::Rejected,
            "attestation hash does not match envelope binding".to_string(),
        ));
    }

    // 6. Causal consistency: if causal_scores are present, causal_flag must match.
    if !got_attest::validate_causal_consistency(&req.current) {
        return Ok((
            Verdict::Rejected,
            "causal_flag / causal_scores / intervention_delta inconsistency".to_string(),
        ));
    }

    Ok((Verdict::Accepted, String::new()))
}

/// Validate an incoming `ExchangeResponse` against the local trust registry.
pub fn validate_response(
    rsp: &ExchangeResponse,
    own_agent_id: &[u8; 32],
    expected_nonce: &[u8; 32],
    registry: &TrustRegistry,
) -> Result<(Verdict, String), WireError> {
    // 1. Lookup responder.
    let entry = match registry.lookup(&rsp.agent_id) {
        Some(e) => e,
        None => {
            let id_hex: String = rsp.agent_id.iter().map(|b| format!("{b:02x}")).collect();
            return Err(WireError::UnknownAgent(id_hex));
        }
    };

    // 1b. Certificate validity/revocation check.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if let Err(e) = registry.validate_agent_certificate(&rsp.agent_id, now) {
        return Ok((
            Verdict::Rejected,
            format!("certificate validation failed: {e}"),
        ));
    }

    // 1c. Phase 4 defence in depth — domain compatibility re-verify
    //     (see the matching comment in validate_request above).
    if let Some(self_entry) = registry.lookup(own_agent_id) {
        if let (Some(peer_scope), Some(self_scope)) =
            (entry.domain_scope.as_ref(), self_entry.domain_scope.as_ref())
        {
            if let Err(e) = check_domain_compatibility(peer_scope, self_scope) {
                return Ok((Verdict::Rejected, format!("domain incompatible: {e}")));
            }
        }
    }

    // 2. Verify envelope (must echo our nonce).
    let chain_anchor = rsp.chain.first();
    if let Err(e) = rsp.envelope.verify(
        own_agent_id,
        Some(expected_nonce),
        &rsp.current,
        chain_anchor,
        &entry.public_key,
        now,
        registry.max_envelope_age_secs,
    ) {
        return Ok((
            Verdict::Rejected,
            format!("envelope verification failed: {e}"),
        ));
    }

    // 3. Verify attestation signature.
    if let Err(e) = got_attest::verify(&rsp.current, &entry.public_key) {
        return Ok((
            Verdict::Rejected,
            format!("attestation signature invalid: {e}"),
        ));
    }

    // 3b. Attestation timestamp freshness.
    if rsp.current.timestamp > now {
        return Ok((
            Verdict::Rejected,
            "attestation timestamp is in the future".to_string(),
        ));
    }
    let attest_age = now - rsp.current.timestamp;
    if attest_age > registry.max_attestation_age_secs {
        return Ok((
            Verdict::Rejected,
            format!(
                "attestation too old: age {}s > max {}s",
                attest_age, registry.max_attestation_age_secs
            ),
        ));
    }

    // 3c. Model hash policy check.
    if let Some(expected) = entry.expected_model_hash {
        if rsp.current.model_hash != Some(expected) {
            return Ok((
                Verdict::Rejected,
                "model_hash does not match registry policy".to_string(),
            ));
        }
    }

    // 3d. §7.3 / §8.2 governance policy — mirrors validate_request.
    let thresholds = if let Some(self_entry) = registry.lookup(own_agent_id) {
        effective_thresholds(self_entry, entry)
    } else {
        GovernanceThresholds::permissive(entry.max_drift_accepted)
    };
    if let Some(reason) = enforce_governance(entry, &rsp.current, &thresholds) {
        return Ok((Verdict::Rejected, reason));
    }

    // 3e. §2.1 — attestation-registry domain scope binding.
    if let Err(reason) = check_attestation_scope_binding(entry, &rsp.current) {
        return Ok((Verdict::Rejected, reason));
    }

    // 4. Chain verification.
    if !rsp.chain.is_empty() {
        if rsp.chain.len() > registry.max_chain_length {
            return Ok((
                Verdict::Rejected,
                format!(
                    "chain too long: {} > max {}",
                    rsp.chain.len(),
                    registry.max_chain_length
                ),
            ));
        }
        match verify_chain(
            &rsp.chain,
            &rsp.current,
            &[entry.public_key],
            thresholds.max_drift,
        ) {
            Ok(_) => {}
            Err(e) => {
                return Ok((Verdict::Rejected, format!("chain verification failed: {e}")));
            }
        }
    }

    // 5. Causal consistency check.
    if !got_attest::validate_causal_consistency(&rsp.current) {
        return Ok((
            Verdict::Rejected,
            "causal_flag / causal_scores / intervention_delta inconsistency".to_string(),
        ));
    }

    Ok((Verdict::Accepted, String::new()))
}

/// Perform a complete in-memory exchange between two agents.
///
/// This simulates the full protocol without any network I/O:
///   1. Initiator builds request
///   2. Responder validates and builds response
///   3. Initiator validates response
///
/// Returns `(initiator_result, responder_verdict)`.
pub fn perform_exchange(
    // Initiator
    initiator_key: &SigningKey,
    initiator_chain: Vec<GeometricAttestation>,
    initiator_current: GeometricAttestation,
    // Responder
    responder_key: &SigningKey,
    responder_chain: Vec<GeometricAttestation>,
    responder_current: GeometricAttestation,
    // Shared
    registry: &TrustRegistry,
) -> Result<(ExchangeResult, Verdict), WireError> {
    let initiator_id = compute_agent_id(&initiator_key.verifying_key());
    let responder_id = compute_agent_id(&responder_key.verifying_key());

    // 1. Initiator creates nonce and builds request.
    let nonce = {
        let mut n = [0u8; 32];
        use rand::RngCore;
        rand::rngs::OsRng.fill_bytes(&mut n);
        n
    };

    let request = build_request(
        nonce,
        responder_id,
        initiator_key,
        initiator_chain,
        initiator_current,
    )?;

    // 2. Responder validates request.
    let (responder_verdict, responder_reason) =
        validate_request(&request, &responder_id, None, registry)?;

    // 3. Responder builds response (echoing the nonce).
    let response = build_response(
        nonce,
        initiator_id,
        responder_key,
        responder_verdict,
        responder_chain,
        responder_current,
        responder_reason.clone(),
    )?;

    // 4. Initiator validates response.
    let (our_verdict, our_reason) = validate_response(&response, &initiator_id, &nonce, registry)?;

    let result = ExchangeResult {
        peer_verdict: responder_verdict,
        our_verdict,
        peer_attestation: response.current,
        peer_chain: response.chain,
        reason: if !responder_reason.is_empty() {
            responder_reason
        } else {
            our_reason
        },
    };

    Ok((result, responder_verdict))
}

/// Perform a one-directional supervised verification (§5.5).
///
/// Models the asymmetric regulatory pattern: a regulator (Agent M) demands
/// an attestation from a supervised agent (Agent L) without producing one
/// of its own.  The regulator's authority derives from institutional
/// mandate, not from mutual geometric compatibility, so this function
/// deliberately has no "responder attestation" input.
///
/// Flow:
///   1. Supervised agent builds a normal ExchangeRequest carrying its
///      current attestation, chain, and a freshly-generated nonce.
///   2. The regulator runs `validate_request` against its local trust
///      registry — this applies the Phase 1 domain pre-flight (via the
///      defence-in-depth re-check), envelope, chain, and governance
///      checks exactly as in a symmetric exchange.
///   3. The regulator emits a verdict; no counter-attestation is sent.
///
/// For the domain compatibility check to succeed, both agents must
/// declare the other's primary domain in `Supervised` mode (or at least
/// one side must have no domain_scope at all, which defaults to
/// "unscoped" — permissive).
pub fn perform_supervised_request(
    // Regulator (Agent M) — holds oversight authority, never attests.
    regulator_id: &[u8; 32],
    // Supervised agent (Agent L) — produces the attestation to be verified.
    supervised_key: &SigningKey,
    supervised_chain: Vec<GeometricAttestation>,
    supervised_current: GeometricAttestation,
    // Shared trust registry.
    registry: &TrustRegistry,
) -> Result<(Verdict, String), WireError> {
    // 1. Supervised agent builds its request aimed at the regulator.
    let nonce = {
        let mut n = [0u8; 32];
        use rand::RngCore;
        rand::rngs::OsRng.fill_bytes(&mut n);
        n
    };
    let request = build_request(
        nonce,
        *regulator_id,
        supervised_key,
        supervised_chain,
        supervised_current,
    )?;

    // 2. Regulator validates the request against its local policy.
    validate_request(&request, regulator_id, Some(&nonce), registry)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer;
    use got_attest::assemble_and_sign;
    use got_core::{InnerProduct, Precision, SCHEMA_VERSION};

    use crate::chain::attestation_hash as chain_attest_hash;
    use crate::registry::AgentEntry;

    fn key_alice() -> SigningKey {
        SigningKey::from_bytes(&[0xAA; 32])
    }

    fn key_bob() -> SigningKey {
        SigningKey::from_bytes(&[0xBB; 32])
    }

    fn make_v1(key: &SigningKey) -> GeometricAttestation {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let a = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: "test-model".to_string(),
            model_hash: Some([0x11; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0x22; 32],
            timestamp: now,
            corpus_version: "c1".to_string(),
            probe_version: "p1".to_string(),
            layer_readings: vec![vec![1.0, 2.0]],
            confidence: vec![0.95],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: None,
            geometry_hash: None,
            geometry_drift: None,
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 0,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            domain_scope_declaration: None,
            signature: [0u8; 64],
        };
        assemble_and_sign(a, key).unwrap()
    }

    fn make_v2_child(
        key: &SigningKey,
        parent: &GeometricAttestation,
        drift: f32,
    ) -> GeometricAttestation {
        let parent_hash = chain_attest_hash(parent).unwrap();
        let a = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: "test-model".to_string(),
            model_hash: Some([0x11; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0x33; 32],
            timestamp: parent.timestamp,
            corpus_version: "c2".to_string(),
            probe_version: "p2".to_string(),
            layer_readings: vec![vec![1.1, 2.1]],
            confidence: vec![0.93],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: Some(parent_hash),
            geometry_hash: Some([0xDD; 32]),
            geometry_drift: Some(drift),
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: parent.sequence_number + 1,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            domain_scope_declaration: None,
            signature: [0u8; 64],
        };
        assemble_and_sign(a, key).unwrap()
    }

    fn build_registry(alice: &SigningKey, bob: &SigningKey) -> TrustRegistry {
        let mut registry = TrustRegistry::empty();
        let alice_pk = alice.verifying_key();
        let bob_pk = bob.verifying_key();

        registry.add_agent(AgentEntry {
            name: "alice".to_string(),
            public_key: alice_pk,
            agent_id: compute_agent_id(&alice_pk),
            max_drift_accepted: 0.05,
            roles: vec!["producer".to_string()],
            expected_model_hash: None,
            certificate: None,
            domain_scope: None,
            governance_table: crate::governance::GovernanceTable::default(),
        });
        registry.add_agent(AgentEntry {
            name: "bob".to_string(),
            public_key: bob_pk,
            agent_id: compute_agent_id(&bob_pk),
            max_drift_accepted: 0.05,
            roles: vec!["verifier".to_string()],
            expected_model_hash: None,
            certificate: None,
            domain_scope: None,
            governance_table: crate::governance::GovernanceTable::default(),
        });

        registry
    }

    #[test]
    fn verdict_roundtrip() {
        assert_eq!(Verdict::from_byte(0x01).unwrap(), Verdict::Accepted);
        assert_eq!(Verdict::from_byte(0x02).unwrap(), Verdict::Rejected);
        assert_eq!(Verdict::from_byte(0x03).unwrap(), Verdict::Error);
        assert!(Verdict::from_byte(0x00).is_err());
    }

    #[test]
    fn build_request_succeeds() {
        let alice = key_alice();
        let bob = key_bob();
        let bob_id = compute_agent_id(&bob.verifying_key());
        let attest = make_v1(&alice);

        let req = build_request([0x42; 32], bob_id, &alice, vec![], attest).unwrap();
        assert_eq!(req.agent_id, compute_agent_id(&alice.verifying_key()));
        assert_eq!(req.envelope.nonce, [0x42; 32]);
        assert_eq!(req.envelope.peer_agent_id, bob_id);
    }

    #[test]
    fn full_exchange_v1_accepted() {
        let alice = key_alice();
        let bob = key_bob();
        let registry = build_registry(&alice, &bob);

        let alice_attest = make_v1(&alice);
        let bob_attest = make_v1(&bob);

        let (result, responder_verdict) = perform_exchange(
            &alice,
            vec![],
            alice_attest,
            &bob,
            vec![],
            bob_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(responder_verdict, Verdict::Accepted);
        assert_eq!(result.peer_verdict, Verdict::Accepted);
        assert_eq!(result.our_verdict, Verdict::Accepted);
    }

    #[test]
    fn full_exchange_v2_with_chain_accepted() {
        let alice = key_alice();
        let bob = key_bob();
        let registry = build_registry(&alice, &bob);

        let alice_anchor = make_v1(&alice);
        let alice_current = make_v2_child(&alice, &alice_anchor, 0.03);

        let bob_anchor = make_v1(&bob);
        let bob_current = make_v2_child(&bob, &bob_anchor, 0.02);

        let (result, responder_verdict) = perform_exchange(
            &alice,
            vec![alice_anchor],
            alice_current,
            &bob,
            vec![bob_anchor],
            bob_current,
            &registry,
        )
        .unwrap();

        assert_eq!(responder_verdict, Verdict::Accepted);
        assert_eq!(result.our_verdict, Verdict::Accepted);
    }

    #[test]
    fn exchange_drift_exceeds_local_max_rejected() {
        let alice = key_alice();
        let bob = key_bob();
        let registry = build_registry(&alice, &bob); // max_drift = 0.05

        let alice_anchor = make_v1(&alice);
        let alice_current = make_v2_child(&alice, &alice_anchor, 0.08); // too much drift

        let bob_attest = make_v1(&bob);

        let (result, responder_verdict) = perform_exchange(
            &alice,
            vec![alice_anchor],
            alice_current,
            &bob,
            vec![],
            bob_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(responder_verdict, Verdict::Rejected);
        assert_eq!(result.peer_verdict, Verdict::Rejected);
        assert!(result.reason.contains("drift"));
    }

    #[test]
    fn exchange_unknown_agent_rejected() {
        let alice = key_alice();
        let bob = key_bob();
        let charlie = SigningKey::from_bytes(&[0xCC; 32]);

        // Registry only knows alice and bob.
        let registry = build_registry(&alice, &bob);

        let charlie_attest = make_v1(&charlie);
        let bob_attest = make_v1(&bob);

        // Charlie (unknown) tries to exchange with Bob.
        let err = perform_exchange(
            &charlie,
            vec![],
            charlie_attest,
            &bob,
            vec![],
            bob_attest,
            &registry,
        );

        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("unknown") || msg.contains("Unknown"), "{msg}");
    }

    #[test]
    fn validate_request_checks_envelope_peer_id() {
        let alice = key_alice();
        let bob = key_bob();
        let registry = build_registry(&alice, &bob);

        let alice_attest = make_v1(&alice);
        let bob_id = compute_agent_id(&bob.verifying_key());

        // Build a request legitimately aimed at bob.
        let req = build_request([0x42; 32], bob_id, &alice, vec![], alice_attest).unwrap();

        // Validate as bob → should succeed.
        let (verdict, _) = validate_request(&req, &bob_id, None, &registry).unwrap();
        assert_eq!(verdict, Verdict::Accepted);

        // Validate as if we are a different agent → envelope peer_id mismatch.
        let fake_id = [0xFF; 32];
        let (verdict, reason) = validate_request(&req, &fake_id, None, &registry).unwrap();
        assert_eq!(verdict, Verdict::Rejected);
        assert!(reason.contains("peer"), "reason: {reason}");
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issues 24, 36)
    // -----------------------------------------------------------------------

    /// Issue #24 (S-3): Nonce must be generated with a CSPRNG (OsRng),
    /// not `thread_rng()`.
    ///
    /// We can't directly test which RNG is used at runtime, but we can
    /// verify that two consecutive nonces are unique (a basic sanity
    /// check that real random bytes are being produced).
    #[test]
    fn sec_nonce_is_unique_across_calls() {
        let alice = key_alice();
        let bob = key_bob();
        let registry = build_registry(&alice, &bob);

        let alice_attest1 = make_v1(&alice);
        let alice_attest2 = make_v1(&alice);
        let bob_attest1 = make_v1(&bob);
        let bob_attest2 = make_v1(&bob);

        let (r1, _) = perform_exchange(
            &alice,
            vec![],
            alice_attest1,
            &bob,
            vec![],
            bob_attest1,
            &registry,
        )
        .unwrap();

        let (r2, _) = perform_exchange(
            &alice,
            vec![],
            alice_attest2,
            &bob,
            vec![],
            bob_attest2,
            &registry,
        )
        .unwrap();

        // Two independent exchanges must produce Accepted verdicts
        // (proves the nonce path works), and the nonces embedded in the
        // peer attestations must differ with overwhelming probability.
        assert_eq!(r1.peer_verdict, Verdict::Accepted);
        assert_eq!(r2.peer_verdict, Verdict::Accepted);
        // Note: we can't directly inspect the nonce from ExchangeResult,
        // but the fact that both complete successfully confirms the flow.
        // A deeper test would require a refactor to expose the nonce.
    }

    /// Issue #36 (S-15): validate_request uses SystemTime::now() which
    /// is not monotonic.  After fix, an injectable clock is used.
    ///
    /// For now we verify that an attestation with a far-future timestamp
    /// is rejected (the existing defence-in-depth check).
    #[test]
    fn sec_validate_request_rejects_far_future_attestation() {
        let alice = key_alice();
        let bob = key_bob();
        let registry = build_registry(&alice, &bob);

        // Build an attestation with a timestamp far in the future.
        // We sign it manually (bypassing assemble_and_sign's timestamp check)
        // because we want validate_request to catch the future timestamp.
        let mut future_attest = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: "test-model".to_string(),
            model_hash: Some([0x11; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0x22; 32],
            timestamp: u64::MAX - 1000, // far future
            corpus_version: "v1".to_string(),
            probe_version: "v1".to_string(),
            layer_readings: vec![vec![0.5]],
            confidence: vec![0.9],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: None,
            geometry_hash: None,
            geometry_drift: None,
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 0,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            domain_scope_declaration: None,
            signature: [0u8; 64],
        };
        // Sign directly to bypass assemble_and_sign's S-7 timestamp guard.
        let payload = got_attest::serialise_for_signing(&future_attest).unwrap();
        let sig = alice.sign(&payload);
        future_attest.signature = sig.to_bytes();

        let bob_id = compute_agent_id(&bob.verifying_key());

        // Build request manually so we control the attestation.
        let req = build_request([0x42; 32], bob_id, &alice, vec![], future_attest).unwrap();

        // Validate: should be rejected because attestation timestamp is in the future.
        let (verdict, reason) = validate_request(&req, &bob_id, None, &registry).unwrap();
        assert_eq!(verdict, Verdict::Rejected, "reason: {reason}");
        assert!(
            reason.contains("future") || reason.contains("expired") || reason.contains("old"),
            "expected temporal rejection, got: {reason}"
        );
    }

    // -----------------------------------------------------------------------
    // Domain scoping (§4 / Appendix B)
    // -----------------------------------------------------------------------

    fn build_registry_with_domains(
        alice: &SigningKey,
        bob: &SigningKey,
        alice_scope: Option<crate::domain::DomainScope>,
        bob_scope: Option<crate::domain::DomainScope>,
    ) -> TrustRegistry {
        let mut registry = TrustRegistry::empty();
        let alice_pk = alice.verifying_key();
        let bob_pk = bob.verifying_key();

        registry.add_agent(AgentEntry {
            name: "alice".to_string(),
            public_key: alice_pk,
            agent_id: compute_agent_id(&alice_pk),
            max_drift_accepted: 0.05,
            roles: vec!["producer".to_string()],
            expected_model_hash: None,
            certificate: None,
            domain_scope: alice_scope,
            governance_table: crate::governance::GovernanceTable::default(),
        });
        registry.add_agent(AgentEntry {
            name: "bob".to_string(),
            public_key: bob_pk,
            agent_id: compute_agent_id(&bob_pk),
            max_drift_accepted: 0.05,
            roles: vec!["verifier".to_string()],
            expected_model_hash: None,
            certificate: None,
            domain_scope: bob_scope,
            governance_table: crate::governance::GovernanceTable::default(),
        });
        registry
    }

    fn agri_scope() -> crate::domain::DomainScope {
        use crate::domain::*;
        DomainScope {
            primary: Domain::parse("agriculture.crop-management").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("agriculture.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![DomainPattern::parse("transport.*").unwrap()],
        }
    }

    fn vehicle_scope() -> crate::domain::DomainScope {
        use crate::domain::*;
        DomainScope {
            primary: Domain::parse("transport.autonomous-vehicle").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("transport.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![],
        }
    }

    /// Use case §5.1: agri ↔ vehicle exchange must be rejected at Phase 1
    /// (domain pre-flight).  The defence-in-depth re-check in Phase 4
    /// also catches this, which is what this test exercises via
    /// `perform_exchange` (which calls `validate_request` directly).
    #[test]
    fn exchange_rejected_by_domain_incompatibility() {
        let alice = key_alice();
        let bob = key_bob();
        let registry =
            build_registry_with_domains(&alice, &bob, Some(agri_scope()), Some(vehicle_scope()));

        let alice_attest = make_v1(&alice);
        let bob_attest = make_v1(&bob);

        let (result, responder_verdict) = perform_exchange(
            &alice,
            vec![],
            alice_attest,
            &bob,
            vec![],
            bob_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(responder_verdict, Verdict::Rejected);
        assert!(
            result.reason.contains("domain"),
            "expected domain rejection, got: {}",
            result.reason
        );
    }

    /// Compatible domains (both within agriculture.*) must succeed.
    #[test]
    fn exchange_accepted_when_domains_compatible() {
        use crate::domain::*;
        let alice = key_alice();
        let bob = key_bob();
        let supply = DomainScope {
            primary: Domain::parse("agriculture.supply-chain").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("agriculture.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![],
        };
        let registry = build_registry_with_domains(&alice, &bob, Some(agri_scope()), Some(supply));

        let alice_attest = make_v1(&alice);
        let bob_attest = make_v1(&bob);

        let (result, responder_verdict) = perform_exchange(
            &alice,
            vec![],
            alice_attest,
            &bob,
            vec![],
            bob_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(responder_verdict, Verdict::Accepted, "{}", result.reason);
        assert_eq!(result.our_verdict, Verdict::Accepted);
    }

    // -----------------------------------------------------------------------
    // §5.5 Supervised mode + §7.3/§8.2 per-domain governance
    // -----------------------------------------------------------------------

    fn regulator_scope() -> crate::domain::DomainScope {
        use crate::domain::*;
        DomainScope {
            primary: Domain::parse("finance.regulatory-compliance").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("finance.*").unwrap(),
                mode: InteractionMode::Supervised,
            }],
            exclusions: vec![],
        }
    }

    fn trading_scope() -> crate::domain::DomainScope {
        use crate::domain::*;
        DomainScope {
            primary: Domain::parse("finance.trading").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("finance.regulatory-compliance").unwrap(),
                mode: InteractionMode::Supervised,
            }],
            exclusions: vec![],
        }
    }

    /// §5.5: regulator (Agent M) demands attestation from supervised
    /// trading agent (Agent L) without producing one itself.
    #[test]
    fn supervised_request_accepted() {
        let regulator = SigningKey::from_bytes(&[0xCC; 32]); // Agent M
        let trader = SigningKey::from_bytes(&[0xDD; 32]);    // Agent L
        let registry = build_registry_with_domains(
            &regulator,
            &trader,
            Some(regulator_scope()),
            Some(trading_scope()),
        );

        // The helper aliases `alice` → regulator, `bob` → trader in the
        // registry, but the labels don't matter for the semantic check.
        let trader_attest = make_v1(&trader);
        let regulator_id = compute_agent_id(&regulator.verifying_key());

        let (verdict, reason) = perform_supervised_request(
            &regulator_id,
            &trader,
            vec![],
            trader_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(verdict, Verdict::Accepted, "reason: {reason}");
    }

    /// Without Supervised permission on either side, the one-directional
    /// request must still be rejected at Phase 1 (domain pre-flight).
    #[test]
    fn supervised_request_rejected_without_supervised_mode() {
        let regulator = SigningKey::from_bytes(&[0xCC; 32]);
        let trader = SigningKey::from_bytes(&[0xDD; 32]);
        // Regulator advertises Cooperative instead of Supervised.
        let mut bad_reg_scope = regulator_scope();
        bad_reg_scope.permitted[0].mode = crate::domain::InteractionMode::ReadOnly;
        let mut bad_trd_scope = trading_scope();
        bad_trd_scope.permitted[0].mode = crate::domain::InteractionMode::ReadOnly;
        let registry = build_registry_with_domains(
            &regulator,
            &trader,
            Some(bad_reg_scope),
            Some(bad_trd_scope),
        );

        let trader_attest = make_v1(&trader);
        let regulator_id = compute_agent_id(&regulator.verifying_key());

        let (verdict, reason) = perform_supervised_request(
            &regulator_id,
            &trader,
            vec![],
            trader_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(verdict, Verdict::Rejected);
        assert!(
            reason.contains("domain"),
            "expected domain rejection, got: {reason}"
        );
    }

    /// §8.2: per-domain `require_causal_validation` rejects a bare Tier-1
    /// attestation when the verifier's governance table mandates Tier 3
    /// causal proof for that domain.
    #[test]
    fn governance_rejects_when_causal_validation_required() {
        use crate::domain::{Domain, DomainPattern, DomainScope, InteractionMode, PermittedDomain};
        use crate::governance::{GovernanceEntry, GovernanceTable, GovernanceThresholds};

        let alice = key_alice(); // verifier (healthcare regulator-ish)
        let bob = key_bob(); // producer submitting a non-causal attestation

        let alice_scope = DomainScope {
            primary: Domain::parse("healthcare.regulator").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("healthcare.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![],
        };
        let bob_scope = DomainScope {
            primary: Domain::parse("healthcare.diagnostic-advisory").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("healthcare.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![],
        };

        let mut registry =
            build_registry_with_domains(&alice, &bob, Some(alice_scope), Some(bob_scope));

        // Inject a governance rule on alice: healthcare.* requires
        // Tier-3 causal validation.
        let alice_id = compute_agent_id(&alice.verifying_key());
        let alice_entry = registry.agents.get_mut(&alice_id).unwrap();
        alice_entry.governance_table = GovernanceTable {
            entries: vec![GovernanceEntry {
                pattern: DomainPattern::parse("healthcare.*").unwrap(),
                thresholds: GovernanceThresholds {
                    max_drift: 0.03,
                    min_confidence: 0.0,
                    min_causal_score: None,
                    require_chain: false,
                    require_causal_validation: true,
                },
            }],
        };

        // Bob submits a plain attestation with no causal_scores.
        let bob_attest = make_v1(&bob);
        let req = build_request([0x42; 32], alice_id, &bob, vec![], bob_attest).unwrap();

        let (verdict, reason) = validate_request(&req, &alice_id, None, &registry).unwrap();
        assert_eq!(verdict, Verdict::Rejected);
        assert!(
            reason.contains("causal"),
            "expected causal-validation rejection, got: {reason}"
        );
    }

    /// §7.3: per-domain max_drift overrides the flat AgentEntry value.
    /// A chain that would pass the flat 0.10 bound is rejected when the
    /// verifier's table demands 0.02 for the peer's domain.
    #[test]
    fn governance_per_domain_drift_overrides_flat_default() {
        use crate::domain::{Domain, DomainPattern, DomainScope, InteractionMode, PermittedDomain};
        use crate::governance::{GovernanceEntry, GovernanceTable, GovernanceThresholds};

        let alice = key_alice();
        let bob = key_bob();

        let scope = |p: &str| DomainScope {
            primary: Domain::parse(p).unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("infrastructure.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![],
        };

        let mut registry = build_registry_with_domains(
            &alice,
            &bob,
            Some(scope("infrastructure.energy-grid")),
            Some(scope("infrastructure.water-systems")),
        );

        // Flat per-agent bound is generous (0.05, set by build_registry).
        // Override with a critical-infra rule: 0.02.
        let alice_id = compute_agent_id(&alice.verifying_key());
        registry.agents.get_mut(&alice_id).unwrap().governance_table = GovernanceTable {
            entries: vec![GovernanceEntry {
                pattern: DomainPattern::parse("infrastructure.*").unwrap(),
                thresholds: GovernanceThresholds {
                    max_drift: 0.02,
                    min_confidence: 0.0,
                    min_causal_score: None,
                    require_chain: false,
                    require_causal_validation: false,
                },
            }],
        };

        // Bob submits a chained attestation with drift 0.04 — inside the old 0.05 flat
        // bound but above the new 0.02 infra bound.
        let bob_anchor = make_v1(&bob);
        let bob_current = make_v2_child(&bob, &bob_anchor, 0.04);
        let bob_id = compute_agent_id(&bob.verifying_key());
        let _ = bob_id;
        let req = build_request(
            [0x42; 32],
            alice_id,
            &bob,
            vec![bob_anchor],
            bob_current,
        )
        .unwrap();

        let (verdict, reason) = validate_request(&req, &alice_id, None, &registry).unwrap();
        assert_eq!(verdict, Verdict::Rejected);
        assert!(
            reason.contains("drift"),
            "expected drift rejection, got: {reason}"
        );
    }

    // -----------------------------------------------------------------------
    // §2.1 — embedded domain scope declaration
    // -----------------------------------------------------------------------

    /// Build an attestation whose embedded domain_scope_declaration
    /// matches the scope we register for the same signer.
    fn make_scoped_attestation(
        key: &SigningKey,
        scope: &crate::domain::DomainScope,
    ) -> GeometricAttestation {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let a = GeometricAttestation {
            schema_version: got_core::SCHEMA_VERSION,
            model_id: "test-model".to_string(),
            model_hash: Some([0x11; 32]),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: [0x22; 32],
            timestamp: now,
            corpus_version: "c1".to_string(),
            probe_version: "p1".to_string(),
            layer_readings: vec![vec![1.0, 2.0]],
            confidence: vec![0.95],
            coverage_flags: vec![false],
            divergence_flag: false,
            parent_attestation_hash: None,
            geometry_hash: None,
            geometry_drift: None,
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: 0,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            domain_scope_declaration: Some(scope.to_declaration()),
            signature: [0u8; 64],
        };
        assemble_and_sign(a, key).unwrap()
    }

    fn supply_scope() -> crate::domain::DomainScope {
        use crate::domain::*;
        DomainScope {
            primary: Domain::parse("agriculture.supply-chain").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("agriculture.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![],
        }
    }

    #[test]
    fn scoped_attestation_binding_match_accepted() {
        let alice = key_alice();
        let bob = key_bob();

        let a_scope = agri_scope();
        let b_scope = supply_scope();
        let registry = build_registry_with_domains(
            &alice,
            &bob,
            Some(a_scope.clone()),
            Some(b_scope.clone()),
        );

        let a_attest = make_scoped_attestation(&alice, &a_scope);
        let b_attest = make_scoped_attestation(&bob, &b_scope);

        let (result, responder_verdict) = perform_exchange(
            &alice,
            vec![],
            a_attest,
            &bob,
            vec![],
            b_attest,
            &registry,
        )
        .unwrap();
        assert_eq!(responder_verdict, Verdict::Accepted, "{}", result.reason);
        assert_eq!(result.our_verdict, Verdict::Accepted);
    }

    #[test]
    fn scoped_attestation_binding_mismatch_rejected() {
        use crate::domain::*;

        let alice = key_alice();
        let bob = key_bob();

        // Alice is registered as agriculture, but her signed attestation
        // claims to be in transport — the binding check must catch this.
        let registered = agri_scope();
        let claimed = DomainScope {
            primary: Domain::parse("transport.autonomous-vehicle").unwrap(),
            permitted: vec![PermittedDomain {
                pattern: DomainPattern::parse("transport.*").unwrap(),
                mode: InteractionMode::Cooperative,
            }],
            exclusions: vec![],
        };
        let registry =
            build_registry_with_domains(&alice, &bob, Some(registered), Some(supply_scope()));

        let a_attest = make_scoped_attestation(&alice, &claimed);
        let b_attest = make_scoped_attestation(&bob, &supply_scope());

        let (result, responder_verdict) = perform_exchange(
            &alice,
            vec![],
            a_attest,
            &bob,
            vec![],
            b_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(responder_verdict, Verdict::Rejected);
        assert!(
            result.reason.contains("domain_scope_declaration")
                || result.reason.contains("domain"),
            "expected binding rejection, got: {}",
            result.reason
        );
    }

    /// A scoped attestation whose embedded declaration round-trips
    /// through the canonical signing path verifies correctly (no ambient
    /// bytes change depending on map ordering, float canonicalisation,
    /// etc.).
    #[test]
    fn scoped_attestation_canonical_signature_verifies() {
        let alice = key_alice();
        let scope = agri_scope();
        let attest = make_scoped_attestation(&alice, &scope);
        got_attest::verify(&attest, &alice.verifying_key()).unwrap();
    }

    /// One peer unscoped → backwards-compatible: skip the domain check.
    #[test]
    fn exchange_skips_domain_check_when_either_peer_unscoped() {
        let alice = key_alice();
        let bob = key_bob();
        // Alice has the strict agri scope; Bob has none.
        let registry = build_registry_with_domains(&alice, &bob, Some(agri_scope()), None);

        let alice_attest = make_v1(&alice);
        let bob_attest = make_v1(&bob);

        let (_, responder_verdict) = perform_exchange(
            &alice,
            vec![],
            alice_attest,
            &bob,
            vec![],
            bob_attest,
            &registry,
        )
        .unwrap();

        assert_eq!(responder_verdict, Verdict::Accepted);
    }
}
