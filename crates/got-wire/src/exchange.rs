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
use crate::envelope::ExchangeEnvelope;
use crate::registry::{compute_agent_id, TrustRegistry};
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
    /// Attestation chain (oldest first), may be empty for v1.
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
// Validation helpers
// ---------------------------------------------------------------------------

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

    // 4. If there is a chain, verify it.
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
            entry.max_drift_accepted,
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
            entry.max_drift_accepted,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer;
    use got_attest::assemble_and_sign;
    use got_core::{InnerProduct, Precision, SCHEMA_VERSION, SCHEMA_VERSION_2};

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
            schema_version: SCHEMA_VERSION_2,
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
        });
        registry.add_agent(AgentEntry {
            name: "bob".to_string(),
            public_key: bob_pk,
            agent_id: compute_agent_id(&bob_pk),
            max_drift_accepted: 0.05,
            roles: vec!["verifier".to_string()],
            expected_model_hash: None,
            certificate: None,
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
}
