// ---------------------------------------------------------------------------
// Behavioral Exchange Protocol — parallel to geometric exchange.
//
// New message types for exchanging behavioral attestations between agents
// monitoring closed-source models via the proxy architecture.
//
// Agent role: "behavioral-observer" — authorised to produce and exchange
// Tier 0 (behavioral) attestations.
// ---------------------------------------------------------------------------

use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};

use got_proxy::attestation::{
    attestation_hash, verify_attestation, BehavioralAttestation,
};

use crate::registry::{compute_agent_id, TrustRegistry};
use crate::exchange::Verdict;
use crate::WireError;

/// The agent role string for behavioral observers.
pub const BEHAVIORAL_OBSERVER_ROLE: &str = "behavioral-observer";

// ---------------------------------------------------------------------------
// Behavioral Exchange Request / Response
// ---------------------------------------------------------------------------

/// A BEHAVIORAL_EXCHANGE_REQ payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralExchangeRequest {
    /// SHA-256 of the sender's public key.
    pub agent_id: [u8; 32],
    /// Random nonce for this exchange.
    pub nonce: [u8; 32],
    /// The behavioral attestation chain (oldest first).
    pub chain: Vec<BehavioralAttestation>,
    /// The most recent behavioral attestation.
    pub current: BehavioralAttestation,
}

/// A BEHAVIORAL_EXCHANGE_RSP payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralExchangeResponse {
    /// SHA-256 of the responder's public key.
    pub agent_id: [u8; 32],
    /// Echo of the request nonce.
    pub nonce: [u8; 32],
    /// Accept / Reject / Error.
    pub verdict: u8,
    /// The responder's behavioral attestation chain.
    pub chain: Vec<BehavioralAttestation>,
    /// The responder's most recent behavioral attestation.
    pub current: BehavioralAttestation,
    /// Human-readable reason for the verdict.
    pub reason: String,
}

/// Build a behavioral exchange request.
pub fn build_behavioral_request(
    nonce: [u8; 32],
    own_key: &SigningKey,
    chain: Vec<BehavioralAttestation>,
    current: BehavioralAttestation,
) -> Result<BehavioralExchangeRequest, WireError> {
    let agent_id = compute_agent_id(&own_key.verifying_key());

    // Validate the chain links
    verify_behavioral_chain(&chain, &current)?;

    Ok(BehavioralExchangeRequest {
        agent_id,
        nonce,
        chain,
        current,
    })
}

/// Build a behavioral exchange response.
pub fn build_behavioral_response(
    request_nonce: [u8; 32],
    own_key: &SigningKey,
    verdict: Verdict,
    chain: Vec<BehavioralAttestation>,
    current: BehavioralAttestation,
    reason: String,
) -> Result<BehavioralExchangeResponse, WireError> {
    let agent_id = compute_agent_id(&own_key.verifying_key());

    Ok(BehavioralExchangeResponse {
        agent_id,
        nonce: request_nonce,
        verdict: verdict.to_byte(),
        chain,
        current,
        reason,
    })
}

/// Validate a behavioral exchange request against the trust registry.
pub fn validate_behavioral_request(
    req: &BehavioralExchangeRequest,
    registry: &TrustRegistry,
) -> Result<(Verdict, String), WireError> {
    // Check agent is known
    let entry = registry.agents.get(&req.agent_id).ok_or_else(|| {
        WireError::UnknownAgent(hex_encode(&req.agent_id))
    })?;

    // Check agent has behavioral-observer role
    if !entry.roles.contains(&BEHAVIORAL_OBSERVER_ROLE.to_string()) {
        return Ok((
            Verdict::Rejected,
            format!(
                "agent '{}' not authorised for role '{}'",
                entry.name, BEHAVIORAL_OBSERVER_ROLE
            ),
        ));
    }

    // Verify the current attestation's signature
    verify_attestation(&req.current, &entry.public_key).map_err(|_| {
        WireError::Protocol("behavioral attestation signature invalid".into())
    })?;

    // Verify chain integrity
    verify_behavioral_chain(&req.chain, &req.current)?;

    Ok((Verdict::Accepted, "behavioral attestation accepted".into()))
}

/// Verify that a behavioral attestation chain is properly linked.
fn verify_behavioral_chain(
    chain: &[BehavioralAttestation],
    current: &BehavioralAttestation,
) -> Result<(), WireError> {
    // Verify parent hash links
    let mut prev_hash: Option<[u8; 32]> = None;

    for att in chain {
        if att.parent_hash != prev_hash {
            return Err(WireError::Chain(
                "behavioral chain parent hash mismatch".into(),
            ));
        }
        prev_hash = Some(attestation_hash(att));
    }

    // Current must link to last chain element (or be first if chain is empty)
    if current.parent_hash != prev_hash {
        return Err(WireError::Chain(
            "current behavioral attestation does not link to chain".into(),
        ));
    }

    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use got_proxy::attestation::{
        sign_attestation, AttestationSummary, AttestationType, BEHAVIORAL_SCHEMA_VERSION,
    };

    fn make_signed_attestation(
        sk: &SigningKey,
        seq: u64,
        parent: Option<[u8; 32]>,
    ) -> BehavioralAttestation {
        let att = BehavioralAttestation {
            schema_version: BEHAVIORAL_SCHEMA_VERSION.into(),
            target_model_id: "test-model".into(),
            reference_geometry_hash: [0xAA; 32],
            attestation_type: AttestationType::Snapshot,
            observation_count: seq * 10,
            sequence_number: seq,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            value_space_hash: [seq as u8; 32],
            parent_hash: parent,
            summary: AttestationSummary {
                top_values: vec![("honesty".into(), 0.8)],
                coherence_score: 0.9,
                cumulative_drift: 0.0,
            },
            deviation: None,
            signature: [0; 64],
        };
        sign_attestation(att, sk).unwrap()
    }

    #[test]
    fn build_and_validate_chain() {
        let sk = SigningKey::from_bytes(&[42u8; 32]);

        let att1 = make_signed_attestation(&sk, 1, None);
        let hash1 = attestation_hash(&att1);
        let att2 = make_signed_attestation(&sk, 2, Some(hash1));

        // Chain verification should pass
        assert!(verify_behavioral_chain(&[att1], &att2).is_ok());
    }

    #[test]
    fn broken_chain_rejected() {
        let sk = SigningKey::from_bytes(&[42u8; 32]);

        let att1 = make_signed_attestation(&sk, 1, None);
        // att2 claims wrong parent
        let att2 = make_signed_attestation(&sk, 2, Some([0xFF; 32]));

        assert!(verify_behavioral_chain(&[att1], &att2).is_err());
    }

    #[test]
    fn build_request_success() {
        let sk = SigningKey::from_bytes(&[42u8; 32]);
        let att = make_signed_attestation(&sk, 1, None);

        let req = build_behavioral_request([0u8; 32], &sk, vec![], att).unwrap();
        assert_eq!(req.agent_id, compute_agent_id(&sk.verifying_key()));
    }
}
