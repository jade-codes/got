// ---------------------------------------------------------------------------
// got-net::client — initiator side of the TCP exchange.
//
// Mirrors the in-memory `perform_exchange` flow over a real socket:
//   1. Open a TCP connection to the responder.
//   2. Run Noise NK as the initiator, binding to the responder's
//      Ed25519 verifying key looked up from the trust registry.
//   3. Build a signed `ExchangeRequest` and send it through the
//      encrypted session.
//   4. Receive the `ExchangeResponse`, validate it through
//      `validate_response`, and return the resulting `(verdict, reason,
//      response)` triple to the caller.
//
// Both a sync `request_blocking` (for callers that already own a
// blocking thread or are not running tokio) and an async `request`
// (which wraps the blocking variant in `spawn_blocking`) are exposed.
// ---------------------------------------------------------------------------

use std::net::{SocketAddr, TcpStream};

use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::RngCore;

use got_core::GeometricAttestation;
use got_wire::exchange::{
    build_request, check_domain_before_exchange, validate_response, ExchangeResponse, Verdict,
};
use got_wire::noise::{noise_connect_ed25519, NoiseSession};
use got_wire::registry::{compute_agent_id, TrustRegistry};

use crate::codec::{decode_exchange_response, encode_exchange_request};
use crate::error::NetError;
use crate::transport::TcpTransport;

/// Result of a single exchange round-trip.
#[derive(Debug)]
pub struct ExchangeOutcome {
    pub verdict: Verdict,
    pub reason: String,
    pub response: ExchangeResponse,
}

/// Inputs to one exchange request.
pub struct RequestParams {
    /// Initiator's signing key (this client).
    pub signing_key: SigningKey,
    /// Responder's verifying key — used both as the Noise NK static key
    /// and to derive the responder's agent ID for envelope addressing.
    pub responder_vk: VerifyingKey,
    /// Initiator's attestation chain (oldest first; may be empty).
    pub chain: Vec<GeometricAttestation>,
    /// Initiator's current attestation.
    pub current: GeometricAttestation,
}

/// Connect to `addr`, perform a single exchange, and return the result.
///
/// Synchronous: blocks the calling thread for the duration of the
/// connect + handshake + exchange.  Use [`request`] from an async
/// context.
pub fn request_blocking(
    addr: SocketAddr,
    params: RequestParams,
    registry: &TrustRegistry,
) -> Result<ExchangeOutcome, NetError> {
    let stream = TcpStream::connect(addr)?;
    request_on_stream(stream, params, registry)
}

/// Same as [`request_blocking`] but takes an already-connected stream.
/// Useful for testing or for callers that want to set socket options
/// (timeouts, TCP_NODELAY) before the handshake begins.
pub fn request_on_stream(
    stream: TcpStream,
    params: RequestParams,
    registry: &TrustRegistry,
) -> Result<ExchangeOutcome, NetError> {
    let RequestParams {
        signing_key,
        responder_vk,
        chain,
        current,
    } = params;

    // Phase 0: TCP connect + Noise NK handshake (already done by caller
    // for request_on_stream; request_blocking does both).
    let transport = TcpTransport::new(stream);
    let mut session: NoiseSession<TcpTransport> =
        noise_connect_ed25519(transport, &responder_vk)?;

    // Phase 1: domain compatibility pre-flight.  Runs BEFORE any
    // attestation work (geometry, probes, signing).  If the two agents
    // are domain-incompatible, the connection closes immediately
    // without wasting compute.
    let initiator_id = compute_agent_id(&signing_key.verifying_key());
    let responder_id = compute_agent_id(&responder_vk);
    check_domain_before_exchange(&initiator_id, &responder_id, registry)?;

    // Phase 2 (self-attest) is the caller's responsibility — they
    // pass in the already-signed `current` attestation.

    // Phase 3: build and send the exchange request.
    let mut nonce = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    let request = build_request(nonce, responder_id, &signing_key, chain, current)?;

    // 3. Send through the encrypted session.
    let req_bytes = encode_exchange_request(&request)?;
    session.send_encrypted(&req_bytes)?;

    // 4. Receive and decode the response.
    let rsp_bytes = session.recv_encrypted()?;
    let response = decode_exchange_response(&rsp_bytes)?;

    // Phase 4: validate the response against our local registry.
    //    Re-runs domain compat (defence in depth), envelope, attestation,
    //    governance, and scope-binding checks — we don't trust the
    //    server's verdict blindly.
    let (our_verdict, our_reason) =
        validate_response(&response, &initiator_id, &nonce, registry)?;

    // The verdict we surface to the caller is the *intersection*: if
    // either side rejected, the exchange has failed.  We surface the
    // server's reason if present, otherwise our own.
    let verdict = match (response.verdict, our_verdict) {
        (Verdict::Accepted, Verdict::Accepted) => Verdict::Accepted,
        (Verdict::Error, _) | (_, Verdict::Error) => Verdict::Error,
        _ => Verdict::Rejected,
    };
    let reason = if !response.reason.is_empty() {
        response.reason.clone()
    } else {
        our_reason
    };

    Ok(ExchangeOutcome {
        verdict,
        reason,
        response,
    })
}

/// Async wrapper around [`request_blocking`].  Spawns the blocking work
/// onto tokio's blocking thread pool so callers can `.await` from an
/// async context without bridging the sync Noise handshake themselves.
pub async fn request(
    addr: SocketAddr,
    params: RequestParams,
    registry: TrustRegistry,
) -> Result<ExchangeOutcome, NetError> {
    tokio::task::spawn_blocking(move || request_blocking(addr, params, &registry))
        .await
        .map_err(|e| NetError::Join(e.to_string()))?
}
