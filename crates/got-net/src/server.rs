// ---------------------------------------------------------------------------
// got-net::server — TCP listener that accepts inbound exchange requests.
//
// The async tokio half of the architecture.  `serve` binds a listener,
// loops on `accept`, and dispatches each inbound connection to a blocking
// thread via `tokio::task::spawn_blocking`.  The actual per-connection
// work — Noise NK handshake, request decode, validate_request, response
// encode — runs synchronously inside that blocking thread, exactly the
// same code path the in-memory tests use.  No async/sync bridging
// happens inside the cryptography.
//
// The server is single-shot per connection: one Noise handshake, one
// ExchangeRequest in, one ExchangeResponse out, then the socket closes.
// This matches the protocol's exchange semantics — there is no notion
// of a long-lived "session" with multiple back-to-back attestations.
// A peer that wants to exchange again opens a new connection.
//
// `AttestationProvider` is a trait so callers can wire in whatever
// strategy they want for producing the server's own attestation
// (cached snapshot, lazy re-probe on each request, fetched from a
// hardware enclave, etc.).  The integration test uses a static
// implementation that hands out a fixed attestation.
// ---------------------------------------------------------------------------

use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;

use ed25519_dalek::SigningKey;

use got_core::GeometricAttestation;
use got_wire::exchange::{build_response, validate_request, Verdict};
use got_wire::noise::{noise_accept_ed25519, NoiseSession};
use got_wire::registry::{compute_agent_id, TrustRegistry};

use crate::codec::{decode_exchange_request, encode_exchange_response};
use crate::error::NetError;
use crate::transport::TcpTransport;

/// Strategy for producing the server's own attestation when responding
/// to an exchange request.
///
/// Implementations must be cheap to call — they run on a blocking
/// thread but a slow provider blocks the connection's reply.  Cache or
/// pre-compute as appropriate.
pub trait AttestationProvider: Send + Sync {
    /// Returns `(current_attestation, chain)`.  `chain` is the ordered
    /// list of historical attestations from the anchor through the
    /// most recent ancestor of `current`; an empty `chain` is fine for
    /// a fresh attestation.
    fn current(&self) -> (GeometricAttestation, Vec<GeometricAttestation>);
}

/// Trivial provider that returns the same `(current, chain)` tuple on
/// every call.  Used in tests and for fixed-attestation deployments.
pub struct StaticAttestationProvider {
    pub current: GeometricAttestation,
    pub chain: Vec<GeometricAttestation>,
}

impl AttestationProvider for StaticAttestationProvider {
    fn current(&self) -> (GeometricAttestation, Vec<GeometricAttestation>) {
        (self.current.clone(), self.chain.clone())
    }
}

/// Configuration for a running exchange server.
#[derive(Clone)]
pub struct ServerConfig {
    pub signing_key: Arc<SigningKey>,
    pub registry: Arc<TrustRegistry>,
    pub attestation: Arc<dyn AttestationProvider>,
}

/// Bind a tokio listener and accept inbound exchange connections until
/// the future is dropped.  Each connection is handled on a blocking
/// thread; this function only returns on a fatal `accept` error.
///
/// To run a server in the background, spawn this with
/// `tokio::spawn(serve(addr, config))` and `JoinHandle::abort()` it
/// when you want to stop accepting.
pub async fn serve(addr: SocketAddr, config: ServerConfig) -> Result<(), NetError> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    accept_loop(listener, config).await
}

/// Same as [`serve`] but takes a pre-bound listener.  Useful in tests
/// where you want to bind to `127.0.0.1:0` and read back the assigned
/// port before clients try to connect.
pub async fn accept_loop(
    listener: tokio::net::TcpListener,
    config: ServerConfig,
) -> Result<(), NetError> {
    loop {
        let (stream, _peer_addr) = listener.accept().await?;
        let config = config.clone();
        // Convert to a std::net::TcpStream so the sync Noise + exchange
        // path can use it directly.  set_nonblocking(false) is required
        // because tokio leaves the underlying fd in non-blocking mode.
        let std_stream = stream.into_std()?;
        std_stream.set_nonblocking(false)?;
        tokio::task::spawn_blocking(move || {
            // Errors here are per-connection — log them but keep the
            // accept loop running.  In a real deployment you would
            // wire this through `tracing` or whatever observability
            // layer the host uses.
            if let Err(e) = handle_connection(std_stream, &config) {
                eprintln!("got-net: connection handler error: {e}");
            }
        });
    }
}

/// Handle one inbound connection synchronously.  Performs the Noise NK
/// accept, decodes one `ExchangeRequest`, runs the registry-backed
/// validation, builds and encodes the matching `ExchangeResponse`, and
/// closes the socket.
///
/// Exposed publicly so callers that want to drive the server with their
/// own thread pool (or no pool at all) can skip [`accept_loop`] and
/// invoke this directly on each accepted stream.
pub fn handle_connection(stream: TcpStream, config: &ServerConfig) -> Result<(), NetError> {
    let transport = TcpTransport::new(stream);
    let mut session: NoiseSession<TcpTransport> =
        noise_accept_ed25519(transport, &config.signing_key)?;

    // 1. Receive and decode the request.
    let req_bytes = session.recv_encrypted()?;
    let request = decode_exchange_request(&req_bytes)?;

    // 2. Run the standard validation.  The verifier's own agent ID is
    //    derived from its signing key — that's the ID the initiator
    //    will have used to address the envelope.
    let own_agent_id = compute_agent_id(&config.signing_key.verifying_key());
    let (verdict, reason) =
        validate_request(&request, &own_agent_id, None, &config.registry)?;

    // 3. Build the response, signed with our own key.
    let (current, chain) = config.attestation.current();
    let response = build_response(
        request.envelope.nonce,
        request.agent_id,
        &config.signing_key,
        verdict,
        chain,
        current,
        reason,
    )?;

    // 4. Encode and send.
    let rsp_bytes = encode_exchange_response(&response)?;
    session.send_encrypted(&rsp_bytes)?;

    // The socket is dropped here — TCP FIN flushes the underlying
    // stream and closes the connection.
    Ok(())
}

/// Convenience constructor: a `ServerConfig` whose attestation provider
/// is a [`StaticAttestationProvider`].
pub fn static_config(
    signing_key: SigningKey,
    registry: TrustRegistry,
    current: GeometricAttestation,
    chain: Vec<GeometricAttestation>,
) -> ServerConfig {
    ServerConfig {
        signing_key: Arc::new(signing_key),
        registry: Arc::new(registry),
        attestation: Arc::new(StaticAttestationProvider { current, chain }),
    }
}

// ---------------------------------------------------------------------------
// Track which Verdict variants the response can carry — used by the
// caller to translate the server's signed verdict back into a Result.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub(crate) fn verdict_is_terminal(v: Verdict) -> bool {
    matches!(v, Verdict::Accepted | Verdict::Rejected | Verdict::Error)
}
