// ---------------------------------------------------------------------------
// got-net — concrete network transport for the GoT/1 wire protocol.
//
// The protocol's `got_wire::noise::Transport` trait is sync and message-
// oriented; this crate plugs that trait into a real TCP socket and adds
// a tokio-based listener that supervises connections.  The cryptography
// (Noise NK handshake, attestation signing, attestation verification) is
// unchanged from the in-memory test path — only the bytes-on-the-wire
// substrate is different.
//
// Architectural split:
//   * `transport` — sync `TcpTransport` over `std::net::TcpStream` that
//     implements `got_wire::noise::Transport` with length-prefixed
//     framing.  This is the *only* code in the crate that touches a
//     socket directly.
//   * `codec` — wire format for `ExchangeRequest` / `ExchangeResponse`
//     (32-byte agent_id + 200-byte envelope + length-prefixed JSON for
//     each `GeometricAttestation`).
//   * `server` — async tokio listener that calls `accept()` and hands
//     each inbound connection to `tokio::task::spawn_blocking`, where
//     the sync handshake + exchange runs without contaminating the
//     async runtime.
//   * `client` — async helpers that wrap the sync request flow in
//     `spawn_blocking` so callers from an async context can `.await`
//     the round-trip.
//
// The blocking-thread-per-connection model is deliberate.  Noise's
// `snow` is sync; bridging it into an async transport requires either
// running each handshake message through a `block_on` (which deadlocks
// inside a tokio worker) or rewriting the handshake state machine to
// be poll-based.  Since each connection only does one Noise handshake
// followed by a single message exchange, parking it on a blocking
// thread is the simplest correct answer.
// ---------------------------------------------------------------------------

pub mod client;
pub mod codec;
pub mod error;
pub mod server;
pub mod transport;

pub use error::NetError;
pub use transport::{TcpTransport, MAX_MESSAGE_SIZE};
