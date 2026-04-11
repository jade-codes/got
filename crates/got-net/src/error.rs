// ---------------------------------------------------------------------------
// Error type for got-net.
// ---------------------------------------------------------------------------

use thiserror::Error;

#[derive(Debug, Error)]
pub enum NetError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("wire protocol error: {0}")]
    Wire(#[from] got_wire::WireError),
    #[error("attestation error: {0}")]
    Attestation(#[from] got_attest::AttestationError),
    #[error("json codec error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("wire codec error: {0}")]
    Codec(String),
    #[error("message too large: {size} bytes exceeds limit {limit}")]
    MessageTooLarge { size: usize, limit: usize },
    #[error("connection closed before {expected} bytes")]
    UnexpectedEof { expected: usize },
    #[error("unsupported message type: 0x{0:02x}")]
    UnsupportedMessageType(u8),
    #[error("server task join error: {0}")]
    Join(String),
}
