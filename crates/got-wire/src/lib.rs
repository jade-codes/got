// ---------------------------------------------------------------------------
// GOT/1 Wire Protocol — Phase 10.
//
// Binary protocol for agent-to-agent attestation exchange.
// Implements: frame codec, signed exchange envelopes, trust registry,
// chain verification, and the full exchange protocol logic.
//
// The Noise NK transport layer is defined in the plan but intentionally
// left as a trait so the protocol logic can be tested with in-memory
// transports.  A real deployment would plug in `snow` + TCP.
// ---------------------------------------------------------------------------

pub mod behavioral;
pub mod certificate;
pub mod chain;
pub mod envelope;
pub mod exchange;
pub mod frame;
pub mod noise;
pub mod registry;

use thiserror::Error;

/// Wire protocol errors.
#[derive(Debug, Error)]
pub enum WireError {
    #[error("bad magic: expected GOT1 (0x474F5431), got {0:#010x}")]
    BadMagic(u32),
    #[error("unknown message type: 0x{0:02x}")]
    UnknownMessageType(u8),
    #[error("payload too large: {size} bytes exceeds limit {limit}")]
    PayloadTooLarge { size: u32, limit: u32 },
    #[error("incomplete frame: needed {needed} bytes, got {got}")]
    IncompleteFrame { needed: usize, got: usize },
    #[error("envelope signature invalid")]
    EnvelopeSignatureInvalid,
    #[error("nonce mismatch")]
    NonceMismatch,
    #[error("peer agent ID mismatch: expected {expected}, got {got}")]
    PeerIdMismatch { expected: String, got: String },
    #[error("attestation hash mismatch")]
    AttestationHashMismatch,
    #[error("chain root hash mismatch")]
    ChainRootHashMismatch,
    #[error("timestamp too old: age {age_secs}s exceeds max {max_secs}s")]
    TimestampExpired { age_secs: u64, max_secs: u64 },
    #[error("unknown agent ID: {0}")]
    UnknownAgent(String),
    #[error("attestation error: {0}")]
    Attestation(#[from] got_attest::AttestationError),
    #[error("chain error: {0}")]
    Chain(String),
    #[error("registry parse error: {0}")]
    RegistryParse(String),
    #[error("registry integrity check failed: expected {expected}, got {actual}")]
    RegistryIntegrity { expected: String, actual: String },
    #[error("certificate signature invalid")]
    CertificateSignatureInvalid,
    #[error("certificate expired at timestamp {now} (valid {not_before}–{not_after})")]
    CertificateExpired {
        now: u64,
        not_before: u64,
        not_after: u64,
    },
    #[error("certificate subject key does not match expected key")]
    CertificateSubjectMismatch,
    #[error("certificate not issued by a known CA")]
    CertificateUnknownIssuer,
    #[error("certificate has been revoked")]
    CertificateRevoked,
    #[error("key rotation: old key signature invalid")]
    RotationOldSignatureInvalid,
    #[error("key rotation: new key signature invalid")]
    RotationNewSignatureInvalid,
    #[error("CRL signature invalid")]
    CrlSignatureInvalid,
    #[error("CRL issuer does not match expected CA key")]
    CrlIssuerMismatch,
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("io error: {0}")]
    Io(String),
}

/// Maximum payload size (16 MiB).
pub const MAX_PAYLOAD_SIZE: u32 = 16 * 1024 * 1024;

/// GOT1 magic bytes: 0x474F5431.
pub const MAGIC: u32 = 0x474F5431;
