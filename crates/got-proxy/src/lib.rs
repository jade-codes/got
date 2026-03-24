// ---------------------------------------------------------------------------
// got-proxy: Proxy architecture for closed-source model value monitoring.
//
// Builds its own behavioral value space by embedding model outputs and
// value anchors through the same embedding model (e.g. Ollama), ensuring
// consistent measurement within a single embedding space. Uses absolute
// cosine similarity for value detection and Welford online statistics
// with EWMA recency weighting for drift detection.
//
// Trust tier: "Tier 0 — Behavioral" (weaker than geometric attestations,
// which have direct access to model internals).
// ---------------------------------------------------------------------------

pub mod attestation;
pub mod config;
pub mod deviation;
pub mod session;
pub mod store;
pub mod value_space;

use thiserror::Error;

/// Errors from the proxy subsystem.
#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("geometry error: {0}")]
    Geometry(#[from] got_core::geometry::GeometryError),

    #[error("incoherence error: {0}")]
    Incoherence(#[from] got_incoherence::IncoherenceError),

    #[error("signature verification failed")]
    SignatureInvalid,

    #[error("invalid schema version: {0}")]
    InvalidSchemaVersion(String),

    #[error("timestamp too far in the future ({delta}s > max {max}s)")]
    TimestampFuture { delta: u64, max: u64 },

    #[error("geometry mismatch: value space pinned to different reference geometry")]
    GeometryMismatch,

    #[error("session not found: {0}")]
    SessionNotFound(String),

    #[error("serialisation error: {0}")]
    Serialisation(String),

    #[error("IO error: {0}")]
    Io(String),
}
