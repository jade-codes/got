pub mod api;
pub mod attestation_api;
pub mod chat_api;
pub mod demo;
pub mod embed_api;
pub mod metrics_api;
pub mod proxy_api;

use std::collections::HashMap;

use ed25519_dalek::SigningKey;
use got_core::geometry::CausalGeometry;
use got_incoherence::coherence::CoherenceConfig;
use got_incoherence::embeddings::EmbeddingSource;
use got_probe::ProbeSet;
use tokio::sync::Mutex;

/// Full-vocabulary lookup that reads embedding rows on demand from the .gotue bytes.
pub struct VocabLookup {
    /// Token (lowercase, BPE-stripped) → row index in the unembedding matrix.
    pub index: HashMap<String, usize>,
    /// Raw .gotue file bytes (kept in memory for row lookups).
    pub data: Vec<u8>,
    /// Byte offset where the float data starts in the file.
    pub data_start: usize,
    pub hidden_dim: usize,
}

impl VocabLookup {
    /// Look up an embedding for a token. Returns None if not in vocabulary.
    pub fn embed(&self, token: &str) -> Option<Vec<f32>> {
        let clean = token.trim().to_lowercase();
        let row_idx = *self.index.get(&clean)?;
        let row_bytes = self.hidden_dim * 4;
        let start = self.data_start + row_idx * row_bytes;
        if start + row_bytes > self.data.len() {
            return None;
        }
        Some(
            self.data[start..start + row_bytes]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Hex-encode a byte slice.
pub fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Shared error response for all API endpoints.
#[derive(Debug, serde::Serialize)]
pub struct ApiError {
    pub error: String,
}

/// Build an axum error response.
pub fn api_err(
    status: axum::http::StatusCode,
    msg: impl Into<String>,
) -> (axum::http::StatusCode, axum::Json<ApiError>) {
    (status, axum::Json(ApiError { error: msg.into() }))
}

/// Clean a token for vocabulary lookup: lowercase, keep only alphanumeric + hyphens + apostrophes.
fn clean_token(token: &str) -> String {
    token
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
        .collect()
}

/// Embed text by averaging token embeddings from available sources.
///
/// Tries vocab_lookup first, then embedding_source, then term_embeddings.
/// Returns (embedding, matched_count, total_count).
pub fn embed_text_bow(
    text: &str,
    dim: usize,
    vocab_lookup: Option<&VocabLookup>,
    embedding_source: &dyn EmbeddingSource,
    term_embeddings: &HashMap<String, Vec<f32>>,
) -> (Vec<f32>, usize, usize) {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let total = tokens.len();
    let mut sum = vec![0.0f32; dim];
    let mut matched = 0usize;

    for token in &tokens {
        let clean = clean_token(token);
        if clean.is_empty() {
            continue;
        }

        let found = vocab_lookup
            .and_then(|vl| vl.embed(&clean))
            .or_else(|| embedding_source.embed(&clean))
            .or_else(|| term_embeddings.get(&clean).cloned());

        if let Some(emb) = found {
            for (s, e) in sum.iter_mut().zip(emb.iter()) {
                *s += e;
            }
            matched += 1;
        }
    }

    if matched > 0 {
        let scale = 1.0 / matched as f32;
        for s in sum.iter_mut() {
            *s *= scale;
        }
    }

    (sum, matched, total)
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

/// Shared application state, built at startup and shared across all handlers.
pub struct AppState {
    pub geometry: CausalGeometry,
    pub term_embeddings: HashMap<String, Vec<f32>>,
    pub embedding_source: Box<dyn EmbeddingSource + Send + Sync>,
    pub available_terms: Vec<String>,
    pub hidden_dim: usize,
    pub mode: String,
    pub demo_conversation_json: String,
    /// Default coherence config for this mode (thresholds calibrated at startup).
    pub default_config: CoherenceConfig,
    /// Minimum z-score to introduce a value into the cumulative set.
    pub introduction_threshold: f32,
    /// Proxy session state for closed-source model monitoring.
    pub proxy: proxy_api::ProxyState,
    /// Full-vocabulary lookup for embedding arbitrary text. None in synthetic mode.
    pub vocab_lookup: Option<VocabLookup>,
    /// URL of the activation server for intermediate-layer hidden states.
    /// When set, /api/embed routes through the sidecar instead of bag-of-words.
    pub activation_server_url: Option<String>,
    /// Pre-trained probe set for real geometric attestation.
    pub probe_set: Option<ProbeSet>,
    /// Ed25519 signing key for attestations (generated at startup).
    pub signing_key: Option<SigningKey>,
    /// Hex-encoded public key for verification.
    pub verifying_key_hex: Option<String>,
    /// Accumulated attestation state (readings, chaining).
    pub attestation_state: Option<Mutex<attestation_api::AttestationState>>,
}
