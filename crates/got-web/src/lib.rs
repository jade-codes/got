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
