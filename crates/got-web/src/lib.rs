pub mod api;
pub mod demo;

use std::collections::HashMap;

use got_core::geometry::CausalGeometry;
use got_incoherence::coherence::CoherenceConfig;
use got_incoherence::embeddings::EmbeddingSource;

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
}
