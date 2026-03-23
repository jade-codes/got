// ---------------------------------------------------------------------------
// Embed API: turn text into an embedding vector using the reference model.
//
// POST /api/embed
//   Input:  { "text": "I believe in honesty and fairness" }
//   Output: { "embedding": [...], "dim": 32, "matched_tokens": 2, "total_tokens": 6 }
//
// Tokenizes by whitespace, looks up each token in the embedding source,
// and averages the found vectors. Unmatched tokens are skipped.
// ---------------------------------------------------------------------------

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
    pub dim: usize,
    pub matched_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbedErrorResponse {
    pub error: String,
}

/// POST /api/embed — generate an embedding vector from text.
pub async fn embed_text(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<EmbedErrorResponse>)> {
    let text = req.text.trim();
    if text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(EmbedErrorResponse {
                error: "text is empty".into(),
            }),
        ));
    }

    let dim = state.hidden_dim;

    // Tokenize by whitespace, normalize to lowercase
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let total_tokens = tokens.len();

    // Look up each token in the embedding source and accumulate
    let mut sum = vec![0.0f32; dim];
    let mut matched = 0usize;

    for token in &tokens {
        let lower = token.to_lowercase();
        // Strip common punctuation for better matching
        let clean: String = lower
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
            .collect();
        if clean.is_empty() {
            continue;
        }

        // Try the embedding source first (covers full vocab in real mode)
        if let Some(emb) = state.embedding_source.embed(&clean) {
            for (s, e) in sum.iter_mut().zip(emb.iter()) {
                *s += e;
            }
            matched += 1;
        }
        // Also check term_embeddings (always available, includes raw embeddings)
        else if let Some(emb) = state.term_embeddings.get(&clean) {
            for (s, e) in sum.iter_mut().zip(emb.iter()) {
                *s += e;
            }
            matched += 1;
        }
    }

    // Average if we got matches
    if matched > 0 {
        let scale = 1.0 / matched as f32;
        for s in sum.iter_mut() {
            *s *= scale;
        }
    }

    Ok(Json(EmbedResponse {
        embedding: sum,
        dim,
        matched_tokens: matched,
        total_tokens,
    }))
}
