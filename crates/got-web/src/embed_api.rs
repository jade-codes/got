// ---------------------------------------------------------------------------
// Embed API: turn text into an embedding vector.
//
// POST /api/embed
//   Input:  { "text": "I believe in honesty and fairness" }
//   Output: { "embedding": [...], "dim": 4096, "matched_tokens": 6, "total_tokens": 6 }
//
// When an activation server is configured (--activation-server), sends text
// to the sidecar and gets back a real residual stream activation from an
// intermediate transformer layer. Falls back to bag-of-words unembedding
// lookup when no activation server is available.
// ---------------------------------------------------------------------------

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::{api_err, embed_text_bow, ApiError, AppState};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// POST /api/embed — generate an embedding vector from text.
pub async fn embed_text(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<ApiError>)> {
    let text = req.text.trim();
    if text.is_empty() {
        return Err(api_err(StatusCode::BAD_REQUEST, "text is empty"));
    }

    // Try activation server first (real residual stream activations)
    if let Some(ref url) = state.activation_server_url {
        match embed_via_activation_server(text, url).await {
            Ok(resp) => return Ok(Json(resp)),
            Err(e) => {
                eprintln!("Activation server error, falling back to bag-of-words: {e}");
            }
        }
    }

    // Fallback: bag-of-words averaging
    let (embedding, matched, total) = embed_text_bow(
        text,
        state.hidden_dim,
        state.vocab_lookup.as_ref(),
        state.embedding_source.as_ref(),
        &state.term_embeddings,
    );

    Ok(Json(EmbedResponse {
        embedding,
        dim: state.hidden_dim,
        matched_tokens: matched,
        total_tokens: total,
        source: Some("bag-of-words".into()),
    }))
}

/// Call the activation server sidecar to get a real hidden state.
pub async fn embed_via_activation_server(text: &str, url: &str) -> Result<EmbedResponse, String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({ "text": text });

    let resp = client
        .post(format!("{url}/hidden_states"))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {text}"));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("parse response: {e}"))?;

    let hidden_state = json["hidden_state"]
        .as_array()
        .ok_or("no hidden_state array")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect::<Vec<f32>>();

    let n_tokens = json["n_tokens"].as_u64().unwrap_or(0) as usize;
    let dim = hidden_state.len();

    Ok(EmbedResponse {
        embedding: hidden_state,
        dim,
        matched_tokens: n_tokens,
        total_tokens: n_tokens,
        source: Some("activation".into()),
    })
}
