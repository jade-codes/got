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
    /// "activation" if from the sidecar, "bag-of-words" if from vocab lookup.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
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

    // Try activation server first (real residual stream activations)
    if let Some(ref url) = state.activation_server_url {
        match embed_via_activation_server(text, url).await {
            Ok(resp) => return Ok(Json(resp)),
            Err(e) => {
                eprintln!("Activation server error, falling back to bag-of-words: {e}");
            }
        }
    }

    // Fallback: bag-of-words averaging from unembedding matrix rows
    let dim = state.hidden_dim;
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let total_tokens = tokens.len();

    let mut sum = vec![0.0f32; dim];
    let mut matched = 0usize;

    for token in &tokens {
        let lower = token.to_lowercase();
        let clean: String = lower
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
            .collect();
        if clean.is_empty() {
            continue;
        }

        let found = if let Some(ref lookup) = state.vocab_lookup {
            lookup.embed(&clean)
        } else {
            None
        };
        let found = found
            .or_else(|| state.embedding_source.embed(&clean))
            .or_else(|| state.term_embeddings.get(&clean).cloned());

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

    Ok(Json(EmbedResponse {
        embedding: sum,
        dim,
        matched_tokens: matched,
        total_tokens,
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
