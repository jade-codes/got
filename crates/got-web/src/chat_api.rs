// ---------------------------------------------------------------------------
// Chat API: relay messages to a closed-source LLM and return the response.
//
// POST /api/chat
//   Input:  { provider, api_key, model, messages: [{role, content}] }
//   Output: { response: "..." }
//
// Supports OpenAI-compatible and Anthropic APIs.
// The API key is provided per-request from the frontend — never stored.
// ---------------------------------------------------------------------------

use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// "openai" or "anthropic"
    pub provider: String,
    /// API key (sent per-request, never stored)
    pub api_key: String,
    /// Model name (e.g. "gpt-4o", "claude-sonnet-4-20250514")
    pub model: String,
    /// Conversation messages
    pub messages: Vec<ChatMessage>,
    /// Optional base URL override (for local/custom endpoints)
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub response: String,
}

#[derive(Debug, Serialize)]
pub struct ChatErrorResponse {
    pub error: String,
}

fn chat_err(
    status: StatusCode,
    msg: impl Into<String>,
) -> (StatusCode, Json<ChatErrorResponse>) {
    (status, Json(ChatErrorResponse { error: msg.into() }))
}

/// POST /api/chat — relay to LLM provider and return response.
pub async fn chat(
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ChatErrorResponse>)> {
    if req.messages.is_empty() {
        return Err(chat_err(StatusCode::BAD_REQUEST, "messages cannot be empty"));
    }

    let response_text = match req.provider.as_str() {
        "openai" => call_openai(&req).await,
        "anthropic" => call_anthropic(&req).await,
        other => return Err(chat_err(
            StatusCode::BAD_REQUEST,
            format!("unknown provider: {other} (use 'openai' or 'anthropic')"),
        )),
    }
    .map_err(|e| chat_err(StatusCode::BAD_GATEWAY, format!("LLM API error: {e}")))?;

    Ok(Json(ChatResponse {
        response: response_text,
    }))
}

async fn call_openai(req: &ChatRequest) -> Result<String, String> {
    let base_url = req
        .base_url
        .as_deref()
        .unwrap_or("https://api.openai.com/v1");

    let body = serde_json::json!({
        "model": req.model,
        "messages": req.messages,
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .header("Authorization", format!("Bearer {}", req.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("read body: {e}"))?;

    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }

    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse response: {e}"))?;

    parsed["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("unexpected response format: {text}"))
}

async fn call_anthropic(req: &ChatRequest) -> Result<String, String> {
    let base_url = req
        .base_url
        .as_deref()
        .unwrap_or("https://api.anthropic.com/v1");

    // Anthropic separates system messages from the messages array
    let mut system_text = String::new();
    let mut messages = Vec::new();
    for msg in &req.messages {
        if msg.role == "system" {
            system_text.push_str(&msg.content);
        } else {
            messages.push(serde_json::json!({
                "role": msg.role,
                "content": msg.content,
            }));
        }
    }

    let mut body = serde_json::json!({
        "model": req.model,
        "max_tokens": 1024,
        "messages": messages,
    });
    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/messages"))
        .header("x-api-key", &req.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("read body: {e}"))?;

    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }

    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse response: {e}"))?;

    parsed["content"][0]["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("unexpected response format: {text}"))
}
