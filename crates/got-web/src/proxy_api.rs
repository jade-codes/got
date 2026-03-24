// ---------------------------------------------------------------------------
// Proxy API endpoints for closed-source model value monitoring.
//
// POST /api/proxy/session            — Create a proxy session
// POST /api/proxy/session/:id/observe — Submit an observation
// GET  /api/proxy/session/:id/status  — Value space summary + deviation
// GET  /api/proxy/session/:id/history — Deviation history
// POST /api/proxy/session/:id/snapshot — Force snapshot + attestation
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use ed25519_dalek::SigningKey;
use got_proxy::attestation::AttestationType;
use got_proxy::config::ProxyConfig;
use got_proxy::deviation::{DeviationReport, DeviationVerdict};
use got_proxy::session::ProxySession;
use got_proxy::store::MemoryValueSpaceStore;
use got_incoherence::embeddings::PrecomputedEmbeddings;

use crate::AppState;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Per-session embedding config for calling the external embedding model.
#[derive(Debug, Clone)]
pub struct SessionEmbeddingConfig {
    pub url: String,
    pub model: String,
}

/// Shared proxy state: sessions keyed by session ID.
pub struct ProxyState {
    pub sessions: Mutex<HashMap<String, ProxySession<MemoryValueSpaceStore, PrecomputedEmbeddings>>>,
    /// Per-session embedding config (for text → embedding conversion).
    pub embedding_configs: Mutex<HashMap<String, SessionEmbeddingConfig>>,
}

impl ProxyState {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            embedding_configs: Mutex::new(HashMap::new()),
        }
    }
}

/// Embed text via an Ollama-compatible /api/embeddings endpoint.
async fn embed_text_via_api(text: &str, config: &SessionEmbeddingConfig) -> Result<Vec<f32>, String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": config.model,
        "prompt": text,
    });

    let resp = client
        .post(format!("{}/api/embeddings", config.url))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("embedding request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("embedding HTTP {status}: {text}"));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("parse embedding response: {e}"))?;

    json["embedding"]
        .as_array()
        .ok_or_else(|| "no embedding array in response".to_string())?
        .iter()
        .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| "non-numeric embedding value".to_string()))
        .collect()
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub session_id: Option<String>,
    pub target_model_id: String,
    /// Ollama-compatible embedding endpoint URL (e.g. "http://localhost:11434").
    /// When provided, the proxy embeds text internally for observe() calls.
    pub embedding_url: Option<String>,
    /// Embedding model name (e.g. "nomic-embed-text"). Required with embedding_url.
    pub embedding_model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub target_model_id: String,
    pub reference_geometry_hash: String,
}

#[derive(Debug, Deserialize)]
pub struct ObserveRequest {
    /// The text to observe. The proxy embeds this via the session's embedding endpoint.
    /// Provide either `text` or `embedding` — if both are given, `embedding` takes priority.
    pub text: Option<String>,
    /// Pre-computed embedding vector (for demo replay or pre-embedded content).
    pub embedding: Option<Vec<f32>>,
    /// Speaker ID: "assistant" for the model, "user" for the human.
    /// Defaults to "assistant" if omitted (backward compatible).
    #[serde(default = "default_speaker")]
    pub speaker: String,
}

fn default_speaker() -> String {
    "assistant".to_string()
}

#[derive(Debug, Serialize)]
pub struct ObserveResponse {
    pub observation_count: u64,
    pub speaker: String,
    pub detected_values: Vec<DetectedValueResponse>,
    pub deviation: Option<DeviationResponse>,
}

#[derive(Debug, Serialize)]
pub struct DetectedValueResponse {
    pub term: String,
    pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct DeviationResponse {
    pub term_score: f64,
    pub profile_drift: f64,
    pub relationship_score: f64,
    pub manifold_density_score: f64,
    pub combined_score: f64,
    pub verdict: String,
    pub baseline_sufficient: bool,
}

#[derive(Debug, Serialize)]
pub struct SessionStatusResponse {
    pub session_id: String,
    pub target_model_id: String,
    pub observation_count: u64,
    pub value_space_version: u64,
    pub top_values: Vec<(String, f64)>,
    pub latest_deviation: Option<DeviationResponse>,
    pub attestation_count: u64,
}

#[derive(Debug, Serialize)]
pub struct HistoryResponse {
    pub session_id: String,
    pub deviations: Vec<DeviationResponse>,
}

#[derive(Debug, Deserialize)]
pub struct SnapshotRequest {
    pub attestation_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SnapshotResponse {
    pub attestation_hash: String,
    pub sequence_number: u64,
    pub observation_count: u64,
    pub attestation_type: String,
    /// Manifold density summary, if sufficient activations were collected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifold_density: Option<ManifoldSummary>,
    /// Manifold curvature summary, if sufficient activations were collected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifold_curvature: Option<CurvatureSummary>,
    /// Per-term log-density on the activation manifold.
    /// Maps term name → log-density. Empty if insufficient activations.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub term_densities: HashMap<String, f32>,
}

#[derive(Debug, Serialize)]
pub struct ManifoldSummary {
    pub mean_intrinsic_dim: f32,
    pub std_intrinsic_dim: f32,
    pub mean_log_density: f32,
    pub num_points: usize,
    pub num_degenerate: u32,
}

#[derive(Debug, Serialize)]
pub struct CurvatureSummary {
    pub mean_curvature: f32,
    pub std_curvature: f32,
    pub num_points: usize,
    pub num_degenerate: u32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn deviation_to_response(d: &DeviationReport) -> DeviationResponse {
    DeviationResponse {
        term_score: d.term_score,
        profile_drift: d.profile_drift,
        relationship_score: d.relationship_score,
        manifold_density_score: d.manifold_density_score,
        combined_score: d.combined_score,
        verdict: match d.verdict {
            DeviationVerdict::WithinBaseline => "within_baseline".into(),
            DeviationVerdict::Drifting => "drifting".into(),
            DeviationVerdict::Deviated => "deviated".into(),
        },
        baseline_sufficient: d.baseline_sufficient,
    }
}

fn parse_attestation_type(s: Option<&str>) -> AttestationType {
    match s {
        Some("baseline") => AttestationType::Baseline,
        Some("alert") => AttestationType::Alert,
        Some("session_start") => AttestationType::SessionStart,
        _ => AttestationType::Snapshot,
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/proxy/session
pub async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, (StatusCode, Json<crate::ApiError>)> {
    let session_id = req
        .session_id
        .unwrap_or_else(|| format!("proxy-{}", rand::random::<u64>()));

    let sk = SigningKey::generate(&mut rand::thread_rng());

    // Determine embedding source: external API or reference model fallback.
    let (source, embedding_config, hidden_dim) =
        if let (Some(ref url), Some(ref model)) = (&req.embedding_url, &req.embedding_model) {
            let emb_config = SessionEmbeddingConfig {
                url: url.clone(),
                model: model.clone(),
            };

            // Embed all value term names (or descriptions from state) via the embedding API.
            let mut term_embeddings = HashMap::new();
            for term in &state.available_terms {
                match embed_text_via_api(term, &emb_config).await {
                    Ok(emb) => { term_embeddings.insert(term.clone(), emb); }
                    Err(e) => {
                        eprintln!("  warning: failed to embed term '{}': {e}", term);
                    }
                }
            }

            if term_embeddings.is_empty() {
                return Err(crate::api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "no value terms could be embedded via the embedding API",
                ));
            }

            let dim = term_embeddings.values().next().unwrap().len();
            let source = PrecomputedEmbeddings::new(term_embeddings)
                .map_err(|e| crate::api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("embedding source: {e}")))?;

            (source, Some(emb_config), dim)
        } else {
            // Fallback: use reference model embeddings from AppState.
            let source = PrecomputedEmbeddings::new(state.term_embeddings.clone())
                .map_err(|e| crate::api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("embedding source: {e}")))?;

            (source, None, state.hidden_dim)
        };

    let geometry = got_core::geometry::CausalGeometry::identity(hidden_dim);

    let session = ProxySession::new(
        session_id.clone(),
        req.target_model_id.clone(),
        sk,
        geometry,
        source,
        ProxyConfig::default(),
        MemoryValueSpaceStore::new(),
    )
    .map_err(|e| crate::api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("create session: {e}")))?;

    let geometry_hash = crate::hex_encode(&state.geometry.geometry_hash());

    state
        .proxy
        .sessions
        .lock()
        .await
        .insert(session_id.clone(), session);

    // Store embedding config if provided.
    if let Some(config) = embedding_config {
        state
            .proxy
            .embedding_configs
            .lock()
            .await
            .insert(session_id.clone(), config);
    }

    Ok(Json(CreateSessionResponse {
        session_id,
        target_model_id: req.target_model_id,
        reference_geometry_hash: geometry_hash,
    }))
}

/// POST /api/proxy/session/:id/observe
pub async fn observe(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<ObserveRequest>,
) -> Result<Json<ObserveResponse>, (StatusCode, Json<crate::ApiError>)> {
    // Resolve embedding: use pre-computed if provided, otherwise embed text.
    let embedding = if let Some(emb) = req.embedding {
        emb
    } else if let Some(ref text) = req.text {
        let configs = state.proxy.embedding_configs.lock().await;
        if let Some(config) = configs.get(&session_id) {
            embed_text_via_api(text, config)
                .await
                .map_err(|e| crate::api_err(StatusCode::BAD_GATEWAY, format!("embed: {e}")))?
        } else {
            let (emb, _, _) = crate::embed_text_bow(
                text, state.hidden_dim,
                state.vocab_lookup.as_ref(),
                state.embedding_source.as_ref(),
                &state.term_embeddings,
            );
            emb
        }
    } else {
        return Err(crate::api_err(StatusCode::BAD_REQUEST, "provide either 'text' or 'embedding'"));
    };

    let mut sessions = state.proxy.sessions.lock().await;
    let session = sessions
        .get_mut(&session_id)
        .ok_or_else(|| crate::api_err(StatusCode::NOT_FOUND, format!("session not found: {session_id}")))?;

    let result = session
        .observe(&embedding, &req.speaker)
        .map_err(|e| crate::api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("observe: {e}")))?;

    Ok(Json(ObserveResponse {
        observation_count: result.observation_count,
        speaker: result.speaker.clone(),
        detected_values: result
            .detected_values
            .iter()
            .map(|v| DetectedValueResponse {
                term: v.term.clone(),
                score: v.score,
            })
            .collect(),
        deviation: result.deviation.as_ref().map(deviation_to_response),
    }))
}

/// GET /api/proxy/session/:id/status
pub async fn session_status(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<SessionStatusResponse>, (StatusCode, Json<crate::ApiError>)> {
    let sessions = state.proxy.sessions.lock().await;
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| crate::api_err(StatusCode::NOT_FOUND, format!("session not found: {session_id}")))?;

    let status = session.status();
    Ok(Json(SessionStatusResponse {
        session_id: status.session_id,
        target_model_id: status.target_model_id,
        observation_count: status.observation_count,
        value_space_version: status.value_space_version,
        top_values: status.top_values,
        latest_deviation: status.latest_deviation.as_ref().map(deviation_to_response),
        attestation_count: status.attestation_count,
    }))
}

/// GET /api/proxy/session/:id/history
pub async fn deviation_history(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<HistoryResponse>, (StatusCode, Json<crate::ApiError>)> {
    let sessions = state.proxy.sessions.lock().await;
    let session = sessions
        .get(&session_id)
        .ok_or_else(|| crate::api_err(StatusCode::NOT_FOUND, format!("session not found: {session_id}")))?;

    let history = session.deviation_history();
    Ok(Json(HistoryResponse {
        session_id: session_id.clone(),
        deviations: history.iter().map(deviation_to_response).collect(),
    }))
}

/// POST /api/proxy/session/:id/manifold — attested manifold geometry.
///
/// Produces a signed Snapshot attestation that includes manifold density and
/// curvature readings, then returns the readings plus per-term densities.
/// Every response is backed by a verifiable Ed25519 signature.
pub async fn manifold(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<ManifoldResponse>, (StatusCode, Json<crate::ApiError>)> {
    let mut sessions = state.proxy.sessions.lock().await;
    let session = sessions
        .get_mut(&session_id)
        .ok_or_else(|| crate::api_err(StatusCode::NOT_FOUND, format!("session not found: {session_id}")))?;

    let (attestation, term_densities) = session
        .attest_manifold()
        .map_err(|e| crate::api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("attestation: {e}")))?;

    let hash = got_proxy::attestation::attestation_hash(&attestation);

    let manifold_density = attestation.density_reading.as_ref().map(|dr| ManifoldSummary {
        mean_intrinsic_dim: dr.mean_intrinsic_dim,
        std_intrinsic_dim: dr.std_intrinsic_dim,
        mean_log_density: dr.mean_log_density,
        num_points: dr.points.len(),
        num_degenerate: dr.num_degenerate,
    });

    let manifold_curvature = attestation.curvature_reading.as_ref().map(|cr| CurvatureSummary {
        mean_curvature: cr.mean_curvature,
        std_curvature: cr.std_curvature,
        num_points: cr.points.len(),
        num_degenerate: cr.num_degenerate,
    });

    // Get EWMA activation weights from the value space
    let term_weights: HashMap<String, f64> = session
        .value_space()
        .term_profiles
        .iter()
        .map(|(t, p)| (t.clone(), p.ewma))
        .collect();

    Ok(Json(ManifoldResponse {
        attestation_hash: crate::hex_encode(&hash),
        sequence_number: attestation.sequence_number,
        observation_count: attestation.observation_count,
        manifold_density,
        manifold_curvature,
        term_densities,
        term_weights,
    }))
}

#[derive(Debug, Serialize)]
pub struct ManifoldResponse {
    /// SHA-256 of the signed attestation backing this data.
    pub attestation_hash: String,
    /// Monotonic sequence number of the attestation.
    pub sequence_number: u64,
    pub observation_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifold_density: Option<ManifoldSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifold_curvature: Option<CurvatureSummary>,
    /// Per-term log-density on the activation manifold.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub term_densities: HashMap<String, f32>,
    /// Per-term EWMA activation weight (how strongly the model expresses each value).
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub term_weights: HashMap<String, f64>,
}

/// POST /api/proxy/session/:id/snapshot
pub async fn snapshot(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<SnapshotRequest>,
) -> Result<Json<SnapshotResponse>, (StatusCode, Json<crate::ApiError>)> {
    let mut sessions = state.proxy.sessions.lock().await;
    let session = sessions
        .get_mut(&session_id)
        .ok_or_else(|| crate::api_err(StatusCode::NOT_FOUND, format!("session not found: {session_id}")))?;

    let att_type = parse_attestation_type(req.attestation_type.as_deref());
    let attestation = session
        .snapshot_and_attest(att_type)
        .map_err(|e| crate::api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("attestation: {e}")))?;

    let hash = got_proxy::attestation::attestation_hash(&attestation);

    let manifold_density = attestation.density_reading.as_ref().map(|dr| ManifoldSummary {
        mean_intrinsic_dim: dr.mean_intrinsic_dim,
        std_intrinsic_dim: dr.std_intrinsic_dim,
        mean_log_density: dr.mean_log_density,
        num_points: dr.points.len(),
        num_degenerate: dr.num_degenerate,
    });

    let manifold_curvature = attestation.curvature_reading.as_ref().map(|cr| CurvatureSummary {
        mean_curvature: cr.mean_curvature,
        std_curvature: cr.std_curvature,
        num_points: cr.points.len(),
        num_degenerate: cr.num_degenerate,
    });

    let term_densities = session.term_densities().clone();

    Ok(Json(SnapshotResponse {
        attestation_hash: crate::hex_encode(&hash),
        sequence_number: attestation.sequence_number,
        observation_count: attestation.observation_count,
        attestation_type: format!("{:?}", attestation.attestation_type),
        manifold_density,
        manifold_curvature,
        term_densities,
    }))
}
