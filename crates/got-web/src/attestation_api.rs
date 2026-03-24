// ---------------------------------------------------------------------------
// Attestation API: real probe readings + signed geometric attestations.
//
// POST /api/attest/read   — Run probes on a message's hidden state
// POST /api/attest/sign   — Produce a signed attestation from readings
// POST /api/attest/verify — Verify a signed attestation
// ---------------------------------------------------------------------------

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use got_core::{GeometricAttestation, InnerProduct, Precision, SCHEMA_VERSION};
use got_probe::read_probe;

use crate::{api_err, ApiError, AppState};

// ---------------------------------------------------------------------------
// Shared attestation state (accumulated readings for signing)
// ---------------------------------------------------------------------------

/// Accumulated probe readings for producing an attestation.
pub struct AttestationState {
    pub readings_per_layer: Vec<Vec<f32>>,   // [layer][dim] raw scores
    pub confidences: Vec<f32>,                // per-dimension confidence
    pub coverage_flags: Vec<bool>,            // per-dimension coverage
    pub observation_count: u64,
    pub last_attestation_hash: Option<[u8; 32]>,
}

impl AttestationState {
    pub fn new() -> Self {
        Self {
            readings_per_layer: Vec::new(),
            confidences: Vec::new(),
            coverage_flags: Vec::new(),
            observation_count: 0,
            last_attestation_hash: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ReadRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ReadResponse {
    pub readings: Vec<ProbeReading>,
    pub layer: usize,
    pub source: String,
}

#[derive(Debug, Serialize)]
pub struct ProbeReading {
    pub dimension: String,
    pub raw: f32,
    pub confidence: f32,
    pub coverage_flag: bool,
}

#[derive(Debug, Deserialize)]
pub struct SignRequest {
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SignResponse {
    pub attestation_hash: String,
    pub signature: String,
    pub schema_version: u16,
    pub observation_count: u64,
    pub readings_count: usize,
    pub divergence_flag: bool,
}

#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    pub attestation_json: String,
    pub public_key_hex: String,
}

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub error: Option<String>,
}


// ---------------------------------------------------------------------------
// POST /api/attest/read — Run probes on a message's hidden state
// ---------------------------------------------------------------------------

pub async fn attest_read(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReadRequest>,
) -> Result<Json<ReadResponse>, (StatusCode, Json<ApiError>)> {
    let probe_set = state.probe_set.as_ref()
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, "no probes loaded (start with --probes)"))?;

    // Get hidden state from activation server
    let activation_url = state.activation_server_url.as_ref()
        .ok_or_else(|| api_err(StatusCode::SERVICE_UNAVAILABLE, "no activation server configured"))?;

    let embedding = crate::embed_api::embed_via_activation_server(&req.text, activation_url)
        .await
        .map_err(|e| api_err(StatusCode::BAD_GATEWAY, format!("activation server: {e}")))?;

    let h = &embedding.embedding;

    // Run each probe
    let mut readings = Vec::new();
    for probe in &probe_set.probes {
        match read_probe(probe, h, &state.geometry) {
            Ok((raw, confidence, coverage_flag)) => {
                readings.push(ProbeReading {
                    dimension: probe.dimension_name.clone(),
                    raw,
                    confidence,
                    coverage_flag,
                });
            }
            Err(e) => {
                return Err(api_err(StatusCode::INTERNAL_SERVER_ERROR,
                    format!("probe '{}': {e}", probe.dimension_name)));
            }
        }
    }

    // Accumulate readings for attestation signing
    if let Some(ref attest_state) = state.attestation_state {
        let mut ast = attest_state.lock().await;
        let layer_readings: Vec<f32> = readings.iter().map(|r| r.raw).collect();
        ast.readings_per_layer.push(layer_readings);
        ast.confidences = readings.iter().map(|r| r.confidence).collect();
        ast.coverage_flags = readings.iter().map(|r| r.coverage_flag).collect();
        ast.observation_count += 1;
    }

    Ok(Json(ReadResponse {
        readings,
        layer: probe_set.layer,
        source: "probe".into(),
    }))
}

// ---------------------------------------------------------------------------
// POST /api/attest/sign — Produce a signed attestation
// ---------------------------------------------------------------------------

pub async fn attest_sign(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SignRequest>,
) -> Result<Json<SignResponse>, (StatusCode, Json<ApiError>)> {
    let probe_set = state.probe_set.as_ref()
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, "no probes loaded"))?;
    let signing_key = state.signing_key.as_ref()
        .ok_or_else(|| api_err(StatusCode::INTERNAL_SERVER_ERROR, "no signing key"))?;
    let attest_state_mutex = state.attestation_state.as_ref()
        .ok_or_else(|| api_err(StatusCode::INTERNAL_SERVER_ERROR, "no attestation state"))?;

    let mut ast = attest_state_mutex.lock().await;

    if ast.readings_per_layer.is_empty() {
        return Err(api_err(StatusCode::BAD_REQUEST, "no readings accumulated — send messages first"));
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let model_id = req.model_id.unwrap_or_else(|| "interactive".into());
    let divergence_flag = ast.coverage_flags.iter().any(|&f| f);

    let attestation = GeometricAttestation {
        schema_version: SCHEMA_VERSION,
        model_id,
        model_hash: None,
        precision: Precision::Fp32,
        inner_product: InnerProduct::Euclidean,
        input_hash: [0; 32], // no deterministic input file in interactive mode
        timestamp: now,
        corpus_version: "interactive".into(),
        probe_version: probe_set.version.clone(),
        layer_readings: ast.readings_per_layer.clone(),
        confidence: ast.confidences.clone(),
        coverage_flags: ast.coverage_flags.clone(),
        divergence_flag,
        parent_attestation_hash: ast.last_attestation_hash,
        geometry_hash: Some(state.geometry.geometry_hash()),
        geometry_drift: None,
        causal_scores: Vec::new(),
        intervention_delta: None,
        causal_flag: None,
        sequence_number: ast.observation_count,
        directional_drifts: Vec::new(),
        probe_commitment: None,
        density_reading: None,
        curvature_reading: None,
        signature: [0; 64],
    };

    let signed = got_attest::assemble_and_sign(attestation, signing_key)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("sign: {e}")))?;

    let hash = got_attest::attestation_hash(&signed)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, format!("hash: {e}")))?;

    let readings_count = ast.readings_per_layer.len();

    // Chain: next attestation references this one
    ast.last_attestation_hash = Some(hash);
    // Reset readings for next batch
    ast.readings_per_layer.clear();

    Ok(Json(SignResponse {
        attestation_hash: crate::hex_encode(&hash),
        signature: crate::hex_encode(&signed.signature),
        schema_version: signed.schema_version,
        observation_count: ast.observation_count,
        readings_count,
        divergence_flag: signed.divergence_flag,
    }))
}

// ---------------------------------------------------------------------------
// POST /api/attest/verify — Verify a signed attestation
// ---------------------------------------------------------------------------

pub async fn attest_verify(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (StatusCode, Json<ApiError>)> {
    let attestation: GeometricAttestation = serde_json::from_str(&req.attestation_json)
        .map_err(|e| api_err(StatusCode::BAD_REQUEST, format!("invalid attestation JSON: {e}")))?;

    // Parse public key from hex
    let pk_bytes: Vec<u8> = (0..req.public_key_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&req.public_key_hex[i..i + 2], 16))
        .collect::<Result<Vec<u8>, _>>()
        .map_err(|e| api_err(StatusCode::BAD_REQUEST, format!("invalid public key hex: {e}")))?;

    if pk_bytes.len() != 32 {
        return Err(api_err(StatusCode::BAD_REQUEST, "public key must be 32 bytes"));
    }

    let vk = ed25519_dalek::VerifyingKey::from_bytes(
        pk_bytes.as_slice().try_into().unwrap(),
    ).map_err(|e| api_err(StatusCode::BAD_REQUEST, format!("invalid public key: {e}")))?;

    match got_attest::verify(&attestation, &vk) {
        Ok(()) => Ok(Json(VerifyResponse { valid: true, error: None })),
        Err(e) => Ok(Json(VerifyResponse { valid: false, error: Some(e.to_string()) })),
    }
}

// ---------------------------------------------------------------------------
// GET /api/attest/status — Current attestation state
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct AttestStatusResponse {
    pub probes_loaded: bool,
    pub probe_count: usize,
    pub probe_dimensions: Vec<String>,
    pub probe_layer: usize,
    pub accumulated_readings: usize,
    pub observation_count: u64,
    pub public_key: Option<String>,
    pub has_activation_server: bool,
}

pub async fn attest_status(
    State(state): State<Arc<AppState>>,
) -> Json<AttestStatusResponse> {
    let (accumulated, obs_count) = if let Some(ref ast) = state.attestation_state {
        let ast = ast.lock().await;
        (ast.readings_per_layer.len(), ast.observation_count)
    } else {
        (0, 0)
    };

    let (probe_count, dimensions, layer) = if let Some(ref ps) = state.probe_set {
        (
            ps.probes.len(),
            ps.probes.iter().map(|p| p.dimension_name.clone()).collect(),
            ps.layer,
        )
    } else {
        (0, Vec::new(), 0)
    };

    Json(AttestStatusResponse {
        probes_loaded: state.probe_set.is_some(),
        probe_count,
        probe_dimensions: dimensions,
        probe_layer: layer,
        accumulated_readings: accumulated,
        observation_count: obs_count,
        public_key: state.verifying_key_hex.clone(),
        has_activation_server: state.activation_server_url.is_some(),
    })
}
