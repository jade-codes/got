// ---------------------------------------------------------------------------
// Store API: attestation storage, querying, and audit reports.
// ---------------------------------------------------------------------------

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use ed25519_dalek::VerifyingKey;
use got_core::GeometricAttestation;
use got_store::AttestationStore;
use serde::{Deserialize, Serialize};

use crate::AppState;

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct AttestationSummary {
    pub id: String,
    pub model_id: String,
    pub schema_version: u16,
    pub timestamp: u64,
    pub num_layers: usize,
    pub divergence_flag: bool,
    pub causal_flag: Option<bool>,
    pub geometry_drift: Option<f32>,
    pub sequence_number: u64,
}

#[derive(Debug, Serialize)]
pub struct AttestationDetail {
    pub id: String,
    pub attestation: GeometricAttestation,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub model_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ImportRequest {
    pub attestation: GeometricAttestation,
    /// Hex-encoded Ed25519 verifying key (32 bytes = 64 hex chars).
    pub verifying_key_hex: String,
}

#[derive(Debug, Serialize)]
pub struct ImportResponse {
    pub status: String,
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct StoreErrorResponse {
    pub error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn hex_id(id: &[u8; 32]) -> String {
    id.iter().map(|b| format!("{b:02x}")).collect()
}

fn summarise(id_bytes: [u8; 32], a: &GeometricAttestation) -> AttestationSummary {
    AttestationSummary {
        id: hex_id(&id_bytes),
        model_id: a.model_id.clone(),
        schema_version: a.schema_version,
        timestamp: a.timestamp,
        num_layers: a.layer_readings.len(),
        divergence_flag: a.divergence_flag,
        causal_flag: a.causal_flag,
        geometry_drift: a.geometry_drift,
        sequence_number: a.sequence_number,
    }
}

/// GET /api/attestations?model_id=...
///
/// List all attestations, optionally filtered by model_id.
pub async fn list_attestations(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Json<Vec<AttestationSummary>> {
    let lock = state.lock().unwrap();

    let filter = if let Some(ref mid) = query.model_id {
        got_store::StoreFilter::new().model_id(mid.clone())
    } else {
        got_store::StoreFilter::new()
    };

    let attestations = lock.store.query(&filter);

    let summaries: Vec<AttestationSummary> = attestations
        .into_iter()
        .filter_map(|a| {
            let id = got_store::store::attestation_store_id(a).ok()?;
            Some(summarise(id, a))
        })
        .collect();

    Json(summaries)
}

/// POST /api/attestations
///
/// Import a signed attestation into the store.
pub async fn import_attestation(
    State(state): State<AppState>,
    Json(req): Json<ImportRequest>,
) -> Result<Json<ImportResponse>, (StatusCode, Json<StoreErrorResponse>)> {
    // Parse verifying key from hex
    let key_bytes = parse_hex_32(&req.verifying_key_hex).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(StoreErrorResponse {
                error: format!("invalid verifying_key_hex: {e}"),
            }),
        )
    })?;
    let vk = VerifyingKey::from_bytes(&key_bytes).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(StoreErrorResponse {
                error: format!("invalid Ed25519 key: {e}"),
            }),
        )
    })?;

    let mut lock = state.lock().unwrap();
    let id = lock.store.append(&req.attestation, &vk).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(StoreErrorResponse {
                error: format!("{e}"),
            }),
        )
    })?;

    Ok(Json(ImportResponse {
        status: "ok".into(),
        id: hex_id(&id),
    }))
}

/// GET /api/audit/:model_id
///
/// Generate an audit report for the given model.
pub async fn audit_report(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Json<serde_json::Value> {
    let lock = state.lock().unwrap();
    let report = lock.store.audit(&model_id);
    // Serialize the audit report, converting signer hashes to hex
    Json(serde_json::json!({
        "model_id": report.model_id,
        "total_attestations": report.total_attestations,
        "chain_length": report.chain_length,
        "chain_valid": report.chain_valid,
        "first_timestamp": report.first_timestamp,
        "last_timestamp": report.last_timestamp,
        "schema_versions_seen": report.schema_versions_seen,
        "drift_summary": {
            "readings_with_drift": report.drift_summary.readings_with_drift,
            "max_drift": report.drift_summary.max_drift,
            "mean_drift": report.drift_summary.mean_drift,
        },
        "causal_summary": {
            "attestations_with_causal": report.causal_summary.attestations_with_causal,
            "causal_pass_count": report.causal_summary.causal_pass_count,
            "causal_fail_count": report.causal_summary.causal_fail_count,
            "mean_consistency": report.causal_summary.mean_consistency,
        },
        "signers": report.signers.iter().map(hex_id).collect::<Vec<_>>(),
    }))
}

/// POST /api/attestations/import-file
///
/// Accept a file upload containing the CLI attestation JSON output
/// (which embeds both the attestation and verifying_key_hex).
pub async fn import_attestation_file(
    State(state): State<AppState>,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<ImportResponse>, (StatusCode, Json<StoreErrorResponse>)> {
    let mut file_text: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(StoreErrorResponse {
            error: format!("multipart error: {e}"),
        })))?
    {
        let text = field
            .text()
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, Json(StoreErrorResponse {
                error: format!("failed to read field: {e}"),
            })))?;
        if !text.is_empty() {
            file_text = Some(text);
            break;
        }
    }

    let text = file_text.ok_or_else(|| (StatusCode::BAD_REQUEST, Json(StoreErrorResponse {
        error: "no file uploaded".into(),
    })))?;

    let req: ImportRequest = serde_json::from_str(&text).map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(StoreErrorResponse {
            error: format!("invalid attestation JSON: {e}"),
        }))
    })?;

    // Reuse the existing import logic
    let key_bytes = parse_hex_32(&req.verifying_key_hex).map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(StoreErrorResponse {
            error: format!("invalid verifying_key_hex: {e}"),
        }))
    })?;
    let vk = VerifyingKey::from_bytes(&key_bytes).map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(StoreErrorResponse {
            error: format!("invalid Ed25519 key: {e}"),
        }))
    })?;

    let mut lock = state.lock().unwrap();
    let id = lock.store.append(&req.attestation, &vk).map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(StoreErrorResponse {
            error: format!("{e}"),
        }))
    })?;

    Ok(Json(ImportResponse {
        status: "ok".into(),
        id: hex_id(&id),
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_hex_32(hex: &str) -> Result<[u8; 32], String> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", hex.len()));
    }
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("non-hex characters".into());
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|e| format!("{e}"))?;
    }
    Ok(out)
}
