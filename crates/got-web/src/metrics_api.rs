// ---------------------------------------------------------------------------
// Metrics API: coherence scoring, manifold collapse, and model comparison.
//
// POST /api/coherence — value-ordering coherence C(h) per message
// POST /api/collapse  — effective value dimensionality (dim_eff)
// POST /api/compare   — value alignment distance between two models
// ---------------------------------------------------------------------------

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::AppState;

// ---------------------------------------------------------------------------
// POST /api/coherence
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CoherenceRequest {
    /// Value-ordering constraints using term names (resolved to embeddings).
    pub ordering: Vec<ConstraintInput>,
    /// Sharpness parameter (default 1.0).
    #[serde(default = "default_sharpness")]
    pub sharpness: f32,
    /// Message embeddings to score. If empty, uses all available term
    /// embeddings as "hidden states" (useful for a static geometry check).
    #[serde(default)]
    pub embeddings: Vec<Vec<f32>>,
}

fn default_sharpness() -> f32 { 1.0 }

#[derive(Debug, Deserialize)]
pub struct ConstraintInput {
    pub dominant: String,
    pub subordinate: String,
    pub label: String,
}

#[derive(Debug, Serialize)]
pub struct CoherenceResponse {
    pub per_message: Vec<f32>,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
    pub violated: Vec<ViolationInfo>,
}

#[derive(Debug, Serialize)]
pub struct ViolationInfo {
    pub position: usize,
    pub label: String,
    pub margin: f32,
}

pub async fn coherence(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CoherenceRequest>,
) -> Result<Json<CoherenceResponse>, (StatusCode, Json<ErrorResponse>)> {
    use got_core::coherence::{coherence_score, ValueConstraint, ValueOrdering};

    // Resolve term names to embedding vectors
    let mut constraints = Vec::new();
    for c in &req.ordering {
        let dom = state.term_embeddings.get(&c.dominant)
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, format!("unknown term: {}", c.dominant)))?;
        let sub = state.term_embeddings.get(&c.subordinate)
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, format!("unknown term: {}", c.subordinate)))?;
        constraints.push(ValueConstraint {
            dominant: dom.clone(),
            subordinate: sub.clone(),
            label: c.label.clone(),
        });
    }

    if constraints.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "no valid constraints"));
    }

    let ordering = ValueOrdering { constraints };

    // Use provided embeddings, or fall back to term embeddings as test states
    let hidden_states: Vec<&[f32]> = if !req.embeddings.is_empty() {
        req.embeddings.iter().map(|e| e.as_slice()).collect()
    } else {
        state.term_embeddings.values().map(|e| e.as_slice()).collect()
    };

    let mut per_message = Vec::with_capacity(hidden_states.len());
    let mut violated = Vec::new();

    for (pos, h) in hidden_states.iter().enumerate() {
        let score = coherence_score(h, &ordering, &state.geometry, req.sharpness)
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("coherence: {e}")))?;
        per_message.push(score);

        // Check individual constraints for violations at this position
        if score < 0.5 {
            for c in &ordering.constraints {
                let dot_dom = state.geometry.inner_product(&c.dominant, h)
                    .unwrap_or(0.0);
                let dot_sub = state.geometry.inner_product(&c.subordinate, h)
                    .unwrap_or(0.0);
                let margin = dot_dom - dot_sub;
                if margin < 0.0 {
                    violated.push(ViolationInfo {
                        position: pos,
                        label: c.label.clone(),
                        margin,
                    });
                }
            }
        }
    }

    let mean = if per_message.is_empty() { 1.0 }
        else { per_message.iter().sum::<f32>() / per_message.len() as f32 };
    let min = per_message.iter().cloned().reduce(f32::min).unwrap_or(1.0);
    let max = per_message.iter().cloned().reduce(f32::max).unwrap_or(1.0);

    Ok(Json(CoherenceResponse { per_message, mean, min, max, violated }))
}

// ---------------------------------------------------------------------------
// POST /api/collapse
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CollapseRequest {
    /// Term names to use as probe directions. If empty, uses all available terms.
    #[serde(default)]
    pub probe_terms: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CollapseResponse {
    pub k: usize,
    pub eigenvalues: Vec<f32>,
    pub dim_eff: f32,
    pub dim_eff_ratio: f32,
    pub assessment: String,
}

pub async fn collapse(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CollapseRequest>,
) -> Result<Json<CollapseResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Collect probe directions from term embeddings
    let terms: Vec<&String> = if req.probe_terms.is_empty() {
        state.available_terms.iter().collect()
    } else {
        req.probe_terms.iter().collect()
    };

    let mut weights: Vec<Vec<f32>> = Vec::new();
    for term in &terms {
        if let Some(emb) = state.term_embeddings.get(*term) {
            weights.push(emb.clone());
        }
    }

    if weights.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "no valid probe terms found"));
    }

    let weight_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();

    let proj = state.geometry.value_projected_gram(&weight_refs)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("collapse: {e}")))?;

    let ratio = proj.dim_eff / proj.k as f32;
    let assessment = if ratio > 0.8 { "fully spread" }
        else if ratio > 0.4 { "partially collapsed" }
        else { "severely collapsed" };

    Ok(Json(CollapseResponse {
        k: proj.k,
        eigenvalues: proj.eigenvalues,
        dim_eff: proj.dim_eff,
        dim_eff_ratio: ratio,
        assessment: assessment.to_string(),
    }))
}

// ---------------------------------------------------------------------------
// POST /api/compare
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CompareRequest {
    /// Path to a second .gotue file to compare against the loaded model.
    pub comparison_gotue_path: String,
    /// Term names to use as probe directions. If empty, uses all available terms.
    #[serde(default)]
    pub probe_terms: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CompareResponse {
    pub global_distance: f32,
    pub probe_projected_distance: Option<f32>,
    pub per_probe: Vec<ProbeDistance>,
    pub ratio: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ProbeDistance {
    pub label: String,
    pub distance: f32,
}

pub async fn compare(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompareRequest>,
) -> Result<Json<CompareResponse>, (StatusCode, Json<ErrorResponse>)> {
    use got_core::geometry::value_alignment_distance;
    use got_core::UnembeddingMatrix;

    // Load the second unembedding matrix
    let data = std::fs::read(&req.comparison_gotue_path)
        .map_err(|e| err(StatusCode::BAD_REQUEST, format!("cannot read {}: {e}", req.comparison_gotue_path)))?;

    if data.len() < 14 || &data[0..4] != b"GOTU" {
        return Err(err(StatusCode::BAD_REQUEST, "not a valid .gotue file"));
    }

    let mut offset = 6; // skip magic + version
    let vocab_size = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
    offset += 4;
    let hidden_dim = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
    offset += 4;

    let total = vocab_size * hidden_dim;
    if offset + total * 4 > data.len() {
        return Err(err(StatusCode::BAD_REQUEST, "gotue file truncated"));
    }

    let mut values = Vec::with_capacity(total);
    for i in 0..total {
        let start = offset + i * 4;
        values.push(f32::from_le_bytes(data[start..start+4].try_into().unwrap()));
    }

    let matrix_b = UnembeddingMatrix::new(vocab_size, hidden_dim, values)
        .map_err(|e| err(StatusCode::BAD_REQUEST, format!("invalid gotue: {e}")))?;

    // Build geometry B with same epsilon as geometry A
    let geo_b = got_core::geometry::CausalGeometry::from_unembedding(&matrix_b, state.geometry.epsilon());

    // Collect probe directions
    let terms: Vec<&String> = if req.probe_terms.is_empty() {
        state.available_terms.iter().collect()
    } else {
        req.probe_terms.iter().collect()
    };

    let mut probe_weights: Vec<Vec<f32>> = Vec::new();
    let mut probe_labels: Vec<String> = Vec::new();
    for term in &terms {
        if let Some(emb) = state.term_embeddings.get(*term) {
            probe_weights.push(emb.clone());
            probe_labels.push((*term).clone());
        }
    }

    let probes: Option<Vec<&[f32]>> = if probe_weights.is_empty() {
        None
    } else {
        Some(probe_weights.iter().map(|w| w.as_slice()).collect())
    };

    let dist = value_alignment_distance(
        &state.geometry,
        &geo_b,
        probes.as_deref(),
    ).map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("compare: {e}")))?;

    let per_probe = if let Some(ref per) = dist.per_probe_distances {
        per.iter().enumerate().map(|(i, &d)| ProbeDistance {
            label: probe_labels.get(i).cloned().unwrap_or_else(|| format!("probe_{i}")),
            distance: d,
        }).collect()
    } else {
        Vec::new()
    };

    let ratio = match (dist.probe_projected_distance, dist.global_distance > 1e-9) {
        (Some(pd), true) => Some(pd / dist.global_distance),
        _ => None,
    };

    Ok(Json(CompareResponse {
        global_distance: dist.global_distance,
        probe_projected_distance: dist.probe_projected_distance,
        per_probe,
        ratio,
    }))
}

// ---------------------------------------------------------------------------
// Shared error type
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

fn err(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.into() }))
}
