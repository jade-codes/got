// ---------------------------------------------------------------------------
// API handlers for conversation-based incoherence analysis.
//
// The core endpoint (POST /api/conversation/analyse) takes a conversation
// where each message carries a 32-d embedding vector. The handler:
//   1. Builds Φ = EᵀE from the value-term embeddings
//   2. Projects each message embedding against all value terms using cos_Φ
//   3. Accumulates detected values turn by turn
//   4. Runs contradiction analysis at each turn via got_incoherence
//
// This reveals how incoherence *emerges* over the course of a dialogue,
// with the causal geometry doing the actual value detection — not hand-tags.
// ---------------------------------------------------------------------------

use axum::{http::StatusCode, Json};
use got_core::geometry::CausalGeometry;
use got_incoherence::coherence::{self, CoherenceConfig, Contradiction, Redundancy, PairwiseRelation};
use got_incoherence::embeddings::PrecomputedEmbeddings;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::demo;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ConversationRequest {
    pub messages: Vec<MessageInput>,
    pub antonym_threshold: Option<f32>,
    pub synonym_threshold: Option<f32>,
    /// cos_Φ threshold for value detection (default: 0.3).
    pub detection_threshold: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct MessageInput {
    pub speaker: String,
    pub text: String,
    /// Pre-computed embedding for this message (32-d).
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ConversationResponse {
    pub turns: Vec<TurnAnalysis>,
    pub available_terms: Vec<String>,
}

/// Analysis state after each message in the conversation.
#[derive(Debug, Serialize)]
pub struct TurnAnalysis {
    pub turn: usize,
    pub speaker: String,
    pub text: String,
    /// Values detected in *this* message by causal projection.
    pub detected_values: Vec<DetectedValue>,
    /// Values newly added to the cumulative set at this turn.
    pub values_introduced: Vec<String>,
    /// All values accumulated up to this point.
    pub cumulative_values: Vec<String>,
    pub coherence_score: f32,
    /// Trust score ∈ [0, 1]. Combines coherence with drift rate.
    /// Drops faster than coherence because sudden coherence loss is a red flag.
    pub trust_score: f32,
    /// Contradictions that are new at this turn.
    pub new_contradictions: Vec<Contradiction>,
    /// All contradictions at this point.
    pub all_contradictions: Vec<Contradiction>,
    /// All redundancies at this point.
    pub all_redundancies: Vec<Redundancy>,
    /// All pairwise relations at this point.
    pub pairwise: Vec<PairwiseRelation>,
    pub num_terms: usize,
    pub num_unresolved: usize,
}

/// A value detected in a message via causal cosine projection.
#[derive(Debug, Clone, Serialize)]
pub struct DetectedValue {
    pub term: String,
    /// Causal cosine similarity between message embedding and term embedding.
    pub cos_phi: f32,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contradiction_key(c: &Contradiction) -> (String, String) {
    let (a, b) = if c.term_a < c.term_b {
        (c.term_a.clone(), c.term_b.clone())
    } else {
        (c.term_b.clone(), c.term_a.clone())
    };
    (a, b)
}

/// Build Φ = EᵀE from the demo value-term embeddings.
///
/// E is the 28×32 matrix of term embeddings. Φ = EᵀE is 32×32 — the
/// causal geometry of the value space.
fn build_geometry_from_embeddings(
    embeddings_map: &HashMap<String, Vec<f32>>,
    dim: usize,
) -> Result<CausalGeometry, String> {
    let n = embeddings_map.len();
    if n == 0 {
        return Err("no embeddings".into());
    }

    // Stack embeddings into matrix E (n × dim), row-major
    let mut e_data = Vec::with_capacity(n * dim);
    for emb in embeddings_map.values() {
        if emb.len() != dim {
            return Err(format!("embedding dim mismatch: expected {dim}, got {}", emb.len()));
        }
        e_data.extend_from_slice(emb);
    }

    // Compute Φ = EᵀE (dim × dim)
    let mut gram = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut dot = 0.0f32;
            for k in 0..n {
                dot += e_data[k * dim + i] * e_data[k * dim + j];
            }
            gram[i * dim + j] = dot;
            gram[j * dim + i] = dot; // symmetric
        }
    }

    CausalGeometry::from_raw_gram(gram, dim)
        .map_err(|e| format!("geometry error: {e}"))
}

/// Detect which value terms are active in a message embedding.
///
/// Computes cos_Φ(message, term) for every term. Returns the top values
/// sorted by descending |cos_Φ|, keeping only those above the threshold
/// and capped at `max_per_message` results.
fn detect_values(
    msg_embedding: &[f32],
    term_embeddings: &HashMap<String, Vec<f32>>,
    geometry: &CausalGeometry,
    threshold: f32,
    max_per_message: usize,
) -> Vec<DetectedValue> {
    let mut all_scores: Vec<DetectedValue> = term_embeddings
        .iter()
        .filter_map(|(term, term_emb)| {
            coherence::causal_cosine(msg_embedding, term_emb, geometry)
                .ok()
                .map(|cos| DetectedValue { term: term.clone(), cos_phi: cos })
        })
        .collect();

    // Sort by |cos_Φ| descending so strongest activations come first
    all_scores.sort_by(|a, b| {
        b.cos_phi.abs().partial_cmp(&a.cos_phi.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top N that are above threshold
    all_scores
        .into_iter()
        .filter(|dv| dv.cos_phi.abs() >= threshold)
        .take(max_per_message)
        .collect()
}

struct DemoResources {
    source: PrecomputedEmbeddings,
    geometry: CausalGeometry,
    term_embeddings: HashMap<String, Vec<f32>>,
    dim: usize,
    available_terms: Vec<String>,
}

fn load_demo_resources() -> Result<DemoResources, (StatusCode, Json<ErrorResponse>)> {
    let err =
        |msg: String| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: msg }));

    let term_embeddings: HashMap<String, Vec<f32>> =
        serde_json::from_str(demo::demo_embeddings_json())
            .map_err(|e| err(format!("failed to parse embeddings: {e}")))?;

    let dim = term_embeddings
        .values()
        .next()
        .map(|v| v.len())
        .ok_or_else(|| err("no embeddings".into()))?;

    let geometry =
        build_geometry_from_embeddings(&term_embeddings, dim).map_err(|e| err(e))?;

    let source = PrecomputedEmbeddings::from_json(demo::demo_embeddings_json())
        .map_err(|e| err(format!("failed to load embeddings: {e}")))?;

    let mut available_terms: Vec<String> = term_embeddings.keys().cloned().collect();
    available_terms.sort();

    Ok(DemoResources {
        source,
        geometry,
        term_embeddings,
        dim,
        available_terms,
    })
}

// ---------------------------------------------------------------------------
// Handler: POST /api/conversation/analyse
// ---------------------------------------------------------------------------

pub async fn analyse_conversation(
    Json(req): Json<ConversationRequest>,
) -> Result<Json<ConversationResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "no messages provided".into(),
            }),
        ));
    }

    if req.messages.len() > 100 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "too many messages (max 100)".into(),
            }),
        ));
    }

    let resources = load_demo_resources()?;

    let config = CoherenceConfig {
        antonym_threshold: req.antonym_threshold.unwrap_or(-0.5),
        synonym_threshold: req.synonym_threshold.unwrap_or(0.8),
    };
    let detection_threshold = req.detection_threshold.unwrap_or(0.3);

    let mut cumulative_values: Vec<String> = Vec::new();
    let mut seen_values: HashSet<String> = HashSet::new();
    let mut previous_contradiction_keys: HashSet<(String, String)> = HashSet::new();
    let mut turns: Vec<TurnAnalysis> = Vec::new();

    for (idx, msg) in req.messages.iter().enumerate() {
        // Validate embedding dimension
        if msg.embedding.len() != resources.dim {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "message {} embedding dimension mismatch: expected {}, got {}",
                        idx, resources.dim, msg.embedding.len()
                    ),
                }),
            ));
        }

        // Detect values from message embedding via causal projection
        let detected = detect_values(
            &msg.embedding,
            &resources.term_embeddings,
            &resources.geometry,
            detection_threshold,
            6, // top 6 values per message
        );

        // Collect newly introduced values (positive cos_Φ only = affirmed)
        let mut values_introduced = Vec::new();
        for dv in &detected {
            if dv.cos_phi > 0.0 {
                let normalised = dv.term.trim().to_lowercase();
                if seen_values.insert(normalised.clone()) {
                    cumulative_values.push(normalised.clone());
                    values_introduced.push(normalised);
                }
            }
        }

        // Need at least 2 values to run analysis
        if cumulative_values.len() < 2 {
            turns.push(TurnAnalysis {
                turn: idx,
                speaker: msg.speaker.clone(),
                text: msg.text.clone(),
                detected_values: detected,
                values_introduced,
                cumulative_values: cumulative_values.clone(),
                coherence_score: 1.0,
                trust_score: 0.0, // computed in post-pass
                new_contradictions: vec![],
                all_contradictions: vec![],
                all_redundancies: vec![],
                pairwise: vec![],
                num_terms: cumulative_values.len(),
                num_unresolved: 0,
            });
            continue;
        }

        let term_refs: Vec<&str> = cumulative_values.iter().map(|s| s.as_str()).collect();

        match got_incoherence::analyse_value_system(
            &term_refs,
            &resources.source,
            &resources.geometry,
            &config,
        ) {
            Ok(report) => {
                let new_contradictions: Vec<Contradiction> = report
                    .analysis
                    .contradictions
                    .iter()
                    .filter(|c| {
                        let key = contradiction_key(c);
                        !previous_contradiction_keys.contains(&key)
                    })
                    .cloned()
                    .collect();

                for c in &report.analysis.contradictions {
                    previous_contradiction_keys.insert(contradiction_key(c));
                }

                turns.push(TurnAnalysis {
                    turn: idx,
                    speaker: msg.speaker.clone(),
                    text: msg.text.clone(),
                    detected_values: detected,
                    values_introduced,
                    cumulative_values: cumulative_values.clone(),
                    coherence_score: report.analysis.coherence_score,
                    trust_score: 0.0, // computed in post-pass
                    new_contradictions,
                    all_contradictions: report.analysis.contradictions,
                    all_redundancies: report.analysis.redundancies,
                    pairwise: report.analysis.pairwise,
                    num_terms: report.analysis.num_terms,
                    num_unresolved: report.analysis.num_unresolved,
                });
            }
            Err(_) => {
                turns.push(TurnAnalysis {
                    turn: idx,
                    speaker: msg.speaker.clone(),
                    text: msg.text.clone(),
                    detected_values: detected,
                    values_introduced,
                    cumulative_values: cumulative_values.clone(),
                    coherence_score: 1.0,
                    trust_score: 0.0, // computed in post-pass
                    new_contradictions: vec![],
                    all_contradictions: vec![],
                    all_redundancies: vec![],
                    pairwise: vec![],
                    num_terms: cumulative_values.len(),
                    num_unresolved: cumulative_values.len(),
                });
            }
        }
    }

    // Also handle the early-exit TurnAnalysis (< 2 values) - read back to update
    // Post-pass: compute trust_score from coherence + drift
    compute_trust_scores(&mut turns);

    Ok(Json(ConversationResponse {
        turns,
        available_terms: resources.available_terms,
    }))
}

/// Compute trust scores as a post-pass over all turns.
///
/// Trust = coherence × stability.
///
/// - coherence: the raw coherence_score (how contradictory the current value set is)
/// - stability: penalises sudden coherence *drops*. Each drop (delta < 0) accumulates
///   as a "drift penalty" that decays slowly. The idea: a single drop from 1.0→0.8
///   is worse than starting at 0.8, because it signals active destabilisation.
///
/// trust = coherence × (1 - drift_penalty)
fn compute_trust_scores(turns: &mut [TurnAnalysis]) {
    if turns.is_empty() {
        return;
    }

    let mut drift_penalty: f32 = 0.0;
    let decay = 0.7; // drift memory: 70% carried forward each turn
    let drift_weight = 2.0; // amplify drops

    for i in 0..turns.len() {
        let coherence = turns[i].coherence_score;

        if i > 0 {
            let prev_coherence = turns[i - 1].coherence_score;
            let delta = coherence - prev_coherence;
            if delta < 0.0 {
                // Coherence dropped — accumulate drift penalty
                drift_penalty = (drift_penalty * decay) + (-delta * drift_weight);
            } else {
                // Coherence stable or improving — decay penalty
                drift_penalty *= decay;
            }
        }

        let stability = (1.0 - drift_penalty).clamp(0.0, 1.0);
        turns[i].trust_score = (coherence * stability).clamp(0.0, 1.0);
    }
}
