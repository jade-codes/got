// ---------------------------------------------------------------------------
// API handlers for conversation-based incoherence analysis.
//
// The core endpoint (POST /api/conversation/analyse) takes a conversation
// where each message carries an embedding vector. The handler:
//   1. Uses the pre-built Φ from AppState (either real model or synthetic)
//   2. Projects each message embedding against all value terms using cos_Φ
//   3. Accumulates detected values turn by turn
//   4. Runs contradiction analysis at each turn via got_incoherence
// ---------------------------------------------------------------------------

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use got_core::geometry::CausalGeometry;
use got_incoherence::coherence::{CoherenceConfig, Contradiction, Redundancy, PairwiseRelation};
use serde::{Deserialize, Serialize};

use crate::AppState;

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
    /// Pre-computed message embedding (dimension must match geometry).
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ConversationResponse {
    pub turns: Vec<TurnAnalysis>,
    pub available_terms: Vec<String>,
    /// "gpt2" or "synthetic-demo" — indicates geometry source.
    pub mode: String,
    /// Per-speaker summary statistics.
    pub speaker_summary: Vec<SpeakerSummary>,
    /// Overall manipulation risk assessment.
    pub assessment: Assessment,
}

/// Per-speaker aggregate statistics across the conversation.
#[derive(Debug, Serialize)]
pub struct SpeakerSummary {
    pub speaker: String,
    pub message_count: usize,
    /// Cosine similarity between speaker's first and last message embeddings.
    /// Low values mean semantic drift — the speaker changed position.
    pub semantic_drift: f32,
    /// Top value terms activated by this speaker (aggregated z-scores).
    pub top_values: Vec<(String, f32)>,
}

/// Overall assessment combining coherence, trust, and manipulation signals.
#[derive(Debug, Serialize)]
pub struct Assessment {
    /// "manipulative", "inconsistent", "coherent"
    pub verdict: String,
    /// Human-readable explanation.
    pub summary: String,
    /// How much the first speaker's semantic position drifted toward
    /// the second speaker's framing. ∈ [0, 1]. High = strong influence.
    pub influence_score: f32,
    /// Final coherence.
    pub final_coherence: f32,
    /// Final trust.
    pub final_trust: f32,
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
    pub trust_score: f32,
    /// Cosine between this message and the speaker's first message.
    /// Tracks how much each speaker's semantic position is shifting.
    pub speaker_drift: f32,
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

/// Build Φ = EᵀE from a set of embeddings, with ε-regularisation.
///
/// E is the n×d matrix of embeddings. Φ = EᵀE is d×d.
/// When n < d (fewer embeddings than dimensions), Φ is rank-deficient,
/// so we add εI to ensure positive-definiteness.
pub fn build_geometry_from_embeddings(
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

    // Try without regularisation first
    match CausalGeometry::from_raw_gram(gram.clone(), dim) {
        Ok(g) => Ok(g),
        Err(_) => {
            // Rank-deficient (n < dim) — add εI regularisation
            let epsilon = 1e-6_f32;
            for i in 0..dim {
                gram[i * dim + i] += epsilon;
            }
            CausalGeometry::from_raw_gram(gram, dim)
                .map_err(|e| format!("geometry error after regularisation: {e}"))
        }
    }
}

/// Detect which value terms are active in a message embedding.
///
/// Computes z-scored logits: for each term, the raw dot product h·u_i
/// (the model's logit for that term) is standardized across all terms.
/// Terms with above-average activation (z > 0) are detected.
/// Returns the top values sorted by descending z-score.
fn detect_values(
    msg_embedding: &[f32],
    term_embeddings: &HashMap<String, Vec<f32>>,
    _geometry: &CausalGeometry,
    _threshold: f32,
    max_per_message: usize,
) -> Vec<DetectedValue> {
    // Compute raw logits: h · u_i (standard dot product)
    let scores: Vec<(String, f32)> = term_embeddings
        .iter()
        .map(|(term, term_emb)| {
            let dot: f32 = msg_embedding.iter()
                .zip(term_emb.iter())
                .map(|(a, b)| a * b)
                .sum();
            (term.clone(), dot)
        })
        .collect();

    if scores.is_empty() {
        return vec![];
    }

    // Standardize: z = (logit - mean) / std
    let n = scores.len() as f32;
    let mean = scores.iter().map(|(_, s)| s).sum::<f32>() / n;
    let variance = scores.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt().max(1e-10);

    let mut detected: Vec<DetectedValue> = scores
        .iter()
        .map(|(term, score)| DetectedValue {
            term: term.clone(),
            cos_phi: (score - mean) / std_dev, // z-score
        })
        .collect();

    // Sort by z-score descending (strongest activations first)
    detected.sort_by(|a, b| {
        b.cos_phi.partial_cmp(&a.cos_phi).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top N with positive z-score (above-average activation)
    detected
        .into_iter()
        .filter(|dv| dv.cos_phi > 0.0)
        .take(max_per_message)
        .collect()
}

// ---------------------------------------------------------------------------
// Handler: POST /api/conversation/analyse
// ---------------------------------------------------------------------------

pub async fn analyse_conversation(
    State(state): State<Arc<AppState>>,
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

    // Calibrate thresholds based on mode.
    // Synthetic: cos_Φ ranges [-1, +1], antonym at -0.5, synonym at 0.8.
    // Real model: term embeddings are mean-centred before pairwise analysis,
    //   giving cosines ≈ [-0.23, 0.53] with mean ≈ -0.04, std ≈ 0.10.
    //   severity_scale = std so that small but significant deviations
    //   produce meaningful severity values.
    let is_real_model = state.mode != "synthetic-demo";
    let default_antonym = if is_real_model { -0.15 } else { -0.5 };
    let default_synonym = if is_real_model { 0.20 } else { 0.8 };
    let severity_scale = if is_real_model { Some(0.10) } else { None };

    let config = CoherenceConfig {
        antonym_threshold: req.antonym_threshold.unwrap_or(default_antonym),
        synonym_threshold: req.synonym_threshold.unwrap_or(default_synonym),
        severity_scale,
    };
    let detection_threshold = req.detection_threshold.unwrap_or(0.3);

    // For introduction, require stronger activation (z > 1.0 for real models)
    let introduction_threshold = if is_real_model { 1.0 } else { 0.0 };

    let mut cumulative_values: Vec<String> = Vec::new();
    let mut seen_values: HashSet<String> = HashSet::new();
    let mut previous_contradiction_keys: HashSet<(String, String)> = HashSet::new();
    let mut turns: Vec<TurnAnalysis> = Vec::new();

    for (idx, msg) in req.messages.iter().enumerate() {
        // Validate embedding dimension
        if msg.embedding.len() != state.hidden_dim {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "message {} embedding dimension mismatch: expected {}, got {}",
                        idx, state.hidden_dim, msg.embedding.len()
                    ),
                }),
            ));
        }

        // Detect values from message embedding via causal projection
        let detected = detect_values(
            &msg.embedding,
            &state.term_embeddings,
            &state.geometry,
            detection_threshold,
            6, // top 6 values per message
        );

        // Collect newly introduced values
        // For real models, require z > 1.0 (strong activation).
        // For synthetic, any positive cos_Φ = affirmed.
        let mut values_introduced = Vec::new();
        for dv in &detected {
            if dv.cos_phi > introduction_threshold {
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
                speaker_drift: 0.0, // computed in post-pass
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
            state.embedding_source.as_ref(),
            &state.geometry,
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
                    speaker_drift: 0.0, // computed in post-pass
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
                    speaker_drift: 0.0, // computed in post-pass
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

    // Post-pass: compute trust_score and speaker_drift
    compute_trust_scores(&mut turns);
    compute_speaker_drift(&mut turns, &req.messages);

    // Build per-speaker summaries and overall assessment
    let speaker_summary = build_speaker_summaries(&turns);
    let assessment = build_assessment(&turns, &req.messages);

    Ok(Json(ConversationResponse {
        turns,
        available_terms: state.available_terms.clone(),
        mode: state.mode.clone(),
        speaker_summary,
        assessment,
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

/// Cosine similarity between two embedding vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute per-speaker semantic drift: cosine(first_msg, current_msg) for each speaker.
fn compute_speaker_drift(turns: &mut [TurnAnalysis], messages: &[MessageInput]) {
    // Track first message embedding per speaker
    let mut first_embedding: HashMap<String, &[f32]> = HashMap::new();

    for (i, msg) in messages.iter().enumerate() {
        let speaker = &msg.speaker;
        if !first_embedding.contains_key(speaker) {
            first_embedding.insert(speaker.clone(), &msg.embedding);
        }
        if let Some(first) = first_embedding.get(speaker) {
            // drift = 1 - cosine(first, current). 0 = identical, 1 = orthogonal.
            let cos = cosine_similarity(first, &msg.embedding);
            turns[i].speaker_drift = 1.0 - cos;
        }
    }
}

/// Build per-speaker summary: aggregate detected values and semantic drift.
fn build_speaker_summaries(turns: &[TurnAnalysis]) -> Vec<SpeakerSummary> {
    let mut speakers: Vec<String> = Vec::new();
    let mut value_totals: HashMap<String, HashMap<String, f32>> = HashMap::new();
    let mut msg_counts: HashMap<String, usize> = HashMap::new();
    let mut last_drift: HashMap<String, f32> = HashMap::new();

    for turn in turns {
        let s = &turn.speaker;
        if !speakers.contains(s) {
            speakers.push(s.clone());
        }
        *msg_counts.entry(s.clone()).or_default() += 1;
        last_drift.insert(s.clone(), turn.speaker_drift);

        let entry = value_totals.entry(s.clone()).or_default();
        for dv in &turn.detected_values {
            *entry.entry(dv.term.clone()).or_default() += dv.cos_phi;
        }
    }

    speakers.iter().map(|s| {
        let mut top: Vec<(String, f32)> = value_totals
            .get(s).cloned().unwrap_or_default()
            .into_iter().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top.truncate(5);

        SpeakerSummary {
            speaker: s.clone(),
            message_count: *msg_counts.get(s).unwrap_or(&0),
            semantic_drift: *last_drift.get(s).unwrap_or(&0.0),
            top_values: top,
        }
    }).collect()
}

/// Build overall assessment from the analysis.
///
/// Influence is computed from VALUE ACTIVATION PROFILES, not raw embeddings.
/// GPT-2 message embeddings capture topic (similar across all messages about
/// the same subject) but not stance. Value z-scores capture stance.
fn build_assessment(turns: &[TurnAnalysis], _messages: &[MessageInput]) -> Assessment {
    let final_coherence = turns.last().map(|t| t.coherence_score).unwrap_or(1.0);
    let final_trust = turns.last().map(|t| t.trust_score).unwrap_or(1.0);

    // Identify speakers (in order of appearance)
    let speakers: Vec<String> = {
        let mut seen = Vec::new();
        for t in turns {
            if !seen.contains(&t.speaker) {
                seen.push(t.speaker.clone());
            }
        }
        seen
    };

    // Compute influence from value activation profiles.
    //
    // For a 2-party conversation:
    // - Build the first speaker's "early" and "late" value vectors
    //   (sum of z-scores per term in first half vs second half)
    // - Compute cosine between early and late profiles
    // - 1 - cosine = value reorientation
    //
    // Also: check if second speaker has stable values while first speaker shifts.
    // That asymmetry is the manipulation fingerprint.
    let influence_score = if speakers.len() >= 2 {
        let speaker_a = &speakers[0];
        let speaker_b = &speakers[1];

        // Collect all value terms seen
        let all_terms: Vec<String> = turns.last()
            .map(|t| t.cumulative_values.clone())
            .unwrap_or_default();

        if all_terms.is_empty() {
            0.0
        } else {
            // Build value profile vectors: z-scores summed per term
            let build_profile = |speaker: &str, turns_slice: &[&TurnAnalysis]| -> Vec<f32> {
                let mut profile = vec![0.0f32; all_terms.len()];
                for turn in turns_slice {
                    if turn.speaker == speaker {
                        for dv in &turn.detected_values {
                            if let Some(idx) = all_terms.iter().position(|t| t == &dv.term) {
                                profile[idx] += dv.cos_phi;
                            }
                        }
                    }
                }
                profile
            };

            let mid = turns.len() / 2;
            let first_half: Vec<&TurnAnalysis> = turns[..mid].iter().collect();
            let second_half: Vec<&TurnAnalysis> = turns[mid..].iter().collect();

            let a_early = build_profile(speaker_a, &first_half);
            let a_late = build_profile(speaker_a, &second_half);
            let b_early = build_profile(speaker_b, &first_half);
            let b_late = build_profile(speaker_b, &second_half);

            // Speaker A's value reorientation
            let a_shift = 1.0 - cosine_similarity(&a_early, &a_late);
            // Speaker B's stability (low shift = consistent values)
            let b_shift = 1.0 - cosine_similarity(&b_early, &b_late);

            // Asymmetry: A shifted but B didn't → B influenced A
            let asymmetry = (a_shift - b_shift).max(0.0);

            // Cross-check: did A's late profile move toward B's profile?
            let a_b_early_cos = cosine_similarity(&a_early, &b_early);
            let a_late_b_cos = cosine_similarity(&a_late, &b_late);
            let convergence = (a_late_b_cos - a_b_early_cos).max(0.0);

            // Influence = asymmetric shift + convergence toward B
            ((asymmetry + convergence) / 2.0).clamp(0.0, 1.0)
        }
    } else {
        0.0
    };

    // Determine verdict
    let n_contradictions = turns.last()
        .map(|t| t.all_contradictions.len())
        .unwrap_or(0);

    let (verdict, summary) = if final_trust < 0.25 && influence_score > 0.02 {
        ("manipulative".to_string(), format!(
            "Manipulation pattern detected. Coherence collapsed to {:.0}% \
             with {:.0}% value-profile influence on the first speaker. \
             {} contradiction{} emerged as values were gradually reframed.",
            final_coherence * 100.0, influence_score * 100.0,
            n_contradictions,
            if n_contradictions != 1 { "s" } else { "" }
        ))
    } else if final_trust < 0.3 {
        ("inconsistent".to_string(), format!(
            "Significant value contradictions detected. Final trust: {:.0}%. \
             The stated values in this conversation are internally incoherent.",
            final_trust * 100.0
        ))
    } else if final_coherence < 0.5 {
        ("drifting".to_string(), format!(
            "Moderate value drift detected. Coherence: {:.0}%. \
             Some contradictions between stated values.",
            final_coherence * 100.0
        ))
    } else {
        ("coherent".to_string(), format!(
            "Values appear consistent. Coherence: {:.0}%, Trust: {:.0}%.",
            final_coherence * 100.0, final_trust * 100.0
        ))
    };

    Assessment {
        verdict,
        summary,
        influence_score,
        final_coherence,
        final_trust,
    }
}
