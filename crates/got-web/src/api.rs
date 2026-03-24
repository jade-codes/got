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
    /// Mean message coherence across all turns.
    pub mean_message_coherence: f32,
    /// Final convergence between speakers' value profiles.
    pub final_convergence: f32,
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
    /// Coherence of just THIS message's detected values (not cumulative).
    /// Computed from the pairwise geometry of the 6 terms in this message.
    /// Dynamic every turn — never saturates.
    pub message_coherence: f32,
    /// Trust score ∈ [0, 1]. Combines coherence with drift rate.
    pub trust_score: f32,
    /// Drift in value-activation profile from this speaker's first message.
    /// 0 = identical value profile, 1 = orthogonal.
    pub speaker_drift: f32,
    /// Cosine similarity between the two speakers' cumulative value profiles.
    /// Rises as one speaker adopts the other's framing.
    pub convergence: f32,
    /// Contradictions that are new at this turn.
    pub new_contradictions: Vec<Contradiction>,
    /// Contradictions from all_contradictions where at least one term
    /// was detected in THIS message. Surfaces ongoing tensions.
    pub turn_contradictions: Vec<Contradiction>,
    /// Contradictions within THIS message's detected values only.
    pub message_contradictions: Vec<Contradiction>,
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

/// Detect which value terms are active in a message embedding.
///
/// Computes z-scored logits: for each term, the raw dot product h·u_i
/// (the model's logit for that term) is standardized across all terms.
/// Terms with above-average activation (z > 0) are detected.
/// Returns the top values sorted by descending z-score.
fn detect_values(
    msg_embedding: &[f32],
    term_embeddings: &HashMap<String, Vec<f32>>,
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

    // Use the mode-specific thresholds configured at startup.
    // Request can override antonym/synonym thresholds if provided.
    let config = CoherenceConfig {
        antonym_threshold: req.antonym_threshold.unwrap_or(state.default_config.antonym_threshold),
        synonym_threshold: req.synonym_threshold.unwrap_or(state.default_config.synonym_threshold),
        severity_scale: state.default_config.severity_scale,
    };
    let introduction_threshold = state.introduction_threshold;

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
            6, // top 6 values per message
        );

        // Collect newly introduced values (z > introduction_threshold)
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

        // Collect term names detected in this message (for filtering active contradictions)
        let turn_term_set: HashSet<String> = detected.iter().map(|dv| dv.term.clone()).collect();

        // Per-message coherence: analyse just this message's detected values.
        // This is dynamic every turn (never saturates) because each message
        // has its own set of 6 values.
        let turn_terms: Vec<&str> = detected.iter().map(|dv| dv.term.as_str()).collect();
        let (msg_coherence, msg_contradictions) = if turn_terms.len() >= 2 {
            match got_incoherence::analyse_value_system(
                &turn_terms,
                state.embedding_source.as_ref(),
                &state.geometry,
                &config,
            ) {
                Ok(report) => (
                    report.analysis.coherence_score,
                    report.analysis.contradictions,
                ),
                Err(e) => {
                    eprintln!("warning: message coherence analysis failed for turn {idx}: {e}");
                    (1.0, vec![])
                }
            }
        } else {
            (1.0, vec![])
        };

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
                message_coherence: msg_coherence,
                trust_score: 0.0, // computed in post-pass
                speaker_drift: 0.0, // computed in post-pass
                convergence: 0.0, // computed in post-pass
                new_contradictions: vec![],
                turn_contradictions: vec![],
                message_contradictions: msg_contradictions,
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

                // Active contradictions: those where at least one term
                // was detected in this message's values.
                let turn_contradictions: Vec<Contradiction> = report
                    .analysis
                    .contradictions
                    .iter()
                    .filter(|c| {
                        turn_term_set.contains(&c.term_a)
                            || turn_term_set.contains(&c.term_b)
                    })
                    .cloned()
                    .collect();

                turns.push(TurnAnalysis {
                    turn: idx,
                    speaker: msg.speaker.clone(),
                    text: msg.text.clone(),
                    detected_values: detected,
                    values_introduced,
                    cumulative_values: cumulative_values.clone(),
                    coherence_score: report.analysis.coherence_score,
                    message_coherence: msg_coherence,
                    trust_score: 0.0, // computed in post-pass
                    speaker_drift: 0.0, // computed in post-pass
                    convergence: 0.0, // computed in post-pass
                    new_contradictions,
                    turn_contradictions,
                    message_contradictions: msg_contradictions,
                    all_contradictions: report.analysis.contradictions,
                    all_redundancies: report.analysis.redundancies,
                    pairwise: report.analysis.pairwise,
                    num_terms: report.analysis.num_terms,
                    num_unresolved: report.analysis.num_unresolved,
                });
            }
            Err(e) => {
                eprintln!("warning: cumulative coherence analysis failed at turn {idx}: {e}");
                turns.push(TurnAnalysis {
                    turn: idx,
                    speaker: msg.speaker.clone(),
                    text: msg.text.clone(),
                    detected_values: detected,
                    values_introduced,
                    cumulative_values: cumulative_values.clone(),
                    coherence_score: 1.0,
                    message_coherence: msg_coherence,
                    trust_score: 0.0, // computed in post-pass
                    speaker_drift: 0.0, // computed in post-pass
                    convergence: 0.0, // computed in post-pass
                    new_contradictions: vec![],
                    turn_contradictions: vec![],
                    message_contradictions: msg_contradictions,
                    all_contradictions: vec![],
                    all_redundancies: vec![],
                    pairwise: vec![],
                    num_terms: cumulative_values.len(),
                    num_unresolved: cumulative_values.len(),
                });
            }
        }
    }

    // Post-pass: compute trust_score, speaker_drift, and convergence
    compute_trust_scores(&mut turns);
    compute_speaker_drift(&mut turns);
    compute_convergence(&mut turns);

    // Build per-speaker summaries and overall assessment
    let speaker_summary = build_speaker_summaries(&turns);
    let assessment = build_assessment(&turns);

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

    let mut contradiction_memory: f32 = 0.0;
    let decay = 0.8; // memory of past contradictions decays per turn
    let contradiction_weight = 0.5; // how much each contradiction-bearing message hurts

    for i in 0..turns.len() {
        let msg_coh = turns[i].message_coherence;

        // Accumulate memory of contradictory messages
        if !turns[i].message_contradictions.is_empty() {
            // This message contained contradictions — penalise
            let severity = 1.0 - msg_coh;
            contradiction_memory = (contradiction_memory * decay) + (severity * contradiction_weight);
        } else {
            // Clean message — let memory decay
            contradiction_memory *= decay;
        }

        // Trust = message coherence × (1 - contradiction memory)
        // A coherent message after a string of contradictions still gets
        // partial trust reduction from memory. But it CAN recover.
        let memory_factor = (1.0 - contradiction_memory).clamp(0.0, 1.0);
        turns[i].trust_score = (msg_coh * memory_factor).clamp(0.0, 1.0);
    }
}

/// Cosine similarity between two embedding vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    got_core::geometry::euclidean_cosine(a, b)
}

/// Compute per-speaker value-profile drift.
///
/// Instead of comparing raw embeddings (which all cluster together for same-topic
/// messages), we compare each message's VALUE ACTIVATION PROFILE — the z-score
/// vector over all detected terms. This captures stance shifts even when the
/// topic stays the same.
fn compute_speaker_drift(turns: &mut [TurnAnalysis]) {
    // Collect all value terms seen in the conversation
    let all_terms: Vec<String> = if let Some(last) = turns.last() {
        last.cumulative_values.clone()
    } else {
        return;
    };
    if all_terms.is_empty() {
        return;
    }

    // Build value profile vector for a turn: z-scores indexed by term position
    let build_profile = |turn: &TurnAnalysis| -> Vec<f32> {
        let mut profile = vec![0.0f32; all_terms.len()];
        for dv in &turn.detected_values {
            if let Some(idx) = all_terms.iter().position(|t| t == &dv.term) {
                profile[idx] = dv.cos_phi;
            }
        }
        profile
    };

    // Track first profile per speaker
    let mut first_profile: HashMap<String, Vec<f32>> = HashMap::new();

    for turn in turns.iter_mut() {
        let profile = build_profile(turn);
        let speaker = &turn.speaker;

        if !first_profile.contains_key(speaker) {
            first_profile.insert(speaker.clone(), profile.clone());
        }

        if let Some(first) = first_profile.get(speaker) {
            let cos = cosine_similarity(first, &profile);
            turn.speaker_drift = (1.0 - cos).max(0.0);
        }
    }
}

/// Compute per-turn convergence between speakers' cumulative value profiles.
///
/// For each turn, maintains a running sum of z-scores per term per speaker.
/// Convergence = cosine similarity between the two running profiles.
/// Rises from ~0 (different priorities) toward 1 as one speaker adopts
/// the other's value framing.
fn compute_convergence(turns: &mut [TurnAnalysis]) {
    // Collect all value terms from the final cumulative set.
    let all_terms: Vec<String> = if let Some(last) = turns.last() {
        last.cumulative_values.clone()
    } else {
        return;
    };
    if all_terms.is_empty() {
        return;
    }

    // Identify speakers in order of appearance
    let speakers: Vec<String> = {
        let mut seen = Vec::new();
        for t in turns.iter() {
            if !seen.contains(&t.speaker) {
                seen.push(t.speaker.clone());
            }
        }
        seen
    };
    if speakers.len() < 2 {
        return;
    }

    // Running sum of z-scores per speaker
    let mut profiles: HashMap<String, Vec<f32>> = HashMap::new();
    for s in &speakers {
        profiles.insert(s.clone(), vec![0.0f32; all_terms.len()]);
    }

    for turn in turns.iter_mut() {
        // Accumulate this turn's z-scores into the speaker's running profile
        if let Some(profile) = profiles.get_mut(&turn.speaker) {
            for dv in &turn.detected_values {
                if let Some(idx) = all_terms.iter().position(|t| t == &dv.term) {
                    profile[idx] += dv.cos_phi;
                }
            }
        }

        // Convergence = cosine between the two speakers' running profiles
        let p0 = &profiles[&speakers[0]];
        let p1 = &profiles[&speakers[1]];
        let cos = cosine_similarity(p0, p1);
        // Normalise: cosine 0→1 maps to convergence 0→1
        // Negative cosine (opposing profiles) → 0
        turn.convergence = cos.max(0.0);
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
fn build_assessment(turns: &[TurnAnalysis]) -> Assessment {
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

    // Aggregate message coherence stats
    let mean_msg_coh = if turns.is_empty() {
        1.0
    } else {
        turns.iter().map(|t| t.message_coherence).sum::<f32>() / turns.len() as f32
    };
    let final_convergence = turns.last().map(|t| t.convergence).unwrap_or(0.0);

    // Determine verdict using both cumulative and per-message signals
    let n_contradictions = turns.last()
        .map(|t| t.all_contradictions.len())
        .unwrap_or(0);

    // Count turns with message-level contradictions (dynamic signal)
    let turns_with_msg_contradictions = turns.iter()
        .filter(|t| !t.message_contradictions.is_empty())
        .count();

    let (verdict, summary) = if final_trust < 0.25 && influence_score > 0.02 {
        ("manipulative".to_string(), format!(
            "Manipulation pattern detected. {}/{} messages contain internal value contradictions. \
             Speaker convergence: {:.0}%. {} cumulative contradiction{} with {:.0}% influence.",
            turns_with_msg_contradictions, turns.len(),
            final_convergence * 100.0,
            n_contradictions,
            if n_contradictions != 1 { "s" } else { "" },
            influence_score * 100.0,
        ))
    } else if final_trust < 0.3 {
        ("inconsistent".to_string(), format!(
            "Significant value contradictions detected. Final trust: {:.0}%. \
             Mean message coherence: {:.0}%.",
            final_trust * 100.0, mean_msg_coh * 100.0
        ))
    } else if final_coherence < 0.5 || mean_msg_coh < 0.5 {
        ("drifting".to_string(), format!(
            "Moderate value drift detected. Coherence: {:.0}%, Message coherence: {:.0}%. \
             Some contradictions between stated values.",
            final_coherence * 100.0, mean_msg_coh * 100.0
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
        mean_message_coherence: mean_msg_coh,
        final_convergence,
    }
}
