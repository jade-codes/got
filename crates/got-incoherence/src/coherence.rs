// ---------------------------------------------------------------------------
// Coherence analysis: the mathematical core.
//
// Given a set of resolved value embeddings, computes pairwise angles under
// the causal inner product and detects contradictions.
//
// Key metric: causal cosine similarity
//   cos_Φ(u, v) = ⟨u, v⟩_Φ / (‖u‖_Φ · ‖v‖_Φ)
//
// This differs from ordinary cosine similarity because it weights dimensions
// by their influence on model output (via Φ = UᵀU).  Two terms might look
// similar in Euclidean space but be functionally opposed through the lens
// of the unembedding matrix.
//
// Contradiction detection:
//   If a value system claims both X and Y, and cos_Φ(X, Y) ≈ -1,
//   then X and Y are near-antonyms in the model's output geometry.
//   That's a measurable incoherence.
//
// The analysis is fully deterministic: same embeddings + same Φ → same scores.
// ---------------------------------------------------------------------------

use got_core::geometry::{CausalGeometry, GeometryError};
use serde::{Deserialize, Serialize};

use crate::embeddings::ResolvedValue;
use crate::IncoherenceError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Pairwise relationship between two value terms under the causal geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseRelation {
    /// First term.
    pub term_a: String,
    /// Second term.
    pub term_b: String,
    /// Causal cosine similarity ∈ [-1, 1].
    ///  +1 = synonyms (aligned in output space)
    ///   0 = orthogonal (independent)
    ///  -1 = antonyms (opposed in output space)
    pub causal_cosine: f32,
    /// Causal distance: ‖a - b‖_Φ.
    pub causal_distance: f32,
    /// Classification of this relationship.
    pub relation: RelationType,
}

/// Classification of a pairwise relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    /// cos_Φ > synonym_threshold — these terms are functionally aligned.
    Aligned,
    /// cos_Φ < antonym_threshold — these terms are functionally opposed.
    Opposed,
    /// Between the two thresholds — independent or weakly related.
    Independent,
}

/// A detected contradiction in a stated value system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    /// The two contradicting terms.
    pub term_a: String,
    pub term_b: String,
    /// How opposed they are: 0.0 = barely opposed, 1.0 = perfect antonyms.
    /// Computed as: (-causal_cosine - antonym_threshold) / (1.0 - antonym_threshold)
    /// clamped to [0, 1].
    pub severity: f32,
    /// The raw causal cosine similarity (negative).
    pub causal_cosine: f32,
    /// Angle in degrees between the two embeddings under Φ.
    pub angle_degrees: f32,
}

/// A detected redundancy: two terms that are near-synonyms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Redundancy {
    pub term_a: String,
    pub term_b: String,
    /// How similar they are: closer to 1.0 = more redundant.
    pub similarity: f32,
    pub causal_cosine: f32,
}

/// Full coherence analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceAnalysis {
    /// All pairwise relations (n*(n-1)/2 pairs).
    pub pairwise: Vec<PairwiseRelation>,
    /// Detected contradictions (opposed pairs within the stated values).
    pub contradictions: Vec<Contradiction>,
    /// Detected redundancies (near-synonym pairs).
    pub redundancies: Vec<Redundancy>,
    /// Overall coherence score ∈ [0, 1].
    /// 1.0 = perfectly coherent (no contradictions, all independent or aligned).
    /// 0.0 = maximally incoherent.
    pub coherence_score: f32,
    /// Number of value terms analysed.
    pub num_terms: usize,
    /// Number of terms that could not be embedded (for reference).
    pub num_unresolved: usize,
    /// Effective dimensionality of the value subspace (participation ratio).
    ///
    /// PR = (Σλ_i)² / Σλ_i²  where λ_i are eigenvalues of the n×n cosine matrix.
    /// PR = 1 means all values collapsed to one direction.
    /// PR = n means values span n independent directions.
    pub effective_dimensionality: f32,
    /// Sorted eigenvalues of the pairwise cosine matrix (descending).
    /// Useful for visualising the spectrum and detecting collapse.
    pub eigenspectrum: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Thresholds for coherence analysis.
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    /// Causal cosine below this → opposed / antonym (default: -0.5).
    pub antonym_threshold: f32,
    /// Causal cosine above this → aligned / synonym (default: 0.8).
    pub synonym_threshold: f32,
    /// Denominator for severity scaling.
    ///
    /// Default: `1.0 + antonym_threshold` (full theoretical range).
    /// For real models with centered embeddings, set to the standard
    /// deviation of pairwise cosines (~0.10) so that small deviations
    /// past the threshold produce meaningful severities.
    pub severity_scale: Option<f32>,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            antonym_threshold: -0.5,
            synonym_threshold: 0.8,
            severity_scale: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Core analysis functions
// ---------------------------------------------------------------------------

/// Compute the causal cosine similarity between two vectors.
///
/// cos_Φ(u, v) = ⟨u, v⟩_Φ / (‖u‖_Φ · ‖v‖_Φ)
///
/// where ⟨u, v⟩_Φ = uᵀΦv and ‖u‖_Φ = √(uᵀΦu).
pub fn causal_cosine(
    u: &[f32],
    v: &[f32],
    geometry: &CausalGeometry,
) -> Result<f32, GeometryError> {
    let dot = geometry.inner_product(u, v)?;
    let norm_u = geometry.inner_product(u, u)?.sqrt();
    let norm_v = geometry.inner_product(v, v)?.sqrt();

    if norm_u < f32::EPSILON || norm_v < f32::EPSILON {
        return Ok(0.0); // zero vector → undefined, treat as orthogonal
    }

    Ok((dot / (norm_u * norm_v)).clamp(-1.0, 1.0))
}

/// Compute the causal distance between two vectors: ‖u - v‖_Φ.
pub fn causal_distance(
    u: &[f32],
    v: &[f32],
    geometry: &CausalGeometry,
) -> Result<f32, GeometryError> {
    let diff: Vec<f32> = u.iter().zip(v.iter()).map(|(a, b)| a - b).collect();
    let quad = geometry.inner_product(&diff, &diff)?;
    Ok(if quad > 0.0 { quad.sqrt() } else { 0.0 })
}

/// Classify a pairwise relationship based on causal cosine similarity.
fn classify_relation(causal_cos: f32, config: &CoherenceConfig) -> RelationType {
    if causal_cos < config.antonym_threshold {
        RelationType::Opposed
    } else if causal_cos > config.synonym_threshold {
        RelationType::Aligned
    } else {
        RelationType::Independent
    }
}

/// Compute all pairwise relations between resolved values.
pub fn pairwise_relations(
    values: &[ResolvedValue],
    geometry: &CausalGeometry,
    config: &CoherenceConfig,
) -> Result<Vec<PairwiseRelation>, IncoherenceError> {
    let n = values.len();
    let mut relations = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let cos = causal_cosine(
                &values[i].embedding,
                &values[j].embedding,
                geometry,
            )?;
            let dist = causal_distance(
                &values[i].embedding,
                &values[j].embedding,
                geometry,
            )?;
            let relation = classify_relation(cos, config);

            relations.push(PairwiseRelation {
                term_a: values[i].term.clone(),
                term_b: values[j].term.clone(),
                causal_cosine: cos,
                causal_distance: dist,
                relation,
            });
        }
    }

    Ok(relations)
}

/// Extract contradictions from pairwise relations.
pub fn find_contradictions(
    relations: &[PairwiseRelation],
    config: &CoherenceConfig,
) -> Vec<Contradiction> {
    relations
        .iter()
        .filter(|r| r.relation == RelationType::Opposed)
        .map(|r| {
            // Severity: how far past the threshold the cosine is.
            //
            // Default scale: 1.0 + threshold (full theoretical range).
            // For real models with centered embeddings, use severity_scale
            // (typically the std of pairwise cosines) so that small but
            // statistically significant deviations produce meaningful scores.
            let scale = config.severity_scale
                .unwrap_or(1.0 + config.antonym_threshold);
            let severity = if scale > f32::EPSILON {
                (config.antonym_threshold - r.causal_cosine) / scale
            } else {
                1.0
            };

            let angle_rad = r.causal_cosine.clamp(-1.0, 1.0).acos();
            let angle_degrees = angle_rad.to_degrees();

            Contradiction {
                term_a: r.term_a.clone(),
                term_b: r.term_b.clone(),
                severity: severity.clamp(0.0, 1.0),
                causal_cosine: r.causal_cosine,
                angle_degrees,
            }
        })
        .collect()
}

/// Extract redundancies from pairwise relations.
pub fn find_redundancies(
    relations: &[PairwiseRelation],
    _config: &CoherenceConfig,
) -> Vec<Redundancy> {
    relations
        .iter()
        .filter(|r| r.relation == RelationType::Aligned)
        .map(|r| Redundancy {
            term_a: r.term_a.clone(),
            term_b: r.term_b.clone(),
            similarity: r.causal_cosine,
            causal_cosine: r.causal_cosine,
        })
        .collect()
}

/// Compute the overall coherence score ∈ [0, 1].
///
/// Score = (1 - max_severity) × (1 - contradiction_ratio).
///
/// Two independent signals:
///   - max_severity: a single severe contradiction tanks the score.
///   - contradiction_ratio: many mild contradictions also reduce it.
///
/// A system with no contradictions scores 1.0.
/// A system with one perfect antonym pair (severity 1.0) scores 0.0.
pub fn coherence_score(
    relations: &[PairwiseRelation],
    contradictions: &[Contradiction],
) -> f32 {
    if relations.is_empty() {
        return 1.0;
    }

    if contradictions.is_empty() {
        return 1.0;
    }

    let max_severity = contradictions
        .iter()
        .map(|c| c.severity)
        .fold(0.0f32, f32::max);

    let contradiction_ratio = contradictions.len() as f32 / relations.len() as f32;

    ((1.0 - max_severity) * (1.0 - contradiction_ratio)).clamp(0.0, 1.0)
}

/// Compute the participation ratio (effective dimensionality) from pairwise cosines.
///
/// Builds the n×n cosine matrix C where C[i][j] = causal_cosine(v_i, v_j),
/// computes its eigenvalues, and returns:
///   PR = (Σλ_i)² / Σλ_i²
///
/// PR ∈ [1, n]:
///   - PR ≈ 1: all values collapsed onto a single direction (manifold collapse)
///   - PR ≈ n: values span n independent directions (rich structure)
///
/// Also returns the sorted eigenspectrum (descending) for visualisation.
pub fn participation_ratio(
    values: &[ResolvedValue],
    geometry: &CausalGeometry,
) -> Result<(f32, Vec<f32>), IncoherenceError> {
    let n = values.len();
    if n < 2 {
        return Ok((1.0, vec![1.0]));
    }

    // Build n×n cosine matrix
    let mut cosine_matrix = faer::Mat::zeros(n, n);
    for i in 0..n {
        cosine_matrix[(i, i)] = 1.0; // self-similarity
        for j in (i + 1)..n {
            let cos = causal_cosine(
                &values[i].embedding,
                &values[j].embedding,
                geometry,
            )?;
            cosine_matrix[(i, j)] = cos as f64;
            cosine_matrix[(j, i)] = cos as f64;
        }
    }

    // Eigendecomposition of symmetric matrix
    let eigenvalues = cosine_matrix.selfadjoint_eigendecomposition(faer::Side::Lower).s().column_vector().try_as_slice().unwrap().to_vec();

    // Clamp negative eigenvalues to zero (numerical noise)
    let lambdas: Vec<f64> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();

    let sum: f64 = lambdas.iter().sum();
    let sum_sq: f64 = lambdas.iter().map(|l| l * l).sum();

    let pr = if sum_sq > 1e-15 {
        (sum * sum / sum_sq) as f32
    } else {
        1.0
    };

    // Return sorted descending eigenspectrum
    let mut spectrum: Vec<f32> = lambdas.iter().map(|&l| l as f32).collect();
    spectrum.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok((pr, spectrum))
}

/// Run the full coherence analysis.
pub fn analyse(
    values: &[ResolvedValue],
    num_unresolved: usize,
    geometry: &CausalGeometry,
    config: &CoherenceConfig,
) -> Result<CoherenceAnalysis, IncoherenceError> {
    if values.len() < 2 {
        return Err(IncoherenceError::EmptyInput(
            "need at least 2 resolved values for coherence analysis",
        ));
    }

    let pairwise = pairwise_relations(values, geometry, config)?;
    let contradictions = find_contradictions(&pairwise, config);
    let redundancies = find_redundancies(&pairwise, config);
    let score = coherence_score(&pairwise, &contradictions);
    let (eff_dim, eigenspectrum) = participation_ratio(values, geometry)?;

    Ok(CoherenceAnalysis {
        pairwise,
        contradictions,
        redundancies,
        coherence_score: score,
        num_terms: values.len(),
        num_unresolved,
        effective_dimensionality: eff_dim,
        eigenspectrum,
    })
}

// ---------------------------------------------------------------------------
// Euclidean fallback (when no unembedding matrix is available)
// ---------------------------------------------------------------------------

/// Ordinary cosine similarity (no geometry weighting).
/// Re-exported from got-core for backward compatibility.
pub use got_core::geometry::euclidean_cosine;

// ---------------------------------------------------------------------------
// Conversation-level types
// ---------------------------------------------------------------------------

/// A single turn in a conversation with coherence state at that point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Turn index (0-based).
    pub turn: usize,
    /// Speaker identifier.
    pub speaker: String,
    /// The message text.
    pub text: String,
    /// Value terms introduced by this specific message.
    pub values_introduced: Vec<String>,
    /// All values accumulated up to and including this turn.
    pub cumulative_values: Vec<String>,
    /// Coherence analysis at this point in the conversation.
    pub analysis: CoherenceAnalysis,
    /// Contradictions that first appeared at *this* turn.
    pub new_contradictions: Vec<Contradiction>,
}

/// Full conversation analysis: coherence state at every turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationAnalysis {
    /// Per-turn coherence snapshots in chronological order.
    pub turns: Vec<ConversationTurn>,
}

impl ConversationAnalysis {
    /// Coherence scores over time, one per turn.
    pub fn score_series(&self) -> Vec<f32> {
        self.turns.iter().map(|t| t.analysis.coherence_score).collect()
    }

    /// Turn at which the first contradiction appeared, if any.
    pub fn first_contradiction_turn(&self) -> Option<usize> {
        self.turns.iter().position(|t| !t.new_contradictions.is_empty())
    }

    /// Final coherence score.
    pub fn final_score(&self) -> f32 {
        self.turns.last().map(|t| t.analysis.coherence_score).unwrap_or(1.0)
    }

    /// Total number of unique contradictions across the conversation.
    pub fn total_contradictions(&self) -> usize {
        self.turns.last().map(|t| t.analysis.contradictions.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use got_core::UnembeddingMatrix;

    fn test_geometry() -> CausalGeometry {
        // Identity-like geometry: Φ ≈ I (orthonormal unembedding rows)
        let u = UnembeddingMatrix::new(4, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]).unwrap();
        CausalGeometry::from_unembedding(&u, 1e-6)
    }

    fn test_values() -> Vec<ResolvedValue> {
        vec![
            ResolvedValue {
                term: "innovation".into(),
                normalised: "innovation".into(),
                embedding: vec![1.0, 0.5, 0.0],
            },
            ResolvedValue {
                term: "risk-aversion".into(),
                normalised: "risk-aversion".into(),
                embedding: vec![-1.0, -0.5, 0.0],  // opposite direction
            },
            ResolvedValue {
                term: "transparency".into(),
                normalised: "transparency".into(),
                embedding: vec![0.0, 1.0, 0.5],
            },
            ResolvedValue {
                term: "confidentiality".into(),
                normalised: "confidentiality".into(),
                embedding: vec![0.0, -1.0, -0.5],  // opposite direction
            },
        ]
    }

    #[test]
    fn causal_cosine_of_identical_is_one() {
        let geom = test_geometry();
        let v = [1.0, 0.0, 0.0];
        let cos = causal_cosine(&v, &v, &geom).unwrap();
        assert!((cos - 1.0).abs() < 1e-4, "self-cosine should be ~1.0, got {cos}");
    }

    #[test]
    fn causal_cosine_of_opposites_is_negative() {
        let geom = test_geometry();
        let a = [1.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0];
        let cos = causal_cosine(&a, &b, &geom).unwrap();
        assert!(cos < -0.9, "opposite vectors should have cos ≈ -1, got {cos}");
    }

    #[test]
    fn causal_cosine_of_orthogonal_is_near_zero() {
        let geom = test_geometry();
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let cos = causal_cosine(&a, &b, &geom).unwrap();
        // Not exactly 0 because Φ ≠ I (the [1,1,1] row adds cross terms).
        // With Φ = I + 11ᵀ: cos_Φ([1,0,0], [0,1,0]) = 1/√2·√2 = 0.5
        assert!(cos.abs() < 0.6, "orthogonal vectors should be below 0.6, got {cos}");
    }

    #[test]
    fn pairwise_detects_opposed_pairs() {
        let geom = test_geometry();
        let values = test_values();
        let config = CoherenceConfig::default();
        let relations = pairwise_relations(&values, &geom, &config).unwrap();

        // Should have n*(n-1)/2 = 6 pairs
        assert_eq!(relations.len(), 6);

        // innovation vs risk-aversion should be opposed
        let inno_risk = relations.iter().find(|r|
            (r.term_a == "innovation" && r.term_b == "risk-aversion") ||
            (r.term_a == "risk-aversion" && r.term_b == "innovation")
        ).unwrap();
        assert_eq!(inno_risk.relation, RelationType::Opposed,
            "innovation/risk-aversion should be opposed, cos={}", inno_risk.causal_cosine);

        // transparency vs confidentiality should be opposed
        let trans_conf = relations.iter().find(|r|
            (r.term_a == "transparency" && r.term_b == "confidentiality") ||
            (r.term_a == "confidentiality" && r.term_b == "transparency")
        ).unwrap();
        assert_eq!(trans_conf.relation, RelationType::Opposed,
            "transparency/confidentiality should be opposed, cos={}", trans_conf.causal_cosine);
    }

    #[test]
    fn contradictions_detected_from_opposed_values() {
        let geom = test_geometry();
        let values = test_values();
        let config = CoherenceConfig::default();
        let relations = pairwise_relations(&values, &geom, &config).unwrap();
        let contradictions = find_contradictions(&relations, &config);

        assert!(contradictions.len() >= 2,
            "should detect at least 2 contradictions, got {}", contradictions.len());

        for c in &contradictions {
            assert!(c.severity > 0.0, "contradiction severity should be > 0");
            assert!(c.angle_degrees > 90.0, "angle should be > 90° for opposed terms");
        }
    }

    #[test]
    fn coherence_score_penalised_by_contradictions() {
        let geom = test_geometry();
        let values = test_values();
        let config = CoherenceConfig::default();
        let analysis = analyse(&values, 0, &geom, &config).unwrap();

        assert!(analysis.coherence_score < 1.0,
            "system with contradictions should score < 1.0, got {}", analysis.coherence_score);
        assert!(analysis.coherence_score > 0.0,
            "score should still be > 0, got {}", analysis.coherence_score);
    }

    #[test]
    fn coherent_system_scores_high() {
        let geom = test_geometry();
        // Values that are orthogonal (independent), not opposed
        let values = vec![
            ResolvedValue {
                term: "honesty".into(),
                normalised: "honesty".into(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            ResolvedValue {
                term: "kindness".into(),
                normalised: "kindness".into(),
                embedding: vec![0.0, 1.0, 0.0],
            },
            ResolvedValue {
                term: "courage".into(),
                normalised: "courage".into(),
                embedding: vec![0.0, 0.0, 1.0],
            },
        ];
        let config = CoherenceConfig::default();
        let analysis = analyse(&values, 0, &geom, &config).unwrap();

        assert_eq!(analysis.contradictions.len(), 0, "no contradictions expected");
        assert!((analysis.coherence_score - 1.0).abs() < 1e-4,
            "coherent system should score ~1.0, got {}", analysis.coherence_score);
    }

    #[test]
    fn too_few_values_rejected() {
        let geom = test_geometry();
        let values = vec![ResolvedValue {
            term: "lonely".into(),
            normalised: "lonely".into(),
            embedding: vec![1.0, 0.0, 0.0],
        }];
        let config = CoherenceConfig::default();
        assert!(analyse(&values, 0, &geom, &config).is_err());
    }

    #[test]
    fn euclidean_cosine_matches_for_unit_vectors() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let cos = euclidean_cosine(&a, &b);
        assert!(cos.abs() < 1e-6, "orthogonal unit vectors should have cos=0");
    }

    #[test]
    fn zero_vector_handled() {
        let geom = test_geometry();
        let zero = [0.0, 0.0, 0.0];
        let v = [1.0, 0.0, 0.0];
        let cos = causal_cosine(&zero, &v, &geom).unwrap();
        assert_eq!(cos, 0.0, "zero vector should give cosine 0");
    }
}
