// ---------------------------------------------------------------------------
// Value-ordering coherence scoring.
//
// Given a hidden state h and a set of value-ordering constraints {(uᵢ, uⱼ)},
// computes:
//
//   C(h) = (1/n) Σ σ(α · (⟨uᵢ, h⟩_c - ⟨uⱼ, h⟩_c))
//
// where ⟨u, h⟩_c = uᵀΦh is the causal inner product and σ is the sigmoid.
// C(h) ∈ [0, 1]. Near 1 = all constraints satisfied. Near 0 = most violated.
// ---------------------------------------------------------------------------

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::geometry::{CausalGeometry, GeometryError};

/// A value-ordering constraint: direction `dominant` should have
/// higher causal activation than `subordinate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueConstraint {
    pub dominant: Vec<f32>,
    pub subordinate: Vec<f32>,
    pub label: String,
}

/// Set of constraints defining a value ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueOrdering {
    pub constraints: Vec<ValueConstraint>,
}

impl ValueOrdering {
    /// Load from a JSON file.
    ///
    /// Expected format:
    /// ```json
    /// [
    ///   {"dominant": [0.1, ...], "subordinate": [0.2, ...], "label": "honesty > deception"},
    ///   ...
    /// ]
    /// ```
    pub fn from_json(path: &Path) -> Result<Self, CoherenceError> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| CoherenceError::Io(format!("{path:?}: {e}")))?;
        let constraints: Vec<ValueConstraint> =
            serde_json::from_str(&data).map_err(|e| CoherenceError::Json(e.to_string()))?;
        if constraints.is_empty() {
            return Err(CoherenceError::Empty);
        }
        Ok(Self { constraints })
    }
}

/// Sigmoid function σ(x) = 1 / (1 + e^{-x}).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute coherence score for a single hidden state.
///
/// Returns C(h) ∈ [0, 1].
pub fn coherence_score(
    h: &[f32],
    ordering: &ValueOrdering,
    geometry: &CausalGeometry,
    sharpness: f32,
) -> Result<f32, GeometryError> {
    let n = ordering.constraints.len();
    if n == 0 {
        return Ok(1.0);
    }

    let mut sum = 0.0f32;
    for c in &ordering.constraints {
        let dot_dom = geometry.inner_product(&c.dominant, h)?;
        let dot_sub = geometry.inner_product(&c.subordinate, h)?;
        sum += sigmoid(sharpness * (dot_dom - dot_sub));
    }

    Ok(sum / n as f32)
}

/// Per-position coherence report for a sequence of hidden states.
#[derive(Debug, Clone)]
pub struct CoherenceReport {
    pub per_position: Vec<f32>,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
    /// (position index, constraint label) for positions where C < 0.5.
    pub violated_constraints: Vec<(usize, String)>,
}

/// Compute coherence scores for a sequence of hidden states.
///
/// Returns per-position scores and aggregate statistics.
pub fn conversational_coherence(
    hidden_states: &[Vec<f32>],
    ordering: &ValueOrdering,
    geometry: &CausalGeometry,
    sharpness: f32,
) -> Result<CoherenceReport, GeometryError> {
    let mut per_position = Vec::with_capacity(hidden_states.len());
    let mut violated_constraints = Vec::new();

    for (pos, h) in hidden_states.iter().enumerate() {
        let score = coherence_score(h, ordering, geometry, sharpness)?;
        per_position.push(score);

        // Check individual constraints at this position for violations.
        if score < 0.5 {
            for c in &ordering.constraints {
                let dot_dom = geometry.inner_product(&c.dominant, h)?;
                let dot_sub = geometry.inner_product(&c.subordinate, h)?;
                if sigmoid(sharpness * (dot_dom - dot_sub)) < 0.5 {
                    violated_constraints.push((pos, c.label.clone()));
                }
            }
        }
    }

    let mean = if per_position.is_empty() {
        1.0
    } else {
        per_position.iter().sum::<f32>() / per_position.len() as f32
    };
    let min = per_position.iter().cloned().reduce(f32::min).unwrap_or(1.0);
    let max = per_position.iter().cloned().reduce(f32::max).unwrap_or(1.0);

    Ok(CoherenceReport {
        per_position,
        mean,
        min,
        max,
        violated_constraints,
    })
}

#[derive(Debug)]
pub enum CoherenceError {
    Io(String),
    Json(String),
    Empty,
    Geometry(GeometryError),
}

impl std::fmt::Display for CoherenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoherenceError::Io(e) => write!(f, "IO error: {e}"),
            CoherenceError::Json(e) => write!(f, "JSON error: {e}"),
            CoherenceError::Empty => write!(f, "empty value ordering"),
            CoherenceError::Geometry(e) => write!(f, "geometry error: {e}"),
        }
    }
}

impl std::error::Error for CoherenceError {}

impl From<GeometryError> for CoherenceError {
    fn from(e: GeometryError) -> Self {
        CoherenceError::Geometry(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_geometry(dim: usize) -> CausalGeometry {
        let mut gram = vec![0.0f32; dim * dim];
        for i in 0..dim {
            gram[i * dim + i] = 1.0;
        }
        CausalGeometry::from_raw_gram(gram, dim).unwrap()
    }

    fn make_ordering(pairs: Vec<(Vec<f32>, Vec<f32>, &str)>) -> ValueOrdering {
        ValueOrdering {
            constraints: pairs
                .into_iter()
                .map(|(d, s, l)| ValueConstraint {
                    dominant: d,
                    subordinate: s,
                    label: l.to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn coherence_all_satisfied() {
        let geo = identity_geometry(4);
        // dominant direction = [1,0,0,0], subordinate = [0,1,0,0]
        // h is strongly aligned with dominant
        let ordering = make_ordering(vec![
            (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0], "a > b"),
            (vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0], "c > d"),
        ]);
        let h = vec![5.0, 0.0, 5.0, 0.0]; // dot(dom_a, h)=5, dot(sub_a, h)=0 → σ(5)≈1
        let score = coherence_score(&h, &ordering, &geo, 1.0).unwrap();
        assert!(score > 0.9, "expected > 0.9, got {score}");
    }

    #[test]
    fn coherence_all_violated() {
        let geo = identity_geometry(4);
        let ordering = make_ordering(vec![
            (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0], "a > b"),
            (vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0], "c > d"),
        ]);
        // h is strongly aligned with subordinate directions
        let h = vec![0.0, 5.0, 0.0, 5.0]; // dot(dom, h)=0, dot(sub, h)=5 → σ(-5)≈0
        let score = coherence_score(&h, &ordering, &geo, 1.0).unwrap();
        assert!(score < 0.1, "expected < 0.1, got {score}");
    }

    #[test]
    fn coherence_mixed() {
        let geo = identity_geometry(4);
        let ordering = make_ordering(vec![
            (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0], "a > b"),
            (vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0], "c > d"),
        ]);
        // First constraint satisfied (h[0]=5 > h[1]=0), second violated (h[2]=0 < h[3]=5)
        let h = vec![5.0, 0.0, 0.0, 5.0];
        let score = coherence_score(&h, &ordering, &geo, 1.0).unwrap();
        assert!(
            (score - 0.5).abs() < 0.05,
            "expected ~0.5, got {score}"
        );
    }

    #[test]
    fn coherence_sharpness_effect() {
        let geo = identity_geometry(4);
        let ordering = make_ordering(vec![(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            "a > b",
        )]);
        // Small margin: dot diff = 0.5
        let h = vec![1.0, 0.5, 0.0, 0.0];

        let score_low = coherence_score(&h, &ordering, &geo, 0.1).unwrap();
        let score_high = coherence_score(&h, &ordering, &geo, 10.0).unwrap();

        // Both above 0.5 (constraint satisfied), but high sharpness → closer to 1.0
        assert!(score_low > 0.5);
        assert!(score_high > score_low, "higher α should push score further from 0.5");
        assert!(score_high > 0.99, "α=10 with margin 0.5 should be near 1.0");
    }

    #[test]
    fn conversational_coherence_reports_violations() {
        let geo = identity_geometry(4);
        let ordering = make_ordering(vec![(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            "a > b",
        )]);

        let states = vec![
            vec![5.0, 0.0, 0.0, 0.0], // satisfied
            vec![0.0, 5.0, 0.0, 0.0], // violated
            vec![3.0, 0.0, 0.0, 0.0], // satisfied
        ];

        let report = conversational_coherence(&states, &ordering, &geo, 1.0).unwrap();

        assert_eq!(report.per_position.len(), 3);
        assert!(report.per_position[0] > 0.9);
        assert!(report.per_position[1] < 0.1);
        assert!(report.per_position[2] > 0.9);

        assert!(!report.violated_constraints.is_empty());
        assert_eq!(report.violated_constraints[0].0, 1); // position 1
        assert_eq!(report.violated_constraints[0].1, "a > b");
    }
}
