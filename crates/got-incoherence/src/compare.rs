// ---------------------------------------------------------------------------
// Model comparison: measure how instruction-tuning / RLHF changes value geometry.
//
// Given two sets of resolved value embeddings (e.g. base model vs instruct),
// computes:
//   - Participation ratio for each (effective dimensionality)
//   - Per-term drift (how much each value embedding moved)
//   - Pairwise cosine deltas (which relationships changed most)
//   - Frobenius distance between cosine matrices
//
// This is the measurement tool for Conjecture 3: "RLHF flattens value geometry."
// ---------------------------------------------------------------------------

use got_core::geometry::CausalGeometry;
use serde::{Deserialize, Serialize};

use crate::coherence::{causal_cosine, euclidean_cosine, participation_ratio};
use crate::embeddings::ResolvedValue;
use crate::IncoherenceError;

/// Result of comparing two models' value geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Name/label for the base model.
    pub base_label: String,
    /// Name/label for the compared model.
    pub compared_label: String,

    /// Number of value terms resolved in both models.
    pub shared_terms: usize,
    /// Terms resolved in base but not compared.
    pub base_only_terms: Vec<String>,
    /// Terms resolved in compared but not base.
    pub compared_only_terms: Vec<String>,

    /// Effective dimensionality (participation ratio) for each model.
    pub base_dimensionality: f32,
    pub compared_dimensionality: f32,
    /// Change in dimensionality: compared - base. Negative = collapse.
    pub dimensionality_delta: f32,

    /// Eigenspectra for each model (sorted descending).
    pub base_eigenspectrum: Vec<f32>,
    pub compared_eigenspectrum: Vec<f32>,

    /// Per-term drift: how much each value embedding moved (cosine distance).
    pub term_drifts: Vec<TermDrift>,

    /// Mean cosine distance across all shared terms.
    pub mean_term_drift: f32,

    /// Pairwise relationship changes (sorted by absolute delta, descending).
    pub relationship_changes: Vec<RelationshipChange>,

    /// Frobenius distance between the two n×n cosine matrices.
    pub cosine_matrix_distance: f32,
}

/// How much a single value term's embedding changed between models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermDrift {
    pub term: String,
    /// Cosine similarity between the term's embedding in base vs compared model.
    /// 1.0 = identical direction, 0.0 = orthogonal, -1.0 = reversed.
    pub cosine_similarity: f32,
    /// Norm of the embedding in the base model.
    pub base_norm: f32,
    /// Norm of the embedding in the compared model.
    pub compared_norm: f32,
}

/// How a pairwise relationship changed between models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipChange {
    pub term_a: String,
    pub term_b: String,
    /// Cosine between term_a and term_b in the base model.
    pub base_cosine: f32,
    /// Cosine between term_a and term_b in the compared model.
    pub compared_cosine: f32,
    /// Change: compared - base. Positive = more aligned, negative = more opposed.
    pub delta: f32,
}

/// Compare two models' value geometry.
///
/// Both models must use identity geometry (Φ = I) for the comparison to be
/// meaningful across different hidden dimensions. If the models share the same
/// hidden dimension, causal geometry can optionally be used.
pub fn compare_models(
    base_label: &str,
    base_values: &[ResolvedValue],
    base_geometry: &CausalGeometry,
    compared_label: &str,
    compared_values: &[ResolvedValue],
    compared_geometry: &CausalGeometry,
) -> Result<ModelComparison, IncoherenceError> {
    // Find shared terms (present in both models)
    let base_terms: std::collections::HashSet<&str> =
        base_values.iter().map(|v| v.normalised.as_str()).collect();
    let compared_terms: std::collections::HashSet<&str> =
        compared_values.iter().map(|v| v.normalised.as_str()).collect();

    let shared: Vec<&str> = base_terms.intersection(&compared_terms).copied().collect();
    let base_only: Vec<String> = base_terms.difference(&compared_terms).map(|s| s.to_string()).collect();
    let compared_only: Vec<String> = compared_terms.difference(&base_terms).map(|s| s.to_string()).collect();

    if shared.len() < 2 {
        return Err(IncoherenceError::EmptyInput(
            "need at least 2 shared terms between models for comparison",
        ));
    }

    // Collect shared-term embeddings in matched order
    let base_shared: Vec<&ResolvedValue> = shared.iter()
        .map(|t| base_values.iter().find(|v| v.normalised == *t).unwrap())
        .collect();
    let compared_shared: Vec<&ResolvedValue> = shared.iter()
        .map(|t| compared_values.iter().find(|v| v.normalised == *t).unwrap())
        .collect();

    // Owned copies for participation_ratio (which needs &[ResolvedValue])
    let base_owned: Vec<ResolvedValue> = base_shared.iter().map(|v| (*v).clone()).collect();
    let compared_owned: Vec<ResolvedValue> = compared_shared.iter().map(|v| (*v).clone()).collect();

    // Participation ratio for each
    let (base_pr, base_spectrum) = participation_ratio(&base_owned, base_geometry)?;
    let (compared_pr, compared_spectrum) = participation_ratio(&compared_owned, compared_geometry)?;

    // Per-term drift (euclidean cosine between same term across models)
    // Only meaningful when models share the same hidden dimension
    let same_dim = base_geometry.hidden_dim() == compared_geometry.hidden_dim();
    let term_drifts: Vec<TermDrift> = if same_dim {
        shared.iter().enumerate().map(|(i, term)| {
            let base_emb = &base_shared[i].embedding;
            let comp_emb = &compared_shared[i].embedding;
            let cos = euclidean_cosine(base_emb, comp_emb);
            let base_norm: f32 = base_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let comp_norm: f32 = comp_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            TermDrift {
                term: term.to_string(),
                cosine_similarity: cos,
                base_norm,
                compared_norm: comp_norm,
            }
        }).collect()
    } else {
        vec![] // Can't compare embeddings across different dimensions
    };

    let mean_term_drift = if term_drifts.is_empty() {
        0.0
    } else {
        term_drifts.iter().map(|d| 1.0 - d.cosine_similarity).sum::<f32>() / term_drifts.len() as f32
    };

    // Pairwise cosine changes
    let n = shared.len();
    let mut relationship_changes = Vec::with_capacity(n * (n - 1) / 2);
    let mut frobenius_sq = 0.0f32;

    for i in 0..n {
        for j in (i + 1)..n {
            let base_cos = causal_cosine(
                &base_shared[i].embedding,
                &base_shared[j].embedding,
                base_geometry,
            )?;
            let comp_cos = causal_cosine(
                &compared_shared[i].embedding,
                &compared_shared[j].embedding,
                compared_geometry,
            )?;
            let delta = comp_cos - base_cos;
            frobenius_sq += 2.0 * delta * delta; // ×2: symmetric matrix, upper+lower

            relationship_changes.push(RelationshipChange {
                term_a: shared[i].to_string(),
                term_b: shared[j].to_string(),
                base_cosine: base_cos,
                compared_cosine: comp_cos,
                delta,
            });
        }
    }

    // Sort by absolute delta descending (biggest changes first)
    relationship_changes.sort_by(|a, b| {
        b.delta.abs().partial_cmp(&a.delta.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });

    let cosine_matrix_distance = frobenius_sq.sqrt();

    Ok(ModelComparison {
        base_label: base_label.to_string(),
        compared_label: compared_label.to_string(),
        shared_terms: shared.len(),
        base_only_terms: base_only,
        compared_only_terms: compared_only,
        base_dimensionality: base_pr,
        compared_dimensionality: compared_pr,
        dimensionality_delta: compared_pr - base_pr,
        base_eigenspectrum: base_spectrum,
        compared_eigenspectrum: compared_spectrum,
        term_drifts,
        mean_term_drift,
        relationship_changes,
        cosine_matrix_distance,
    })
}

/// Format a comparison as a human-readable report.
pub fn render_comparison(comp: &ModelComparison) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Model Comparison: {} vs {}\n",
        comp.base_label, comp.compared_label
    ));
    out.push_str(&"=".repeat(60));
    out.push('\n');

    out.push_str(&format!("\nShared terms: {}\n", comp.shared_terms));
    if !comp.base_only_terms.is_empty() {
        out.push_str(&format!("  Base only: {}\n", comp.base_only_terms.join(", ")));
    }
    if !comp.compared_only_terms.is_empty() {
        out.push_str(&format!("  Compared only: {}\n", comp.compared_only_terms.join(", ")));
    }

    out.push_str(&format!(
        "\nEffective Dimensionality (Participation Ratio):\n  {}: {:.2} / {}\n  {}: {:.2} / {}\n  Delta: {:.2} ({})\n",
        comp.base_label,
        comp.base_dimensionality,
        comp.shared_terms,
        comp.compared_label,
        comp.compared_dimensionality,
        comp.shared_terms,
        comp.dimensionality_delta,
        if comp.dimensionality_delta < -0.5 { "COLLAPSE DETECTED" }
        else if comp.dimensionality_delta < 0.0 { "slight contraction" }
        else if comp.dimensionality_delta > 0.5 { "expansion" }
        else { "stable" }
    ));

    // Eigenspectrum summary
    out.push_str("\nEigenspectrum (top 5):\n");
    let max_show = 5.min(comp.base_eigenspectrum.len());
    out.push_str("  Base:     ");
    for v in &comp.base_eigenspectrum[..max_show] {
        out.push_str(&format!("{:.3} ", v));
    }
    out.push_str("\n  Compared: ");
    let max_show = 5.min(comp.compared_eigenspectrum.len());
    for v in &comp.compared_eigenspectrum[..max_show] {
        out.push_str(&format!("{:.3} ", v));
    }
    out.push('\n');

    // Per-term drift
    if !comp.term_drifts.is_empty() {
        out.push_str(&format!(
            "\nPer-term drift (mean cosine distance: {:.4}):\n",
            comp.mean_term_drift,
        ));
        let mut sorted_drifts = comp.term_drifts.clone();
        sorted_drifts.sort_by(|a, b| {
            a.cosine_similarity.partial_cmp(&b.cosine_similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        for d in &sorted_drifts {
            let bar_len = ((1.0 - d.cosine_similarity) * 40.0) as usize;
            let bar: String = "#".repeat(bar_len.min(40));
            out.push_str(&format!(
                "  {:<20} cos={:.4}  {}\n",
                d.term, d.cosine_similarity, bar
            ));
        }
    }

    // Top relationship changes
    out.push_str(&format!(
        "\nCosine matrix Frobenius distance: {:.4}\n",
        comp.cosine_matrix_distance,
    ));
    let top_n = 10.min(comp.relationship_changes.len());
    if top_n > 0 {
        out.push_str(&format!("\nTop {} relationship changes:\n", top_n));
        for rc in &comp.relationship_changes[..top_n] {
            let arrow = if rc.delta > 0.0 { "+" } else { "" };
            out.push_str(&format!(
                "  {:<15} <-> {:<15}  {:.3} -> {:.3}  ({}{:.3})\n",
                rc.term_a, rc.term_b, rc.base_cosine, rc.compared_cosine, arrow, rc.delta
            ));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::ResolvedValue;
    use got_core::geometry::CausalGeometry;

    fn identity_geometry(dim: usize) -> CausalGeometry {
        let mut gram = vec![0.0f32; dim * dim];
        for i in 0..dim {
            gram[i * dim + i] = 1.0;
        }
        CausalGeometry::from_raw_gram(gram, dim).unwrap()
    }

    fn make_value(term: &str, embedding: Vec<f32>) -> ResolvedValue {
        ResolvedValue {
            term: term.to_string(),
            normalised: term.to_lowercase(),
            embedding,
        }
    }

    #[test]
    fn identical_models_have_zero_delta() {
        let geom = identity_geometry(3);
        let values = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            make_value("wisdom", vec![0.0, 0.0, 1.0]),
        ];

        let result = compare_models(
            "base", &values, &geom,
            "same", &values, &geom,
        ).unwrap();

        assert_eq!(result.dimensionality_delta, 0.0);
        assert_eq!(result.cosine_matrix_distance, 0.0);
        assert!(result.term_drifts.iter().all(|d| (d.cosine_similarity - 1.0).abs() < 1e-6));
    }

    #[test]
    fn collapsed_model_has_lower_dimensionality() {
        let geom = identity_geometry(3);

        // Base: orthogonal values — max spread
        let base = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            make_value("wisdom", vec![0.0, 0.0, 1.0]),
        ];

        // Compared: all values collapsed toward same direction
        let collapsed = vec![
            make_value("honesty", vec![1.0, 0.1, 0.0]),
            make_value("courage", vec![0.9, 0.2, 0.0]),
            make_value("wisdom", vec![0.8, 0.3, 0.0]),
        ];

        let result = compare_models(
            "base", &base, &geom,
            "collapsed", &collapsed, &geom,
        ).unwrap();

        assert!(result.base_dimensionality > result.compared_dimensionality,
            "base PR ({}) should be > collapsed PR ({})",
            result.base_dimensionality, result.compared_dimensionality);
        assert!(result.dimensionality_delta < 0.0, "delta should be negative");
    }

    #[test]
    fn different_vocab_tracked() {
        let geom = identity_geometry(3);
        let base = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            make_value("cowardice", vec![0.0, 0.0, 1.0]),
        ];
        let compared = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            // cowardice missing, but has wisdom
            make_value("wisdom", vec![0.5, 0.5, 0.0]),
        ];

        let result = compare_models("base", &base, &geom, "comp", &compared, &geom).unwrap();
        assert_eq!(result.shared_terms, 2);
        assert!(result.base_only_terms.contains(&"cowardice".to_string()));
        assert!(result.compared_only_terms.contains(&"wisdom".to_string()));
    }

    #[test]
    fn report_renders() {
        let geom = identity_geometry(3);
        let base = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            make_value("wisdom", vec![0.0, 0.0, 1.0]),
        ];
        let collapsed = vec![
            make_value("honesty", vec![1.0, 0.1, 0.0]),
            make_value("courage", vec![0.9, 0.2, 0.0]),
            make_value("wisdom", vec![0.8, 0.3, 0.0]),
        ];

        let result = compare_models("GPT-2 Base", &base, &geom, "GPT-2 RLHF", &collapsed, &geom).unwrap();
        let report = render_comparison(&result);
        assert!(report.contains("GPT-2 Base"));
        assert!(report.contains("GPT-2 RLHF"));
        assert!(report.contains("Participation Ratio"));
        assert!(report.contains("COLLAPSE DETECTED") || report.contains("contraction") || report.contains("stable"));
    }
}
