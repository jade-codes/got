// ---------------------------------------------------------------------------
// got-incoherence: detect geometric contradictions in stated value systems.
//
// When an organisation claims to value both X and Y, and X and Y are
// near-antonyms in the model's causal embedding space (cos_Φ ≈ -1),
// that's a measurable mathematical incoherence.
//
// Pipeline:
//   1. embeddings.rs — map value terms to vectors in ℝ^d
//   2. coherence.rs  — compute pairwise causal cosines, detect contradictions
//   3. report.rs     — format results for humans (text) or machines (JSON)
//
// Zero-training: no probes, no labels, no SGD.  Just geometry.
//
// Usage:
//   let source = PrecomputedEmbeddings::from_json(json)?;
//   let geometry = CausalGeometry::from_unembedding(&u, epsilon);
//   let report = analyse_value_system(&["innovation", "risk-aversion", ...], &source, &geometry, &config)?;
//   println!("{}", report::render_text(&report.analysis));
// ---------------------------------------------------------------------------

pub mod category;
pub mod coherence;
pub mod compare;
pub mod curvature;
pub mod embeddings;
pub mod report;
pub mod visual;

use got_core::geometry::{CausalGeometry, GeometryError};
use thiserror::Error;

use coherence::{CoherenceAnalysis, CoherenceConfig};
use embeddings::{EmbeddingSource, ResolvedValue};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum IncoherenceError {
    #[error("geometry error: {0}")]
    Geometry(#[from] GeometryError),
    #[error("vocabulary size mismatch: vocab has {vocab_len} tokens, matrix has {matrix_rows} rows")]
    VocabMismatch { vocab_len: usize, matrix_rows: usize },
    #[error("dimension inconsistency for term '{term}': expected {expected}, got {got}")]
    DimensionInconsistency {
        term: String,
        expected: usize,
        got: usize,
    },
    #[error("empty input: {0}")]
    EmptyInput(&'static str),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// Top-level API
// ---------------------------------------------------------------------------

/// Result of analysing a value system.
pub struct ValueSystemReport {
    /// The coherence analysis with scores and contradictions.
    pub analysis: CoherenceAnalysis,
    /// Terms that were successfully resolved to embeddings.
    pub resolved: Vec<ResolvedValue>,
    /// Terms that could not be embedded.
    pub unresolved: Vec<String>,
}

/// Analyse a list of value terms for geometric coherence.
///
/// This is the main entry point.  Takes raw term strings, resolves them
/// to embeddings, runs the coherence analysis, and returns the full report.
///
/// ```text
/// let report = analyse_value_system(
///     &["innovation", "risk-aversion", "transparency", "confidentiality"],
///     &embedding_source,
///     &geometry,
///     &CoherenceConfig::default(),
/// )?;
/// assert!(report.analysis.coherence_score < 1.0); // contradictions detected
/// ```
pub fn analyse_value_system(
    terms: &[&str],
    source: &dyn EmbeddingSource,
    geometry: &CausalGeometry,
    config: &CoherenceConfig,
) -> Result<ValueSystemReport, IncoherenceError> {
    if terms.is_empty() {
        return Err(IncoherenceError::EmptyInput("no value terms provided"));
    }

    // Step 1: Resolve terms to embeddings
    let (resolved, unresolved) = embeddings::resolve_values(terms, source);

    if resolved.len() < 2 {
        return Err(IncoherenceError::EmptyInput(
            "need at least 2 resolvable terms for coherence analysis",
        ));
    }

    // Step 2: Run coherence analysis
    let analysis = coherence::analyse(&resolved, unresolved.len(), geometry, config)?;

    Ok(ValueSystemReport {
        analysis,
        resolved,
        unresolved,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use embeddings::PrecomputedEmbeddings;
    use got_core::UnembeddingMatrix;
    use std::collections::HashMap;

    fn test_geometry() -> CausalGeometry {
        let u = UnembeddingMatrix::new(4, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]).unwrap();
        CausalGeometry::from_unembedding(&u, 1e-6)
    }

    fn test_source() -> PrecomputedEmbeddings {
        let mut map = HashMap::new();
        map.insert("innovation".into(), vec![1.0, 0.5, 0.0]);
        map.insert("risk-aversion".into(), vec![-1.0, -0.5, 0.0]);
        map.insert("transparency".into(), vec![0.0, 1.0, 0.5]);
        map.insert("confidentiality".into(), vec![0.0, -1.0, -0.5]);
        map.insert("integrity".into(), vec![0.5, 0.5, 0.5]);
        PrecomputedEmbeddings::new(map).unwrap()
    }

    #[test]
    fn end_to_end_detects_contradictions() {
        let geom = test_geometry();
        let source = test_source();
        let config = CoherenceConfig::default();

        let report = analyse_value_system(
            &["innovation", "risk-aversion", "transparency", "confidentiality"],
            &source,
            &geom,
            &config,
        ).unwrap();

        assert!(report.analysis.contradictions.len() >= 2,
            "should detect innovation/risk-aversion and transparency/confidentiality contradictions, got {}",
            report.analysis.contradictions.len());
        assert!(report.analysis.coherence_score < 1.0,
            "should penalise contradictions");
        assert!(report.unresolved.is_empty(), "all terms should resolve");
    }

    #[test]
    fn coherent_system_scores_high() {
        let geom = test_geometry();
        let source = test_source();
        let config = CoherenceConfig::default();

        // These terms are not opposed
        let report = analyse_value_system(
            &["innovation", "transparency", "integrity"],
            &source,
            &geom,
            &config,
        ).unwrap();

        assert!(report.analysis.contradictions.is_empty(),
            "no contradictions expected in coherent system");
        assert!(report.analysis.coherence_score > 0.9,
            "coherent system should score high: {}", report.analysis.coherence_score);
    }

    #[test]
    fn unresolved_terms_tracked() {
        let geom = test_geometry();
        let source = test_source();
        let config = CoherenceConfig::default();

        let report = analyse_value_system(
            &["innovation", "transparency", "synergy_buzzword"],
            &source,
            &geom,
            &config,
        ).unwrap();

        assert_eq!(report.unresolved, vec!["synergy_buzzword"]);
        assert_eq!(report.analysis.num_unresolved, 1);
    }

    #[test]
    fn too_few_resolvable_terms_rejected() {
        let geom = test_geometry();
        let source = test_source();
        let config = CoherenceConfig::default();

        let result = analyse_value_system(&["innovation", "unknown1"], &source, &geom, &config);
        assert!(result.is_err(), "only 1 resolvable term should fail");
    }

    #[test]
    fn empty_terms_rejected() {
        let geom = test_geometry();
        let source = test_source();
        let config = CoherenceConfig::default();
        assert!(analyse_value_system(&[], &source, &geom, &config).is_err());
    }

    #[test]
    fn text_report_renders() {
        let geom = test_geometry();
        let source = test_source();
        let config = CoherenceConfig::default();

        let report = analyse_value_system(
            &["innovation", "risk-aversion"],
            &source,
            &geom,
            &config,
        ).unwrap();

        let text = report::render_text(&report.analysis);
        assert!(text.contains("Coherence score"), "report should have header");
        assert!(text.contains("innovation"), "should mention terms");
    }

    #[test]
    fn json_report_roundtrips() {
        let geom = test_geometry();
        let source = test_source();
        let config = CoherenceConfig::default();

        let result = analyse_value_system(
            &["innovation", "risk-aversion", "transparency", "confidentiality"],
            &source,
            &geom,
            &config,
        ).unwrap();

        let json = report::render_json(&result.analysis).unwrap();
        let parsed: coherence::CoherenceAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.coherence_score, result.analysis.coherence_score);
    }
}
