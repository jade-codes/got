// ---------------------------------------------------------------------------
// Interpolation intervention experiments.
//
// Takes two on-manifold activation vectors, linearly interpolates between
// them in ambient space, and at each step: injects the vector into a model,
// measures output entropy/confidence, checks manifold membership via
// local density, and scores incoherence from output-vector consistency.
//
// The output is a fully attestable ExperimentReport.
// ---------------------------------------------------------------------------

use got_core::geometry::{euclidean_cosine, CausalGeometry, GeometryError};
use got_core::manifold::{ManifoldError, ValueManifold};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::intervention::ModelHandle;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ExperimentError {
    #[error(transparent)]
    Geometry(#[from] GeometryError),

    #[error(transparent)]
    Manifold(#[from] ManifoldError),

    #[error("activation dimension {got} does not match geometry dimension {expected}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("need at least 2 interpolation steps, got {0}")]
    TooFewSteps(usize),

    #[error("interpolation parameter out of range [0,1]: {0}")]
    StepOutOfRange(f32),

    #[error("empty model output at step t={0}")]
    EmptyOutput(f32),
}

// ---------------------------------------------------------------------------
// Config & output types
// ---------------------------------------------------------------------------

/// Configuration for an interpolation experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Interpolation parameters, e.g. [0.0, 0.25, 0.5, 0.75, 1.0].
    /// Must be sorted, each in [0, 1], at least 2 entries.
    pub steps: Vec<f32>,
    /// Log-density below this threshold → off-manifold.
    pub density_threshold: f32,
    /// Effective dimension for density estimation (from DensityReading::mean_intrinsic_dim).
    pub d_eff: f32,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            steps: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            density_threshold: -10.0,
            d_eff: 2.0,
        }
    }
}

/// Per-step attestable output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationStep {
    /// Interpolation parameter t ∈ [0, 1].
    pub t: f32,
    /// Causal distance from interpolated point to endpoint a.
    pub causal_distance_from_a: f32,
    /// Causal distance from interpolated point to endpoint b.
    pub causal_distance_from_b: f32,
    /// Log-density at this point (from manifold k-NN estimator).
    /// None if the manifold query returned degenerate.
    pub log_density: Option<f32>,
    /// Whether this point is on-manifold (log_density >= threshold).
    pub on_manifold: bool,
    /// Shannon entropy of softmax(output_logits): H = -Σ p_i ln(p_i).
    pub output_entropy: f32,
    /// Model confidence: max(softmax(output_logits)).
    pub model_confidence: f32,
    /// Incoherence score: 1 - cosine(output_here, output_at_t0).
    /// Measures divergence from the starting point's output.
    pub incoherence_score: f32,
}

/// Full experiment report — attestable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub steps: Vec<InterpolationStep>,
    /// Cosine similarity between outputs at t=0 and t=1.
    pub endpoint_cosine: f32,
    /// Mean cosine between consecutive step outputs.
    pub mean_step_consistency: f32,
    /// Fraction of interpolation points that are off-manifold.
    pub fraction_off_manifold: f32,
    /// The experiment configuration used.
    pub config: ExperimentConfig,
}

// ---------------------------------------------------------------------------
// InterventionExperiment
// ---------------------------------------------------------------------------

/// Orchestrates a linear interpolation experiment between two activation vectors.
pub struct InterventionExperiment<'a> {
    /// Starting activation vector (on-manifold).
    pub a: &'a [f32],
    /// Ending activation vector (on-manifold).
    pub b: &'a [f32],
    /// Causal geometry for distance and density computations.
    pub geometry: &'a CausalGeometry,
    /// Reference manifold for density queries (manifold membership test).
    pub manifold: &'a ValueManifold,
    /// Experiment configuration.
    pub config: ExperimentConfig,
}

impl<'a> InterventionExperiment<'a> {
    /// Validate inputs and create the experiment.
    pub fn new(
        a: &'a [f32],
        b: &'a [f32],
        geometry: &'a CausalGeometry,
        manifold: &'a ValueManifold,
        config: ExperimentConfig,
    ) -> Result<Self, ExperimentError> {
        let d = geometry.hidden_dim();
        if a.len() != d {
            return Err(ExperimentError::DimensionMismatch {
                expected: d,
                got: a.len(),
            });
        }
        if b.len() != d {
            return Err(ExperimentError::DimensionMismatch {
                expected: d,
                got: b.len(),
            });
        }
        if config.steps.len() < 2 {
            return Err(ExperimentError::TooFewSteps(config.steps.len()));
        }
        for &t in &config.steps {
            if !(0.0..=1.0).contains(&t) {
                return Err(ExperimentError::StepOutOfRange(t));
            }
        }

        Ok(Self {
            a,
            b,
            geometry,
            manifold,
            config,
        })
    }

    /// Run the experiment: interpolate, forward through model, score each step.
    pub fn run(&self, model: &dyn ModelHandle) -> Result<ExperimentReport, ExperimentError> {
        let steps = &self.config.steps;

        // Forward pass at each interpolation step
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(steps.len());
        let mut interpolated_points: Vec<Vec<f32>> = Vec::with_capacity(steps.len());

        for &t in steps {
            // h(t) = (1-t)·a + t·b
            let h_t: Vec<f32> = self
                .a
                .iter()
                .zip(self.b.iter())
                .map(|(ai, bi)| (1.0 - t) * ai + t * bi)
                .collect();

            let output = model.forward(&h_t);
            if output.is_empty() {
                return Err(ExperimentError::EmptyOutput(t));
            }

            interpolated_points.push(h_t);
            outputs.push(output);
        }

        // Compute per-step metrics
        let mut result_steps: Vec<InterpolationStep> = Vec::with_capacity(steps.len());
        let output_0 = &outputs[0]; // reference output at t=0

        for (idx, &t) in steps.iter().enumerate() {
            let h_t = &interpolated_points[idx];
            let output = &outputs[idx];

            // Causal distances from h(t) to a and b
            let diff_a: Vec<f32> = h_t.iter().zip(self.a.iter()).map(|(hi, ai)| hi - ai).collect();
            let diff_b: Vec<f32> = h_t.iter().zip(self.b.iter()).map(|(hi, bi)| hi - bi).collect();
            let dist_a_sq = self.geometry.inner_product(&diff_a, &diff_a)?;
            let dist_b_sq = self.geometry.inner_product(&diff_b, &diff_b)?;
            let dist_a = if dist_a_sq > 0.0 { dist_a_sq.sqrt() } else { 0.0 };
            let dist_b = if dist_b_sq > 0.0 { dist_b_sq.sqrt() } else { 0.0 };

            // Manifold membership via density query
            let log_density = self
                .manifold
                .query_log_density(h_t, self.geometry, self.config.d_eff)?;
            let on_manifold = log_density
                .map(|ld| ld >= self.config.density_threshold)
                .unwrap_or(false);

            // Output entropy: H = -Σ p_i ln(p_i) where p = softmax(logits)
            let entropy = softmax_entropy(output);

            // Model confidence: max(softmax(logits))
            let confidence = softmax_max(output);

            // Incoherence: 1 - cos(output_t, output_0)
            let cos_from_start = euclidean_cosine(output, output_0);
            let incoherence = 1.0 - cos_from_start;

            result_steps.push(InterpolationStep {
                t,
                causal_distance_from_a: dist_a,
                causal_distance_from_b: dist_b,
                log_density,
                on_manifold,
                output_entropy: entropy,
                model_confidence: confidence,
                incoherence_score: incoherence,
            });
        }

        // Global metrics
        let endpoint_cosine = euclidean_cosine(&outputs[0], outputs.last().unwrap());

        let step_cosines: Vec<f32> = outputs
            .windows(2)
            .map(|w| euclidean_cosine(&w[0], &w[1]))
            .collect();
        let mean_step_consistency = if step_cosines.is_empty() {
            1.0
        } else {
            step_cosines.iter().sum::<f32>() / step_cosines.len() as f32
        };

        let off_manifold_count = result_steps.iter().filter(|s| !s.on_manifold).count();
        let fraction_off_manifold = off_manifold_count as f32 / result_steps.len() as f32;

        Ok(ExperimentReport {
            steps: result_steps,
            endpoint_cosine,
            mean_step_consistency,
            fraction_off_manifold,
            config: self.config.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute softmax entropy: H = -Σ p_i ln(p_i).
fn softmax_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    // Numerically stable softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let ln_sum = exp_sum.ln();

    let mut entropy = 0.0f32;
    for &x in logits {
        let log_p = (x - max_logit) - ln_sum;
        let p = log_p.exp();
        if p > 0.0 {
            entropy -= p * log_p;
        }
    }
    entropy
}

/// Compute max(softmax(logits)) — model confidence.
fn softmax_max(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    // max softmax = exp(max_logit - max_logit) / exp_sum = 1.0 / exp_sum
    1.0 / exp_sum
}

/// Linear interpolation: h(t) = (1-t)·a + t·b.
pub fn lerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (1.0 - t) * ai + t * bi)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intervention::ClosureModelHandle;
    use got_core::geometry::CausalGeometry;
    use got_core::manifold::ManifoldConfig;

    fn identity_geometry(d: usize) -> CausalGeometry {
        let mut gram = vec![0.0f32; d * d];
        for i in 0..d {
            gram[i * d + i] = 1.0;
        }
        CausalGeometry::from_raw_gram(gram, d).unwrap()
    }

    /// Build a simple manifold from a grid of points.
    fn test_manifold(geom: &CausalGeometry) -> ValueManifold {
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(vec![i as f32 * 0.5, j as f32 * 0.5]);
            }
        }
        ValueManifold::new(points, geom, ManifoldConfig { k: 3 }).unwrap()
    }

    // --- Helper tests ---

    #[test]
    fn softmax_entropy_uniform() {
        // Uniform distribution over 4 items: logits all equal → max entropy = ln(4)
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let h = softmax_entropy(&logits);
        let expected = (4.0f32).ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "uniform entropy should be ln(4)={expected}, got {h}"
        );
    }

    #[test]
    fn softmax_entropy_peaked() {
        // One large logit → low entropy
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let h = softmax_entropy(&logits);
        assert!(h < 0.01, "peaked distribution should have near-zero entropy, got {h}");
    }

    #[test]
    fn softmax_max_uniform() {
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let m = softmax_max(&logits);
        assert!(
            (m - 0.25).abs() < 1e-4,
            "uniform max softmax should be 0.25, got {m}"
        );
    }

    #[test]
    fn softmax_max_peaked() {
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let m = softmax_max(&logits);
        assert!(m > 0.99, "peaked max softmax should be ~1.0, got {m}");
    }

    #[test]
    fn cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = euclidean_cosine(&a, &b);
        assert!((c - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = euclidean_cosine(&a, &b);
        assert!(c.abs() < 1e-5);
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let c = euclidean_cosine(&a, &b);
        assert!((c + 1.0).abs() < 1e-5);
    }

    #[test]
    fn lerp_endpoints() {
        let a = vec![0.0, 0.0];
        let b = vec![2.0, 4.0];
        assert_eq!(lerp(&a, &b, 0.0), vec![0.0, 0.0]);
        assert_eq!(lerp(&a, &b, 1.0), vec![2.0, 4.0]);
    }

    #[test]
    fn lerp_midpoint() {
        let a = vec![0.0, 0.0];
        let b = vec![2.0, 4.0];
        let mid = lerp(&a, &b, 0.5);
        assert!((mid[0] - 1.0).abs() < 1e-6);
        assert!((mid[1] - 2.0).abs() < 1e-6);
    }

    // --- Experiment validation tests ---

    #[test]
    fn too_few_steps_rejected() {
        let geom = identity_geometry(2);
        let manifold = test_manifold(&geom);
        let config = ExperimentConfig {
            steps: vec![0.5],
            ..Default::default()
        };
        let result =
            InterventionExperiment::new(&[0.0, 0.0], &[1.0, 1.0], &geom, &manifold, config);
        assert!(matches!(result, Err(ExperimentError::TooFewSteps(1))));
    }

    #[test]
    fn step_out_of_range_rejected() {
        let geom = identity_geometry(2);
        let manifold = test_manifold(&geom);
        let config = ExperimentConfig {
            steps: vec![0.0, 1.5],
            ..Default::default()
        };
        let result =
            InterventionExperiment::new(&[0.0, 0.0], &[1.0, 1.0], &geom, &manifold, config);
        assert!(matches!(result, Err(ExperimentError::StepOutOfRange(_))));
    }

    #[test]
    fn dim_mismatch_rejected() {
        let geom = identity_geometry(2);
        let manifold = test_manifold(&geom);
        let config = ExperimentConfig::default();
        let result = InterventionExperiment::new(
            &[0.0, 0.0, 0.0], // 3D, geometry is 2D
            &[1.0, 1.0],
            &geom,
            &manifold,
            config,
        );
        assert!(matches!(result, Err(ExperimentError::DimensionMismatch { .. })));
    }

    // --- Full experiment test ---

    #[test]
    fn linear_model_smooth_interpolation() {
        let geom = identity_geometry(2);
        let manifold = test_manifold(&geom);

        // Linear model: output = 2*h (stretches the input)
        let model = ClosureModelHandle::new(|h: &[f32]| h.iter().map(|x| 2.0 * x).collect());

        let config = ExperimentConfig {
            steps: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            density_threshold: -20.0, // lenient
            d_eff: 2.0,
        };

        let a = [0.5, 0.5];
        let b = [1.5, 1.5];
        let exp =
            InterventionExperiment::new(&a, &b, &geom, &manifold, config).unwrap();
        let report = exp.run(&model).unwrap();

        assert_eq!(report.steps.len(), 5);

        // Linear model → output changes linearly → high step consistency
        assert!(
            report.mean_step_consistency > 0.99,
            "linear model should have very high step consistency, got {}",
            report.mean_step_consistency
        );

        // Endpoints have identical direction (both positive) → high cosine
        assert!(
            report.endpoint_cosine > 0.99,
            "same-direction endpoints should have high cosine, got {}",
            report.endpoint_cosine
        );

        // t=0 incoherence should be 0 (comparing to self)
        assert!(
            report.steps[0].incoherence_score.abs() < 1e-4,
            "t=0 incoherence should be ~0, got {}",
            report.steps[0].incoherence_score
        );

        // Distances should be monotonic
        for s in &report.steps {
            assert!(s.causal_distance_from_a.is_finite());
            assert!(s.causal_distance_from_b.is_finite());
        }
    }

    #[test]
    fn opposing_endpoints_high_incoherence() {
        let geom = identity_geometry(2);
        let manifold = test_manifold(&geom);

        // Model where output flips sign: output = h
        let model = ClosureModelHandle::new(|h: &[f32]| h.to_vec());

        let config = ExperimentConfig {
            steps: vec![0.0, 0.5, 1.0],
            density_threshold: -20.0,
            d_eff: 2.0,
        };

        let a = [1.0, 0.0];
        let b = [-1.0, 0.0]; // opposite direction
        let exp =
            InterventionExperiment::new(&a, &b, &geom, &manifold, config).unwrap();
        let report = exp.run(&model).unwrap();

        // Endpoint cosine should be -1 (opposite directions)
        assert!(
            report.endpoint_cosine < -0.99,
            "opposite endpoints should have cosine ~-1, got {}",
            report.endpoint_cosine
        );

        // Last step incoherence should be ~2.0 (1 - (-1))
        assert!(
            report.steps[2].incoherence_score > 1.5,
            "opposite endpoint should have high incoherence, got {}",
            report.steps[2].incoherence_score
        );
    }

    #[test]
    fn off_manifold_detection() {
        let geom = identity_geometry(2);
        let manifold = test_manifold(&geom); // grid from 0..2

        let model = ClosureModelHandle::new(|h: &[f32]| h.to_vec());

        let config = ExperimentConfig {
            steps: vec![0.0, 0.5, 1.0],
            density_threshold: 0.0, // strict threshold
            d_eff: 2.0,
        };

        // b is far from the manifold (grid is 0..2, b is at 100)
        let a = [1.0, 1.0];
        let b = [100.0, 100.0];
        let exp =
            InterventionExperiment::new(&a, &b, &geom, &manifold, config).unwrap();
        let report = exp.run(&model).unwrap();

        // At least the far endpoint should be off-manifold
        assert!(
            report.fraction_off_manifold > 0.0,
            "far-away interpolation should have off-manifold points"
        );
    }

    // --- Serde round-trip ---

    #[test]
    fn experiment_report_serde_roundtrip() {
        let report = ExperimentReport {
            steps: vec![
                InterpolationStep {
                    t: 0.0,
                    causal_distance_from_a: 0.0,
                    causal_distance_from_b: 1.5,
                    log_density: Some(-3.2),
                    on_manifold: true,
                    output_entropy: 2.1,
                    model_confidence: 0.4,
                    incoherence_score: 0.0,
                },
                InterpolationStep {
                    t: 1.0,
                    causal_distance_from_a: 1.5,
                    causal_distance_from_b: 0.0,
                    log_density: Some(-5.1),
                    on_manifold: false,
                    output_entropy: 3.5,
                    model_confidence: 0.2,
                    incoherence_score: 0.8,
                },
            ],
            endpoint_cosine: -0.3,
            mean_step_consistency: 0.7,
            fraction_off_manifold: 0.5,
            config: ExperimentConfig::default(),
        };

        let json = serde_json::to_string(&report).unwrap();
        let decoded: ExperimentReport = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.steps.len(), 2);
        assert!((decoded.endpoint_cosine - (-0.3)).abs() < 1e-6);
        assert!((decoded.fraction_off_manifold - 0.5).abs() < 1e-6);
    }
}
