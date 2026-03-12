// ---------------------------------------------------------------------------
// Causal intervention engine.
//
// Perturbs hidden-state activations along probe directions and checks whether
// the model's output changes proportionally. This is the **keystone** of the
// system: without causal validation, probes might measure surface correlations
// rather than real mechanisms.
//
// Algorithm (per probe):
//   1. Compute baseline output from h.
//   2. Perturb h in the probe direction: h+ = h + δ·ŵ, h- = h − δ·ŵ
//   3. Compute outputs for h+ and h-.
//   4. Measure Δ+ = ‖output(h+) − output(h)‖₂, Δ- = ‖output(h-) − output(h)‖₂
//   5. Consistency = sign(Δ+ − Δ-) × min(Δ+/Δ-, Δ-/Δ+)
//      +1 = perfectly symmetric causal effect
//       0 = one direction has no effect
//      -1 = paradoxical reversal
// ---------------------------------------------------------------------------

use got_core::geometry::CausalGeometry;
use got_core::CausalScoreRecord;

use crate::{ProbeError, ProbeVector};

/// Encapsulates a model's forward pass from a probed layer to output.
///
/// In production, the implementation lives inside the TEE and is loaded
/// from a verified model shard. The enclave owns the handle; the caller
/// never supplies it per-call.
///
/// Phase 13: Replaces the raw `&dyn Fn(&[f32]) -> Vec<f32>` closure
/// parameter, making the ownership boundary explicit.
pub trait ModelHandle {
    fn forward(&self, h: &[f32]) -> Vec<f32>;
}

/// PoC convenience wrapper: wraps a closure as a ModelHandle.
/// In production, this is replaced by a TEE-internal model shard loader.
pub struct ClosureModelHandle<F: Fn(&[f32]) -> Vec<f32>> {
    f: F,
}

impl<F: Fn(&[f32]) -> Vec<f32>> ClosureModelHandle<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F: Fn(&[f32]) -> Vec<f32>> ModelHandle for ClosureModelHandle<F> {
    fn forward(&self, h: &[f32]) -> Vec<f32> {
        (self.f)(h)
    }
}

/// Default causal consistency threshold. Probes with consistency above this
/// are considered to measure a real causal mechanism.
pub const DEFAULT_CAUSAL_THRESHOLD: f32 = 0.5;

/// Result of a single causal intervention check on one probe.
#[derive(Debug, Clone)]
pub struct CausalScore {
    /// ‖output(h + δŵ) − output(h)‖₂
    pub delta_plus: f32,
    /// ‖output(h − δŵ) − output(h)‖₂
    pub delta_minus: f32,
    /// Causal consistency ∈ [-1, 1].
    pub consistency: f32,
    /// consistency > threshold
    pub is_causal: bool,
    /// The δ perturbation magnitude used.
    pub perturbation_delta: f32,
}

impl CausalScore {
    /// Convert to the serde-friendly record stored in attestations.
    pub fn to_record(&self) -> CausalScoreRecord {
        CausalScoreRecord {
            delta_plus: self.delta_plus,
            delta_minus: self.delta_minus,
            consistency: self.consistency,
            is_causal: self.is_causal,
        }
    }
}

/// Perform a causal intervention check on a single probe.
///
/// `probe`     – the linear probe to test.
/// `h`         – hidden-state activation vector (ℝ^d).
/// `geometry`  – causal geometry (Φ = UᵀU) for normalisation.
/// `delta`     – perturbation magnitude (scalar, must be > 0).
/// `model_fn`  – callback: hidden state → output vector. Encapsulates the
///               model's forward pass from the probed layer to output.
/// `threshold` – causal consistency threshold (typically 0.5).
///
/// Returns a `CausalScore` describing the intervention result.
pub fn causal_check(
    probe: &ProbeVector,
    h: &[f32],
    geometry: &CausalGeometry,
    delta: f32,
    model: &dyn ModelHandle,
    threshold: f32,
) -> Result<CausalScore, ProbeError> {
    let d = geometry.hidden_dim();
    if h.len() != d {
        return Err(ProbeError::DimensionMismatch {
            act_dim: h.len(),
            geom_dim: d,
        });
    }
    if probe.weights.len() != d {
        return Err(ProbeError::DimensionMismatch {
            act_dim: probe.weights.len(),
            geom_dim: d,
        });
    }
    if delta <= 0.0 || !delta.is_finite() {
        return Err(ProbeError::InvalidDelta(delta));
    }

    // Normalise probe weights: ŵ = w / ‖w‖₂
    let w_norm: f32 = probe.weights.iter().map(|x| x * x).sum::<f32>().sqrt();
    if w_norm == 0.0 {
        // Zero-weight probe has no causal direction to perturb along.
        return Ok(CausalScore {
            delta_plus: 0.0,
            delta_minus: 0.0,
            consistency: 0.0,
            is_causal: false,
            perturbation_delta: delta,
        });
    }
    let w_hat: Vec<f32> = probe.weights.iter().map(|x| x / w_norm).collect();

    // Baseline output
    let output_original = model.forward(h);

    // Positive perturbation: h + δŵ
    let h_plus: Vec<f32> = h
        .iter()
        .zip(w_hat.iter())
        .map(|(hi, wi)| hi + delta * wi)
        .collect();
    let output_plus = model.forward(&h_plus);

    // Negative perturbation: h − δŵ
    let h_minus: Vec<f32> = h
        .iter()
        .zip(w_hat.iter())
        .map(|(hi, wi)| hi - delta * wi)
        .collect();
    let output_minus = model.forward(&h_minus);

    // Measure output shifts (L2 distance)
    let delta_plus = l2_distance(&output_plus, &output_original);
    let delta_minus = l2_distance(&output_minus, &output_original);

    // Causal consistency score
    let consistency = compute_consistency(delta_plus, delta_minus);
    let is_causal = consistency > threshold;

    Ok(CausalScore {
        delta_plus,
        delta_minus,
        consistency,
        is_causal,
        perturbation_delta: delta,
    })
}

/// Result of causal intervention across multiple layers.
#[derive(Debug, Clone)]
pub struct MultiLayerCausalResult {
    /// Per-layer (layer_index, CausalScore) pairs.
    pub layer_scores: Vec<(usize, CausalScore)>,
    /// Count of layers where is_causal == true.
    pub consistent_layers: usize,
    /// Total layers tested.
    pub total_layers: usize,
    /// consistent_layers >= min_consistent_layers.
    pub cross_layer_consistent: bool,
}

/// Perform causal intervention across multiple layers and check consistency.
///
/// `probes_by_layer` – (layer_index, probe) pairs.
/// `h_by_layer`      – (layer_index, hidden-state) pairs.
/// `geometry`         – causal geometry (same for all layers in PoC).
/// `delta`            – perturbation magnitude.
/// `model_fn_by_layer` – callback: (layer_index, hidden-state) → output.
/// `min_consistent_layers` – minimum layers that must pass for cross-layer consistency.
/// `threshold`        – causal consistency threshold.
pub fn causal_check_multi_layer(
    probes_by_layer: &[(usize, &ProbeVector)],
    h_by_layer: &[(usize, &[f32])],
    geometry: &CausalGeometry,
    delta: f32,
    model_fn_by_layer: &dyn Fn(usize, &[f32]) -> Vec<f32>,
    min_consistent_layers: usize,
    threshold: f32,
) -> Result<MultiLayerCausalResult, ProbeError> {
    let mut layer_scores = Vec::new();

    for (layer_idx, probe) in probes_by_layer {
        // Find matching activation
        let h = h_by_layer
            .iter()
            .find(|(l, _)| l == layer_idx)
            .map(|(_, h)| *h)
            .ok_or(ProbeError::DimensionMismatch {
                act_dim: 0,
                geom_dim: geometry.hidden_dim(),
            })?;

        // Wrap the per-layer model_fn into a ModelHandle for causal_check
        let layer = *layer_idx;
        let handle = ClosureModelHandle::new(|h_in: &[f32]| model_fn_by_layer(layer, h_in));

        let score = causal_check(probe, h, geometry, delta, &handle, threshold)?;
        layer_scores.push((*layer_idx, score));
    }

    let consistent_layers = layer_scores.iter().filter(|(_, s)| s.is_causal).count();
    let total_layers = layer_scores.len();
    let cross_layer_consistent = consistent_layers >= min_consistent_layers;

    Ok(MultiLayerCausalResult {
        layer_scores,
        consistent_layers,
        total_layers,
        cross_layer_consistent,
    })
}

/// A library of probes for random sampling during intervention rounds.
///
/// If the model knows which probes will be applied, it could learn to present
/// the "right" activations for those specific directions. Random sampling from
/// a large library prevents this: the model would need to fake activations
/// along all N directions simultaneously.
pub struct ProbeLibrary {
    /// Full set of probes available for a given concept.
    pub probes: Vec<ProbeVector>,
    /// How many to sample per intervention round.
    pub sample_size: usize,
}

impl ProbeLibrary {
    /// Randomly sample probes for this intervention round.
    /// Uses a cryptographic RNG so the selection is unpredictable.
    ///
    /// Returns indices into self.probes (and references) — never duplicates.
    /// If sample_size >= probes.len(), returns all probes.
    pub fn sample(&self) -> Vec<&ProbeVector> {
        use rand::seq::SliceRandom;

        let effective_size = self.sample_size.min(self.probes.len());
        let mut indices: Vec<usize> = (0..self.probes.len()).collect();
        let mut rng = rand::rngs::OsRng;
        indices.shuffle(&mut rng);
        indices.truncate(effective_size);
        indices.iter().map(|&i| &self.probes[i]).collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// L2 distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum::<f32>()
        .sqrt()
}

/// Compute causal consistency score.
///
/// consistency = sign(Δ+ − Δ-) × min(Δ+/Δ-, Δ-/Δ+)
///
/// Domain: [-1, 1]
///   +1 = perfectly symmetric (Δ+ ≈ Δ-)
///    0 = one direction has no effect
///   -1 = paradoxical
///
/// When Δ+ and Δ- are nearly equal (relative difference < 1e-6),
/// the sign is treated as positive to avoid floating-point noise
/// randomly flipping the result to -1.
fn compute_consistency(delta_plus: f32, delta_minus: f32) -> f32 {
    // If both are zero, there's no causal effect in either direction.
    if delta_plus == 0.0 && delta_minus == 0.0 {
        return 0.0;
    }
    // If exactly one is zero, the ratio is 0 → consistency = 0.
    if delta_plus == 0.0 || delta_minus == 0.0 {
        return 0.0;
    }

    let ratio = (delta_plus / delta_minus).min(delta_minus / delta_plus);

    // When the two deltas are nearly equal, the sign of their difference
    // is dominated by floating-point noise.  Treat as positive (symmetric).
    let max_delta = delta_plus.max(delta_minus);
    let rel_diff = (delta_plus - delta_minus).abs() / max_delta;
    // When deltas are nearly equal or positive direction dominates,
    // sign is positive (symmetric or expected asymmetry).
    let sign = if rel_diff < 1e-6 || delta_plus >= delta_minus {
        1.0
    } else {
        -1.0
    };
    sign * ratio
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use got_core::UnembeddingMatrix;

    /// Tiny geometry for testing: 3×2 unembedding → 2×2 Gram.
    fn test_geometry() -> CausalGeometry {
        let u = UnembeddingMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        CausalGeometry::from_unembedding(&u, 1e-6)
    }

    fn test_probe() -> ProbeVector {
        ProbeVector {
            dimension_name: "test_concept".to_string(),
            weights: vec![1.0, 0.0],
            bias: 0.0,
            platt_scale: 1.0,
            platt_shift: 0.0,
            reliability_threshold: 0.7,
        }
    }

    // --- causal_check tests ---

    #[test]
    fn causal_check_linear_model_is_causal() {
        // A synthetic model where output = Φ·h (linear in h).
        // Perturbing h along the probe direction should produce a symmetric output shift.
        let geom = test_geometry();
        let probe = test_probe();
        let h = vec![1.0, 1.0];

        let model_fn = ClosureModelHandle::new(|h_in: &[f32]| -> Vec<f32> {
            // Simple linear model: output = Φ·h
            let d = 2;
            let gram = geom.gram();
            (0..d)
                .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
                .collect()
        });

        let score =
            causal_check(&probe, &h, &geom, 1.0, &model_fn, DEFAULT_CAUSAL_THRESHOLD).unwrap();

        // For a linear model, Δ+ and Δ- should be approximately equal
        assert!(
            score.delta_plus > 0.0,
            "positive perturbation should shift output"
        );
        assert!(
            score.delta_minus > 0.0,
            "negative perturbation should shift output"
        );
        // Consistency should be high (close to 1.0)
        assert!(
            score.consistency > 0.8,
            "linear model should give high consistency, got {}",
            score.consistency
        );
        assert!(score.is_causal, "linear model should be causal");
    }

    #[test]
    fn causal_check_constant_model_not_causal() {
        // A model that always returns the same output regardless of input.
        let geom = test_geometry();
        let probe = test_probe();
        let h = vec![1.0, 1.0];

        let model_fn = ClosureModelHandle::new(|_h_in: &[f32]| -> Vec<f32> {
            vec![42.0, 42.0] // constant output
        });

        let score =
            causal_check(&probe, &h, &geom, 1.0, &model_fn, DEFAULT_CAUSAL_THRESHOLD).unwrap();

        assert_eq!(score.delta_plus, 0.0);
        assert_eq!(score.delta_minus, 0.0);
        assert_eq!(score.consistency, 0.0);
        assert!(!score.is_causal, "constant model should not be causal");
    }

    #[test]
    fn causal_check_nonlinear_model_reduced_consistency() {
        // A model with a rectifier (ReLU) that breaks symmetry.
        let geom = test_geometry();
        let probe = ProbeVector {
            dimension_name: "test".to_string(),
            weights: vec![1.0, 0.0],
            bias: 0.0,
            platt_scale: 1.0,
            platt_shift: 0.0,
            reliability_threshold: 0.7,
        };
        // Pick h near the ReLU threshold so positive perturbation activates
        // but negative perturbation is clipped.
        let h = vec![0.1, 0.5];

        let model_fn = ClosureModelHandle::new(|h_in: &[f32]| -> Vec<f32> {
            // ReLU-like: output = max(0, h_in[i])
            h_in.iter().map(|&x| x.max(0.0)).collect()
        });

        let score =
            causal_check(&probe, &h, &geom, 0.5, &model_fn, DEFAULT_CAUSAL_THRESHOLD).unwrap();

        // The ReLU clips the negative perturbation → asymmetric effects.
        // Consistency should be lower than 1.0 (and might fail the threshold).
        assert!(
            score.consistency < 1.0,
            "nonlinear model should have reduced consistency, got {}",
            score.consistency
        );
    }

    #[test]
    fn causal_check_dimension_mismatch_rejected() {
        let geom = test_geometry();
        let probe = test_probe();
        let h = vec![1.0, 2.0, 3.0]; // 3-dim, geometry is 2-dim

        let model_fn = ClosureModelHandle::new(|_: &[f32]| -> Vec<f32> { vec![0.0] });
        let result = causal_check(&probe, &h, &geom, 1.0, &model_fn, DEFAULT_CAUSAL_THRESHOLD);
        assert!(result.is_err());
    }

    #[test]
    fn causal_check_zero_delta_rejected() {
        let geom = test_geometry();
        let probe = test_probe();
        let h = vec![1.0, 1.0];

        let model_fn = ClosureModelHandle::new(|_: &[f32]| -> Vec<f32> { vec![0.0] });
        let result = causal_check(&probe, &h, &geom, 0.0, &model_fn, DEFAULT_CAUSAL_THRESHOLD);
        assert!(result.is_err());
    }

    #[test]
    fn causal_check_negative_delta_rejected() {
        let geom = test_geometry();
        let probe = test_probe();
        let h = vec![1.0, 1.0];

        let model_fn = ClosureModelHandle::new(|_: &[f32]| -> Vec<f32> { vec![0.0] });
        let result = causal_check(&probe, &h, &geom, -1.0, &model_fn, DEFAULT_CAUSAL_THRESHOLD);
        assert!(result.is_err());
    }

    #[test]
    fn causal_check_zero_weights_not_causal() {
        let geom = test_geometry();
        let probe = ProbeVector {
            dimension_name: "zero".to_string(),
            weights: vec![0.0, 0.0],
            bias: 0.0,
            platt_scale: 1.0,
            platt_shift: 0.0,
            reliability_threshold: 0.7,
        };
        let h = vec![1.0, 1.0];

        let model_fn = ClosureModelHandle::new(|h_in: &[f32]| -> Vec<f32> { h_in.to_vec() });
        let score =
            causal_check(&probe, &h, &geom, 1.0, &model_fn, DEFAULT_CAUSAL_THRESHOLD).unwrap();

        assert!(!score.is_causal, "zero-weight probe cannot be causal");
        assert_eq!(score.consistency, 0.0);
    }

    #[test]
    fn causal_score_to_record_roundtrip() {
        let score = CausalScore {
            delta_plus: 1.5,
            delta_minus: 1.3,
            consistency: 0.87,
            is_causal: true,
            perturbation_delta: 0.5,
        };
        let record = score.to_record();
        assert_eq!(record.delta_plus, 1.5);
        assert_eq!(record.delta_minus, 1.3);
        assert_eq!(record.consistency, 0.87);
        assert!(record.is_causal);
    }

    // --- consistency formula tests ---

    #[test]
    fn consistency_symmetric_is_one() {
        // Δ+ = Δ- → ratio = 1.0 → consistency = +1.0
        let c = compute_consistency(5.0, 5.0);
        assert!((c - 1.0).abs() < 1e-6, "symmetric should be 1.0, got {c}");
    }

    #[test]
    fn consistency_one_zero_is_zero() {
        assert_eq!(compute_consistency(5.0, 0.0), 0.0);
        assert_eq!(compute_consistency(0.0, 5.0), 0.0);
    }

    #[test]
    fn consistency_both_zero_is_zero() {
        assert_eq!(compute_consistency(0.0, 0.0), 0.0);
    }

    #[test]
    fn consistency_asymmetric_below_one() {
        // Δ+ = 10, Δ- = 5 → ratio = 0.5, sign = +1 → consistency = 0.5
        let c = compute_consistency(10.0, 5.0);
        assert!((c - 0.5).abs() < 1e-6, "expected 0.5, got {c}");
    }

    #[test]
    fn consistency_reversed_sign_is_negative() {
        // Δ+ = 2, Δ- = 10 → ratio = 0.2, sign = -1 → consistency = -0.2
        let c = compute_consistency(2.0, 10.0);
        assert!((c - (-0.2)).abs() < 1e-6, "expected -0.2, got {c}");
    }

    // --- ProbeLibrary tests ---

    #[test]
    fn probe_library_sample_correct_count() {
        let probes: Vec<ProbeVector> = (0..10)
            .map(|i| ProbeVector {
                dimension_name: format!("dim_{i}"),
                weights: vec![i as f32, 0.0],
                bias: 0.0,
                platt_scale: 1.0,
                platt_shift: 0.0,
                reliability_threshold: 0.7,
            })
            .collect();

        let lib = ProbeLibrary {
            probes,
            sample_size: 3,
        };

        let sampled = lib.sample();
        assert_eq!(sampled.len(), 3, "should return sample_size probes");
    }

    #[test]
    fn probe_library_sample_no_duplicates() {
        let probes: Vec<ProbeVector> = (0..10)
            .map(|i| ProbeVector {
                dimension_name: format!("dim_{i}"),
                weights: vec![i as f32, (i as f32) * 0.5],
                bias: 0.0,
                platt_scale: 1.0,
                platt_shift: 0.0,
                reliability_threshold: 0.7,
            })
            .collect();

        let lib = ProbeLibrary {
            probes,
            sample_size: 5,
        };

        let sampled = lib.sample();
        // Check uniqueness by dimension name
        let names: Vec<&str> = sampled.iter().map(|p| p.dimension_name.as_str()).collect();
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "no duplicate probes");
    }

    #[test]
    fn probe_library_sample_caps_at_library_size() {
        let probes: Vec<ProbeVector> = (0..3)
            .map(|i| ProbeVector {
                dimension_name: format!("dim_{i}"),
                weights: vec![1.0, 0.0],
                bias: 0.0,
                platt_scale: 1.0,
                platt_shift: 0.0,
                reliability_threshold: 0.7,
            })
            .collect();

        let lib = ProbeLibrary {
            probes,
            sample_size: 100, // More than available
        };

        let sampled = lib.sample();
        assert_eq!(
            sampled.len(),
            3,
            "should return all probes when sample_size > len"
        );
    }

    #[test]
    fn probe_library_sample_shuffles() {
        // Run multiple samples and check that we get different orderings.
        let probes: Vec<ProbeVector> = (0..20)
            .map(|i| ProbeVector {
                dimension_name: format!("dim_{i}"),
                weights: vec![i as f32, 0.0],
                bias: 0.0,
                platt_scale: 1.0,
                platt_shift: 0.0,
                reliability_threshold: 0.7,
            })
            .collect();

        let lib = ProbeLibrary {
            probes,
            sample_size: 5,
        };

        // Sample multiple times and collect the name sequences
        let mut all_same = true;
        let first: Vec<String> = lib
            .sample()
            .iter()
            .map(|p| p.dimension_name.clone())
            .collect();
        for _ in 0..20 {
            let names: Vec<String> = lib
                .sample()
                .iter()
                .map(|p| p.dimension_name.clone())
                .collect();
            if names != first {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "samples should vary across calls (RNG)");
    }

    // --- Multi-layer tests ---

    #[test]
    fn multi_layer_all_causal() {
        let geom = test_geometry();

        // Linear model at every layer
        let model_fn = |_layer: usize, h_in: &[f32]| -> Vec<f32> {
            let d = 2;
            let gram = geom.gram();
            (0..d)
                .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
                .collect()
        };

        let probe = test_probe();
        let h = vec![1.0, 1.0];

        let probes_by_layer: Vec<(usize, &ProbeVector)> =
            vec![(0, &probe), (1, &probe), (2, &probe)];
        let h_by_layer: Vec<(usize, &[f32])> = vec![(0, &h), (1, &h), (2, &h)];

        let result = causal_check_multi_layer(
            &probes_by_layer,
            &h_by_layer,
            &geom,
            1.0,
            &model_fn,
            2,
            DEFAULT_CAUSAL_THRESHOLD,
        )
        .unwrap();

        assert_eq!(result.total_layers, 3);
        assert_eq!(result.consistent_layers, 3);
        assert!(result.cross_layer_consistent);
    }

    #[test]
    fn multi_layer_only_one_causal() {
        let geom = test_geometry();

        // Only layer 0 is linear, layers 1 and 2 return constant
        let model_fn = |layer: usize, h_in: &[f32]| -> Vec<f32> {
            if layer == 0 {
                let d = 2;
                let gram = geom.gram();
                (0..d)
                    .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
                    .collect()
            } else {
                vec![1.0, 1.0] // constant
            }
        };

        let probe = test_probe();
        let h = vec![1.0, 1.0];

        let probes_by_layer: Vec<(usize, &ProbeVector)> =
            vec![(0, &probe), (1, &probe), (2, &probe)];
        let h_by_layer: Vec<(usize, &[f32])> = vec![(0, &h), (1, &h), (2, &h)];

        let result = causal_check_multi_layer(
            &probes_by_layer,
            &h_by_layer,
            &geom,
            1.0,
            &model_fn,
            2, // need at least 2 causal layers
            DEFAULT_CAUSAL_THRESHOLD,
        )
        .unwrap();

        assert_eq!(result.consistent_layers, 1);
        assert!(
            !result.cross_layer_consistent,
            "only 1 causal layer < min 2"
        );
    }
}
