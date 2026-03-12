// ---------------------------------------------------------------------------
// Probe training and inference under the causal inner product.
//
// Training: direct gradient descent in ℝ^d using the causal IP.
//   logit = wᵀΦh + b
//   gradient w.r.t. w = (σ(logit) − y) · Φh
//
// Inference: same operation — geometry.inner_product(w, h) + bias.
// ---------------------------------------------------------------------------

use got_core::geometry::{CausalGeometry, GeometryError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod hooks;
pub mod intervention;

#[derive(Debug, Error)]
pub enum ProbeError {
    #[error("geometry error: {0}")]
    Geometry(#[from] GeometryError),
    #[error("empty training set")]
    EmptyTrainingSet,
    #[error("activation dimension {act_dim} does not match geometry dimension {geom_dim}")]
    DimensionMismatch { act_dim: usize, geom_dim: usize },
    #[error("probes are stale: geometry drift {drift:.6} exceeds max {max_drift:.6}")]
    ProbeStale { drift: f32, max_drift: f32 },
    #[error("geometry hash mismatch: probes were trained on a different geometry")]
    GeometryMismatch,
    #[error("invalid perturbation delta: {0} (must be finite and > 0)")]
    InvalidDelta(f32),
    #[error("directional drift {drift:.6} exceeds max {max_drift:.6} along probe direction")]
    DirectionalDriftExceeded { drift: f32, max_drift: f32 },
}

/// A single trained linear probe for one value dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeVector {
    pub dimension_name: String,
    /// Probe weights w ∈ ℝ^d (hidden-dim space).
    pub weights: Vec<f32>,
    pub bias: f32,
    /// Platt calibration parameters. PoC uses (1.0, 0.0) = uncalibrated.
    pub platt_scale: f32,
    pub platt_shift: f32,
    /// Below this confidence → coverage_flag = true.
    pub reliability_threshold: f32,
}

/// A set of probes for one layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeSet {
    pub probes: Vec<ProbeVector>,
    pub version: String,
    pub corpus_version: String,
    pub layer: usize,
    /// SHA-256 of the Φ matrix these probes were trained against.
    /// None for legacy probe sets created before drift tracking.
    #[serde(default)]
    pub geometry_hash: Option<[u8; 32]>,
    /// Maximum normalised Frobenius drift before probes are stale.
    /// None means no drift bound (probes are always valid).
    #[serde(default)]
    pub max_drift: Option<f32>,
    /// Maximum directional drift (along probe weight vector) before
    /// probes are stale. None means no per-direction check.
    /// Phase 13: prevents adversary from hiding geometry changes
    /// in probe-relevant directions behind a favourable global norm.
    #[serde(default)]
    pub max_directional_drift: Option<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Train a single probe for a value dimension using gradient descent
/// under the causal inner product.
///
/// `activations`: pairs of (h ∈ ℝ^d, label).
/// Returns weights in ℝ^d that can be used directly with `geometry.inner_product`.
pub fn train_probe(
    activations: &[(Vec<f32>, bool)],
    geometry: &CausalGeometry,
    dimension_name: &str,
    learning_rate: f32,
    epochs: usize,
) -> Result<ProbeVector, ProbeError> {
    if activations.is_empty() {
        return Err(ProbeError::EmptyTrainingSet);
    }

    let d = geometry.hidden_dim();

    // Validate dimensions
    for (h, _) in activations {
        if h.len() != d {
            return Err(ProbeError::DimensionMismatch {
                act_dim: h.len(),
                geom_dim: d,
            });
        }
    }

    let mut w = vec![0f32; d];
    let mut bias = 0f32;

    // Precompute Φh for every training sample (these don't change across epochs)
    let phi_hs: Vec<Vec<f32>> = activations
        .iter()
        .map(|(h, _)| geometry.gram_vec(h))
        .collect::<Result<Vec<_>, _>>()?;

    let labels: Vec<f32> = activations
        .iter()
        .map(|(_, y)| if *y { 1.0 } else { 0.0 })
        .collect();

    // SGD on logistic loss under causal IP:
    //   logit = wᵀ(Φh) + b    (note: wᵀΦh = ⟨w, h⟩_c)
    //   pred  = σ(logit)
    //   error = pred − y
    //   w  ← w  − lr · error · Φh
    //   b  ← b  − lr · error
    for _ in 0..epochs {
        for (idx, phi_h) in phi_hs.iter().enumerate() {
            let logit: f32 = w
                .iter()
                .zip(phi_h.iter())
                .map(|(wi, phi_hi)| wi * phi_hi)
                .sum::<f32>()
                + bias;

            let pred = sigmoid(logit);
            let error = pred - labels[idx];

            for (wi, phi_hi) in w.iter_mut().zip(phi_h.iter()) {
                *wi -= learning_rate * error * phi_hi;
            }
            bias -= learning_rate * error;
        }
    }

    Ok(ProbeVector {
        dimension_name: dimension_name.to_string(),
        weights: w,
        bias,
        platt_scale: 1.0, // uncalibrated — use train_probe_calibrated for real values
        platt_shift: 0.0,
        reliability_threshold: 0.7,
    })
}

/// Train a probe AND fit Platt calibration parameters from a held-out
/// validation split.
///
/// `train_data`: pairs of (h ∈ ℝ^d, label) used for probe weight training.
/// `validation_data`: held-out pairs used ONLY for fitting Platt parameters.
///
/// The validation set must not overlap with the training set — otherwise
/// the calibration is overfit and confidence values are meaningless.
///
/// Returns a `ProbeVector` with calibrated `platt_scale` and `platt_shift`.
pub fn train_probe_calibrated(
    train_data: &[(Vec<f32>, bool)],
    validation_data: &[(Vec<f32>, bool)],
    geometry: &CausalGeometry,
    dimension_name: &str,
    learning_rate: f32,
    epochs: usize,
    platt_lr: f32,
    platt_epochs: usize,
) -> Result<ProbeVector, ProbeError> {
    // Step 1: train probe weights on the training set.
    let mut probe = train_probe(train_data, geometry, dimension_name, learning_rate, epochs)?;

    // Step 2: compute raw logits on the held-out validation set.
    let val_logits: Vec<(f32, bool)> = validation_data
        .iter()
        .map(|(h, label)| {
            let raw = geometry
                .inner_product(&probe.weights, h)
                .map(|v| v + probe.bias);
            raw.map(|r| (r, *label))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if val_logits.is_empty() {
        return Err(ProbeError::EmptyTrainingSet);
    }

    // Step 3: fit Platt parameters.
    let (a, b) = fit_platt(&val_logits, platt_lr, platt_epochs);
    probe.platt_scale = a;
    probe.platt_shift = b;

    Ok(probe)
}

/// Fit Platt scaling parameters (a, b) by logistic regression on
/// held-out (raw_logit, true_label) pairs.
///
/// After fitting, calibrated confidence is: σ(a · raw + b)
///
/// The target for Platt scaling follows Lin, Lin & Weng (2007):
///   t_i = (N_+ + 1) / (N_+ + 2)   if y_i = true
///   t_i = 1 / (N_- + 2)           if y_i = false
/// This avoids overfitting to 0/1 hard targets on small validation sets.
///
/// # Returns
///
/// `(platt_scale, platt_shift)` — the fitted `a` and `b`.
pub fn fit_platt(validation: &[(f32, bool)], lr: f32, epochs: usize) -> (f32, f32) {
    if validation.is_empty() {
        return (1.0, 0.0); // fallback to identity
    }

    // Soft targets per Lin, Lin & Weng (2007) to avoid overfitting.
    let n_pos = validation.iter().filter(|(_, y)| *y).count() as f32;
    let n_neg = validation.len() as f32 - n_pos;
    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    let mut a = 1.0f32; // scale
    let mut b = 0.0f32; // shift

    // SGD on cross-entropy: L = -[ t·log(σ(a·f+b)) + (1-t)·log(1-σ(a·f+b)) ]
    // Gradient w.r.t. a: (σ(a·f+b) - t) · f
    // Gradient w.r.t. b: (σ(a·f+b) - t)
    for _ in 0..epochs {
        for &(raw, label) in validation {
            let t = if label { t_pos } else { t_neg };
            let p = sigmoid(a * raw + b);
            let err = p - t;
            a -= lr * err * raw;
            b -= lr * err;
        }
    }

    (a, b)
}

/// Compute Expected Calibration Error (ECE) over binned predictions.
///
/// Each entry is `(calibrated_confidence, true_label)`.
/// Predictions are bucketed into `bins` equal-width bins by confidence.
/// ECE = Σ_b (|bin_count| / total) × |avg_confidence_b − accuracy_b|.
///
/// Returns 0.0 for empty input. A perfectly calibrated model returns ≈ 0.
pub fn expected_calibration_error(predictions: &[(f32, bool)], bins: usize) -> f32 {
    if predictions.is_empty() || bins == 0 {
        return 0.0;
    }

    let total = predictions.len() as f32;
    let mut ece = 0.0f32;

    for b in 0..bins {
        let lo = b as f32 / bins as f32;
        let hi = (b + 1) as f32 / bins as f32;

        let in_bin: Vec<&(f32, bool)> = predictions
            .iter()
            .filter(|(c, _)| {
                if b == bins - 1 {
                    *c >= lo && *c <= hi // inclusive upper bound on last bin
                } else {
                    *c >= lo && *c < hi
                }
            })
            .collect();

        if in_bin.is_empty() {
            continue;
        }

        let bin_count = in_bin.len() as f32;
        let avg_conf: f32 = in_bin.iter().map(|(c, _)| c).sum::<f32>() / bin_count;
        let accuracy: f32 = in_bin.iter().filter(|(_, y)| *y).count() as f32 / bin_count;

        ece += (bin_count / total) * (avg_conf - accuracy).abs();
    }

    ece
}

/// Run a probe against an activation vector.
///
/// Returns (raw_reading, confidence, coverage_flag).
///  - raw_reading = ⟨w, h⟩_c + bias
///  - confidence  = σ(platt_scale · raw + platt_shift)
///  - coverage_flag = confidence < reliability_threshold
pub fn read_probe(
    probe: &ProbeVector,
    h: &[f32],
    geometry: &CausalGeometry,
) -> Result<(f32, f32, bool), ProbeError> {
    let raw = geometry.inner_product(&probe.weights, h)? + probe.bias;
    let confidence = sigmoid(probe.platt_scale * raw + probe.platt_shift);
    let coverage_flag = confidence < probe.reliability_threshold;
    Ok((raw, confidence, coverage_flag))
}

/// Drift-aware probe reading. Checks that the current geometry hasn’t drifted
/// too far from the reference geometry the probes were trained on.
///
/// Returns the same tuple as `read_probe`, or an error if:
///  - `probe_set.geometry_hash` doesn’t match the reference geometry’s hash
///  - drift exceeds `probe_set.max_drift`
pub fn read_probe_checked(
    probe: &ProbeVector,
    probe_set: &ProbeSet,
    h: &[f32],
    current_geometry: &CausalGeometry,
    reference_geometry: &CausalGeometry,
) -> Result<(f32, f32, bool), ProbeError> {
    // Verify geometry_hash matches the reference (if the probe set tracks it)
    if let Some(expected_hash) = probe_set.geometry_hash {
        let ref_hash = reference_geometry.geometry_hash();
        if ref_hash != expected_hash {
            return Err(ProbeError::GeometryMismatch);
        }
    }
    // Check global drift bound (if one is configured)
    if let Some(max_drift) = probe_set.max_drift {
        let drift = current_geometry.drift_from(reference_geometry)?;
        if drift > max_drift {
            return Err(ProbeError::ProbeStale { drift, max_drift });
        }
    }
    // Check directional drift bound (Phase 13: per-probe direction check)
    if let Some(max_dir_drift) = probe_set.max_directional_drift {
        let dir_drift = current_geometry.directional_drift(reference_geometry, &probe.weights)?;
        if dir_drift > max_dir_drift {
            return Err(ProbeError::DirectionalDriftExceeded {
                drift: dir_drift,
                max_drift: max_dir_drift,
            });
        }
    }
    // Probes still valid — proceed
    read_probe(probe, h, current_geometry)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use got_core::UnembeddingMatrix;

    /// Create a tiny geometry for testing (3×2 unembedding).
    fn test_geometry() -> (UnembeddingMatrix, CausalGeometry) {
        let u = UnembeddingMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);
        (u, geom)
    }

    #[test]
    fn train_separable_clusters() {
        let (_, geom) = test_geometry();

        // Two clearly separated clusters in ℝ^2
        let activations: Vec<(Vec<f32>, bool)> = vec![
            // Positive cluster (high values)
            (vec![5.0, 5.0], true),
            (vec![4.5, 5.5], true),
            (vec![5.5, 4.5], true),
            (vec![6.0, 6.0], true),
            // Negative cluster (low values)
            (vec![-5.0, -5.0], false),
            (vec![-4.5, -5.5], false),
            (vec![-5.5, -4.5], false),
            (vec![-6.0, -6.0], false),
        ];

        let probe = train_probe(&activations, &geom, "test_dim", 0.001, 100).unwrap();

        // Probe should classify positive examples as positive
        for (h, expected) in &activations {
            let (raw, _conf, _flag) = read_probe(&probe, h, &geom).unwrap();
            if *expected {
                assert!(raw > 0.0, "expected positive for {h:?}, got {raw}");
            } else {
                assert!(raw < 0.0, "expected negative for {h:?}, got {raw}");
            }
        }
    }

    #[test]
    fn confidence_in_unit_range() {
        let (_, geom) = test_geometry();

        let activations: Vec<(Vec<f32>, bool)> =
            vec![(vec![1.0, 1.0], true), (vec![-1.0, -1.0], false)];

        let probe = train_probe(&activations, &geom, "test", 0.01, 50).unwrap();

        for (h, _) in &activations {
            let (_raw, conf, _flag) = read_probe(&probe, h, &geom).unwrap();
            assert!(conf >= 0.0 && conf <= 1.0, "confidence {conf} out of range");
        }
    }

    #[test]
    fn coverage_flag_triggers() {
        let (_, geom) = test_geometry();

        // A probe with high threshold — should flag low-confidence predictions
        let mut probe = train_probe(
            &[(vec![1.0, 0.0], true), (vec![0.0, 1.0], false)],
            &geom,
            "test",
            0.001,
            10,
        )
        .unwrap();

        // Set an absurdly high threshold so everything is flagged
        probe.reliability_threshold = 0.999;

        let (_, conf, flag) = read_probe(&probe, &[0.5, 0.5], &geom).unwrap();
        // With sigmoid and uncalibrated Platt, conf near 0.5 is below 0.999
        assert!(flag, "expected coverage flag, conf={conf}");
    }

    #[test]
    fn empty_training_set_rejected() {
        let (_, geom) = test_geometry();
        let result = train_probe(&[], &geom, "test", 0.01, 10);
        assert!(result.is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let (_, geom) = test_geometry();
        let activations = vec![(vec![1.0, 2.0, 3.0], true)]; // 3-dim, geometry is 2-dim
        let result = train_probe(&activations, &geom, "test", 0.01, 10);
        assert!(result.is_err());
    }

    #[test]
    fn training_is_deterministic() {
        let (_, geom) = test_geometry();

        let activations: Vec<(Vec<f32>, bool)> =
            vec![(vec![5.0, 5.0], true), (vec![-5.0, -5.0], false)];

        let p1 = train_probe(&activations, &geom, "test", 0.01, 50).unwrap();
        let p2 = train_probe(&activations, &geom, "test", 0.01, 50).unwrap();

        assert_eq!(p1.weights, p2.weights, "weights differ across runs");
        assert_eq!(p1.bias, p2.bias, "bias differs across runs");
    }

    #[test]
    fn read_probe_nan_activation_rejected() {
        let (_, geom) = test_geometry();

        let probe = train_probe(
            &[(vec![1.0, 0.0], true), (vec![0.0, 1.0], false)],
            &geom,
            "test",
            0.01,
            50,
        )
        .unwrap();

        let result = read_probe(&probe, &[f32::NAN, 1.0], &geom);
        assert!(result.is_err(), "NaN activation should be rejected");
    }

    #[test]
    fn read_probe_dimension_mismatch_rejected() {
        let (_, geom) = test_geometry();

        let probe = train_probe(
            &[(vec![1.0, 0.0], true), (vec![0.0, 1.0], false)],
            &geom,
            "test",
            0.01,
            50,
        )
        .unwrap();

        // Geometry is 2-dim but we pass a 3-dim activation
        let result = read_probe(&probe, &[1.0, 2.0, 3.0], &geom);
        assert!(
            result.is_err(),
            "wrong-dimension activation should be rejected"
        );
    }

    #[test]
    fn fit_platt_identity_on_empty() {
        let (a, b) = fit_platt(&[], 0.01, 100);
        assert_eq!(a, 1.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn fit_platt_improves_calibration() {
        // Construct validation pairs where raw logits are well-separated:
        // positives have high logits, negatives have low logits.
        let validation: Vec<(f32, bool)> = vec![
            (3.0, true),
            (2.5, true),
            (4.0, true),
            (2.0, true),
            (-3.0, false),
            (-2.5, false),
            (-4.0, false),
            (-2.0, false),
        ];

        let (a, b) = fit_platt(&validation, 0.01, 200);

        // After fitting, positive logits should map to high confidence
        // and negative logits should map to low confidence.
        for &(raw, label) in &validation {
            let conf = sigmoid(a * raw + b);
            if label {
                assert!(
                    conf > 0.5,
                    "positive should have conf > 0.5, got {conf} for raw={raw}"
                );
            } else {
                assert!(
                    conf < 0.5,
                    "negative should have conf < 0.5, got {conf} for raw={raw}"
                );
            }
        }

        // Scale should be positive (preserving direction).
        assert!(a > 0.0, "platt_scale should be positive, got {a}");
    }

    #[test]
    fn fit_platt_deterministic() {
        let validation: Vec<(f32, bool)> =
            vec![(2.0, true), (-2.0, false), (1.5, true), (-1.5, false)];

        let (a1, b1) = fit_platt(&validation, 0.01, 100);
        let (a2, b2) = fit_platt(&validation, 0.01, 100);

        assert_eq!(a1, a2, "platt_scale not deterministic");
        assert_eq!(b1, b2, "platt_shift not deterministic");
    }

    #[test]
    fn train_probe_calibrated_produces_better_confidence() {
        let (_, geom) = test_geometry();

        // Well-separated clusters
        let train_data: Vec<(Vec<f32>, bool)> = vec![
            (vec![5.0, 5.0], true),
            (vec![4.5, 5.5], true),
            (vec![5.5, 4.5], true),
            (vec![6.0, 6.0], true),
            (vec![-5.0, -5.0], false),
            (vec![-4.5, -5.5], false),
            (vec![-5.5, -4.5], false),
            (vec![-6.0, -6.0], false),
        ];

        // Held-out validation set (different points, same clusters)
        let val_data: Vec<(Vec<f32>, bool)> = vec![
            (vec![4.0, 4.0], true),
            (vec![5.0, 6.0], true),
            (vec![-4.0, -4.0], false),
            (vec![-5.0, -6.0], false),
        ];

        let probe = train_probe_calibrated(
            &train_data,
            &val_data,
            &geom,
            "honesty",
            0.001,
            100,
            0.01,
            200,
        )
        .unwrap();

        // Calibrated probe should have non-default Platt params
        assert!(
            probe.platt_scale != 1.0 || probe.platt_shift != 0.0,
            "expected non-default Platt params after calibration"
        );

        // Confidence should be well-separated for clearly positive/negative inputs
        let (_, conf_pos, _) = read_probe(&probe, &[5.0, 5.0], &geom).unwrap();
        let (_, conf_neg, _) = read_probe(&probe, &[-5.0, -5.0], &geom).unwrap();
        assert!(
            conf_pos > 0.8,
            "positive confidence should be high, got {conf_pos}"
        );
        assert!(
            conf_neg < 0.2,
            "negative confidence should be low, got {conf_neg}"
        );
    }

    #[test]
    fn fit_platt_soft_targets_avoid_extreme() {
        // With only 1 positive and 1 negative sample, soft targets
        // should prevent the sigmoid from saturating to 0/1.
        let validation = vec![(5.0, true), (-5.0, false)];
        let (a, b) = fit_platt(&validation, 0.01, 500);

        // The fitted sigmoid should be well-behaved (not infinite scale)
        assert!(a.is_finite(), "scale should be finite");
        assert!(b.is_finite(), "shift should be finite");
        assert!(a.abs() < 100.0, "scale should not be extreme, got {a}");
    }

    #[test]
    fn probe_set_serde_roundtrip() {
        let (_, geom) = test_geometry();

        let probe = train_probe(
            &[(vec![5.0, 5.0], true), (vec![-5.0, -5.0], false)],
            &geom,
            "test_dim",
            0.01,
            50,
        )
        .unwrap();

        let ps = ProbeSet {
            probes: vec![probe],
            version: "v1".to_string(),
            corpus_version: "corpus-v1".to_string(),
            layer: 3,
            geometry_hash: None,
            max_drift: None,
            max_directional_drift: None,
        };

        // Round-trip through JSON (the format used by got-cli)
        let json = serde_json::to_string(&ps).unwrap();
        let ps2: ProbeSet = serde_json::from_str(&json).unwrap();

        assert_eq!(ps.probes.len(), ps2.probes.len());
        assert_eq!(ps.probes[0].weights, ps2.probes[0].weights);
        assert_eq!(ps.probes[0].bias, ps2.probes[0].bias);
        assert_eq!(ps.probes[0].dimension_name, ps2.probes[0].dimension_name);
        assert_eq!(ps.probes[0].platt_scale, ps2.probes[0].platt_scale);
        assert_eq!(ps.probes[0].platt_shift, ps2.probes[0].platt_shift);
        assert_eq!(
            ps.probes[0].reliability_threshold,
            ps2.probes[0].reliability_threshold
        );
        assert_eq!(ps.version, ps2.version);
        assert_eq!(ps.corpus_version, ps2.corpus_version);
        assert_eq!(ps.layer, ps2.layer);
    }

    // -----------------------------------------------------------------------
    // ECE tests (Issue #47)
    // -----------------------------------------------------------------------

    #[test]
    fn ece_empty_returns_zero() {
        assert_eq!(expected_calibration_error(&[], 10), 0.0);
    }

    #[test]
    fn ece_perfect_calibration_near_zero() {
        // Confidence ≈ accuracy within each bin → ECE ≈ 0.
        let predictions: Vec<(f32, bool)> = vec![
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, true),
            (0.9, false), // 80% accuracy in bin 0.8–1.0, avg conf 0.9 => gap 0.1
            (0.1, false),
            (0.1, false),
            (0.1, false),
            (0.1, false),
            (0.1, true), // 20% accuracy in bin 0.0–0.2, avg conf 0.1 => gap 0.1
        ];
        let ece = expected_calibration_error(&predictions, 10);
        assert!(
            ece < 0.15,
            "ECE should be small for near-calibrated, got {ece}"
        );
    }

    #[test]
    fn ece_maximally_wrong_is_high() {
        // All confident but all wrong.
        let predictions: Vec<(f32, bool)> =
            vec![(0.95, false), (0.95, false), (0.95, false), (0.95, false)];
        let ece = expected_calibration_error(&predictions, 10);
        assert!(
            ece > 0.8,
            "ECE should be high for all-wrong confident, got {ece}"
        );
    }

    #[test]
    fn ece_is_deterministic() {
        let predictions: Vec<(f32, bool)> =
            vec![(0.7, true), (0.3, false), (0.8, true), (0.2, false)];
        let e1 = expected_calibration_error(&predictions, 10);
        let e2 = expected_calibration_error(&predictions, 10);
        assert_eq!(e1, e2);
    }
}
