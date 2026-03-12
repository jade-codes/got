// ---------------------------------------------------------------------------
// Integration tests: end-to-end attestation pipeline.
//
// These tests prove:
//  1. The pipeline works from geometry → probes → attestation → verification.
//  2. Attestations are deterministic (same inputs → same readings).
//  3. Serialisation is a pure function.
// ---------------------------------------------------------------------------

use ed25519_dalek::SigningKey;
use got_attest::{assemble_and_sign, serialise_for_signing, verify};
use got_core::geometry::CausalGeometry;
use got_core::{GeometricAttestation, InnerProduct, Precision, UnembeddingMatrix, SCHEMA_VERSION};
use got_core::{SCHEMA_VERSION_2, SCHEMA_VERSION_3};
use got_probe::hooks::{
    detect_distribution_shift, ActivationStats, CollectingHook, MeasurementHook,
    MeasurementSidecar, SidecarConfig,
};
use got_probe::intervention::{
    causal_check, causal_check_multi_layer, ClosureModelHandle, ModelHandle, ProbeLibrary,
    DEFAULT_CAUSAL_THRESHOLD,
};
use got_probe::{read_probe, train_probe, ProbeSet};

/// Deterministic signing key for tests (fixed 32-byte seed).
fn test_key() -> SigningKey {
    SigningKey::from_bytes(&[42u8; 32])
}

/// Build the full pipeline from scratch and return a signed attestation.
/// Every input is synthetic — no external files needed.
fn produce_attestation(timestamp: u64) -> GeometricAttestation {
    // 1. Synthetic unembedding (4×3: V=4 tokens, d=3 hidden dim)
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, //
        ],
    )
    .unwrap();

    // 2. Build geometry
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);

    // 3. Train probe on synthetic labelled data
    let training_data: Vec<(Vec<f32>, bool)> = vec![
        (vec![3.0, 3.0, 3.0], true),
        (vec![2.5, 3.5, 3.0], true),
        (vec![3.5, 2.5, 3.0], true),
        (vec![-3.0, -3.0, -3.0], false),
        (vec![-2.5, -3.5, -3.0], false),
        (vec![-3.5, -2.5, -3.0], false),
    ];

    let probe = train_probe(&training_data, &geometry, "test_value", 0.001, 200).unwrap();

    let probe_set = ProbeSet {
        probes: vec![probe],
        version: "test-v1".to_string(),
        corpus_version: "test-corpus-v1".to_string(),
        layer: 0,
        geometry_hash: None,
        max_drift: None,
        max_directional_drift: None,
    };

    // 4. "New input" activation to attest
    let test_activation = vec![1.0, 2.0, 1.5];

    // 5. Run probes
    let mut readings = Vec::new();
    let mut confidences = Vec::new();
    let mut coverage_flags = Vec::new();

    for probe in &probe_set.probes {
        let (raw, conf, flag) = read_probe(probe, &test_activation, &geometry).unwrap();
        readings.push(raw);
        confidences.push(conf);
        coverage_flags.push(flag);
    }

    // 6. Assemble attestation
    let input_hash = got_core::sha256(&[1, 2, 3, 4]); // synthetic "input tokens"

    let attestation = GeometricAttestation {
        schema_version: SCHEMA_VERSION,
        model_id: "synthetic-test".to_string(),
        model_hash: Some([0xAA; 32]),
        precision: Precision::Fp32,
        inner_product: if geometry.is_positive_definite() {
            InnerProduct::Causal
        } else {
            InnerProduct::CausalRegularised {
                epsilon: geometry.epsilon(),
            }
        },
        input_hash,
        timestamp,
        corpus_version: probe_set.corpus_version.clone(),
        probe_version: probe_set.version.clone(),
        layer_readings: vec![readings],
        confidence: confidences,
        coverage_flags,
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };

    // 7. Sign
    let key = test_key();
    assemble_and_sign(attestation, &key).unwrap()
}

// ---------------------------------------------------------------------------
// The reproducibility test — the core proof.
// ---------------------------------------------------------------------------

#[test]
fn attestation_is_deterministic() {
    let ts = 1709568000u64; // fixed timestamp for determinism

    let a1 = produce_attestation(ts);
    let a2 = produce_attestation(ts);

    // Every field must match.
    assert_eq!(a1.layer_readings, a2.layer_readings, "readings differ");
    assert_eq!(a1.confidence, a2.confidence, "confidence differs");
    assert_eq!(
        a1.coverage_flags, a2.coverage_flags,
        "coverage flags differ"
    );
    assert_eq!(a1.model_hash, a2.model_hash, "model hash differs");
    assert_eq!(a1.input_hash, a2.input_hash, "input hash differs");
    assert_eq!(
        a1.inner_product, a2.inner_product,
        "inner product type differs"
    );
    assert_eq!(a1.precision, a2.precision, "precision differs");
    assert_eq!(a1.divergence_flag, a2.divergence_flag, "divergence differs");

    // With fixed timestamp, even the signature should match
    // (Ed25519 with ed25519-dalek is deterministic given identical key + message)
    assert_eq!(a1.signature, a2.signature, "signatures differ");
}

// ---------------------------------------------------------------------------
// End-to-end: produce → verify
// ---------------------------------------------------------------------------

#[test]
fn end_to_end_sign_and_verify() {
    let key = test_key();
    let attestation = produce_attestation(1709568000);

    verify(&attestation, &key.verifying_key()).expect("attestation should verify");
}

// ---------------------------------------------------------------------------
// Serialisation purity
// ---------------------------------------------------------------------------

#[test]
fn serialise_for_signing_is_pure() {
    let attestation = produce_attestation(1709568000);

    let baseline = serialise_for_signing(&attestation).unwrap();

    for i in 0..1000 {
        let bytes = serialise_for_signing(&attestation).unwrap();
        assert_eq!(baseline, bytes, "serialisation differed on iteration {i}");
    }
}

// ---------------------------------------------------------------------------
// Tamper detection
// ---------------------------------------------------------------------------

#[test]
fn tamper_detection_end_to_end() {
    let key = test_key();
    let mut attestation = produce_attestation(1709568000);

    // Tamper with readings
    attestation.layer_readings[0][0] += 0.001;

    assert!(
        verify(&attestation, &key.verifying_key()).is_err(),
        "tampered attestation should not verify"
    );
}

// ---------------------------------------------------------------------------
// Multi-run with different timestamp — readings still match
// ---------------------------------------------------------------------------

#[test]
fn readings_independent_of_timestamp() {
    let a1 = produce_attestation(1000);
    let a2 = produce_attestation(2000);

    assert_eq!(
        a1.layer_readings, a2.layer_readings,
        "readings should not depend on timestamp"
    );
    assert_eq!(a1.confidence, a2.confidence);
    assert_eq!(a1.coverage_flags, a2.coverage_flags);

    // But signatures should differ (different timestamp in payload)
    assert_ne!(a1.signature, a2.signature);
}

// ===========================================================================
// Model-profile tests: full pipeline across different model geometries.
//
// Each profile varies vocab_size (V), hidden_dim (d), number of layers, and
// optionally the structure of the unembedding matrix.  The goal is to prove
// the pipeline is correct for *any* well-formed model shape, not just the
// single 4×3 matrix used above.
// ===========================================================================

/// Description of a synthetic model configuration.
struct ModelProfile {
    name: &'static str,
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    precision: Precision,
}

/// Generate a deterministic unembedding matrix for a profile.
/// Uses a simple deterministic sequence so tests are reproducible.
fn make_unembedding(p: &ModelProfile) -> UnembeddingMatrix {
    let n = p.vocab_size * p.hidden_dim;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            // Deterministic pseudo-random values in [-1, 1] using a simple hash
            let x = ((i as f32 + 1.0) * 0.6180339887).fract() * 2.0 - 1.0;
            x
        })
        .collect();
    UnembeddingMatrix::new(p.vocab_size, p.hidden_dim, data).unwrap()
}

/// Generate synthetic labelled training data in ℝ^d.
/// Positive cluster centred at +magnitude, negative at −magnitude.
fn make_training_data(d: usize, magnitude: f32) -> Vec<(Vec<f32>, bool)> {
    let mut data = Vec::new();
    for i in 0..4 {
        let offset = (i as f32) * 0.1;
        data.push((vec![magnitude + offset; d], true));
        data.push((vec![-(magnitude + offset); d], false));
    }
    data
}

/// Run the full pipeline for a given profile and return a signed attestation.
fn produce_attestation_for_profile(profile: &ModelProfile, timestamp: u64) -> GeometricAttestation {
    let u = make_unembedding(profile);
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);

    let training_data = make_training_data(profile.hidden_dim, 3.0);

    // Train one probe per layer
    let mut all_layer_readings = Vec::new();
    let mut all_confidences = Vec::new();
    let mut all_coverage_flags = Vec::new();

    for layer in 0..profile.num_layers {
        let probe = train_probe(
            &training_data,
            &geometry,
            &format!("{}_layer{layer}", profile.name),
            0.001,
            200,
        )
        .unwrap();

        // A test activation that varies slightly per layer
        let test_h: Vec<f32> = (0..profile.hidden_dim)
            .map(|j| 1.0 + (layer as f32) * 0.1 + (j as f32) * 0.01)
            .collect();

        let (raw, conf, flag) = read_probe(&probe, &test_h, &geometry).unwrap();
        all_layer_readings.push(vec![raw]);
        all_confidences.push(conf);
        all_coverage_flags.push(flag);
    }

    let input_hash = got_core::sha256(&[profile.vocab_size as u8, profile.hidden_dim as u8]);

    let ip = if geometry.is_positive_definite() {
        InnerProduct::Causal
    } else {
        InnerProduct::CausalRegularised {
            epsilon: geometry.epsilon(),
        }
    };

    let attestation = GeometricAttestation {
        schema_version: SCHEMA_VERSION,
        model_id: format!("synthetic-{}", profile.name),
        model_hash: Some(got_core::sha256(profile.name.as_bytes())),
        precision: profile.precision,
        inner_product: ip,
        input_hash,
        timestamp,
        corpus_version: "multi-model-test-v1".to_string(),
        probe_version: format!("{}-probes-v1", profile.name),
        layer_readings: all_layer_readings,
        confidence: all_confidences,
        coverage_flags: all_coverage_flags,
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };

    let key = test_key();
    assemble_and_sign(attestation, &key).unwrap()
}

/// Assert the full invariant suite for a profile.
fn assert_profile_invariants(profile: &ModelProfile) {
    let ts = 1709568000u64;

    // 1. Produces a valid attestation
    let a1 = produce_attestation_for_profile(profile, ts);
    assert_eq!(
        a1.layer_readings.len(),
        profile.num_layers,
        "{}: expected {} layers of readings",
        profile.name,
        profile.num_layers
    );

    // 2. Signature verifies
    let key = test_key();
    verify(&a1, &key.verifying_key())
        .unwrap_or_else(|e| panic!("{}: signature should verify: {e}", profile.name));

    // 3. Deterministic: second run produces identical attestation
    let a2 = produce_attestation_for_profile(profile, ts);
    assert_eq!(
        a1.layer_readings, a2.layer_readings,
        "{}: readings differ across runs",
        profile.name
    );
    assert_eq!(
        a1.confidence, a2.confidence,
        "{}: confidence differs across runs",
        profile.name
    );
    assert_eq!(
        a1.coverage_flags, a2.coverage_flags,
        "{}: coverage flags differ across runs",
        profile.name
    );
    assert_eq!(
        a1.signature, a2.signature,
        "{}: signatures differ across runs",
        profile.name
    );

    // 4. Serialisation is pure
    let bytes1 = serialise_for_signing(&a1).unwrap();
    let bytes2 = serialise_for_signing(&a2).unwrap();
    assert_eq!(bytes1, bytes2, "{}: serialisation not pure", profile.name);

    // 5. Tamper detection works
    let mut tampered = produce_attestation_for_profile(profile, ts);
    tampered.layer_readings[0][0] += 0.001;
    assert!(
        verify(&tampered, &key.verifying_key()).is_err(),
        "{}: tampered attestation should not verify",
        profile.name
    );
}

// ---------------------------------------------------------------------------
// Profile: Wide vocabulary (V=64, d=8)
// Exercises Gram matrix computation where V >> d.
// ---------------------------------------------------------------------------

#[test]
fn profile_wide_vocab() {
    assert_profile_invariants(&ModelProfile {
        name: "wide-vocab",
        vocab_size: 64,
        hidden_dim: 8,
        num_layers: 1,
        precision: Precision::Fp32,
    });
}

// ---------------------------------------------------------------------------
// Profile: Tall hidden dim (V=4, d=16)
// d > V means UᵀU has rank ≤ V < d → regularisation path.
// ---------------------------------------------------------------------------

#[test]
fn profile_tall_hidden_rank_deficient() {
    let profile = ModelProfile {
        name: "tall-hidden",
        vocab_size: 4,
        hidden_dim: 16,
        num_layers: 1,
        precision: Precision::Fp16,
    };

    let u = make_unembedding(&profile);
    let geom = CausalGeometry::from_unembedding(&u, 1e-6);

    // With V=4 < d=16, the Gram matrix should be rank-deficient
    // (our heuristic may or may not trigger depending on trace vs threshold,
    //  but the pipeline must work either way)
    let _ = geom.is_positive_definite(); // just check it doesn't panic

    assert_profile_invariants(&profile);
}

// ---------------------------------------------------------------------------
// Profile: Square (V=8, d=8)
// V = d means U is square; Gram matrix is UᵀU where U is 8×8.
// ---------------------------------------------------------------------------

#[test]
fn profile_square() {
    assert_profile_invariants(&ModelProfile {
        name: "square",
        vocab_size: 8,
        hidden_dim: 8,
        num_layers: 1,
        precision: Precision::Fp32,
    });
}

// ---------------------------------------------------------------------------
// Profile: Multi-layer (V=16, d=8, 3 layers)
// Tests that multi-layer attestation works — readings from each layer are
// independent and all land in the signed attestation.
// ---------------------------------------------------------------------------

#[test]
fn profile_multi_layer() {
    let profile = ModelProfile {
        name: "multi-layer",
        vocab_size: 16,
        hidden_dim: 8,
        num_layers: 3,
        precision: Precision::Fp32,
    };

    let ts = 1709568000u64;
    let a = produce_attestation_for_profile(&profile, ts);

    // Each layer should have distinct readings (different test activations)
    assert_eq!(a.layer_readings.len(), 3);
    assert_ne!(
        a.layer_readings[0], a.layer_readings[1],
        "layer 0 and 1 readings should differ"
    );
    assert_ne!(
        a.layer_readings[1], a.layer_readings[2],
        "layer 1 and 2 readings should differ"
    );

    // Full invariants
    assert_profile_invariants(&profile);
}

// ---------------------------------------------------------------------------
// Profile: Sparse unembedding (V=32, d=6)
// Most entries near zero — tests numerical stability.
// ---------------------------------------------------------------------------

#[test]
fn profile_sparse_unembedding() {
    let profile = ModelProfile {
        name: "sparse",
        vocab_size: 32,
        hidden_dim: 6,
        num_layers: 1,
        precision: Precision::Bfloat16,
    };

    // Override with a sparse matrix (90% zeros)
    let n = profile.vocab_size * profile.hidden_dim;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            if i % 10 == 0 {
                ((i as f32 + 1.0) * 0.6180339887).fract() * 2.0 - 1.0
            } else {
                0.0
            }
        })
        .collect();
    let u = UnembeddingMatrix::new(profile.vocab_size, profile.hidden_dim, data).unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);

    // Pipeline should still work (geometry may be regularised)
    let training_data = make_training_data(profile.hidden_dim, 3.0);
    let probe = train_probe(&training_data, &geometry, "sparse_v", 0.001, 200).unwrap();

    let test_h: Vec<f32> = vec![1.0; profile.hidden_dim];
    let (raw, conf, _flag) = read_probe(&probe, &test_h, &geometry).unwrap();

    // Values should be finite
    assert!(raw.is_finite(), "raw reading should be finite, got {raw}");
    assert!(
        conf >= 0.0 && conf <= 1.0,
        "confidence out of range: {conf}"
    );

    // Deterministic: rerun with same sparse matrix → same results
    let geometry2 = CausalGeometry::from_unembedding(&u, 1e-6);
    let probe2 = train_probe(&training_data, &geometry2, "sparse_v", 0.001, 200).unwrap();
    let (raw2, conf2, _) = read_probe(&probe2, &test_h, &geometry2).unwrap();
    assert_eq!(raw, raw2, "sparse pipeline should be deterministic (raw)");
    assert_eq!(
        conf, conf2,
        "sparse pipeline should be deterministic (conf)"
    );
}

// ---------------------------------------------------------------------------
// Profile: Large-ish (V=128, d=32, 4 layers)
// Approaching realistic ratios. Proves pipeline scales beyond toy sizes.
// ---------------------------------------------------------------------------

#[test]
fn profile_larger_model() {
    assert_profile_invariants(&ModelProfile {
        name: "larger",
        vocab_size: 128,
        hidden_dim: 32,
        num_layers: 4,
        precision: Precision::Fp32,
    });
}

// ---------------------------------------------------------------------------
// Cross-model: attestations from different models must NOT be interchangeable.
// Different geometry → different readings, even on "same" input.
// ---------------------------------------------------------------------------

#[test]
fn different_models_produce_different_readings() {
    let ts = 1709568000u64;

    let a_wide = produce_attestation_for_profile(
        &ModelProfile {
            name: "cross-wide",
            vocab_size: 64,
            hidden_dim: 8,
            num_layers: 1,
            precision: Precision::Fp32,
        },
        ts,
    );

    let a_square = produce_attestation_for_profile(
        &ModelProfile {
            name: "cross-square",
            vocab_size: 8,
            hidden_dim: 8,
            num_layers: 1,
            precision: Precision::Fp32,
        },
        ts,
    );

    // Same hidden dim, but different unembedding → different Gram → different readings
    assert_ne!(
        a_wide.layer_readings, a_square.layer_readings,
        "different models should produce different readings"
    );
    assert_ne!(
        a_wide.model_hash, a_square.model_hash,
        "different models should have different hashes"
    );
}

// ---------------------------------------------------------------------------
// Precision tag round-trip: different precision values in attestation.
// ---------------------------------------------------------------------------

#[test]
fn precision_variants_all_work() {
    for (prec, label) in [
        (Precision::Fp32, "prec-fp32"),
        (Precision::Fp16, "prec-fp16"),
        (Precision::Bfloat16, "prec-bf16"),
        (Precision::Int8, "prec-int8"),
    ] {
        let profile = ModelProfile {
            name: label,
            vocab_size: 16,
            hidden_dim: 4,
            num_layers: 1,
            precision: prec,
        };
        let ts = 1709568000u64;
        let a = produce_attestation_for_profile(&profile, ts);
        assert_eq!(a.precision, prec, "precision should match for {label}");

        let key = test_key();
        verify(&a, &key.verifying_key())
            .unwrap_or_else(|e| panic!("should verify for precision {label}: {e}"));
    }
}

// ===========================================================================
// Phase 8 — Causal Intervention Protocol integration tests.
//
// These tests prove:
//  1. causal_check distinguishes causal from non-causal probes end-to-end.
//  2. Schema v3 attestation (with causal fields) signs and verifies correctly.
//  3. Schema v1 and v2 attestations remain verifiable (backward compat).
//  4. End-to-end: sample probes → intervene → attest with causal_flag → verify.
// ===========================================================================

/// Build a signed v3 attestation with causal intervention scores.
fn produce_causal_attestation(timestamp: u64) -> GeometricAttestation {
    // 1. Synthetic unembedding (4×3)
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, //
        ],
    )
    .unwrap();

    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);

    // 2. Train probe
    let training_data: Vec<(Vec<f32>, bool)> = vec![
        (vec![3.0, 3.0, 3.0], true),
        (vec![2.5, 3.5, 3.0], true),
        (vec![3.5, 2.5, 3.0], true),
        (vec![-3.0, -3.0, -3.0], false),
        (vec![-2.5, -3.5, -3.0], false),
        (vec![-3.5, -2.5, -3.0], false),
    ];

    let probe = train_probe(&training_data, &geometry, "test_value", 0.001, 200).unwrap();

    // 3. Run causal intervention
    let test_h = vec![1.0, 2.0, 1.5];
    let delta = 1.0;

    // Linear model: output = Φ·h
    let gram: Vec<f32> = geometry.gram().to_vec();
    let d = geometry.hidden_dim();
    let model_fn = ClosureModelHandle::new(move |h_in: &[f32]| -> Vec<f32> {
        (0..d)
            .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
            .collect()
    });

    let score = causal_check(
        &probe,
        &test_h,
        &geometry,
        delta,
        &model_fn,
        DEFAULT_CAUSAL_THRESHOLD,
    )
    .unwrap();

    // 4. Run regular probe reading too
    let (raw, conf, coverage_flag) = read_probe(&probe, &test_h, &geometry).unwrap();

    // 5. Build v3 attestation
    let input_hash = got_core::sha256(&[1, 2, 3, 4]);

    let attestation = GeometricAttestation {
        schema_version: SCHEMA_VERSION_3,
        model_id: "synthetic-causal-test".to_string(),
        model_hash: Some([0xAA; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash,
        timestamp,
        corpus_version: "test-corpus-v1".to_string(),
        probe_version: "test-probe-v1".to_string(),
        layer_readings: vec![vec![raw]],
        confidence: vec![conf],
        coverage_flags: vec![coverage_flag],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: Some(geometry.geometry_hash()),
        geometry_drift: Some(0.0),
        causal_scores: vec![score.to_record()],
        intervention_delta: Some(delta),
        causal_flag: Some(score.is_causal),
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };

    let key = test_key();
    assemble_and_sign(attestation, &key).unwrap()
}

// ---------------------------------------------------------------------------
// V3 round-trip: sign → verify
// ---------------------------------------------------------------------------

#[test]
fn v3_causal_attestation_sign_and_verify() {
    let key = test_key();
    let attestation = produce_causal_attestation(1709568000);

    verify(&attestation, &key.verifying_key()).expect("v3 causal attestation should verify");

    // Check causal fields are populated
    assert_eq!(attestation.schema_version, SCHEMA_VERSION_3);
    assert!(!attestation.causal_scores.is_empty());
    assert!(attestation.intervention_delta.is_some());
    assert!(attestation.causal_flag.is_some());
}

// ---------------------------------------------------------------------------
// V3 attestation is deterministic
// ---------------------------------------------------------------------------

#[test]
fn v3_causal_attestation_is_deterministic() {
    let ts = 1709568000u64;
    let a1 = produce_causal_attestation(ts);
    let a2 = produce_causal_attestation(ts);

    assert_eq!(a1.layer_readings, a2.layer_readings, "readings differ");
    assert_eq!(
        a1.causal_scores.len(),
        a2.causal_scores.len(),
        "score count"
    );
    for (s1, s2) in a1.causal_scores.iter().zip(a2.causal_scores.iter()) {
        assert_eq!(s1.delta_plus, s2.delta_plus, "delta_plus");
        assert_eq!(s1.delta_minus, s2.delta_minus, "delta_minus");
        assert_eq!(s1.consistency, s2.consistency, "consistency");
        assert_eq!(s1.is_causal, s2.is_causal, "is_causal");
    }
    assert_eq!(a1.causal_flag, a2.causal_flag, "causal_flag");
    assert_eq!(a1.signature, a2.signature, "signatures differ");
}

// ---------------------------------------------------------------------------
// V3 serialisation is pure
// ---------------------------------------------------------------------------

#[test]
fn v3_serialise_for_signing_is_pure() {
    let attestation = produce_causal_attestation(1709568000);
    let baseline = serialise_for_signing(&attestation).unwrap();
    for i in 0..100 {
        let bytes = serialise_for_signing(&attestation).unwrap();
        assert_eq!(baseline, bytes, "v3 serialisation differed on iter {i}");
    }
}

// ---------------------------------------------------------------------------
// V3 tamper detection on causal fields
// ---------------------------------------------------------------------------

#[test]
fn v3_tamper_causal_scores_detected() {
    let key = test_key();
    let mut a = produce_causal_attestation(1709568000);
    a.causal_scores[0].consistency += 0.001;

    let valid = verify(&a, &key.verifying_key());
    assert!(
        valid.is_err(),
        "tampered causal_scores should fail verification"
    );
}

#[test]
fn v3_tamper_causal_flag_detected() {
    let key = test_key();
    let mut a = produce_causal_attestation(1709568000);
    a.causal_flag = Some(!a.causal_flag.unwrap());

    let valid = verify(&a, &key.verifying_key());
    assert!(
        valid.is_err(),
        "tampered causal_flag should fail verification"
    );
}

#[test]
fn v3_tamper_intervention_delta_detected() {
    let key = test_key();
    let mut a = produce_causal_attestation(1709568000);
    a.intervention_delta = Some(999.0);

    assert!(
        verify(&a, &key.verifying_key()).is_err(),
        "tampered intervention_delta should fail verification"
    );
}

// ---------------------------------------------------------------------------
// Backward compatibility: v1 and v2 still work
// ---------------------------------------------------------------------------

#[test]
fn v1_attestation_still_verifiable_after_v3() {
    let key = test_key();
    let a = produce_attestation(1709568000); // v1
    assert_eq!(a.schema_version, SCHEMA_VERSION);
    verify(&a, &key.verifying_key()).expect("v1 must remain verifiable");
}

#[test]
fn v2_attestation_still_verifiable_after_v3() {
    // Build a v2 attestation (chained, no causal fields)
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);
    let probe = train_probe(
        &[(vec![3.0, 3.0, 3.0], true), (vec![-3.0, -3.0, -3.0], false)],
        &geometry,
        "test",
        0.001,
        100,
    )
    .unwrap();

    let (raw, conf, flag) = read_probe(&probe, &[1.0, 2.0, 1.5], &geometry).unwrap();

    let attestation = GeometricAttestation {
        schema_version: SCHEMA_VERSION_2,
        model_id: "v2-test".to_string(),
        model_hash: Some([0xBB; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: got_core::sha256(&[5, 6, 7]),
        timestamp: 1709568000,
        corpus_version: "cv1".to_string(),
        probe_version: "pv1".to_string(),
        layer_readings: vec![vec![raw]],
        confidence: vec![conf],
        coverage_flags: vec![flag],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: Some(geometry.geometry_hash()),
        geometry_drift: Some(0.0),
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };

    let key = test_key();
    let signed = assemble_and_sign(attestation, &key).unwrap();
    verify(&signed, &key.verifying_key()).expect("v2 must remain verifiable after v3 additions");
}

// ---------------------------------------------------------------------------
// End-to-end causal attestation flow: sample → intervene → attest → verify
// ---------------------------------------------------------------------------

#[test]
fn causal_attestation_flow_end_to_end() {
    // 1. Set up geometry and train a library of probes
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);
    let d = geometry.hidden_dim();

    let training_data: Vec<(Vec<f32>, bool)> =
        vec![(vec![3.0, 3.0, 3.0], true), (vec![-3.0, -3.0, -3.0], false)];

    // Train 5 probes with slightly varied training data (simulating probe diversity)
    let mut probes = Vec::new();
    for i in 0..5 {
        let offset = i as f32 * 0.1;
        let data: Vec<(Vec<f32>, bool)> = training_data
            .iter()
            .map(|(h, y)| (h.iter().map(|x| x + offset).collect(), *y))
            .collect();
        let p = train_probe(&data, &geometry, &format!("concept_{i}"), 0.001, 200).unwrap();
        probes.push(p);
    }

    // 2. Build probe library and sample
    let library = ProbeLibrary {
        probes: probes.clone(),
        sample_size: 3,
    };
    let sampled = library.sample();
    assert_eq!(sampled.len(), 3);

    // 3. Run causal intervention on each sampled probe
    let test_h = vec![1.0, 2.0, 1.5];
    let delta = 1.0;

    let gram: Vec<f32> = geometry.gram().to_vec();
    let model_fn = ClosureModelHandle::new(move |h_in: &[f32]| -> Vec<f32> {
        (0..d)
            .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
            .collect()
    });

    let mut scores = Vec::new();
    let mut all_causal = true;
    for probe in &sampled {
        let score = causal_check(
            *probe,
            &test_h,
            &geometry,
            delta,
            &model_fn,
            DEFAULT_CAUSAL_THRESHOLD,
        )
        .unwrap();
        if !score.is_causal {
            all_causal = false;
        }
        scores.push(score.to_record());
    }

    // 4. Build v3 attestation with causal fields
    let mut readings = Vec::new();
    let mut confs = Vec::new();
    let mut flags = Vec::new();
    for probe in &sampled {
        let (raw, conf, flag) = read_probe(probe, &test_h, &geometry).unwrap();
        readings.push(raw);
        confs.push(conf);
        flags.push(flag);
    }

    let attestation = GeometricAttestation {
        schema_version: SCHEMA_VERSION_3,
        model_id: "flow-test".to_string(),
        model_hash: Some([0xCC; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: got_core::sha256(&[10, 20]),
        timestamp: 1709568000,
        corpus_version: "cv1".to_string(),
        probe_version: "pv1".to_string(),
        layer_readings: vec![readings],
        confidence: confs,
        coverage_flags: flags,
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: Some(geometry.geometry_hash()),
        geometry_drift: Some(0.0),
        causal_scores: scores.clone(),
        intervention_delta: Some(delta),
        causal_flag: Some(all_causal),
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };

    // 5. Sign and verify
    let key = test_key();
    let signed = assemble_and_sign(attestation, &key).unwrap();
    verify(&signed, &key.verifying_key()).expect("causal attestation flow should verify");

    // 6. Verify causal fields are present and correct
    assert_eq!(signed.causal_scores.len(), 3);
    assert_eq!(signed.intervention_delta, Some(1.0));
    assert!(
        signed.causal_flag.unwrap(),
        "linear model should be causal for all probes"
    );

    // 7. JSON round-trip preserves causal fields
    let json = serde_json::to_string_pretty(&signed).unwrap();
    let deser: GeometricAttestation = serde_json::from_str(&json).unwrap();
    verify(&deser, &key.verifying_key()).expect("JSON round-tripped v3 attestation should verify");
    assert_eq!(deser.causal_scores.len(), 3);
    assert_eq!(deser.causal_flag, Some(true));
}

// ---------------------------------------------------------------------------
// Multi-layer causal check integration
// ---------------------------------------------------------------------------

#[test]
fn multi_layer_causal_check_integration() {
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);
    let d = geometry.hidden_dim();

    let probe = train_probe(
        &[(vec![3.0, 3.0, 3.0], true), (vec![-3.0, -3.0, -3.0], false)],
        &geometry,
        "multi_layer_test",
        0.001,
        200,
    )
    .unwrap();

    let h = vec![1.0, 2.0, 1.5];

    // Linear model at all layers
    let gram: Vec<f32> = geometry.gram().to_vec();
    let model_fn = move |_layer: usize, h_in: &[f32]| -> Vec<f32> {
        (0..d)
            .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
            .collect()
    };

    let probes_by_layer: Vec<(usize, &got_probe::ProbeVector)> =
        vec![(0, &probe), (1, &probe), (2, &probe)];
    let h_slice: &[f32] = &h;
    let h_by_layer: Vec<(usize, &[f32])> = vec![(0, h_slice), (1, h_slice), (2, h_slice)];

    let result = causal_check_multi_layer(
        &probes_by_layer,
        &h_by_layer,
        &geometry,
        1.0,
        &model_fn,
        2,
        DEFAULT_CAUSAL_THRESHOLD,
    )
    .unwrap();

    assert_eq!(result.total_layers, 3);
    assert_eq!(result.consistent_layers, 3);
    assert!(
        result.cross_layer_consistent,
        "linear model should be consistent across layers"
    );
}

// ===========================================================================
// Phase 9 — Inline Measurement Architecture integration tests.
//
// These tests prove:
//  1. Sidecar produces signed attestation chains end-to-end.
//  2. Signed sidecar attestations verify correctly.
//  3. AttestationChain links are valid (parent hash == hash of prev signed attestation).
//  4. CollectingHook + sidecar integration works.
//  5. Causal checks inline with sidecar.
//  6. Activation distribution shift detection.
// ===========================================================================

/// Full end-to-end: sidecar → sign → verify → chain across windows.
#[test]
fn sidecar_end_to_end_signed_chain() {
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);
    let probes: Vec<_> = (0..5)
        .map(|i| {
            let off = i as f32 * 0.1;
            train_probe(
                &[
                    (vec![3.0 + off, 3.0, 3.0], true),
                    (vec![-3.0, -3.0, -3.0], false),
                ],
                &geometry,
                &format!("dim_{i}"),
                0.001,
                200,
            )
            .unwrap()
        })
        .collect();

    let config = SidecarConfig {
        window_size: 3,
        probes_per_window: 2,
        model_id: "e2e-sidecar".to_string(),
        model_hash: Some([0xDD; 32]),
        precision: Precision::Fp32,
        causal_enabled: false,
        causal_delta: 1.0,
        causal_threshold: DEFAULT_CAUSAL_THRESHOLD,
    };
    let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

    let key = test_key();
    let h = vec![1.0, 2.0, 1.5];

    let mut prev_signed: Option<GeometricAttestation> = None;

    // Produce 3 windows
    for window in 0..3u64 {
        let mut attestation = None;
        for req in 0..3u64 {
            let rid = window * 3 + req;
            attestation = sidecar.ingest(rid, 0, &h, None);
        }

        let unsigned = attestation.expect("window should close");
        assert_eq!(unsigned.schema_version, SCHEMA_VERSION_3);

        // Sign it
        let signed = assemble_and_sign(unsigned, &key).unwrap();

        // Verify it
        verify(&signed, &key.verifying_key())
            .unwrap_or_else(|e| panic!("sidecar attestation window {window} should verify: {e}"));

        // Check chain link
        if let Some(ref prev) = prev_signed {
            let prev_bytes = serialise_for_signing(prev).unwrap();
            let expected_parent = got_core::sha256(&prev_bytes);
            assert_eq!(
                signed.parent_attestation_hash,
                Some(expected_parent),
                "window {window} should chain to previous"
            );
        } else {
            assert!(
                signed.parent_attestation_hash.is_none(),
                "first window should have no parent"
            );
        }

        // Set parent hash for next window
        let signed_bytes = serialise_for_signing(&signed).unwrap();
        sidecar.set_parent_hash(got_core::sha256(&signed_bytes));

        prev_signed = Some(signed);
    }

    assert_eq!(sidecar.window_index(), 3);
}

/// CollectingHook → drain → feed to sidecar → produce attestation.
#[test]
fn collecting_hook_to_sidecar_pipeline() {
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);
    let probes: Vec<_> = (0..3)
        .map(|i| {
            train_probe(
                &[(vec![3.0, 3.0, 3.0], true), (vec![-3.0, -3.0, -3.0], false)],
                &geometry,
                &format!("dim_{i}"),
                0.001,
                200,
            )
            .unwrap()
        })
        .collect();

    let config = SidecarConfig {
        window_size: 4,
        probes_per_window: 2,
        ..SidecarConfig::default()
    };
    let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

    // Simulate model forward passes writing to the hook
    let hook = CollectingHook::new();
    for rid in 0..4u64 {
        hook.on_activation(rid, 0, &[1.0, 2.0 + rid as f32 * 0.1, 1.5]);
    }
    assert_eq!(hook.len(), 4);

    // Drain and feed to sidecar
    let activations = hook.drain();
    assert!(hook.is_empty());

    let mut attestation = None;
    for (rid, layer, h) in activations {
        attestation = sidecar.ingest(rid, layer, &h, None);
    }

    let a = attestation.expect("feeding 4 activations with window_size=4 should close window");
    assert!(!a.layer_readings[0].is_empty());

    // Sign and verify
    let key = test_key();
    let signed = assemble_and_sign(a, &key).unwrap();
    verify(&signed, &key.verifying_key()).unwrap();
}

/// Sidecar with causal checks enabled produces signed attestation with causal fields.
#[test]
fn sidecar_inline_causal_sign_and_verify() {
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);
    let d = geometry.hidden_dim();
    let probes: Vec<_> = (0..3)
        .map(|i| {
            let off = i as f32 * 0.05;
            train_probe(
                &[
                    (vec![3.0 + off, 3.0, 3.0], true),
                    (vec![-3.0, -3.0, -3.0], false),
                ],
                &geometry,
                &format!("dim_{i}"),
                0.001,
                200,
            )
            .unwrap()
        })
        .collect();

    let config = SidecarConfig {
        window_size: 2,
        probes_per_window: 2,
        model_id: "causal-sidecar".to_string(),
        model_hash: Some([0xEE; 32]),
        precision: Precision::Fp32,
        causal_enabled: true,
        causal_delta: 1.0,
        causal_threshold: DEFAULT_CAUSAL_THRESHOLD,
    };
    let mut sidecar = MeasurementSidecar::new(config, geometry.clone(), probes, "cv1", "pv1");

    let h = vec![1.0, 2.0, 1.5];
    let gram: Vec<f32> = geometry.gram().to_vec();
    let model_fn = ClosureModelHandle::new(move |h_in: &[f32]| -> Vec<f32> {
        (0..d)
            .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
            .collect()
    });

    sidecar.ingest(0, 0, &h, Some(&model_fn));
    let unsigned = sidecar.ingest(1, 0, &h, Some(&model_fn)).unwrap();

    // Causal fields should be populated
    assert!(
        !unsigned.causal_scores.is_empty(),
        "should have causal scores"
    );
    assert!(unsigned.causal_flag.is_some());
    assert!(unsigned.intervention_delta.is_some());

    // Sign and verify
    let key = test_key();
    let signed = assemble_and_sign(unsigned, &key).unwrap();
    verify(&signed, &key.verifying_key()).expect("causal sidecar attestation should verify");

    // Tamper causal field — should fail
    let mut tampered = signed;
    tampered.causal_flag = Some(false);
    assert!(
        verify(&tampered, &key.verifying_key()).is_err(),
        "tampered causal sidecar attestation should fail"
    );
}

/// Activation distribution shift detected between baseline and shifted model.
#[test]
fn activation_shift_detection_end_to_end() {
    // Baseline: activations centred around [5, 5, 5]
    let mut baseline = ActivationStats::new(0, 3);
    for i in 0..100 {
        let x = 5.0 + (i as f32 % 3.0) * 0.1;
        baseline.update(&[x, x, x]);
    }

    // Current: activations centred around [50, 50, 50] — massive shift
    let mut current = ActivationStats::new(0, 3);
    for i in 0..100 {
        let x = 50.0 + (i as f32 % 3.0) * 0.1;
        current.update(&[x, x, x]);
    }

    let shift = detect_distribution_shift(&baseline, &current, 3.0);
    assert!(
        shift > 0.5,
        "massive distribution shift should be detected, got {shift}"
    );

    // No shift variant
    let mut same = ActivationStats::new(0, 3);
    for i in 0..100 {
        let x = 5.0 + (i as f32 % 3.0) * 0.1;
        same.update(&[x, x, x]);
    }
    let no_shift = detect_distribution_shift(&baseline, &same, 3.0);
    assert!(
        no_shift < 0.1,
        "same distribution should show no shift, got {no_shift}"
    );
}

/// Stratified coverage: over many windows, all probes eventually get sampled.
#[test]
fn stratified_coverage_across_many_windows() {
    let u = UnembeddingMatrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let geometry = CausalGeometry::from_unembedding(&u, 1e-6);
    let probes: Vec<_> = (0..20)
        .map(|i| {
            let off = i as f32 * 0.05;
            train_probe(
                &[
                    (vec![3.0 + off, 3.0, 3.0], true),
                    (vec![-3.0, -3.0, -3.0], false),
                ],
                &geometry,
                &format!("dim_{i}"),
                0.001,
                100,
            )
            .unwrap()
        })
        .collect();

    let config = SidecarConfig {
        window_size: 1,
        probes_per_window: 3,
        ..SidecarConfig::default()
    };
    let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

    let h = vec![1.0, 2.0, 1.5];

    // Run 100 windows → should cover all 20 probes
    for i in 0..100u64 {
        let _ = sidecar.ingest(i, 0, &h, None);
    }

    let coverage = sidecar.coverage_count();
    assert_eq!(
        coverage,
        sidecar.library_size(),
        "after 100 windows sampling 3 from 20, all probes should be covered"
    );
}

// ===========================================================================
// Phase 10 — GOT Wire Protocol integration tests
// ===========================================================================

use got_wire::chain::{attestation_hash, verify_chain};
use got_wire::envelope::ExchangeEnvelope;
use got_wire::exchange::{perform_exchange, Verdict};
use got_wire::frame::{Frame, MessageType};
use got_wire::registry::{compute_agent_id, AgentEntry, TrustRegistry};

/// Helper: build a registry with alice and bob.
fn wire_registry(alice: &SigningKey, bob: &SigningKey) -> TrustRegistry {
    let mut registry = TrustRegistry::empty();
    let alice_pk = alice.verifying_key();
    let bob_pk = bob.verifying_key();
    registry.add_agent(AgentEntry {
        name: "alice".to_string(),
        public_key: alice_pk,
        agent_id: compute_agent_id(&alice_pk),
        max_drift_accepted: 0.05,
        roles: vec!["producer".to_string()],
        expected_model_hash: None,
        certificate: None,
    });
    registry.add_agent(AgentEntry {
        name: "bob".to_string(),
        public_key: bob_pk,
        agent_id: compute_agent_id(&bob_pk),
        max_drift_accepted: 0.05,
        roles: vec!["verifier".to_string()],
        expected_model_hash: None,
        certificate: None,
    });
    registry
}

fn wire_key_alice() -> SigningKey {
    SigningKey::from_bytes(&[0xAA; 32])
}

fn wire_key_bob() -> SigningKey {
    SigningKey::from_bytes(&[0xBB; 32])
}

fn wire_make_v1(key: &SigningKey) -> GeometricAttestation {
    wire_make_v1_seq(key, 0)
}

fn wire_make_v1_seq(key: &SigningKey, seq: u64) -> GeometricAttestation {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let a = GeometricAttestation {
        schema_version: SCHEMA_VERSION,
        model_id: "wire-test-model".to_string(),
        model_hash: Some([0x11; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0x22; 32],
        timestamp: now,
        corpus_version: "c1".to_string(),
        probe_version: "p1".to_string(),
        layer_readings: vec![vec![1.0, 2.0, 3.0]],
        confidence: vec![0.95],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: seq,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    assemble_and_sign(a, key).unwrap()
}

fn wire_make_v2_child(
    key: &SigningKey,
    parent: &GeometricAttestation,
    drift: f32,
) -> GeometricAttestation {
    wire_make_v2_child_seq(key, parent, drift, parent.sequence_number + 1)
}

fn wire_make_v2_child_seq(
    key: &SigningKey,
    parent: &GeometricAttestation,
    drift: f32,
    seq: u64,
) -> GeometricAttestation {
    let parent_hash = attestation_hash(parent).unwrap();
    let a = GeometricAttestation {
        schema_version: SCHEMA_VERSION_2,
        model_id: "wire-test-model".to_string(),
        model_hash: Some([0x11; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0x33; 32],
        timestamp: parent.timestamp,
        corpus_version: "c2".to_string(),
        probe_version: "p2".to_string(),
        layer_readings: vec![vec![1.1, 2.1, 3.1]],
        confidence: vec![0.93],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: Some(parent_hash),
        geometry_hash: Some([0xDD; 32]),
        geometry_drift: Some(drift),
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: seq,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    assemble_and_sign(a, key).unwrap()
}

// --- Frame codec integration ---

#[test]
fn wire_frame_roundtrip_all_types() {
    let types = [
        MessageType::ExchangeReq,
        MessageType::ExchangeRsp,
        MessageType::VerifyReq,
        MessageType::VerifyRsp,
        MessageType::ChainReq,
        MessageType::ChainRsp,
        MessageType::Error,
    ];
    for msg_type in types {
        let payload = b"test payload for frame roundtrip";
        let frame = Frame {
            message_type: msg_type,
            payload: payload.to_vec(),
        };
        let encoded = frame.encode().unwrap();
        let (decoded, _consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.message_type, msg_type);
        assert_eq!(decoded.payload, payload.to_vec());
    }
}

// --- Envelope integration ---

#[test]
fn wire_envelope_create_verify_roundtrip() {
    let alice = wire_key_alice();
    let bob = wire_key_bob();
    let attest = wire_make_v1(&alice);
    let bob_id = compute_agent_id(&bob.verifying_key());
    let now = 1700000000u64;

    let envelope =
        ExchangeEnvelope::create([0x42; 32], bob_id, &attest, None, now, &alice).unwrap();

    // Verify as bob.
    envelope
        .verify(
            &bob_id,
            None,
            &attest,
            None,
            &alice.verifying_key(),
            now,
            300,
        )
        .unwrap();
}

#[test]
fn wire_envelope_bytes_roundtrip() {
    use got_wire::envelope::ENVELOPE_SIZE;

    let alice = wire_key_alice();
    let attest = wire_make_v1(&alice);
    let now = 1700000000u64;

    let envelope =
        ExchangeEnvelope::create([0x99; 32], [0xAB; 32], &attest, None, now, &alice).unwrap();

    let bytes = envelope.to_bytes();
    assert_eq!(bytes.len(), ENVELOPE_SIZE);

    let restored = ExchangeEnvelope::from_bytes(&bytes);
    assert_eq!(restored.nonce, envelope.nonce);
    assert_eq!(restored.peer_agent_id, envelope.peer_agent_id);
    assert_eq!(restored.attestation_hash, envelope.attestation_hash);
    assert_eq!(restored.timestamp, envelope.timestamp);
    assert_eq!(restored.signature, envelope.signature);
}

// --- Trust registry integration ---

#[test]
fn wire_registry_toml_load_and_lookup() {
    let alice = wire_key_alice();
    let bob = wire_key_bob();

    let alice_hex: String = alice
        .verifying_key()
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let bob_hex: String = bob
        .verifying_key()
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    let toml_str = format!(
        r#"
[registry]
max_chain_length = 50
max_envelope_age_secs = 600

[[agents]]
id = "alice"
public_key = "{alice_hex}"
max_drift_accepted = 0.03
roles = ["producer"]

[[agents]]
id = "bob"
public_key = "{bob_hex}"
max_drift_accepted = 0.05
roles = ["verifier"]
"#
    );

    let registry = TrustRegistry::from_toml(&toml_str).unwrap();
    assert_eq!(registry.max_chain_length, 50);
    assert_eq!(registry.agents.len(), 2);

    let alice_id = compute_agent_id(&alice.verifying_key());
    let entry = registry.lookup(&alice_id).unwrap();
    assert_eq!(entry.name, "alice");
    assert!((entry.max_drift_accepted - 0.03).abs() < 1e-6);
}

#[test]
fn wire_agent_id_is_sha256_of_pk() {
    let key = wire_key_alice();
    let pk = key.verifying_key();
    let id = compute_agent_id(&pk);
    let expected = got_core::sha256(pk.as_bytes());
    assert_eq!(id, expected);
}

// --- Chain walk integration ---

#[test]
fn wire_chain_walk_valid_three_link() {
    let key = wire_key_alice();
    let a0 = wire_make_v1(&key);
    let a1 = wire_make_v2_child(&key, &a0, 0.02);
    let a2 = wire_make_v2_child(&key, &a1, 0.04);

    let verdict = verify_chain(&[a0, a1], &a2, &[key.verifying_key()], 0.05).unwrap();
    assert_eq!(verdict.length, 3);
    assert!((verdict.max_drift_observed - 0.04).abs() < 1e-6);
}

#[test]
fn wire_chain_walk_broken_link_rejected() {
    let key = wire_key_alice();
    let a0 = wire_make_v1(&key);
    // Build a child that points to a different parent.
    let mut bad = GeometricAttestation {
        schema_version: SCHEMA_VERSION_2,
        model_id: "wire-test-model".to_string(),
        model_hash: Some([0x11; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0xCC; 32],
        timestamp: a0.timestamp + 100,
        corpus_version: "c2".to_string(),
        probe_version: "p2".to_string(),
        layer_readings: vec![vec![1.1]],
        confidence: vec![0.9],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: Some([0x00; 32]), // wrong
        geometry_hash: Some([0xDD; 32]),
        geometry_drift: Some(0.01),
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 1,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    bad = assemble_and_sign(bad, &key).unwrap();

    let err = verify_chain(&[a0], &bad, &[key.verifying_key()], 1.0);
    assert!(err.is_err());
    assert!(format!("{}", err.unwrap_err()).contains("broken link"));
}

#[test]
fn wire_chain_walk_drift_exceeds_rejected() {
    let key = wire_key_alice();
    let anchor = wire_make_v1(&key);
    let child = wire_make_v2_child(&key, &anchor, 0.10);

    let err = verify_chain(&[anchor], &child, &[key.verifying_key()], 0.05);
    assert!(err.is_err());
    assert!(format!("{}", err.unwrap_err()).contains("drift"));
}

#[test]
fn wire_chain_walk_missing_anchor_rejected() {
    let key = wire_key_alice();
    // An attestation with a parent hash but no chain before it.
    let anchor = wire_make_v1(&key);
    let child = wire_make_v2_child(&key, &anchor, 0.01);
    // Pass the child as if it were the anchor — it has a parent_attestation_hash.
    let err = verify_chain(&[], &child, &[key.verifying_key()], 1.0);
    // child has parent_attestation_hash set but is presented as first element.
    // For v2 child, verify_chain checks anchor has no parent hash.
    // Wait — the "current" is the child, not chain[0]. chain is empty, so
    // all = [child] and child.parent_attestation_hash.is_some() → error.
    assert!(err.is_err());
    assert!(format!("{}", err.unwrap_err()).contains("anchor"));
}

// --- Full exchange integration ---

#[test]
fn wire_full_exchange_v1_both_accepted() {
    let alice = wire_key_alice();
    let bob = wire_key_bob();
    let registry = wire_registry(&alice, &bob);

    let alice_attest = wire_make_v1(&alice);
    let bob_attest = wire_make_v1(&bob);

    let (result, rv) = perform_exchange(
        &alice,
        vec![],
        alice_attest,
        &bob,
        vec![],
        bob_attest,
        &registry,
    )
    .unwrap();

    assert_eq!(rv, Verdict::Accepted);
    assert_eq!(result.peer_verdict, Verdict::Accepted);
    assert_eq!(result.our_verdict, Verdict::Accepted);
}

#[test]
fn wire_full_exchange_v2_chain_accepted() {
    let alice = wire_key_alice();
    let bob = wire_key_bob();
    let registry = wire_registry(&alice, &bob);

    let alice_a0 = wire_make_v1(&alice);
    let alice_a1 = wire_make_v2_child(&alice, &alice_a0, 0.03);

    let bob_attest = wire_make_v1(&bob);

    let (result, rv) = perform_exchange(
        &alice,
        vec![alice_a0],
        alice_a1,
        &bob,
        vec![],
        bob_attest,
        &registry,
    )
    .unwrap();

    assert_eq!(rv, Verdict::Accepted);
    assert_eq!(result.our_verdict, Verdict::Accepted);
}

#[test]
fn wire_exchange_drift_rejected() {
    let alice = wire_key_alice();
    let bob = wire_key_bob();
    let registry = wire_registry(&alice, &bob); // max_drift = 0.05

    let alice_a0 = wire_make_v1(&alice);
    let alice_a1 = wire_make_v2_child(&alice, &alice_a0, 0.08); // exceeds 0.05

    let bob_attest = wire_make_v1(&bob);

    let (result, rv) = perform_exchange(
        &alice,
        vec![alice_a0],
        alice_a1,
        &bob,
        vec![],
        bob_attest,
        &registry,
    )
    .unwrap();

    assert_eq!(rv, Verdict::Rejected);
    assert!(result.reason.contains("drift"));
}

#[test]
fn wire_exchange_unknown_agent_error() {
    let alice = wire_key_alice();
    let bob = wire_key_bob();
    let charlie = SigningKey::from_bytes(&[0xCC; 32]);
    let registry = wire_registry(&alice, &bob);

    let charlie_attest = wire_make_v1(&charlie);
    let bob_attest = wire_make_v1(&bob);

    let err = perform_exchange(
        &charlie,
        vec![],
        charlie_attest,
        &bob,
        vec![],
        bob_attest,
        &registry,
    );
    assert!(err.is_err());
}

// ===========================================================================
// Phase 11 — Hardware-Isolated Measurement integration tests
// ===========================================================================

use got_enclave::capture::{HardwareCapture, MockDmaTap};
use got_enclave::enclave::{enclave_pipeline, MockEnclave, MockEnclaveConfig};
use got_enclave::MeasurementEnclave;

fn enclave_geometry(dim: usize) -> CausalGeometry {
    let mut data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }
    let u = UnembeddingMatrix::new(dim, dim, data).unwrap();
    CausalGeometry::from_unembedding(&u, 0.0)
}

fn enclave_probe(dim: usize, name: &str) -> got_probe::ProbeVector {
    let geometry = enclave_geometry(dim);
    let activations: Vec<(Vec<f32>, bool)> = {
        let mut v = Vec::new();
        for _ in 0..10 {
            let mut pos = vec![0.0f32; dim];
            pos[0] = 1.0;
            v.push((pos, true));
            let mut neg = vec![0.0f32; dim];
            neg[0] = -1.0;
            v.push((neg, false));
        }
        v
    };
    train_probe(&activations, &geometry, name, 0.1, 100).unwrap()
}

fn enclave_key() -> SigningKey {
    SigningKey::from_bytes(&[0xEE; 32])
}

/// Mock enclave receives hardware-captured activations and produces valid attestation.
#[test]
fn enclave_capture_to_attestation() {
    let dim = 4;
    let geometry = enclave_geometry(dim);
    let probe = enclave_probe(dim, "honesty");
    let key = enclave_key();
    let mut enclave = MockEnclave::new(
        key,
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        None,
    );

    let tap = MockDmaTap::new(dim, vec![]);
    let h = vec![1.0, 0.5, -0.3, 0.8];
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();

    let attest = enclave
        .attest("integration-model", [0xAA; 32], None, None, None)
        .unwrap();

    verify(&attest, &enclave.verifying_key()).unwrap();
    assert_eq!(attest.schema_version, SCHEMA_VERSION_3);
    assert!(!attest.layer_readings.is_empty());
}

/// Mock enclave runs causal intervention and returns scores.
#[test]
fn enclave_causal_intervention_integration() {
    let dim = 4;
    let geometry = enclave_geometry(dim);
    let probe = enclave_probe(dim, "fairness");
    let key = enclave_key();
    let model_fn: Box<dyn ModelHandle> =
        Box::new(ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> {
            h.to_vec()
        }));
    let mut enclave = MockEnclave::new(
        key,
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        Some(model_fn),
    );

    let tap = MockDmaTap::new(dim, vec![]);
    let h = vec![1.0, 0.5, -0.3, 0.8];
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();

    let scores = enclave.run_causal_check(0.1).unwrap();
    assert_eq!(scores.len(), 1);
    assert!(scores[0].delta_plus >= 0.0);
    assert!(scores[0].delta_minus >= 0.0);
}

/// Attestation signed inside enclave is verifiable by external party.
#[test]
fn enclave_attestation_external_verify() {
    let dim = 4;
    let geometry = enclave_geometry(dim);
    let probe = enclave_probe(dim, "ext");
    let key = enclave_key();
    let pk = key.verifying_key();
    let mut enclave = MockEnclave::new(
        key,
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        None,
    );

    let tap = MockDmaTap::new(dim, vec![]);
    let frame = tap.capture(0, 0, &vec![0.5; dim]).unwrap();
    enclave.receive_activations(frame).unwrap();

    let attest = enclave
        .attest("ext-model", [0xBB; 32], None, None, None)
        .unwrap();

    // "External" verifier only has the public key.
    verify(&attest, &pk).unwrap();

    // Serialisation is deterministic.
    let bytes1 = serialise_for_signing(&attest).unwrap();
    let bytes2 = serialise_for_signing(&attest).unwrap();
    assert_eq!(bytes1, bytes2);
}

/// End-to-end: hardware capture → enclave → attestation → wire protocol verify chain.
#[test]
fn enclave_to_wire_protocol_chain() {
    let dim = 4;
    let geometry = enclave_geometry(dim);
    let probe = enclave_probe(dim, "e2e");
    let key = enclave_key();
    let config = MockEnclaveConfig {
        delta: 0.1,
        causal_threshold: 0.0, // low threshold so probes pass
        ..MockEnclaveConfig::default()
    };
    let model_fn: Box<dyn ModelHandle> =
        Box::new(ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> {
            h.to_vec()
        }));
    let mut enclave = MockEnclave::new(key, geometry, vec![probe], config, Some(model_fn));

    let tap = MockDmaTap::new(dim, vec![]);

    // First attestation (anchor).
    let activations1 = vec![(0usize, 0usize, vec![1.0f32, 0.0, 0.0, 0.0])];
    let (a0, _scores0) = enclave_pipeline(
        &mut enclave,
        &tap,
        &activations1,
        "e2e-model",
        [0xCC; 32],
        None,
        None,
        None,
    )
    .unwrap();

    let a0_hash = {
        let bytes = serialise_for_signing(&a0).unwrap();
        got_core::sha256(&bytes)
    };

    enclave.reset();

    // Second attestation (chained to a0).
    let activations2 = vec![(0usize, 0usize, vec![0.9f32, 0.1, 0.0, 0.0])];
    let (a1, _scores1) = enclave_pipeline(
        &mut enclave,
        &tap,
        &activations2,
        "e2e-model",
        [0xCC; 32],
        Some(a0_hash),
        Some([0xDD; 32]),
        Some(0.01),
    )
    .unwrap();

    // Both attestations are signed and valid.
    let pk = enclave.verifying_key();
    verify(&a0, &pk).unwrap();
    verify(&a1, &pk).unwrap();

    // Use wire protocol's chain verification.
    let verdict = verify_chain(&[a0], &a1, &[pk], 0.05).unwrap();
    assert_eq!(verdict.length, 2);
}

/// Enclave rejects tampered activation data (integrity hash mismatch).
#[test]
fn enclave_rejects_tampered_hardware_capture() {
    let dim = 4;
    let geometry = enclave_geometry(dim);
    let probe = enclave_probe(dim, "tamper");
    let key = enclave_key();
    let model_fn: Box<dyn ModelHandle> =
        Box::new(ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> {
            h.to_vec()
        }));
    let mut enclave = MockEnclave::new(
        key,
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        Some(model_fn),
    );

    let tap = MockDmaTap::new(dim, vec![]).with_tamper();

    let activations = vec![(0usize, 0usize, vec![1.0f32, 0.5, -0.3, 0.8])];
    let err = enclave_pipeline(
        &mut enclave,
        &tap,
        &activations,
        "tamper-model",
        [0xAA; 32],
        None,
        None,
        None,
    );
    assert!(err.is_err());
}

/// Enclave attestation integrates with wire protocol exchange.
#[test]
fn enclave_attestation_wire_exchange() {
    let dim = 4;
    let geometry = enclave_geometry(dim);
    let probe = enclave_probe(dim, "wire");

    // Two enclave keys (alice and bob each have their own enclave).
    let alice_key = SigningKey::from_bytes(&[0xA1; 32]);
    let bob_key = SigningKey::from_bytes(&[0xB2; 32]);

    let config = MockEnclaveConfig {
        delta: 0.1,
        causal_threshold: 0.0,
        ..MockEnclaveConfig::default()
    };

    let alice_model: Box<dyn ModelHandle> =
        Box::new(ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> {
            h.to_vec()
        }));
    let mut alice_enclave = MockEnclave::new(
        alice_key.clone(),
        geometry.clone(),
        vec![probe.clone()],
        config.clone(),
        Some(alice_model),
    );
    let bob_model: Box<dyn ModelHandle> =
        Box::new(ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> {
            h.to_vec()
        }));
    let mut bob_enclave = MockEnclave::new(
        bob_key.clone(),
        geometry,
        vec![probe],
        config,
        Some(bob_model),
    );

    let tap = MockDmaTap::new(dim, vec![]);

    // Alice produces attestation.
    let alice_acts = vec![(0usize, 0usize, vec![1.0f32, 0.0, 0.0, 0.0])];
    let (alice_attest, _) = enclave_pipeline(
        &mut alice_enclave,
        &tap,
        &alice_acts,
        "alice-model",
        [0x11; 32],
        None,
        None,
        None,
    )
    .unwrap();

    // Bob produces attestation.
    let bob_acts = vec![(0usize, 0usize, vec![0.0f32, 1.0, 0.0, 0.0])];
    let (bob_attest, _) = enclave_pipeline(
        &mut bob_enclave,
        &tap,
        &bob_acts,
        "bob-model",
        [0x22; 32],
        None,
        None,
        None,
    )
    .unwrap();

    // Build a wire protocol trust registry with enclave public keys.
    let mut registry = TrustRegistry::empty();
    let alice_pk = alice_key.verifying_key();
    let bob_pk = bob_key.verifying_key();
    registry.add_agent(AgentEntry {
        name: "alice-enclave".to_string(),
        public_key: alice_pk,
        agent_id: compute_agent_id(&alice_pk),
        max_drift_accepted: 0.1,
        roles: vec!["producer".to_string()],
        expected_model_hash: None,
        certificate: None,
    });
    registry.add_agent(AgentEntry {
        name: "bob-enclave".to_string(),
        public_key: bob_pk,
        agent_id: compute_agent_id(&bob_pk),
        max_drift_accepted: 0.1,
        roles: vec!["producer".to_string()],
        expected_model_hash: None,
        certificate: None,
    });

    // Perform wire protocol exchange between enclave-attested agents.
    let (result, rv) = perform_exchange(
        &alice_key,
        vec![],
        alice_attest,
        &bob_key,
        vec![],
        bob_attest,
        &registry,
    )
    .unwrap();

    assert_eq!(rv, Verdict::Accepted);
    assert_eq!(result.our_verdict, Verdict::Accepted);
    assert!(result.peer_attestation.causal_scores.len() > 0);
    assert!(result.peer_attestation.causal_flag.is_some());
}

// ===========================================================================
// Phase 12 — Persistent Attestation Store & Audit Trail integration tests
// ===========================================================================

use got_store::{AttestationStore, FileStore, MemoryStore, StoreFilter};

/// Helper: build a signed attestation with optional chain/causal fields.
fn store_attestation(
    key: &SigningKey,
    model_id: &str,
    timestamp: u64,
    parent: Option<[u8; 32]>,
    drift: Option<f32>,
    causal_flag: Option<bool>,
) -> got_core::GeometricAttestation {
    use ed25519_dalek::Signer;
    let mut a = got_core::GeometricAttestation {
        schema_version: SCHEMA_VERSION_3,
        model_id: model_id.to_string(),
        model_hash: Some([0xAA; 32]),
        precision: got_core::Precision::Fp32,
        inner_product: got_core::InnerProduct::Causal,
        input_hash: [0xBB; 32],
        timestamp,
        corpus_version: "test-v1".to_string(),
        probe_version: "probe-v1".to_string(),
        layer_readings: vec![vec![0.5, 0.6]],
        confidence: vec![0.9, 0.8],
        coverage_flags: vec![false, false],
        divergence_flag: false,
        parent_attestation_hash: parent,
        geometry_hash: Some([0xCC; 32]),
        geometry_drift: drift,
        causal_scores: if causal_flag.is_some() {
            vec![got_core::CausalScoreRecord {
                delta_plus: 0.5,
                delta_minus: 0.4,
                consistency: 0.9,
                is_causal: causal_flag.unwrap_or(false),
            }]
        } else {
            vec![]
        },
        intervention_delta: if causal_flag.is_some() {
            Some(0.1)
        } else {
            None
        },
        causal_flag,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    let payload = serialise_for_signing(&a).unwrap();
    let sig = key.sign(&payload);
    a.signature = sig.to_bytes();
    a
}

fn store_content_hash(a: &got_core::GeometricAttestation) -> [u8; 32] {
    let bytes = serialise_for_signing(a).unwrap();
    got_core::sha256(&bytes)
}

/// End-to-end: build attestation chain → store → query → audit.
#[test]
fn store_chain_query_audit_end_to_end() {
    let key = test_key();
    let vk = key.verifying_key();
    let mut store = MemoryStore::new();

    // Build a 3-link chain.
    let a0 = store_attestation(&key, "e2e-model", 1000, None, Some(0.01), Some(true));
    let a0_hash = store_content_hash(&a0);
    let a1 = store_attestation(
        &key,
        "e2e-model",
        2000,
        Some(a0_hash),
        Some(0.02),
        Some(true),
    );
    let a1_hash = store_content_hash(&a1);
    let a2 = store_attestation(
        &key,
        "e2e-model",
        3000,
        Some(a1_hash),
        Some(0.03),
        Some(false),
    );

    store.append(&a0, &vk).unwrap();
    store.append(&a1, &vk).unwrap();
    store.append(&a2, &vk).unwrap();
    assert_eq!(store.len(), 3);

    // Chain retrieval.
    let chain = store.chain("e2e-model");
    assert_eq!(chain.len(), 3);
    assert_eq!(chain[0].timestamp, 1000);
    assert_eq!(chain[2].timestamp, 3000);

    // Query by causal flag.
    let passing = store.query(&StoreFilter::new().causal_flag(true));
    assert_eq!(passing.len(), 2);
    let failing = store.query(&StoreFilter::new().causal_flag(false));
    assert_eq!(failing.len(), 1);

    // Query by time range.
    let middle = store.query(&StoreFilter::new().after(1500).before(2500));
    assert_eq!(middle.len(), 1);
    assert_eq!(middle[0].timestamp, 2000);

    // Audit.
    let report = store.audit("e2e-model");
    assert_eq!(report.total_attestations, 3);
    assert_eq!(report.chain_length, 3);
    assert!(report.chain_valid);
    assert_eq!(report.first_timestamp, Some(1000));
    assert_eq!(report.last_timestamp, Some(3000));
    assert_eq!(report.drift_summary.readings_with_drift, 3);
    assert!((report.drift_summary.max_drift.unwrap() - 0.03).abs() < 1e-4);
    assert_eq!(report.causal_summary.causal_pass_count, 2);
    assert_eq!(report.causal_summary.causal_fail_count, 1);
    assert_eq!(report.signers.len(), 1);
}

/// FileStore persists across process restart (simulated by drop + reopen).
#[test]
fn store_file_persistence_across_restart() {
    let dir = std::env::temp_dir().join(format!("got-int-persist-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let key = test_key();
    let vk = key.verifying_key();

    let a0 = store_attestation(&key, "persist-m", 1000, None, None, None);
    let a0_hash = store_content_hash(&a0);
    let a1 = store_attestation(&key, "persist-m", 2000, Some(a0_hash), None, None);

    // Write session.
    {
        let mut store = FileStore::open(&dir).unwrap();
        store.append(&a0, &vk).unwrap();
        store.append(&a1, &vk).unwrap();
        assert_eq!(store.len(), 2);
    }
    // Reopen session.
    {
        let store = FileStore::open(&dir).unwrap();
        assert_eq!(store.len(), 2);
        let chain = store.chain("persist-m");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].timestamp, 1000);
        assert_eq!(chain[1].timestamp, 2000);

        let report = store.audit("persist-m");
        assert_eq!(report.total_attestations, 2);
        assert_eq!(report.chain_length, 2);
        assert!(report.chain_valid);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Store rejects tampered attestation (signature check on insert).
#[test]
fn store_rejects_tampered_on_insert() {
    let key = test_key();
    let mut a = store_attestation(&key, "tamper-m", 1000, None, None, None);
    a.timestamp = 9999; // tamper after signing

    let mut store = MemoryStore::new();
    let err = store.append(&a, &key.verifying_key());
    assert!(err.is_err());
}

/// Store rejects orphaned attestation (missing parent).
#[test]
fn store_rejects_orphaned_chain() {
    let key = test_key();
    let a = store_attestation(&key, "orphan-m", 1000, Some([0xFF; 32]), None, None);

    let mut store = MemoryStore::new();
    let err = store.append(&a, &key.verifying_key());
    assert!(err.is_err());
}

/// Multi-model store correctly separates chains.
#[test]
fn store_multi_model_isolation() {
    let key = test_key();
    let vk = key.verifying_key();
    let mut store = MemoryStore::new();

    let a = store_attestation(&key, "model-alpha", 1000, None, None, Some(true));
    let b = store_attestation(&key, "model-beta", 2000, None, None, Some(false));
    let c = store_attestation(&key, "model-alpha", 3000, None, None, Some(true));

    store.append(&a, &vk).unwrap();
    store.append(&b, &vk).unwrap();
    store.append(&c, &vk).unwrap();

    assert_eq!(store.chain("model-alpha").len(), 2);
    assert_eq!(store.chain("model-beta").len(), 1);
    assert_eq!(
        store
            .query(&StoreFilter::new().model_id("model-alpha"))
            .len(),
        2
    );

    let report_a = store.audit("model-alpha");
    assert_eq!(report_a.total_attestations, 2);
    assert_eq!(report_a.causal_summary.causal_pass_count, 2);

    let report_b = store.audit("model-beta");
    assert_eq!(report_b.total_attestations, 1);
    assert_eq!(report_b.causal_summary.causal_fail_count, 1);
}

/// Enclave → wire → store → audit end-to-end pipeline.
#[test]
fn enclave_wire_store_audit_pipeline() {
    use got_enclave::capture::MockDmaTap;
    use got_enclave::enclave::{enclave_pipeline, MockEnclaveConfig};

    let dim = 4;
    let geometry = enclave_geometry(dim);
    let probe = enclave_probe(dim, "pipe");
    let enc_key = SigningKey::from_bytes(&[0xF1; 32]);
    let config = MockEnclaveConfig {
        delta: 0.1,
        causal_threshold: 0.0,
        ..MockEnclaveConfig::default()
    };
    let model_fn: Box<dyn ModelHandle> =
        Box::new(ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> {
            h.to_vec()
        }));
    let mut enclave = got_enclave::MockEnclave::new(
        enc_key.clone(),
        geometry,
        vec![probe],
        config,
        Some(model_fn),
    );

    let tap = MockDmaTap::new(dim, vec![]);

    // Produce attestations from enclave.
    let acts1 = vec![(0usize, 0usize, vec![1.0f32, 0.0, 0.0, 0.0])];
    let (a0, _) = enclave_pipeline(
        &mut enclave,
        &tap,
        &acts1,
        "enc-model",
        [0xDD; 32],
        None,
        None,
        None,
    )
    .unwrap();

    let a0_hash = store_content_hash(&a0);
    enclave.reset();

    let acts2 = vec![(0usize, 0usize, vec![0.9f32, 0.1, 0.0, 0.0])];
    let (a1, _) = enclave_pipeline(
        &mut enclave,
        &tap,
        &acts2,
        "enc-model",
        [0xDD; 32],
        Some(a0_hash),
        Some([0xEE; 32]),
        Some(0.01),
    )
    .unwrap();

    // Store both attestations.
    let vk = enc_key.verifying_key();
    let mut store = MemoryStore::new();
    store.append(&a0, &vk).unwrap();
    store.append(&a1, &vk).unwrap();

    assert_eq!(store.len(), 2);

    // Audit the chain.
    let report = store.audit("enc-model");
    assert_eq!(report.total_attestations, 2);
    assert!(report.chain_valid);
    assert!(report.causal_summary.attestations_with_causal > 0);
    assert_eq!(report.signers.len(), 1);
}

/// Duplicate insert is idempotent (integration-level).
#[test]
fn store_duplicate_insert_idempotent_integration() {
    let key = test_key();
    let vk = key.verifying_key();
    let a = store_attestation(&key, "dup-m", 1000, None, None, None);

    let mut store = MemoryStore::new();
    let id1 = store.append(&a, &vk).unwrap();
    let id2 = store.append(&a, &vk).unwrap();
    assert_eq!(id1, id2);
    assert_eq!(store.len(), 1);
}

// ===========================================================================
// Phase 13 — Adversarial Hardening Tests
// ===========================================================================

use got_core::DirectionalDrift;

fn phase13_geometry(dim: usize) -> CausalGeometry {
    let mut data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }
    let u = UnembeddingMatrix::new(dim, dim, data).unwrap();
    CausalGeometry::from_unembedding(&u, 0.0)
}

fn phase13_probe(dim: usize, name: &str) -> got_probe::ProbeVector {
    let geometry = phase13_geometry(dim);
    let mut training: Vec<(Vec<f32>, bool)> = Vec::new();
    for _ in 0..10 {
        let mut pos = vec![0.0f32; dim];
        pos[0] = 1.0;
        training.push((pos, true));
        let mut neg = vec![0.0f32; dim];
        neg[0] = -1.0;
        training.push((neg, false));
    }
    train_probe(&training, &geometry, name, 0.1, 100).unwrap()
}

fn phase13_enclave_key() -> SigningKey {
    SigningKey::from_bytes(&[0xEE; 32])
}

// ---------------------------------------------------------------------------
// Sequence number tests
// ---------------------------------------------------------------------------

#[test]
fn phase13_consecutive_attestations_have_seq_0_1() {
    let dim = 4;
    let geometry = phase13_geometry(dim);
    let probe = phase13_probe(dim, "seq-test");
    let mut enclave = MockEnclave::new(
        phase13_enclave_key(),
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        None,
    );

    let tap = MockDmaTap::new(dim, vec![]);
    let h = vec![1.0, 0.5, -0.3, 0.8];

    // First attestation → sequence 0
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();
    let a0 = enclave
        .attest("seq-m", [0xAA; 32], None, None, None)
        .unwrap();
    assert_eq!(a0.sequence_number, 0);

    // Reset for next window, counter preserved
    enclave.reset();

    // Second attestation → sequence 1
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();
    let a1 = enclave
        .attest("seq-m", [0xAA; 32], None, None, None)
        .unwrap();
    assert_eq!(a1.sequence_number, 1);
}

#[test]
fn phase13_verify_chain_rejects_gap() {
    let dim = 4;
    let key = phase13_enclave_key();
    let geometry = phase13_geometry(dim);
    let probe = phase13_probe(dim, "gap-test");
    let mut enclave = MockEnclave::new(
        key.clone(),
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        None,
    );
    let tap = MockDmaTap::new(dim, vec![]);
    let h = vec![1.0, 0.5, -0.3, 0.8];

    // Attestation 0
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();
    let a0 = enclave
        .attest("gap-m", [0xAA; 32], None, Some([0xDD; 32]), None)
        .unwrap();
    let a0_hash = got_attest::attestation_hash(&a0).unwrap();
    enclave.reset();

    // Attestation 1 (sequence 1) — we produce but discard
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();
    let _a1_discarded = enclave
        .attest("gap-m", [0xAA; 32], Some(a0_hash), Some([0xDD; 32]), None)
        .unwrap();
    enclave.reset();

    // Attestation 2 (sequence 2) — present this as child of a0
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();
    let a2 = enclave
        .attest("gap-m", [0xAA; 32], Some(a0_hash), Some([0xDD; 32]), None)
        .unwrap();
    assert_eq!(a2.sequence_number, 2);

    // verify_chain should reject: a0 (seq 0) → a2 (seq 2), gap at index 1
    let result = verify_chain(&[a0], &a2, &[key.verifying_key()], 1.0);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("sequence gap"),
        "expected sequence gap error, got: {err_msg}"
    );
}

#[test]
fn phase13_verify_chain_rejects_duplicate_seq() {
    // Construct two attestations with same sequence number 0.
    // The second attestation is manually built to have seq 0 but chain to the first.
    // Since MockEnclave auto-increments, we forge this with assemble_and_sign directly.
    let key = phase13_enclave_key();

    let a0 = GeometricAttestation {
        schema_version: SCHEMA_VERSION_3,
        model_id: "dupe-m".to_string(),
        model_hash: Some([0xAA; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0u8; 32],
        timestamp: 1000,
        corpus_version: "c1".to_string(),
        probe_version: "p1".to_string(),
        layer_readings: vec![vec![0.5]],
        confidence: vec![0.9],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    let a0 = assemble_and_sign(a0, &key).unwrap();
    let a0_hash = got_attest::attestation_hash(&a0).unwrap();

    // Second attestation with SAME sequence_number 0
    let a1 = GeometricAttestation {
        schema_version: SCHEMA_VERSION_3,
        model_id: "dupe-m".to_string(),
        model_hash: Some([0xAA; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0u8; 32],
        timestamp: 1001,
        corpus_version: "c1".to_string(),
        probe_version: "p1".to_string(),
        layer_readings: vec![vec![0.5]],
        confidence: vec![0.9],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: Some(a0_hash),
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0, // DUPLICATE: should be 1
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    let a1 = assemble_and_sign(a1, &key).unwrap();

    let result = verify_chain(&[a0], &a1, &[key.verifying_key()], 1.0);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("sequence gap"),
        "duplicate seq should trigger gap error, got: {err_msg}"
    );
}

#[test]
fn phase13_reset_preserves_sequence_counter() {
    let dim = 4;
    let geometry = phase13_geometry(dim);
    let probe = phase13_probe(dim, "reset-seq");
    let mut enclave = MockEnclave::new(
        phase13_enclave_key(),
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        None,
    );
    let tap = MockDmaTap::new(dim, vec![]);
    let h = vec![1.0, 0.5, -0.3, 0.8];

    // Produce attestation 0
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();
    let a0 = enclave.attest("r-m", [0xAA; 32], None, None, None).unwrap();
    assert_eq!(a0.sequence_number, 0);

    // Reset
    enclave.reset();
    assert_eq!(enclave.frame_count(), 0);

    // Next attestation must be 1, not 0
    let frame = tap.capture(0, 0, &h).unwrap();
    enclave.receive_activations(frame).unwrap();
    let a1 = enclave.attest("r-m", [0xAA; 32], None, None, None).unwrap();
    assert_eq!(
        a1.sequence_number, 1,
        "reset must not reset sequence counter"
    );
}

#[test]
fn phase13_sequence_in_signed_payload() {
    // Verify that tampering with the sequence number invalidates the signature.
    let key = phase13_enclave_key();
    let dim = 4;
    let geometry = phase13_geometry(dim);
    let probe = phase13_probe(dim, "sig-seq");
    let mut enclave = MockEnclave::new(
        key.clone(),
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        None,
    );
    let tap = MockDmaTap::new(dim, vec![]);
    let frame = tap.capture(0, 0, &[1.0, 0.5, -0.3, 0.8]).unwrap();
    enclave.receive_activations(frame).unwrap();

    let mut attest = enclave
        .attest("sig-m", [0xAA; 32], None, None, None)
        .unwrap();
    verify(&attest, &key.verifying_key()).unwrap();

    // Tamper with sequence_number
    attest.sequence_number = 999;
    assert!(
        verify(&attest, &key.verifying_key()).is_err(),
        "tampered sequence_number must invalidate signature"
    );
}

// ---------------------------------------------------------------------------
// Directional drift tests
// ---------------------------------------------------------------------------

#[test]
fn phase13_directional_drift_zero_for_identical() {
    let geometry = phase13_geometry(3);
    let direction = vec![1.0, 0.0, 0.0];
    let drift = geometry.directional_drift(&geometry, &direction).unwrap();
    assert!(
        drift.abs() < 1e-6,
        "identical geometries should have 0 directional drift, got {drift}"
    );
}

#[test]
fn phase13_directional_drift_detects_targeted_change() {
    // Reference: identity Gram. Modified: scale dimension 0 by 2x.
    // Global Frobenius drift might be moderate, but directional drift
    // along [1,0,0] should be large.
    let dim = 3;
    let ref_geom = phase13_geometry(dim);

    // Modified geometry: scale first row/col of Gram by 2x
    let mut data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }
    data[0] = 2.0; // Gram[0,0] = 2 instead of 1
    let u_mod = UnembeddingMatrix::new(dim, dim, {
        // To get this Gram we need: UᵀU = Gram
        // For a diagonal Gram with [2,1,1], U can be diag(√2, 1, 1)
        let mut u = vec![0.0f32; dim * dim];
        u[0] = 2.0f32.sqrt();
        u[dim + 1] = 1.0;
        u[2 * dim + 2] = 1.0;
        u
    })
    .unwrap();
    let mod_geom = CausalGeometry::from_unembedding(&u_mod, 0.0);

    // Global drift
    let global_drift = mod_geom.drift_from(&ref_geom).unwrap();

    // Directional drift along [1,0,0] — the modified direction
    let dir_drift = mod_geom
        .directional_drift(&ref_geom, &[1.0, 0.0, 0.0])
        .unwrap();

    // Directional drift along [0,1,0] — untouched direction
    let unmod_drift = mod_geom
        .directional_drift(&ref_geom, &[0.0, 1.0, 0.0])
        .unwrap();

    assert!(
        dir_drift > 0.5,
        "directional drift along modified direction should be large, got {dir_drift}"
    );
    assert!(
        unmod_drift < 1e-6,
        "directional drift along untouched direction should be ~0, got {unmod_drift}"
    );
    // The directional drift catches what global might dilute
    assert!(
        dir_drift > global_drift || global_drift > 0.1,
        "directional drift ({dir_drift}) should catch targeted manipulation"
    );
}

#[test]
fn phase13_read_probe_checked_rejects_directional_drift() {
    let dim = 3;
    let ref_geom = phase13_geometry(dim);

    // Modified geometry with changed first dimension
    let mut u = vec![0.0f32; dim * dim];
    u[0] = 2.0f32.sqrt();
    u[dim + 1] = 1.0;
    u[2 * dim + 2] = 1.0;
    let u_mat = UnembeddingMatrix::new(dim, dim, u).unwrap();
    let cur_geom = CausalGeometry::from_unembedding(&u_mat, 0.0);

    let probe = phase13_probe(dim, "dir-reject");
    let probe_set = ProbeSet {
        probes: vec![probe.clone()],
        version: "p13".to_string(),
        corpus_version: "c13".to_string(),
        layer: 0,
        geometry_hash: None,
        max_drift: None,
        max_directional_drift: Some(0.01), // very tight bound
    };

    let h = vec![1.0, 0.5, 0.3];
    let result = got_probe::read_probe_checked(&probe, &probe_set, &h, &cur_geom, &ref_geom);
    assert!(
        result.is_err(),
        "should reject when directional drift exceeds tight bound"
    );
}

#[test]
fn phase13_read_probe_checked_passes_within_bound() {
    let dim = 3;
    let ref_geom = phase13_geometry(dim);
    let probe = phase13_probe(dim, "dir-pass");
    let probe_set = ProbeSet {
        probes: vec![probe.clone()],
        version: "p13".to_string(),
        corpus_version: "c13".to_string(),
        layer: 0,
        geometry_hash: None,
        max_drift: None,
        max_directional_drift: Some(0.5), // generous bound
    };

    // Same geometry → drift = 0 → passes
    let h = vec![1.0, 0.5, 0.3];
    let result = got_probe::read_probe_checked(&probe, &probe_set, &h, &ref_geom, &ref_geom);
    assert!(
        result.is_ok(),
        "should pass when directional drift is zero (same geometry)"
    );
}

#[test]
fn phase13_directional_drifts_in_signed_payload() {
    // Verify that tampering with directional_drifts invalidates the signature.
    let key = phase13_enclave_key();

    let attest = GeometricAttestation {
        schema_version: SCHEMA_VERSION_3,
        model_id: "dd-m".to_string(),
        model_hash: Some([0xAA; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0u8; 32],
        timestamp: 1000,
        corpus_version: "c1".to_string(),
        probe_version: "p1".to_string(),
        layer_readings: vec![vec![0.5]],
        confidence: vec![0.9],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    let mut signed = assemble_and_sign(attest, &key).unwrap();
    verify(&signed, &key.verifying_key()).unwrap();

    // Tamper: inject a directional drift record
    signed.directional_drifts.push(DirectionalDrift {
        probe_name: "fake".to_string(),
        drift: 0.42,
    });
    assert!(
        verify(&signed, &key.verifying_key()).is_err(),
        "tampered directional_drifts must invalidate signature"
    );
}

// ---------------------------------------------------------------------------
// ModelHandle tests
// ---------------------------------------------------------------------------

#[test]
fn phase13_closure_model_handle_same_result() {
    let geom = phase13_geometry(3);
    let probe = phase13_probe(3, "mh-closure");
    let h = vec![1.0, 0.5, 0.3];

    // Direct closure wrapping via ClosureModelHandle
    let handle = ClosureModelHandle::new(|h_in: &[f32]| h_in.to_vec());
    let score = causal_check(&probe, &h, &geom, 0.1, &handle, DEFAULT_CAUSAL_THRESHOLD).unwrap();

    // Must produce meaningful results
    assert!(score.delta_plus >= 0.0);
    assert!(score.delta_minus >= 0.0);
}

#[test]
fn phase13_enclave_runs_causal_with_model_handle() {
    let dim = 4;
    let geometry = phase13_geometry(dim);
    let probe = phase13_probe(dim, "mh-enclave");
    let model: Box<dyn ModelHandle> = Box::new(ClosureModelHandle::new(|h: &[f32]| h.to_vec()));
    let mut enclave = MockEnclave::new(
        phase13_enclave_key(),
        geometry,
        vec![probe],
        MockEnclaveConfig::default(),
        Some(model),
    );

    let tap = MockDmaTap::new(dim, vec![]);
    let frame = tap.capture(0, 0, &[1.0, 0.5, -0.3, 0.8]).unwrap();
    enclave.receive_activations(frame).unwrap();

    // Model is enclave-owned — run_causal_check takes only delta
    let scores = enclave.run_causal_check(0.1).unwrap();
    assert_eq!(scores.len(), 1);
    assert!(scores[0].delta_plus >= 0.0);
}

#[test]
fn phase13_enclave_pipeline_with_model_handle() {
    let dim = 4;
    let geometry = phase13_geometry(dim);
    let probe = phase13_probe(dim, "mh-pipe");
    let config = MockEnclaveConfig {
        delta: 0.1,
        causal_threshold: 0.0,
        ..MockEnclaveConfig::default()
    };
    let model: Box<dyn ModelHandle> = Box::new(ClosureModelHandle::new(|h: &[f32]| h.to_vec()));
    let mut enclave = MockEnclave::new(
        phase13_enclave_key(),
        geometry,
        vec![probe],
        config,
        Some(model),
    );
    let tap = MockDmaTap::new(dim, vec![]);

    let activations = vec![(0usize, 0usize, vec![1.0f32, 0.5, -0.3, 0.8])];

    let (attest, scores) = enclave_pipeline(
        &mut enclave,
        &tap,
        &activations,
        "mh-model",
        [0xAA; 32],
        None,
        None,
        None,
    )
    .unwrap();

    verify(&attest, &enclave.verifying_key()).unwrap();
    assert_eq!(scores.len(), 1);
    assert!(attest.causal_flag.is_some());
}

#[test]
fn phase13_existing_causal_tests_pass_with_closure_handle() {
    // This test verifies that wrapping in ClosureModelHandle
    // produces identical results to the old direct-closure path.
    let geom = phase13_geometry(3);
    let probe = phase13_probe(3, "compat");
    let h = vec![1.0, 0.5, 0.3];

    let gram: Vec<f32> = geom.gram().to_vec();
    let d = geom.hidden_dim();
    let model = ClosureModelHandle::new(move |h_in: &[f32]| -> Vec<f32> {
        (0..d)
            .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
            .collect()
    });

    let score = causal_check(&probe, &h, &geom, 1.0, &model, DEFAULT_CAUSAL_THRESHOLD).unwrap();
    assert!(score.delta_plus > 0.0);
    assert!(score.delta_minus > 0.0);
    assert!(
        score.consistency > 0.5,
        "linear model should give good consistency via ClosureModelHandle"
    );
}

// ---------------------------------------------------------------------------
// Security regression tests (Issue 42)
// ---------------------------------------------------------------------------

/// Issue #42 (S-21): `model_hash` is now `Option<[u8; 32]>`.
/// `None` means "model shards not provided" — structurally distinct from any real hash.
#[test]
fn sec_model_hash_option_none_is_distinct() {
    let key = test_key();

    // 1. Attestation with model_hash = None (no shards provided).
    let mut a_none = GeometricAttestation {
        schema_version: SCHEMA_VERSION,
        model_id: "sentinel-test".to_string(),
        model_hash: None,
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0x22; 32],
        timestamp: 1000,
        corpus_version: "v1".to_string(),
        probe_version: "v1".to_string(),
        layer_readings: vec![vec![0.5]],
        confidence: vec![0.9],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    a_none = assemble_and_sign(a_none, &key).unwrap();
    verify(&a_none, &key.verifying_key()).unwrap();
    assert_eq!(
        a_none.model_hash, None,
        "None model_hash must survive sign/verify round-trip"
    );

    // 2. Attestation with model_hash = Some([0u8; 32]) (all-zero hash).
    let mut a_zero = GeometricAttestation {
        schema_version: SCHEMA_VERSION,
        model_id: "sentinel-test".to_string(),
        model_hash: Some([0u8; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0x22; 32],
        timestamp: 1000,
        corpus_version: "v1".to_string(),
        probe_version: "v1".to_string(),
        layer_readings: vec![vec![0.5]],
        confidence: vec![0.9],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        signature: [0u8; 64],
    };
    a_zero = assemble_and_sign(a_zero, &key).unwrap();
    verify(&a_zero, &key.verifying_key()).unwrap();
    assert_eq!(
        a_zero.model_hash,
        Some([0u8; 32]),
        "Some([0; 32]) must survive sign/verify round-trip"
    );

    // 3. None and Some([0; 32]) produce different attestation hashes
    //    (proves they are structurally distinct in the signing payload).
    let h_none = got_attest::attestation_hash(&a_none).unwrap();
    let h_zero = got_attest::attestation_hash(&a_zero).unwrap();
    assert_ne!(
        h_none, h_zero,
        "None and Some([0; 32]) must produce different attestation hashes"
    );
}
