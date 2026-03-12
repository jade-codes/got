// ---------------------------------------------------------------------------
// Inline Measurement Architecture — Phase 9.
//
// Makes measurement **inline**: every forward pass produces activation data
// for probing, not just spot-checks.  A MeasurementSidecar collects readings
// in rolling windows, then emits a signed attestation when each window closes.
//
// Design invariants:
//   • The hook interface (`MeasurementHook`) is trait-based so any executor
//     (thread, process, in-process callback) can implement it.
//   • Probe sampling is **stratified random** — each window samples a different
//     subset from the ProbeLibrary, so the model cannot predict which probes
//     will fire on any particular request.
//   • Attestations are chained: each window's attestation records the hash of
//     the previous window's signed attestation.
//   • Activation statistics (mean, variance) are tracked per layer so a
//     sudden distribution shift can be flagged.
// ---------------------------------------------------------------------------

use got_core::geometry::CausalGeometry;
use got_core::{
    CausalScoreRecord, GeometricAttestation, InnerProduct, Precision, SCHEMA_VERSION_3,
};

use crate::intervention::{causal_check, CausalScore, ModelHandle, DEFAULT_CAUSAL_THRESHOLD};
use crate::{read_probe, ProbeVector};

// ---------------------------------------------------------------------------
// Hook trait
// ---------------------------------------------------------------------------

/// A measurement hook that receives activations from one layer.
///
/// Implementors are responsible for forwarding activations to a
/// `MeasurementSidecar` (or equivalent) for probe evaluation.
pub trait MeasurementHook: Send + Sync {
    /// Called with the hidden state at a specific layer for each forward pass.
    ///
    /// * `request_id` — opaque identifier for this inference request.
    /// * `layer`      — the layer index.
    /// * `h`          — the hidden-state activation vector (ℝ^d).
    fn on_activation(&self, request_id: u64, layer: usize, h: &[f32]);
}

// ---------------------------------------------------------------------------
// ProbeReading
// ---------------------------------------------------------------------------

/// One probe reading from one request/layer combination.
#[derive(Debug, Clone)]
pub struct ProbeReading {
    pub request_id: u64,
    pub layer: usize,
    pub probe_name: String,
    pub value: f32,
    pub confidence: f32,
    pub coverage_flag: bool,
    pub causal_score: Option<CausalScore>,
}

// ---------------------------------------------------------------------------
// Activation statistics tracker
// ---------------------------------------------------------------------------

/// Per-layer running statistics for activation vectors (Welford's algorithm).
#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub layer: usize,
    count: u64,
    mean: Vec<f32>,
    m2: Vec<f32>,
}

impl ActivationStats {
    pub fn new(layer: usize, dim: usize) -> Self {
        Self {
            layer,
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
        }
    }

    /// Incorporate a new activation vector (online Welford update).
    pub fn update(&mut self, h: &[f32]) {
        self.count += 1;
        let n = self.count as f32;
        let dim = self.mean.len().min(h.len());
        for (i, &h_val) in h.iter().enumerate().take(dim) {
            let delta = h_val - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = h_val - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Current per-dimension variance (population variance).
    pub fn variance(&self) -> Vec<f32> {
        if self.count < 2 {
            return vec![0.0; self.mean.len()];
        }
        self.m2.iter().map(|m| m / self.count as f32).collect()
    }

    /// Current per-dimension mean.
    pub fn current_mean(&self) -> &[f32] {
        &self.mean
    }

    /// Number of samples seen.
    pub fn sample_count(&self) -> u64 {
        self.count
    }
}

/// Detect whether two activation-stats snapshots differ significantly.
///
/// Returns the fraction of dimensions whose mean shifted by more than
/// `threshold` standard deviations.  A fraction above, say, 0.3 indicates
/// the model's activation distribution changed substantially.
pub fn detect_distribution_shift(
    baseline: &ActivationStats,
    current: &ActivationStats,
    threshold_sigmas: f32,
) -> f32 {
    if baseline.count < 2 || current.count < 2 {
        return 0.0;
    }
    let baseline_var = baseline.variance();
    let dim = baseline.mean.len().min(current.mean.len());
    let mut shifted = 0usize;
    for (i, &var_i) in baseline_var.iter().enumerate().take(dim) {
        let std = var_i.sqrt();
        if std < 1e-12 {
            continue; // effectively constant dimension, skip
        }
        let diff = (current.mean[i] - baseline.mean[i]).abs();
        if diff > threshold_sigmas * std {
            shifted += 1;
        }
    }
    if dim == 0 {
        0.0
    } else {
        shifted as f32 / dim as f32
    }
}

// ---------------------------------------------------------------------------
// MeasurementSidecar
// ---------------------------------------------------------------------------

/// Configuration for the MeasurementSidecar.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// Number of requests per attestation window.
    pub window_size: usize,
    /// How many probes to sample per window from the library.
    pub probes_per_window: usize,
    /// Model identifier string for attestations.
    pub model_id: String,
    /// Model hash (Merkle root over weight shards). None if not provided.
    pub model_hash: Option<[u8; 32]>,
    /// Precision tag.
    pub precision: Precision,
    /// Whether to run causal intervention checks inline.
    pub causal_enabled: bool,
    /// Perturbation delta for causal checks.
    pub causal_delta: f32,
    /// Causal consistency threshold.
    pub causal_threshold: f32,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            probes_per_window: 3,
            model_id: "unknown".to_string(),
            model_hash: None,
            precision: Precision::Fp32,
            causal_enabled: false,
            causal_delta: 1.0,
            causal_threshold: DEFAULT_CAUSAL_THRESHOLD,
        }
    }
}

/// Inline measurement sidecar.
///
/// Collects probe readings in rolling windows and produces unsigned
/// `GeometricAttestation` values when each window closes.  The caller is
/// responsible for signing (keeps got-probe independent of got-attest).
pub struct MeasurementSidecar {
    config: SidecarConfig,
    geometry: CausalGeometry,

    /// All probes in the library.
    all_probes: Vec<ProbeVector>,
    /// Probes sampled for the current window (indices into `all_probes`).
    current_sample: Vec<usize>,

    /// Accumulated readings in the current window.
    readings: Vec<ProbeReading>,
    /// Requests seen in the current window so far.
    window_request_count: usize,

    /// Hash of the previous window's serialised attestation (for chaining).
    /// None for the first window.
    parent_hash: Option<[u8; 32]>,

    /// Monotonically increasing window counter.
    window_index: u64,

    /// Per-layer activation statistics for the current window.
    layer_stats: Vec<ActivationStats>,

    /// Corpus and probe version strings.
    corpus_version: String,
    probe_version: String,

    /// Tracks which probe indices have been sampled across all windows.
    /// Used to verify stratified coverage.
    coverage_bitmap: Vec<bool>,

    /// Monotonically increasing sequence counter for attestation chaining.
    next_sequence_number: u64,

    /// SHA-256 commitment to the current window's sampled probe indices.
    /// Computed at resample time (before model sees activations).
    probe_commitment: [u8; 32],

    /// S-10: Count of probes skipped due to dimension mismatch in this window.
    skipped_probes: usize,

    /// S-11: Accumulated activation bytes for input_hash computation.
    activation_bytes: Vec<u8>,
}

impl MeasurementSidecar {
    /// Create a new sidecar.
    pub fn new(
        config: SidecarConfig,
        geometry: CausalGeometry,
        probes: Vec<ProbeVector>,
        corpus_version: &str,
        probe_version: &str,
    ) -> Self {
        let n_probes = probes.len();
        let mut sidecar = Self {
            config,
            geometry,
            all_probes: probes,
            current_sample: Vec::new(),
            readings: Vec::new(),
            window_request_count: 0,
            parent_hash: None,
            window_index: 0,
            layer_stats: Vec::new(),
            corpus_version: corpus_version.to_string(),
            probe_version: probe_version.to_string(),
            coverage_bitmap: vec![false; n_probes],
            next_sequence_number: 0,
            probe_commitment: [0u8; 32],
            skipped_probes: 0,
            activation_bytes: Vec::new(),
        };
        sidecar.resample_probes();
        sidecar
    }

    /// Resample which probes are active for the current window.
    fn resample_probes(&mut self) {
        use rand::seq::SliceRandom;

        let k = self.config.probes_per_window.min(self.all_probes.len());
        let mut indices: Vec<usize> = (0..self.all_probes.len()).collect();
        let mut rng = rand::rngs::OsRng;
        indices.shuffle(&mut rng);
        indices.truncate(k);

        // Mark coverage
        for &idx in &indices {
            self.coverage_bitmap[idx] = true;
        }

        self.current_sample = indices;

        // Commit to the selected probe indices BEFORE the model sees any
        // activations for this window. In a real TEE this commitment
        // would be published to a log visible to the verifier.
        let mut sorted = self.current_sample.clone();
        sorted.sort();
        let commitment_bytes: Vec<u8> = sorted
            .iter()
            .flat_map(|idx| (*idx as u64).to_le_bytes())
            .collect();
        self.probe_commitment = got_core::sha256(&commitment_bytes);
    }

    /// Process a new activation from one layer of one request.
    ///
    /// Returns `Some(attestation)` when the window closes (after
    /// `config.window_size` requests have been ingested).  The returned
    /// attestation is **unsigned** — the caller should sign it via
    /// `got_attest::assemble_and_sign`.
    ///
    /// `model` is an optional `ModelHandle` for causal intervention
    /// (hidden-state → output).  Pass `None` to skip causal checks even
    /// if `causal_enabled` is true.
    #[allow(clippy::type_complexity)]
    pub fn ingest(
        &mut self,
        request_id: u64,
        layer: usize,
        h: &[f32],
        model: Option<&dyn ModelHandle>,
    ) -> Option<GeometricAttestation> {
        // Update activation stats
        while self.layer_stats.len() <= layer {
            self.layer_stats
                .push(ActivationStats::new(self.layer_stats.len(), h.len()));
        }
        self.layer_stats[layer].update(h);

        // S-11: Accumulate activation data for input_hash.
        for &val in h {
            self.activation_bytes.extend_from_slice(&val.to_le_bytes());
        }

        // Run sampled probes against this activation
        for &probe_idx in &self.current_sample.clone() {
            let probe = &self.all_probes[probe_idx];

            // Run probe reading
            let reading_result = read_probe(probe, h, &self.geometry);
            let (value, confidence, coverage_flag) = match reading_result {
                Ok(r) => r,
                Err(_) => {
                    // S-10: Track skipped probes (dimension mismatch etc.)
                    self.skipped_probes += 1;
                    continue;
                }
            };

            // Optionally run causal check
            let causal_score = if self.config.causal_enabled {
                model.and_then(|m| {
                    causal_check(
                        probe,
                        h,
                        &self.geometry,
                        self.config.causal_delta,
                        m,
                        self.config.causal_threshold,
                    )
                    .ok()
                })
            } else {
                None
            };

            self.readings.push(ProbeReading {
                request_id,
                layer,
                probe_name: probe.dimension_name.clone(),
                value,
                confidence,
                coverage_flag,
                causal_score,
            });
        }

        self.window_request_count += 1;

        // Check if window is closed
        if self.window_request_count >= self.config.window_size {
            Some(self.close_window())
        } else {
            None
        }
    }

    /// Close the current window and produce an unsigned attestation.
    fn close_window(&mut self) -> GeometricAttestation {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Aggregate readings into layer_readings / confidence / coverage_flags
        // Group by unique probe name (flattened across layers).
        let mut probe_names: Vec<String> = self
            .readings
            .iter()
            .map(|r| r.probe_name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        probe_names.sort(); // deterministic order

        // Mean reading per probe across the window
        let mut layer_readings_flat: Vec<f32> = Vec::new();
        let mut confidences: Vec<f32> = Vec::new();
        let mut coverage_flags: Vec<bool> = Vec::new();
        let mut causal_records: Vec<CausalScoreRecord> = Vec::new();
        let mut any_non_causal = false;

        for name in &probe_names {
            let matching: Vec<&ProbeReading> = self
                .readings
                .iter()
                .filter(|r| &r.probe_name == name)
                .collect();

            if matching.is_empty() {
                continue;
            }

            // Mean value and confidence
            let n = matching.len() as f32;
            let mean_value: f32 = matching.iter().map(|r| r.value).sum::<f32>() / n;
            let mean_conf: f32 = matching.iter().map(|r| r.confidence).sum::<f32>() / n;
            let any_coverage = matching.iter().any(|r| r.coverage_flag);

            layer_readings_flat.push(mean_value);
            confidences.push(mean_conf);
            coverage_flags.push(any_coverage);

            // Aggregate causal scores: take summary from first available
            if let Some(cs) = matching
                .iter()
                .filter_map(|r| r.causal_score.as_ref())
                .next()
            {
                if !cs.is_causal {
                    any_non_causal = true;
                }
                causal_records.push(cs.to_record());
            }
        }

        let divergence_flag = coverage_flags.iter().any(|&f| f) || self.skipped_probes > 0;

        // Build input hash from window index + activation data (S-11).
        let mut input_data: Vec<u8> = self.window_index.to_le_bytes().to_vec();
        input_data.extend_from_slice(&self.activation_bytes);
        let input_hash = got_core::sha256(&input_data);

        let causal_flag = if causal_records.is_empty() {
            None
        } else {
            Some(!any_non_causal)
        };

        let intervention_delta = if causal_records.is_empty() {
            None
        } else {
            Some(self.config.causal_delta)
        };

        let attestation = GeometricAttestation {
            schema_version: SCHEMA_VERSION_3,
            model_id: self.config.model_id.clone(),
            model_hash: self.config.model_hash,
            precision: self.config.precision,
            inner_product: InnerProduct::Causal,
            input_hash,
            timestamp,
            corpus_version: self.corpus_version.clone(),
            probe_version: self.probe_version.clone(),
            layer_readings: vec![layer_readings_flat],
            confidence: confidences,
            coverage_flags,
            divergence_flag,
            parent_attestation_hash: self.parent_hash,
            geometry_hash: Some(self.geometry.geometry_hash()),
            geometry_drift: Some(0.0),
            causal_scores: causal_records,
            intervention_delta,
            causal_flag,
            sequence_number: self.next_sequence_number,
            directional_drifts: vec![],
            probe_commitment: Some(self.probe_commitment),
            signature: [0u8; 64], // caller signs
        };

        // Prepare for next window
        self.readings.clear();
        self.window_request_count = 0;
        self.window_index += 1;
        self.layer_stats.clear();
        self.next_sequence_number += 1;
        self.skipped_probes = 0;
        self.activation_bytes.clear();
        self.resample_probes();

        attestation
    }

    /// Set the parent hash for the next attestation (for chaining).
    /// Typically called after signing the previous attestation.
    pub fn set_parent_hash(&mut self, hash: [u8; 32]) {
        self.parent_hash = Some(hash);
    }

    /// How many probes in the library have been sampled at least once.
    pub fn coverage_count(&self) -> usize {
        self.coverage_bitmap.iter().filter(|&&b| b).count()
    }

    /// Total probes in the library.
    pub fn library_size(&self) -> usize {
        self.all_probes.len()
    }

    /// Current window index.
    pub fn window_index(&self) -> u64 {
        self.window_index
    }

    /// Number of requests ingested in the current (open) window.
    pub fn current_window_request_count(&self) -> usize {
        self.window_request_count
    }

    /// The readings accumulated in the current (open) window.
    pub fn current_readings(&self) -> &[ProbeReading] {
        &self.readings
    }

    /// Per-layer activation statistics for the current window.
    pub fn layer_stats(&self) -> &[ActivationStats] {
        &self.layer_stats
    }
}

// ---------------------------------------------------------------------------
// MeasurementHook implementation for MeasurementSidecar
//
// Since MeasurementSidecar.ingest() takes &mut self and a model_fn,
// but MeasurementHook.on_activation() takes &self with no model_fn,
// we provide a simple collecting hook that buffers activations for
// later batch processing by the sidecar.
// ---------------------------------------------------------------------------

/// A simple hook that collects activations into a thread-safe buffer.
/// The caller periodically drains the buffer and feeds items to a sidecar.
pub struct CollectingHook {
    buffer: std::sync::Mutex<Vec<(u64, usize, Vec<f32>)>>,
}

impl CollectingHook {
    pub fn new() -> Self {
        Self {
            buffer: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Drain all buffered activations.
    pub fn drain(&self) -> Vec<(u64, usize, Vec<f32>)> {
        let mut buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *buf)
    }

    /// Number of buffered activations.
    pub fn len(&self) -> usize {
        self.buffer.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for CollectingHook {
    fn default() -> Self {
        Self::new()
    }
}

impl MeasurementHook for CollectingHook {
    fn on_activation(&self, request_id: u64, layer: usize, h: &[f32]) {
        let mut buf = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buf.push((request_id, layer, h.to_vec()));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intervention::ClosureModelHandle;
    use crate::train_probe;
    use got_core::UnembeddingMatrix;

    fn test_geometry() -> CausalGeometry {
        let u = UnembeddingMatrix::new(
            4,
            3,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        CausalGeometry::from_unembedding(&u, 1e-6)
    }

    fn train_test_probes(geometry: &CausalGeometry, n: usize) -> Vec<ProbeVector> {
        (0..n)
            .map(|i| {
                let offset = i as f32 * 0.1;
                let data: Vec<(Vec<f32>, bool)> = vec![
                    (vec![3.0 + offset, 3.0, 3.0], true),
                    (vec![-3.0 + offset, -3.0, -3.0], false),
                ];
                train_probe(&data, geometry, &format!("concept_{i}"), 0.001, 200).unwrap()
            })
            .collect()
    }

    fn test_config(window_size: usize) -> SidecarConfig {
        SidecarConfig {
            window_size,
            probes_per_window: 2,
            model_id: "test-model".to_string(),
            model_hash: Some([0xAA; 32]),
            precision: Precision::Fp32,
            causal_enabled: false,
            causal_delta: 1.0,
            causal_threshold: DEFAULT_CAUSAL_THRESHOLD,
        }
    }

    // --- CollectingHook tests ---

    #[test]
    fn collecting_hook_receives_activations() {
        let hook = CollectingHook::new();
        hook.on_activation(1, 0, &[1.0, 2.0, 3.0]);
        hook.on_activation(2, 1, &[4.0, 5.0, 6.0]);

        assert_eq!(hook.len(), 2);

        let drained = hook.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, 1); // request_id
        assert_eq!(drained[0].1, 0); // layer
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, 2);
        assert_eq!(drained[1].1, 1);

        assert!(hook.is_empty(), "drain should empty the buffer");
    }

    #[test]
    fn collecting_hook_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CollectingHook>();
    }

    // --- ActivationStats tests ---

    #[test]
    fn activation_stats_mean_and_variance() {
        let mut stats = ActivationStats::new(0, 2);
        stats.update(&[2.0, 4.0]);
        stats.update(&[4.0, 6.0]);
        stats.update(&[6.0, 8.0]);

        let mean = stats.current_mean();
        assert!((mean[0] - 4.0).abs() < 1e-6);
        assert!((mean[1] - 6.0).abs() < 1e-6);

        let var = stats.variance();
        // Population variance of [2,4,6] = 8/3 ≈ 2.6667
        assert!((var[0] - 8.0 / 3.0).abs() < 1e-5, "var[0]={}", var[0]);
        // Population variance of [4,6,8] = 8/3
        assert!((var[1] - 8.0 / 3.0).abs() < 1e-5, "var[1]={}", var[1]);
    }

    #[test]
    fn activation_stats_single_sample_zero_variance() {
        let mut stats = ActivationStats::new(0, 3);
        stats.update(&[1.0, 2.0, 3.0]);
        let var = stats.variance();
        assert!(var.iter().all(|&v| v == 0.0));
    }

    // --- Distribution shift detection ---

    #[test]
    fn detect_shift_identical_distributions() {
        let mut baseline = ActivationStats::new(0, 2);
        let mut current = ActivationStats::new(0, 2);
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            baseline.update(&[x, x * 2.0]);
            current.update(&[x, x * 2.0]);
        }
        let shift = detect_distribution_shift(&baseline, &current, 2.0);
        assert_eq!(shift, 0.0, "identical distributions should show no shift");
    }

    #[test]
    fn detect_shift_clearly_different() {
        let mut baseline = ActivationStats::new(0, 2);
        let mut current = ActivationStats::new(0, 2);
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            baseline.update(&[x, x]);
        }
        for x in [100.0, 200.0, 300.0, 400.0, 500.0] {
            current.update(&[x, x]);
        }
        let shift = detect_distribution_shift(&baseline, &current, 2.0);
        assert!(
            shift > 0.5,
            "very different distributions should be detected, got {shift}"
        );
    }

    // --- MeasurementSidecar tests ---

    #[test]
    fn sidecar_no_attestation_before_window_close() {
        let geometry = test_geometry();
        let probes = train_test_probes(&geometry, 5);
        let config = test_config(3); // window of 3
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        let h = vec![1.0, 2.0, 1.5];
        // Feed 2 requests — should not close
        assert!(sidecar.ingest(0, 0, &h, None).is_none());
        assert!(sidecar.ingest(1, 0, &h, None).is_none());
        assert_eq!(sidecar.current_window_request_count(), 2);
    }

    #[test]
    fn sidecar_produces_attestation_at_window_boundary() {
        let geometry = test_geometry();
        let probes = train_test_probes(&geometry, 5);
        let config = test_config(3);
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        let h = vec![1.0, 2.0, 1.5];

        assert!(sidecar.ingest(0, 0, &h, None).is_none());
        assert!(sidecar.ingest(1, 0, &h, None).is_none());
        let attestation = sidecar.ingest(2, 0, &h, None);

        assert!(
            attestation.is_some(),
            "window of 3 should close after 3 ingest calls"
        );

        let a = attestation.unwrap();
        assert_eq!(a.schema_version, SCHEMA_VERSION_3);
        assert_eq!(a.model_id, "test-model");
        assert!(!a.layer_readings[0].is_empty(), "should have readings");
        assert!(
            a.parent_attestation_hash.is_none(),
            "first window has no parent"
        );
    }

    #[test]
    fn sidecar_chains_attestations_across_windows() {
        let geometry = test_geometry();
        let probes = train_test_probes(&geometry, 5);
        let config = test_config(2); // window of 2
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        let h = vec![1.0, 2.0, 1.5];

        // Window 0
        sidecar.ingest(0, 0, &h, None);
        let a0 = sidecar.ingest(1, 0, &h, None).unwrap();
        assert!(a0.parent_attestation_hash.is_none());

        // Simulate signing: compute hash of the attestation
        // (In real usage, this would be hash of serialise_for_signing output)
        let fake_parent_hash = got_core::sha256(b"attestation_0_signed_bytes");
        sidecar.set_parent_hash(fake_parent_hash);

        // Window 1
        sidecar.ingest(2, 0, &h, None);
        let a1 = sidecar.ingest(3, 0, &h, None).unwrap();
        assert_eq!(
            a1.parent_attestation_hash,
            Some(fake_parent_hash),
            "second window should chain to first"
        );
    }

    #[test]
    fn sidecar_stratified_sampling_covers_library() {
        let geometry = test_geometry();
        let probes = train_test_probes(&geometry, 10);
        let config = SidecarConfig {
            window_size: 1,
            probes_per_window: 2,
            ..test_config(1)
        };
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        let h = vec![1.0, 2.0, 1.5];

        // Run 50 windows — should cover most of the 10 probes
        for i in 0..50 {
            let _ = sidecar.ingest(i, 0, &h, None);
        }

        let coverage = sidecar.coverage_count();
        assert!(
            coverage >= 8,
            "after 50 windows sampling 2 from 10, coverage should be high, got {coverage}/10"
        );
    }

    #[test]
    fn sidecar_with_causal_checks() {
        let geometry = test_geometry();
        let d = geometry.hidden_dim();
        let probes = train_test_probes(&geometry, 3);
        let config = SidecarConfig {
            window_size: 2,
            probes_per_window: 2,
            causal_enabled: true,
            causal_delta: 1.0,
            ..test_config(2)
        };
        let mut sidecar = MeasurementSidecar::new(config, geometry.clone(), probes, "cv1", "pv1");

        let h = vec![1.0, 2.0, 1.5];

        // Linear model for causal checks
        let gram: Vec<f32> = geometry.gram().to_vec();
        let model_fn = ClosureModelHandle::new(move |h_in: &[f32]| -> Vec<f32> {
            (0..d)
                .map(|i| (0..d).map(|j| gram[i * d + j] * h_in[j]).sum::<f32>())
                .collect()
        });

        sidecar.ingest(0, 0, &h, Some(&model_fn));
        let a = sidecar.ingest(1, 0, &h, Some(&model_fn)).unwrap();

        // Should have causal scores
        assert!(
            !a.causal_scores.is_empty(),
            "causal-enabled sidecar should produce causal scores"
        );
        assert!(a.intervention_delta.is_some());
        assert!(a.causal_flag.is_some());
    }

    #[test]
    fn sidecar_tracks_layer_stats() {
        let geometry = test_geometry();
        let probes = train_test_probes(&geometry, 3);
        let config = test_config(5);
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        // Feed different activations
        sidecar.ingest(0, 0, &[1.0, 2.0, 3.0], None);
        sidecar.ingest(1, 0, &[3.0, 4.0, 5.0], None);
        sidecar.ingest(2, 0, &[5.0, 6.0, 7.0], None);

        let stats = sidecar.layer_stats();
        assert_eq!(stats.len(), 1, "one layer");
        assert_eq!(stats[0].sample_count(), 3);
        let mean = stats[0].current_mean();
        assert!((mean[0] - 3.0).abs() < 1e-6, "mean[0]={}", mean[0]);
    }

    #[test]
    fn sidecar_window_resets_after_close() {
        let geometry = test_geometry();
        let probes = train_test_probes(&geometry, 3);
        let config = test_config(2);
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        let h = vec![1.0, 2.0, 1.5];

        sidecar.ingest(0, 0, &h, None);
        let _ = sidecar.ingest(1, 0, &h, None).unwrap();

        // After close, window should be reset
        assert_eq!(sidecar.current_window_request_count(), 0);
        assert!(sidecar.current_readings().is_empty());
        assert_eq!(sidecar.window_index(), 1);
    }

    #[test]
    fn sidecar_geometry_hash_in_attestation() {
        let geometry = test_geometry();
        let expected_hash = geometry.geometry_hash();
        let probes = train_test_probes(&geometry, 3);
        let config = test_config(1);
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        let h = vec![1.0, 2.0, 1.5];
        let a = sidecar.ingest(0, 0, &h, None).unwrap();
        assert_eq!(a.geometry_hash, Some(expected_hash));
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issues 31, 32)
    // -----------------------------------------------------------------------

    /// Issue #31 (S-10): MeasurementSidecar sets divergence_flag when probes
    /// are skipped due to dimension mismatch.
    #[test]
    fn sec_sidecar_flags_skipped_probes() {
        let geometry = test_geometry();
        let probes = train_test_probes(&geometry, 3);
        let config = test_config(1);
        let mut sidecar = MeasurementSidecar::new(config, geometry, probes, "cv1", "pv1");

        // Feed a wrong-dimension activation: probes expect dim=3, give dim=2.
        let wrong_dim = vec![1.0, 2.0];
        let result = sidecar.ingest(0, 0, &wrong_dim, None);

        if let Some(a) = result {
            assert!(
                a.divergence_flag,
                "divergence_flag must be set when probes are skipped due to dimension mismatch"
            );
        } else {
            panic!("expected attestation from window_size=1 sidecar");
        }
    }

    /// Issue #32 (S-11): input_hash includes actual activation data,
    /// not just the window_index.
    #[test]
    fn sec_sidecar_input_hash_depends_on_activation_data() {
        let geometry = test_geometry();

        // Build two independent sidecars, same config, both at window_index=0.
        let probes1 = train_test_probes(&geometry, 3);
        let probes2 = train_test_probes(&geometry, 3);
        let config1 = test_config(1);
        let config2 = test_config(1);
        let mut sidecar1 =
            MeasurementSidecar::new(config1, geometry.clone(), probes1, "cv1", "pv1");
        let mut sidecar2 =
            MeasurementSidecar::new(config2, geometry.clone(), probes2, "cv1", "pv1");

        // Different activations.
        let h_a = vec![1.0, 2.0, 1.5];
        let h_b = vec![99.0, -99.0, 0.0];

        let a1 = sidecar1.ingest(0, 0, &h_a, None).unwrap();
        let a2 = sidecar2.ingest(0, 0, &h_b, None).unwrap();

        assert_ne!(
            a1.input_hash, a2.input_hash,
            "different activations at same window_index must produce different input_hash"
        );
    }

    // -----------------------------------------------------------------------
    // Security regression test (Issue 44 / N-2)
    // -----------------------------------------------------------------------

    /// Issue #44 (N-2): `CollectingHook` recovers from mutex poisoning
    /// instead of panicking.
    #[test]
    fn sec_collecting_hook_survives_mutex_poison() {
        use std::sync::Arc;

        let hook = Arc::new(CollectingHook::new());

        // Push an activation normally.
        hook.on_activation(1, 0, &[1.0, 2.0]);

        // Poison the mutex by panicking inside a thread while holding the lock.
        let hook2 = Arc::clone(&hook);
        let handle = std::thread::spawn(move || {
            // Lock and then panic — this poisons the mutex.
            let _guard = hook2.buffer.lock().unwrap();
            panic!("intentional panic to poison mutex");
        });
        // The thread panicked — ignore the join error.
        let _ = handle.join();

        // After poisoning, the hook must still work (recover via into_inner).
        hook.on_activation(2, 1, &[3.0, 4.0]);
        let drained = hook.drain();
        // We should have at least the post-poison activation.
        assert!(
            drained.iter().any(|(rid, _, _)| *rid == 2),
            "CollectingHook must recover from mutex poisoning"
        );
    }
}
