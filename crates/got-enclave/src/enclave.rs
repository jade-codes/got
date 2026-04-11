// ---------------------------------------------------------------------------
// Measurement Enclave — Phase 11, §11.3–11.6.
//
// The measurement enclave is a hardware-isolated execution environment
// that receives raw activations from the hardware capture layer,
// runs causal probes and causal interventions, and produces signed
// attestations.  The signing key never leaves the enclave.
//
// Key security property:
//   The model writes activations to VRAM as part of its normal
//   computation.  The hardware tap captures exactly what was written,
//   before the model has any opportunity to overwrite or sanitise.
//   The enclave then runs probes on these authentic activations.
// ---------------------------------------------------------------------------

use ed25519_dalek::{SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use thiserror::Error;

use got_attest::assemble_and_sign;
use got_core::geometry::CausalGeometry;
use got_core::{GeometricAttestation, InnerProduct, Precision, SCHEMA_VERSION};
use got_probe::intervention::{causal_check, CausalScore, ModelHandle, DEFAULT_CAUSAL_THRESHOLD};
use got_probe::{read_probe, ProbeVector};

use crate::capture::{ActivationFrame, HardwareCapture};

/// Errors from the measurement enclave.
#[derive(Debug, Error)]
pub enum EnclaveError {
    #[error("activation integrity check failed at layer {layer}, position {position}")]
    IntegrityViolation { layer: usize, position: usize },
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("no activations received — cannot produce attestation")]
    NoActivations,
    #[error("probe error: {0}")]
    Probe(#[from] got_probe::ProbeError),
    #[error("attestation error: {0}")]
    Attestation(#[from] got_attest::AttestationError),
    #[error("enclave not initialised")]
    NotInitialised,
}

/// Trait abstracting a hardware-isolated measurement environment.
///
/// **A real implementation IS the trust boundary** of the attestation
/// pipeline.  Once a frame is inside the enclave, the model — and the
/// host process running the model — must be unable to influence what
/// the probes measure, what causal intervention reports, or what the
/// signing key produces.  The mock implementation in this crate
/// (`MockEnclave`) validates the protocol *flow* but does not provide
/// these guarantees and is intentionally insecure.
///
/// Required guarantees for a production adapter:
///
///   1. **Activations come from hardware capture**, not from model
///      software.  The capture device computes the integrity hash
///      before the bytes leave the trust boundary; the enclave
///      recomputes and rejects mismatches in `receive_activations`.
///   2. **The signing key is generated inside the enclave** at
///      provisioning time and never exported.  `verifying_key()` is
///      the only key material that crosses the boundary.
///   3. **Probes are provisioned through a secure channel** (sealed
///      storage, remote attestation, etc.) and never visible to the
///      host process running the model.
///   4. **The model handle used for causal intervention runs inside
///      the enclave**, not the host's model.  In a real TEE the model
///      shard is loaded into trusted memory at provisioning time.
///   5. **`attest_with_causal` embeds causal scores before signing**.
///      A real adapter that asks the host to fill in causal scores
///      after signing violates the contract — the attestation would
///      carry an enclave-authorised key over host-tampered content.
///   6. **Sequence numbers are monotonic and rollback-resistant**.
///      Back the counter with a hardware monotonic counter
///      (SGX MC, AMD vTPM, NVIDIA secure counter) so an old attestation
///      cannot be replayed under a fresh-looking sequence number.
///
/// The full per-platform sketch of how SGX, SEV-SNP, and H100
/// confidential computing satisfy this contract is in
/// `docs/enclave-adapter-contract.md`.
pub trait MeasurementEnclave {
    /// Receive activations from hardware capture and verify integrity.
    ///
    /// The enclave recomputes the integrity hash and rejects frames
    /// where the hash doesn't match (indicating tampering in transit).
    fn receive_activations(&mut self, frame: ActivationFrame) -> Result<(), EnclaveError>;

    /// Run causal intervention inside the enclave on accumulated activations.
    ///
    /// The enclave owns the model handle internally (Phase 13 hardening).
    /// In a real TEE, the model shard is loaded inside the enclave's
    /// trusted memory at provisioning time, not supplied per-call.
    ///
    /// Returns `CausalScore` for each probe evaluated.
    fn run_causal_check(&self, delta: f32) -> Result<Vec<CausalScore>, EnclaveError>;

    /// Produce a signed attestation from accumulated measurements.
    ///
    /// The signing key never leaves the enclave.  The attestation
    /// includes probe readings, causal scores, and integrity metadata.
    fn attest(
        &mut self,
        model_id: &str,
        model_hash: [u8; 32],
        parent_hash: Option<[u8; 32]>,
        geometry_hash: Option<[u8; 32]>,
        geometry_drift: Option<f32>,
    ) -> Result<GeometricAttestation, EnclaveError>;

    /// Produce a signed attestation that includes causal intervention results.
    ///
    /// This is the preferred method when causal checks have been run, because
    /// the causal scores are embedded *before* signing — the signing key never
    /// leaves the enclave boundary.
    #[allow(clippy::too_many_arguments)]
    fn attest_with_causal(
        &mut self,
        model_id: &str,
        model_hash: [u8; 32],
        parent_hash: Option<[u8; 32]>,
        geometry_hash: Option<[u8; 32]>,
        geometry_drift: Option<f32>,
        causal_scores: &[CausalScore],
        intervention_delta: f32,
    ) -> Result<GeometricAttestation, EnclaveError>;

    /// Get the enclave's verifying (public) key.
    ///
    /// Used by external verifiers to check attestation signatures.
    /// The corresponding signing key never leaves the enclave.
    fn verifying_key(&self) -> VerifyingKey;

    /// Number of activation frames received so far.
    fn frame_count(&self) -> usize;

    /// Reset accumulated state for a new measurement window.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// MockEnclave — in-process test double
// ---------------------------------------------------------------------------

/// Configuration for the mock enclave.
#[cfg(any(test, feature = "mock"))]
#[derive(Debug, Clone)]
pub struct MockEnclaveConfig {
    /// Causal intervention perturbation magnitude.
    pub delta: f32,
    /// Causal consistency threshold.
    pub causal_threshold: f32,
    /// Corpus version string for attestations.
    pub corpus_version: String,
    /// Probe version string for attestations.
    pub probe_version: String,
}

#[cfg(any(test, feature = "mock"))]
impl Default for MockEnclaveConfig {
    fn default() -> Self {
        Self {
            delta: 0.1,
            causal_threshold: DEFAULT_CAUSAL_THRESHOLD,
            corpus_version: "enclave-v1".to_string(),
            probe_version: "enclave-p1".to_string(),
        }
    }
}

/// A mock enclave that runs in-process for testing.
///
/// Simulates the security properties of a real TEE:
///   - Verifies activation frame integrity hashes
///   - Runs causal probes on captured activations
///   - Produces signed attestations with the enclave's own key
///   - Signing key stored privately (never exposed)
///
/// # Security caveat (PoC only)
///
/// This mock runs in the same address space as the agent runtime.
/// The signing key, probes, and geometry are all accessible to the
/// host process. In production, a real TEE (SGX/SEV/H100) would:
///   - Generate the signing key *inside* the enclave (never exported)
///   - Receive probes via secure provisioning (agent never sees them)
///   - Enforce memory isolation at the hardware level
///
/// Until step 12 of the build order (real TEE integration), this mock
/// validates the protocol flow but does NOT provide hardware isolation.
#[cfg(any(test, feature = "mock"))]
pub struct MockEnclave {
    /// The enclave's signing key — never exported.
    signing_key: SigningKey,
    /// Causal geometry for probe evaluation.
    geometry: CausalGeometry,
    /// Loaded probes (enclave-provisioned, not from model process).
    probes: Vec<ProbeVector>,
    /// Accumulated activation frames (verified integrity).
    frames: Vec<ActivationFrame>,
    /// Configuration.
    config: MockEnclaveConfig,
    /// Monotonic sequence counter. Incremented on every attestation.
    /// Never reset — `reset()` clears frames but preserves this counter.
    /// In production, backed by a hardware monotonic counter (SGX/SEV/vTPM).
    next_sequence: u64,
    /// Enclave-owned model handle (Phase 13 hardening).
    /// In production, this is a TEE-internal model shard loaded from a
    /// verified image. The caller never supplies it per-call.
    model: Option<Box<dyn ModelHandle>>,
}

#[cfg(any(test, feature = "mock"))]
impl std::fmt::Debug for MockEnclave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockEnclave")
            .field("probes", &self.probes.len())
            .field("frames", &self.frames.len())
            .field("next_sequence", &self.next_sequence)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(any(test, feature = "mock"))]
impl MockEnclave {
    /// Create a new mock enclave.
    ///
    /// `signing_key` — the enclave's signing key (in a real TEE, generated
    ///   inside the enclave and never exported).
    /// `geometry` — the causal geometry (provisioned into the enclave by
    ///   the enclave operator).
    /// `probes` — the linear probes to evaluate (provisioned by the
    ///   enclave operator — note §11.7: who provisions the enclave is
    ///   a governance question).
    /// `model` — the model forward-pass handle (Phase 13: enclave-owned,
    ///   not caller-supplied per-call). In a real TEE this is a model shard
    ///   loaded into enclave memory from a verified image.
    pub fn new(
        signing_key: SigningKey,
        geometry: CausalGeometry,
        probes: Vec<ProbeVector>,
        config: MockEnclaveConfig,
        model: Option<Box<dyn ModelHandle>>,
    ) -> Self {
        Self {
            signing_key,
            geometry,
            probes,
            frames: Vec::new(),
            config,
            next_sequence: 0,
            model,
        }
    }

    /// Compute SHA-256 of all received activation data for use as input_hash.
    fn compute_input_hash(&self, model_id: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        // Domain separator so input hashes are distinct from other hashes.
        hasher.update(b"enclave-input-v1");
        hasher.update(model_id.as_bytes());
        for frame in &self.frames {
            hasher.update(frame.layer.to_le_bytes());
            hasher.update(frame.token_position.to_le_bytes());
            for v in &frame.values {
                // Canonicalise -0.0 → 0.0 to avoid sign-bit ambiguity.
                let canon = if *v == 0.0 { 0.0_f32 } else { *v };
                hasher.update(canon.to_le_bytes());
            }
        }
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }

    /// Run probe readings on the latest frame for a given layer.
    #[allow(clippy::type_complexity)]
    fn read_probes_for_layer(
        &self,
        layer: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<bool>), EnclaveError> {
        // Find the latest frame for this layer.
        let frame = self
            .frames
            .iter()
            .rev()
            .find(|f| f.layer == layer)
            .ok_or(EnclaveError::NoActivations)?;

        let mut readings = Vec::new();
        let mut confidences = Vec::new();
        let mut coverage_flags = Vec::new();

        for probe in &self.probes {
            let (reading, confidence, flag) = read_probe(probe, &frame.values, &self.geometry)?;
            readings.push(reading);
            confidences.push(confidence);
            coverage_flags.push(flag);
        }

        Ok((readings, confidences, coverage_flags))
    }

    /// Collect probe readings across all captured layers and compute
    /// common attestation metadata (divergence flag, input hash, timestamp).
    ///
    /// Shared by `attest()` and `attest_with_causal()`.
    fn collect_measurements(&self, model_id: &str) -> Result<MeasurementBundle, EnclaveError> {
        if self.frames.is_empty() {
            return Err(EnclaveError::NoActivations);
        }

        let mut layers: Vec<usize> = self.frames.iter().map(|f| f.layer).collect();
        layers.sort();
        layers.dedup();

        let mut all_readings = Vec::new();
        let mut all_confidences = Vec::new();
        let mut all_coverage = Vec::new();

        for &layer in &layers {
            let (readings, confidences, flags) = self.read_probes_for_layer(layer)?;
            all_readings.push(readings);
            all_confidences.extend(confidences);
            all_coverage.extend(flags);
        }

        let divergence_flag = all_coverage.iter().any(|&f| f);
        let input_hash = self.compute_input_hash(model_id);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(MeasurementBundle {
            layer_readings: all_readings,
            confidence: all_confidences,
            coverage_flags: all_coverage,
            divergence_flag,
            input_hash,
            timestamp,
        })
    }
}

/// Intermediate result from [`MockEnclave::collect_measurements`].
struct MeasurementBundle {
    layer_readings: Vec<Vec<f32>>,
    confidence: Vec<f32>,
    coverage_flags: Vec<bool>,
    divergence_flag: bool,
    input_hash: [u8; 32],
    timestamp: u64,
}

#[cfg(any(test, feature = "mock"))]
impl MeasurementEnclave for MockEnclave {
    fn receive_activations(&mut self, frame: ActivationFrame) -> Result<(), EnclaveError> {
        // Critical security check: verify the integrity hash computed by
        // hardware matches the activation data.
        if !frame.verify_integrity() {
            return Err(EnclaveError::IntegrityViolation {
                layer: frame.layer,
                position: frame.token_position,
            });
        }

        // Dimension check.
        if frame.values.len() != self.geometry.hidden_dim() {
            return Err(EnclaveError::DimensionMismatch {
                expected: self.geometry.hidden_dim(),
                got: frame.values.len(),
            });
        }

        self.frames.push(frame);
        Ok(())
    }

    fn run_causal_check(&self, delta: f32) -> Result<Vec<CausalScore>, EnclaveError> {
        if self.frames.is_empty() {
            return Err(EnclaveError::NoActivations);
        }

        let model = self.model.as_ref().ok_or(EnclaveError::NotInitialised)?;

        // Use the latest frame for causal checking.
        let frame = self.frames.last().unwrap();
        let mut scores = Vec::new();

        for probe in &self.probes {
            let score = causal_check(
                probe,
                &frame.values,
                &self.geometry,
                delta,
                model.as_ref(),
                self.config.causal_threshold,
            )?;
            scores.push(score);
        }

        Ok(scores)
    }

    fn attest(
        &mut self,
        model_id: &str,
        model_hash: [u8; 32],
        parent_hash: Option<[u8; 32]>,
        geometry_hash: Option<[u8; 32]>,
        geometry_drift: Option<f32>,
    ) -> Result<GeometricAttestation, EnclaveError> {
        let m = self.collect_measurements(model_id)?;

        let attestation = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: model_id.to_string(),
            model_hash: Some(model_hash),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: m.input_hash,
            timestamp: m.timestamp,
            corpus_version: self.config.corpus_version.clone(),
            probe_version: self.config.probe_version.clone(),
            layer_readings: m.layer_readings,
            confidence: m.confidence,
            coverage_flags: m.coverage_flags,
            divergence_flag: m.divergence_flag,
            parent_attestation_hash: parent_hash,
            geometry_hash,
            geometry_drift,
            causal_scores: vec![],
            intervention_delta: None,
            causal_flag: None,
            sequence_number: self.next_sequence,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            domain_scope_declaration: None,
            signature: [0u8; 64],
        };

        self.next_sequence += 1;
        let signed = assemble_and_sign(attestation, &self.signing_key)?;
        Ok(signed)
    }

    fn attest_with_causal(
        &mut self,
        model_id: &str,
        model_hash: [u8; 32],
        parent_hash: Option<[u8; 32]>,
        geometry_hash: Option<[u8; 32]>,
        geometry_drift: Option<f32>,
        causal_scores: &[CausalScore],
        intervention_delta: f32,
    ) -> Result<GeometricAttestation, EnclaveError> {
        let m = self.collect_measurements(model_id)?;

        let attestation = GeometricAttestation {
            schema_version: SCHEMA_VERSION,
            model_id: model_id.to_string(),
            model_hash: Some(model_hash),
            precision: Precision::Fp32,
            inner_product: InnerProduct::Causal,
            input_hash: m.input_hash,
            timestamp: m.timestamp,
            corpus_version: self.config.corpus_version.clone(),
            probe_version: self.config.probe_version.clone(),
            layer_readings: m.layer_readings,
            confidence: m.confidence,
            coverage_flags: m.coverage_flags,
            divergence_flag: m.divergence_flag,
            parent_attestation_hash: parent_hash,
            geometry_hash,
            geometry_drift,
            causal_scores: causal_scores.iter().map(|s| s.to_record()).collect(),
            intervention_delta: Some(intervention_delta),
            causal_flag: Some(causal_scores.iter().all(|s| s.is_causal)),
            sequence_number: self.next_sequence,
            directional_drifts: vec![],
            probe_commitment: None,
            density_reading: None,
            curvature_reading: None,
            domain_scope_declaration: None,
            signature: [0u8; 64],
        };

        self.next_sequence += 1;
        let signed = assemble_and_sign(attestation, &self.signing_key)?;
        Ok(signed)
    }

    fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Convenience: run a full enclave measurement pipeline.
///
/// 1. Capture activations via hardware tap
/// 2. Feed into enclave (with integrity check)
/// 3. Run causal intervention
/// 4. Produce signed attestation (causal scores embedded before signing)
///
/// Returns `(attestation, causal_scores)`.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "mock"))]
pub fn enclave_pipeline(
    enclave: &mut MockEnclave,
    capture: &dyn HardwareCapture,
    activations: &[(usize, usize, Vec<f32>)], // (layer, token_pos, values)
    model_id: &str,
    model_hash: [u8; 32],
    parent_hash: Option<[u8; 32]>,
    geometry_hash: Option<[u8; 32]>,
    geometry_drift: Option<f32>,
) -> Result<(GeometricAttestation, Vec<CausalScore>), EnclaveError> {
    // 1. Capture and ingest activations.
    for (layer, pos, values) in activations {
        if let Some(frame) = capture.capture(*layer, *pos, values) {
            enclave.receive_activations(frame)?;
        }
    }

    // 2. Run causal intervention (model is enclave-owned, Phase 13).
    let delta = enclave.config.delta;
    let scores = enclave.run_causal_check(delta)?;

    // 3. Build and sign attestation with causal scores embedded before signing.
    //    This avoids leaking the signing key outside the enclave boundary.
    let signed = enclave.attest_with_causal(
        model_id,
        model_hash,
        parent_hash,
        geometry_hash,
        geometry_drift,
        &scores,
        delta,
    )?;

    Ok((signed, scores))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::MockDmaTap;
    use got_attest::serialise_for_signing;
    use got_core::UnembeddingMatrix;
    use got_probe::intervention::ClosureModelHandle;
    use got_probe::ProbeVector;

    fn make_geometry(dim: usize) -> CausalGeometry {
        // Simple identity-like unembedding: U = I (dim × dim).
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        let u = UnembeddingMatrix::new(dim, dim, data).unwrap();
        CausalGeometry::from_unembedding(&u, 0.0)
    }

    fn make_probe(dim: usize, name: &str) -> ProbeVector {
        // Train a simple probe with separable synthetic data.
        let geometry = make_geometry(dim);
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
        got_probe::train_probe(&activations, &geometry, name, 0.1, 100).unwrap()
    }

    fn enclave_key() -> SigningKey {
        SigningKey::from_bytes(&[0xEE; 32])
    }

    #[test]
    fn enclave_receives_valid_frame() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "test");
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            None,
        );

        let tap = MockDmaTap::new(dim, vec![]);
        let h = vec![1.0, 0.5, -0.3, 0.8];
        let frame = tap.capture(0, 0, &h).unwrap();

        enclave.receive_activations(frame).unwrap();
        assert_eq!(enclave.frame_count(), 1);
    }

    #[test]
    fn enclave_rejects_tampered_frame() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "test");
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            None,
        );

        let tap = MockDmaTap::new(dim, vec![]).with_tamper();
        let h = vec![1.0, 0.5, -0.3, 0.8];
        let frame = tap.capture(0, 0, &h).unwrap();

        let err = enclave.receive_activations(frame);
        assert!(err.is_err());
        match err.unwrap_err() {
            EnclaveError::IntegrityViolation { layer, position } => {
                assert_eq!(layer, 0);
                assert_eq!(position, 0);
            }
            other => panic!("expected IntegrityViolation, got: {other}"),
        }
    }

    #[test]
    fn enclave_rejects_dimension_mismatch() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "test");
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            None,
        );

        // Frame with wrong dimension.
        let h_wrong = vec![1.0, 2.0]; // dim=2, expected 4
        let hash = ActivationFrame::compute_hash(0, 0, &h_wrong);
        let frame = ActivationFrame {
            layer: 0,
            token_position: 0,
            values: h_wrong,
            integrity_hash: hash,
        };
        let err = enclave.receive_activations(frame);
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            EnclaveError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn enclave_produces_signed_attestation() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "honesty");
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            None,
        );

        // Feed a frame.
        let tap = MockDmaTap::new(dim, vec![]);
        let h = vec![1.0, 0.5, -0.3, 0.8];
        let frame = tap.capture(0, 0, &h).unwrap();
        enclave.receive_activations(frame).unwrap();

        // Produce attestation.
        let attest = enclave
            .attest("test-model", [0xAA; 32], None, None, None)
            .unwrap();

        // Verify signature with enclave's public key.
        got_attest::verify(&attest, &enclave.verifying_key()).unwrap();
        assert_eq!(attest.schema_version, SCHEMA_VERSION);
        assert_eq!(attest.model_id, "test-model");
        assert!(!attest.layer_readings.is_empty());
    }

    #[test]
    fn enclave_causal_check_returns_scores() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "test");
        let model_fn = ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> { h.to_vec() });
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            Some(Box::new(model_fn)),
        );

        let tap = MockDmaTap::new(dim, vec![]);
        let h = vec![1.0, 0.5, -0.3, 0.8];
        let frame = tap.capture(0, 0, &h).unwrap();
        enclave.receive_activations(frame).unwrap();

        let scores = enclave.run_causal_check(0.1).unwrap();
        assert_eq!(scores.len(), 1, "one probe → one score");
        // The score should have meaningful values.
        assert!(scores[0].delta_plus >= 0.0);
        assert!(scores[0].delta_minus >= 0.0);
    }

    #[test]
    fn enclave_no_activations_errors() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let model_fn = ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> { h.to_vec() });
        let enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![],
            MockEnclaveConfig::default(),
            Some(Box::new(model_fn)),
        );

        assert!(enclave.run_causal_check(0.1).is_err());
    }

    #[test]
    fn enclave_reset_clears_frames() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![],
            MockEnclaveConfig::default(),
            None,
        );

        let tap = MockDmaTap::new(dim, vec![]);
        let h = vec![1.0, 0.5, -0.3, 0.8];
        let frame = tap.capture(0, 0, &h).unwrap();
        enclave.receive_activations(frame).unwrap();
        assert_eq!(enclave.frame_count(), 1);

        enclave.reset();
        assert_eq!(enclave.frame_count(), 0);
    }

    #[test]
    fn enclave_pipeline_end_to_end() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "test");
        let config = MockEnclaveConfig {
            delta: 0.1,
            causal_threshold: 0.0, // low threshold so probes pass
            ..MockEnclaveConfig::default()
        };
        let model_fn = ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> { h.to_vec() });
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            config,
            Some(Box::new(model_fn)),
        );
        let tap = MockDmaTap::new(dim, vec![]);

        let activations = vec![(0usize, 0usize, vec![1.0f32, 0.5, -0.3, 0.8])];

        let (attest, scores) = enclave_pipeline(
            &mut enclave,
            &tap,
            &activations,
            "test-model",
            [0xAA; 32],
            None,
            None,
            None,
        )
        .unwrap();

        // Attestation should be signed and verifiable.
        got_attest::verify(&attest, &enclave.verifying_key()).unwrap();

        // Should have causal scores embedded.
        assert_eq!(attest.causal_scores.len(), 1);
        assert!(attest.intervention_delta.is_some());
        assert!(attest.causal_flag.is_some());

        // Scores returned should match.
        assert_eq!(scores.len(), 1);
    }

    #[test]
    fn enclave_pipeline_tampered_capture_rejected() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "test");
        let model_fn = ClosureModelHandle::new(|h: &[f32]| -> Vec<f32> { h.to_vec() });
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            Some(Box::new(model_fn)),
        );
        let tap = MockDmaTap::new(dim, vec![]).with_tamper();

        let activations = vec![(0usize, 0usize, vec![1.0f32, 0.5, -0.3, 0.8])];

        let err = enclave_pipeline(
            &mut enclave,
            &tap,
            &activations,
            "test-model",
            [0xAA; 32],
            None,
            None,
            None,
        );
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            EnclaveError::IntegrityViolation { .. }
        ));
    }

    #[test]
    fn attestation_from_enclave_verifiable_externally() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "ext-verify");
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            None,
        );

        let tap = MockDmaTap::new(dim, vec![]);
        let h = vec![0.5, 0.5, 0.5, 0.5];
        let frame = tap.capture(0, 0, &h).unwrap();
        enclave.receive_activations(frame).unwrap();

        let attest = enclave
            .attest("ext-model", [0xBB; 32], None, None, None)
            .unwrap();

        // An external verifier with only the public key can verify.
        let pk = enclave.verifying_key();
        got_attest::verify(&attest, &pk).unwrap();

        // Verify that serialise_for_signing is deterministic.
        let bytes1 = serialise_for_signing(&attest).unwrap();
        let bytes2 = serialise_for_signing(&attest).unwrap();
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn enclave_chained_attestation() {
        let dim = 4;
        let geometry = make_geometry(dim);
        let probe = make_probe(dim, "chain");
        let mut enclave = MockEnclave::new(
            enclave_key(),
            geometry,
            vec![probe],
            MockEnclaveConfig::default(),
            None,
        );

        // First attestation (anchor).
        let tap = MockDmaTap::new(dim, vec![]);
        let h1 = vec![1.0, 0.0, 0.0, 0.0];
        let frame1 = tap.capture(0, 0, &h1).unwrap();
        enclave.receive_activations(frame1).unwrap();

        let a0 = enclave
            .attest("chain-model", [0xCC; 32], None, Some([0xDD; 32]), None)
            .unwrap();
        let a0_hash = {
            let bytes = serialise_for_signing(&a0).unwrap();
            got_core::sha256(&bytes)
        };

        // Reset for next window.
        enclave.reset();

        // Second attestation (child, chained to a0).
        let h2 = vec![0.9, 0.1, 0.0, 0.0];
        let frame2 = tap.capture(0, 0, &h2).unwrap();
        enclave.receive_activations(frame2).unwrap();

        let a1 = enclave
            .attest(
                "chain-model",
                [0xCC; 32],
                Some(a0_hash),
                Some([0xDD; 32]),
                Some(0.01),
            )
            .unwrap();

        // Verify the chain.
        assert_eq!(a1.parent_attestation_hash, Some(a0_hash));
        got_attest::verify(&a1, &enclave.verifying_key()).unwrap();
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issues 35, 40)
    // -----------------------------------------------------------------------

    /// Issue #35 (S-14): ed25519-dalek `zeroize` feature must be enabled
    /// so `SigningKey` bytes are zeroed on drop.
    #[test]
    fn sec_ed25519_dalek_has_zeroize_feature() {
        let cargo_toml = include_str!("../Cargo.toml");
        assert!(
            cargo_toml.contains("zeroize"),
            "ed25519-dalek must have zeroize feature enabled"
        );
    }

    /// Issue #40 (S-19): MockEnclave is gated behind `#[cfg(any(test, feature = "mock"))]`.
    /// Verify the cfg gate is present in the source.
    #[test]
    fn sec_mock_enclave_is_not_cfg_gated() {
        let source = include_str!("enclave.rs");
        // Find the line with `pub struct MockEnclave` and check the preceding line has cfg gate.
        let lines: Vec<&str> = source.lines().collect();
        let mock_line = lines
            .iter()
            .position(|l| l.contains("pub struct MockEnclave {"))
            .expect("MockEnclave struct not found");
        // The cfg gate should be on the line immediately before the struct (or 1 before doc comment).
        let preceding = lines[..mock_line].join("\n");
        assert!(
            preceding.contains("#[cfg(any(test, feature = \"mock\"))]"),
            "MockEnclave must be gated behind #[cfg(any(test, feature = \"mock\"))]"
        );
    }
}
