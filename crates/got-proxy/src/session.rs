// ---------------------------------------------------------------------------
// ProxySession — lifecycle management for proxy value monitoring.
//
// Manages: new() → observe() → detect() → snapshot_and_attest() → resume()
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use ed25519_dalek::SigningKey;
use got_core::geometry::CausalGeometry;
use got_incoherence::coherence::causal_cosine;
use got_incoherence::embeddings::EmbeddingSource;

use got_core::manifold::{DensityReading, CurvatureReading, ManifoldConfig, ValueManifold};

use crate::attestation::{
    sign_attestation, AttestationSummary, AttestationType, BehavioralAttestation,
    BEHAVIORAL_SCHEMA_VERSION,
};
use crate::config::ProxyConfig;
use crate::deviation::{detect_deviation, DeviationReport};
use crate::store::ValueSpaceStore;
use crate::value_space::BehavioralValueSpace;
use crate::ProxyError;

/// A detected value from an observation.
#[derive(Debug, Clone)]
pub struct DetectedValue {
    pub term: String,
    pub score: f64,
}

/// Result of observing one output from the closed-source model.
#[derive(Debug, Clone)]
pub struct ObservationResult {
    /// Values detected in this observation.
    pub detected_values: Vec<DetectedValue>,
    /// Current deviation report (None if baseline insufficient or non-model speaker).
    pub deviation: Option<DeviationReport>,
    /// Current observation count (model observations only).
    pub observation_count: u64,
    /// The speaker who produced this observation.
    pub speaker: String,
}

/// Proxy session status summary.
#[derive(Debug, Clone)]
pub struct SessionStatus {
    pub session_id: String,
    pub target_model_id: String,
    pub observation_count: u64,
    pub value_space_version: u64,
    pub top_values: Vec<(String, f64)>,
    pub latest_deviation: Option<DeviationReport>,
    pub attestation_count: u64,
}

/// ProxySession manages the full lifecycle of proxy value monitoring.
pub struct ProxySession<S: ValueSpaceStore, E: EmbeddingSource> {
    /// Unique session identifier.
    pub session_id: String,
    /// Target model being monitored.
    pub target_model_id: String,
    /// Ed25519 signing key for attestations.
    signing_key: SigningKey,
    /// Reference geometry (Φ = U^T U) from the open-source reference model.
    geometry: CausalGeometry,
    /// Embedding source for value term lookup and enumeration.
    embedding_source: E,
    /// The evolving value space.
    value_space: BehavioralValueSpace,
    /// Configuration.
    config: ProxyConfig,
    /// Persistent store.
    store: S,
    /// Monotonic attestation sequence number.
    sequence_number: u64,
    /// Hash of the most recent attestation (for chaining).
    last_attestation_hash: Option<[u8; 32]>,
    /// Most recent deviation report.
    latest_deviation: Option<DeviationReport>,
    /// Cumulative profile drift.
    cumulative_drift: f64,
    /// Deviation history.
    deviation_history: Vec<DeviationReport>,
    /// Stored activation vectors for manifold analysis (model speaker only).
    activation_history: Vec<Vec<f32>>,
    /// Per-term log-densities from the most recent snapshot. Empty before first snapshot.
    latest_term_densities: HashMap<String, f32>,
    /// Lightweight user context: which values has the user expressed, and how strongly.
    /// Not used for deviation detection or trust — purely descriptive context.
    user_value_profile: HashMap<String, f64>,
}

impl<S: ValueSpaceStore, E: EmbeddingSource> ProxySession<S, E> {
    /// Create a new proxy session.
    pub fn new(
        session_id: String,
        target_model_id: String,
        signing_key: SigningKey,
        geometry: CausalGeometry,
        embedding_source: E,
        config: ProxyConfig,
        store: S,
    ) -> Result<Self, ProxyError> {
        let geometry_hash = geometry.geometry_hash();
        let value_space = BehavioralValueSpace::new(geometry_hash);

        Ok(Self {
            session_id,
            target_model_id,
            signing_key,
            geometry,
            embedding_source,
            value_space,
            config,
            store,
            sequence_number: 0,
            last_attestation_hash: None,
            latest_deviation: None,
            cumulative_drift: 0.0,
            deviation_history: Vec::new(),
            activation_history: Vec::new(),
            latest_term_densities: HashMap::new(),
            user_value_profile: HashMap::new(),
        })
    }

    /// Observe a single message from the conversation.
    ///
    /// `speaker` identifies who produced the message:
    /// - `"assistant"` (or any model ID): full pipeline — value space, deviation,
    ///   manifold tracking, activation history. This is the monitored subject.
    /// - Any other speaker (e.g. `"user"`): lightweight value detection only.
    ///   Accumulates a context profile of what the user is expressing, but no
    ///   deviation baseline, no trust computation, no contradiction flagging.
    pub fn observe(
        &mut self,
        output_embedding: &[f32],
        speaker: &str,
    ) -> Result<ObservationResult, ProxyError> {
        // Step 1: Detect values (same for all speakers)
        let (detected, scores) = self.detect_values(output_embedding);

        // Step 2: Speaker-dependent processing
        let is_model = speaker == "assistant" || speaker == &self.target_model_id;

        if is_model {
            // Model speaker: full pipeline
            self.activation_history.push(output_embedding.to_vec());

            self.value_space
                .update_terms(&scores, self.config.ewma_alpha as f64);

            let pairwise_cosines = self.compute_pairwise(&scores);
            self.value_space.update_pairwise(&pairwise_cosines);

            let manifold_density_score = self.compute_manifold_density_signal();

            let deviation_report = detect_deviation(
                &scores,
                &pairwise_cosines,
                &self.value_space,
                &self.config,
                manifold_density_score,
            );

            let deviation = if deviation_report.baseline_sufficient {
                self.cumulative_drift += deviation_report.profile_drift;
                self.latest_deviation = Some(deviation_report.clone());
                self.deviation_history.push(deviation_report.clone());
                Some(deviation_report)
            } else {
                None
            };

            Ok(ObservationResult {
                detected_values: detected,
                deviation,
                observation_count: self.value_space.observation_count,
                speaker: speaker.to_string(),
            })
        } else {
            // Non-model speaker: lightweight context tracking
            for (term, &score) in &scores {
                let entry = self.user_value_profile.entry(term.clone()).or_insert(0.0);
                // EWMA update for the user profile
                *entry = *entry * (1.0 - self.config.ewma_alpha as f64)
                    + score * self.config.ewma_alpha as f64;
            }

            Ok(ObservationResult {
                detected_values: detected,
                deviation: None,
                observation_count: self.value_space.observation_count,
                speaker: speaker.to_string(),
            })
        }
    }

    /// Detect values in an embedding: project against all terms, z-score, threshold.
    fn detect_values(&self, embedding: &[f32]) -> (Vec<DetectedValue>, HashMap<String, f64>) {
        let mut detected = Vec::new();
        let mut scores = HashMap::new();

        let mut raw_scores: Vec<(String, f64)> = Vec::new();
        for term in self.embedding_source.available_terms() {
            if let Some(term_emb) = self.embedding_source.embed(&term) {
                match causal_cosine(embedding, &term_emb, &self.geometry) {
                    Ok(cos) => raw_scores.push((term, cos as f64)),
                    Err(_) => continue,
                }
            }
        }

        if !raw_scores.is_empty() {
            let mean: f64 =
                raw_scores.iter().map(|(_, s)| s).sum::<f64>() / raw_scores.len() as f64;
            let variance: f64 = raw_scores
                .iter()
                .map(|(_, s)| (s - mean).powi(2))
                .sum::<f64>()
                / raw_scores.len() as f64;
            let stddev = variance.sqrt();

            let mut z_scored: Vec<(String, f64)> = if stddev > f64::EPSILON {
                raw_scores
                    .iter()
                    .map(|(t, s)| (t.clone(), (s - mean) / stddev))
                    .collect()
            } else {
                raw_scores
                    .iter()
                    .map(|(t, _)| (t.clone(), 0.0))
                    .collect()
            };

            z_scored
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (term, zscore) in z_scored
                .into_iter()
                .filter(|(_, z)| *z > self.config.value_detection_threshold as f64)
                .take(self.config.max_values_per_observation)
            {
                detected.push(DetectedValue {
                    term: term.clone(),
                    score: zscore,
                });
                scores.insert(term, zscore);
            }
        }

        (detected, scores)
    }

    /// Compute pairwise causal cosines for a set of detected terms.
    fn compute_pairwise(
        &self,
        scores: &HashMap<String, f64>,
    ) -> HashMap<(String, String), f64> {
        let terms: Vec<String> = scores.keys().cloned().collect();
        let mut pairwise = HashMap::new();
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                if let (Some(emb_a), Some(emb_b)) = (
                    self.embedding_source.embed(&terms[i]),
                    self.embedding_source.embed(&terms[j]),
                ) {
                    if let Ok(cos) = causal_cosine(&emb_a, &emb_b, &self.geometry) {
                        pairwise.insert((terms[i].clone(), terms[j].clone()), cos as f64);
                    }
                }
            }
        }
        pairwise
    }

    /// Compute Signal 4: is the latest activation off-manifold?
    ///
    /// Uses a sliding window of recent activations (capped at 4× min_activations)
    /// to keep cost bounded. Returns 0.0 if manifold analysis is disabled or
    /// there are insufficient activations.
    fn compute_manifold_density_signal(&self) -> f64 {
        if self.config.weight_manifold <= 0.0 {
            return 0.0;
        }
        let k = self.config.manifold_k;
        let min_n = self.config.min_activations_for_manifold.max(k + 1);
        if self.activation_history.len() < min_n {
            return 0.0;
        }

        // Use a sliding window: at most 4× min_activations recent points
        let history_len = self.activation_history.len();
        let window_cap = min_n * 4;
        let window_start = if history_len > window_cap + 1 {
            history_len - window_cap - 1
        } else {
            0
        };

        // Build manifold from the window, excluding the latest observation
        let manifold_points: Vec<Vec<f32>> =
            self.activation_history[window_start..history_len - 1].to_vec();
        if manifold_points.len() < k + 1 {
            return 0.0;
        }
        let manifold = match ValueManifold::new(
            manifold_points,
            &self.geometry,
            ManifoldConfig { k },
        ) {
            Ok(m) => m,
            Err(_) => return 0.0,
        };

        let d_eff = match manifold.density_map() {
            Ok(dr) => dr.mean_intrinsic_dim,
            Err(_) => return 0.0,
        };

        let latest = &self.activation_history[history_len - 1];
        match manifold.query_log_density(latest, &self.geometry, d_eff) {
            Ok(Some(ld)) if ld < self.config.manifold_density_threshold as f32 => 1.0,
            Ok(Some(_)) => 0.0,
            _ => 0.0,
        }
    }

    /// Build manifold readings from the activation history.
    ///
    /// Returns (density, curvature, per-term densities). All None/empty if insufficient data.
    fn build_manifold_readings(
        &self,
    ) -> (
        Option<DensityReading>,
        Option<CurvatureReading>,
        HashMap<String, f32>,
    ) {
        let k = self.config.manifold_k;
        let min_n = self.config.min_activations_for_manifold.max(k + 1);
        if self.activation_history.len() < min_n {
            return (None, None, HashMap::new());
        }

        let manifold = match ValueManifold::new(
            self.activation_history.clone(),
            &self.geometry,
            ManifoldConfig { k },
        ) {
            Ok(m) => m,
            Err(_) => return (None, None, HashMap::new()),
        };

        let density = manifold.density_map().ok();
        let curvature = manifold.curvature_map(None).ok();

        // Query each known term's density on the activation manifold
        let d_eff = density.as_ref().map(|d| d.mean_intrinsic_dim).unwrap_or(2.0);
        let mut term_densities = HashMap::new();
        for term in self.embedding_source.available_terms() {
            if let Some(emb) = self.embedding_source.embed(&term) {
                if let Ok(Some(ld)) = manifold.query_log_density(&emb, &self.geometry, d_eff) {
                    term_densities.insert(term, ld);
                }
            }
        }

        (density, curvature, term_densities)
    }

    /// Take a snapshot of the current value space and produce a signed attestation.
    pub fn snapshot_and_attest(
        &mut self,
        attestation_type: AttestationType,
    ) -> Result<BehavioralAttestation, ProxyError> {
        // Store the snapshot
        self.store.store_snapshot(&self.value_space)?;

        // Build top values from EWMA
        let mut top_values: Vec<(String, f64)> = self
            .value_space
            .term_profiles
            .iter()
            .map(|(t, p)| (t.clone(), p.ewma))
            .collect();
        top_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_values.truncate(10);

        // Compute coherence score from current pairwise baselines
        let coherence_score = if self.value_space.pairwise_baselines.is_empty() {
            1.0
        } else {
            let total: f64 = self
                .value_space
                .pairwise_baselines
                .iter()
                .map(|pb| pb.mean.max(0.0))
                .sum();
            (total / self.value_space.pairwise_baselines.len() as f64).clamp(0.0, 1.0)
        };

        self.sequence_number += 1;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let deviation = match attestation_type {
            AttestationType::Alert => self.latest_deviation.clone(),
            _ => None,
        };

        // Build manifold readings from activation history
        let (density_reading, curvature_reading, term_densities) =
            self.build_manifold_readings();
        self.latest_term_densities = term_densities;

        let attestation = BehavioralAttestation {
            schema_version: BEHAVIORAL_SCHEMA_VERSION.into(),
            target_model_id: self.target_model_id.clone(),
            reference_geometry_hash: self.value_space.reference_geometry_hash,
            attestation_type,
            observation_count: self.value_space.observation_count,
            sequence_number: self.sequence_number,
            timestamp,
            value_space_hash: self.value_space.hash(),
            parent_hash: self.last_attestation_hash,
            summary: AttestationSummary {
                top_values,
                coherence_score,
                cumulative_drift: self.cumulative_drift,
            },
            deviation,
            density_reading,
            curvature_reading,
            signature: [0; 64],
        };

        let signed = sign_attestation(attestation, &self.signing_key)?;
        let hash = crate::attestation::attestation_hash(&signed);
        self.store.store_attestation(&signed)?;
        self.last_attestation_hash = Some(hash);

        // Update value space parent hash
        self.value_space.parent_hash = Some(self.value_space.hash());

        Ok(signed)
    }

    /// Get current session status.
    pub fn status(&self) -> SessionStatus {
        let mut top_values: Vec<(String, f64)> = self
            .value_space
            .term_profiles
            .iter()
            .map(|(t, p)| (t.clone(), p.ewma))
            .collect();
        top_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_values.truncate(10);

        SessionStatus {
            session_id: self.session_id.clone(),
            target_model_id: self.target_model_id.clone(),
            observation_count: self.value_space.observation_count,
            value_space_version: self.value_space.version,
            top_values,
            latest_deviation: self.latest_deviation.clone(),
            attestation_count: self.sequence_number,
        }
    }

    /// Get deviation history.
    pub fn deviation_history(&self) -> &[DeviationReport] {
        &self.deviation_history
    }

    /// Access the current value space (read-only).
    pub fn value_space(&self) -> &BehavioralValueSpace {
        &self.value_space
    }

    /// Access the store (read-only).
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Per-term log-densities from the most recent manifold computation.
    pub fn term_densities(&self) -> &HashMap<String, f32> {
        &self.latest_term_densities
    }

    /// The user's value context profile (EWMA of detected value scores).
    /// Descriptive only — not used for trust or deviation computation.
    pub fn user_value_profile(&self) -> &HashMap<String, f64> {
        &self.user_value_profile
    }

    /// Produce a signed attestation that includes manifold readings.
    ///
    /// Convenience wrapper: creates a Snapshot attestation and returns it
    /// along with per-term densities (for visualisation).
    pub fn attest_manifold(
        &mut self,
    ) -> Result<(BehavioralAttestation, HashMap<String, f32>), ProxyError> {
        let attestation = self.snapshot_and_attest(AttestationType::Snapshot)?;
        let term_densities = self.latest_term_densities.clone();
        Ok((attestation, term_densities))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryValueSpaceStore;
    use got_core::geometry::CausalGeometry;
    use got_incoherence::embeddings::PrecomputedEmbeddings;

    fn make_test_session() -> ProxySession<MemoryValueSpaceStore, PrecomputedEmbeddings> {
        let dim = 4;
        // Simple identity geometry
        let mut gram = vec![0.0f32; dim * dim];
        for i in 0..dim {
            gram[i * dim + i] = 1.0;
        }
        let geometry = CausalGeometry::from_raw_gram(gram, dim).unwrap();

        // Simple term embeddings (unit-ish vectors)
        let mut embeddings = HashMap::new();
        embeddings.insert("honesty".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        embeddings.insert("courage".to_string(), vec![0.0, 1.0, 0.0, 0.0]);
        embeddings.insert("fairness".to_string(), vec![0.0, 0.0, 1.0, 0.0]);

        let source = PrecomputedEmbeddings::new(embeddings).unwrap();
        let sk = SigningKey::from_bytes(&[42u8; 32]);

        ProxySession::new(
            "test-session".into(),
            "test-model".into(),
            sk,
            geometry,
            source,
            ProxyConfig::default(),
            MemoryValueSpaceStore::new(),
        )
        .unwrap()
    }

    #[test]
    fn observe_detects_values() {
        let mut session = make_test_session();
        // Embedding close to "honesty" direction
        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        let result = session.observe(&embedding, "assistant").unwrap();
        assert!(!result.detected_values.is_empty());
        assert_eq!(result.observation_count, 1);
    }

    #[test]
    fn full_lifecycle() {
        let mut session = make_test_session();

        // Build up baseline with 25 observations
        for i in 0..25 {
            let angle = (i as f32) * 0.1;
            let embedding = vec![angle.cos(), angle.sin(), 0.1, 0.0];
            session.observe(&embedding, "assistant").unwrap();
        }

        assert_eq!(session.value_space().observation_count, 25);

        // Snapshot and attest
        let att = session
            .snapshot_and_attest(AttestationType::Baseline)
            .unwrap();
        assert_eq!(att.schema_version, "B1");
        assert_eq!(att.observation_count, 25);
        assert_eq!(att.sequence_number, 1);
        assert!(att.parent_hash.is_none()); // First attestation

        // Status check
        let status = session.status();
        assert_eq!(status.observation_count, 25);
        assert_eq!(status.attestation_count, 1);
    }

    #[test]
    fn attestation_chaining() {
        let mut session = make_test_session();

        for _ in 0..5 {
            session.observe(&[0.5, 0.5, 0.0, 0.0], "assistant").unwrap();
        }

        let att1 = session
            .snapshot_and_attest(AttestationType::Baseline)
            .unwrap();
        assert!(att1.parent_hash.is_none());

        for _ in 0..5 {
            session.observe(&[0.3, 0.7, 0.0, 0.0], "assistant").unwrap();
        }

        let att2 = session
            .snapshot_and_attest(AttestationType::Snapshot)
            .unwrap();
        assert!(att2.parent_hash.is_some());
        assert_eq!(att2.sequence_number, 2);
    }
}
