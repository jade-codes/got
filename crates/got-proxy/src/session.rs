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
    /// Current deviation report (None if baseline insufficient).
    pub deviation: Option<DeviationReport>,
    /// Current observation count.
    pub observation_count: u64,
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
    /// Value term embeddings from the reference model.
    term_embeddings: HashMap<String, Vec<f32>>,
    /// Embedding source for value term lookup.
    #[allow(dead_code)]
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
}

impl<S: ValueSpaceStore, E: EmbeddingSource> ProxySession<S, E> {
    /// Create a new proxy session.
    pub fn new(
        session_id: String,
        target_model_id: String,
        signing_key: SigningKey,
        geometry: CausalGeometry,
        term_embeddings: HashMap<String, Vec<f32>>,
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
            term_embeddings,
            embedding_source,
            value_space,
            config,
            store,
            sequence_number: 0,
            last_attestation_hash: None,
            latest_deviation: None,
            cumulative_drift: 0.0,
            deviation_history: Vec::new(),
        })
    }

    /// Observe a single output from the closed-source model.
    ///
    /// `output_embedding` is the output text embedded through the reference
    /// model's geometry. The session detects values, updates the value space,
    /// and runs deviation detection.
    pub fn observe(&mut self, output_embedding: &[f32]) -> Result<ObservationResult, ProxyError> {
        // Detect values: project output against each term embedding using causal cosine
        let mut detected = Vec::new();
        let mut scores = HashMap::new();

        // Compute raw projections
        let mut raw_scores: Vec<(String, f64)> = Vec::new();
        for (term, term_emb) in &self.term_embeddings {
            match causal_cosine(output_embedding, term_emb, &self.geometry) {
                Ok(cos) => raw_scores.push((term.clone(), cos as f64)),
                Err(_) => continue,
            }
        }

        // Z-score the raw projections
        if !raw_scores.is_empty() {
            let mean: f64 = raw_scores.iter().map(|(_, s)| s).sum::<f64>() / raw_scores.len() as f64;
            let variance: f64 = raw_scores.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f64>()
                / raw_scores.len() as f64;
            let stddev = variance.sqrt();

            let mut z_scored: Vec<(String, f64)> = if stddev > f64::EPSILON {
                raw_scores
                    .iter()
                    .map(|(t, s)| (t.clone(), (s - mean) / stddev))
                    .collect()
            } else {
                raw_scores.iter().map(|(t, _)| (t.clone(), 0.0)).collect()
            };

            // Sort by z-score descending, take top N above threshold
            z_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

        // Update value space
        self.value_space
            .update_terms(&scores, self.config.ewma_alpha as f64);

        // Compute pairwise causal cosines for detected terms
        let detected_terms: Vec<String> = scores.keys().cloned().collect();
        let mut pairwise_cosines = HashMap::new();
        for i in 0..detected_terms.len() {
            for j in (i + 1)..detected_terms.len() {
                let term_a = &detected_terms[i];
                let term_b = &detected_terms[j];
                if let (Some(emb_a), Some(emb_b)) = (
                    self.term_embeddings.get(term_a),
                    self.term_embeddings.get(term_b),
                ) {
                    if let Ok(cos) = causal_cosine(emb_a, emb_b, &self.geometry) {
                        pairwise_cosines
                            .insert((term_a.clone(), term_b.clone()), cos as f64);
                    }
                }
            }
        }
        self.value_space.update_pairwise(&pairwise_cosines);

        // Run deviation detection
        let deviation_report =
            detect_deviation(&scores, &pairwise_cosines, &self.value_space, &self.config);

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
        })
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

        let source = PrecomputedEmbeddings::new(embeddings.clone()).unwrap();
        let sk = SigningKey::from_bytes(&[42u8; 32]);

        ProxySession::new(
            "test-session".into(),
            "test-model".into(),
            sk,
            geometry,
            embeddings,
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
        let result = session.observe(&embedding).unwrap();
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
            session.observe(&embedding).unwrap();
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
            session.observe(&[0.5, 0.5, 0.0, 0.0]).unwrap();
        }

        let att1 = session
            .snapshot_and_attest(AttestationType::Baseline)
            .unwrap();
        assert!(att1.parent_hash.is_none());

        for _ in 0..5 {
            session.observe(&[0.3, 0.7, 0.0, 0.0]).unwrap();
        }

        let att2 = session
            .snapshot_and_attest(AttestationType::Snapshot)
            .unwrap();
        assert!(att2.parent_hash.is_some());
        assert_eq!(att2.sequence_number, 2);
    }
}
