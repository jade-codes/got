// ---------------------------------------------------------------------------
// ProxyConfig — all tuneable thresholds for proxy value monitoring.
// ---------------------------------------------------------------------------

/// Configuration for proxy session behaviour.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    // --- Deviation detection thresholds ---
    /// Z-score significance threshold for term-level shift (Signal 1).
    /// Default: 2.5 standard deviations.
    pub term_zscore_threshold: f32,

    /// Cosine drift threshold for profile-level shift (Signal 2).
    /// Not used as a hard gate — the raw drift feeds into the combined score.
    pub profile_drift_warning: f32,

    /// Pairwise relationship disruption threshold (Signal 3).
    /// A pair is "disrupted" when its causal cosine moves by more than this
    /// many baseline standard deviations from the baseline mean.
    pub pairwise_disruption_sigmas: f32,

    // --- Combined score weights ---
    /// Weight for term-level z-score signal.
    pub weight_term: f32,
    /// Weight for profile cosine drift signal.
    pub weight_profile: f32,
    /// Weight for pairwise relationship disruption signal.
    pub weight_pairwise: f32,

    // --- Verdict thresholds ---
    /// Combined score below this → WithinBaseline.
    pub threshold_within: f32,
    /// Combined score at or above this → Deviated (between → Drifting).
    pub threshold_deviated: f32,

    // --- Observation requirements ---
    /// Minimum observations before deviation detection activates.
    pub min_observations_for_detection: u64,

    // --- Value detection ---
    /// Minimum z-score for a value term to be "detected" in a message.
    pub value_detection_threshold: f32,
    /// Maximum number of value terms to detect per observation.
    pub max_values_per_observation: usize,

    // --- EWMA ---
    /// EWMA decay factor for recency-weighted term profiles.
    /// α ∈ (0, 1]; higher = more weight on recent observations.
    pub ewma_alpha: f32,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            term_zscore_threshold: 2.5,
            profile_drift_warning: 0.3,
            pairwise_disruption_sigmas: 2.5,

            weight_term: 0.4,
            weight_profile: 0.3,
            weight_pairwise: 0.3,

            threshold_within: 0.3,
            threshold_deviated: 0.6,

            min_observations_for_detection: 20,

            value_detection_threshold: 0.0,
            max_values_per_observation: 10,

            ewma_alpha: 0.1,
        }
    }
}
