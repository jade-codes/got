// ---------------------------------------------------------------------------
// BehavioralValueSpace — the evolving statistical profile of a closed-source
// model's value expression, built from observable outputs.
//
// Each value term has a running mean/variance (Welford's online algorithm)
// plus an EWMA for recency weighting.  Pairwise baselines track relationship
// structure between value terms over time.
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Per-value-term running statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermProfile {
    /// Number of observations that included this term.
    pub count: u64,
    /// Welford online mean of the term's causal cosine / z-score.
    pub mean: f64,
    /// Welford M2 accumulator (sum of squared deltas from the mean).
    pub m2: f64,
    /// EWMA (exponentially weighted moving average) of recent scores.
    pub ewma: f64,
}

impl TermProfile {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            ewma: 0.0,
        }
    }

    /// Welford online update with a new score observation.
    pub fn update(&mut self, score: f64, alpha: f64) {
        self.count += 1;
        let delta = score - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = score - self.mean;
        self.m2 += delta * delta2;

        // EWMA update
        if self.count == 1 {
            self.ewma = score;
        } else {
            self.ewma = alpha * score + (1.0 - alpha) * self.ewma;
        }
    }

    /// Population variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Standard deviation.
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Z-score of a given value relative to baseline.
    pub fn zscore(&self, value: f64) -> f64 {
        let sd = self.stddev();
        if sd < f64::EPSILON {
            return 0.0;
        }
        (value - self.mean) / sd
    }
}

/// Running statistics for the causal cosine between a pair of value terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseBaseline {
    pub term_a: String,
    pub term_b: String,
    /// Number of observations for this pair.
    pub count: u64,
    /// Welford online mean of pairwise causal cosine.
    pub mean: f64,
    /// Welford M2 accumulator.
    pub m2: f64,
}

impl PairwiseBaseline {
    pub fn new(term_a: String, term_b: String) -> Self {
        Self {
            term_a,
            term_b,
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Welford online update.
    pub fn update(&mut self, cosine: f64) {
        self.count += 1;
        let delta = cosine - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = cosine - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// The evolving statistical profile of a closed-source model's value expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralValueSpace {
    /// Per-value-term running statistics.
    pub term_profiles: HashMap<String, TermProfile>,
    /// Running mean/variance of causal cosines between value term pairs.
    pub pairwise_baselines: Vec<PairwiseBaseline>,
    /// Total number of observations processed.
    pub observation_count: u64,
    /// Monotonic version counter (incremented on each observation).
    pub version: u64,
    /// SHA-256 of the previous snapshot (None for first snapshot).
    pub parent_hash: Option<[u8; 32]>,
    /// SHA-256 of the reference geometry's Gram matrix.
    /// If the reference model changes, the value space must be re-baselined.
    pub reference_geometry_hash: [u8; 32],
}

impl BehavioralValueSpace {
    /// Create a new empty value space pinned to a reference geometry.
    pub fn new(reference_geometry_hash: [u8; 32]) -> Self {
        Self {
            term_profiles: HashMap::new(),
            pairwise_baselines: Vec::new(),
            observation_count: 0,
            version: 0,
            parent_hash: None,
            reference_geometry_hash,
        }
    }

    /// Update term profiles with a set of detected values from one observation.
    ///
    /// `detected` maps term name → score (causal cosine or z-score).
    pub fn update_terms(&mut self, detected: &HashMap<String, f64>, alpha: f64) {
        for (term, &score) in detected {
            self.term_profiles
                .entry(term.clone())
                .or_insert_with(TermProfile::new)
                .update(score, alpha);
        }
        self.observation_count += 1;
        self.version += 1;
    }

    /// Update pairwise baselines with current pairwise causal cosines.
    ///
    /// `pairwise_cosines` maps `(term_a, term_b)` → causal cosine.
    pub fn update_pairwise(&mut self, pairwise_cosines: &HashMap<(String, String), f64>) {
        for ((term_a, term_b), &cosine) in pairwise_cosines {
            if let Some(baseline) = self.pairwise_baselines.iter_mut().find(|b| {
                (&b.term_a == term_a && &b.term_b == term_b)
                    || (&b.term_a == term_b && &b.term_b == term_a)
            }) {
                baseline.update(cosine);
            } else {
                let mut baseline = PairwiseBaseline::new(term_a.clone(), term_b.clone());
                baseline.update(cosine);
                self.pairwise_baselines.push(baseline);
            }
        }
    }

    /// Build a profile vector from current EWMA values (for cosine drift detection).
    /// Returns `(terms, vector)` where terms[i] corresponds to vector[i].
    pub fn profile_vector(&self) -> (Vec<String>, Vec<f64>) {
        let mut terms: Vec<String> = self.term_profiles.keys().cloned().collect();
        terms.sort();
        let vector: Vec<f64> = terms.iter().map(|t| self.term_profiles[t].ewma).collect();
        (terms, vector)
    }

    /// Compute SHA-256 hash of the current value space state.
    pub fn hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        // Hash reference geometry binding
        hasher.update(self.reference_geometry_hash);

        // Hash observation count and version
        hasher.update(self.observation_count.to_le_bytes());
        hasher.update(self.version.to_le_bytes());

        // Hash term profiles in sorted order for determinism
        let mut terms: Vec<&String> = self.term_profiles.keys().collect();
        terms.sort();
        for term in &terms {
            let profile = &self.term_profiles[*term];
            hasher.update(term.as_bytes());
            hasher.update(profile.count.to_le_bytes());
            hasher.update(profile.mean.to_le_bytes());
            hasher.update(profile.m2.to_le_bytes());
            hasher.update(profile.ewma.to_le_bytes());
        }

        // Hash pairwise baselines (sorted by term pair for determinism)
        let mut sorted_pairs = self.pairwise_baselines.clone();
        sorted_pairs.sort_by(|a, b| (&a.term_a, &a.term_b).cmp(&(&b.term_a, &b.term_b)));
        for pair in &sorted_pairs {
            hasher.update(pair.term_a.as_bytes());
            hasher.update(pair.term_b.as_bytes());
            hasher.update(pair.count.to_le_bytes());
            hasher.update(pair.mean.to_le_bytes());
            hasher.update(pair.m2.to_le_bytes());
        }

        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_term_profile() {
        let mut tp = TermProfile::new();
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for v in &values {
            tp.update(*v, 0.1);
        }
        assert_eq!(tp.count, 8);
        let expected_mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!((tp.mean - expected_mean).abs() < 1e-10);
        // Population variance for this dataset = 4.0
        assert!((tp.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn welford_pairwise_baseline() {
        let mut pb = PairwiseBaseline::new("a".into(), "b".into());
        pb.update(0.5);
        pb.update(0.7);
        pb.update(0.3);
        assert_eq!(pb.count, 3);
        assert!((pb.mean - 0.5).abs() < 1e-10);
    }

    #[test]
    fn value_space_hash_determinism() {
        let gh = [0xAA; 32];
        let mut vs1 = BehavioralValueSpace::new(gh);
        let mut vs2 = BehavioralValueSpace::new(gh);

        let detected: HashMap<String, f64> =
            [("honesty".into(), 0.8), ("courage".into(), 0.5)]
                .into_iter()
                .collect();

        vs1.update_terms(&detected, 0.1);
        vs2.update_terms(&detected, 0.1);

        assert_eq!(vs1.hash(), vs2.hash());
    }

    #[test]
    fn profile_vector_sorted() {
        let mut vs = BehavioralValueSpace::new([0; 32]);
        let detected: HashMap<String, f64> =
            [("z_term".into(), 1.0), ("a_term".into(), 2.0)]
                .into_iter()
                .collect();
        vs.update_terms(&detected, 0.1);
        let (terms, _) = vs.profile_vector();
        assert_eq!(terms, vec!["a_term", "z_term"]);
    }
}
