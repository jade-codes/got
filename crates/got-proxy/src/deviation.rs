// ---------------------------------------------------------------------------
// 3-Signal Deviation Detection
//
// Detects when a closed-source model's expressed values deviate from its
// established baseline using three complementary signals:
//   Signal 1: Term-level z-score shift
//   Signal 2: Profile cosine drift
//   Signal 3: Pairwise relationship disruption
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::ProxyConfig;
use crate::value_space::BehavioralValueSpace;

/// Deviation verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviationVerdict {
    /// Combined score < threshold_within (default 0.3).
    WithinBaseline,
    /// threshold_within ≤ score < threshold_deviated.
    Drifting,
    /// score ≥ threshold_deviated (default 0.6).
    Deviated,
}

/// A single flagged term that exceeded the z-score threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlaggedTerm {
    pub term: String,
    pub current_score: f64,
    pub baseline_mean: f64,
    pub baseline_stddev: f64,
    pub significance: f64,
}

/// A pairwise relationship that was disrupted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisruptedPair {
    pub term_a: String,
    pub term_b: String,
    pub current_cosine: f64,
    pub baseline_mean: f64,
    pub baseline_stddev: f64,
    pub shift_sigmas: f64,
}

/// Full deviation report for one observation window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationReport {
    /// Signal 1: normalised term-level z-score shift score ∈ [0, 1].
    pub term_score: f64,
    /// Signal 2: profile cosine drift ∈ [0, 2] (1 - cosine; 0 = identical).
    pub profile_drift: f64,
    /// Signal 3: normalised pairwise relationship disruption score ∈ [0, 1].
    pub relationship_score: f64,
    /// Signal 4: manifold density anomaly score ∈ [0, 1].
    /// Fraction of recent observations that fall off-manifold.
    /// 0 when manifold analysis is disabled or insufficient data.
    #[serde(default)]
    pub manifold_density_score: f64,
    /// Weighted combination of all signals.
    pub combined_score: f64,
    /// Verdict based on combined score.
    pub verdict: DeviationVerdict,
    /// Terms that exceeded the z-score threshold.
    pub flagged_terms: Vec<FlaggedTerm>,
    /// Pairwise relationships that were disrupted.
    pub disrupted_pairs: Vec<DisruptedPair>,
    /// Whether the baseline had enough observations for reliable detection.
    pub baseline_sufficient: bool,
}

/// Compute Signal 1: term-level z-score shift.
///
/// For each detected term in the current observation, compute the z-score
/// relative to the baseline. Return the fraction of terms that exceed the
/// significance threshold.
fn signal_term_zscore(
    current_scores: &HashMap<String, f64>,
    baseline: &BehavioralValueSpace,
    config: &ProxyConfig,
) -> (f64, Vec<FlaggedTerm>) {
    if current_scores.is_empty() {
        return (0.0, Vec::new());
    }

    let mut flagged = Vec::new();
    let mut significant_count = 0usize;
    let total = current_scores.len();

    for (term, &score) in current_scores {
        if let Some(profile) = baseline.term_profiles.get(term) {
            let significance = profile.zscore(score).abs();
            if significance > config.term_zscore_threshold as f64 {
                significant_count += 1;
                flagged.push(FlaggedTerm {
                    term: term.clone(),
                    current_score: score,
                    baseline_mean: profile.mean,
                    baseline_stddev: profile.stddev(),
                    significance,
                });
            }
        }
        // New terms not in baseline are not flagged (they haven't deviated
        // from anything — they're novel).
    }

    let score = significant_count as f64 / total as f64;
    (score.min(1.0), flagged)
}

/// Compute Signal 2: profile cosine drift.
///
/// Builds a profile vector from EWMA values and compares against the
/// baseline profile vector using cosine similarity.
/// Returns `1.0 - cosine(current, baseline)` ∈ [0, 2].
fn signal_profile_drift(
    current_scores: &HashMap<String, f64>,
    baseline: &BehavioralValueSpace,
) -> f64 {
    let (baseline_terms, baseline_vec) = baseline.profile_vector();
    if baseline_terms.is_empty() {
        return 0.0;
    }

    // Build current vector aligned to baseline terms
    let current_vec: Vec<f64> = baseline_terms
        .iter()
        .map(|t| current_scores.get(t).copied().unwrap_or(0.0))
        .collect();

    let dot: f64 = current_vec.iter().zip(&baseline_vec).map(|(a, b)| a * b).sum();
    let norm_c: f64 = current_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = baseline_vec.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_c < f64::EPSILON || norm_b < f64::EPSILON {
        return 0.0;
    }

    let cosine = (dot / (norm_c * norm_b)).clamp(-1.0, 1.0);
    1.0 - cosine
}

/// Compute Signal 3: pairwise relationship disruption.
///
/// For each pairwise baseline, compare the current causal cosine against
/// the baseline mean. Flag when the shift exceeds the configured number
/// of standard deviations.
fn signal_pairwise_disruption(
    current_pairwise: &HashMap<(String, String), f64>,
    baseline: &BehavioralValueSpace,
    config: &ProxyConfig,
) -> (f64, Vec<DisruptedPair>) {
    if baseline.pairwise_baselines.is_empty() {
        return (0.0, Vec::new());
    }

    let mut disrupted = Vec::new();
    let mut disrupted_count = 0usize;
    let total = baseline.pairwise_baselines.len();

    for pb in &baseline.pairwise_baselines {
        // Look up current cosine for this pair (check both orderings)
        let current = current_pairwise
            .get(&(pb.term_a.clone(), pb.term_b.clone()))
            .or_else(|| current_pairwise.get(&(pb.term_b.clone(), pb.term_a.clone())));

        if let Some(&current_cosine) = current {
            let sd = pb.stddev();
            if sd < f64::EPSILON {
                continue;
            }
            let shift = ((current_cosine - pb.mean) / sd).abs();
            if shift > config.pairwise_disruption_sigmas as f64 {
                disrupted_count += 1;
                disrupted.push(DisruptedPair {
                    term_a: pb.term_a.clone(),
                    term_b: pb.term_b.clone(),
                    current_cosine,
                    baseline_mean: pb.mean,
                    baseline_stddev: sd,
                    shift_sigmas: shift,
                });
            }
        }
    }

    let score = disrupted_count as f64 / total as f64;
    (score.min(1.0), disrupted)
}

/// Run the full deviation detection algorithm (3 signals + optional manifold signal).
///
/// `manifold_density_score`: fraction of recent observations that are off-manifold.
/// Pass 0.0 when manifold analysis is not available.
pub fn detect_deviation(
    current_scores: &HashMap<String, f64>,
    current_pairwise: &HashMap<(String, String), f64>,
    baseline: &BehavioralValueSpace,
    config: &ProxyConfig,
    manifold_density_score: f64,
) -> DeviationReport {
    let baseline_sufficient =
        baseline.observation_count >= config.min_observations_for_detection;

    if !baseline_sufficient {
        return DeviationReport {
            term_score: 0.0,
            profile_drift: 0.0,
            relationship_score: 0.0,
            manifold_density_score: 0.0,
            combined_score: 0.0,
            verdict: DeviationVerdict::WithinBaseline,
            flagged_terms: Vec::new(),
            disrupted_pairs: Vec::new(),
            baseline_sufficient: false,
        };
    }

    let (term_score, flagged_terms) = signal_term_zscore(current_scores, baseline, config);
    let profile_drift = signal_profile_drift(current_scores, baseline);
    // Clamp profile_drift to [0, 1] for weighting (raw is [0, 2])
    let profile_drift_normalised = (profile_drift / 2.0).min(1.0);
    let (relationship_score, disrupted_pairs) =
        signal_pairwise_disruption(current_pairwise, baseline, config);

    let combined_score = config.weight_term as f64 * term_score
        + config.weight_profile as f64 * profile_drift_normalised
        + config.weight_pairwise as f64 * relationship_score
        + config.weight_manifold as f64 * manifold_density_score;

    let verdict = if combined_score < config.threshold_within as f64 {
        DeviationVerdict::WithinBaseline
    } else if combined_score < config.threshold_deviated as f64 {
        DeviationVerdict::Drifting
    } else {
        DeviationVerdict::Deviated
    };

    DeviationReport {
        term_score,
        profile_drift,
        relationship_score,
        manifold_density_score,
        combined_score,
        verdict,
        flagged_terms,
        disrupted_pairs,
        baseline_sufficient: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_space::{BehavioralValueSpace, TermProfile, PairwiseBaseline};

    fn make_baseline(n: u64) -> BehavioralValueSpace {
        let mut vs = BehavioralValueSpace::new([0; 32]);
        vs.observation_count = n;

        // Honesty: mean=0.7, some variance
        let mut tp = TermProfile::new();
        for v in &[0.6, 0.7, 0.8, 0.65, 0.75, 0.7, 0.72, 0.68, 0.71, 0.69,
                    0.6, 0.7, 0.8, 0.65, 0.75, 0.7, 0.72, 0.68, 0.71, 0.69] {
            tp.update(*v, 0.1);
        }
        vs.term_profiles.insert("honesty".into(), tp);

        // Courage: mean=0.5
        let mut tp2 = TermProfile::new();
        for v in &[0.4, 0.5, 0.6, 0.45, 0.55, 0.5, 0.52, 0.48, 0.51, 0.49,
                    0.4, 0.5, 0.6, 0.45, 0.55, 0.5, 0.52, 0.48, 0.51, 0.49] {
            tp2.update(*v, 0.1);
        }
        vs.term_profiles.insert("courage".into(), tp2);

        // Pairwise: honesty↔courage baseline ~0.3
        let mut pb = PairwiseBaseline::new("honesty".into(), "courage".into());
        for v in &[0.28, 0.32, 0.30, 0.29, 0.31, 0.30, 0.28, 0.32, 0.30, 0.29,
                    0.28, 0.32, 0.30, 0.29, 0.31, 0.30, 0.28, 0.32, 0.30, 0.29] {
            pb.update(*v);
        }
        vs.pairwise_baselines.push(pb);

        vs
    }

    #[test]
    fn within_baseline() {
        let baseline = make_baseline(20);
        let config = ProxyConfig::default();

        let current: HashMap<String, f64> =
            [("honesty".into(), 0.71), ("courage".into(), 0.50)]
                .into_iter()
                .collect();
        let pairwise: HashMap<(String, String), f64> =
            [(("honesty".into(), "courage".into()), 0.30)]
                .into_iter()
                .collect();

        let report = detect_deviation(&current, &pairwise, &baseline, &config, 0.0);
        assert!(report.baseline_sufficient);
        assert_eq!(report.verdict, DeviationVerdict::WithinBaseline);
        assert!(report.flagged_terms.is_empty());
        assert!(report.disrupted_pairs.is_empty());
    }

    #[test]
    fn insufficient_baseline() {
        let baseline = make_baseline(5);
        let config = ProxyConfig::default();

        let current: HashMap<String, f64> =
            [("honesty".into(), 0.0)].into_iter().collect();
        let pairwise = HashMap::new();

        let report = detect_deviation(&current, &pairwise, &baseline, &config, 0.0);
        assert!(!report.baseline_sufficient);
        assert_eq!(report.verdict, DeviationVerdict::WithinBaseline);
    }

    #[test]
    fn extreme_deviation_detected() {
        let baseline = make_baseline(20);
        let config = ProxyConfig::default();

        // Honesty score wildly different from baseline mean of ~0.7
        let current: HashMap<String, f64> =
            [("honesty".into(), -0.5), ("courage".into(), -0.5)]
                .into_iter()
                .collect();
        // Pairwise relationship inverted
        let pairwise: HashMap<(String, String), f64> =
            [(("honesty".into(), "courage".into()), -0.8)]
                .into_iter()
                .collect();

        let report = detect_deviation(&current, &pairwise, &baseline, &config, 0.0);
        assert!(report.baseline_sufficient);
        assert!(!report.flagged_terms.is_empty());
        assert!(report.combined_score > 0.3);
    }
}
