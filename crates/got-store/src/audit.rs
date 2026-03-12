// ---------------------------------------------------------------------------
// Audit report types and generation logic
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};

use got_core::GeometricAttestation;

use crate::store::attestation_store_id;

/// Summary of geometry drift across an attestation chain.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DriftSummary {
    /// How many attestations contained a non-None geometry_drift.
    pub readings_with_drift: usize,
    pub max_drift: Option<f64>,
    pub mean_drift: Option<f64>,
}

/// Summary of causal intervention results across an attestation chain.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalSummary {
    /// Attestations that had at least one causal_score entry.
    pub attestations_with_causal: usize,
    pub causal_pass_count: usize,
    pub causal_fail_count: usize,
    pub mean_consistency: Option<f64>,
}

/// Structured audit report for a model's attestation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub model_id: String,
    pub total_attestations: usize,
    pub chain_length: usize,
    /// True iff every parent_attestation_hash link resolves within the chain.
    pub chain_valid: bool,
    pub first_timestamp: Option<u64>,
    pub last_timestamp: Option<u64>,
    pub schema_versions_seen: Vec<u16>,
    pub drift_summary: DriftSummary,
    pub causal_summary: CausalSummary,
    /// SHA-256 hashes of unique signer public keys (derived from content IDs).
    pub signers: Vec<[u8; 32]>,
}

/// Build an audit report from a sorted slice of attestations and their
/// signer key hashes.
pub fn build_audit_report(
    model_id: &str,
    attestations: &[&GeometricAttestation],
    signer_hashes: &[[u8; 32]],
) -> AuditReport {
    if attestations.is_empty() {
        return AuditReport {
            model_id: model_id.to_string(),
            total_attestations: 0,
            chain_length: 0,
            chain_valid: true,
            first_timestamp: None,
            last_timestamp: None,
            schema_versions_seen: Vec::new(),
            drift_summary: DriftSummary::default(),
            causal_summary: CausalSummary::default(),
            signers: Vec::new(),
        };
    }

    // Sort by timestamp for chain analysis.
    let mut sorted: Vec<&GeometricAttestation> = attestations.to_vec();
    sorted.sort_by_key(|a| a.timestamp);

    // --- schema versions ---
    let mut schema_versions: Vec<u16> = sorted.iter().map(|a| a.schema_version).collect();
    schema_versions.sort();
    schema_versions.dedup();

    // --- timestamps ---
    let first_timestamp = sorted.first().map(|a| a.timestamp);
    let last_timestamp = sorted.last().map(|a| a.timestamp);

    // --- chain validation ---
    // Build a set of known content IDs.
    let mut known_ids: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    let mut chain_valid = true;
    let mut chain_length: usize = 0;

    for a in &sorted {
        let id = attestation_store_id(a).unwrap_or([0u8; 32]);
        known_ids.insert(id);
    }

    // Walk the chain: for each attestation with a parent hash, check it exists.
    for a in &sorted {
        if let Some(parent) = a.parent_attestation_hash {
            if !known_ids.contains(&parent) {
                chain_valid = false;
            }
        }
    }

    // Compute chain length: walk backwards from the latest attestation's parent chain.
    if let Some(tip) = sorted.last() {
        chain_length = 1;
        let mut current_parent = tip.parent_attestation_hash;
        while let Some(parent_hash) = current_parent {
            if let Some(parent) = sorted
                .iter()
                .find(|a| attestation_store_id(a).ok() == Some(parent_hash))
            {
                chain_length += 1;
                current_parent = parent.parent_attestation_hash;
            } else {
                break;
            }
        }
    }

    // --- drift summary ---
    let mut drift_vals: Vec<f64> = Vec::new();
    for a in &sorted {
        if let Some(d) = a.geometry_drift {
            drift_vals.push(d as f64);
        }
    }
    let drift_summary = DriftSummary {
        readings_with_drift: drift_vals.len(),
        max_drift: drift_vals.iter().cloned().reduce(f64::max),
        mean_drift: if drift_vals.is_empty() {
            None
        } else {
            Some(drift_vals.iter().sum::<f64>() / drift_vals.len() as f64)
        },
    };

    // --- causal summary ---
    let mut causal_attestations = 0usize;
    let mut pass_count = 0usize;
    let mut fail_count = 0usize;
    let mut consistency_vals: Vec<f64> = Vec::new();

    for a in &sorted {
        if !a.causal_scores.is_empty() {
            causal_attestations += 1;
            for cs in &a.causal_scores {
                consistency_vals.push(cs.consistency as f64);
                if cs.is_causal {
                    pass_count += 1;
                } else {
                    fail_count += 1;
                }
            }
        }
    }
    let causal_summary = CausalSummary {
        attestations_with_causal: causal_attestations,
        causal_pass_count: pass_count,
        causal_fail_count: fail_count,
        mean_consistency: if consistency_vals.is_empty() {
            None
        } else {
            Some(consistency_vals.iter().sum::<f64>() / consistency_vals.len() as f64)
        },
    };

    // --- unique signers ---
    let mut signers: Vec<[u8; 32]> = signer_hashes.to_vec();
    signers.sort();
    signers.dedup();

    AuditReport {
        model_id: model_id.to_string(),
        total_attestations: sorted.len(),
        chain_length,
        chain_valid,
        first_timestamp,
        last_timestamp,
        schema_versions_seen: schema_versions,
        drift_summary,
        causal_summary,
        signers,
    }
}
