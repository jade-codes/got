// ---------------------------------------------------------------------------
// ModelContext — cached expensive invariants for attestation production.
//
// The attestation pipeline has two cost tiers:
//
//   Expensive (cached, changes on model update):
//     - CausalGeometry Φ = UᵀU        — O(Vd²)
//     - Trained probe weights          — SGD under causal IP
//     - Causal validation results      — model forward passes for
//                                        each probe's causal_check
//     - geometry_hash                  — SHA-256(Φ)
//     - parent_attestation_hash        — chain link to the previous
//                                        attestation
//
//   Cheap (computed fresh per attestation, depends on input context):
//     - Forward pass → activations
//     - read_probe() per probe per layer → layer_readings, confidence,
//       coverage_flags
//     - assemble_and_sign() → signed GeometricAttestation
//
// ModelContext holds the expensive tier.  The per-attestation work is
// the caller's responsibility — typically the MeasurementSidecar or
// an explicit pipeline function that takes a ModelContext and a set of
// activations and produces a signed attestation.
//
// Invalidation triggers:
//   1. Agent startup (no context exists yet)
//   2. Model update (new U → recompute Φ, retrain probes, re-run
//      causal checks, chain link to the previous attestation)
//   3. detect_distribution_shift() fires from the MeasurementSidecar
//      (probe staleness detected → retrain probes against current Φ)
//   4. Manual operator trigger
//
// The ModelContext does NOT hold a signed attestation.  Every attestation
// is computed fresh from current activations + the cached context.
// ---------------------------------------------------------------------------

use std::sync::RwLock;

use got_core::geometry::CausalGeometry;
use got_core::CausalScoreRecord;

/// Cached expensive invariants for attestation production.
///
/// Thread-safe via interior `RwLock` — safe to share across tokio tasks
/// via `Arc<ModelContext>`.  The context is read-heavy (every exchange
/// reads it) and write-rare (only on model update or invalidation).
#[derive(Debug)]
pub struct ModelContext {
    inner: RwLock<Option<CachedInvariants>>,
}

/// The expensive invariants that change only on model update.
#[derive(Clone)]
pub struct CachedInvariants {
    /// Causal geometry Φ = UᵀU.
    pub geometry: CausalGeometry,
    /// Trained probe weights, bound to `geometry` via `geometry_hash`.
    pub probe_weights: Vec<ProbeEntry>,
    /// Causal validation results (one per probe, from `causal_check`).
    /// Empty if causal validation was not run.
    pub causal_scores: Vec<CausalScoreRecord>,
    /// SHA-256(Φ) — deterministic fingerprint of the geometry.
    pub geometry_hash: [u8; 32],
    /// Hash of the previous attestation in the chain.  `None` for
    /// the first attestation (chain anchor).
    pub parent_attestation_hash: Option<[u8; 32]>,
    /// Frobenius drift from the reference geometry (if chained).
    pub geometry_drift: Option<f32>,
    /// Unix timestamp when these invariants were computed.
    pub computed_at: u64,
    /// Model identifier string.
    pub model_id: String,
    /// Model hash (Merkle root over weight shards), if available.
    pub model_hash: Option<[u8; 32]>,
}

/// One trained probe and its metadata, stored inside the model context.
#[derive(Debug, Clone)]
pub struct ProbeEntry {
    /// Probe weight vector (length = hidden_dim).
    pub weights: Vec<f32>,
    /// Platt scaling bias.
    pub bias: f32,
    /// Platt scaling: scale parameter.
    pub platt_scale: f32,
    /// Platt scaling: shift parameter.
    pub platt_shift: f32,
    /// Human-readable label for this probe direction.
    pub label: String,
}

impl ModelContext {
    /// Create an empty context.  The first call to `get()` returns
    /// `None`, signalling that the caller must compute the invariants
    /// and call `update()`.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(None),
        }
    }

    /// Create a context pre-loaded with invariants (e.g. from a
    /// persisted checkpoint).
    pub fn with_invariants(invariants: CachedInvariants) -> Self {
        Self {
            inner: RwLock::new(Some(invariants)),
        }
    }

    /// Read the current invariants.  Returns `None` if the context
    /// has not been populated yet (startup) or has been invalidated.
    pub fn get(&self) -> Option<CachedInvariants> {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        guard.clone()
    }

    /// True if the context holds valid invariants.
    pub fn is_ready(&self) -> bool {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        guard.is_some()
    }

    /// Replace the invariants.  Call this after recomputing Φ,
    /// retraining probes, and re-running causal checks.
    ///
    /// The previous invariants (if any) are dropped.  If you need
    /// the previous `parent_attestation_hash` for chain linking,
    /// read it via `get()` before calling `update()`.
    pub fn update(&self, invariants: CachedInvariants) {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        *guard = Some(invariants);
    }

    /// Clear the invariants, forcing the next consumer to recompute.
    ///
    /// Call this when:
    ///   - `detect_distribution_shift()` fires (probes are stale)
    ///   - The model's unembedding matrix U has changed
    ///   - An operator forces a re-probe
    pub fn invalidate(&self) {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }

    /// Unix timestamp when the invariants were last computed, or
    /// `None` if the context has never been populated.
    pub fn computed_at(&self) -> Option<u64> {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().map(|i| i.computed_at)
    }
}

impl std::fmt::Debug for CachedInvariants {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedInvariants")
            .field("model_id", &self.model_id)
            .field("probe_count", &self.probe_weights.len())
            .field("causal_score_count", &self.causal_scores.len())
            .field("computed_at", &self.computed_at)
            .finish()
    }
}

impl Default for ModelContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use got_core::geometry::CausalGeometry;

    fn dummy_geometry() -> CausalGeometry {
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        CausalGeometry::from_raw_gram(data, 3).unwrap()
    }

    fn dummy_invariants() -> CachedInvariants {
        let geo = dummy_geometry();
        CachedInvariants {
            geometry_hash: geo.geometry_hash(),
            geometry: geo,
            probe_weights: vec![ProbeEntry {
                weights: vec![1.0, 0.0, 0.0],
                bias: 0.0,
                platt_scale: 1.0,
                platt_shift: 0.0,
                label: "test-probe".into(),
            }],
            causal_scores: vec![],
            parent_attestation_hash: None,
            geometry_drift: None,
            computed_at: 1_000_000,
            model_id: "test-model".into(),
            model_hash: Some([0x11; 32]),
        }
    }

    #[test]
    fn empty_context_returns_none() {
        let ctx = ModelContext::new();
        assert!(!ctx.is_ready());
        assert!(ctx.get().is_none());
        assert!(ctx.computed_at().is_none());
    }

    #[test]
    fn update_populates_context() {
        let ctx = ModelContext::new();
        ctx.update(dummy_invariants());
        assert!(ctx.is_ready());
        let inv = ctx.get().unwrap();
        assert_eq!(inv.model_id, "test-model");
        assert_eq!(inv.computed_at, 1_000_000);
    }

    #[test]
    fn invalidate_clears_context() {
        let ctx = ModelContext::with_invariants(dummy_invariants());
        assert!(ctx.is_ready());
        ctx.invalidate();
        assert!(!ctx.is_ready());
        assert!(ctx.get().is_none());
    }

    #[test]
    fn update_replaces_previous() {
        let ctx = ModelContext::new();
        ctx.update(dummy_invariants());
        let mut inv2 = dummy_invariants();
        inv2.model_id = "updated-model".into();
        inv2.computed_at = 2_000_000;
        ctx.update(inv2);
        let inv = ctx.get().unwrap();
        assert_eq!(inv.model_id, "updated-model");
        assert_eq!(inv.computed_at, 2_000_000);
    }

    #[test]
    fn thread_safe_concurrent_reads() {
        use std::sync::Arc;
        let ctx = Arc::new(ModelContext::with_invariants(dummy_invariants()));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = ctx.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        assert!(c.is_ready());
                        let _ = c.get();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn thread_safe_concurrent_read_write() {
        use std::sync::Arc;
        let ctx = Arc::new(ModelContext::new());
        let writer = {
            let c = ctx.clone();
            std::thread::spawn(move || {
                for i in 0..10 {
                    let mut inv = dummy_invariants();
                    inv.computed_at = i;
                    c.update(inv);
                    std::thread::yield_now();
                }
            })
        };
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let c = ctx.clone();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        let _ = c.get(); // may be None or Some
                    }
                })
            })
            .collect();
        writer.join().unwrap();
        for h in readers {
            h.join().unwrap();
        }
        // After the writer finishes, context should be populated.
        assert!(ctx.is_ready());
    }
}
