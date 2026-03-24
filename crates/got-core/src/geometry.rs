// ---------------------------------------------------------------------------
// The causal inner product — the maths this whole system rests on.
//
//   ⟨u, v⟩_c = uᵀ Φ v     where Φ = UᵀU
//
// Equivalent to: transform both vectors as Uh, then take Euclidean dot.
// ---------------------------------------------------------------------------

use thiserror::Error;

use crate::UnembeddingMatrix;

#[derive(Debug, Error)]
pub enum GeometryError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("NaN detected in input vector")]
    NaN,
    #[error("infinity detected in input vector")]
    Infinity,
    #[error("NaN detected in Gram matrix at [{row}, {col}]")]
    GramNaN { row: usize, col: usize },
    #[error("infinity detected in Gram matrix at [{row}, {col}]")]
    GramInfinity { row: usize, col: usize },
    #[error("Gram matrix is not symmetric: [{row},{col}]={a} vs [{col},{row}]={b}")]
    NotSymmetric {
        row: usize,
        col: usize,
        a: f32,
        b: f32,
    },
}

/// Result of projecting probe directions through the causal geometry.
pub struct ValueProjection {
    /// k×k projected Gram matrix G_W = WᵀΦW, row-major.
    pub gram_w: Vec<f32>,
    /// Eigenvalues of G_W, sorted descending.
    pub eigenvalues: Vec<f32>,
    /// Effective dimensionality (participation ratio). ∈ [1, k].
    pub dim_eff: f32,
    /// Number of probes (k).
    pub k: usize,
}

/// Precomputed causal geometry derived from an unembedding matrix.
///
/// Holds Φ = UᵀU (d × d) and metadata about the matrix's rank / regularisation.
#[derive(Clone)]
pub struct CausalGeometry {
    /// Precomputed Φ = UᵀU, shape d × d, row-major.
    gram: Vec<f32>,
    hidden_dim: usize,
    is_full_rank: bool,
    epsilon: f32,
    /// True when Φ = I — inner product degenerates to plain dot product.
    is_identity: bool,
}

/// Cholesky factorisation check: returns `true` if the `d×d` row-major
/// matrix `gram` is positive-definite (all diagonal elements of L remain
/// strictly positive during decomposition).
fn cholesky_check(gram: &[f32], d: usize) -> bool {
    let mut l = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * d + k] * l[j * d + k];
            }
            if i == j {
                let diag = gram[i * d + i] - sum;
                if diag <= 0.0 {
                    return false; // not positive-definite
                }
                l[i * d + j] = diag.sqrt();
            } else {
                let denom = l[j * d + j];
                if denom.abs() < f32::EPSILON {
                    return false;
                }
                l[i * d + j] = (gram[i * d + j] - sum) / denom;
            }
        }
    }
    true
}

/// Check whether a d×d row-major matrix is the identity (within f32 tolerance).
fn is_identity_matrix(gram: &[f32], d: usize) -> bool {
    for i in 0..d {
        for j in 0..d {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (gram[i * d + j] - expected).abs() > 1e-6 {
                return false;
            }
        }
    }
    true
}

impl CausalGeometry {
    /// Create an identity geometry (Φ = I) without allocating the full d×d matrix
    /// or running a Cholesky check. All inner products degenerate to plain dot products.
    pub fn identity(hidden_dim: usize) -> Self {
        Self {
            gram: Vec::new(), // not used when is_identity = true
            hidden_dim,
            is_full_rank: true,
            epsilon: 0.0,
            is_identity: true,
        }
    }

    /// Build the causal geometry from an unembedding matrix.
    ///
    /// Computes Φ = UᵀU using `faer` for efficient matrix multiplication.
    /// If the matrix appears rank-deficient (trace heuristic), regularises
    /// with Φ_ε = UᵀU + εI.
    pub fn from_unembedding(u: &UnembeddingMatrix, epsilon: f32) -> Self {
        let v = u.vocab_size;
        let d = u.hidden_dim;

        // Build faer matrix from raw data (V × d)
        let u_mat = faer::Mat::from_fn(v, d, |i, j| u.data[i * d + j]);

        // Φ = Uᵀ · U  →  (d × V) · (V × d) = d × d
        let phi_mat = u_mat.transpose() * &u_mat;

        // Extract to row-major Vec<f32>
        let mut gram = vec![0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                gram[i * d + j] = phi_mat[(i, j)];
            }
        }

        // Rank check via Cholesky factorisation (S-12).
        let is_full_rank = cholesky_check(&gram, d);

        // Regularise if needed: Φ_ε = Φ + εI
        if !is_full_rank {
            for i in 0..d {
                gram[i * d + i] += epsilon;
            }
        }

        Self {
            gram,
            hidden_dim: d,
            is_full_rank,
            epsilon,
            is_identity: false,
        }
    }

    /// Reconstruct a `CausalGeometry` from a raw Gram matrix (e.g. loaded from a checkpoint).
    ///
    /// Assumes the Gram matrix is valid and already regularised if needed.
    pub fn from_raw_gram(gram: Vec<f32>, hidden_dim: usize) -> Result<Self, GeometryError> {
        if gram.len() != hidden_dim * hidden_dim {
            return Err(GeometryError::DimensionMismatch {
                expected: hidden_dim * hidden_dim,
                got: gram.len(),
            });
        }

        // S-4: Validate entries — reject NaN, Infinity, and asymmetry.
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                let v = gram[i * hidden_dim + j];
                if v.is_nan() {
                    return Err(GeometryError::GramNaN { row: i, col: j });
                }
                if v.is_infinite() {
                    return Err(GeometryError::GramInfinity { row: i, col: j });
                }
                if j > i {
                    let a = gram[i * hidden_dim + j];
                    let b = gram[j * hidden_dim + i];
                    if (a - b).abs() > 1e-6 * (1.0 + a.abs() + b.abs()) {
                        return Err(GeometryError::NotSymmetric {
                            row: i,
                            col: j,
                            a,
                            b,
                        });
                    }
                }
            }
        }

        // S-12: Cholesky factorisation to check positive-definiteness.
        let is_full_rank = cholesky_check(&gram, hidden_dim);

        // Detect identity matrix for fast-path inner products.
        let is_identity = is_identity_matrix(&gram, hidden_dim);

        Ok(Self {
            gram,
            hidden_dim,
            is_full_rank,
            epsilon: 0.0,
            is_identity,
        })
    }

    /// Compute the causal inner product ⟨w, h⟩_c = wᵀ Φ h.
    ///
    /// When Φ = I this is a plain dot product (O(d) instead of O(d²)).
    pub fn inner_product(&self, w: &[f32], h: &[f32]) -> Result<f32, GeometryError> {
        let d = self.hidden_dim;
        self.check_vec(w)?;
        self.check_vec(h)?;

        if self.is_identity {
            let result: f32 = w.iter().zip(h.iter()).map(|(a, b)| a * b).sum();
            return Ok(result);
        }

        // Compute Φh (d-dimensional vector)
        let phi_h: Vec<f32> = (0..d)
            .map(|i| (0..d).map(|j| self.gram[i * d + j] * h[j]).sum::<f32>())
            .collect();

        // Dot w · (Φh)
        let result: f32 = w.iter().zip(phi_h.iter()).map(|(wi, pi)| wi * pi).sum();
        Ok(result)
    }

    /// Compute Φh — the Gram-matrix–vector product.
    ///
    /// When Φ = I, returns a clone of h.
    pub fn gram_vec(&self, h: &[f32]) -> Result<Vec<f32>, GeometryError> {
        let d = self.hidden_dim;
        self.check_vec(h)?;

        if self.is_identity {
            return Ok(h.to_vec());
        }

        let phi_h: Vec<f32> = (0..d)
            .map(|i| (0..d).map(|j| self.gram[i * d + j] * h[j]).sum::<f32>())
            .collect();

        Ok(phi_h)
    }

    /// Transform h → Uh ∈ ℝ^V for diagnostic / visualisation use.
    ///
    /// Not used in the training path (we train directly in ℝ^d with the
    /// causal inner product), but useful for verifying equivalence.
    pub fn transform(&self, u: &UnembeddingMatrix, h: &[f32]) -> Result<Vec<f32>, GeometryError> {
        let v = u.vocab_size;
        let d = u.hidden_dim;
        if h.len() != d {
            return Err(GeometryError::DimensionMismatch {
                expected: d,
                got: h.len(),
            });
        }
        if h.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(GeometryError::NaN);
        }

        let result: Vec<f32> = (0..v)
            .map(|k| (0..d).map(|j| u.data[k * d + j] * h[j]).sum())
            .collect();

        Ok(result)
    }

    /// Whether the Gram matrix is full-rank (causal IP is positive definite).
    pub fn is_positive_definite(&self) -> bool {
        self.is_full_rank
    }

    /// The hidden dimension d.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// The epsilon used for regularisation.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Access the raw Gram matrix (d × d, row-major).
    pub fn gram(&self) -> &[f32] {
        &self.gram
    }

    /// SHA-256 hash of the Gram matrix (f32 LE bytes, row-major).
    ///
    /// Deterministic fingerprint of the current geometry.
    /// Same Φ → same hash, always.
    ///
    /// Includes `hidden_dim` as a prefix to prevent dimension-confusion
    /// attacks where matrices of different shapes have identical byte
    /// representations.
    pub fn geometry_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        // Domain: include dimension so different-shaped matrices can't collide
        hasher.update((self.hidden_dim as u64).to_le_bytes());
        // S-5: Include epsilon so different regularisation → different hash
        let eps_canonical = if self.epsilon == 0.0 {
            0.0f32
        } else {
            self.epsilon
        };
        hasher.update(eps_canonical.to_le_bytes());
        for &val in &self.gram {
            // Canonicalise: -0.0 → 0.0 before hashing
            let canonical = if val == 0.0 { 0.0f32 } else { val };
            hasher.update(canonical.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Normalised Frobenius distance between this geometry and a reference.
    ///
    /// Returns ‖Φ_self − Φ_ref‖_F / ‖Φ_ref‖_F.
    /// Zero if identical, positive otherwise.
    /// Returns `f32::MAX` if the reference norm is zero but the difference
    /// is non-zero (avoids INFINITY which would fail serialisation).
    pub fn drift_from(&self, reference: &CausalGeometry) -> Result<f32, GeometryError> {
        if self.hidden_dim != reference.hidden_dim {
            return Err(GeometryError::DimensionMismatch {
                expected: reference.hidden_dim,
                got: self.hidden_dim,
            });
        }
        let frobenius_delta_sq: f32 = self
            .gram
            .iter()
            .zip(reference.gram.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let frobenius_ref_sq: f32 = reference.gram.iter().map(|x| x * x).sum();
        if frobenius_ref_sq == 0.0 {
            return Ok(if frobenius_delta_sq == 0.0 {
                0.0
            } else {
                f32::MAX
            });
        }
        Ok((frobenius_delta_sq / frobenius_ref_sq).sqrt())
    }

    /// Drift along a specific probe direction.
    ///
    /// Computes |wᵀ(Φ_new − Φ_ref)w| / |wᵀΦ_ref w|
    ///
    /// This measures how much the geometry has changed *specifically*
    /// in the direction the probe measures, not just globally.
    /// An adversary who surgically modifies probe-relevant directions
    /// while keeping global Frobenius drift small will be caught here.
    ///
    /// Returns 0.0 for identical geometries, `f32::MAX` if reference
    /// quadratic form is zero but new is not (avoids INFINITY which would
    /// fail serialisation and float comparisons).
    pub fn directional_drift(
        &self,
        reference: &CausalGeometry,
        direction: &[f32],
    ) -> Result<f32, GeometryError> {
        if self.hidden_dim != reference.hidden_dim {
            return Err(GeometryError::DimensionMismatch {
                expected: reference.hidden_dim,
                got: self.hidden_dim,
            });
        }
        self.check_vec(direction)?;

        let quad_new = self.quadratic_form(direction);
        let quad_ref = reference.quadratic_form(direction);

        if quad_ref.abs() < f32::EPSILON {
            return Ok(if (quad_new - quad_ref).abs() < f32::EPSILON {
                0.0
            } else {
                f32::MAX
            });
        }
        Ok((quad_new - quad_ref).abs() / quad_ref.abs())
    }

    /// Compute wᵀΦw for a direction vector w.
    ///
    /// This is the quadratic form of the Gram matrix applied to w.
    /// Used internally for directional drift computation.
    fn quadratic_form(&self, w: &[f32]) -> f32 {
        let n = self.hidden_dim;
        let mut result = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                result += w[i] * self.gram[i * n + j] * w[j];
            }
        }
        result
    }

    /// Compute the value-projected Gram matrix G_W = WᵀΦW and its eigenvalues.
    ///
    /// `probe_weights` is a slice of k probe weight vectors, each of dimension d.
    /// Returns a `ValueProjection` with the k×k matrix, eigenvalues, and dim_eff.
    pub fn value_projected_gram(
        &self,
        probe_weights: &[&[f32]],
    ) -> Result<ValueProjection, GeometryError> {
        let k = probe_weights.len();
        if k == 0 {
            return Err(GeometryError::DimensionMismatch { expected: 1, got: 0 });
        }
        for w in probe_weights {
            self.check_vec(w)?;
        }

        // Compute ΦW: for each probe wⱼ, compute Φwⱼ (d-dimensional vector)
        let mut phi_w: Vec<Vec<f32>> = Vec::with_capacity(k);
        for w in probe_weights {
            phi_w.push(self.gram_vec(w)?);
        }

        // Compute G_W = WᵀΦW: G_W[i][j] = wᵢᵀ (Φwⱼ)
        let mut gram_w = vec![0.0f32; k * k];
        for i in 0..k {
            for j in i..k {
                let dot: f32 = probe_weights[i]
                    .iter()
                    .zip(phi_w[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                gram_w[i * k + j] = dot;
                gram_w[j * k + i] = dot; // symmetric
            }
        }

        // Eigendecomposition of the k×k symmetric matrix using faer
        let mat = faer::Mat::from_fn(k, k, |i, j| gram_w[i * k + j] as f64);
        let eigendecomp = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eig_vals = eigendecomp.s().column_vector();

        let mut eigenvalues: Vec<f32> = (0..k)
            .map(|i| eig_vals.read(i) as f32)
            .collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Participation ratio: dim_eff = (Σλᵢ)² / Σλᵢ²
        // Use only positive eigenvalues (numerical noise can produce tiny negatives)
        let sum: f32 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();
        let sum_sq: f32 = eigenvalues.iter().filter(|&&v| v > 0.0).map(|v| v * v).sum();
        let dim_eff = if sum_sq > f32::EPSILON {
            (sum * sum) / sum_sq
        } else {
            1.0
        };

        Ok(ValueProjection {
            gram_w,
            eigenvalues,
            dim_eff,
            k,
        })
    }

    /// Compute effective value dimensionality (participation ratio of G_W eigenvalues).
    ///
    /// Shorthand for `value_projected_gram(probe_weights)?.dim_eff`.
    /// Returns dim_eff ∈ [1, k].
    pub fn effective_value_dimensionality(
        &self,
        probe_weights: &[&[f32]],
    ) -> Result<f32, GeometryError> {
        Ok(self.value_projected_gram(probe_weights)?.dim_eff)
    }

    // --- internal helpers ---

    pub(crate) fn check_vec(&self, v: &[f32]) -> Result<(), GeometryError> {
        if v.len() != self.hidden_dim {
            return Err(GeometryError::DimensionMismatch {
                expected: self.hidden_dim,
                got: v.len(),
            });
        }
        for x in v {
            if x.is_infinite() {
                return Err(GeometryError::Infinity);
            }
            if x.is_nan() {
                return Err(GeometryError::NaN);
            }
        }
        Ok(())
    }
}

/// Euclidean cosine similarity between two vectors (no geometry weighting).
///
/// Returns 0.0 if either vector has near-zero norm.
/// Result is clamped to [-1, 1].
pub fn euclidean_cosine(u: &[f32], v: &[f32]) -> f32 {
    let dot: f32 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let norm_u: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_u < f32::EPSILON || norm_v < f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_u * norm_v)).clamp(-1.0, 1.0)
}

/// Result of comparing value geometry between two models.
pub struct AlignmentDistance {
    /// Global Frobenius distance: ‖Φ_A − Φ_B‖_F / max(‖Φ_A‖_F, ‖Φ_B‖_F).
    pub global_distance: f32,
    /// Probe-projected distance d_V(A, B), if probes were provided.
    pub probe_projected_distance: Option<f32>,
    /// Per-probe distances, if probes were provided.
    pub per_probe_distances: Option<Vec<f32>>,
}

/// Compare value geometry between two models.
///
/// Global distance: d(A,B) = ‖Φ_A − Φ_B‖_F / max(‖Φ_A‖_F, ‖Φ_B‖_F)
///
/// Probe-projected distance (when probes provided):
/// d_V(A,B) = (1/k) Σⱼ |wⱼᵀ(Φ_A − Φ_B)wⱼ| / max(|wⱼᵀΦ_Awⱼ|, |wⱼᵀΦ_Bwⱼ|)
pub fn value_alignment_distance(
    geo_a: &CausalGeometry,
    geo_b: &CausalGeometry,
    probe_weights: Option<&[&[f32]]>,
) -> Result<AlignmentDistance, GeometryError> {
    if geo_a.hidden_dim() != geo_b.hidden_dim() {
        return Err(GeometryError::DimensionMismatch {
            expected: geo_a.hidden_dim(),
            got: geo_b.hidden_dim(),
        });
    }

    // Global Frobenius distance: ‖Φ_A − Φ_B‖_F / max(‖Φ_A‖_F, ‖Φ_B‖_F)
    let frob_delta_sq: f32 = geo_a
        .gram()
        .iter()
        .zip(geo_b.gram().iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let frob_a: f32 = geo_a.gram().iter().map(|x| x * x).sum::<f32>().sqrt();
    let frob_b: f32 = geo_b.gram().iter().map(|x| x * x).sum::<f32>().sqrt();
    let max_frob = frob_a.max(frob_b);
    let global_distance = if max_frob > f32::EPSILON {
        frob_delta_sq.sqrt() / max_frob
    } else {
        0.0
    };

    // Probe-projected distance
    let (probe_projected_distance, per_probe_distances) = if let Some(probes) = probe_weights {
        if probes.is_empty() {
            (Some(0.0), Some(Vec::new()))
        } else {
            let mut per_probe = Vec::with_capacity(probes.len());
            for w in probes {
                geo_a.check_vec(w)?;
                let quad_a = quadratic_form_raw(geo_a.gram(), geo_a.hidden_dim(), w);
                let quad_b = quadratic_form_raw(geo_b.gram(), geo_b.hidden_dim(), w);
                let max_quad = quad_a.abs().max(quad_b.abs());
                let d_w = if max_quad > f32::EPSILON {
                    (quad_a - quad_b).abs() / max_quad
                } else {
                    0.0
                };
                per_probe.push(d_w);
            }
            let mean = per_probe.iter().sum::<f32>() / per_probe.len() as f32;
            (Some(mean), Some(per_probe))
        }
    } else {
        (None, None)
    };

    Ok(AlignmentDistance {
        global_distance,
        probe_projected_distance,
        per_probe_distances,
    })
}

/// Compute wᵀΦw from raw gram data.
fn quadratic_form_raw(gram: &[f32], d: usize, w: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..d {
        for j in 0..d {
            result += w[i] * gram[i * d + j] * w[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny 3×2 unembedding matrix where we can hand-verify Φ.
    ///
    /// U = [[1, 2],
    ///      [3, 4],
    ///      [5, 6]]
    ///
    /// UᵀU = [[1,3,5],[2,4,6]] · [[1,2],[3,4],[5,6]]
    ///      = [[1+9+25, 2+12+30], [2+12+30, 4+16+36]]
    ///      = [[35, 44], [44, 56]]
    fn tiny_unembedding() -> UnembeddingMatrix {
        UnembeddingMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn gram_matrix_is_correct() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let gram = geom.gram();
        let d = 2;
        // Expected: [[35, 44], [44, 56]]
        assert!((gram[0 * d + 0] - 35.0).abs() < 1e-4, "Φ[0,0]={}", gram[0]);
        assert!((gram[0 * d + 1] - 44.0).abs() < 1e-4, "Φ[0,1]={}", gram[1]);
        assert!((gram[1 * d + 0] - 44.0).abs() < 1e-4, "Φ[1,0]={}", gram[2]);
        assert!((gram[1 * d + 1] - 56.0).abs() < 1e-4, "Φ[1,1]={}", gram[3]);
    }

    #[test]
    fn inner_product_matches_hand_calculation() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [1.0, 0.0];
        let h = [0.0, 1.0];

        // ⟨w, h⟩_c = wᵀ Φ h
        // Φh = [[35,44],[44,56]] · [0,1] = [44, 56]
        // wᵀ · [44, 56] = [1,0] · [44,56] = 44
        let result = geom.inner_product(&w, &h).unwrap();
        assert!((result - 44.0).abs() < 1e-4, "got {result}");
    }

    #[test]
    fn inner_product_general_case() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [2.0, 3.0];
        let h = [4.0, 5.0];

        // Φh = [[35,44],[44,56]] · [4,5] = [140+220, 176+280] = [360, 456]
        // wᵀ(Φh) = 2*360 + 3*456 = 720 + 1368 = 2088
        let result = geom.inner_product(&w, &h).unwrap();
        assert!((result - 2088.0).abs() < 1e-2, "got {result}");
    }

    #[test]
    fn inner_product_is_symmetric() {
        // Φ = UᵀU is symmetric, so ⟨w,h⟩_c = wᵀΦh = hᵀΦw = ⟨h,w⟩_c
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [2.0, 3.0];
        let h = [4.0, 5.0];

        let wh = geom.inner_product(&w, &h).unwrap();
        let hw = geom.inner_product(&h, &w).unwrap();
        assert!(
            (wh - hw).abs() < 1e-4,
            "causal IP should be symmetric: ⟨w,h⟩={wh}, ⟨h,w⟩={hw}"
        );
    }

    #[test]
    fn transform_produces_correct_output() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let h = [1.0, 1.0];
        // Uh: row k = U[k,0]*h[0] + U[k,1]*h[1]
        //   row 0 = 1*1 + 2*1 = 3
        //   row 1 = 3*1 + 4*1 = 7
        //   row 2 = 5*1 + 6*1 = 11
        let result = geom.transform(&u, &h).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 3.0).abs() < 1e-4);
        assert!((result[1] - 7.0).abs() < 1e-4);
        assert!((result[2] - 11.0).abs() < 1e-4);
    }

    #[test]
    fn causal_ip_equals_euclidean_on_transformed() {
        // The fundamental equivalence: ⟨w, h⟩_c = (Uw) · (Uh)
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [2.0, 3.0];
        let h = [4.0, 5.0];

        let causal = geom.inner_product(&w, &h).unwrap();

        let uw = geom.transform(&u, &w).unwrap();
        let uh = geom.transform(&u, &h).unwrap();
        let euclidean: f32 = uw.iter().zip(uh.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (causal - euclidean).abs() < 1e-2,
            "causal={causal}, euclidean={euclidean}"
        );
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [1.0, 2.0, 3.0]; // wrong dimension
        let h = [1.0, 2.0];
        assert!(geom.inner_product(&w, &h).is_err());
    }

    #[test]
    fn nan_in_w_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [1.0, f32::NAN];
        let h = [1.0, 2.0];
        assert!(geom.inner_product(&w, &h).is_err());
    }

    #[test]
    fn nan_in_h_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [1.0, 2.0];
        let h = [f32::NAN, 1.0];
        assert!(geom.inner_product(&w, &h).is_err());
    }

    #[test]
    fn gram_vec_nan_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        assert!(geom.gram_vec(&[f32::NAN, 1.0]).is_err());
    }

    #[test]
    fn gram_vec_matches_inner_product() {
        // gram_vec(h) = Φh, so wᵀ·gram_vec(h) should equal inner_product(w, h)
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        let w = [2.0, 3.0];
        let h = [4.0, 5.0];

        let phi_h = geom.gram_vec(&h).unwrap();
        let dot: f32 = w.iter().zip(phi_h.iter()).map(|(a, b)| a * b).sum();
        let ip = geom.inner_product(&w, &h).unwrap();

        assert!((dot - ip).abs() < 1e-4, "wᵀΦh={dot} vs inner_product={ip}");
    }

    #[test]
    fn regularisation_on_rank_deficient() {
        // U with near-zero entries: UᵀU ≈ 0, trace ≈ 0.
        // Threshold = ε * d = 1.0 * 2 = 2.0 → triggers regularisation.
        let u = UnembeddingMatrix::new(1, 2, vec![1e-8, 1e-8]).unwrap();
        let geom = CausalGeometry::from_unembedding(&u, 1.0);

        assert!(!geom.is_positive_definite());

        // Diagonal should have epsilon added: Φ[0,0] ≈ 0 + 1.0 = 1.0
        let gram = geom.gram();
        assert!(
            (gram[0] - 1.0).abs() < 1e-4,
            "Φ[0,0] should be ~ε, got {}",
            gram[0]
        );

        // Off-diagonal should remain near zero (no ε added)
        assert!(gram[1].abs() < 1e-4, "Φ[0,1] should be ~0, got {}", gram[1]);

        // Inner product should still work on regularised geometry
        let result = geom.inner_product(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(result.is_finite(), "regularised IP should be finite");
    }

    #[test]
    fn positive_definite_tiny() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);
        assert!(geom.is_positive_definite());
    }

    #[test]
    fn infinity_in_w_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        assert!(geom
            .inner_product(&[f32::INFINITY, 1.0], &[1.0, 2.0])
            .is_err());
        assert!(geom
            .inner_product(&[f32::NEG_INFINITY, 1.0], &[1.0, 2.0])
            .is_err());
    }

    #[test]
    fn infinity_in_h_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        assert!(geom
            .inner_product(&[1.0, 2.0], &[1.0, f32::INFINITY])
            .is_err());
        assert!(geom
            .inner_product(&[1.0, 2.0], &[f32::NEG_INFINITY, 1.0])
            .is_err());
    }

    #[test]
    fn gram_vec_infinity_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        assert!(geom.gram_vec(&[f32::INFINITY, 1.0]).is_err());
    }

    #[test]
    fn transform_infinity_rejected() {
        let u = tiny_unembedding();
        let geom = CausalGeometry::from_unembedding(&u, 1e-6);

        assert!(geom.transform(&u, &[1.0, f32::INFINITY]).is_err());
    }

    // -----------------------------------------------------------------------
    // Security regression tests (Issues 25, 26, 33)
    // -----------------------------------------------------------------------

    /// Issue #25 (S-4): from_raw_gram must reject NaN entries.
    #[test]
    fn sec_from_raw_gram_rejects_nan() {
        let mut gram = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        gram[1] = f32::NAN;

        let result = CausalGeometry::from_raw_gram(gram, 2);
        assert!(result.is_err(), "from_raw_gram must reject NaN entry");
    }

    /// Issue #25 (S-4): from_raw_gram must reject Infinity entries.
    #[test]
    fn sec_from_raw_gram_rejects_infinity() {
        let mut gram = vec![1.0, 0.0, 0.0, 1.0];
        gram[0] = f32::INFINITY;

        let result = CausalGeometry::from_raw_gram(gram, 2);
        assert!(result.is_err(), "from_raw_gram must reject Infinity entry");
    }

    /// Issue #25 (S-4): from_raw_gram must reject asymmetric matrices.
    #[test]
    fn sec_from_raw_gram_rejects_asymmetric() {
        // Φ[0,1] = 0.5, Φ[1,0] = -0.5 → asymmetric
        let gram = vec![1.0, 0.5, -0.5, 1.0];

        let result = CausalGeometry::from_raw_gram(gram, 2);
        assert!(result.is_err(), "from_raw_gram must reject asymmetric matrix");
    }

    /// Issue #26 (S-5): geometry_hash must include epsilon.
    ///
    /// Two geometries from the same gram data but different epsilon values
    /// should produce different hashes.
    #[test]
    fn sec_geometry_hash_includes_epsilon() {
        let u = tiny_unembedding();
        let geom_a = CausalGeometry::from_unembedding(&u, 1e-6);
        let geom_b = CausalGeometry::from_unembedding(&u, 1e-2);

        let hash_a = geom_a.geometry_hash();
        let hash_b = geom_b.geometry_hash();

        assert_ne!(hash_a, hash_b, "different epsilon → different geometry_hash");
    }

    /// Issue #33 (S-12): is_positive_definite should detect degenerate matrices
    /// with large trace.
    ///
    /// Matrix [[1000, 0], [0, 0]] has trace=1000 but is rank-deficient.
    #[test]
    fn sec_rank_check_detects_degenerate_with_large_trace() {
        let gram = vec![1000.0, 0.0, 0.0, 0.0]; // rank 1, trace 1000
        let geom = CausalGeometry::from_raw_gram(gram, 2).unwrap();

        let pd = geom.is_positive_definite();
        assert!(!pd, "rank-1 matrix with large trace should not be PD");
    }

    // --- Value dimensionality (manifold collapse) tests ---

    #[test]
    fn dim_eff_identity_metric() {
        // Φ = I, orthogonal probes → G_W = I_k → all eigenvalues = 1 → dim_eff = k
        let geom = CausalGeometry::from_raw_gram(vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ], 4).unwrap();

        let w1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let w2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0];
        let w3: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
        let probes: Vec<&[f32]> = vec![&w1, &w2, &w3];

        let proj = geom.value_projected_gram(&probes).unwrap();
        assert_eq!(proj.k, 3);
        assert!((proj.dim_eff - 3.0).abs() < 0.01, "expected dim_eff ≈ 3, got {}", proj.dim_eff);
    }

    #[test]
    fn dim_eff_collapsed_metric() {
        // Φ = vvᵀ (rank 1), probes aligned with v → dim_eff ≈ 1
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let mut gram = vec![0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                gram[i * 4 + j] = v[i] * v[j];
            }
        }
        // Add tiny epsilon for PD
        for i in 0..4 {
            gram[i * 4 + i] += 1e-6;
        }
        let geom = CausalGeometry::from_raw_gram(gram, 4).unwrap();

        // Three probes, all partially aligned with v
        let w1: Vec<f32> = vec![1.0, 0.1, 0.0, 0.0];
        let w2: Vec<f32> = vec![1.0, 0.0, 0.1, 0.0];
        let w3: Vec<f32> = vec![1.0, 0.0, 0.0, 0.1];
        let probes: Vec<&[f32]> = vec![&w1, &w2, &w3];

        let dim = geom.effective_value_dimensionality(&probes).unwrap();
        assert!(dim < 1.5, "expected dim_eff ≈ 1 for rank-1 Φ, got {dim}");
    }

    #[test]
    fn dim_eff_partial_collapse() {
        // Φ with two strong and one weak direction
        let gram = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.001, 0.0,
            0.0, 0.0, 0.0, 0.001,
        ];
        let geom = CausalGeometry::from_raw_gram(gram, 4).unwrap();

        let w1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0]; // strong
        let w2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0]; // strong
        let w3: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0]; // weak
        let probes: Vec<&[f32]> = vec![&w1, &w2, &w3];

        let proj = geom.value_projected_gram(&probes).unwrap();
        // Two dominant eigenvalues ≈ 1.0, one ≈ 0.001 → dim_eff ≈ 2
        assert!(proj.dim_eff > 1.5 && proj.dim_eff < 2.5,
            "expected dim_eff ≈ 2, got {}", proj.dim_eff);
    }

    #[test]
    fn dim_eff_rejects_dimension_mismatch() {
        let geom = CausalGeometry::from_raw_gram(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let w_bad: Vec<f32> = vec![1.0, 0.0, 0.0]; // dim 3, not 2
        let probes: Vec<&[f32]> = vec![&w_bad];
        assert!(geom.value_projected_gram(&probes).is_err());
    }

    #[test]
    fn dim_eff_rejects_empty_probes() {
        let geom = CausalGeometry::from_raw_gram(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let probes: Vec<&[f32]> = vec![];
        assert!(geom.value_projected_gram(&probes).is_err());
    }

    // --- Value alignment distance tests ---

    #[test]
    fn alignment_distance_identical_models() {
        let gram = vec![2.0, 0.5, 0.5, 3.0];
        let geo_a = CausalGeometry::from_raw_gram(gram.clone(), 2).unwrap();
        let geo_b = CausalGeometry::from_raw_gram(gram, 2).unwrap();

        let w1: Vec<f32> = vec![1.0, 0.0];
        let w2: Vec<f32> = vec![0.0, 1.0];
        let probes: Vec<&[f32]> = vec![&w1, &w2];

        let dist = value_alignment_distance(&geo_a, &geo_b, Some(&probes)).unwrap();
        assert!(dist.global_distance < 1e-6, "identical → d=0, got {}", dist.global_distance);
        assert!(dist.probe_projected_distance.unwrap() < 1e-6);
        for d in dist.per_probe_distances.unwrap() {
            assert!(d < 1e-6);
        }
    }

    #[test]
    fn alignment_distance_orthogonal_change() {
        // Φ_A and Φ_B differ only in the [1,1] entry.
        // Probe is along [1,0] — orthogonal to the change.
        let geo_a = CausalGeometry::from_raw_gram(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let geo_b = CausalGeometry::from_raw_gram(vec![1.0, 0.0, 0.0, 5.0], 2).unwrap();

        let w: Vec<f32> = vec![1.0, 0.0];
        let probes: Vec<&[f32]> = vec![&w];

        let dist = value_alignment_distance(&geo_a, &geo_b, Some(&probes)).unwrap();
        assert!(dist.global_distance > 0.1, "global should detect change");
        assert!(
            dist.probe_projected_distance.unwrap() < 1e-6,
            "probe-projected should be ~0 for orthogonal change, got {}",
            dist.probe_projected_distance.unwrap()
        );
    }

    #[test]
    fn alignment_distance_probe_relevant_change() {
        // Φ_A = I, Φ_B differs specifically in probe direction [1,0]
        let geo_a = CausalGeometry::from_raw_gram(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let geo_b = CausalGeometry::from_raw_gram(vec![5.0, 0.0, 0.0, 1.0], 2).unwrap();

        let w: Vec<f32> = vec![1.0, 0.0];
        let probes: Vec<&[f32]> = vec![&w];

        let dist = value_alignment_distance(&geo_a, &geo_b, Some(&probes)).unwrap();
        assert!(dist.global_distance > 0.1);
        assert!(
            dist.probe_projected_distance.unwrap() > 0.1,
            "probe-projected should detect change along probe direction"
        );
    }

    #[test]
    fn alignment_distance_dimension_mismatch() {
        let geo_a = CausalGeometry::from_raw_gram(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let geo_b = CausalGeometry::from_raw_gram(vec![1.0; 9], 3).unwrap();
        assert!(value_alignment_distance(&geo_a, &geo_b, None).is_err());
    }
}
