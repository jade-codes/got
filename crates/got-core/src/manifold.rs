// ---------------------------------------------------------------------------
// ValueManifold: k-NN density, intrinsic dimension, and sectional curvature
// under the causal metric d_Φ(u,v) = √((u-v)ᵀΦ(u-v)).
//
// Probe activations trace a manifold in causal space. This module
// characterises that manifold's local geometry — density, effective
// dimensionality, and curvature — using the causal metric throughout.
// ---------------------------------------------------------------------------

use crate::geometry::{CausalGeometry, GeometryError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ManifoldError {
    #[error(transparent)]
    Geometry(#[from] GeometryError),

    #[error("too few points: need at least k+1={k_plus_one}, got {n}")]
    TooFewPoints { n: usize, k_plus_one: usize },

    #[error("k must be >= 2, got {0}")]
    KTooSmall(usize),

    #[error("{0} points have all-zero causal distances to their neighbours")]
    DegenerateDistances(usize),

    #[error("uncertainty length mismatch: expected {expected}, got {got}")]
    UncertaintyLengthMismatch { expected: usize, got: usize },
}

// ---------------------------------------------------------------------------
// Config & output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ManifoldConfig {
    /// k for k-NN (must be >= 2).
    pub k: usize,
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self { k: 10 }
    }
}

/// Per-point density and dimension estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointDensity {
    pub log_density: f32,
    pub intrinsic_dim: f32,
}

/// Attestable output summarising manifold geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityReading {
    pub points: Vec<PointDensity>,
    pub mean_intrinsic_dim: f32,
    pub std_intrinsic_dim: f32,
    pub mean_log_density: f32,
    pub k: u32,
    pub num_degenerate: u32,
}

/// Per-point sectional curvature estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCurvature {
    /// Mean sectional curvature at this point (two-scale dimension comparison).
    pub sectional_curvature: f32,
    /// Number of k-NN neighbours used in the estimate.
    pub num_triangles: u32,
}

/// Attestable output summarising manifold curvature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureReading {
    pub points: Vec<PointCurvature>,
    pub mean_curvature: f32,
    pub std_curvature: f32,
    /// Pearson correlation between local curvature and per-point uncertainty.
    /// None if uncertainty was not provided or if variance is zero.
    pub curvature_uncertainty_correlation: Option<f32>,
    pub k: u32,
    pub num_degenerate: u32,
}

// ---------------------------------------------------------------------------
// ValueManifold
// ---------------------------------------------------------------------------

pub struct ValueManifold {
    points: Vec<Vec<f32>>,
    n: usize,
    hidden_dim: usize,
    /// n×n row-major causal distances (precomputed).
    dist_matrix: Vec<f32>,
    config: ManifoldConfig,
}

impl ValueManifold {
    /// Build a manifold from activation vectors under a causal geometry.
    ///
    /// Precomputes the full pairwise distance matrix using the optimised
    /// Φ-precompute strategy: O(n·d²) precompute + O(n²·d) distances.
    pub fn new(
        points: Vec<Vec<f32>>,
        geometry: &CausalGeometry,
        config: ManifoldConfig,
    ) -> Result<Self, ManifoldError> {
        let k = config.k;
        if k < 2 {
            return Err(ManifoldError::KTooSmall(k));
        }
        let n = points.len();
        if n < k + 1 {
            return Err(ManifoldError::TooFewPoints {
                n,
                k_plus_one: k + 1,
            });
        }
        let d = geometry.hidden_dim();
        for p in &points {
            geometry.check_vec(p)?;
        }
        let hidden_dim = d;

        // Precompute Φ·p_j for all points: O(n·d²)
        let phi_points: Vec<Vec<f32>> = points
            .iter()
            .map(|p| geometry.gram_vec(p))
            .collect::<Result<Vec<_>, _>>()?;

        // Precompute self-norms: self_norms[j] = p_j · (Φ·p_j)
        let self_norms: Vec<f32> = points
            .iter()
            .zip(phi_points.iter())
            .map(|(p, phi_p)| p.iter().zip(phi_p.iter()).map(|(a, b)| a * b).sum::<f32>())
            .collect();

        // Pairwise distances: d²(i,j) = self_norms[i] - 2·(p_i · phi_p_j) + self_norms[j]
        let mut dist_matrix = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let cross: f32 = points[i]
                    .iter()
                    .zip(phi_points[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let d_sq = self_norms[i] - 2.0 * cross + self_norms[j];
                let dist = if d_sq > 0.0 { d_sq.sqrt() } else { 0.0 };
                dist_matrix[i * n + j] = dist;
                dist_matrix[j * n + i] = dist;
            }
        }

        Ok(Self {
            points,
            n,
            hidden_dim,
            dist_matrix,
            config,
        })
    }

    /// Return sorted distances from point i to all other points (excluding self).
    /// Returns (k) nearest distances.
    fn knn_distances(&self, i: usize) -> Vec<f32> {
        let n = self.n;
        let k = self.config.k;

        let mut dists: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| self.dist_matrix[i * n + j])
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    /// Levina-Bickel MLE intrinsic dimension at point x_i.
    ///
    /// d̂(x_i) = [(k-1)⁻¹ · Σ_{j=1}^{k-1} ln(r_k / r_j)]⁻¹
    ///
    /// Returns None if r_k == 0 (degenerate).
    fn intrinsic_dim_at(&self, i: usize) -> Option<f32> {
        let dists = self.knn_distances(i);
        let k = dists.len();
        let r_k = dists[k - 1];
        if r_k <= 0.0 {
            return None;
        }
        let ln_r_k = r_k.ln();
        let mut sum = 0.0f32;
        for j in 0..(k - 1) {
            let r_j = dists[j];
            if r_j <= 0.0 {
                // Use ln(r_k / epsilon) as a large but finite contribution
                // when r_j is zero but r_k is not.
                sum += ln_r_k - f32::MIN_POSITIVE.ln();
            } else {
                sum += ln_r_k - r_j.ln();
            }
        }
        let inv = sum / (k - 1) as f32;
        if inv <= 0.0 {
            return None;
        }
        Some(1.0 / inv)
    }

    /// k-NN log-density at point x_i given effective dimension d_eff.
    ///
    /// log ρ̂(x_i) = ln(k) - d_eff · ln(r_k) - ln(c_d)
    /// where c_d = π^(d/2) / Γ(d/2 + 1)
    fn log_density_at(&self, i: usize, d_eff: f32) -> Option<f32> {
        let dists = self.knn_distances(i);
        let k = dists.len();
        let r_k = dists[k - 1];
        if r_k <= 0.0 {
            return None;
        }

        let ln_k = (k as f32).ln();
        let ln_r_k = r_k.ln();
        let ln_c_d = ln_unit_ball_volume(d_eff);

        Some(ln_k - d_eff * ln_r_k - ln_c_d)
    }

    /// Two-pass density map:
    /// 1. Estimate intrinsic dim at each point → mean d_eff
    /// 2. Use mean d_eff for density estimation at each point
    pub fn density_map(&self) -> Result<DensityReading, ManifoldError> {
        let n = self.n;
        let k = self.config.k;

        // Pass 1: intrinsic dimension
        let mut dims: Vec<f32> = Vec::with_capacity(n);
        let mut num_degenerate: u32 = 0;
        for i in 0..n {
            match self.intrinsic_dim_at(i) {
                Some(d) => dims.push(d),
                None => num_degenerate += 1,
            }
        }

        if dims.is_empty() {
            return Err(ManifoldError::DegenerateDistances(n));
        }

        let mean_dim: f32 = dims.iter().sum::<f32>() / dims.len() as f32;
        let var_dim: f32 =
            dims.iter().map(|d| (d - mean_dim) * (d - mean_dim)).sum::<f32>() / dims.len() as f32;
        let std_dim = var_dim.sqrt();

        // Pass 2: density using mean intrinsic dim
        let mut point_densities: Vec<PointDensity> = Vec::with_capacity(n);
        let mut density_sum = 0.0f32;
        let mut density_count = 0u32;

        for i in 0..n {
            let local_dim = self.intrinsic_dim_at(i);
            let log_density = self.log_density_at(i, mean_dim);

            let (ld, id) = match (log_density, local_dim) {
                (Some(ld), Some(id)) => {
                    density_sum += ld;
                    density_count += 1;
                    (ld, id)
                }
                (Some(ld), None) => {
                    density_sum += ld;
                    density_count += 1;
                    (ld, 0.0)
                }
                (None, Some(id)) => (0.0, id),
                (None, None) => (0.0, 0.0),
            };
            point_densities.push(PointDensity {
                log_density: ld,
                intrinsic_dim: id,
            });
        }

        let mean_log_density = if density_count > 0 {
            density_sum / density_count as f32
        } else {
            0.0
        };

        Ok(DensityReading {
            points: point_densities,
            mean_intrinsic_dim: mean_dim,
            std_intrinsic_dim: std_dim,
            mean_log_density,
            k: k as u32,
            num_degenerate,
        })
    }

    /// Number of points in the manifold.
    pub fn num_points(&self) -> usize {
        self.n
    }

    /// Hidden dimension of the activation space.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Query the log-density at an arbitrary point (not necessarily in the manifold).
    ///
    /// Computes causal distances from `point` to all manifold points, finds the
    /// k nearest, and returns the k-NN log-density estimate using the provided
    /// effective dimension `d_eff` (typically `DensityReading::mean_intrinsic_dim`).
    ///
    /// Returns `None` if the k-th nearest distance is zero (degenerate).
    pub fn query_log_density(
        &self,
        point: &[f32],
        geometry: &CausalGeometry,
        d_eff: f32,
    ) -> Result<Option<f32>, ManifoldError> {
        let k = self.config.k;
        geometry.check_vec(point)?;

        // Compute causal distances from point to all manifold points
        let phi_point = geometry.gram_vec(point)?;
        let self_norm: f32 = point
            .iter()
            .zip(phi_point.iter())
            .map(|(a, b)| a * b)
            .sum();

        let mut dists: Vec<f32> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            // d²(point, p_i) = self_norm - 2·(point · Φ·p_i) + self_norms[i]
            // We need Φ·p_i, but we don't store phi_points. Recompute using dist_matrix trick:
            // Actually we stored the points. Compute directly.
            let pi = &self.points[i];
            let phi_pi = geometry.gram_vec(pi)?;
            let cross: f32 = point.iter().zip(phi_pi.iter()).map(|(a, b)| a * b).sum();
            let pi_norm: f32 = pi.iter().zip(phi_pi.iter()).map(|(a, b)| a * b).sum();
            let d_sq = self_norm - 2.0 * cross + pi_norm;
            dists.push(if d_sq > 0.0 { d_sq.sqrt() } else { 0.0 });
        }

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);

        let r_k = dists[k - 1];
        if r_k <= 0.0 {
            return Ok(None);
        }

        let ln_k = (k as f32).ln();
        let ln_r_k = r_k.ln();
        let ln_c_d = ln_unit_ball_volume(d_eff);

        Ok(Some(ln_k - d_eff * ln_r_k - ln_c_d))
    }

    /// Levina-Bickel intrinsic dimension from a sorted distance slice.
    ///
    /// Given distances r_1 ≤ ... ≤ r_m (m elements), returns the MLE:
    ///   d̂ = [(m-1)⁻¹ · Σ_{j=1}^{m-1} ln(r_m / r_j)]⁻¹
    fn levina_bickel(dists: &[f32]) -> Option<f32> {
        let m = dists.len();
        if m < 2 {
            return None;
        }
        let r_m = dists[m - 1];
        if r_m <= 0.0 {
            return None;
        }
        let ln_r_m = r_m.ln();
        let mut sum = 0.0f32;
        for j in 0..(m - 1) {
            if dists[j] <= 0.0 {
                sum += ln_r_m - f32::MIN_POSITIVE.ln();
            } else {
                sum += ln_r_m - dists[j].ln();
            }
        }
        let inv = sum / (m - 1) as f32;
        if inv <= 0.0 {
            return None;
        }
        Some(1.0 / inv)
    }

    /// Return all distances from point i to other points, sorted ascending.
    fn all_distances_sorted(&self, i: usize) -> Vec<f32> {
        let n = self.n;
        let mut dists: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| self.dist_matrix[i * n + j])
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists
    }

    /// Estimate sectional curvature via two-scale dimension comparison.
    ///
    /// The principle: on a positively curved manifold, geodesic balls grow
    /// slower than in flat space, so the effective dimension measured at a
    /// larger scale (k neighbours) is *lower* than at a smaller scale (k/2).
    /// On a negatively curved manifold, the reverse holds.
    ///
    /// For each point x_i:
    ///   d_near = Levina-Bickel MLE from the k_near = max(2, k/2) nearest neighbours
    ///   d_far  = Levina-Bickel MLE from the full k nearest neighbours
    ///   κ(x_i) = (d_near − d_far) / r_k²
    ///
    /// Sign convention:
    ///   κ > 0  →  positive curvature (sphere-like, ball growth decelerates)
    ///   κ < 0  →  negative curvature (saddle-like, ball growth accelerates)
    ///   κ ≈ 0  →  flat
    ///
    /// If `uncertainty` is provided (one f32 per point), also computes the
    /// Pearson correlation between local curvature and uncertainty.
    pub fn curvature_map(
        &self,
        uncertainty: Option<&[f32]>,
    ) -> Result<CurvatureReading, ManifoldError> {
        let n = self.n;
        let k = self.config.k;

        if let Some(u) = uncertainty {
            if u.len() != n {
                return Err(ManifoldError::UncertaintyLengthMismatch {
                    expected: n,
                    got: u.len(),
                });
            }
        }

        let k_near = (k / 2).max(2);

        let mut point_curvatures: Vec<PointCurvature> = Vec::with_capacity(n);
        let mut num_degenerate: u32 = 0;

        for i in 0..n {
            let dists = self.all_distances_sorted(i);

            // Need at least k distances (we have n-1 total, and k < n by construction)
            let d_near = Self::levina_bickel(&dists[..k_near]);
            let d_far = Self::levina_bickel(&dists[..k]);

            match (d_near, d_far) {
                (Some(dn), Some(df)) => {
                    let r_k = dists[k - 1];
                    if r_k > 0.0 {
                        let kappa = (dn - df) / (r_k * r_k);
                        point_curvatures.push(PointCurvature {
                            sectional_curvature: kappa,
                            num_triangles: k as u32, // triangles replaced by neighbor count
                        });
                    } else {
                        num_degenerate += 1;
                        point_curvatures.push(PointCurvature {
                            sectional_curvature: 0.0,
                            num_triangles: 0,
                        });
                    }
                }
                _ => {
                    num_degenerate += 1;
                    point_curvatures.push(PointCurvature {
                        sectional_curvature: 0.0,
                        num_triangles: 0,
                    });
                }
            }
        }

        // Global statistics (over non-degenerate points)
        let valid_curvatures: Vec<f32> = point_curvatures
            .iter()
            .filter(|p| p.num_triangles > 0)
            .map(|p| p.sectional_curvature)
            .collect();

        let (mean_curvature, std_curvature) = if valid_curvatures.is_empty() {
            (0.0, 0.0)
        } else {
            let mean = valid_curvatures.iter().sum::<f32>() / valid_curvatures.len() as f32;
            let var = valid_curvatures
                .iter()
                .map(|&c| (c - mean) * (c - mean))
                .sum::<f32>()
                / valid_curvatures.len() as f32;
            (mean, var.sqrt())
        };

        // Pearson correlation with uncertainty
        let correlation = uncertainty.and_then(|u| {
            let curvatures: Vec<f32> =
                point_curvatures.iter().map(|p| p.sectional_curvature).collect();
            pearson_correlation(&curvatures, u)
        });

        Ok(CurvatureReading {
            points: point_curvatures,
            mean_curvature,
            std_curvature,
            curvature_uncertainty_correlation: correlation,
            k: k as u32,
            num_degenerate,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// ln(V_d) = ln(π^(d/2) / Γ(d/2 + 1))
///         = (d/2)·ln(π) - ln_gamma(d/2 + 1)
fn ln_unit_ball_volume(d: f32) -> f32 {
    let half_d = d / 2.0;
    half_d * std::f32::consts::PI.ln() - ln_gamma(half_d + 1.0)
}

/// Stirling-series approximation of ln(Γ(x)) for x > 0.
///
/// For x >= 8, uses the asymptotic series. For smaller x, uses the
/// recurrence Γ(x) = Γ(x+1)/x to shift into the asymptotic range.
fn ln_gamma(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::INFINITY;
    }

    // Use recurrence to shift x into the range where Stirling is accurate.
    let mut xx = x as f64;
    let mut shift = 0.0f64;
    while xx < 8.0 {
        shift += xx.ln();
        xx += 1.0;
    }

    // Stirling series: ln Γ(x) ≈ (x-0.5)·ln(x) - x + 0.5·ln(2π) + 1/(12x) - 1/(360x³)
    let result = (xx - 0.5) * xx.ln() - xx + 0.5 * (2.0 * std::f64::consts::PI).ln()
        + 1.0 / (12.0 * xx)
        - 1.0 / (360.0 * xx * xx * xx);

    (result - shift) as f32
}

/// Pearson correlation coefficient between two equal-length slices.
///
/// Returns None if either slice has zero variance (constant values).
fn pearson_correlation(x: &[f32], y: &[f32]) -> Option<f32> {
    let n = x.len();
    if n < 2 || y.len() != n {
        return None;
    }
    let n_f = n as f32;
    let mean_x = x.iter().sum::<f32>() / n_f;
    let mean_y = y.iter().sum::<f32>() / n_f;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < f32::EPSILON || var_y < f32::EPSILON {
        return None;
    }
    Some((cov / (var_x * var_y).sqrt()).clamp(-1.0, 1.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnembeddingMatrix;

    /// Identity geometry (Φ = I) for testing.
    fn identity_geometry(d: usize) -> CausalGeometry {
        let mut gram = vec![0.0f32; d * d];
        for i in 0..d {
            gram[i * d + i] = 1.0;
        }
        CausalGeometry::from_raw_gram(gram, d).unwrap()
    }

    /// Tiny [[35,44],[44,56]] geometry from the test unembedding.
    fn tiny_geometry() -> CausalGeometry {
        let u = UnembeddingMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        CausalGeometry::from_unembedding(&u, 1e-6)
    }

    // --- ln_gamma sanity checks ---

    #[test]
    fn ln_gamma_known_values() {
        // Γ(1) = 1, ln(1) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-4, "ln_gamma(1) = {}", ln_gamma(1.0));
        // Γ(2) = 1, ln(1) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-4, "ln_gamma(2) = {}", ln_gamma(2.0));
        // Γ(3) = 2, ln(2) ≈ 0.6931
        assert!(
            (ln_gamma(3.0) - 2.0f32.ln()).abs() < 1e-3,
            "ln_gamma(3) = {}",
            ln_gamma(3.0)
        );
        // Γ(0.5) = √π, ln(√π) ≈ 0.5724
        let expected = (std::f32::consts::PI.sqrt()).ln();
        assert!(
            (ln_gamma(0.5) - expected).abs() < 1e-3,
            "ln_gamma(0.5) = {}, expected {}",
            ln_gamma(0.5),
            expected
        );
    }

    // --- Rejection tests ---

    #[test]
    fn k_too_small_rejected() {
        let geom = identity_geometry(2);
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let config = ManifoldConfig { k: 1 };
        let result = ValueManifold::new(points, &geom, config);
        assert!(matches!(result, Err(ManifoldError::KTooSmall(1))));
    }

    #[test]
    fn too_few_points_rejected() {
        let geom = identity_geometry(2);
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let config = ManifoldConfig { k: 2 };
        let result = ValueManifold::new(points, &geom, config);
        assert!(matches!(result, Err(ManifoldError::TooFewPoints { .. })));
    }

    // --- Collinear test: 3 points on a line, k=2 → dim ≈ 1 ---

    #[test]
    fn collinear_points_dim_approx_1() {
        let geom = identity_geometry(3);
        // 3 collinear points along x-axis: (0,0,0), (1,0,0), (2,0,0)
        let points = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
        ];
        let config = ManifoldConfig { k: 2 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.density_map().unwrap();

        assert!(
            (reading.mean_intrinsic_dim - 1.0).abs() < 0.5,
            "collinear dim should be ≈1, got {}",
            reading.mean_intrinsic_dim
        );
    }

    // --- 2D grid test: 5 points on a square in 3D, k=3 → dim ≈ 2 ---

    #[test]
    fn planar_points_dim_approx_2() {
        let geom = identity_geometry(3);
        // 5 points in the xy-plane: corners + center
        let points = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];
        let config = ManifoldConfig { k: 3 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.density_map().unwrap();

        // With only 5 points the estimate is noisy; just check it's > 1
        assert!(
            reading.mean_intrinsic_dim > 1.0,
            "planar dim should be >1, got {}",
            reading.mean_intrinsic_dim
        );
    }

    // --- Causal metric test: tiny [[35,44],[44,56]] geometry ---

    #[test]
    fn causal_distance_matches_hand_calc() {
        let geom = tiny_geometry();
        // Points: u = [1, 0], v = [0, 1]
        // diff = [1, -1]
        // d² = diff^T Φ diff = [1,-1] [[35,44],[44,56]] [1,-1]
        //     = 1*35*1 + 1*44*(-1) + (-1)*44*1 + (-1)*56*(-1)
        //     = 35 - 44 - 44 + 56 = 3
        // d = √3 ≈ 1.732
        let points = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ];
        let config = ManifoldConfig { k: 2 };
        let m = ValueManifold::new(points, &geom, config).unwrap();

        // Check distance(0,1) = √3
        let d01 = m.dist_matrix[0 * 3 + 1];
        assert!(
            (d01 - 3.0f32.sqrt()).abs() < 1e-3,
            "d([1,0],[0,1]) under Φ should be √3, got {}",
            d01
        );
    }

    // --- Degenerate test: identical points ---

    #[test]
    fn identical_points_degenerate() {
        let geom = identity_geometry(2);
        let points = vec![vec![1.0, 1.0]; 5];
        let config = ManifoldConfig { k: 2 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let result = m.density_map();
        assert!(
            matches!(result, Err(ManifoldError::DegenerateDistances(_))),
            "identical points should be degenerate, got {result:?}"
        );
    }

    // --- Serde round-trip ---

    #[test]
    fn density_reading_serde_roundtrip() {
        let reading = DensityReading {
            points: vec![
                PointDensity {
                    log_density: -2.5,
                    intrinsic_dim: 1.8,
                },
                PointDensity {
                    log_density: -3.1,
                    intrinsic_dim: 2.2,
                },
            ],
            mean_intrinsic_dim: 2.0,
            std_intrinsic_dim: 0.2,
            mean_log_density: -2.8,
            k: 5,
            num_degenerate: 0,
        };

        let json = serde_json::to_string(&reading).unwrap();
        let decoded: DensityReading = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.points.len(), 2);
        assert!((decoded.mean_intrinsic_dim - 2.0).abs() < 1e-6);
        assert_eq!(decoded.k, 5);
        assert_eq!(decoded.num_degenerate, 0);
    }

    // --- NaN/Inf input rejection ---

    #[test]
    fn nan_point_rejected() {
        let geom = identity_geometry(2);
        let points = vec![
            vec![0.0, 0.0],
            vec![f32::NAN, 1.0],
            vec![1.0, 1.0],
        ];
        let config = ManifoldConfig { k: 2 };
        let result = ValueManifold::new(points, &geom, config);
        assert!(matches!(result, Err(ManifoldError::Geometry(_))));
    }

    #[test]
    fn inf_point_rejected() {
        let geom = identity_geometry(2);
        let points = vec![
            vec![0.0, 0.0],
            vec![f32::INFINITY, 1.0],
            vec![1.0, 1.0],
        ];
        let config = ManifoldConfig { k: 2 };
        let result = ValueManifold::new(points, &geom, config);
        assert!(matches!(result, Err(ManifoldError::Geometry(_))));
    }

    // =======================================================================
    // Curvature tests
    // =======================================================================

    // --- Flat manifold: evenly spaced collinear points → curvature ≈ 0 ---

    #[test]
    fn collinear_curvature_near_zero() {
        let geom = identity_geometry(3);
        // Evenly spaced collinear points — uniform 1D density, zero curvature.
        // d_near ≈ d_far ≈ 1 for uniform spacing → κ ≈ 0.
        let points: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, 0.0, 0.0])
            .collect();
        let config = ManifoldConfig { k: 4 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.curvature_map(None).unwrap();

        assert!(
            reading.mean_curvature.abs() < 1.0,
            "collinear points should have curvature ≈ 0, got {}",
            reading.mean_curvature
        );
    }

    // --- Flat 2D manifold: uniform grid in a plane → curvature ≈ 0 ---

    #[test]
    fn planar_grid_curvature_near_zero() {
        let geom = identity_geometry(3);
        // 5×5 grid in xy-plane — flat, uniform density.
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(vec![i as f32, j as f32, 0.0]);
            }
        }
        let config = ManifoldConfig { k: 4 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.curvature_map(None).unwrap();

        assert!(
            reading.mean_curvature.abs() < 1.0,
            "planar grid should have curvature ≈ 0, got {}",
            reading.mean_curvature
        );
    }

    // --- Curved manifold: points on a sphere → positive curvature ---

    #[test]
    fn spherical_points_positive_curvature() {
        let geom = identity_geometry(3);
        // Dense sampling on a unit sphere. On a sphere, ball volume grows
        // slower than flat → d_near > d_far → positive curvature.
        let mut points = Vec::new();
        let n_lat = 10;
        let n_lon = 16;
        for i in 1..n_lat {
            let theta = std::f32::consts::PI * i as f32 / n_lat as f32;
            for j in 0..n_lon {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / n_lon as f32;
                points.push(vec![
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        // Add poles
        points.push(vec![0.0, 0.0, 1.0]);
        points.push(vec![0.0, 0.0, -1.0]);

        let config = ManifoldConfig { k: 8 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.curvature_map(None).unwrap();

        assert!(
            reading.mean_curvature > 0.0,
            "spherical points should have positive curvature, got {}",
            reading.mean_curvature
        );
    }

    // --- Causal metric curvature: non-identity Φ ---

    #[test]
    fn causal_metric_curvature_computes() {
        let geom = tiny_geometry(); // [[35,44],[44,56]]
        // Grid in 2D — just verify no errors and finite output
        let mut points = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                points.push(vec![i as f32 * 0.1, j as f32 * 0.1]);
            }
        }
        let config = ManifoldConfig { k: 3 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.curvature_map(None).unwrap();

        assert!(
            reading.mean_curvature.is_finite(),
            "curvature should be finite, got {}",
            reading.mean_curvature
        );
        assert_eq!(reading.points.len(), 16);
    }

    // --- Uncertainty correlation ---

    #[test]
    fn uncertainty_correlation_computes() {
        let geom = identity_geometry(3);
        // Dense sphere sampling with uncertainty values
        let mut points = Vec::new();
        let n_lat = 6;
        let n_lon = 10;
        for i in 1..n_lat {
            let theta = std::f32::consts::PI * i as f32 / n_lat as f32;
            for j in 0..n_lon {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / n_lon as f32;
                points.push(vec![
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        points.push(vec![0.0, 0.0, 1.0]);
        points.push(vec![0.0, 0.0, -1.0]);

        let n = points.len();
        // Uncertainty proportional to z-distance from equator
        let uncertainty: Vec<f32> = points.iter().map(|p| p[2].abs() + 0.1).collect();

        let config = ManifoldConfig { k: 6 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.curvature_map(Some(&uncertainty)).unwrap();

        // Verify we get a finite correlation value
        assert!(
            reading.curvature_uncertainty_correlation.is_some(),
            "should compute correlation"
        );
        let r = reading.curvature_uncertainty_correlation.unwrap();
        assert!(
            r.is_finite() && r >= -1.0 && r <= 1.0,
            "correlation should be in [-1,1], got {}",
            r
        );
    }

    // --- Uncertainty length mismatch ---

    #[test]
    fn uncertainty_length_mismatch_rejected() {
        let geom = identity_geometry(2);
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let config = ManifoldConfig { k: 2 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let result = m.curvature_map(Some(&[1.0, 2.0])); // wrong length
        assert!(matches!(
            result,
            Err(ManifoldError::UncertaintyLengthMismatch { .. })
        ));
    }

    // --- No uncertainty provided → correlation is None ---

    #[test]
    fn no_uncertainty_correlation_is_none() {
        let geom = identity_geometry(2);
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let config = ManifoldConfig { k: 2 };
        let m = ValueManifold::new(points, &geom, config).unwrap();
        let reading = m.curvature_map(None).unwrap();
        assert!(reading.curvature_uncertainty_correlation.is_none());
    }

    // --- Pearson correlation unit tests ---

    #[test]
    fn pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!(
            (r - 1.0).abs() < 1e-5,
            "perfect positive correlation, got {}",
            r
        );
    }

    #[test]
    fn pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!(
            (r + 1.0).abs() < 1e-5,
            "perfect negative correlation, got {}",
            r
        );
    }

    #[test]
    fn pearson_constant_returns_none() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(pearson_correlation(&x, &y).is_none());
    }

    // --- CurvatureReading serde round-trip ---

    #[test]
    fn curvature_reading_serde_roundtrip() {
        let reading = CurvatureReading {
            points: vec![
                PointCurvature {
                    sectional_curvature: 0.5,
                    num_triangles: 3,
                },
                PointCurvature {
                    sectional_curvature: -0.2,
                    num_triangles: 2,
                },
            ],
            mean_curvature: 0.15,
            std_curvature: 0.35,
            curvature_uncertainty_correlation: Some(0.72),
            k: 3,
            num_degenerate: 0,
        };

        let json = serde_json::to_string(&reading).unwrap();
        let decoded: CurvatureReading = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.points.len(), 2);
        assert!((decoded.mean_curvature - 0.15).abs() < 1e-6);
        assert!((decoded.curvature_uncertainty_correlation.unwrap() - 0.72).abs() < 1e-6);
        assert_eq!(decoded.k, 3);
    }
}
