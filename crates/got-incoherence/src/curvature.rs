// ---------------------------------------------------------------------------
// Curvature estimation for value subspaces.
//
// Conjecture 2 claims regions of high curvature in the value manifold
// correspond to topics where humans report greater moral uncertainty.
//
// Since we have discrete points (value term embeddings), not a continuous
// manifold, we estimate curvature via:
//
// 1. **Menger curvature** of point triples: for three value terms (A, B, C),
//    the reciprocal of the circumradius of the triangle they form.
//    High Menger curvature = the three terms are close but non-collinear,
//    creating a "bent" region where interpolation is geometrically unstable.
//
// 2. **Local dimensionality variation**: for each term, the participation
//    ratio of its k nearest neighbours. High local PR = the neighbourhood
//    fans out in many directions (saddle-like). Low local PR = the
//    neighbourhood is flat or ridge-like.
//
// 3. **Angle deficit**: for each term, the sum of angles in all triangles
//    containing it, compared to the flat (Euclidean) expectation.
//    Positive deficit = positive curvature (sphere-like).
//    Negative deficit = negative curvature (saddle-like).
//
// These are the geometric quantities that Conjecture 2 predicts will
// correlate with human moral uncertainty data.
// ---------------------------------------------------------------------------

use got_core::geometry::CausalGeometry;
use serde::{Deserialize, Serialize};

use crate::coherence::{causal_cosine, causal_distance};
use crate::embeddings::ResolvedValue;
use crate::IncoherenceError;

/// Curvature analysis of a value subspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureAnalysis {
    /// Per-term local curvature estimates.
    pub term_curvatures: Vec<TermCurvature>,
    /// All triple curvatures, sorted by Menger curvature descending.
    pub triple_curvatures: Vec<TripleCurvature>,
    /// Global statistics.
    pub mean_menger: f32,
    pub max_menger: f32,
    pub mean_local_pr: f32,
}

/// Local curvature estimate for a single value term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermCurvature {
    pub term: String,
    /// Mean Menger curvature across all triples containing this term.
    pub mean_menger_curvature: f32,
    /// Max Menger curvature across triples containing this term.
    pub max_menger_curvature: f32,
    /// Local participation ratio (PR of k nearest neighbours).
    pub local_pr: f32,
    /// Angle deficit: sum of angles at this vertex minus flat expectation.
    /// Positive = sphere-like (convergent), negative = saddle-like (divergent).
    pub angle_deficit: f32,
    /// Number of neighbours used for local PR.
    pub k_neighbours: usize,
}

/// Menger curvature of a triple of value terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleCurvature {
    pub term_a: String,
    pub term_b: String,
    pub term_c: String,
    /// Menger curvature: 1/R where R is the circumradius.
    /// Higher = more "bent" region.
    pub menger_curvature: f32,
    /// Area of the triangle (half the cross-product magnitude).
    pub triangle_area: f32,
    /// Angles at each vertex (radians).
    pub angle_a: f32,
    pub angle_b: f32,
    pub angle_c: f32,
}

/// Compute Menger curvature for a triple of points.
///
/// Menger curvature = 4 * area / (|AB| * |BC| * |CA|)
/// This equals 1/R where R is the circumradius of the triangle.
///
/// Uses causal distances under the geometry.
fn menger_curvature_triple(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    geometry: &CausalGeometry,
) -> Result<(f32, f32, f32, f32, f32), IncoherenceError> {
    let d_ab = causal_distance(a, b, geometry)?;
    let d_bc = causal_distance(b, c, geometry)?;
    let d_ca = causal_distance(c, a, geometry)?;

    // Triangle angles via law of cosines
    let angle_a = triangle_angle(d_ab, d_ca, d_bc);
    let angle_b = triangle_angle(d_ab, d_bc, d_ca);
    let angle_c = triangle_angle(d_ca, d_bc, d_ab);

    // Area via Heron's formula
    let s = (d_ab + d_bc + d_ca) / 2.0;
    let area_sq = s * (s - d_ab) * (s - d_bc) * (s - d_ca);
    let area = if area_sq > 0.0 { area_sq.sqrt() } else { 0.0 };

    // Menger curvature = 4 * area / (|AB| * |BC| * |CA|)
    let product = d_ab * d_bc * d_ca;
    let menger = if product > 1e-10 {
        4.0 * area / product
    } else {
        0.0 // degenerate triangle
    };

    Ok((menger, area, angle_a, angle_b, angle_c))
}

/// Angle at vertex A in triangle with sides AB, AC, and opposite BC.
/// Uses law of cosines: cos(A) = (AB² + AC² - BC²) / (2 * AB * AC)
fn triangle_angle(side_ab: f32, side_ac: f32, side_bc: f32) -> f32 {
    let denom = 2.0 * side_ab * side_ac;
    if denom < 1e-10 {
        return 0.0;
    }
    let cos_a = (side_ab * side_ab + side_ac * side_ac - side_bc * side_bc) / denom;
    cos_a.clamp(-1.0, 1.0).acos()
}

/// Compute curvature analysis for a set of value terms.
///
/// `k` controls how many nearest neighbours are used for local PR estimation.
pub fn analyse_curvature(
    values: &[ResolvedValue],
    geometry: &CausalGeometry,
    k: usize,
) -> Result<CurvatureAnalysis, IncoherenceError> {
    let n = values.len();
    if n < 3 {
        return Err(IncoherenceError::EmptyInput(
            "need at least 3 values for curvature analysis",
        ));
    }

    // Pairwise distances
    let mut distances = vec![0.0f32; n * n];
    let mut cosines = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = causal_distance(
                &values[i].embedding,
                &values[j].embedding,
                geometry,
            )?;
            distances[i * n + j] = d;
            distances[j * n + i] = d;

            let c = causal_cosine(
                &values[i].embedding,
                &values[j].embedding,
                geometry,
            )?;
            cosines[i * n + j] = c;
            cosines[j * n + i] = c;
        }
    }

    // Compute all triple curvatures
    let mut triple_curvatures = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k_idx in (j + 1)..n {
                let (menger, area, angle_a, angle_b, angle_c) = menger_curvature_triple(
                    &values[i].embedding,
                    &values[j].embedding,
                    &values[k_idx].embedding,
                    geometry,
                )?;

                triple_curvatures.push(TripleCurvature {
                    term_a: values[i].normalised.clone(),
                    term_b: values[j].normalised.clone(),
                    term_c: values[k_idx].normalised.clone(),
                    menger_curvature: menger,
                    triangle_area: area,
                    angle_a,
                    angle_b,
                    angle_c,
                });
            }
        }
    }

    // Sort by Menger curvature descending
    triple_curvatures.sort_by(|a, b| {
        b.menger_curvature.partial_cmp(&a.menger_curvature).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Per-term analysis
    let k_actual = k.min(n - 1);
    let mut term_curvatures = Vec::new();

    for i in 0..n {
        // Menger curvatures for triples containing this term
        let term_mengers: Vec<f32> = triple_curvatures.iter()
            .filter(|t| {
                t.term_a == values[i].normalised
                    || t.term_b == values[i].normalised
                    || t.term_c == values[i].normalised
            })
            .map(|t| t.menger_curvature)
            .collect();

        let mean_menger = if term_mengers.is_empty() {
            0.0
        } else {
            term_mengers.iter().sum::<f32>() / term_mengers.len() as f32
        };
        let max_menger = term_mengers.iter().fold(0.0f32, |a, &b| a.max(b));

        // Local PR: PR of k nearest neighbours
        let mut neighbour_dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, distances[i * n + j]))
            .collect();
        neighbour_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let neighbours: Vec<usize> = neighbour_dists.iter()
            .take(k_actual)
            .map(|&(j, _)| j)
            .collect();

        let local_pr = if neighbours.len() >= 2 {
            // Build local cosine matrix for neighbours
            let m = neighbours.len();
            let mut local_cos = vec![0.0f64; m * m];
            for a in 0..m {
                local_cos[a * m + a] = 1.0;
                for b in (a + 1)..m {
                    let c = cosines[neighbours[a] * n + neighbours[b]] as f64;
                    local_cos[a * m + b] = c;
                    local_cos[b * m + a] = c;
                }
            }
            // Eigenvalues
            let mat = faer::Mat::from_fn(m, m, |r, c| local_cos[r * m + c]);
            let eigenvalues = mat.selfadjoint_eigendecomposition(faer::Side::Lower)
                .s().column_vector().try_as_slice().unwrap().to_vec();
            let lambdas: Vec<f64> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();
            let sum: f64 = lambdas.iter().sum();
            let sum_sq: f64 = lambdas.iter().map(|l| l * l).sum();
            if sum_sq > 1e-15 { (sum * sum / sum_sq) as f32 } else { 1.0 }
        } else {
            1.0
        };

        // Angle deficit: sum of angles at this vertex in all triangles
        let angles_at_i: Vec<f32> = triple_curvatures.iter()
            .filter_map(|t| {
                if t.term_a == values[i].normalised { Some(t.angle_a) }
                else if t.term_b == values[i].normalised { Some(t.angle_b) }
                else if t.term_c == values[i].normalised { Some(t.angle_c) }
                else { None }
            })
            .collect();

        // Angle deficit: compare to expected flat angle
        // In flat space, mean angle in a random triangle ≈ π/3.
        // Positive deficit = angles larger than expected (positive curvature).
        let mean_angle = if angles_at_i.is_empty() {
            std::f32::consts::FRAC_PI_3
        } else {
            angles_at_i.iter().sum::<f32>() / angles_at_i.len() as f32
        };
        let angle_deficit = mean_angle - std::f32::consts::FRAC_PI_3;

        term_curvatures.push(TermCurvature {
            term: values[i].normalised.clone(),
            mean_menger_curvature: mean_menger,
            max_menger_curvature: max_menger,
            local_pr,
            angle_deficit,
            k_neighbours: k_actual,
        });
    }

    // Sort by mean Menger curvature descending
    term_curvatures.sort_by(|a, b| {
        b.mean_menger_curvature.partial_cmp(&a.mean_menger_curvature)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Global stats
    let all_mengers: Vec<f32> = triple_curvatures.iter()
        .map(|t| t.menger_curvature)
        .collect();
    let mean_menger = if all_mengers.is_empty() {
        0.0
    } else {
        all_mengers.iter().sum::<f32>() / all_mengers.len() as f32
    };
    let max_menger = all_mengers.iter().fold(0.0f32, |a, &b| a.max(b));
    let mean_local_pr = term_curvatures.iter()
        .map(|t| t.local_pr)
        .sum::<f32>() / term_curvatures.len() as f32;

    Ok(CurvatureAnalysis {
        term_curvatures,
        triple_curvatures,
        mean_menger,
        max_menger,
        mean_local_pr,
    })
}

/// Format curvature analysis for display.
pub fn render_curvature(analysis: &CurvatureAnalysis) -> String {
    let mut out = String::new();

    out.push_str("Curvature Analysis\n");
    out.push_str(&"=".repeat(60));
    out.push('\n');

    out.push_str(&format!(
        "\nGlobal: mean Menger = {:.4}, max Menger = {:.4}, mean local PR = {:.2}\n",
        analysis.mean_menger, analysis.max_menger, analysis.mean_local_pr,
    ));

    out.push_str("\nPer-term curvature (sorted by mean Menger, descending):\n");
    out.push_str(&format!(
        "  {:<20} {:>10} {:>10} {:>10} {:>10}\n",
        "Term", "Mean κ", "Max κ", "Local PR", "Angle Δ"
    ));
    out.push_str(&format!("  {}\n", "-".repeat(62)));

    for tc in &analysis.term_curvatures {
        out.push_str(&format!(
            "  {:<20} {:>10.4} {:>10.4} {:>10.2} {:>+10.4}\n",
            tc.term, tc.mean_menger_curvature, tc.max_menger_curvature,
            tc.local_pr, tc.angle_deficit,
        ));
    }

    // Top 10 highest-curvature triples
    let top_n = 10.min(analysis.triple_curvatures.len());
    out.push_str(&format!("\nTop {} highest-curvature triples:\n", top_n));
    for tc in &analysis.triple_curvatures[..top_n] {
        out.push_str(&format!(
            "  {:<12} {:<12} {:<12}  κ = {:.4}  area = {:.4}  angles = ({:.1}°, {:.1}°, {:.1}°)\n",
            tc.term_a, tc.term_b, tc.term_c,
            tc.menger_curvature, tc.triangle_area,
            tc.angle_a.to_degrees(), tc.angle_b.to_degrees(), tc.angle_c.to_degrees(),
        ));
    }

    // Interpretation
    out.push_str("\nInterpretation:\n");
    out.push_str("  High Menger curvature = terms are close but non-collinear.\n");
    out.push_str("  These are regions where the value landscape is 'bent' —\n");
    out.push_str("  small perturbations change which value dominates.\n");
    out.push_str("  Conjecture 2 predicts these correspond to hard moral dilemmas.\n");

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::ResolvedValue;
    use got_core::geometry::CausalGeometry;

    fn identity_geometry(dim: usize) -> CausalGeometry {
        let mut gram = vec![0.0f32; dim * dim];
        for i in 0..dim {
            gram[i * dim + i] = 1.0;
        }
        CausalGeometry::from_raw_gram(gram, dim).unwrap()
    }

    fn make_value(term: &str, embedding: Vec<f32>) -> ResolvedValue {
        ResolvedValue {
            term: term.to_string(),
            normalised: term.to_lowercase(),
            embedding,
        }
    }

    #[test]
    fn collinear_points_have_zero_curvature() {
        let geom = identity_geometry(3);
        let values = vec![
            make_value("a", vec![1.0, 0.0, 0.0]),
            make_value("b", vec![2.0, 0.0, 0.0]),
            make_value("c", vec![3.0, 0.0, 0.0]),
        ];

        let result = analyse_curvature(&values, &geom, 2).unwrap();
        assert!(result.triple_curvatures[0].menger_curvature < 1e-6,
            "collinear points should have zero curvature: {}",
            result.triple_curvatures[0].menger_curvature);
    }

    #[test]
    fn right_angle_triangle_has_positive_curvature() {
        let geom = identity_geometry(3);
        let values = vec![
            make_value("a", vec![0.0, 0.0, 0.0]),
            make_value("b", vec![1.0, 0.0, 0.0]),
            make_value("c", vec![0.0, 1.0, 0.0]),
        ];

        let result = analyse_curvature(&values, &geom, 2).unwrap();
        assert!(result.triple_curvatures[0].menger_curvature > 0.5,
            "right triangle should have positive curvature: {}",
            result.triple_curvatures[0].menger_curvature);
    }

    #[test]
    fn equilateral_has_maximal_curvature_for_side_length() {
        let geom = identity_geometry(3);
        // Equilateral triangle with side length 1
        let values = vec![
            make_value("a", vec![0.0, 0.0, 0.0]),
            make_value("b", vec![1.0, 0.0, 0.0]),
            make_value("c", vec![0.5, 0.866, 0.0]),
        ];

        let result = analyse_curvature(&values, &geom, 2).unwrap();
        let kappa = result.triple_curvatures[0].menger_curvature;
        // For equilateral triangle: R = side / sqrt(3), so κ = sqrt(3)/side ≈ 1.732
        assert!((kappa - 1.732).abs() < 0.05,
            "equilateral κ should be ~1.732: {}", kappa);
    }

    #[test]
    fn more_terms_produces_richer_analysis() {
        let geom = identity_geometry(4);
        let values = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0, 0.0]),
            make_value("wisdom", vec![0.0, 0.0, 1.0, 0.0]),
            make_value("justice", vec![0.0, 0.0, 0.0, 1.0]),
        ];

        let result = analyse_curvature(&values, &geom, 3).unwrap();
        // C(4,3) = 4 triples
        assert_eq!(result.triple_curvatures.len(), 4);
        assert_eq!(result.term_curvatures.len(), 4);
    }

    #[test]
    fn render_produces_output() {
        let geom = identity_geometry(3);
        let values = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            make_value("wisdom", vec![0.5, 0.5, 0.5]),
        ];

        let result = analyse_curvature(&values, &geom, 2).unwrap();
        let text = render_curvature(&result);
        assert!(text.contains("Curvature Analysis"));
        assert!(text.contains("honesty"));
        assert!(text.contains("Menger"));
    }
}
