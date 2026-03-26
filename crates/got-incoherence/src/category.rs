// ---------------------------------------------------------------------------
// Categorical structure of value geometry.
//
// Each model defines a category enriched over [-1, 1]:
//   - Objects:    value terms
//   - Hom(A, B):  causal cosine similarity cos_Φ(A, B)
//   - Identity:   cos_Φ(A, A) = 1
//   - Symmetry:   cos_Φ(A, B) = cos_Φ(B, A)  (Φ is symmetric)
//
// A model comparison is a functor F: Val_base → Val_tuned.
// Per-term embedding drift gives the components of a natural
// transformation η: Id → F.
//
// The 2-category of models lets us check whether composing
// comparisons along different paths gives the same result —
// a coherence condition on alignment trajectories.
// ---------------------------------------------------------------------------

use got_core::geometry::CausalGeometry;
use serde::{Deserialize, Serialize};

use crate::coherence::{causal_cosine, euclidean_cosine, participation_ratio};
use crate::embeddings::ResolvedValue;
use crate::IncoherenceError;

// ---------------------------------------------------------------------------
// ValueCategory: a model's value structure as an enriched category
// ---------------------------------------------------------------------------

/// A category enriched over [-1, 1], representing a model's value geometry.
///
/// Objects are value terms, morphisms are cosine similarities.
/// The cosine matrix C[i][j] = cos_Φ(term_i, term_j) is the hom-set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueCategory {
    /// Label for this category (model name).
    pub label: String,
    /// Objects of the category (value terms, sorted).
    pub objects: Vec<String>,
    /// Hom-set: cosine matrix, row-major n×n.
    /// hom[i * n + j] = cos_Φ(objects[i], objects[j]).
    pub hom: Vec<f32>,
    /// Effective dimensionality (participation ratio of eigenspectrum).
    pub effective_rank: f32,
    /// Eigenspectrum of the cosine matrix (sorted descending).
    pub spectrum: Vec<f32>,
}

impl ValueCategory {
    /// Build a ValueCategory from resolved values under a causal geometry.
    pub fn from_values(
        label: &str,
        values: &[ResolvedValue],
        geometry: &CausalGeometry,
    ) -> Result<Self, IncoherenceError> {
        let n = values.len();
        if n < 2 {
            return Err(IncoherenceError::EmptyInput(
                "need at least 2 values for a category",
            ));
        }

        let mut objects: Vec<String> = values.iter().map(|v| v.normalised.clone()).collect();

        // Sort objects and reorder values to match, so categories
        // built from the same terms are directly comparable.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| objects[a].cmp(&objects[b]));
        let sorted_values: Vec<&ResolvedValue> = indices.iter().map(|&i| &values[i]).collect();
        objects.sort();

        // Compute hom-set (cosine matrix)
        let mut hom = vec![0.0f32; n * n];
        for i in 0..n {
            hom[i * n + i] = 1.0; // identity morphism
            for j in (i + 1)..n {
                let cos = causal_cosine(
                    &sorted_values[i].embedding,
                    &sorted_values[j].embedding,
                    geometry,
                )?;
                hom[i * n + j] = cos;
                hom[j * n + i] = cos;
            }
        }

        // Effective rank via participation ratio
        let owned: Vec<ResolvedValue> = sorted_values.iter().map(|v| (*v).clone()).collect();
        let (pr, spectrum) = participation_ratio(&owned, geometry)?;

        Ok(ValueCategory {
            label: label.to_string(),
            objects,
            hom,
            effective_rank: pr,
            spectrum,
        })
    }

    /// Number of objects.
    pub fn size(&self) -> usize {
        self.objects.len()
    }

    /// Look up hom(a, b) by term names.
    pub fn hom_by_name(&self, a: &str, b: &str) -> Option<f32> {
        let i = self.objects.iter().position(|o| o == a)?;
        let j = self.objects.iter().position(|o| o == b)?;
        Some(self.hom[i * self.size() + j])
    }

    /// Check the triangle inequality in angle space:
    ///   arccos(cos(A,C)) <= arccos(cos(A,B)) + arccos(cos(B,C))
    ///
    /// Returns violations (triples where composition is inconsistent).
    pub fn composition_violations(&self, tolerance: f32) -> Vec<CompositionViolation> {
        let n = self.size();
        let mut violations = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let cos_ij = self.hom[i * n + j];
                    let cos_jk = self.hom[j * n + k];
                    let cos_ik = self.hom[i * n + k];

                    // Triangle inequality in angle space
                    let angle_ij = cos_ij.clamp(-1.0, 1.0).acos();
                    let angle_jk = cos_jk.clamp(-1.0, 1.0).acos();
                    let angle_ik = cos_ik.clamp(-1.0, 1.0).acos();

                    let excess = angle_ik - (angle_ij + angle_jk);
                    if excess > tolerance {
                        violations.push(CompositionViolation {
                            a: self.objects[i].clone(),
                            b: self.objects[j].clone(),
                            c: self.objects[k].clone(),
                            angle_ab: angle_ij,
                            angle_bc: angle_jk,
                            angle_ac: angle_ik,
                            excess,
                        });
                    }
                }
            }
        }

        violations
    }
}

/// A triple where the triangle inequality is violated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionViolation {
    pub a: String,
    pub b: String,
    pub c: String,
    pub angle_ab: f32,
    pub angle_bc: f32,
    pub angle_ac: f32,
    /// How much angle(A,C) exceeds angle(A,B) + angle(B,C).
    pub excess: f32,
}

// ---------------------------------------------------------------------------
// ValueFunctor: a structure-preserving map between value categories
// ---------------------------------------------------------------------------

/// A functor F: C_base → C_compared between value categories.
///
/// Maps each object (value term) to itself in the other category,
/// and each morphism cos_base(A, B) to cos_compared(A, B).
///
/// The functor's "faithfulness" measures how well it preserves
/// morphism structure. A faithful functor preserves all pairwise
/// relationships; an unfaithful one distorts them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFunctor {
    /// Source category label.
    pub source: String,
    /// Target category label.
    pub target: String,
    /// Objects in the shared domain (terms present in both categories).
    pub objects: Vec<String>,

    /// Components of the natural transformation η: Id → F.
    /// For each object, the cosine similarity between its embedding
    /// in source vs target. η_i = cos(emb_source(i), emb_target(i)).
    /// Only defined when source and target share hidden dimension.
    pub components: Vec<NatComponent>,

    /// Morphism distortion: for each pair (A, B),
    /// how much the functor changed the hom-set value.
    /// delta[i] = cos_target(A, B) - cos_source(A, B).
    pub morphism_deltas: Vec<MorphismDelta>,

    /// Frobenius norm of the morphism distortion matrix.
    /// 0 = perfect isomorphism, large = heavy distortion.
    pub distortion: f32,

    /// Rank preservation: ratio of effective ranks.
    /// 1.0 = same rank, <1 = collapse, >1 = expansion.
    pub rank_ratio: f32,
}

/// Component of the natural transformation at one object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatComponent {
    pub object: String,
    /// cos(source_embedding, target_embedding).
    pub similarity: f32,
}

/// How the functor changed one morphism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphismDelta {
    pub source_obj: String,
    pub target_obj: String,
    pub source_hom: f32,
    pub target_hom: f32,
    pub delta: f32,
}

impl ValueFunctor {
    /// Build a functor between two value categories.
    ///
    /// The functor is defined on the shared objects (terms present in both).
    /// If the categories have different hidden dimensions, the natural
    /// transformation components are not defined (empty vec).
    pub fn new(
        source: &ValueCategory,
        target: &ValueCategory,
        source_values: &[ResolvedValue],
        target_values: &[ResolvedValue],
    ) -> Result<Self, IncoherenceError> {
        // Find shared objects
        let source_set: std::collections::HashSet<&str> =
            source.objects.iter().map(|s| s.as_str()).collect();
        let target_set: std::collections::HashSet<&str> =
            target.objects.iter().map(|s| s.as_str()).collect();
        let mut shared: Vec<String> = source_set
            .intersection(&target_set)
            .map(|s| s.to_string())
            .collect();
        shared.sort();

        if shared.len() < 2 {
            return Err(IncoherenceError::EmptyInput(
                "need at least 2 shared objects for a functor",
            ));
        }

        // Natural transformation components (per-term drift)
        let same_dim = source_values.first().map(|v| v.embedding.len())
            == target_values.first().map(|v| v.embedding.len());

        let components: Vec<NatComponent> = if same_dim {
            shared.iter().filter_map(|term| {
                let sv = source_values.iter().find(|v| v.normalised == *term)?;
                let tv = target_values.iter().find(|v| v.normalised == *term)?;
                let cos = euclidean_cosine(&sv.embedding, &tv.embedding);
                Some(NatComponent {
                    object: term.clone(),
                    similarity: cos,
                })
            }).collect()
        } else {
            vec![]
        };

        // Morphism deltas
        let n = shared.len();
        let mut morphism_deltas = Vec::with_capacity(n * (n - 1) / 2);
        let mut frobenius_sq = 0.0f32;

        for i in 0..n {
            for j in (i + 1)..n {
                let sh = source.hom_by_name(&shared[i], &shared[j]).unwrap_or(0.0);
                let th = target.hom_by_name(&shared[i], &shared[j]).unwrap_or(0.0);
                let delta = th - sh;
                frobenius_sq += delta * delta;

                morphism_deltas.push(MorphismDelta {
                    source_obj: shared[i].clone(),
                    target_obj: shared[j].clone(),
                    source_hom: sh,
                    target_hom: th,
                    delta,
                });
            }
        }

        // Sort by absolute delta descending
        morphism_deltas.sort_by(|a, b| {
            b.delta.abs().partial_cmp(&a.delta.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let rank_ratio = if source.effective_rank > 0.0 {
            target.effective_rank / source.effective_rank
        } else {
            1.0
        };

        Ok(ValueFunctor {
            source: source.label.clone(),
            target: target.label.clone(),
            objects: shared,
            components,
            morphism_deltas,
            distortion: frobenius_sq.sqrt(),
            rank_ratio,
        })
    }

    /// Is this functor approximately an isomorphism?
    ///
    /// An isomorphism preserves all morphisms (distortion ≈ 0)
    /// and all natural transformation components are ≈ 1.
    pub fn is_approximate_isomorphism(&self, threshold: f32) -> bool {
        if self.distortion > threshold {
            return false;
        }
        if !self.components.is_empty() {
            let min_sim = self.components.iter()
                .map(|c| c.similarity)
                .fold(f32::INFINITY, f32::min);
            if 1.0 - min_sim > threshold {
                return false;
            }
        }
        true
    }

    /// Is this functor faithful (preserves morphism structure)?
    ///
    /// A faithful functor has small morphism distortion relative
    /// to the number of morphisms.
    pub fn faithfulness(&self) -> f32 {
        if self.morphism_deltas.is_empty() {
            return 1.0;
        }
        let mean_abs_delta: f32 = self.morphism_deltas.iter()
            .map(|d| d.delta.abs())
            .sum::<f32>() / self.morphism_deltas.len() as f32;
        // Faithfulness: 1.0 = perfect, 0.0 = completely distorted
        (1.0 - mean_abs_delta).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// FunctorComposition: the 2-category coherence check
// ---------------------------------------------------------------------------

/// Result of checking whether functor composition is path-independent.
///
/// Given models A → B → C and A → C directly, the 2-category
/// coherence condition says the composed functor (A→B)∘(B→C) should
/// agree with the direct functor A→C.
///
/// If it doesn't, the alignment trajectory is path-dependent:
/// the order in which you apply training stages matters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathCoherence {
    /// The three models in the path.
    pub model_a: String,
    pub model_b: String,
    pub model_c: String,

    /// Direct functor A → C.
    pub direct_distortion: f32,
    /// Composed functors: A → B distortion + B → C distortion.
    pub composed_distortion_ab: f32,
    pub composed_distortion_bc: f32,

    /// Path coherence error: how different the composed path is from direct.
    /// For each pair of terms, |cos_direct(A→C) - cos_composed(A→B→C)|.
    pub coherence_error: f32,

    /// Per-morphism comparison: direct vs composed.
    pub morphism_comparisons: Vec<MorphismPathComparison>,
}

/// Comparison of one morphism along two paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphismPathComparison {
    pub term_a: String,
    pub term_b: String,
    /// Hom in model A.
    pub hom_a: f32,
    /// Hom in model C via direct comparison.
    pub hom_c_direct: f32,
    /// Hom in model B (intermediate).
    pub hom_b: f32,
    /// |direct - composed| at this morphism.
    /// For enriched categories, "composed" means the hom in C.
    /// The coherence question is whether the *change* A→C equals
    /// the *change* A→B plus B→C.
    pub delta_direct: f32,
    pub delta_via_b: f32,
    pub path_error: f32,
}

/// Check path coherence: does the alignment trajectory A → B → C
/// give the same geometric result as A → C directly?
///
/// In the 2-category, this checks whether the diagram commutes.
pub fn check_path_coherence(
    cat_a: &ValueCategory,
    cat_b: &ValueCategory,
    cat_c: &ValueCategory,
) -> Result<PathCoherence, IncoherenceError> {
    // Find terms shared across all three
    let a_set: std::collections::HashSet<&str> =
        cat_a.objects.iter().map(|s| s.as_str()).collect();
    let b_set: std::collections::HashSet<&str> =
        cat_b.objects.iter().map(|s| s.as_str()).collect();
    let c_set: std::collections::HashSet<&str> =
        cat_c.objects.iter().map(|s| s.as_str()).collect();

    let ab: std::collections::HashSet<&str> = a_set.intersection(&b_set).copied().collect();
    let mut shared: Vec<String> = ab.intersection(&c_set).map(|s| s.to_string()).collect();
    shared.sort();

    if shared.len() < 2 {
        return Err(IncoherenceError::EmptyInput(
            "need at least 2 shared terms across all three models",
        ));
    }

    let n = shared.len();
    let mut morphism_comparisons = Vec::new();
    let mut total_path_error_sq = 0.0f32;
    let mut direct_distortion_sq = 0.0f32;
    let mut ab_distortion_sq = 0.0f32;
    let mut bc_distortion_sq = 0.0f32;

    for i in 0..n {
        for j in (i + 1)..n {
            let hom_a = cat_a.hom_by_name(&shared[i], &shared[j]).unwrap_or(0.0);
            let hom_b = cat_b.hom_by_name(&shared[i], &shared[j]).unwrap_or(0.0);
            let hom_c = cat_c.hom_by_name(&shared[i], &shared[j]).unwrap_or(0.0);

            let delta_direct = hom_c - hom_a;         // A → C
            let delta_ab = hom_b - hom_a;             // A → B
            let delta_bc = hom_c - hom_b;             // B → C
            let delta_via_b = delta_ab + delta_bc;     // composed: should equal delta_direct

            let path_error = (delta_direct - delta_via_b).abs();
            total_path_error_sq += path_error * path_error;
            direct_distortion_sq += delta_direct * delta_direct;
            ab_distortion_sq += delta_ab * delta_ab;
            bc_distortion_sq += delta_bc * delta_bc;

            morphism_comparisons.push(MorphismPathComparison {
                term_a: shared[i].clone(),
                term_b: shared[j].clone(),
                hom_a,
                hom_c_direct: hom_c,
                hom_b,
                delta_direct,
                delta_via_b,
                path_error,
            });
        }
    }

    // Sort by path error descending
    morphism_comparisons.sort_by(|a, b| {
        b.path_error.partial_cmp(&a.path_error).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(PathCoherence {
        model_a: cat_a.label.clone(),
        model_b: cat_b.label.clone(),
        model_c: cat_c.label.clone(),
        direct_distortion: direct_distortion_sq.sqrt(),
        composed_distortion_ab: ab_distortion_sq.sqrt(),
        composed_distortion_bc: bc_distortion_sq.sqrt(),
        coherence_error: total_path_error_sq.sqrt(),
        morphism_comparisons,
    })
}

/// Format a ValueCategory for display.
pub fn render_category(cat: &ValueCategory) -> String {
    let mut out = String::new();
    let n = cat.size();

    out.push_str(&format!("Category: {} ({} objects)\n", cat.label, n));
    out.push_str(&format!("Effective rank: {:.2} / {}\n", cat.effective_rank, n));
    out.push_str(&format!("Spectrum (top 5): "));
    for v in cat.spectrum.iter().take(5) {
        out.push_str(&format!("{:.3} ", v));
    }
    out.push('\n');

    // Hom-set as compact matrix
    out.push_str("\nHom-set (cosine matrix):\n");
    out.push_str(&format!("{:>15}", ""));
    for obj in &cat.objects {
        out.push_str(&format!("{:>10}", &obj[..obj.len().min(9)]));
    }
    out.push('\n');
    for i in 0..n {
        out.push_str(&format!("{:>15}", &cat.objects[i][..cat.objects[i].len().min(14)]));
        for j in 0..n {
            out.push_str(&format!("{:>10.3}", cat.hom[i * n + j]));
        }
        out.push('\n');
    }
    out
}

/// Format a ValueFunctor for display.
pub fn render_functor(f: &ValueFunctor) -> String {
    let mut out = String::new();

    out.push_str(&format!("Functor: {} → {}\n", f.source, f.target));
    out.push_str(&format!("Objects: {} shared\n", f.objects.len()));
    out.push_str(&format!("Distortion (Frobenius): {:.4}\n", f.distortion));
    out.push_str(&format!("Faithfulness: {:.4}\n", f.faithfulness()));
    out.push_str(&format!("Rank ratio: {:.4}\n", f.rank_ratio));
    out.push_str(&format!(
        "Approximate isomorphism (threshold 0.05): {}\n",
        f.is_approximate_isomorphism(0.05)
    ));

    if !f.components.is_empty() {
        out.push_str("\nNatural transformation components (η_i = cos(source, target)):\n");
        for c in &f.components {
            let bar_len = ((1.0 - c.similarity) * 200.0) as usize;
            let bar: String = "#".repeat(bar_len.min(40));
            out.push_str(&format!("  {:<20} η = {:.4}  {}\n", c.object, c.similarity, bar));
        }
    }

    let top_n = 10.min(f.morphism_deltas.len());
    if top_n > 0 {
        out.push_str(&format!("\nTop {} morphism distortions:\n", top_n));
        for d in &f.morphism_deltas[..top_n] {
            let arrow = if d.delta > 0.0 { "+" } else { "" };
            out.push_str(&format!(
                "  {:<15} → {:<15}  {:.3} → {:.3}  ({}{:.3})\n",
                d.source_obj, d.target_obj, d.source_hom, d.target_hom, arrow, d.delta
            ));
        }
    }

    out
}

/// Format a PathCoherence check for display.
pub fn render_path_coherence(pc: &PathCoherence) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Path coherence: {} → {} → {} vs {} → {}\n",
        pc.model_a, pc.model_b, pc.model_c, pc.model_a, pc.model_c
    ));
    out.push_str(&format!("Direct distortion (A→C):   {:.4}\n", pc.direct_distortion));
    out.push_str(&format!("Composed A→B distortion:   {:.4}\n", pc.composed_distortion_ab));
    out.push_str(&format!("Composed B→C distortion:   {:.4}\n", pc.composed_distortion_bc));
    out.push_str(&format!("Path coherence error:      {:.6}\n", pc.coherence_error));

    let label = if pc.coherence_error < 1e-4 {
        "COHERENT (diagram commutes)"
    } else if pc.coherence_error < 1e-2 {
        "APPROXIMATELY COHERENT"
    } else {
        "INCOHERENT (path-dependent)"
    };
    out.push_str(&format!("Verdict: {}\n", label));

    let top_n = 5.min(pc.morphism_comparisons.len());
    if top_n > 0 && pc.coherence_error > 1e-6 {
        out.push_str(&format!("\nTop {} path-dependent morphisms:\n", top_n));
        for m in &pc.morphism_comparisons[..top_n] {
            out.push_str(&format!(
                "  {:<15} ↔ {:<15}  direct: {:.4}  via B: {:.4}  error: {:.6}\n",
                m.term_a, m.term_b, m.delta_direct, m.delta_via_b, m.path_error
            ));
        }
    }

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

    fn orthogonal_values() -> Vec<ResolvedValue> {
        vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            make_value("wisdom", vec![0.0, 0.0, 1.0]),
        ]
    }

    fn collapsed_values() -> Vec<ResolvedValue> {
        vec![
            make_value("honesty", vec![1.0, 0.1, 0.0]),
            make_value("courage", vec![0.9, 0.2, 0.0]),
            make_value("wisdom", vec![0.8, 0.3, 0.0]),
        ]
    }

    #[test]
    fn category_from_orthogonal_has_zero_hom() {
        let geom = identity_geometry(3);
        let cat = ValueCategory::from_values("test", &orthogonal_values(), &geom).unwrap();

        assert_eq!(cat.size(), 3);
        // Diagonal = 1 (identity)
        assert!((cat.hom_by_name("honesty", "honesty").unwrap() - 1.0).abs() < 1e-6);
        // Off-diagonal ≈ 0 (orthogonal)
        assert!(cat.hom_by_name("honesty", "courage").unwrap().abs() < 1e-6);
        // Effective rank ≈ 3 (fully spread)
        assert!((cat.effective_rank - 3.0).abs() < 0.1);
    }

    #[test]
    fn category_from_collapsed_has_lower_rank() {
        let geom = identity_geometry(3);
        let cat = ValueCategory::from_values("test", &collapsed_values(), &geom).unwrap();

        assert!(cat.effective_rank < 2.5,
            "collapsed values should have lower rank: {}", cat.effective_rank);
    }

    #[test]
    fn functor_between_identical_is_isomorphism() {
        let geom = identity_geometry(3);
        let values = orthogonal_values();
        let cat = ValueCategory::from_values("model", &values, &geom).unwrap();

        let f = ValueFunctor::new(&cat, &cat, &values, &values).unwrap();

        assert!(f.distortion < 1e-6, "distortion should be 0: {}", f.distortion);
        assert!((f.faithfulness() - 1.0).abs() < 1e-6);
        assert!((f.rank_ratio - 1.0).abs() < 1e-6);
        assert!(f.is_approximate_isomorphism(0.01));

        // All natural transformation components should be 1.0
        for c in &f.components {
            assert!((c.similarity - 1.0).abs() < 1e-6,
                "component {} should be 1.0: {}", c.object, c.similarity);
        }
    }

    #[test]
    fn functor_to_collapsed_detects_distortion() {
        let geom = identity_geometry(3);
        let base = orthogonal_values();
        let collapsed = collapsed_values();
        let cat_base = ValueCategory::from_values("base", &base, &geom).unwrap();
        let cat_collapsed = ValueCategory::from_values("collapsed", &collapsed, &geom).unwrap();

        let f = ValueFunctor::new(&cat_base, &cat_collapsed, &base, &collapsed).unwrap();

        assert!(f.distortion > 0.1, "should detect distortion: {}", f.distortion);
        assert!(f.rank_ratio < 1.0, "rank should decrease: {}", f.rank_ratio);
        assert!(!f.is_approximate_isomorphism(0.05));
    }

    #[test]
    fn path_coherence_is_exact_for_enriched_categories() {
        // For enriched categories where "composition" is just looking up
        // the hom in the target, path coherence is exact:
        // delta(A→C) = delta(A→B) + delta(B→C) always holds because
        // delta(A→B) = hom_B - hom_A and delta(B→C) = hom_C - hom_B,
        // so their sum = hom_C - hom_A = delta(A→C).
        let geom = identity_geometry(3);

        let a = vec![
            make_value("honesty", vec![1.0, 0.0, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.0]),
            make_value("wisdom", vec![0.0, 0.0, 1.0]),
        ];
        let b = vec![
            make_value("honesty", vec![1.0, 0.1, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.1]),
            make_value("wisdom", vec![0.1, 0.0, 1.0]),
        ];
        let c = vec![
            make_value("honesty", vec![1.0, 0.3, 0.0]),
            make_value("courage", vec![0.0, 1.0, 0.3]),
            make_value("wisdom", vec![0.3, 0.0, 1.0]),
        ];

        let cat_a = ValueCategory::from_values("A", &a, &geom).unwrap();
        let cat_b = ValueCategory::from_values("B", &b, &geom).unwrap();
        let cat_c = ValueCategory::from_values("C", &c, &geom).unwrap();

        let pc = check_path_coherence(&cat_a, &cat_b, &cat_c).unwrap();

        // This should be exactly coherent: delta telescopes
        assert!(pc.coherence_error < 1e-4,
            "enriched category path should be coherent: error = {}", pc.coherence_error);
    }

    #[test]
    fn no_composition_violations_for_real_geometry() {
        // Cosine similarities in a real vector space always satisfy
        // the triangle inequality in angle space.
        let geom = identity_geometry(3);
        let values = orthogonal_values();
        let cat = ValueCategory::from_values("test", &values, &geom).unwrap();

        let violations = cat.composition_violations(1e-6);
        assert!(violations.is_empty(), "real geometry should have no violations");
    }

    #[test]
    fn render_functions_produce_output() {
        let geom = identity_geometry(3);
        let values = orthogonal_values();
        let cat = ValueCategory::from_values("test-model", &values, &geom).unwrap();

        let text = render_category(&cat);
        assert!(text.contains("test-model"));
        assert!(text.contains("Effective rank"));

        let collapsed = collapsed_values();
        let cat2 = ValueCategory::from_values("collapsed", &collapsed, &geom).unwrap();
        let f = ValueFunctor::new(&cat, &cat2, &values, &collapsed).unwrap();
        let text = render_functor(&f);
        assert!(text.contains("Functor"));
        assert!(text.contains("Faithfulness"));
    }
}
