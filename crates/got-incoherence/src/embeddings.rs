// ---------------------------------------------------------------------------
// Value embeddings: map text terms to vectors in ℝ^d.
//
// The core abstraction is `EmbeddingSource` — a trait that takes a term
// string and returns its embedding vector.  This decouples the coherence
// analysis from any particular model or API.
//
// Shipped implementations:
//   • `UnembeddingLookup` — finds the closest vocabulary token in the
//     model's unembedding matrix U.  Zero external dependencies.
//   • `PrecomputedEmbeddings` — loads a JSON map of term → vector.
//     For use with external embedding models (sentence-transformers, etc.)
//
// Design: embeddings are always in the model's hidden space ℝ^d, so they
// can be measured under the causal inner product ⟨·,·⟩_Φ.
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use got_core::UnembeddingMatrix;
use serde::{Deserialize, Serialize};

use crate::IncoherenceError;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Source of embeddings for value terms.
///
/// Implementors map a string term (e.g. "honesty") to a vector in ℝ^d.
/// Returns `None` if the term cannot be embedded (unknown vocabulary, etc.).
pub trait EmbeddingSource {
    /// The hidden dimension of the embedding space.
    fn hidden_dim(&self) -> usize;

    /// Look up the embedding for a single term.
    fn embed(&self, term: &str) -> Option<Vec<f32>>;

    /// Embed multiple terms at once. Default: sequential calls to `embed`.
    fn embed_batch(&self, terms: &[&str]) -> Vec<Option<Vec<f32>>> {
        terms.iter().map(|t| self.embed(t)).collect()
    }
}

// ---------------------------------------------------------------------------
// UnembeddingLookup: vocabulary-based embedding lookup
// ---------------------------------------------------------------------------

/// Embeds terms by finding the closest token in the model's vocabulary.
///
/// Uses the unembedding matrix rows as token embeddings.  Requires a
/// vocabulary mapping (token index → string) provided at construction time.
///
/// This is a shallow embedding — it maps terms to single-token representations.
/// For multi-word concepts, use `PrecomputedEmbeddings` or average sub-tokens.
pub struct UnembeddingLookup {
    /// Token string → row index in the unembedding matrix.
    vocab: HashMap<String, usize>,
    /// The unembedding matrix U ∈ ℝ^{V×d}, row-major.
    matrix: UnembeddingMatrix,
}

impl UnembeddingLookup {
    /// Build a lookup from a vocabulary list and unembedding matrix.
    ///
    /// `vocab_tokens`: ordered list of token strings, one per row of U.
    /// Length must equal `matrix.vocab_size`.
    pub fn new(
        vocab_tokens: Vec<String>,
        matrix: UnembeddingMatrix,
    ) -> Result<Self, IncoherenceError> {
        if vocab_tokens.len() != matrix.vocab_size {
            return Err(IncoherenceError::VocabMismatch {
                vocab_len: vocab_tokens.len(),
                matrix_rows: matrix.vocab_size,
            });
        }

        let mut vocab = HashMap::with_capacity(vocab_tokens.len());
        for (idx, token) in vocab_tokens.into_iter().enumerate() {
            // First occurrence wins (handles duplicate tokens gracefully).
            vocab.entry(token.to_lowercase()).or_insert(idx);
        }

        Ok(Self { vocab, matrix })
    }

    /// Get the raw embedding row for a vocabulary index.
    fn row(&self, idx: usize) -> Vec<f32> {
        let d = self.matrix.hidden_dim;
        let start = idx * d;
        self.matrix.data[start..start + d].to_vec()
    }
}

impl EmbeddingSource for UnembeddingLookup {
    fn hidden_dim(&self) -> usize {
        self.matrix.hidden_dim
    }

    fn embed(&self, term: &str) -> Option<Vec<f32>> {
        let key = term.trim().to_lowercase();
        self.vocab.get(&key).map(|&idx| self.row(idx))
    }
}

// ---------------------------------------------------------------------------
// PrecomputedEmbeddings: load from external source
// ---------------------------------------------------------------------------

/// Embeddings pre-computed externally and loaded as a JSON map.
///
/// Use this when embeddings come from sentence-transformers, OpenAI, or
/// any other source that produces vectors offline.
///
/// Expected JSON format:
/// ```json
/// {
///   "honesty": [0.1, -0.3, ...],
///   "deception": [-0.2, 0.4, ...],
///   ...
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecomputedEmbeddings {
    embeddings: HashMap<String, Vec<f32>>,
    hidden_dim: usize,
}

impl PrecomputedEmbeddings {
    /// Build from a map of term → embedding vector.
    ///
    /// All vectors must have the same dimension. Returns an error if
    /// dimensions are inconsistent or the map is empty.
    pub fn new(embeddings: HashMap<String, Vec<f32>>) -> Result<Self, IncoherenceError> {
        if embeddings.is_empty() {
            return Err(IncoherenceError::EmptyInput("no embeddings provided"));
        }

        let hidden_dim = embeddings.values().next().unwrap().len();
        if hidden_dim == 0 {
            return Err(IncoherenceError::EmptyInput("embedding dimension is 0"));
        }

        for (term, vec) in &embeddings {
            if vec.len() != hidden_dim {
                return Err(IncoherenceError::DimensionInconsistency {
                    term: term.clone(),
                    expected: hidden_dim,
                    got: vec.len(),
                });
            }
        }

        // Normalise keys to lowercase for consistent lookup.
        let normalised: HashMap<String, Vec<f32>> = embeddings
            .into_iter()
            .map(|(k, v)| (k.to_lowercase(), v))
            .collect();

        Ok(Self {
            embeddings: normalised,
            hidden_dim,
        })
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, IncoherenceError> {
        let map: HashMap<String, Vec<f32>> =
            serde_json::from_str(json).map_err(IncoherenceError::Json)?;
        Self::new(map)
    }
}

impl EmbeddingSource for PrecomputedEmbeddings {
    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn embed(&self, term: &str) -> Option<Vec<f32>> {
        let key = term.trim().to_lowercase();
        self.embeddings.get(&key).cloned()
    }
}

// ---------------------------------------------------------------------------
// Resolved value: a term successfully mapped to an embedding
// ---------------------------------------------------------------------------

/// A value term that has been successfully resolved to an embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedValue {
    /// The original term string.
    pub term: String,
    /// The normalised (lowercase, trimmed) form used for lookup.
    pub normalised: String,
    /// The embedding vector in ℝ^d.
    pub embedding: Vec<f32>,
}

/// Resolve a list of value terms against an embedding source.
///
/// Returns (resolved, unresolved) — the caller decides how to handle
/// terms that couldn't be embedded.
pub fn resolve_values(
    terms: &[&str],
    source: &dyn EmbeddingSource,
) -> (Vec<ResolvedValue>, Vec<String>) {
    let mut resolved = Vec::new();
    let mut unresolved = Vec::new();

    for &term in terms {
        let normalised = term.trim().to_lowercase();
        match source.embed(term) {
            Some(embedding) => resolved.push(ResolvedValue {
                term: term.to_string(),
                normalised,
                embedding,
            }),
            None => unresolved.push(term.to_string()),
        }
    }

    (resolved, unresolved)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precomputed_lookup_case_insensitive() {
        let mut map = HashMap::new();
        map.insert("Honesty".to_string(), vec![1.0, 0.0, 0.0]);
        map.insert("Deception".to_string(), vec![-1.0, 0.0, 0.0]);
        let source = PrecomputedEmbeddings::new(map).unwrap();

        assert!(source.embed("honesty").is_some());
        assert!(source.embed("HONESTY").is_some());
        assert!(source.embed("unknown").is_none());
        assert_eq!(source.hidden_dim(), 3);
    }

    #[test]
    fn precomputed_rejects_inconsistent_dims() {
        let mut map = HashMap::new();
        map.insert("a".to_string(), vec![1.0, 0.0]);
        map.insert("b".to_string(), vec![1.0, 0.0, 0.0]); // wrong dim
        assert!(PrecomputedEmbeddings::new(map).is_err());
    }

    #[test]
    fn precomputed_rejects_empty() {
        let map: HashMap<String, Vec<f32>> = HashMap::new();
        assert!(PrecomputedEmbeddings::new(map).is_err());
    }

    #[test]
    fn unembedding_lookup_works() {
        let u = UnembeddingMatrix::new(3, 2, vec![
            1.0, 2.0,   // token 0: "hello"
            3.0, 4.0,   // token 1: "world"
            5.0, 6.0,   // token 2: "test"
        ]).unwrap();
        let vocab = vec!["hello".into(), "world".into(), "test".into()];
        let source = UnembeddingLookup::new(vocab, u).unwrap();

        assert_eq!(source.embed("hello"), Some(vec![1.0, 2.0]));
        assert_eq!(source.embed("WORLD"), Some(vec![3.0, 4.0]));
        assert!(source.embed("missing").is_none());
        assert_eq!(source.hidden_dim(), 2);
    }

    #[test]
    fn unembedding_lookup_rejects_vocab_mismatch() {
        let u = UnembeddingMatrix::new(3, 2, vec![1.0; 6]).unwrap();
        let vocab = vec!["a".into(), "b".into()]; // only 2, matrix has 3 rows
        assert!(UnembeddingLookup::new(vocab, u).is_err());
    }

    #[test]
    fn resolve_values_partitions_correctly() {
        let mut map = HashMap::new();
        map.insert("honesty".to_string(), vec![1.0, 0.0]);
        map.insert("kindness".to_string(), vec![0.0, 1.0]);
        let source = PrecomputedEmbeddings::new(map).unwrap();

        let terms = &["honesty", "missing_term", "kindness"];
        let (resolved, unresolved) = resolve_values(terms, &source);

        assert_eq!(resolved.len(), 2);
        assert_eq!(unresolved, vec!["missing_term"]);
        assert_eq!(resolved[0].term, "honesty");
        assert_eq!(resolved[1].term, "kindness");
    }

    #[test]
    fn from_json_works() {
        let json = r#"{"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}"#;
        let source = PrecomputedEmbeddings::from_json(json).unwrap();
        assert_eq!(source.hidden_dim(), 2);
        assert!(source.embed("alpha").is_some());
    }
}
