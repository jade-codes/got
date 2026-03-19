// ---------------------------------------------------------------------------
// Demo data: pre-computed embeddings and a sample conversation with
// per-message embeddings that show incoherence emerging over turns.
//
// All data is compiled into the binary via include_str! — zero runtime
// file dependencies.
// ---------------------------------------------------------------------------

/// Demo value-term embeddings JSON (28 terms, 32-d each).
pub fn demo_embeddings_json() -> &'static str {
    include_str!("../../../data/demo/embeddings.json")
}

/// Demo conversation with per-message embeddings.
///
/// Each message carries a 32-d embedding (pre-computed blend of value-term
/// vectors).  The API projects these against value terms using the causal
/// inner product to *detect* which values are active — no hand-tagging.
pub fn demo_conversation_json() -> &'static str {
    include_str!("../../../data/demo/demo_conversation.json")
}
