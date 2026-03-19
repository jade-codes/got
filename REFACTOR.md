# got-web Refactoring Plan

## The Problem

The web demo is broken in two fundamental ways:

### 1. Dimension Mismatch — Analysis is a No-Op

The demo Gram matrix (`demo_gram_matrix()`) is **8×8**, but the demo embeddings
(`data/demo/embeddings.json`) are **32-dimensional**.

`CausalGeometry::from_raw_gram(gram, 8)` creates a geometry that expects 8-d
vectors. When `analyse_value_system()` passes 32-d embedding vectors to
`geometry.inner_product()`, the `check_vec()` guard returns
`DimensionMismatch { expected: 8, got: 32 }`.

The API handler catches this error silently and returns
`coherence_score: 1.0` with empty contradictions for every turn.

**The entire analysis pipeline is a no-op. Every demo run shows a flat line
at 1.0 and zero contradictions.**

### 2. Hand-Tagged Values — The Geometry Does Nothing

Each message in the demo JSON carries a hand-written `"values": [...]` array.
The API just passes these labels to `analyse_value_system()` as lookup keys.
Even if the dimensions matched, the causal geometry would only be computing
pairwise cosines between pre-existing term embeddings — it wouldn't be
*detecting* anything from the message content.

The whole point of the causal inner product ⟨u,v⟩_Φ is to measure how
message content activates value directions in the embedding space. Hand-tagging
bypasses that entirely.

---

## What to Fix

### A. Fix the Gram Matrix (32×32)

**File**: `crates/got-web/src/demo.rs`

Replace the hand-coded 8×8 matrix with a proper 32×32 Gram matrix derived
from the embeddings: Φ = EᵀE where E is the 28×32 term embedding matrix.

**How**: Write a Python script (`scripts/generate_gram_matrix.py`) that:
1. Loads `data/demo/embeddings.json`
2. Stacks the 28 embeddings into a 28×32 matrix E
3. Computes Φ = EᵀE (32×32)
4. Writes it as a Rust `vec![...]` literal or a JSON file

Then either:
- `include_str!` a JSON gram matrix file and parse at startup, or
- Compute Φ = EᵀE at startup in Rust from the embeddings (simpler, no
  extra data file)

**Decision**: Compute at startup. It's 28×32 × 32×28 = 32×32 — trivial.
Delete `demo_gram_matrix()` entirely. Build the geometry from the embeddings
in `load_demo_resources()`.

### B. Generate Message Embeddings and Auto-Detect Values

**File**: `crates/got-web/src/demo.rs`, `crates/got-web/src/api.rs`

Each demo message needs a 32-d embedding vector. Since we don't have a
sentence-transformer at runtime, we pre-compute these as weighted blends of
value term vectors.

**How**: Write a Python script (`scripts/generate_message_embeddings.py`) that:
1. Loads `data/demo/embeddings.json`
2. For each of the 13 demo messages, creates a 32-d vector as a weighted sum
   of value term embeddings:
   - **Explicit** values (words clearly present in the text) get weight 1.0
   - **Implicit** values (contextually implied) get weight 0.3–0.5
   - Normalize to unit length
3. For the manipulation messages (advisor), blend in small amounts of
   contradictory value vectors (e.g. `secrecy * 0.2` while talking about
   openness) — this is how the geometry catches the manipulation
4. Writes the result to `data/demo/message_embeddings.json`

**Demo JSON changes**:
- Remove `"values": [...]` from each message (the geometry detects these now)
- Add `"embedding": [0.1, -0.3, ...]` to each message (32-d vector)

**API changes** (`api.rs`):
- At each turn, compute `cos_Φ(message_embedding, term_embedding)` for all
  28 value terms
- Values with |cos_Φ| above a detection threshold (e.g. 0.3) are "detected"
  in that message
- Positive cos_Φ = value affirmed; negative cos_Φ = value negated
- Accumulate detected values per turn, run contradiction analysis as before

### C. Remove Dead / Duplicate Code

#### Dead endpoint: `GET /api/terms`
- **File**: `crates/got-web/src/main.rs` lines 36–41 (`terms()` handler)
- **File**: `crates/got-web/src/main.rs` line 61 (`.route("/api/terms", ...)`)
- **Action**: Delete both. The frontend never calls this endpoint. The analysis
  response already includes `available_terms`.

#### Dead Gram matrix function: `demo_gram_matrix()`
- **File**: `crates/got-web/src/demo.rs` lines 15–34
- **Action**: Delete after implementing startup Φ = EᵀE computation in api.rs.

#### Duplicate type hierarchy
The following types in `api.rs` are 1:1 copies of types in `coherence.rs`
with slightly different names:

| coherence.rs | api.rs | Keep |
|---|---|---|
| `Contradiction` | `ContradictionDto` | coherence.rs (add `Serialize`) |
| `Redundancy` | `RedundancyDto` | coherence.rs (already `Serialize`) |
| `PairwiseRelation` | `PairwiseDto` | coherence.rs (already `Serialize`) |
| `ConversationTurn` | `TurnAnalysis` | **neither as-is** — merge into one |
| `ConversationAnalysis` | `ConversationResponse` | **neither as-is** — merge |

**Action**: The domain types in `coherence.rs` already derive `Serialize`.
Use them directly from the API handler. Delete the DTO wrappers in `api.rs`
and the manual field-by-field conversion code (api.rs lines 207–260).

One exception: `PairwiseRelation.relation` is a `RelationType` enum. The
frontend expects a string like `"Opposed"`. Either:
- Add `#[serde(rename_all = "PascalCase")]` to `RelationType`, or
- Keep a thin `relation: String` in the response and format it there

#### Unused import / variable warnings
- `crates/got-incoherence/src/coherence.rs` line 247: `config` → rename to `_config`
- `crates/got-incoherence/src/visual.rs` line 229: `lookup` → rename to `_lookup`

---

## Detailed TODO Steps

### Phase 1: Fix the Geometry (make analysis actually work)

- [ ] **1.1** Delete `demo_gram_matrix()` from `crates/got-web/src/demo.rs`
- [ ] **1.2** In `api.rs` `load_demo_resources()`: compute Φ = EᵀE from embeddings
  - Load `PrecomputedEmbeddings`
  - Extract all 28 embedding vectors (32-d each)
  - Stack into matrix E (28×32), compute EᵀE (32×32)
  - Build `CausalGeometry::from_raw_gram(gram, 32)`
- [ ] **1.3** Verify: `cargo test -p got-incoherence` still passes (no changes to library)
- [ ] **1.4** Verify: `cargo build -p got-web` compiles

### Phase 2: Generate Message Embeddings

- [ ] **2.1** Write `scripts/generate_message_embeddings.py`
  - Input: `data/demo/embeddings.json` + message texts from demo
  - Output: `data/demo/message_embeddings.json` — array of 13 objects with
    `{"speaker": "...", "text": "...", "embedding": [...]}`
  - Blend logic: each message embedding = weighted sum of value term vectors
  - Advisor manipulation: blend contradictory values at low weight
- [ ] **2.2** Run the script, check output is sane (13 messages × 32-d)
- [ ] **2.3** Update `demo_conversation_json()` in `demo.rs`:
  - Remove `"values"` arrays from messages
  - Add `"embedding"` arrays (or `include_str!` the generated file)

### Phase 3: Value Detection via Causal Projection

- [ ] **3.1** Add `detect_values()` function to `api.rs`:
  ```
  fn detect_values(
      msg_embedding: &[f32],
      source: &PrecomputedEmbeddings,
      geometry: &CausalGeometry,
      terms: &[String],
      threshold: f32,
  ) -> Vec<DetectedValue>
  ```
  - For each term: compute cos_Φ(message, term)
  - Return terms above threshold with their cos_Φ score and polarity
- [ ] **3.2** Update `MessageInput` to accept `embedding: Vec<f32>` instead of
  (or in addition to) `values: Vec<String>`
- [ ] **3.3** In `analyse_conversation()`: if message has embedding, call
  `detect_values()` instead of reading the `values` array
- [ ] **3.4** Add detected values + scores to `TurnAnalysis` response so the
  frontend can show what was detected and with what confidence

### Phase 4: Remove Duplicate Types

- [ ] **4.1** Delete from `api.rs`:
  - `ContradictionDto` struct
  - `RedundancyDto` struct
  - `PairwiseDto` struct
  - All the `.iter().map(|c| ContradictionDto { ... })` conversion blocks
- [ ] **4.2** In `api.rs`: import and use `coherence::Contradiction`,
  `coherence::Redundancy`, `coherence::PairwiseRelation` directly
- [ ] **4.3** For `PairwiseRelation.relation: RelationType`:
  - Add `#[serde(serialize_with = "...")]` or derive a string representation
  - Or: keep one `relation_label: String` field on TurnAnalysis computed
    from `format!("{:?}", p.relation)`
- [ ] **4.4** Decide on `ConversationTurn` vs `TurnAnalysis`:
  - `ConversationTurn` (in coherence.rs) embeds full `CoherenceAnalysis`
  - `TurnAnalysis` (in api.rs) flattens it into top-level fields
  - **Keep `TurnAnalysis` in api.rs** (flat shape is better for JSON/frontend)
  - **Delete `ConversationTurn` and `ConversationAnalysis` from coherence.rs**
    only if nothing in `report.rs` / `visual.rs` needs them

#### ConversationTurn/ConversationAnalysis dependency check:
These types are used by:
- `report.rs` → `render_conversation(&ConversationAnalysis)` (line 92)
- `visual.rs` → `render_timeline(&ConversationAnalysis)` (line 360)
- `coherence.rs` → methods `score_series()`, `first_contradiction_turn()`,
  `final_score()`, `total_contradictions()`

**Decision**: Keep these types in `coherence.rs`. They serve the library's
report/visual renderers. The API doesn't need to use them — it has its own
flat response shape. This is fine. The duplication is intentional:
library types (rich, nested) vs API types (flat, serializable).

**Revised action**: Keep both hierarchies but don't duplicate individual
field types. Use `coherence::Contradiction` etc. directly in `TurnAnalysis`.

### Phase 5: Clean Up Dead Code

- [ ] **5.1** Delete `GET /api/terms` handler and route from `main.rs`
- [ ] **5.2** Fix unused variable warnings:
  - `coherence.rs:247` → `_config`
  - `visual.rs:229` → `_lookup`
- [ ] **5.3** Remove `available_terms` from `ConversationResponse` if the
  frontend doesn't need it (check index.html)

### Phase 6: Frontend Updates

- [ ] **6.1** Update `index.html` to show **detected** values (from geometry)
  instead of hand-tagged values
- [ ] **6.2** Show detection confidence (cos_Φ score) on value chips
- [ ] **6.3** Distinguish affirmed (+cos_Φ) vs negated (-cos_Φ) values visually
- [ ] **6.4** Remove any reference to `/api/terms` endpoint

### Phase 7: Build & Verify

- [ ] **7.1** `cargo test -p got-incoherence` — all existing tests pass
- [ ] **7.2** `cargo build -p got-web` — compiles clean
- [ ] **7.3** `cargo run -p got-web` — server starts
- [ ] **7.4** Load demo in browser — coherence score drops over conversation,
  contradictions appear, manipulation is visible in the geometry

### Tests: What to Keep, Remove, or Add

#### Existing tests — ALL KEEP
No existing tests need to be removed. The 49+ tests across the workspace
test the underlying maths, protocol, and storage — none of them are invalidated
by this refactor. Specifically:

| Test Module | Tests | Verdict | Reason |
|---|---|---|---|
| `coherence.rs` tests (10) | causal cosine, pairwise, contradictions, scoring | **KEEP** | Core maths, unchanged |
| `embeddings.rs` tests (7) | lookup, dimension checks, JSON parsing | **KEEP** | Embedding layer unchanged |
| `report.rs` tests (6) | text/JSON report formatting | **KEEP** | Report layer unchanged |
| `visual.rs` tests (7) | SVG rendering, colour mapping | **KEEP** | Visual layer unchanged |
| `lib.rs` tests (7) | end-to-end analyse_value_system | **KEEP** | Top-level API, unchanged |
| `integration.rs` tests (27+) | attestation, geometry, wire, store | **KEEP** | Unrelated subsystems |

#### New tests to ADD

- [ ] **T1** `api.rs` or `lib.rs`: test that `detect_values()` returns
  expected values when given a message embedding that's a blend of
  "honesty" + "transparency" vectors → should detect both terms
- [ ] **T2** `api.rs`: test that a message embedding blending "openness"
  with a small "secrecy" component → detects openness (positive),
  and secrecy weakly or negatively
- [ ] **T3** `demo.rs` or `api.rs`: test that full demo conversation
  produces declining coherence scores (not flat 1.0)
- [ ] **T4** `api.rs`: test dimension mismatch is handled (message
  embedding of wrong size → clear error, not silent fallback)

---

## File Change Summary

| File | Action |
|---|---|
| `crates/got-web/src/demo.rs` | Delete `demo_gram_matrix()`. Update `demo_conversation_json()` to use embeddings instead of hand-tagged values. |
| `crates/got-web/src/api.rs` | Compute Φ from embeddings. Add `detect_values()`. Remove DTO wrappers. Use library types directly. |
| `crates/got-web/src/main.rs` | Delete `/api/terms` route and handler. |
| `crates/got-web/static/index.html` | Show detected values with confidence. Remove `/api/terms` reference. |
| `crates/got-incoherence/src/coherence.rs` | Fix `_config` warning. Keep `ConversationTurn`/`ConversationAnalysis`. |
| `crates/got-incoherence/src/visual.rs` | Fix `_lookup` warning. |
| `scripts/generate_message_embeddings.py` | **NEW** — generates demo message embeddings. |
| `data/demo/message_embeddings.json` | **NEW** — generated output. |
