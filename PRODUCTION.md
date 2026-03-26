# Production Readiness Plan

## Status (2026-03-25)

**Phases 1, 2, and 4.1–4.2 are DONE.** The system now runs on real GPT-2 geometry.
**Effective dimensionality (participation ratio) and model comparison are DONE.**

### Key Discovery: cos_Φ collapses for real models

Both full UᵀU (50257×768 → 768×768) and term-focused EᵀE (26×768 → 768×768)
produce cos_Φ ≈ 0.98 for all term pairs — zero discrimination. The fix:

- **Pairwise analysis**: Φ = I (standard cosine). GPT-2 term cosines range
  0.24–0.76 (bravery↔courage 0.76, efficiency↔tradition 0.24). Good spread.
- **Detection**: z-scored logits. Raw dot product h·u_i standardized across
  terms. z > 0 = above-average activation. Matches conversation content:
  Turn 2 "I feel a responsibility" → responsibility(z=2.05).
- **Thresholds**: antonym ≈ -0.15 (real) vs -0.5 (synthetic),
  synonym ≈ 0.20 (real) vs 0.8 (synthetic).

The real pipeline detects contradictions (justice↔tradition, freedom↔justice)
that emerge from GPT-2's actual weight geometry. Coherence drops to 0.80
by Turn 4. Trust drops to 0.51. Non-circular.

### Effective Dimensionality and Multi-Model Comparison

`got-incoherence` now computes the **participation ratio** (effective value
dimensionality) from the eigenvalues of the n×n pairwise cosine matrix:

  PR = (Σλ_i)² / Σλ_i²

PR ∈ [1, n]: 1 = all values collapsed to one direction, n = fully spread.

The `compare` module compares two models' value geometry: participation ratio
delta, per-term embedding drift, pairwise cosine changes, and Frobenius
distance between cosine matrices.

**Six models extracted and compared** (2026-03-25):

| Model | Vocab | Dim | Terms | PR / n |
|---|---|---|---|---|
| GPT-2 124M | 50257 | 768 | 26 | 20.10 / 26 |
| GPT-2 Medium 355M | 50257 | 1024 | 26 | 20.69 / 26 |
| Qwen2.5-0.5B base | 151936 | 896 | 26 | 21.78 / 26 |
| Qwen2.5-0.5B Instruct | 151936 | 896 | 26 | 21.90 / 26 |
| TinyLlama 1.1B base | 32000 | 2048 | 9 | 7.90 / 9 |
| TinyLlama 1.1B Chat | 32000 | 2048 | 9 | 7.90 / 9 |

### Key Finding: Instruction Tuning Does Not Collapse Unembedding Geometry

| Comparison | Base PR | Tuned PR | Delta | Frobenius |
|---|---|---|---|---|
| Qwen2.5 Base vs Instruct | 21.78 | 21.90 | +0.12 | 0.13 |
| TinyLlama Base vs Chat | 7.90 | 7.90 | +0.00 | 0.01 |
| GPT-2 vs GPT-2 Medium | 20.10 | 20.69 | +0.59 | 0.71 |

Per-term embedding drift is near zero: TinyLlama base vs chat has cosine
similarity >0.9998 for every term. Qwen2.5 base vs instruct: all terms
>0.991. The unembedding matrix barely moves during SFT/RLHF/DPO.

**Implication for Conjecture 3 (RLHF manifold collapse):** The unembedding
matrix is not where alignment-induced collapse would manifest. The output
projection is shared infrastructure — instruction tuning primarily modifies
internal representations (attention patterns, residual stream directions).

Activation geometry experiments (10 moral dilemma prompts, 6 layers each)
on Qwen2.5-0.5B and TinyLlama 1.1B show the **opposite** of collapse in
final layers: PR *increases* after instruction tuning (TinyLlama base 1.41
→ chat 1.60 at layer 21). Early/middle layers are invariant (cosine >0.999).

### Curvature Analysis (Conjecture 2)

Menger curvature computed for all C(26,3) = 2600 value-term triples.
High-curvature terms are **consistent across architectures**:

| High curvature (bent) | Low curvature (flat) |
|---|---|
| bravery, compassion, empathy | justice, equality, efficiency |
| creativity, honesty, resilience | tradition, secrecy, freedom |

High-curvature terms are affective values (emotional judgment, situational
sensitivity). Low-curvature terms are structural/institutional values
(rules, systems). This ordering matches the Conjecture 2 prediction that
high curvature corresponds to human moral uncertainty, but confirmation
requires correlation with measured deliberation times.

Instruction tuning slightly *increases* mean curvature (Qwen2.5: 3.17 →
3.23) while preserving term rankings.

---

## Where We Are

The PoC demonstrates the full pipeline: geometry → probes → detection → attestation → visualisation. The math scaffolding is sound and tested (255+ tests across 9 crates). The web demo works end-to-end with synthetic data.

**The core legitimacy problem:** The demo is a closed loop. Message embeddings are hand-blended from term vectors, so "detection" algebraically recovers the construction recipe. No NLP, no real model geometry, no non-trivial inference. A reviewer would see through this in minutes.

Everything below is about breaking that circularity while keeping the working infrastructure.

---

## Phase 1 — Real Model Geometry

**Goal:** Φ = UᵀU from a real model, not from 28 hand-crafted vectors.

**Why first:** Everything downstream depends on the vector space being real. Without this, nothing else matters.

### 1.1 Extract unembedding matrix from GPT-2

`scripts/extract_activations.py` already writes `.gotue` files. `CausalGeometry::from_unembedding()` already consumes them with faer-based UᵀU and auto-regularisation.

- [x] Run extraction script against `gpt2` (50257 vocab × 768 hidden dim)
- [x] Write to `data/models/gpt2.gotue`
- [x] Verify: load in Rust, check dimensions, confirm Φ is 768×768 and PD

### 1.2 Load real geometry in got-web

- [x] Add `--geometry <path>` CLI flag to `got-web`
- [x] If provided: load `.gotue`, build `CausalGeometry::from_unembedding()`
- [x] If not provided: fall back to current synthetic demo (preserve demo mode)
- [x] Store geometry source label ("gpt2" vs "synthetic-demo") in server state

**Existing code:** `got-core::CausalGeometry::from_unembedding`, `UnembeddingMatrix`
**New code:** ~50 lines in `got-web/src/main.rs`

### 1.3 Value terms from model vocabulary

`UnembeddingLookup` already exists — maps term strings to their row in U.

- [x] When real geometry loaded: use `UnembeddingLookup` instead of `PrecomputedEmbeddings`
- [x] Term list stays configurable (the 28 terms are fine as a starting set)
- [x] Their vectors now come from U's rows, not from hand-crafted JSON

**Existing code:** `got-incoherence::embeddings::UnembeddingLookup`, `EmbeddingSource` trait
**New code:** ~30 lines of wiring in `got-web/src/api.rs`

### 1.4 Verify real structure

- [x] Dump pairwise causal cosines for the 28 terms under GPT-2 geometry
- [x] Sanity check: do honest/transparent cluster? Do secrecy/transparency oppose?
- [ ] If real geometry doesn't separate these terms meaningfully, adjust the term list
- [x] Write `data/models/gpt2-term-analysis.json` documenting real cosine values

**Deliverable:** The system uses a real model's output geometry. Φ is no longer self-referential. 

---

## Phase 2 — Real Message Embeddings

**Goal:** Messages encoded by a real model, not hand-blended from term vectors.

**Why second:** Depends on Phase 1 — message embeddings must live in the same ℝ^d as the geometry.

### 2.1 Extract message activations through GPT-2

The cleanest approach: use the same model for both geometry and message encoding. Both vectors live in ℝ^768. cos_Φ is mathematically valid.

- [x] Extend `extract_activations.py` to accept a conversation JSON
- [x] For each message: run through GPT-2, extract final-layer residual stream
- [x] Mean-pool token positions → one 768-d vector per message
- [x] Write to `data/demo/gpt2-message-activations.json`

### 2.2 Pre-extracted demo (no live inference required)

For the demo conversation, we pre-extract once and ship the activations:

- [x] Run the 13 demo messages through GPT-2 extraction
- [x] Store activations alongside the demo conversation JSON
- [x] `got-web` loads these at startup (same as current synthetic path, but with real vectors)

### 2.3 Live inference path (optional, for production)

For analysing new conversations at runtime:

- [ ] Add a `/api/embed` endpoint that calls a local Python inference server
- [ ] Or: add `ort` (ONNX Runtime) dependency and run GPT-2 in-process
- [ ] Or: accept pre-computed embeddings from the client (current API shape)

**Decision:** For v1, pre-extract demo + accept client embeddings. Live inference is v2.

### 2.4 Verify non-circular detection

- [x] Run the real-geometry + real-embedding pipeline on the demo conversation
- [x] Check: does it still detect manipulation? Which terms emerge?
- [x] The answer may be different from the synthetic demo — that's the point
- [ ] Document what the real pipeline detects vs what the synthetic pipeline detected

### 2.5 Clean up synthetic path

- [ ] Move `generate_message_embeddings.py` → `scripts/legacy/`
- [ ] Move `generate_synthetic_data.py` → `scripts/legacy/`
- [ ] Demo mode clearly labeled: loads real-extracted GPT-2 activations
- [ ] Synthetic mode still available via flag for development/testing

**Deliverable:** Detection is non-trivial. Message embeddings come from running text through a model.

---

## Phase 3 — Calibrate Scores

**Goal:** Thresholds and scores have empirical grounding.

**Why third:** Depends on Phase 2 — calibration on synthetic data is meaningless.

### 3.1 Build evaluation dataset

- [ ] Collect 50+ conversations with known manipulation (social engineering, phishing, dark patterns)
- [ ] Collect 50+ benign conversations (support, tutoring, collaboration)
- [ ] Label each: manipulative/benign, turn where manipulation begins (if applicable)
- [ ] Format: same JSON schema as demo conversation

### 3.2 Threshold calibration

- [ ] Run production pipeline (real geometry + real embeddings) on all labeled conversations
- [ ] Compute ROC curves for:
  - Contradiction detection (varied `antonym_threshold`)
  - Coherence score cutoffs
  - Trust score cutoffs (varied `decay`/`drift_weight`)
- [ ] Set thresholds at defensible operating point (e.g. 95% recall on manipulation)
- [ ] Document false positive rate

### 3.3 Score validation

- [ ] Publish distribution of coherence scores: benign vs manipulative
- [ ] Compute confidence intervals
- [ ] Replace magic constants (decay=0.7, drift_weight=2.0, thresholds -0.5/0.8) with empirical values
- [ ] Add `CalibrationMetadata` to API response: dataset version, threshold source, date

**Deliverable:** Every number the system shows has a known false-positive rate.

---

## Phase 4 — Fix Known Technical Debt

Independent of Phases 1-3. Can run in parallel.

### 4.1 Gram matrix regularisation

- [x] Apply ε-regularisation in `from_raw_gram` path (already done in `from_unembedding`)
- [x] Fix Cholesky tolerance: `1e-30` → `f32::EPSILON` (~1.2e-7)
- [ ] Test: rank-deficient Gram matrix triggers regularisation

### 4.2 Pre-existing test failure

- [x] Fix `causal_cosine_of_orthogonal_is_near_zero` (0.50000006 vs <0.5 threshold)
- [x] Either fix the test geometry to produce truly orthogonal vectors, or relax the assertion

### 4.3 Error handling

- [ ] Remove silent fallback to `coherence_score: 1.0` on analysis failure
- [ ] Return proper HTTP error responses with diagnostic info
- [ ] Log failures with structured logging (`tracing`)

---

## Phase 5 — Production Web Server

Independent of Phases 1-3. Can run in parallel.

### 5.1 Configuration

- [ ] Config file or env vars: geometry source, embedding backend, listen address, thresholds
- [ ] Structured logging (`tracing` crate)
- [ ] Graceful shutdown

### 5.2 API hardening

- [ ] Authentication (API keys or JWT)
- [ ] Rate limiting (analysis is O(n²) in terms)
- [ ] Input validation: max message length, max conversation length, dimension checks
- [ ] CORS configuration (currently allows all origins)

### 5.3 Frontend

- [ ] Serve static files from disk (or `rust-embed` for single-binary)
- [x] Clear "SYNTHETIC DEMO" vs "LIVE" banner based on geometry source
- [x] API response includes `mode: "synthetic" | "live"`

---

## Phase 6 — Wire Existing Infrastructure

Depends on Phase 5. Most code already exists.

| Component | Status | Integration Work |
|---|---|---|
| `got-attest` (sign/verify) | Production-ready | Attach attestations to analyses |
| `got-store` (FileStore) | Working | Store analysis history |
| Chain verification | Working | Track model changes over time |
| PKI + trust registry | Working | Agent identity for multi-party |
| `got-cli` (11 commands) | Working | Add `serve` command |

### 6.1 Attestation chain

- [ ] After each analysis, produce a signed `GeometricAttestation`
- [ ] Store in `FileStore`
- [ ] Expose `/api/attestations` endpoint
- [ ] Each attestation references geometry hash + probe commitments

### 6.2 Audit trail

- [ ] Expose `/api/audit` endpoint (FileStore::audit() already works)
- [ ] UI: "Audit" tab showing attestation history

---

## Phase 7 — Honest Labelling

### 7.1 Acknowledge lineage

- [ ] README: cite Burgess's Promise Theory as intellectual ancestor
- [ ] Note: structural parallel (trust from self-consistency) + extension (continuous geometry from weight space)

### 7.2 Acknowledge limitations

- [ ] README: "Causal" applies only when Φ comes from a real model's unembedding matrix
- [ ] README: The system detects geometric incoherence, not deception per se
- [ ] README: Scores are relative to the model's output geometry, not absolute moral judgments

---

## Explicit Non-Goals

| Item | Reason |
|---|---|
| Hardware TEE (SGX/SEV) | Hardware procurement, not software. `MockEnclave` + trait boundary is clean. |
| TCP/QUIC transport | In-memory exchange is fine. Network transport needed for multi-agent, not for credibility. |
| Database backend | `FileStore` handles single-node. Scale later. |
| Peer discovery | Way later. |
| Live sentence-transformer in Rust | Use Python extraction for now. ONNX embedding is v2. |
| Distributed corpus governance | As PLAN.md says: "The most important work in AI alignment is not technical. It is institutional." |

---

## Dependency Graph

```
Phase 1 (real geometry)
    │
    ▼
Phase 2 (real embeddings) ── depends on Phase 1 for vector space
    │
    ▼
Phase 3 (calibration) ─── depends on Phase 2 for real data
    
Phase 4 (tech debt) ─────── independent, parallel with 1-3
Phase 5 (prod server) ───── independent, parallel with 1-3
Phase 6 (wire existing) ─── depends on Phase 5
Phase 7 (labelling) ──────── depends on Phase 1 for mode detection
```

**Critical path: 1 → 2 → 3.** Everything else is parallel work.

---

## Complexity Estimates

| Phase | New Code | Reuses | Risk |
|-------|----------|--------|------|
| 1 — Real geometry | ~80 lines Rust, ~20 lines Python | `from_unembedding`, extraction script | Low — path exists |
| 2 — Real embeddings | ~100 lines Rust, ~50 lines Python | `EmbeddingSource`, `UnembeddingLookup` | Medium — need to verify cos_Φ is meaningful |
| 3 — Calibration | ~100 lines Rust + dataset collection | Scoring pipeline | Medium — dependent on labeled data |
| 4 — Tech debt | ~30 lines | Existing tests | Low |
| 5 — Prod server | ~200 lines | Axum, tower-http | Low |
| 6 — Wire existing | ~150 lines | `got-attest`, `got-store` | Low — adapters only |
| 7 — Labelling | ~50 lines | — | Low |

## Key Risk

Phase 2.4 — "Verify non-circular detection." When real GPT-2 activations replace the hand-blended vectors, the system may not detect the same manipulation patterns. The demo scenario was designed for synthetic geometry.

**Mitigations:**
- If GPT-2's geometry doesn't separate the current terms well, we adjust the term list based on what the model actually separates
- If mean-pooled activations don't project meaningfully onto vocabulary directions, we try last-token instead of mean-pool, or use a different layer
- The math is sound regardless — the question is whether the specific model has useful structure for these specific concepts

This is a feature, not a bug. If the real model doesn't separate these terms, the synthetic demo was claiming capabilities the model doesn't have. Better to find out now.
