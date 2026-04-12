# Data Flow Architecture

End-to-end data flow through an agent's internal pipelines and the
agent-to-agent attestation exchange. Follows the plan's pipeline:
**deterministic geometry → signed attestation → independent
reproducibility → causal proof → agent exchange**.

All flows reflect the security-hardened codebase (353 tests passing).

---

## Agent-Internal Pipeline

```
  EXTRACTION                       BINARY FILES
  (Python scripts,                 (consumed by Layer 0)
  step 7 of 12)
  =============                    ==============

  Agent's own model                .gotact (GOTA magic)
  (HuggingFace or                   layer x pos x d f32 LE
   direct weights)                       |
       |                           .gotue (GOTU magic)
       +-- Tokenizer                V x d f32 LE
       |       |                         |
       v       v                         |
  Forward Pass + Hooks             .labels
       |                             0/1 per line
       |   Unembedding extract           |
       |       |                         |
       v       v                         |
   .gotact   .gotue                      |
       |       |                         |
       +---+---+                         |
           |                             |
           v                             v
  =====================================================
   GEOMETRY & CHECKPOINT
  =====================================================

  Why: The Gram matrix Phi = U^T U defines the causal inner product.
  All downstream measurement (probes, drift, causal checks) is
  performed in *this* geometry, not in Euclidean space.

   load_unembedding(.gotue)   --->  UnembeddingMatrix
                                         |
                                         v
                              CausalGeometry::from_unembedding(U, eps)
                                  Phi = U^T U  (+eps*I if rank-deficient)
                                         |
                             +-----------+-----------+
                             |                       |
                             v                       v
                      geometry_hash()          save .gotgeo
                      H(Phi) [u8;32]         (reference checkpoint)
                             |                       |
                             v                       |
  =====================================================
   PROBE TRAINING                                    |
  =====================================================
                                                     |
  Why: Linear probes trained under the causal inner product
  measure whether a direction in the residual stream causally
  encodes a property. The plan calls this "the geometry side".
                                                     |
   load_activations(.gotact)  --->  Vec<LayerAct>    |
                                         |           |
                                         v           |
                              Precompute Phi*h        |
                              for all samples        |
                                         |           |
                    labels.txt ------->   |           |
                                         v           |
                              SGD loop (epochs x samples)
                                logit = w^T(Phi*h) + b
                                pred  = sigma(logit)
                                error = pred - y
                                w <- w - lr * error * Phi*h
                                b <- b - lr * error
                                         |
                                         v
                              ProbeVector { w, b, platt, thresh }
                                         |
                                         v
                              ProbeSet { probes, layer,
                                geometry_hash: H(Phi),
                                max_drift: <policy> }
                                         |
                                         v
                              probes.json (persisted)
                                         |
  =====================================================
   SELF-ATTESTATION (Two-Tier Cost Model)            |
  =====================================================
                                         |
  Why: The attestation is a signed, deterministic record of what
  the probes measured. It is independently reproducible (Tier 3).
  Trust tiers are *content-based* — they are derived from which
  fields the attestation populates, not from a schema version:
    Tier 1 = any signed attestation
    Tier 2 = `parent_attestation_hash.is_some()` (chained)
    Tier 3 = non-empty `causal_scores` with every record causal

  KEY INSIGHT: Probe readings depend on input activations, so
  signed attestations CANNOT be cached. What CAN be cached are
  the expensive model invariants that only change on model update.
  This separation is enforced by ModelContext (got-net::attestation_cache).

  Hardening (defence-in-depth):
    S-7:  timestamp must be ≤ now + 300 s
    S-13: model_id, corpus_version, probe_version ≤ 256 bytes
    S-20: layer_readings ≤ 1024 layers, total readings ≤ 65536
    S-21: model_hash is Option<[u8; 32]> (None if absent)

  ------- TIER 1: Cached Invariants (ModelContext) -------
  |                                                      |
  |  Loaded from ModelContext::get() if ready,           |
  |  otherwise recomputed and stored via update().       |
  |  Invalidated on: startup, model update,              |
  |    detect_distribution_shift(), manual trigger.       |
  |                                                      |
  |  CausalGeometry Phi = U^T U       O(Vd^2) to compute|
  |    geometry_hash: H(Phi) [u8;32]                     |
  |                                                      |
  |  Trained probe weights             SGD under causal  |
  |    bound to geometry_hash          inner product     |
  |    probes.json (persisted)                           |
  |                                                      |
  |  If model updated (chained):                         |
  |    load reference .gotgeo -> Geometry_ref            |
  |    drift = drift_from(current, ref)                  |
  |    if drift > max_drift: invalidate + retrain        |
  |    parent_attestation_hash = H(prev)                 |
  |    geometry_drift = drift                            |
  |                                                      |
  |  Causal validation results         model forward     |
  |    causal_check per probe          passes (expensive)|
  |    causal_scores: [...]                              |
  |                                                      |
  |  All stored in CachedInvariants:                     |
  |    { geometry, probe_weights, causal_scores,          |
  |      geometry_hash, parent_attestation_hash,         |
  |      geometry_drift, computed_at, model_id,          |
  |      model_hash }                                    |
  --------------------------------------------------------
                              |
                              v
  ------- TIER 2: Per-Attestation (Fresh Every Time) -----
  |                                                      |
  |  These depend on input context — NEVER cached.       |
  |                                                      |
   load_activations(.gotact) ----+
   current CausalGeometry ----+  |  (from ModelContext)
   probe_weights ----------+  |  |  (from ModelContext)
                           |  |  |
                           v  v  v
              For each probe in each layer:
                raw  = inner_product(w, h) + bias
                conf = sigma(platt_scale * raw + platt_shift)
                flag = conf < threshold
                              |
                              | (read_probe / read_probe_checked)
                              |
                              v
        merkle_root() -----+  |  +------ sha256(act_bytes)
        (weight shards)    |  |  |       (input hash)
                           v  v  v
              Fill GeometricAttestation struct
              { schema_version: SCHEMA_VERSION,   (always 1)
                model_hash: Option<[u8; 32]>,    ← S-21
                parent_attestation_hash,         ← from ModelContext
                geometry_hash: H(Phi),           ← from ModelContext
                geometry_drift,                  ← from ModelContext
                causal_scores: [...],            ← from ModelContext
                intervention_delta: Some(delta),
                causal_flag: Some(all_pass),
                sequence_number,                  ← Phase 13
                directional_drifts: [...],        ← Phase 13
                probe_commitment: Some(H(...)),   ← Phase 13
                density_reading / curvature_reading (manifold),
                domain_scope_declaration (§2.1 binding),
                layer_readings, confidence,       ← FRESH (never cached)
                coverage_flags, ... }             ← FRESH (never cached)
                              |
                              v
              assemble_and_sign(attestation, sk)
                S-7 / S-13 / S-20 gates pass
                serialise_for_signing() → canonical LE bytes
                Ed25519 sign(bytes, agent's secret_key)
                              |
                              v
              Signed GeometricAttestation
              (held in memory / persisted / stored)
              (NEVER cached — each signing is unique to input)
  --------------------------------------------------------


  =====================================================
   HARDWARE ENCLAVE PIPELINE (alternative to above)
  =====================================================

  Why: If the agent runs in a TEE (SGX, SEV, H100), the signing
  key never leaves the enclave. This makes the attestation
  tamper-evident even against the host OS.

  NOTE: The current MockEnclave runs in the same address space as
  the agent. Probes, signing key, and geometry are all accessible
  to the host process. Real TEE integration (step 12 of the build
  order) would enforce hardware isolation — the agent runtime could
  not read the probes or signing key. Until then, the mock validates
  the protocol flow but not the security boundary.

   Hardware (GPU/DMA)          Enclave (TEE)
        |                          |
        |-- capture(layer, pos, values)
        |   compute integrity_hash |
        |   = SHA-256(layer | pos | values)
        |                          |
        |-- ActivationFrame ------>|
        |                          |
        |   receive_activations(): |
        |     recompute hash       |
        |     verify integrity     |
        |     store frame          |
        |                          |
        |   run_causal_check():    |
        |     for each probe:      |
        |       causal_check(...)  |
        |     -> Vec<CausalScore>  |
        |                          |
        |   attest_with_causal():  |
        |     embed causal scores  |
        |     before signing       |
        |     -> signed attestation|
        |     (key stays in enclave)
        |                          |

   enclave_pipeline() orchestrates the full flow:
     capture -> receive -> causal -> attest_with_causal


  =====================================================
   ATTESTATION STORAGE
  =====================================================

  Why: Signed attestations are persisted in a content-addressed
  store so that any party can later audit a model's full history
  and verify chain integrity.

   Signed attestation
        |
        v
   AttestationStore::append(attestation, verifying_key)
        |
        +-- verify signature (Ed25519)
        +-- compute StoreId = SHA-256(canonical bytes)
        +-- persist (MemoryStore or FileStore)
        |     FileStore: atomic write (temp + rename)
        |     FileStore: hash-on-load integrity check
        |
        v
   Queryable store:
     store.chain(model_id)  -> ordered attestation list
     store.query(filter)    -> filtered results
     store.audit(model_id)  -> AuditReport {
       chain_length, chain_valid,
       drift_summary { max_drift, mean_drift },
       causal_summary { pass/fail_count, mean_consistency },
       signers, timestamps
     }


  =====================================================
   AGENT-TO-AGENT EXCHANGE
  =====================================================

  Why: Two agents must verify each other's alignment properties
  before cooperating. The exchange is symmetric — both sides
  produce and verify attestations. This is the protocol's
  ultimate output: a trust decision backed by deterministic
  geometry, signed attestation, and (optionally) causal proof.

  Security hardening in the exchange path:
    S-8:  verify_chain accepts &[VerifyingKey] for key rotation
    S-9:  envelope has verified flag + from_bytes_verified()
    S-2:  TrustRegistry verified by SHA-256 on load()
    N-1:  Frame::encode returns Result (payload size guard)

              Agent A                          Agent B
                |                                |
                |   self-attest (pipeline above) |
                v                                v
         attest_A (signed)              attest_B (signed)
         + chain [attest_0..A]          + chain [attest_0..B]
                |                                |
                v                                v
         build_request(                 build_response(
           nonce, id_B,                   nonce, id_A,
           key_A, chain, current)         key_B, verdict,
                |                         chain, current)
                v                                v
         ExchangeRequest {              ExchangeResponse {
           agent_id: id_A,               agent_id: id_B,
           envelope: signed,              envelope: signed,
           chain: [...],                  verdict: Accepted/Rejected,
           current: attest_A              chain: [...],
         }                                current: attest_B
                |                         }
                +---------> exchange <-----------+
                |          (channel)             |
                v                                v
         received: rsp                  received: req
                |                                |
                v                                v
         validate_response(             validate_request(
           rsp, id_A, nonce,              req, id_B, nonce,
           registry)                      registry)
                |                                |
   Phase 4 — Domain re-verify (defence in depth):          Phase 4 — Domain re-verify (defence in depth):
     check_domain_                    check_domain_
       compatibility(                   compatibility(
         peer_scope,                      peer_scope,
         self_scope)                      self_scope)
     §4 / Appendix B                  §4 / Appendix B
     Supervised pair OK (§5.5)        Supervised pair OK (§5.5)
     STRUCTURAL — runs                STRUCTURAL — runs
     before envelope                  before envelope
                |                                |
   §7.3/§8.2 Governance:            §7.3/§8.2 Governance:
     effective_thresholds +           effective_thresholds +
     enforce_governance               enforce_governance
     (max_drift, confidence,          (max_drift, confidence,
      require_chain,                   require_chain,
      require_causal_validation)      require_causal_validation)
                |                                |
   §2.1 Scope binding:              §2.1 Scope binding:
     check_attestation_               check_attestation_
       scope_binding                    scope_binding
                |                                |
   Envelope verify:                 Envelope verify:
     Ed25519 sig check                Ed25519 sig check
     peer_agent_id match              peer_agent_id match
     attestation_hash match           attestation_hash match
     chain_root_hash match            chain_root_hash match
     timestamp freshness              timestamp freshness
     (S-9: verified flag set)         (S-9: verified flag set)
                |                                |
   Chain verify (if chained):       Chain verify:
     verify_chain(chain,               verify_chain(chain,
       current, pks, max_drift)          current, pks, max_drift)
       ↑ S-8: &[VerifyingKey]           ↑ S-8: &[VerifyingKey]
     -> ChainVerdict {                 -> ChainVerdict {
         length, max_drift }               length, max_drift }
                |                                |
                v                                v
         (Verdict, reason)              (Verdict, reason)


  =====================================================
   VERIFICATION (receiving agent)
  =====================================================

  Why: Verification implements the plan's three trust tiers.
  Tier 1 checks the signature. Tier 2 checks consistency
  (coverage, confidence, drift). Tier 3 reproduces the full
  pipeline on the same model + input and demands bitwise match.

         Received attestation
         Sender's public key (from TrustRegistry)
           TrustRegistry loaded with S-2 SHA-256 integrity
                              |
                              v
              For each link in chain (if chained):
                check parent_attestation_hash linkage
                check geometry_drift <= accepted_max
                check model_id consistency across chain
                check signature against &[VerifyingKey] (S-8)
                              |
                              v
              Check schema_version == SCHEMA_VERSION
                              |
                              v
              serialise_for_signing(attestation)
              (reject NaN/Inf, canonicalise -0.0)
                              |
                              v
              Ed25519 verify(bytes, signature, sender_pk)
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                 VALID              INVALID
              (cooperate)        (refuse peer)
```

---

## Pipeline Details

### 1. Extraction (Model Weights → Binary Files)

The Python extraction script (~50 lines, step 7 of 12 in the build
order) reads the unembedding matrix U and residual-stream activations
h out of a HuggingFace model. This produces the .gotact and .gotue
binary files that the rest of the pipeline consumes.

**Architecture auto-detection**: GPT-2 (`transformer.h`), LLaMA/Mistral (`model.layers`), OPT (`model.decoder.layers`), GPTNeoX/Pythia (`gpt_neox.layers`).

### 2. Geometry & Checkpoint (Unembedding → Φ → .gotgeo)

Computes the Gram matrix Φ = UᵀU defining the causal inner product.
The geometry hash H(Φ) is a deterministic fingerprint that binds probes
to the specific model they were trained against.

### 3. Probe Training (Activations + Geometry → ProbeSet)

Trains linear probes via SGD under the causal inner product. The ProbeSet
records which geometry it was trained against (`geometry_hash`) and a
maximum drift threshold (`max_drift`).

### 4. Self-Attestation (Probes + Activations → Signed Attestation)

All attestations share a single canonical wire format
(`SCHEMA_VERSION == 1`). Trust tiers are *content-based* — the verifier
derives the tier by inspecting which fields are populated:
- **Tier 1 — Signature**: any attestation with a valid Ed25519 signature.
- **Tier 2 — Consistency + Chain**: `parent_attestation_hash.is_some()`
  and chain drift within governance bounds.
- **Tier 3 — Causal Proof**: non-empty `causal_scores` with every
  record having `is_causal == true`. The KEYSTONE — proves the probed
  directions are causally linked, not surface correlations.

Defence-in-depth gates in `assemble_and_sign()`:
- S-7: timestamp ≤ now + 300 s
- S-13: string fields ≤ 256 bytes
- S-20: ≤ 1 024 layers, ≤ 65 536 readings
- S-21: `model_hash` is `Option<[u8; 32]>` (not a sentinel)

### 5. Hardware Enclave Pipeline (Alternative Attestation Path)

When running inside a TEE, the signing key never leaves the enclave.
`enclave_pipeline()` orchestrates: capture → receive → causal_check → attest_with_causal.

### 6. Peer Verification (Received Attestation + Trust → Decision)

Implements the Verifier role across three trust tiers:
- Envelope verification: Ed25519 sig, peer_agent_id, attestation_hash, chain_root_hash, timestamp
- Chain verification: `verify_chain()` with `&[VerifyingKey]` (S-8 key rotation), drift bounds, model_id consistency
- Trust registry: S-2 SHA-256 integrity, `expected_model_hash` binding, `max_attestation_age_secs`

### 7. Attestation Storage & Audit

Content-addressed storage (`StoreId` = SHA-256 of canonical bytes).
`FileStore` uses atomic writes (temp + rename) and hash-on-load integrity.
`AuditReport` provides chain validity, drift summary, causal summary, signer list.

### 8. Proxy Pipeline (Closed-Source Model Monitoring)

For models where internals are inaccessible (GPT-4, Claude, Gemini).
The proxy builds its own behavioral value space by embedding model outputs
and value anchors through the same embedding model, ensuring consistent
measurement within a single embedding space.

**Session creation** embeds all value terms through the configured embedding
model (e.g. Ollama's nomic-embed-text), creating anchors in that space.
The proxy uses Φ = I (standard cosine) since it operates in the embedding
model's space, not a reference model's causal geometry.

```
  POST /api/proxy/session
       |
       ├── embedding_url + embedding_model configured
       ├── For each value term:
       │     POST {embedding_url}/api/embeddings → anchor vector
       ├── Build PrecomputedEmbeddings from anchors
       ├── Φ = I (identity — plain cosine in embedding space)
       └── Return { session_id, geometry_hash }

  User types message in browser
       |
       v
  POST /api/chat  ──────────>  LLM Provider (Ollama / OpenAI / Anthropic)
       |                              |
       |  <──── AI response text ─────┘
       |
       v
  POST /api/proxy/session/:id/observe  { text: "...", speaker: "assistant" }
       |
       ├── Embed text via same embedding model:
       │     POST {embedding_url}/api/embeddings → [f32; dim]
       │
       ├── cosine(embedding, term_emb) for each value term
       ├── Detected values: terms with cosine > threshold (default 0.3)
       ├── ALL scores recorded for baseline tracking
       ├── Welford update: TermProfile.update(score, α)
       ├── EWMA update for recency weighting
       ├── pairwise cosines → PairwiseBaseline.update()
       |
       ├── IF observation_count ≥ min_observations (baseline sufficient):
       │     Signal 1: fraction of terms with |z-score from baseline| > 2.5σ
       │     Signal 2: 1 − cosine(current_EWMA_profile, baseline_profile)
       │     Signal 3: fraction of pairs shifted > 2.5σ from baseline
       │     Signal 4: manifold density (off-manifold detection, optional)
       │     Combined: w1×S1 + w2×S2 + w3×S3 + w4×S4
       │     → WithinBaseline (<0.3) | Drifting (0.3–0.6) | Deviated (≥0.6)
       │
       └── Return { detected_values, deviation }

  POST /api/proxy/session/:id/snapshot
       |
       ├── BehavioralValueSpace.hash() → SHA-256 of snapshot
       ├── Build BehavioralAttestation { schema: "B1", ... }
       ├── serialise_for_signing() → canonical bytes
       ├── Ed25519 sign → signature
       └── Chain: parent_hash = previous attestation hash
```

**Value Taxonomy**: Value terms can be configured via a TOML file (`--values`).
Each entry has a `name`, `description`, and optional `cluster`/`antonyms`.
In reference model mode, descriptions are embedded by averaging token vectors
from the unembedding matrix. In proxy mode with an external embedding model,
term names are embedded through the same API used for observations.

### 9. Activation Server Pipeline (Real Residual Stream Activations)

For live chat with real intermediate-layer hidden states instead of
bag-of-words token averaging. The activation server is a Python FastAPI
sidecar that loads the model via HuggingFace transformers and registers
hooks to capture residual stream activations at a configured layer.

**Why intermediate layers?** The unembedding matrix Φ = UᵀU collapses
value dimensions at the output layer (dim_eff = 1.1/13 for Qwen3.5 —
all value terms map to the same "fluent English" direction). Middle
layers partially recover value structure: layer 16 with last-token
pooling gives dim_eff = 3.16/13 (3 effective dimensions vs 1).

**Why last-token pooling?** Mean-pooling across all positions destroys
the value signal — function words (the, of, and) dominate the average.
The last token position carries the contextualized meaning of the full
input, preserving value-specific information.

```
  Python sidecar (scripts/activation_server.py)
       |
       ├── Loads model via HuggingFace (4-bit quantized, ~5GB VRAM)
       ├── Registers forward hook on target layer (e.g. layer 16/36)
       ├── POST /hidden_states  { text: "..." }
       │     → tokenize → forward pass → hook captures layer 16 output
       │     → last-token position → (4096,) contextualized residual stream vector
       │     → Return { hidden_state: [f32 x 4096], layer, n_tokens }
       │
       └── POST /v1/chat/completions  (OpenAI-compatible)
             → full generation → response text

  got-web (Rust, --activation-server http://localhost:8100)
       |
       ├── POST /api/embed  { text: "..." }
       │     → calls activation server /hidden_states
       │     → returns real residual stream activation (4096d)
       │     → fallback: bag-of-words from .gotue vocabulary
       │
       ├── Value detection uses the hidden state:
       │     cos_Φ(hidden_state, term_embedding) for each value term
       │     where Φ = UᵀU from the model's unembedding matrix
       │
       └── The same Φ weights the inner product — now measuring
           how much the model's intermediate representation projects
           onto value directions through the output distribution
```

**Key findings**:
- Bag-of-words from the unembedding matrix: dim_eff = 1.1/13 (8%), all values identical
- Layer 16 mean-pooled: still collapsed (function words dominate the average)
- Layer 16 last-token: dim_eff = 3.16/13 (24%), 3 effective value dimensions recovered
- UᵀU geometry should NOT be used with intermediate layers (it's output-specific);
  use Φ = I (standard cosine) instead
- Value descriptions separate better than single value term names (cosine 0.47 vs 0.67
  for compassion/cruelty descriptions vs single words)

### 10. Manifold Collapse and Model Comparison

New metrics for characterising value geometry:

```
  POST /api/collapse
       |
       ├── Collect probe directions W from term embeddings
       ├── Compute G_W = WᵀΦW (k×k projected Gram matrix)
       ├── Eigendecompose G_W → λ₁ ≥ λ₂ ≥ ... ≥ λₖ
       ├── Participation ratio: dim_eff = (Σλᵢ)² / Σλᵢ²
       └── Return { k, eigenvalues, dim_eff, assessment }

  POST /api/compare
       |
       ├── Load second .gotue → build Φ_B = U_B^T U_B
       ├── Global: d(A,B) = ‖Φ_A - Φ_B‖_F / max(‖Φ_A‖_F, ‖Φ_B‖_F)
       ├── Per-probe: d_w = |wᵀ(Φ_A - Φ_B)w| / max(|wᵀΦ_Aw|, |wᵀΦ_Bw|)
       ├── Probe-projected: d_V = mean(d_w)
       └── Return { global_distance, probe_projected_distance, per_probe, ratio }

  POST /api/coherence
       |
       ├── Resolve term names → embedding vectors
       ├── For each message embedding h:
       │     C(h) = (1/n) Σ σ(α · (⟨u_dom, h⟩_c - ⟨u_sub, h⟩_c))
       ├── Track violations: positions where C(h) < 0.5
       └── Return { per_message, mean, min, max, violated }
```

---

## File Formats

### .gotact (Activations)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `GOTA` |
| 4 | 2 | Version (u16 LE) |
| 6 | 4+n | Model ID (u32 LE length + UTF-8) |
| — | 1 | Precision tag (0=fp32, 1=fp16, 2=bf16, 3=int8) |
| — | 4 | hidden_dim d (u32 LE) |
| — | 4 | num_layers (u32 LE) |
| — | 4 | num_positions (u32 LE) |
| — | var | Per layer: layer_index(u32) + per position: pos(u32) + d × f32 LE |

### .gotue (Unembedding)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `GOTU` |
| 4 | 2 | Version (u16 LE) |
| 6 | 4 | vocab_size V (u32 LE) |
| 10 | 4 | hidden_dim d (u32 LE) |
| 14 | V×d×4 | Data: V × d f32 LE row-major |

### .gotgeo (Geometry Checkpoint)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `GOTG` |
| 4 | 2 | Version (u16 LE) |
| 6 | 4 | hidden_dim d (u32 LE) |
| 10 | 32 | geometry_hash (SHA-256 of Φ data) |
| 42 | d×d×4 | Data: d × d f32 LE row-major Gram matrix Φ |
