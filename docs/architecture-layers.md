# Layer Architecture

Dependency stack from core types up to agent orchestration.
The top layer can be either the CLI (for human-operated setup) or
an agent runtime (for autonomous agent-to-agent operation).
Layers 0–5 are identical in both modes.

The pipeline implements three **content-based** trust tiers. The tier
an attestation belongs to is derived from which fields it populates —
it is not a schema-version switch:
- **Tier 1 — Signature**: any valid `got_attest::verify`. Ed25519 over
  deterministic canonical bytes.
- **Tier 2 — Consistency + Chain**: `parent_attestation_hash.is_some()`
  and chain drift within governance-defined bounds. Verified by
  `got_wire::chain::verify_chain` against the effective `max_drift`
  from the verifier's `GovernanceThresholds`.
- **Tier 3 — Causal Proof**: non-empty `causal_scores` with every
  record's `is_causal == true`. Gated in governance via
  `require_causal_validation`.

All attestations share a single canonical wire format:
`got_core::SCHEMA_VERSION == 1`. There are no version branches.

## Security Hardening Summary

All layers incorporate defence-in-depth measures hardened during the security audit:

| ID | Hardening | Layer |
|----|-----------|-------|
| S-2 | Registry integrity (SHA-256 on load) | 3a |
| S-7 | Timestamp future-guard (≤ 300 s skew) | 2 |
| S-8 | Key rotation (`verify_chain` accepts `&[VerifyingKey]`) | 3a |
| S-9 | Envelope `verified` flag + `from_bytes_verified()` | 3a |
| S-13 | String field bounds (≤ 256 bytes) | 2 |
| S-20 | Array bounds (≤ 1 024 layers, ≤ 65 536 readings) | 2 |
| S-21 | `model_hash` is `Option<[u8; 32]>` (no sentinel) | 0 |
| N-1 | `Frame::encode()` returns `Result` + payload size guard | 3a |
| N-2 | Mutex poison recovery in `CollectingHook` | 1 |
| N-3 | Full `anyhow::Result` error propagation in CLI | 6 |

---

```
+=======================================================================+
|  Layer 6 — Orchestration                                              |
|                                                                       |
|  ┌──────────────────────────┐  ┌──────────────────────────────────┐  |
|  │  CLI Mode (got-cli)      │  │  Agent Runtime Mode              │  |
|  │                          │  │                                  │  |
|  │  keygen   train          │  │  On startup:                     │  |
|  │  attest   verify         │  │    generate / load keypair       │  |
|  │  checkpoint  drift       │  │    compute geometry from model   │  |
|  │                          │  │    train or load probes          │  |
|  │  All commands return     │  │                                  │  |
|  │  anyhow::Result<()>      │  │  On peer request:                │  |
|  │  (N-3 — no panics on    │  │    enclave_pipeline()            │  |
|  │   I/O or parse errors)   │  │    perform_exchange()            │  |
|  │                          │  │    verify_chain()                │  |
|  │                          │  │    decide: cooperate or refuse   │  |
|  │                          │  │                                  │  |
|  │                          │  │  On model update:                │  |
|  │                          │  │    measure drift from reference  │  |
|  │                          │  │    chain next attestation        │  |
|  │                          │  │    notify peers                  │  |
|  └──────────────────────────┘  └──────────────────────────────────┘  |
|       |       |       |           |       |       |       |          |
+=======|=======|=======|===========|=======|=======|=======|==========+
        |       |       |           |       |       |       |
        v       v       v           v       v       v       v
+=======================================================================+
|  Layer 5 — Network Transport                 (got-net)               |
|                                                                       |
|  TcpTransport (got-wire::noise::Transport impl over real sockets)    |
|                                                                       |
|  Server:                             Client:                          |
|    serve(addr, config)                 request(addr, params, reg)     |
|    tokio listener + spawn_blocking     request_blocking (sync)        |
|    per-connection Noise NK accept      Noise NK initiate → exchange   |
|                                                                       |
|  Codec:                              Federation Sync:                 |
|    ExchangeRequest/Response            FederationSyncManager          |
|    32B agent_id + 200B envelope        async polling loop             |
|    + length-prefixed JSON attests      RefreshPolicy + exp. backoff   |
|                                        HttpSyncSource (reqwest)       |
|                                        If-None-Match / 304 support    |
|                                                                       |
|  ModelContext (attestation_cache):                                    |
|    Two-tier cost model for attestation lifecycle:                     |
|    CACHED (expensive, changes on model update only):                 |
|      CausalGeometry Phi, trained probe weights,                      |
|      causal validation results, geometry_hash,                       |
|      parent_attestation_hash, geometry_drift                         |
|    PER-ATTESTATION (depends on input context, NEVER cached):         |
|      forward pass -> activations -> read_probe() -> sign             |
|    Invalidation: startup / model update / distribution shift /       |
|      manual operator trigger                                         |
|    RwLock for thread safety (read-heavy, write-rare)                 |
|                                                                       |
+=======================================================================+
        |                              |
        v                              v
+=======================================================================+
|  Layer 4 — Hardware Enclave                  (got-enclave)            |
|                                                                       |
|  MeasurementEnclave trait:                                            |
|    receive_activations(frame)    — ingest frame, verify integrity     |
|    run_causal_check(model_fn, δ) — causal intervention inside TEE    |
|    attest(model_id, ...)         — sign attestation (key in enclave) |
|    attest_with_causal(...)       — sign with embedded causal scores  |
|    verifying_key()               — public key only                   |
|    frame_count() / reset()       — lifecycle                         |
|                                                                       |
|  MockEnclave                     — PoC implementation                |
|    signing key never leaves boundary                                 |
|    recomputes integrity hash on receive                              |
|    runs read_probe inside enclave                                    |
|    attest_with_causal embeds scores before signing                   |
|                                                                       |
|  ActivationFrame                 — captured from hardware            |
|    compute_hash(layer, pos, vals)— SHA-256(layer ‖ pos ‖ values)    |
|    verify_integrity()            — compare stored vs recomputed      |
|                                                                       |
|  HardwareCapture trait           — GPU DMA / TEE copy-out            |
|    MockDmaTap                    — test double, optional tamper      |
|                                                                       |
|  enclave_pipeline()              — end-to-end:                       |
|    capture → receive → causal_check → attest_with_causal             |
|                                                                       |
+=======================================================================+
        |                              |
        v                              v
+===================================+=======================================+
|  Layer 3a — Wire Protocol         |  Layer 3b — Attestation Store         |
|                   (got-wire)      |                        (got-store)    |
|                                   |                                       |
|  Frame { encode → Result, decode }|  AttestationStore trait:              |
|    magic: 0x474F5431              |    append / get / chain               |
|    N-1: payload ≤ 16 MiB guard   |    query / audit                      |
|  MessageType                      |                                       |
|    ExchangeReq/Rsp (0x01/0x02)   |  MemoryStore (in-memory HashMap)      |
|    VerifyReq/Rsp                  |  FileStore   (on-disk JSON, atomic)   |
|    ChainReq/Rsp                   |    hash-on-load integrity check       |
|    BehavioralExchangeReq/Rsp      |                                       |
|      (0x10/0x11)                  |                                       |
|    Error                          |                                       |
|                                   |  StoreFilter (builder)                |
|  ExchangeEnvelope (200 bytes)     |    model_id / signer / time range     |
|    nonce ‖ peer_agent_id          |    schema_version / causal_flag       |
|    ‖ attestation_hash             |                                       |
|    ‖ chain_root ‖ timestamp       |  AuditReport                          |
|    Ed25519 sig over 136 bytes     |    chain_valid, drift_summary         |
|    S-9: verified flag             |    causal_summary, signers            |
|    from_bytes_verified()          |                                       |
|    is_verified() accessor         +---------------------------------------+
|                                   |
|  Exchange protocol:               |
|    build_request/response         |
|    validate_request/response      |
|    perform_exchange (in-memory)   |
|                                   |
|  Chain verification:              |
|    attestation_hash()             |
|    verify_chain(chain, current,   |
|      signer_pks: &[VerifyingKey], |  ← S-8: key rotation support
|      max_drift)                   |
|    → ChainVerdict                 |
|                                   |
|  TrustRegistry (TOML)            |
|    S-2: SHA-256 integrity on load |
|    AgentEntry { agent_id,         |
|      expected_model_hash,         |
|      max_drift_accepted, roles,   |
|      domain_scope }               |
|    max_attestation_age_secs       |
|                                   |
|  Domain scoping (§4):             |
|    DomainScope { primary,         |
|      permitted, exclusions }      |
|    InteractionMode includes       |
|      Supervised (§5.5)            |
|    check_domain_compatibility()   |
|    → Phase 0, before crypto       |
|                                   |
|  Governance (§7.3 / §8.2):        |
|    GovernanceTable keyed by       |
|      DomainPattern →              |
|      GovernanceThresholds         |
|      { max_drift, min_confidence, |
|        min_causal_score,          |
|        require_chain,             |
|        require_causal_validation }|
|    effective_thresholds() falls   |
|    back to flat max_drift_accepted|
|                                   |
|  Attestation scope binding (§2.1):|
|    check_attestation_scope_       |
|      binding() — rejects when the |
|    peer's embedded declaration    |
|    disagrees with the registry    |
|                                   |
|  Supervised request (§5.5):       |
|    perform_supervised_request()   |
|    → one-directional verdict      |
|                                   |
|  Federation (got-wire::federation)|
|    Multi-hop voucher chains       |
|      verify_vouchers_with_depth() |
|      DEFAULT_MAX_VOUCHER_CHAIN_   |
|        DEPTH = 10                 |
|      fixed-point with snapshot-   |
|        per-iteration              |
|    OperatorKeyRotation            |
|      cross-signed, temporal       |
|      constraint (not_before)      |
|    FederationRevocationList (FRL) |
|      signed fingerprint list      |
|      only in-chain FRLs honoured  |
|    FederationSyncSource trait     |
|      StaticSyncSource             |
|      FileSyncSource               |
+===================================+
        |
        v
+=======================================================================+
|  Layer 2 — Attestation & Signing             (got-attest)             |
|                                                                       |
|  assemble_and_sign(attestation, key) → Result<GeometricAttestation>   |
|    S-7:  reject timestamp > now + 300 s                               |
|    S-13: reject model_id / corpus_version / probe_version > 256 bytes |
|    S-20: reject layer_readings > 1 024 layers or > 65 536 readings    |
|                                                                       |
|  verify(attestation, pubkey) → Result<bool>                           |
|                                                                       |
|  serialise_for_signing()  — deterministic canonical LE bytes          |
|    LINEAR layout (no version branches):                               |
|      header/model/precision/input_hash/timestamp                      |
|      readings / confidence / coverage_flags                           |
|      parent_attestation_hash / geometry_hash / geometry_drift         |
|      causal_scores / intervention_delta / causal_flag                 |
|      sequence_number / directional_drifts / probe_commitment         |
|      density_reading / curvature_reading                              |
|      domain_scope_declaration                                         |
|                                                                       |
|  attestation_hash() — SHA-256 of canonical bytes (chain linkage)      |
|  merkle_root()      — RFC 6962 domain-separated SHA-256 Merkle tree   |
|  is_supported_schema()  v == 1 → true  (single wire format)          |
|                                                                       |
+=======================================================================+
        |
        v
+=======================================================================+
|  Layer 1 — Probe Training, Inference & Causal Intervention            |
|                                                    (got-probe)        |
|                                                                       |
|  — lib.rs (training & inference) —                                    |
|                                                                       |
|  train_probe(data, geometry, ...)                                     |
|    SGD: w ← w − lr·err·Φh  using gram_vec(), inner_product()         |
|    Platt scaling + reliability threshold                              |
|                                                                       |
|  read_probe(probe, h, geometry)                                       |
|    raw = ⟨w,h⟩_c + b    conf = σ(scale·raw + shift)    flag check   |
|                                                                       |
|  read_probe_checked(probe, set, h, current_geo, ref_geo)             |
|    validates geometry_hash match, enforces max_drift bound            |
|    returns ProbeStale / GeometryMismatch on failure                   |
|                                                                       |
|  — intervention.rs (causal checks) —                                  |
|                                                                       |
|  causal_check(probe, h, geometry, δ, model_fn, threshold)            |
|    ŵ_c = Φw/‖Φw‖    h± = h ± δ·ŵ_c    compare model outputs        |
|    → CausalScore { delta_plus, delta_minus, consistency, is_causal }  |
|                                                                       |
|  causal_check_multi_layer(probes_by_layer, h_by_layer, ...)          |
|    → MultiLayerCausalResult { layer_scores, cross_layer_consistent }  |
|                                                                       |
|  CausalScore → CausalScoreRecord (for attestation embedding)         |
|  ProbeLibrary { probes, sample_size }                                 |
|                                                                       |
|  — hooks.rs (runtime measurement) —                                   |
|                                                                       |
|  MeasurementHook trait: on_activation(request_id, layer, h)           |
|  CollectingHook: thread-safe buffer, drain()                          |
|    N-2: mutex poison recovery via unwrap_or_else(|e| e.into_inner())  |
|                                                                       |
|  ActivationStats: Welford online mean/variance                        |
|    update(h) / variance() / current_mean()                            |
|                                                                       |
|  detect_distribution_shift(baseline, current, threshold_sigmas)       |
|    fraction of dimensions shifted beyond z-score threshold            |
|                                                                       |
|  MeasurementSidecar:                                                  |
|    windowed probe sampling (stratified random from ProbeLibrary)      |
|    automatic close_window() → signed attestation                     |
|    optional causal_check per reading                                  |
|    set_parent_hash() for chaining                                     |
|    coverage tracking across windows                                   |
|    can trigger ModelContext::invalidate() on distribution shift       |
|                                                                       |
+=======================================================================+
        |
        v
+=======================================================================+
|  Layer 0 — Core Types & Causal Geometry      (got-core)               |
|                                                                       |
|  — geometry.rs —                                                      |
|                                                                       |
|  CausalGeometry                                                       |
|    from_unembedding(U, ε)   Φ = UᵀU  (+εI if rank-deficient)        |
|    from_raw_gram(data, d)   rebuild from .gotgeo checkpoint           |
|    inner_product(w, h)      wᵀΦh                                     |
|    gram_vec(h)              Φh (gradient multiplier in training)      |
|    transform(U, h)          Uh → ℝ^V (diagnostic only)               |
|    geometry_hash()          SHA-256(Φ) deterministic fingerprint      |
|    drift_from(ref)          ‖Φ_new − Φ_ref‖_F / ‖Φ_ref‖_F          |
|    is_positive_definite()   epsilon()  hidden_dim()  gram()           |
|                                                                       |
|  — lib.rs —                                                           |
|                                                                       |
|  GeometricAttestation (single canonical layout)                       |
|    S-21: model_hash is Option<[u8; 32]> (no sentinel zeros)          |
|    schema_version = SCHEMA_VERSION (always 1)                         |
|    parent_attestation_hash, geometry_hash, geometry_drift             |
|    causal_scores, intervention_delta, causal_flag                     |
|    sequence_number (Phase 13 monotonic counter)                       |
|    directional_drifts (Phase 13 per-probe drift)                     |
|    probe_commitment (Phase 13 pre-computation binding)               |
|    density_reading, curvature_reading (manifold analysis)             |
|    domain_scope_declaration (§2.1 embedded scope)                     |
|    signature: [u8; 64]                                                |
|                                                                       |
|  UnsignedAttestation newtype (prevents accidental unsigned use)       |
|  CausalScoreRecord  UnembeddingMatrix  LayerActivation                |
|  Precision { Fp32, Fp16, Bfloat16, Int8 }                            |
|  InnerProduct { Causal, Euclidean, CausalRegularised{ε} }            |
|  DirectionalDrift                                                     |
|  DomainScopeDeclaration / PermittedDomainDeclaration /                |
|    InteractionModeTag (wire-level mirrors of got-wire::domain)       |
|  sha256()  hex32/hex64/optional_hex32 (serde)                         |
|  SCHEMA_VERSION constant (single value)                               |
|                                                                       |
+=======================================================================+
        ^                ^
        |                |
 .gotact / .gotue / .gotgeo
        |                |
+=======================================================================+
|  Python Scripts (extraction)                                          |
|                                                                       |
|  extract_activations.py    HuggingFace model → .gotact + .gotue      |
|  test_real_models.py       end-to-end test with real models           |
|                                                                       |
|  ~50-line bridge: reads U and h from a transformer, serialises to    |
|  binary (.gotact / .gotue) for Layer 0.  Supports GPT-2, LLaMA,     |
|  Mistral, OPT, GPTNeoX/Pythia via architecture auto-detection.       |
+=======================================================================+
```

---

## Layer Descriptions

### Layer 0 — Core Types & Causal Geometry (`got-core`)

Foundation types with no business logic dependencies:

- **`GeometricAttestation`** — the single canonical attestation layout. All capability fields travel in every attestation; whether they are populated determines the content-based trust tier.
  - Header: `schema_version` (always `SCHEMA_VERSION`), `model_id`, `input_hash`, `model_hash` (`Option<[u8; 32]>` — S-21), `precision`, `inner_product`, `timestamp`, `corpus_version`, `probe_version`
  - Probe readings: `layer_readings`, `confidence`, `coverage_flags`, `divergence_flag`
  - Chaining: `parent_attestation_hash`, `geometry_hash`, `geometry_drift`
  - Causal (Tier 3): `causal_scores`, `intervention_delta`, `causal_flag`
  - Phase 13 hardening: `sequence_number`, `directional_drifts`, `probe_commitment`
  - Manifold analysis: `density_reading`, `curvature_reading`
  - §2.1 binding: `domain_scope_declaration`
  - `signature: [u8; 64]` (Ed25519 over all preceding fields)
- **`UnsignedAttestation`** — newtype wrapper preventing accidental unsigned use
- **`CausalScoreRecord`** — serialisable causal score (delta_plus, delta_minus, consistency, is_causal)
- **`DirectionalDrift`** — per-probe directional drift record (Phase 13)
- **`UnembeddingMatrix`** — V × d row-major matrix from model weights
- **`LayerActivation`** — residual stream values at one layer/position
- **`Precision`** — enum: Fp32, Fp16, Bfloat16, Int8
- **`InnerProduct`** — enum: Causal, Euclidean, CausalRegularised{ε}
- **`CausalGeometry`** — the mathematical core:
  - `from_unembedding(U, ε)` — computes Φ = UᵀU, regularises if rank-deficient
  - `from_raw_gram(data, d)` — reconstructs from a .gotgeo checkpoint
  - `inner_product(w, h)` — wᵀΦh
  - `gram_vec(h)` — Φh (gradient multiplier in probe training)
  - `transform(U, h)` — Uh ∈ ℝ^V (diagnostic only)
  - `geometry_hash()` — SHA-256 of Gram matrix (deterministic fingerprint)
  - `drift_from(ref)` — ‖Φ−Φ_ref‖_F / ‖Φ_ref‖_F
- **`ValueConstraint`/`ValueOrdering`** — value-ordering constraints for coherence scoring
  - `coherence_score(h, ordering, geometry, α)` — C(h) ∈ [0,1] sigmoid constraint satisfaction
  - `conversational_coherence(states, ...)` — per-position scoring with violation tracking
- **`ValueProjection`** — eigendecomposition of G_W = WᵀΦW
  - `value_projected_gram(probe_weights)` — projected Gram matrix + eigenvalues + dim_eff
  - `effective_value_dimensionality(probe_weights)` — participation ratio for manifold collapse detection
- **`AlignmentDistance`** — value geometry comparison between two models
  - `value_alignment_distance(geo_a, geo_b, probes)` — global Frobenius + probe-projected distance
- **`CausalGeometry::identity(d)`** — zero-allocation fast constructor for Φ = I
- **`sha256(data)`** — canonical SHA-256 utility (used by all crates)
- **`hex32`/`hex64`/`optional_hex32`** — serde helpers for fixed-size byte arrays as hex strings
  - Validates all bytes are ASCII hex before indexing (prevents panic on multi-byte UTF-8)
- **`SCHEMA_VERSION`** — single wire-format version. All attestation capabilities (chaining, causal scores, manifold readings, embedded domain scope) are expressed through Option fields inside one canonical layout. Trust tiers are *content*-based, not version-gated — Tier 2 = `parent_attestation_hash.is_some()`, Tier 3 = non-empty `causal_scores` with every record causal.
- **`DomainScopeDeclaration` / `PermittedDomainDeclaration` / `InteractionModeTag`** — wire-level mirror of `got-wire::domain` types, carried inside the signed payload (§2.1). String-based for stability.

### Layer 1 — Probe Training, Inference & Causal Intervention (`got-probe`)

Machine learning and causal analysis layer. Depends on Layer 0 geometry:

- **`train_probe(data, geometry, ...)`** — SGD on logistic loss under causal inner product
- **`read_probe(probe, h, geometry)`** — inference: raw reading + Platt-scaled confidence + coverage flag
- **`read_probe_checked(probe, set, h, current_geo, ref_geo)`** — drift-aware inference: validates
  geometry hash, checks drift bound, returns `ProbeStale` or `GeometryMismatch` on failure
- **`ProbeSet`** — extended with `geometry_hash` (binds probes to a geometry) and `max_drift` (staleness threshold)
- **`causal_check(probe, h, geometry, δ, model_fn, threshold)`** — single-probe causal intervention:
  perturbs h ± δ·ŵ_c, runs model, compares outputs, returns `CausalScore`
- **`causal_check_multi_layer(...)`** — multi-layer causal check with cross-layer consistency
- **`CausalScore`** — 5-field result (delta_plus, delta_minus, consistency, is_causal, perturbation_delta)
- **`MeasurementSidecar`** — windowed runtime measurement: samples probes, runs optional causal checks,
  produces signed attestations at window boundaries. Can trigger `ModelContext::invalidate()` when `detect_distribution_shift()` fires, forcing recomputation of cached invariants (geometry, probes, causal scores).
- **`CollectingHook`** — thread-safe activation buffer (N-2: recovers from mutex poisoning)
- **`ActivationStats`** — Welford online mean/variance for activation monitoring
- **`detect_distribution_shift(...)`** — z-score-based fraction of shifted dimensions

### Layer 2 — Attestation & Signing (`got-attest`)

Cryptographic layer. Depends on Layer 0 types only:

- **`assemble_and_sign(attestation, key)`** — canonical serialise → Ed25519 sign
  - S-7: rejects timestamps > now + 300 s
  - S-13: rejects string fields > 256 bytes
  - S-20: rejects > 1 024 layers or > 65 536 readings
- **`verify(attestation, pubkey)`** — canonical serialise → Ed25519 verify
- **`serialise_for_signing()`** — deterministic canonical byte serialisation
- **`attestation_hash()`** — SHA-256 of canonical bytes (used as parent hash in chains)
- **`merkle_root(shards)`** — SHA-256 Merkle tree with RFC 6962 domain separation
- **`is_supported_schema(v)`** — {1, 2, 3} → true

### Layer 3a — Wire Protocol (`got-wire`)

Network exchange layer. Depends on got-core and got-attest:

- **`Frame`** — length-prefixed binary framing with magic number and message type
  - N-1: `encode()` returns `Result<Vec<u8>, WireError>` with 16 MiB guard
- **`MessageType`** — ExchangeReq, ExchangeRsp, VerifyReq, VerifyRsp, ChainReq, ChainRsp, Error
- **`ExchangeEnvelope`** — signed 200-byte envelope (S-9: `verified` flag, `from_bytes_verified()`, `is_verified()`)
- **`verify_chain`** — walk chain, check parent linkage, enforce drift (S-8: `&[VerifyingKey]` for key rotation)
- **`TrustRegistry`** — TOML config (S-2: SHA-256 integrity on load; `max_attestation_age_secs`)
- **`AgentEntry`** — name, public_key, agent_id, max_drift_accepted, roles, expected_model_hash, certificate, domain_scope, governance_table
- **`DomainScope`** (`got-wire::domain`) — primary `Domain`, `Vec<PermittedDomain>` (with `InteractionMode` including `Supervised`), exclusion patterns. `check_domain_compatibility()` runs at Phase 0 in `validate_request` / `validate_response`, before any cryptographic or geometric verification (Protocol §4 / Appendix B). `to_declaration()` / `from_declaration()` mirror the rich type into the wire-level struct that rides inside the signed attestation payload.
- **`GovernanceThresholds`** (`got-wire::governance`) — per-domain `max_drift`, `min_confidence`, `min_causal_score`, `require_chain`, `require_causal_validation`. `GovernanceTable` keyed by `DomainPattern` with most-specific lookup. `effective_thresholds()` falls back to `permissive(entry.max_drift_accepted)` when no domain-specific policy matches (§7.3 / §8.2).
- **`perform_supervised_request()`** — one-directional exchange helper (§5.5): a regulator demands an attestation from a supervised agent without producing one of its own. Requires `InteractionMode::Supervised` on both sides of the domain scope.
- **`check_attestation_scope_binding()`** — §2.1 cross-check: if the incoming attestation carries a `DomainScopeDeclaration`, it must match the registry's entry for the same agent. Catches relay attacks and misconfigured agents.
- **Federation** (`got-wire::federation`):
  - Multi-hop voucher chains — `verify_vouchers_with_depth()` walks transitive A→B→C chains with a fixed-point algorithm (snapshot-per-iteration) up to `DEFAULT_MAX_VOUCHER_CHAIN_DEPTH` (10).
  - `OperatorKeyRotation` — cross-signed rotation record binding old key to new key with a temporal constraint (`not_before`).
  - `FederationRevocationList` (FRL) — signed list of revoked voucher fingerprints; only FRLs from in-chain operators are honoured.
  - `FederationSyncSource` trait + `StaticSyncSource` (in-memory) + `FileSyncSource` (disk-based) for providing registry snapshots to the sync layer.

### Layer 3b — Attestation Store (`got-store`)

Persistence and audit layer. Depends on got-core and got-attest:

- **`AttestationStore`** trait — content-addressed storage
- **`MemoryStore`** / **`FileStore`** — in-memory or on-disk JSON (atomic writes, hash-on-load)
- **`StoreFilter`** — builder for querying by model_id, signer, time range, schema, causal flag
- **`AuditReport`** — chain length/validity, drift summary, causal summary, signers

### Layer 4 — Hardware Enclave (`got-enclave`)

Hardware isolation layer. Depends on got-core, got-probe, got-attest, got-wire:

- **`MeasurementEnclave`** trait — trusted boundary for activation measurement
- **`MockEnclave`** — PoC with real crypto, mock hardware
- **`ActivationFrame`** — capture buffer with SHA-256 integrity hash
- **`HardwareCapture`** trait / **`MockDmaTap`** — GPU DMA / TEE copy-out
- **`enclave_pipeline()`** — capture → receive → causal_check → attest_with_causal

### Layer 5 — Network Transport (`got-net`)

Concrete TCP transport with Noise NK encryption over real sockets:

- **`TcpTransport`** — implements `got-wire::noise::Transport` over `TcpStream`; 16 MiB receive guard.
- **Server** — `serve(addr, config)` runs an async tokio listener; each accepted connection is dispatched to a sync handler via `spawn_blocking` that performs Noise NK handshake, receives `ExchangeRequest`, runs `validate_request`, and sends a signed `ExchangeResponse`.
- **Client** — `request_blocking(addr, params, registry)` connects, performs Noise NK initiation, sends request, receives response (sync). `request(addr, params, registry)` wraps this in an async `spawn_blocking` call.
- **Codec** — `encode_exchange_request` / `decode_exchange_request` (and Response variants): 32-byte `agent_id` + 200-byte `ExchangeEnvelope` + length-prefixed JSON for attestation chains.
- **`FederationSyncManager`** — async polling loop that periodically fetches remote federation registry snapshots. Configurable `RefreshPolicy` with exponential backoff on failure and staleness detection.
- **`HttpSyncSource`** — implements `FederationSyncSource` using `reqwest::blocking` with `If-None-Match` / HTTP 304 support for bandwidth-efficient polling.
- **`ModelContext`** (`attestation_cache`) — separates the attestation lifecycle into two cost tiers. Probe readings depend on input activations, so signed attestations cannot be cached. What CAN be cached are the expensive invariants that only change on model update:
  - **Cached in `CachedInvariants`** (expensive, recomputed on model update): `CausalGeometry` Phi = UᵀU (O(Vd²)), trained probe weights (SGD under causal IP, bound to `geometry_hash`), causal validation results (model forward passes for `causal_check`), `geometry_hash`, `parent_attestation_hash`, `geometry_drift`, `model_id`, `model_hash`, `computed_at` timestamp.
  - **Computed fresh per attestation** (depends on input context, NEVER cached): forward pass to get activations, `read_probe()` per probe x layer producing `layer_readings` / `confidence` / `coverage_flags`, then `assemble_and_sign()` to produce the signed `GeometricAttestation`.
  - **API**: `new()`, `with_invariants()`, `get()` -> `Option<CachedInvariants>`, `update()`, `invalidate()`, `is_ready()`, `computed_at()`.
  - **Invalidation triggers**: (1) agent startup, (2) model update (new U -> recompute Phi, retrain probes, re-run causal checks), (3) `detect_distribution_shift()` fires (probe staleness) -- MeasurementSidecar can trigger `ModelContext::invalidate()`, (4) manual operator trigger.
  - **Thread safety**: uses `RwLock` (read-heavy, write-rare pattern). Does NOT implement `AttestationProvider` -- the old `AttestationCache` which cached signed attestations was architecturally wrong because readings depend on input context.

### Layer 6a — Proxy Architecture (`got-proxy`)

Behavioral value monitoring for closed-source models (Tier 0 trust):

- **`BehavioralValueSpace`** — per-term Welford online mean/variance + EWMA for recency weighting; pairwise baselines
- **`ProxySession`** — lifecycle: `new()` → `observe()` → `snapshot_and_attest()`
  - Accepts text or pre-computed embeddings; embeds internally via configurable embedding endpoint
  - Value detection uses absolute cosine similarity (not z-scoring)
- **`detect_deviation()`** — 4-signal algorithm:
  - Signal 1: term-level shift (fraction of terms exceeding baseline σ)
  - Signal 2: profile cosine drift (1 − cosine between current and baseline EWMA vectors)
  - Signal 3: pairwise relationship disruption (fraction of pairs shifting beyond baseline σ)
  - Signal 4: manifold density (off-manifold detection via k-NN log-density)
  - Combined: weighted sum → WithinBaseline / Drifting / Deviated
- **`BehavioralAttestation`** — schema "B1", Ed25519 signed, chained via parent_hash
- **`ValueSpaceStore`** trait — `MemoryValueSpaceStore` + `FileValueSpaceStore`
- **`ProxyConfig`** — all thresholds, weights, EWMA alpha, minimum observations

### Layer 6b — Orchestration

**CLI Mode (`got-cli`)**: keygen, train, attest, verify, checkpoint, drift,
coherence, collapse-report, compare — all return `anyhow::Result<()>` (N-3).

**Web Mode (`got-web`)**: Axum server with unified single-page D3.js frontend:
- LLM chat — activation server (real hidden states), Ollama, OpenAI, Anthropic
- Text embedding (`/api/embed`) — routes through activation server sidecar or falls back to bag-of-words
- Metrics (`/api/coherence`, `/api/collapse`, `/api/compare`) — coherence scoring, manifold collapse, model comparison
- Proxy endpoints (`/api/proxy/session/*`) — session lifecycle, observation, deviation, attestation
- Coherence analysis (`/api/conversation/analyse`) — per-turn value detection, contradictions, trust
- Real Φ = UᵀU geometry from model's unembedding matrix (248K × 4096 for Qwen3.5)
- Configurable value taxonomy (`--values values.toml`) — descriptions embedded through reference model
- 11 visualization tabs in 3 groups (Live / Pairwise / Geometry)
- Static file serving via `tower_http::ServeDir` — modular ES modules + CSS

**Activation Server** (`scripts/activation_server.py`): Python FastAPI sidecar
that loads a model via HuggingFace (4-bit quantized), serves intermediate-layer
residual stream activations via `/hidden_states` and OpenAI-compatible chat via
`/v1/chat/completions`. Enables measuring under the causal inner product at
layers where value concepts actually separate (middle layers, not output layer).

**Agent Runtime Mode**: calls Layer 0–4 directly, manages keypairs, exchanges
attestations, walks chains, stores results, makes cooperate/refuse decisions.

### External — Python Extraction Scripts

~50-line Python bridge reads U and h from a HuggingFace model, serialises
to `.gotact` / `.gotue`. Supports GPT-2, LLaMA/Mistral, OPT, GPTNeoX/Pythia.
