# Layer Architecture

Dependency stack from core types up to agent orchestration.
The top layer can be either the CLI (for human-operated setup) or
an agent runtime (for autonomous agent-to-agent operation).
Layers 0–4 are identical in both modes.

The pipeline implements three trust tiers (from the plan):
- **Tier 1 — Signature**: Ed25519 over deterministic canonical bytes
- **Tier 2 — Consistency**: signature + coverage flags + confidence bounds + chain verification
- **Tier 3 — Reproduction**: full re-extraction + re-probing + bitwise match

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
| N-3 | Full `anyhow::Result` error propagation in CLI | 5 |

---

```
+=======================================================================+
|  Layer 5 — Orchestration                                              |
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
|  │                          │  │    chain attestation (v2/v3)     │  |
|  │                          │  │    notify peers                  │  |
|  └──────────────────────────┘  └──────────────────────────────────┘  |
|       |       |       |           |       |       |       |          |
+=======|=======|=======|===========|=======|=======|=======|==========+
        |       |       |           |       |       |       |
        v       v       v           v       v       v       v
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
|    ExchangeReq/Rsp                |  MemoryStore (in-memory HashMap)      |
|    VerifyReq/Rsp                  |  FileStore   (on-disk JSON, atomic)   |
|    ChainReq/Rsp                   |    hash-on-load integrity check       |
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
|      max_drift_accepted, roles }  |
|    max_attestation_age_secs       |
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
|    v1 branch: core fields (readings, model_hash, ...)                 |
|    v2 branch: + parent_attestation_hash, geometry_hash, drift         |
|    v3 branch: + causal_scores, intervention_delta, causal_flag        |
|                                                                       |
|  attestation_hash() — SHA-256 of canonical bytes (chain linkage)      |
|  merkle_root()      — RFC 6962 domain-separated SHA-256 Merkle tree   |
|  is_supported_schema()  {1, 2, 3} → true                             |
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
|    automatic close_window() → signed attestation (v1/v2/v3)          |
|    optional causal_check per reading                                  |
|    set_parent_hash() for chaining                                     |
|    coverage tracking across windows                                   |
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
|  GeometricAttestation (v1 + v2 + v3 schema)                          |
|    S-21: model_hash is Option<[u8; 32]> (no sentinel zeros)          |
|    schema_version: 1 | 2 | 3                                         |
|    parent_attestation_hash, geometry_hash, geometry_drift             |
|    causal_scores, intervention_delta, causal_flag                     |
|    sequence_number (Phase 13 monotonic counter)                       |
|    directional_drifts (Phase 13 per-probe drift)                     |
|    probe_commitment (Phase 13 pre-computation binding)               |
|    signature: [u8; 64]                                                |
|                                                                       |
|  UnsignedAttestation newtype (prevents accidental unsigned use)       |
|  CausalScoreRecord  UnembeddingMatrix  LayerActivation                |
|  Precision { Fp32, Fp16, Bfloat16, Int8 }                            |
|  InnerProduct { Causal, Euclidean, CausalRegularised{ε} }            |
|  DirectionalDrift  sha256()  hex32/hex64/optional_hex32 (serde)       |
|  SCHEMA_VERSION / _2 / _3 constants                                   |
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

- **`GeometricAttestation`** — the attestation schema (v1 + v2 + v3 fields + Ed25519 signature)
  - v1 fields: `schema_version`, `model_id`, `input_hash`, `model_hash` (`Option<[u8; 32]>` — S-21), `readings`,
    `confidences`, `coverage_flags`, `inner_product`, `precision`,
    `divergence_flag`, `timestamp`, `corpus_version`, `probe_version`, `signature`
  - v2 extensions: `parent_attestation_hash`, `geometry_hash`, `geometry_drift`
  - v3 extensions: `causal_scores`, `intervention_delta`, `causal_flag`
  - Phase 13 extensions: `sequence_number`, `directional_drifts`, `probe_commitment`
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
- **`sha256(data)`** — canonical SHA-256 utility (used by all crates)
- **`hex32`/`hex64`/`optional_hex32`** — serde helpers for fixed-size byte arrays as hex strings
  - Validates all bytes are ASCII hex before indexing (prevents panic on multi-byte UTF-8)
- **`SCHEMA_VERSION`/`SCHEMA_VERSION_2`/`SCHEMA_VERSION_3`** — supported schema version constants

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
  produces signed attestations at window boundaries
- **`CollectingHook`** — thread-safe activation buffer (N-2: recovers from mutex poisoning)
- **`ActivationStats`** — Welford online mean/variance for activation monitoring
- **`detect_distribution_shift(...)`** — z-score-based fraction of shifted dimensions

### Layer 2 — Attestation & Signing (`got-attest`)

Cryptographic layer. Depends on Layer 0 types only:

- **`assemble_and_sign(attestation, key)`** — canonical serialise → Ed25519 sign (v1, v2, v3)
  - S-7: rejects timestamps > now + 300 s
  - S-13: rejects string fields > 256 bytes
  - S-20: rejects > 1 024 layers or > 65 536 readings
- **`verify(attestation, pubkey)`** — canonical serialise → Ed25519 verify (v1, v2, v3)
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
- **`AgentEntry`** — name, public_key, agent_id, max_drift_accepted, roles, expected_model_hash

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

### Layer 5 — Orchestration

**CLI Mode (`got-cli`)**: keygen, train, attest, verify, checkpoint, drift —
all return `anyhow::Result<()>` (N-3).

**Agent Runtime Mode**: calls Layer 0–4 directly, manages keypairs, exchanges
attestations, walks chains, stores results, makes cooperate/refuse decisions.

### External — Python Extraction Scripts

~50-line Python bridge reads U and h from a HuggingFace model, serialises
to `.gotact` / `.gotue`. Supports GPT-2, LLaMA/Mistral, OPT, GPTNeoX/Pythia.
