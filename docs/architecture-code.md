# Code Architecture

Crate dependency graph and internal structure of the Geometry of Trust
system. Arrows show the direction of dependency (caller → callee).
Layer 5 can be either the CLI binary or an agent runtime that calls the
same libraries.

The pipeline runs: **deterministic geometry → signed attestation →
independent reproducibility → causal proof → agent exchange**. Each
layer adds one guarantee on top of the layers below it.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Layer 5 — Orchestration                                │
│                                                                        │
│  ┌──────────────────────────────┐  ┌─────────────────────────────────┐ │
│  │  got-cli  (binary)           │  │  Agent Runtime (library calls)  │ │
│  │                              │  │                                 │ │
│  │  keygen   train   attest     │  │  startup:                       │ │
│  │  verify   checkpoint  drift  │  │    load/generate keypair        │ │
│  │                              │  │    compute geometry             │ │
│  │  .gotact / .gotue parsers    │  │    train/load probes            │ │
│  │  .gotgeo save / load         │  │                                 │ │
│  │                              │  │  peer exchange:                 │ │
│  │  All commands return         │  │    enclave_pipeline()           │ │
│  │  anyhow::Result<()> (N-3)   │  │    perform_exchange()           │ │
│  │                              │  │    verify_chain()               │ │
│  │                              │  │    decide: cooperate/refuse     │ │
│  └──────────────────────────────┘  └─────────────────────────────────┘ │
│       │       │       │                │       │       │               │
└───────┼───────┼───────┼────────────────┼───────┼───────┼───────────────┘
        │       │       │                │       │       │
        v       v       v                v       v       v
┌─────────────────────────────────────────────────────────────────────────┐
│               Layer 4 — Hardware Enclave  (got-enclave)                 │
│                                                                        │
│  MeasurementEnclave trait        MockEnclave                           │
│    receive_activations()           hardware capture + integrity check  │
│    run_causal_check()              probe reading inside enclave        │
│    attest()                        signing key never leaves boundary   │
│    attest_with_causal()                                                │
│    verifying_key()               enclave_pipeline()                    │
│    frame_count() / reset()         capture → ingest → causal → attest  │
│                                                                        │
│  ActivationFrame                 HardwareCapture trait                  │
│    compute_hash(layer, pos, val)   MockDmaTap (test double)            │
│    verify_integrity()              optional tamper injection            │
│                                                                        │
└────────┬──────────────────────────────┬────────────────────────────────┘
         │                              │
         v                              v
┌────────────────────────────────┐ ┌──────────────────────────────────────┐
│  Layer 3b — Store (got-store)  │ │  Layer 3a — Wire Protocol (got-wire) │
│                                │ │                                      │
│  AttestationStore trait        │ │  Frame { encode→Result, decode }     │
│    append / get / chain        │ │  N-1: payload ≤ 16 MiB guard         │
│    query / audit               │ │  MessageType (Req/Rsp/Chain/Error)   │
│                                │ │                                      │
│  MemoryStore (in-memory)       │ │  ExchangeEnvelope (200 bytes)        │
│  FileStore (on-disk JSON)      │ │    S-9: verified flag                │
│    atomic writes               │ │    from_bytes_verified()             │
│    hash-on-load integrity      │ │    is_verified() accessor            │
│                                │ │    create / verify / to_bytes        │
│  StoreFilter (builder)         │ │                                      │
│  StoreId = [u8; 32]           │ │  build_request / build_response      │
│                                │ │  validate_request / validate_response│
│  AuditReport                   │ │  perform_exchange (in-memory)        │
│    drift_summary               │ │                                      │
│    causal_summary              │ │  verify_chain(signer_pks:            │
│    chain_valid, signers        │ │    &[VerifyingKey])  S-8: rotation   │
│                                │ │  attestation_hash / ChainVerdict     │
│                                │ │                                      │
│                                │ │  TrustRegistry (TOML)                │
│                                │ │    S-2: SHA-256 integrity on load    │
│                                │ │    AgentEntry + expected_model_hash  │
│                                │ │    + domain_scope (Option)           │
│                                │ │    max_attestation_age_secs          │
│                                │ │  agent_id = SHA-256(public_key)      │
│                                │ │                                      │
│                                │ │  Domain scoping (§4 / Appendix B):   │
│                                │ │    Domain / DomainPattern            │
│                                │ │    InteractionMode                   │
│                                │ │    DomainScope { primary,            │
│                                │ │      permitted, exclusions }         │
│                                │ │    check_domain_compatibility()      │
│                                │ │    → Phase 0 in validate_request /   │
│                                │ │      validate_response (before crypto)│
└────────┬───────────────────────┘ └──────┬───────────────────────────────┘
         │                                │
         v                                v
┌─────────────────────────────────────────────────────────────────────────┐
│               Layer 2 — Attestation & Signing  (got-attest)             │
│                                                                        │
│  assemble_and_sign() → Result     verify() → Result<bool>              │
│    S-7:  timestamp ≤ now+300s       |                                  │
│    S-13: string fields ≤ 256 B      v                                  │
│    S-20: ≤ 1024 layers,       serialise_for_signing()                  │
│          ≤ 65536 readings     (same function, canonical bytes)         │
│         │                     (v1: original fields)                    │
│         v                     (v2: + parent hash,                      │
│  serialise_for_signing()       geo hash, drift)                        │
│  (v1–v4 branches)            (v3: + causal scores,                    │
│                                intervention_delta,                     │
│  attestation_hash()            causal_flag)                            │
│  sha256(canonical bytes)     (v4: + density_reading,                   │
│                                curvature_reading)                      │
│  merkle_root()               is_supported_schema()                     │
│  (RFC 6962 domain sep)       {1, 2, 3, 4} → true                     │
│                                                                        │
└───────────────┬────────────────────────────────────────────────────────┘
                │
                v
┌─────────────────────────────────────────────────────────────────────────┐
│               Layer 1 — Probe & Intervention  (got-probe)               │
│                                                                        │
│  ─── lib.rs ───────────────────────────────────────────────            │
│  train_probe()                    read_probe()                         │
│    SGD under causal IP              raw = <w,h>_c + b                  │
│                                     conf = sigma(scale*raw+shift)      │
│  ProbeVector { w, b, platt,        flag = conf < threshold             │
│    platt_shift, threshold }                                            │
│                                   read_probe_checked()                 │
│  ProbeSet { probes, layer,          validates geometry_hash            │
│    geometry_hash,                   checks drift bound                 │
│    max_drift }                                                         │
│                                                                        │
│  ─── intervention.rs ──────────────────────────────────────            │
│  causal_check()                 CausalScore (5 fields)                 │
│    perturb h ± δ·ŵ               → CausalScoreRecord (serialisable)   │
│    compare model output        causal_check_multi_layer()              │
│    compute_consistency()       MultiLayerCausalResult                  │
│    is_causal flag              ProbeLibrary { probes, sample_size }    │
│                                                                        │
│  ─── experiment.rs ───────────────────────────────────────            │
│  InterventionExperiment          ExperimentReport (attestable)          │
│    lerp between activation       InterpolationStep {                    │
│    vectors, forward each           causal_distance, log_density,       │
│    through ModelHandle             output_entropy, incoherence_score,   │
│  ExperimentConfig                  model_confidence, on_manifold }      │
│    steps, density_threshold                                             │
│                                                                        │
│  ─── hooks.rs ─────────────────────────────────────────────            │
│  MeasurementHook trait         MeasurementSidecar                      │
│    on_activation()               windowed probe sampling               │
│  CollectingHook                  automatic window → attestation        │
│    N-2: poison recovery          causal checks (optional)              │
│  ActivationStats                 set_parent_hash() for chaining        │
│    Welford online mean/var     detect_distribution_shift()             │
│                                                                        │
└───────────────┬────────────────────────────────────────────────────────┘
                │
                v
┌─────────────────────────────────────────────────────────────────────────┐
│               Layer 0 — Core Types & Geometry  (got-core)               │
│                                                                        │
│  ┌─ geometry.rs ─────────────────────────────┐                         │
│  │                                           │  GeometricAttestation   │
│  │  CausalGeometry                           │  schema_version 1|2|3|4 │
│  │    ├── from_unembedding(U, eps)           │    S-21: model_hash     │
│  │    │     Phi = U^T U  (+eps*I)            │      Option<[u8; 32]>   │
│  │    ├── from_raw_gram(data, d)             │    parent_attest_hash   │
│  │    │     (rebuild from .gotgeo)           │    geometry_hash        │
│  │    ├── inner_product(w, h)  w^T Phi h     │    geometry_drift       │
│  │    ├── gram_vec(h)          Phi h         │    causal_scores: []    │
│  │    ├── transform(U, h)      Uh            │    intervention_delta   │
│  │    ├── geometry_hash()      SHA-256(Phi)  │    causal_flag          │
│  │    └── drift_from(ref)      Frobenius     │    sequence_number      │
│  │                                           │    directional_drifts   │
│  └───────────────────────────────────────────┘    probe_commitment     │
│                                                    signature [u8;64]   │
│  ┌─ manifold.rs ────────────────────────────┐                         │
│  │  ValueManifold                          │  density_reading         │
│  │    ├── new(points, geometry, config)     │  curvature_reading       │
│  │    │     precompute pairwise d_Phi      │                          │
│  │    ├── density_map()  → DensityReading  │                          │
│  │    ├── curvature_map() → CurvatureRead  │                          │
│  │    └── query_log_density(point, geom)   │                          │
│  │  ManifoldConfig { k }                   │                          │
│  │  PointDensity { log_density, dim }      │                          │
│  │  PointCurvature { curvature, count }    │                          │
│  └─────────────────────────────────────────┘                          │
│                                                                        │
│  UnsignedAttestation (newtype wrapper)                                 │
│  CausalScoreRecord  DirectionalDrift                                   │
│  UnembeddingMatrix  LayerActivation                                    │
│  Precision          InnerProduct                                       │
│  euclidean_cosine() (shared utility in geometry.rs)                    │
│  sha256()  (canonical hash utility)                                    │
│  hex32/hex64/optional_hex32 serde (ASCII-hex validated)                │
│  SCHEMA_VERSION / _2 / _3 / _4 constants                              │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                      ^                ^
                      │                │
              .gotact │        .gotue  │        .gotgeo
                      │                │            │
┌─────────────────────────────────────────────────────────────────────────┐
│                     Python Scripts (extraction)                          │
│                                                                        │
│  extract_activations.py     Model → .gotact / .gotue binary files      │
│  test_real_models.py        End-to-end test with real models           │
│                                                                        │
│  ~50-line bridge: reads unembedding matrix U and residual-stream       │
│  activations h out of a HuggingFace model, serialises them into the    │
│  binary formats that Layer 0 consumes. Step 7 of the 12-step build.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Crates

| Crate | Type | Purpose |
|-------|------|---------|
| `got-core` | lib | Core types (`GeometricAttestation` v1+v2+v3, `UnsignedAttestation`, `CausalScoreRecord`, `DirectionalDrift`, `UnembeddingMatrix`, `Precision`, `InnerProduct`), causal geometry (`CausalGeometry`, Gram matrix, inner product, geometry hash, drift), `sha256()`, hex serde helpers |
| `got-probe` | lib | Probe training (SGD under causal IP), inference (`read_probe`), drift-aware inference (`read_probe_checked`), `ProbeSet` with geometry binding; causal intervention (`causal_check`, `CausalScore`, multi-layer); measurement hooks (`MeasurementSidecar`, `CollectingHook` with mutex poison recovery, `ActivationStats`, `detect_distribution_shift`) |
| `got-attest` | lib | Attestation signing/verification (Ed25519, v1+v2+v3) with bounds checking (S-7 timestamp, S-13 strings, S-20 arrays), canonical serialisation, attestation hashing for chain linkage, Merkle tree (SHA-256 + RFC 6962) |
| `got-wire` | lib | Wire protocol framing (`Frame` with Result-returning encode — N-1, `MessageType`), signed exchange envelopes (`ExchangeEnvelope` with verified flag — S-9, `from_bytes_verified()`), request/response exchange (`ExchangeRequest`, `ExchangeResponse`, `perform_exchange`), chain verification (`verify_chain` with `&[VerifyingKey]` — S-8), trust registry (`TrustRegistry` with SHA-256 integrity — S-2, `expected_model_hash`, `max_attestation_age_secs`) |
| `got-enclave` | lib | Hardware isolation abstraction (`HardwareCapture`, `MockDmaTap`), measurement enclave (`MeasurementEnclave` trait, `MockEnclave`), `ActivationFrame` with integrity hashing, `enclave_pipeline()` end-to-end |
| `got-store` | lib | Attestation persistence (`AttestationStore` trait), `MemoryStore` (in-memory), `FileStore` (on-disk JSON with atomic writes + hash-on-load), content-addressed storage (`StoreId`), filtering (`StoreFilter`), audit reporting (`AuditReport`, `DriftSummary`, `CausalSummary`) |
| `got-incoherence` | lib | Zero-training coherence analysis: `causal_cosine()`, `analyse()`, `EmbeddingSource` trait, `PrecomputedEmbeddings`, `UnembeddingLookup`, contradiction/redundancy detection |
| `got-proxy` | lib | Proxy architecture for closed-source models: `BehavioralValueSpace` (Welford + EWMA), `ProxySession`, 3-signal `detect_deviation()`, `BehavioralAttestation` (schema "B1", Ed25519), `ValueSpaceStore` trait (memory + file) |
| `got-cli` | bin | CLI with `keygen`, `train`, `attest`, `verify`, `checkpoint`, `drift` subcommands — all return `anyhow::Result<()>` (N-3); binary `.gotact`/`.gotue`/`.gotgeo` parsers |
| `got-web` | bin | Axum web server: unified D3.js frontend, LLM chat relay (Ollama/OpenAI/Anthropic via `reqwest`), text embedding, proxy session management, coherence analysis; static files via `ServeDir` |

## Cross-Crate Dependency Graph

```
got-core ─────────────────────────────────────────────────────────────
  ^   ^    ^        ^          ^           ^          ^              │
  │   │    │        │          │           │          │              │
  │   │    │        │          │        got-cli    got-incoherence   │
  │   │    │        │          │           │          ^              │
  │   │    │        │          │           │          │              │
  │   │  got-probe  │       got-attest     │       got-proxy        │
  │   │    ^        │          ^           │          ^              │
  │   │    │        │          │           │          │              │
  │   │    │     got-wire ─────┼───────────┼── got-proxy            │
  │   │    │        ^          │           │                        │
  │   │    │        │          │           │                        │
  │   │  got-enclave ── got-wire           │                        │
  │   │    │            got-probe          │                        │
  │   │    │            got-attest         │                        │
  │   │    │                               │                        │
  │   └─ got-store ── got-attest           │                        │
  │                                        │                        │
  │   got-web ── got-core, got-incoherence, got-proxy, reqwest      │
  │                                        │                        │
  └──── workspace root (integration tests) ┘                        │
```

## External Dependencies

| Dependency | Used by | Purpose |
|-----------|---------|---------|
| `faer 0.19` | got-core | Matrix multiplication for Φ = UᵀU |
| `ed25519-dalek 2` | got-attest, got-wire, got-enclave, got-cli | Ed25519 signing and verification |
| `sha2 0.10` | got-core, got-attest, got-enclave | SHA-256 for hashing (geometry, Merkle, frames) |
| `serde 1` | all crates | Serialisation/deserialisation |
| `serde_json 1` | got-wire, got-store | JSON encoding for wire payloads and file store |
| `toml` | got-wire | Trust registry parsing |
| `clap 4` | got-cli | Command-line argument parsing |
| `anyhow 1` | got-cli | Error context propagation (N-3) |
| `zeroize 1` | got-cli | Secure key material cleanup |
| `rand` | got-probe, got-wire | Random sampling, nonce generation |
| `thiserror 1` | got-core, got-probe, got-attest, got-wire, got-enclave, got-store, got-proxy | Error type derivation |
| `reqwest 0.12` | got-web | HTTP client for LLM API relay (Ollama/OpenAI/Anthropic) |
| `axum 0.7` | got-web | Async web framework |
| `tower-http 0.5` | got-web | CORS, static file serving (ServeDir) |

## Agent-to-Agent Integration Points

An agent runtime calls these library entry points directly:

| Operation | Library call | Returns |
|---|---|---|
| Build geometry | `CausalGeometry::from_unembedding(U, eps)` | `CausalGeometry` |
| Fingerprint geometry | `geometry.geometry_hash()` | `[u8; 32]` |
| Measure drift | `geometry.drift_from(&reference)` | `f32` |
| Train probes | `train_probe(data, geometry, ...)` | `ProbeVector` |
| Read probe (frozen) | `read_probe(probe, h, geometry)` | `(f32, f32, bool)` |
| Read probe (drift-aware) | `read_probe_checked(probe, set, h, geo, ref)` | `Result<(f32, f32, bool)>` |
| Causal check (single) | `causal_check(probe, h, geom, delta, model_fn, threshold)` | `CausalScore` |
| Causal check (multi-layer) | `causal_check_multi_layer(...)` | `MultiLayerCausalResult` |
| Capture activations | `MockDmaTap::capture(layer, pos, values)` | `ActivationFrame` |
| Enclave pipeline | `enclave_pipeline(enclave, capture, acts, model_fn, ...)` | `(GeometricAttestation, Vec<CausalScore>)` |
| Sign attestation | `assemble_and_sign(attestation, key)` | `Result<GeometricAttestation>` |
| Verify attestation | `verify(attestation, peer_pk)` | `Result<bool>` |
| Hash for chaining | `attestation_hash(attestation)` | `Result<[u8; 32]>` |
| Verify chain | `verify_chain(chain, current, pks: &[VerifyingKey], max_drift)` | `Result<ChainVerdict>` |
| Build exchange | `build_request(nonce, peer_id, key, chain, current)` | `Result<ExchangeRequest>` |
| Full exchange | `perform_exchange(init_key, ..., resp_key, ..., registry)` | `Result<(ExchangeResult, Verdict)>` |
| Create envelope | `ExchangeEnvelope::create(nonce, peer_id, attest, anchor, ts, sk)` | `Result<ExchangeEnvelope>` |
| Verified deserialise | `ExchangeEnvelope::from_bytes_verified(data, id, nonce, attest, anchor, pk, now, max)` | `Result<ExchangeEnvelope>` |
| Store attestation | `store.append(attestation, verifying_key)` | `Result<StoreId>` |
| Query store | `store.query(&filter)` | `Vec<&GeometricAttestation>` |
| Audit chain | `store.audit(model_id)` | `AuditReport` |
| Distribution shift | `detect_distribution_shift(baseline, current, sigmas)` | `f32` (fraction) |
