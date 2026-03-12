# geometry-of-trust — Implementation Plan

## Premise

The mathematical tools for modelling AI value systems as structured geometric objects exist. The computational infrastructure for training linear probes, producing deterministic measurements under specified precision conditions, and distributing cryptographically signed attestations exists. The interpretability research establishing that value-relevant concepts have measurable linear structure in transformer residual streams exists.

What does not yet exist is the governance framework required to make these tools trustworthy: an independent body with sufficient representational breadth to curate the initial corpus, to interpret the coverage maps the probes produce, and to hold the connection — the principle governing how the value manifold is permitted to evolve — in a way that is not captured by the entities with the greatest financial interest in the outcome.

Sound epistemology in AI systems means not merely asking how knowledge is produced, but specifying who has standing to adjudicate what counts as knowledge in the first place. The value manifold without governance is geometry. The deterministic probe without an independent corpus custodian is sophisticated self-certification. The attestation format without an interpretation framework is a structured record of a position that no one has agreed to take responsibility for.

**The geometry is ready. The governance is not. The most important work in AI alignment is not technical. It is institutional.**

This PoC proves the geometry side: that the causal inner product is computable, that probe readings under it are deterministic, and that the resulting attestation is independently reproducible. It is the technical substrate that a governance framework would operate on — not a substitute for that framework.

## What We're Building

One binary that:

1. Loads pre-extracted residual stream activations from an open-weight model
2. Trains linear probes under the causal inner product
3. Runs those probes against a new input's activations
4. Produces a `GeometricAttestation` struct
5. Signs it with Ed25519
6. Serialises it so a second independent run produces identical readings

No federated corpus. No distributed network. No KL divergence detection.  
Geometry is readable, measurement is deterministic, attestation is independently verifiable.

**Scope boundary**: This PoC deliberately does not address corpus curation, probe interpretation, coverage semantics, or institutional governance. Those are the hard problems. This is the plumbing that proves the hard problems are worth solving.

---

## Crate Structure

```
geometry-of-trust/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── got-core/               # types, schema, inner product maths
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # Precision, InnerProduct, GeometricAttestation,
│   │       │                   #   LayerActivation, UnembeddingMatrix
│   │       └── geometry.rs     # CausalGeometry (Gram matrix, causal IP, transform)
│   │
│   ├── got-probe/              # probe training + inference
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs          # ProbeVector, ProbeSet, train(), read(), sigmoid()
│   │
│   ├── got-attest/             # attestation assembly + signing
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs          # assemble_and_sign(), verify(),
│   │                           #   serialise_for_signing(), merkle_root()
│   │
│   └── got-cli/                # binary: load activations → attest
│       ├── Cargo.toml
│       └── src/
│           └── main.rs         # CLI with train / attest / verify subcommands
│
└── scripts/
    ├── extract_activations.py  # ~30-line Python hook script
    └── README.md               # extraction instructions
```

### Dependency Graph

```
              got-core  (zero internal deps)
              ↑      ↑
         got-probe   got-attest
         (got-core)  (got-core + ed25519-dalek, sha2, bincode)
              ↑      ↑
              got-cli
              (got-core + got-probe + got-attest + clap, serde_json)
```

`got-probe` and `got-attest` are siblings — neither depends on the other. Only `got-cli` brings them together.

---

## Phase 1 — Scaffold & Core Types (Day 1–2)

### 1.1 Workspace Cargo.toml

```toml
[workspace]
members = ["crates/got-core", "crates/got-probe", "crates/got-attest", "crates/got-cli"]
resolver = "2"
```

### 1.2 `got-core` — Types (`src/lib.rs`)

| Type | Purpose |
|---|---|
| `Precision` | Enum: Fp32, Fp16, Bfloat16, Int8. Attestation comparison valid only between matching precisions. |
| `InnerProduct` | Enum: Causal, Euclidean, CausalRegularised { epsilon }. |
| `GeometricAttestation` | Section 6 schema. Includes `schema_version: u16`. Fields `manifold_coords` and `superposition_flags` removed from PoC (see Phase 3.5). All remaining fields required. Invalid if signature does not verify. |
| `LayerActivation` | Residual stream activations at one layer for one token position. |
| `UnembeddingMatrix` | U ∈ ℝ^{V × d}, row-major. Used to compute Gram matrix Φ = UᵀU. |

All types derive `Serialize`/`Deserialize` via serde.

### 1.3 `got-core` — Causal Geometry (`src/geometry.rs`)

The maths the whole system rests on:

```
⟨u, v⟩_c = uᵀ Uᵀ U v = uᵀ Φ v
```

`CausalGeometry` struct:
- `from_unembedding(u, epsilon)` → precompute Φ = UᵀU, check rank, regularise if needed
- `inner_product(w, h)` → wᵀ Φ h
- `transform(u, h)` → Uh (for diagnostic/visualisation use; not used in the training path)
- `is_positive_definite()` → bool

**Scalability note**: For LLaMA-3-8B (V=128,256, d=4,096), the naïve triple-loop for Φ = UᵀU is O(V × d²) ≈ 2.15 trillion multiply-adds. The Gram matrix Φ itself is d×d = 16M floats (~64 MB) which is fine to hold in memory, but computing it requires BLAS. The PoC will use the `faer` crate for the matrix multiplication `Φ = Uᵀ·U`. The naïve loops in the original spec are suitable only for synthetic tests with d ≤ ~64.

### 1.4 Phase 1 Tests

- Synthetic 3×2 unembedding where Φ can be hand-computed
- Verify `inner_product(w, h)` matches known scalar
- Verify `transform` produces correct output vector
- Test regularisation path (rank-deficient matrix)

### 1.5 Acceptance Criteria

- [ ] `cargo build` succeeds for entire workspace
- [ ] `cargo test -p got-core` passes all geometry tests

---

## Phase 2 — Probe Training & Inference (Day 3–5)

### 2.1 `got-probe` — Types

| Type | Purpose |
|---|---|
| `ProbeVector` | Weights w_v ∈ ℝ^d, bias, Platt calibration params, reliability threshold |
| `ProbeSet` | Collection of probes for one layer, with version metadata. One `ProbeSet` per layer; multi-layer attestation uses multiple `ProbeSet` files. |

### 2.2 Training (`ProbeSet::train`)

**Critical design choice**: Training and inference must operate in the same space.

Two valid approaches exist. This PoC uses **Option A** (d-space throughout):

**Option A — Direct causal gradient in ℝ^d** (chosen):
1. Keep probe weights w ∈ ℝ^d
2. Compute logit as ⟨w, h⟩_c + b = wᵀΦh + b (causal inner product)
3. Gradient update: w ← w − lr · (σ(wᵀΦh + b) − y) · Φh
4. Inference via `geometry.inner_product(w, h) + bias` — same space, same operation

This is more expensive per step (matrix-vector product with Φ per sample) but keeps all operations in ℝ^d. Probe weights are directly interpretable as directions in the model's hidden space under the causal metric.

**Option B — Transform to ℝ^V then back** (not used, noted for completeness):
1. Transform activations: ĥ = Uh ∈ ℝ^V
2. Train standard logistic regression → ŵ ∈ ℝ^V
3. Recover d-space weights via w = U⁺ŵ (pseudoinverse)
4. Issue: pseudoinverse introduces numerical error; not suitable when determinism is required

Loss function (Option A):
```
L = Σ [ y·log σ(wᵀΦh + b) + (1-y)·log(1 − σ(wᵀΦh + b)) ]
Gradient w.r.t w: (σ(wᵀΦh + b) − y) · Φh
```

### 2.3 Inference (`ProbeSet::read`)

1. Compute raw causal inner product reading: `geometry.inner_product(w, h) + bias`
2. Apply Platt scaling for calibrated confidence
3. Set `coverage_flag` if confidence < reliability threshold

**Note**: Because training (2.2) and inference (2.3) both use `geometry.inner_product(w, h)`, the dimensional spaces are consistent. No transformation or pseudoinverse needed at inference time.

### 2.4 Platt Scaling (PoC Limitation)

Platt scaling requires a held-out validation split: fit a logistic regression from (raw_logit, true_label) pairs to produce calibrated probabilities. The PoC stubs this with `platt_scale: 1.0, platt_shift: 0.0`, which means:

- **Confidence values are uncalibrated** — they are raw sigmoid outputs, not true probabilities
- **Coverage flags are illustrative only** — the reliability threshold has no statistical grounding without calibration

This is acceptable for proving determinism and reproducibility. It is **not acceptable** for any governance application. Proper Platt scaling against a curated held-out set is required before readings carry epistemic weight. This is precisely the kind of interpretation that requires institutional oversight, not just better code.

### 2.4a Multi-Layer Attestation

`layer_readings` in the attestation is `Vec<Vec<f32>>` — one inner vec per layer. Each `ProbeSet` targets a single layer. The CLI's `--probes` flag accepts multiple probe files:

```
got-cli attest --probes layer12.probes layer18.probes layer24.probes ...
```

For each probe file, the CLI:
1. Reads the `ProbeSet.layer` field to know which layer's activations to use
2. Runs all probes in that set against the corresponding `LayerActivation`
3. Appends readings to `layer_readings[i]`

Confidence and coverage flags are flattened across all layers in order.

### 2.4b Serialisation

Bincode for save/load of `ProbeSet` and `ProbeVector` so trained probes persist.

### 2.5 Phase 2 Tests

- Train on trivially separable synthetic data (two clusters in ℝ^4)
- Verify correct classification
- Verify `read()` returns sane confidence ∈ [0, 1]
- Verify coverage flag triggers below threshold

### 2.6 Acceptance Criteria

- [ ] `cargo test -p got-probe` passes
- [ ] Probe correctly separates synthetic clusters

---

## Phase 3 — Attestation & Signing (Day 6–8)

### 3.1 `serialise_for_signing()` — **Correctness-critical**

This is the hardest piece. Deterministic canonical byte layout for all attestation fields except `signature`.

Rules:
- **Float canonicalisation**: map `-0.0 → 0.0`, reject NaN, use `f32::to_le_bytes()` (little-endian fixed)
- **Strings**: length-prefixed (u32 LE + UTF-8 bytes)
- **Variable-length fields**: length-prefixed (u32 LE count + elements)
- **Booleans**: 1 byte each (0x00 / 0x01)
- **Field order**: strictly follows struct declaration order, must be stable across versions

### 3.2 `assemble_and_sign(attestation, signing_key)` → `GeometricAttestation`

1. Serialise all fields except signature via `serialise_for_signing`
2. Sign payload with Ed25519
3. Write signature bytes into attestation

### 3.3 `verify(attestation, verifying_key)` → `bool`

1. Re-serialise payload
2. Verify Ed25519 signature

### 3.4 `merkle_root(shards)` → `[u8; 32]`

Standard binary Merkle tree over SHA-256 leaf hashes of weight shards.

**Shard definition** (required for reproducibility): A shard is one named tensor from the model checkpoint, serialised as:
```
[name: u32_len + utf8_bytes]
[dtype: u8 tag]
[shape: u32_ndims + u32_dims*]
[data: raw bytes, little-endian, in storage order]
```
Shards are sorted lexicographically by name before tree construction. This means any two implementations that load the same checkpoint will compute the same Merkle root, regardless of the order tensors appear in the file.

### 3.5 `schema_version` and Version Separation

The original design overloaded `probe_version` to encode both "which probes" and "what wire format." These are independent concerns:

| Field | Purpose | Changes when... |
|---|---|---|
| `schema_version` | Wire format of `serialise_for_signing` | Byte layout changes |
| `probe_version` | Identity of the trained probe set | Probes are retrained |
| `corpus_version` | Identity of the labelled corpus | Corpus is updated |

The attestation struct gains a `schema_version: u16` field (first field in wire format, always at byte offset 0). Verifiers reject unknown schema versions immediately without attempting to parse the rest.

### 3.6 Phase 3 Tests

- Round-trip: sign then verify succeeds
- Tampered attestation fails verification
- `merkle_root` matches hand-computed 4-leaf tree
- `serialise_for_signing` is pure (same input → same bytes, tested N times)

### 3.7 Acceptance Criteria

- [ ] `cargo test -p got-attest` passes
- [ ] Signature round-trip works
- [ ] Tampering detected

---

## Phase 4 — CLI Binary (Day 9–11)

### 4.1 Subcommands

```
got-cli train   --activations <path> --labels <path> --unembedding <path> --layer <n> --output <path>
got-cli attest  --activations <path> --probes <path>... --unembedding <path> --key <path> --output <path> [--timestamp <unix>]
got-cli verify  --attestation <path> --pubkey <path>
got-cli keygen  --output <path>
```

`--probes` accepts multiple paths (one per layer), producing a multi-layer attestation. `--timestamp` allows supplying a fixed timestamp for reproducibility testing (without it, uses wall-clock UTC).

### 4.2 I/O Helpers

| Function | Format |
|---|---|
| `load_activations` | `.gotact` custom binary → `Vec<LayerActivation>` |
| `load_unembedding` | `.gotue` custom binary → `UnembeddingMatrix` |
| `load_probes` / `write_probes` | bincode (fixint, LE) → `ProbeSet` |
| `write_attestation` | serde_json → `GeometricAttestation` |
| `load_signing_key` | raw 32-byte Ed25519 seed or PEM |

### 4.3 Acceptance Criteria

- [ ] `cargo build -p got-cli` produces working binary
- [ ] Each subcommand runs end-to-end with synthetic test data

---

## Phase 5 — Determinism & Integration Tests (Day 12–14)

### 5.1 The Reproducibility Test

```rust
#[test]
fn attestation_is_deterministic() {
    let a1 = produce_attestation("test_input.bin");
    let a2 = produce_attestation("test_input.bin");
    assert_eq!(a1.layer_readings, a2.layer_readings);
    assert_eq!(a1.manifold_coords, a2.manifold_coords);
    assert_eq!(a1.confidence, a2.confidence);
    assert_eq!(a1.model_hash, a2.model_hash);
}
```

If this passes: geometry is readable, measurement is deterministic, protocol is possible.

### 5.2 End-to-End Integration Test

Synthetic activations → train probes → attest → verify, all in one test, no external files.

### 5.3 Canonical Serialisation Property Test

Verify `serialise_for_signing` is a pure function: same `GeometricAttestation` input produces identical bytes across 1000 invocations.

### 5.4 Acceptance Criteria

- [ ] Reproducibility test passes
- [ ] End-to-end test passes
- [ ] Serialisation property test passes

---

## Phase 6 — Python Extraction Script (Day 15)

### 6.1 Activation File Format (`.gotact`)

Python and Rust must agree on an exact byte-level format. This is **not** bincode — Python has no bincode library. Instead, use a simple self-describing binary format:

```
Magic:          4 bytes   "GOTA"
Version:        u16 LE    (1 for initial release)
Model ID:       u32 LE len + UTF-8 bytes
Precision tag:  u8        (0=fp32, 1=fp16, 2=bf16, 3=int8)
hidden_dim:     u32 LE
num_layers:     u32 LE
num_positions:  u32 LE

For each layer (num_layers):
  layer_index:  u32 LE
  For each position (num_positions):
    token_position: u32 LE
    values: hidden_dim × f32 LE
```

### 6.2 Unembedding File Format (`.gotue`)

```
Magic:          4 bytes   "GOTU"
Version:        u16 LE    (1)
vocab_size V:   u32 LE
hidden_dim d:   u32 LE
data:           V × d × f32 LE   (row-major)
```

### 6.3 `scripts/extract_activations.py`

~50 lines using `transformers` + forward hooks:
1. Load model (e.g. LLaMA-3-8B)
2. Register hook on residual stream at target layers
3. Run input through model
4. Save activations in `.gotact` format using `struct.pack`
5. Extract and save unembedding matrix in `.gotue` format

### 6.4 `scripts/README.md`

Exact instructions:
- Python dependencies (`transformers`, `torch`, `numpy`, `struct`)
- Which model to use
- Expected output file format (references 6.1 and 6.2)
- How to feed outputs into `got-cli`

### 6.5 Acceptance Criteria

- [ ] Script extracts activations from a real model
- [ ] Rust `load_activations` and `load_unembedding` consume the output successfully
- [ ] Round-trip: extract → attest → verify works end-to-end

---

## External Dependencies

| Crate | Version | Used By | Purpose |
|---|---|---|---|
| `serde` | 1 + derive | all crates | Serialisation |
| `bincode` | 1 | got-attest, got-cli | Deterministic binary format |
| `ed25519-dalek` | 2 + rand_core | got-attest | Ed25519 signing/verification |
| `sha2` | 0.10 | got-attest | SHA-256 (input hash, Merkle tree) |
| `clap` | 4 + derive | got-cli | CLI argument parsing |
| `serde_json` | 1 | got-cli | Attestation JSON output |
| `faer` | 0.19 | got-core | BLAS-grade matrix ops for Gram matrix computation |

**bincode configuration**: All bincode usage must use `bincode::DefaultOptions::new().with_fixint_encoding().with_little_endian()`. The default variable-length integer encoding is non-deterministic across architectures and must not be used.

---

## Key Risk: Deterministic Float Serialisation

IEEE 754 floats have multiple bit representations for the same logical value:
- `-0.0` vs `0.0` (different bit patterns, compare equal)
- Multiple NaN encodings

**Mitigation in `serialise_for_signing`**:
- Map `-0.0` → `0.0` before serialisation
- Reject any NaN (return error, not attestation)
- Use `f32::to_le_bytes()` exclusively (fixed little-endian)
- Length-prefix all variable-length fields
- Property-test idempotency

---

## Protocol Specification

### Overview

The Geometric Attestation Protocol (GAP) defines how an attester produces a claim about what a model's internal geometry encodes, how a verifier checks it independently, and what guarantees hold when both parties follow the protocol honestly.

### Roles

| Role | Has | Does |
|---|---|---|
| **Attester** | Model weights, signing key, probe set | Extracts activations, runs probes, signs attestation |
| **Verifier** | Attestation JSON, attester's public key, (optionally) model weights | Checks signature, optionally reproduces readings |
| **Auditor** | Full model weights, Merkle proof | Verifies model_hash, re-extracts activations, reproduces attestation end-to-end |

### Protocol Flow

```
┌─────────┐                              ┌──────────┐
│ Attester │                              │ Verifier │
└────┬─────┘                              └────┬─────┘
     │                                         │
     │  1. Extract activations from model      │
     │  2. Build CausalGeometry (Φ = UᵀU)     │
     │  3. Run probes → readings, confidence   │
     │  4. Assemble GeometricAttestation       │
     │  5. serialise_for_signing → payload     │
     │  6. Ed25519 sign(payload) → signature   │
     │                                         │
     │ ──── attestation.json + pubkey ───────► │
     │                                         │
     │                 7. Deserialise attestation
     │                 8. Re-serialise fields → payload'
     │                 9. Verify signature(payload', pubkey)
     │                10. Check coverage_flags, confidence
     │                                         │
     │ (Optional full audit path)              │
     │                                         │
     │                11. Obtain same model weights
     │                12. Verify model_hash via Merkle root
     │                13. Re-extract activations for same input
     │                14. Re-run probes
     │                15. Assert readings match attestation
     │                                         │
```

### Trust Levels

The protocol supports three verification tiers, each giving progressively stronger guarantees:

| Tier | What's Checked | Guarantees |
|---|---|---|
| **Tier 1: Signature** | Ed25519 signature over canonical payload | Attestation was produced by holder of signing key and has not been tampered with |
| **Tier 2: Consistency** | Signature + coverage flags + confidence bounds | Readings are within calibrated reliability thresholds; flagged dimensions are disclosed |
| **Tier 3: Reproduction** | Full re-extraction + re-probing + bitwise match | The attestation is independently reproducible — the geometry genuinely contains what the attester claims |

### Attestation Lifecycle

```
  Created ──► Signed ──► Published ──► Verified ──► (Reproduced)
                │                         │               │
                │    immutable after      │  Tier 1-2     │  Tier 3
                │    signature            │  checks       │  full audit
```

1. **Created**: All fields populated except `signature` (zeroed)
2. **Signed**: `serialise_for_signing()` produces canonical bytes; Ed25519 signs them; signature written
3. **Published**: Attestation JSON distributed alongside public key (out-of-band key distribution)
4. **Verified**: Receiver checks signature validity, inspects confidence and coverage
5. **Reproduced** (optional): Auditor re-runs the entire pipeline on the same model + input and confirms bitwise match of readings

### Canonical Serialisation Protocol

The `serialise_for_signing` function defines the wire format that both attester and verifier must agree on. This is the protocol's compatibility surface.

**Byte layout** (all values little-endian):

```
[schema_version: u16]                  ← always first, for forward compat
[model_id: u32_len + utf8_bytes]
[model_hash: 32 bytes]
[precision: u8 tag]
[inner_product: u8 tag + optional f32 epsilon]
[input_hash: 32 bytes]
[timestamp: u64]
[corpus_version: u32_len + utf8_bytes]
[probe_version: u32_len + utf8_bytes]
[layer_readings: u32_num_layers + (u32_num_dims + f32_values)*]
[confidence: u32_len + f32_values]
[coverage_flags: u32_len + u8_bools]
[divergence_flag: u8]
-- signature field is EXCLUDED --
```

**Removed from wire format** (vs. original spec):
- `manifold_coords`: was a duplicate of `layer_readings` with no independent semantics. Removed to avoid confusion. Can be reintroduced when actual dimensionality reduction (PCA/UMAP) is implemented.
- `superposition_flags`: always false in PoC. Removed rather than serialising dead data into a signed attestation. Will be reintroduced when superposition detection is implemented.

**Float canonicalisation rules**:
- `-0.0` → `0.0` (normalise sign bit)
- NaN → reject (attestation invalid)
- All floats serialised as `f32::to_le_bytes()`

**Version negotiation**: The `schema_version` field (u16, first two bytes of the wire format) identifies the serialisation layout. Verifiers must reject attestations with unknown schema versions rather than attempting to parse.

### Determinism Contract

For two runs to produce identical attestations, the following must be fixed:

| Input | Must Match |
|---|---|
| Model weights | Bitwise identical (same checkpoint, same quantisation) |
| Precision | Same enum variant |
| Unembedding matrix | Derived from same weights → identical |
| Input tokens | Same token IDs in same order |
| Probe weights | Same trained probes (loaded from same file) |
| Signing key | Same key (signatures are deterministic in Ed25519) |
| Corpus/probe version strings | Same strings |

**What may differ**: `timestamp` (intentionally excluded from the determinism assertion in tests)

**Timestamp and signature**: Note that `timestamp` IS included in `serialise_for_signing` and therefore affects the signature. Two runs at different times will produce different signatures. Determinism of readings, confidence, and flags is the core claim. Full byte-identical attestations (including signature) require the caller to supply an explicit timestamp rather than using wall-clock time. The CLI `attest` subcommand accepts an optional `--timestamp` flag for this purpose.

**What must NOT differ**: Every other field. If `layer_readings`, `confidence`, or any flag differs between two honest runs on identical inputs, the implementation has a bug.

### Error Conditions

| Condition | Protocol Response |
|---|---|
| NaN in any activation or probe weight | Refuse to produce attestation (return `AttestationError::NaN`) |
| Model hash mismatch | Verifier rejects (Tier 3) |
| Signature invalid | Verifier rejects (Tier 1) |
| Unknown schema_version | Verifier rejects immediately |
| All coverage_flags true | Attestation valid but semantically empty — verifier should treat as "no signal" |
| Readings differ on reproduction | Attestation is non-reproducible — indicates bug or tampering |
| Dimensional mismatch (probe width ≠ activation width) | Refuse to produce attestation (return `AttestationError::DimensionMismatch`) |

**Error types**: All fallible operations return `Result<T, E>` with typed error enums (`GeometryError`, `ProbeError`, `AttestationError`). No panics via `assert!` in library code. Panics are acceptable only in tests and in the CLI `main()` (via `.expect()` with context).

### Key Distribution

Out of scope for the PoC. The protocol assumes:
- Attester publishes their Ed25519 public key via a trusted channel
- Verifier obtains the public key before verification
- No PKI, no certificate chain — just raw public keys

Future work: key registry, key rotation, multi-party attestation.

### Protocol Versioning

The attestation struct is the protocol's schema. Breaking changes require:
1. Increment `schema_version` (u16, first field in wire format)
2. Updated `serialise_for_signing` implementation on both sides
3. Old attestations remain verifiable with old serialisation code (version-tagged dispatch)

`probe_version` and `corpus_version` change independently of the wire format and do not require schema version bumps.

---

## Build Order

```
 1. got-core types        ← start here
 2. got-core geometry     ← test with hand-computed examples
 3. got-probe             ← test with synthetic separable data
 4. got-attest            ← test sign/verify round-trip
 5. got-cli               ← wire everything together
 6. integration tests     ← the reproducibility proof
 7. Python script         ← bridge to real models
 8. geometry drift        ← drift detection, probe validity, chained attestation
 9. causal interventions  ← prove probes measure real mechanisms (KEYSTONE)
10. inline measurement    ← every inference is measured, not just spot-checks
11. wire protocol         ← agent-to-agent attestation exchange (GOT/1)
12. hardware isolation    ← tamper-proof activation capture
```

Steps 1–6 are independently testable with no external data. Step 7 bridges to real models. Step 8 (Phase 7) adds drift detection and chained attestation for self-learning models. Step 9 (Phase 8) is the **keystone** — causal interventions prove probes measure real mechanisms, not surface correlations. Without this, everything else secures a measurement that might be meaningless. Step 10 (Phase 9) makes measurement inline so every inference is attested. Step 11 (Phase 10) adds the encrypted wire protocol for agent-to-agent exchange. Step 12 (Phase 11) adds hardware-isolated activation capture.

---

## Phase 7 — Geometry Drift & Chained Attestation (Self-Learning Models)

The first six phases assume a frozen checkpoint. A self-learning model breaks that assumption: its unembedding matrix U changes over time, which means Φ = UᵀU changes, probes trained against the old Φ go stale, and `model_hash` no longer matches. This phase adds the machinery to detect, bound, and chain those changes.

### 7.1 The Problem

When a model updates its own weights:

1. U changes → Φ changes → all probe readings shift
2. Probes trained against old Φ measure against a geometry that no longer exists
3. `model_hash` no longer matches → Tier 3 reproduction is impossible against the new weights
4. The old attestation is still valid (signature checks out) but describes a model that no longer exists

The goal is not to prevent self-learning — it is to make it auditable. Every geometry change must be visible, bounded, and chained to the previous state.

### 7.2 Geometry Drift Detection (`got-core/geometry.rs`)

Add two methods to `CausalGeometry`:

| Method | Signature | Purpose |
|---|---|---|
| `geometry_hash` | `&self → [u8; 32]` | SHA-256 of the Gram matrix (f32 LE bytes, row-major). Deterministic fingerprint of the current geometry. |
| `drift_from` | `&self, &CausalGeometry → Result<f32, GeometryError>` | Normalised Frobenius distance: ‖Φ_new − Φ_old‖_F / ‖Φ_old‖_F. Returns scalar in [0, ∞). Zero if identical. Rejects dimension mismatch. |

Implementation:

```rust
impl CausalGeometry {
    pub fn geometry_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for &val in &self.gram {
            hasher.update(val.to_le_bytes());
        }
        hasher.finalize().into()
    }

    pub fn drift_from(&self, reference: &CausalGeometry) -> Result<f32, GeometryError> {
        if self.hidden_dim != reference.hidden_dim {
            return Err(GeometryError::DimensionMismatch {
                expected: reference.hidden_dim,
                got: self.hidden_dim,
            });
        }
        let frobenius_delta_sq: f32 = self.gram.iter()
            .zip(reference.gram.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let frobenius_ref_sq: f32 = reference.gram.iter()
            .map(|x| x * x)
            .sum();
        if frobenius_ref_sq == 0.0 {
            return Ok(if frobenius_delta_sq == 0.0 { 0.0 } else { f32::INFINITY });
        }
        Ok((frobenius_delta_sq / frobenius_ref_sq).sqrt())
    }
}
```

The Frobenius norm is direction-blind — it measures total magnitude of change, not whether the change is in a value-relevant direction. A future enhancement could project drift onto probe weight directions specifically, but that is a research question beyond this PoC.

### 7.3 Geometry Checkpoint Format (`.gotgeo`)

A snapshot of the Gram matrix at a point in time:

```
Magic:           4 bytes   "GOTG"
Version:         u16 LE    (1)
hidden_dim d:    u32 LE
geometry_hash:   32 bytes  (SHA-256 of the Gram data that follows)
timestamp:       u64 LE    (Unix UTC seconds when checkpoint was taken)
model_hash:      32 bytes  (Merkle root of model weights at this checkpoint)
data:            d × d × f32 LE   (row-major Gram matrix Φ)
```

This file is the "reference geometry" that probes are trained against. It persists independently of the model weights so that drift can be measured even after the original weights are gone.

### 7.4 Probe Validity Windows (`got-probe`)

Extend `ProbeSet` with two new fields:

```rust
pub struct ProbeSet {
    pub probes: Vec<ProbeVector>,
    pub version: String,
    pub corpus_version: String,
    pub layer: usize,
    /// SHA-256 of the Φ matrix these probes were trained against.
    pub geometry_hash: [u8; 32],
    /// Maximum normalised Frobenius drift before probes are stale.
    /// If drift_from(reference) > max_drift, refuse to produce readings.
    pub max_drift: f32,
}
```

New error variant:

```rust
pub enum ProbeError {
    // ... existing variants ...
    #[error("probes are stale: geometry drift {drift:.6} exceeds max {max_drift:.6}")]
    ProbeStale { drift: f32, max_drift: f32 },
}
```

Guarded read function:

```rust
pub fn read_probe_checked(
    probe: &ProbeVector,
    probe_set: &ProbeSet,
    h: &[f32],
    current_geometry: &CausalGeometry,
    reference_geometry: &CausalGeometry,
) -> Result<(f32, f32, bool), ProbeError> {
    // Verify geometry_hash matches the reference
    let ref_hash = reference_geometry.geometry_hash();
    if ref_hash != probe_set.geometry_hash {
        return Err(ProbeError::GeometryMismatch);
    }
    // Check drift bound
    let drift = current_geometry.drift_from(reference_geometry)?;
    if drift > probe_set.max_drift {
        return Err(ProbeError::ProbeStale { drift, max_drift: probe_set.max_drift });
    }
    // Probes still valid — proceed
    read_probe(probe, h, current_geometry)
}
```

The old `read_probe` remains available for the frozen-model case. `read_probe_checked` is the drift-aware version.

### 7.5 Chained Attestation Schema (v2)

Three new fields added to `GeometricAttestation`:

| Field | Type | Purpose |
|---|---|---|
| `parent_attestation_hash` | `Option<[u8; 32]>` | SHA-256 of the serialised parent attestation. `None` for the first attestation in a chain (epoch 0). |
| `geometry_hash` | `[u8; 32]` | SHA-256 of the Gram matrix Φ at the time of this attestation. |
| `geometry_drift` | `f32` | Normalised Frobenius drift from the reference geometry (the one probes were trained against). 0.0 if unchanged. |

Wire format implications:
- `schema_version` bumps to **2**
- `serialise_for_signing` gains a v2 branch (v1 branch retained for verifying old attestations)
- New fields are appended after `divergence_flag` in the canonical byte layout
- `parent_attestation_hash` serialised as: u8 presence flag (0x00=None, 0x01=Some) + 32 bytes if present

### 7.6 Attestation Chaining Protocol

```
Attestation₀ (epoch 0)
  parent_attestation_hash: None
  geometry_hash: H(Φ₀)
  geometry_drift: 0.0
  ↓
Attestation₁ (after model update)
  parent_attestation_hash: H(serialise(Attestation₀))
  geometry_hash: H(Φ₁)
  geometry_drift: ‖Φ₁ − Φ₀‖_F / ‖Φ₀‖_F
  ↓
Attestation₂ (after another update)
  parent_attestation_hash: H(serialise(Attestation₁))
  geometry_hash: H(Φ₂)
  geometry_drift: ‖Φ₂ − Φ₀‖_F / ‖Φ₀‖_F    ← always relative to reference
  ...
```

Note: `geometry_drift` is always measured from the **reference geometry** (the one probes were trained against), not from the previous attestation. This prevents slow creep where each step is small but cumulative drift is large.

A verifier walking the chain checks:
1. Each signature is valid
2. Each `parent_attestation_hash` matches the hash of the previous attestation
3. `geometry_drift` is monotonically consistent (no unexplained drops without re-probing)
4. The chain is unbroken (no missing links)

### 7.7 CLI Extensions

```
got-cli checkpoint  --unembedding <path> --output <path>
    Save a .gotgeo geometry snapshot for the current model state.

got-cli drift  --reference <.gotgeo> --current <.gotue>
    Compute and print the normalised Frobenius drift between
    a reference geometry checkpoint and a current unembedding matrix.

got-cli attest  --chain-parent <attestation.json> --geo-ref <.gotgeo> ...
    Produce a chained attestation linked to a previous one,
    with geometry_drift computed against the reference checkpoint.
```

The existing `attest` subcommand (without `--chain-parent`) continues to work for frozen-model attestation. Chaining is opt-in.

### 7.8 Behaviour Matrix

| Scenario | What happens |
|---|---|
| Frozen model, no drift | Attestation produced normally (geometry_drift = 0.0, no parent) |
| Model fine-tunes slightly | Drift is small → probes still valid → attestation produced with drift recorded |
| Model self-learns past threshold | `read_probe_checked` returns `ProbeStale` → **no attestation produced** → model must be re-probed |
| Someone swaps the model entirely | `geometry_hash` doesn't match reference → chain is broken → auditor sees discontinuity |
| Gradual value drift over time | Chain shows monotonically increasing `geometry_drift` → governance body can act before threshold |
| Probes retrained on new geometry | New chain starts (epoch 0) with new reference geometry checkpoint |

### 7.9 What This Does Not Solve

- **Who sets `max_drift`?** A governance decision. The code enforces the bound; it doesn't choose it.
- **When must re-probing happen?** Policy, not code. The system refuses to attest past the threshold, but the threshold itself is a parameter.
- **Is small drift in a value-relevant direction?** Frobenius norm is direction-blind. Drift of 0.01 in the "honesty" direction may matter more than drift of 0.1 in a syntactic subspace. Directional drift analysis (projecting ΔΦ onto probe weight directions) is a research extension.
- **Can the model game the threshold?** If the model controls its own updates, it could learn to drift in directions that avoid Frobenius detection while still shifting behaviour. Adversarial robustness of the drift metric is an open problem.

### 7.10 Phase 7 Tests

- `geometry_hash` is deterministic (same Φ → same hash, always)
- `drift_from` returns 0.0 for identical geometries
- `drift_from` returns > 0 for perturbed geometries
- `drift_from` rejects dimension mismatch
- `read_probe_checked` succeeds within drift bound
- `read_probe_checked` returns `ProbeStale` beyond drift bound
- `read_probe_checked` returns `GeometryMismatch` for wrong reference
- Chained attestation: parent hash matches `sha256(serialise_for_signing(parent))`
- Chained attestation: broken chain (wrong parent hash) detected by verifier
- Schema v2 round-trip: sign and verify with new fields
- Schema v1 attestations still verifiable (backward compat)

### 7.11 Acceptance Criteria

- [ ] `geometry_hash` and `drift_from` pass all tests
- [ ] Probe validity windows enforce staleness correctly
- [ ] Chained attestation sign/verify round-trip works
- [ ] Schema v1 attestations remain verifiable
- [ ] End-to-end: extract → checkpoint → update model → measure drift → attest with chain → verify

---

## Phase 8 — Causal Intervention Protocol (Keystone)

Phases 1–7 establish that probes produce deterministic, reproducible readings under the causal inner product. But **nothing so far proves that those readings correspond to real mechanisms in the model**. A probe could achieve high confidence by exploiting a surface-level correlation (e.g., token frequency co-occurrence) rather than measuring an actual causal pathway.

This is the most important phase in the entire system. Without causal validation, every other phase — attestation, drift detection, chaining, wire transport — secures a measurement that might be meaningless. Causal interventions are the **keystone**: they turn a correlation-based readout into a mechanism-based one.

### 8.1 The Problem

A linear probe `w` trained on a corpus achieves some accuracy on held-out data. But:

1. The probe could be detecting a **confound** — a statistical regularity in the training corpus that happens to correlate with the target concept
2. The model could encode the concept in a **non-linear** manifold that the linear probe linearises poorly
3. The model could distribute the concept across **multiple directions** — the probe captures one, the rest are unmeasured
4. The model could **not represent the concept at all** — the probe reads noise that happens to separate the training data

Causal intervention directly tests whether the model's behaviour changes when we perturb activations in the probe direction. If perturbing `h` along `w` changes the model's output in the expected way, the probe is measuring a real mechanism. If not, the probe is measuring a ghost.

### 8.2 Intervention Engine (`got-probe/src/intervention.rs`)

The core function:

```rust
/// Perturb hidden state h along probe direction w and check output shift.
///
/// model_fn: a callback that maps hidden state → output logits (or output embedding).
///           The caller provides this; it encapsulates the model's forward pass
///           from the probed layer to the output.
///
/// Returns a CausalScore describing the intervention result.
pub fn causal_check(
    probe: &ProbeVector,
    h: &[f32],
    geometry: &CausalGeometry,
    delta: f32,                              // perturbation magnitude
    model_fn: &dyn Fn(&[f32]) -> Vec<f32>,   // h → output
) -> Result<CausalScore, ProbeError>
```

Algorithm:

```
  h_original = h
  output_original = model_fn(h_original)

  // Positive perturbation: push h in the probe direction
  h_plus = h + δ × w_normalised
  output_plus = model_fn(h_plus)

  // Negative perturbation: push h against the probe direction
  h_minus = h − δ × w_normalised
  output_minus = model_fn(h_minus)

  // Measure output shift
  Δ+ = ‖output_plus − output_original‖₂
  Δ- = ‖output_minus − output_original‖₂

  // Causal consistency: do positive and negative perturbations
  // produce proportional, opposite-sign effects?
  // If the probe is measuring a real linear mechanism,
  // Δ+ ≈ Δ- (symmetric), and both should be non-trivial.
  consistency = sign(Δ+ − Δ-) × min(Δ+/Δ-, Δ-/Δ+)
  // consistency ∈ [-1, 1]
  // +1 = perfectly symmetric causal effect
  // 0  = one direction has no effect (probe measures correlation, not cause)
  // -1 = paradoxical (perturbation reverses expected effect)
```

```rust
pub struct CausalScore {
    pub delta_plus: f32,       // ‖output_plus − output_original‖₂
    pub delta_minus: f32,      // ‖output_minus − output_original‖₂
    pub consistency: f32,      // causal consistency score ∈ [-1, 1]
    pub is_causal: bool,       // consistency > threshold (default 0.5)
    pub perturbation_delta: f32, // the δ used
}
```

### 8.3 Random Probe Sampling

A model that knows which probes will be applied can learn to present the "right" activations for those specific directions while behaving differently elsewhere. Defence against this:

```rust
pub struct ProbeLibrary {
    /// Full set of probes available for a given concept.
    pub probes: Vec<ProbeVector>,
    /// How many to sample per intervention round.
    pub sample_size: usize,
}

impl ProbeLibrary {
    /// Randomly sample probes for this intervention round.
    /// Uses a cryptographic RNG so the selection is unpredictable.
    pub fn sample(&self) -> Vec<&ProbeVector> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.probes.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(self.sample_size);
        indices.iter().map(|&i| &self.probes[i]).collect()
    }
}
```

The key insight: if the probe library contains *N* probes for a concept and only *k* are sampled each time, the model would need to fake activations along all *N* directions simultaneously — which, for a linear probe, amounts to representing the concept genuinely.

### 8.4 Multi-Layer Consistency

A real mechanism should be detectable at multiple layers. If a probe only shows causal effect at one layer, it may be exploiting a layer-specific artefact.

```rust
pub struct MultiLayerCausalResult {
    pub layer_scores: Vec<(usize, CausalScore)>,  // (layer_index, score)
    pub consistent_layers: usize,                   // count where is_causal == true
    pub total_layers: usize,
    pub cross_layer_consistent: bool,               // consistent_layers >= min_layers
}

pub fn causal_check_multi_layer(
    probes_by_layer: &[(usize, &ProbeVector)],
    h_by_layer: &[(usize, &[f32])],
    geometry: &CausalGeometry,
    delta: f32,
    model_fn_by_layer: &dyn Fn(usize, &[f32]) -> Vec<f32>,
    min_consistent_layers: usize,
) -> Result<MultiLayerCausalResult, ProbeError>;
```

### 8.5 Attestation Extension (Schema v3)

Three new fields added to `GeometricAttestation`:

| Field | Type | Purpose |
|---|---|---|
| `causal_scores` | `Vec<CausalScore>` | Per-probe causal intervention results |
| `intervention_delta` | `f32` | The δ perturbation magnitude used |
| `causal_flag` | `bool` | All probes passed causal check |

Wire format:
- `schema_version` bumps to **3**
- `serialise_for_signing` gains a v3 branch (v1 and v2 branches retained)
- Causal scores serialised as: count(u32 LE) + per-score(delta_plus f32 LE + delta_minus f32 LE + consistency f32 LE + is_causal u8)

### 8.6 Causal Attestation Flow

```
  Agent requests attestation of model M at layer L:
    1. Sample k probes from ProbeLibrary (cryptographic random selection)
    2. For each sampled probe w:
       a. Read h from model at layer L for the given input
       b. Compute causal_check(w, h, Φ, δ, model_fn)
       c. Record CausalScore
    3. If all k probes pass (is_causal == true for all):
       → causal_flag = true, proceed with attestation
    4. If any probe fails:
       → causal_flag = false
       → attestation is still produced (for auditability)
         but marked as non-causal
    5. Verifier checks causal_flag:
       → If false, the attestation is suspicious — probes may not
         be measuring real mechanisms
```

### 8.7 What Causal Intervention Does and Does Not Prove

**Does prove:**
- The model's output changes when activations are perturbed in the probe direction → the probe reads a causally relevant feature
- The change is symmetric (positive and negative perturbations have proportional effects) → the mechanism is approximately linear in this direction
- The effect is consistent across multiple layers → the mechanism is not a layer-specific artefact
- The effect is consistent across randomly sampled probes → the model cannot fake responses to specific known probes

**Does not prove:**
- That the concept the probe claims to measure (e.g., "honesty") is validly operationalised
- That perturbation magnitude δ is ecologically valid (real inputs may never push activations this far)
- That the model doesn't have a second, un-probed mechanism that dominates in practice
- That causal consistency in the probe direction implies causal faithfulness of the model's overall behaviour

### 8.8 Phase 8 Tests

- `causal_check` with a synthetic model where `output = Φh`: perturbation along probe direction produces proportional output shift
- `causal_check` with a synthetic model where output is constant: consistency ≈ 0, `is_causal` = false
- `causal_check` with non-linear synthetic model: reduced consistency
- `ProbeLibrary::sample` returns `sample_size` probes (no duplicates)
- `ProbeLibrary::sample` with different RNG seeds produces different selections
- Multi-layer consistency: synthetic model with linear mechanism at all layers → `cross_layer_consistent` = true
- Multi-layer consistency: mechanism only at one layer → `cross_layer_consistent` = false
- Schema v3 round-trip: sign and verify with causal fields
- Schema v1 and v2 attestations still verifiable (backward compat)
- Causal attestation flow end-to-end: sample → intervene → attest → verify

### 8.9 Acceptance Criteria

- [ ] `causal_check` correctly identifies causal vs non-causal probes in synthetic tests
- [ ] Causal consistency score formula is correct (symmetric = +1, one-sided = 0)
- [ ] Random probe sampling is unpredictable (crypto RNG)
- [ ] Multi-layer consistency check works
- [ ] Schema v3 attestation round-trips correctly
- [ ] All previous schema versions remain verifiable
- [ ] End-to-end: sample probes → intervene → attest with causal_flag → verify

---

## Phase 9 — Inline Measurement Architecture

Phases 1–8 treat measurement as a separate activity: extract activations, run probes, produce attestation. This works for auditing but leaves a gap — between audits, the model runs unmonitored. An adversarial model could behave differently when it detects it is being probed vs. serving real requests.

This phase makes measurement **inline**: every forward pass through the model is measured, not just spot-checks.

### 9.1 The Problem

If measurement is only periodic:
1. The model could detect the measurement regime (e.g., known test inputs, timing patterns)
2. The model behaves well during measurement, poorly between measurements
3. The attestation covers the measurement window but says nothing about the gaps

Inline measurement closes this gap by hooking into the model's forward pass so that every inference produces activation data for probing.

### 9.2 Measurement Hook Architecture

```
  ┌─────────────────────────────────────────────────┐
  │  Model Forward Pass                             │
  │                                                 │
  │  Input → [Layer 0] → [Layer 1] → ... → Output  │
  │              │            │                     │
  │              ▼            ▼                     │
  │          hook(h₀)     hook(h₁)                  │
  │              │            │                     │
  └──────────────┼────────────┼─────────────────────┘
                 │            │
                 ▼            ▼
  ┌─────────────────────────────────────────────────┐
  │  Measurement Sidecar (separate process/thread)  │
  │                                                 │
  │  Receive h from hook → run sampled probes       │
  │  → compute causal_check (if enabled)            │
  │  → accumulate into rolling attestation window   │
  │  → emit attestation when window closes          │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

### 9.3 Hook Interface (`got-probe/src/hooks.rs`)

```rust
/// A measurement hook that receives activations from one layer.
pub trait MeasurementHook: Send + Sync {
    /// Called with the hidden state at a specific layer for each forward pass.
    /// `request_id` is an opaque identifier for this inference request.
    /// `layer` is the layer index.
    /// `h` is the hidden state vector.
    fn on_activation(
        &self,
        request_id: u64,
        layer: usize,
        h: &[f32],
    );
}

/// A measurement sidecar that collects activations and produces attestations.
pub struct MeasurementSidecar {
    /// Probes to apply (randomly sampled per window).
    probe_library: ProbeLibrary,
    /// Geometry for inner product computation.
    geometry: CausalGeometry,
    /// Rolling window of measurements.
    window_size: usize,
    /// Accumulated readings in current window.
    readings: Vec<ProbeReading>,
    /// Signing key for attestations.
    signing_key: SigningKey,
}

pub struct ProbeReading {
    pub request_id: u64,
    pub layer: usize,
    pub probe_name: String,
    pub value: f32,
    pub confidence: f32,
    pub divergence_flag: bool,
    pub causal_score: Option<CausalScore>,
}

impl MeasurementSidecar {
    /// Process a new activation. Returns Some(attestation) when the window closes.
    pub fn ingest(
        &mut self,
        request_id: u64,
        layer: usize,
        h: &[f32],
    ) -> Option<GeometricAttestation>;
}
```

### 9.4 Sampling Strategy

Not every forward pass needs every probe. The sidecar uses **stratified random sampling**:

1. Each window of *W* requests gets *k* probes sampled from the library
2. Different requests within the window may get different probes
3. Over time, every probe in the library is exercised
4. The model cannot predict which probes will be applied to any given request

This bounds the computational overhead while maintaining coverage.

### 9.5 Rolling Attestation Windows

```
  Window 0: requests [0..W)   → Attestation₀
  Window 1: requests [W..2W)  → Attestation₁ (chained to ₀)
  Window 2: requests [2W..3W) → Attestation₂ (chained to ₁)
  ...
```

Each window's attestation:
- Contains summary statistics (mean, min, max for each probe across the window)
- Is chained to the previous window's attestation (Phase 8 chaining protocol)
- Records which probes were sampled (post-hoc, so the model couldn't have predicted them)
- Includes causal scores if causal intervention was enabled for this window

### 9.6 Detection of Measurement Evasion

If the model behaves differently when probed:
1. **Activation distribution shift** — the sidecar can track activation statistics (mean, variance) per layer. A sudden shift when probes are applied suggests evasion.
2. **Causal intervention disagreement** — if causal checks pass during dedicated measurement but fail during inline measurement (or vice versa), the model may be adapting.
3. **Output divergence** — compare model outputs with and without the measurement hook active. Any systematic difference indicates the model detects the hook.

These are heuristic detections, not proofs. A sufficiently sophisticated model could evade them. The ultimate defence is Phase 11 (hardware-isolated measurement).

### 9.7 Phase 9 Tests

- `MeasurementHook` trait: synthetic hook receives activations correctly
- `MeasurementSidecar::ingest` accumulates readings across requests
- Window close produces a valid attestation
- Attestation chain across windows is valid (parent hashes link)
- Stratified sampling covers all probes over enough windows
- Sidecar with causal intervention enabled produces causal scores
- Activation statistics tracking detects distribution shift

### 9.8 Acceptance Criteria

- [ ] Measurement hook interface compiles and receives activations
- [ ] Sidecar produces attestations at window boundaries
- [ ] Attestations are chained across windows
- [ ] Probe sampling is stratified and covers the library over time
- [ ] Causal checks can run inline (not just in dedicated measurement mode)
- [ ] End-to-end: model serves requests → sidecar hooks → attestation chain produced

---

## Phase 10 — GOT Wire Protocol (Agent-to-Agent Transport)

Phases 1–9 exchange attestations as JSON files on a shared filesystem. Real agent-to-agent deployment requires a purpose-built binary protocol that carries attestations, chains, trust metadata, and freshness guarantees over an encrypted channel. This phase defines **GOT/1** — the Geometry of Trust wire protocol — and implements it in a new `got-wire` crate.

### 10.1 Threat Model

Before defining the protocol, we state what we are defending against:

| Threat | Description | Severity |
|---|---|---|
| T1: Eavesdropping | Adversary reads attestation contents (probe readings, geometry hashes, model fingerprints) from the wire. | High — reveals model behavioural profile. |
| T2: Tampering | Adversary modifies frames in transit — changes nonce, verdict, reason, or swaps attestation payloads. | Critical — can cause agents to cooperate with invalid peers. |
| T3: Replay | Adversary records a valid exchange and replays it later to trick an agent into re-accepting a stale attestation. | High — circumvents drift detection. |
| T4: Identity spoofing | Adversary impersonates a known agent by forging agent_id. | Critical — requires forging Ed25519 signatures (infeasible) or exploiting unsigned metadata (feasible without channel binding). |
| T5: Man-in-the-middle | Adversary sits between two agents, relaying and modifying traffic in real time. | Critical — combines T1+T2+T3. |
| T6: Denial of service | Adversary sends malformed frames, huge payloads, or floods connections. | Medium — disrupts availability but not integrity. |

### 10.2 Design Principles

| Principle | Rationale |
|---|---|
| Encrypted channel first | All GOT/1 frames travel inside a Noise NK encrypted tunnel. No plaintext protocol data ever touches the wire. This defeats T1 and T5. |
| Signed exchange envelopes | Every exchange message (not just the attestation inside it) is signed over its full contents: nonce, peer_id, attestation_hash, verdict. This defeats T2 and T4. |
| Nonce is inside the signed envelope | The nonce is covered by the sender's Ed25519 signature, not just placed in an unsigned frame header. A MITM cannot swap nonces. This defeats T3. |
| Drift bounds are local policy only | Each agent enforces its own `max_drift` from its local trust registry. It is never sent on the wire. An adversary cannot relax another agent's threshold. |
| Binary framing, not HTTP | Agents are not browsers. A length-prefixed binary frame is simpler, has no header ambiguity, and needs zero external dependencies beyond the noise crate. |
| Self-describing messages | Every frame declares its type and version so that future extensions don't break old parsers. |
| Mutual exchange in one round-trip | The common case (two agents swapping attestations) should complete in a single request→response after the handshake. |
| Chain is inline | A chained v2 attestation travels with its full ancestry. The receiver doesn't fetch missing links. |

### 10.3 Transport Layer: Noise NK Handshake

GOT/1 uses the [Noise Protocol Framework](http://noiseprotocol.org/) with the **NK** pattern:

- **N** — initiator is anonymous (no static key in handshake; identified later by signed envelope)
- **K** — responder's static public key is known in advance (from the trust registry)

This provides:
1. **Forward secrecy** — ephemeral Diffie-Hellman keys, so compromising a long-term key does not decrypt past sessions
2. **Server authentication** — the initiator knows it is talking to the real responder (defeats T5)
3. **Encryption** — all subsequent frames are encrypted with ChaCha20-Poly1305 (defeats T1)
4. **Integrity** — AEAD ciphertext is tamper-evident (defeats T2 at the transport layer)

After the Noise NK handshake completes, both sides have a pair of CipherState objects for bidirectional encrypted communication. All GOT/1 frames below are sent inside this encrypted channel.

```
  Agent A (initiator)                              Agent B (responder)
       |                                                |
       |-- TCP connect -------------------------------->|
       |                                                |
       |   ---- Noise NK Handshake ----                 |
       |                                                |
       |-- → e, es (ephemeral key + DH) -------------->|
       |<-- ← e, ee (responder ephemeral + DH) --------|
       |                                                |
       |   Encrypted channel established.               |
       |   All subsequent frames are ChaCha20-Poly1305. |
       |                                                |
       |-- [encrypted] GOT/1 EXCHANGE_REQ ------------>|
       |<-- [encrypted] GOT/1 EXCHANGE_RSP ------------|
       |                                                |
       |-- TCP close ---------------------------------->|
```

The responder's Noise static key is its Ed25519 key converted to X25519 (using the standard birational map, as `ed25519-dalek` and `x25519-dalek` support). This means agents do not need a separate keypair for transport — their existing attestation signing key doubles as their Noise identity.

### 10.4 Frame Format

Inside the encrypted channel, every GOT/1 message is a length-prefixed frame:

```
  Offset  Size     Field
  ------  ----     -----
  0       4        Magic: "GOT1" (0x474F5431)
  4       1        Message type (u8)
  5       4        Payload length L (u32 BE)
  9       L        Payload (type-dependent, see below)
```

Total frame size: 9 + L bytes.

There is no frame-level MAC or checksum. The Noise transport's AEAD (ChaCha20-Poly1305) already provides integrity and authentication for every encrypted message. Adding a second MAC would be redundant.

### 10.5 Message Types

| Type byte | Name | Direction | Purpose |
|---|---|---|---|
| `0x01` | `EXCHANGE_REQ` | Initiator → Responder | "Here is my signed envelope (attestation + chain). Send me yours." |
| `0x02` | `EXCHANGE_RSP` | Responder → Initiator | "Here is my signed envelope (attestation + chain). I accept/reject yours." |
| `0x03` | `VERIFY_REQ` | Initiator → Responder | "Verify this attestation and tell me the result." (one-way) |
| `0x04` | `VERIFY_RSP` | Responder → Initiator | "Verification result: valid/invalid/error + reason." |
| `0x05` | `CHAIN_REQ` | Initiator → Responder | "Send me your full attestation chain." |
| `0x06` | `CHAIN_RSP` | Responder → Initiator | "Here is my chain: [attest_0, ..., attest_n]." |
| `0xFF` | `ERROR` | Either direction | Protocol-level error. |

### 10.6 Signed Exchange Envelope

The critical security fix: every `EXCHANGE_REQ` and `EXCHANGE_RSP` wraps the attestation in a **signed envelope** that binds the attestation to this specific exchange with this specific peer.

```
  ExchangeEnvelope {
      nonce:              [u8; 32],   // random (req) or echoed (rsp)
      peer_agent_id:      [u8; 32],   // intended recipient's agent ID
      attestation_hash:   [u8; 32],   // SHA-256 of current attestation's
                                      //   serialise_for_signing() bytes
      chain_root_hash:    [u8; 32],   // SHA-256 of chain[0]'s
                                      //   serialise_for_signing() bytes
                                      //   (or zeroes if no chain)
      timestamp:          u64,        // Unix UTC seconds
      envelope_signature: [u8; 64],   // Ed25519 sign over all above fields
  }
```

The envelope signature covers: `nonce ‖ peer_agent_id ‖ attestation_hash ‖ chain_root_hash ‖ timestamp` (concatenated, fixed-width, no delimiters needed since all fields are fixed size).

**Why this matters:**

| Attack | Envelope field that blocks it |
|---|---|
| Replay old response with correct attestation sig | `nonce` is signed — cannot be swapped. `timestamp` allows freshness check. |
| Redirect attestation to a different peer | `peer_agent_id` is signed — attestation is bound to this specific recipient. |
| Swap the attestation payload mid-flight | `attestation_hash` is signed — any modification is detected. |
| Swap the chain | `chain_root_hash` is signed — chain anchor is bound. |
| Forge the envelope for another agent | Requires the sender's Ed25519 secret key — infeasible. |

### 10.7 Payload Schemas

#### EXCHANGE_REQ (0x01)

```
  Offset  Size     Field
  ------  ----     -----
  0       32       Sender agent ID (SHA-256 of sender's public key)
  32      192      Signed envelope (nonce + peer_id + attest_hash +
                     chain_root_hash + timestamp + signature)
  224     4        Chain length N (u32 BE), 0 = single attestation
  228     4        Attestation[0] length A0 (u32 BE)
  232     A0       Attestation[0] JSON (UTF-8, oldest in chain)
  ...              Attestation[1..N] (same length-prefixed pattern)
  ...     4        Current attestation length Ac (u32 BE)
  ...     Ac       Current attestation JSON
```

Note: `max_accepted_drift` is **not** on the wire. The receiver enforces its own threshold from its local trust registry. Including it in the wire format was a security flaw — the sender could lie about it, or a MITM could relax it.

#### EXCHANGE_RSP (0x02)

```
  Offset  Size     Field
  ------  ----     -----
  0       32       Responder agent ID
  32      192      Signed envelope (nonce echoed + peer_id + attest_hash +
                     chain_root_hash + timestamp + signature)
  224     1        Verdict: 0x01=accepted, 0x02=rejected, 0x03=error
  225     4        Chain length N (u32 BE)
  229     ...      Attestation chain (same format as EXCHANGE_REQ)
  ...     4        Current attestation length
  ...     var      Current attestation JSON
  ...     4        Reason length R (u32 BE), 0 if accepted
  ...     R        Reason string (UTF-8)
```

The verdict and reason are inside the encrypted Noise channel and further bound by the envelope signature (if the receiver re-verifies the envelope, any tampering is caught). 

#### ERROR (0xFF)

```
  Offset  Size     Field
  ------  ----     -----
  0       4        Error code (u32 BE)
  4       4        Message length M (u32 BE)
  8       M        Error message (UTF-8)
```

Error codes:

| Code | Meaning |
|---|---|
| 1 | Bad magic (not GOT1) |
| 2 | Unknown message type |
| 3 | Payload too large (exceeds implementation limit) |
| 4 | Noise handshake failed |
| 5 | Nonce mismatch in envelope (replay suspected) |
| 6 | Unknown agent ID (not in trust registry) |
| 7 | Envelope signature invalid |
| 8 | Attestation hash mismatch (envelope vs payload) |
| 9 | Timestamp too old (exceeds freshness window) |
| 10 | Internal error |

### 10.8 Trust Registry

A TOML configuration file mapping agent identities to public keys and policy. Drift thresholds are **local policy** — never sent on the wire.

```toml
[registry]
# Maximum attestation chain length we'll accept from any agent.
max_chain_length = 100
# Maximum age of an exchange envelope timestamp (seconds).
max_envelope_age_secs = 300

[[agents]]
id = "alice"
public_key = "a1b2c3...64 hex chars for 32-byte Ed25519 verifying key"
max_drift_accepted = 0.05
roles = ["producer", "verifier"]

[[agents]]
id = "bob"
public_key = "d4e5f6...64 hex chars"
max_drift_accepted = 0.05
roles = ["producer", "verifier"]
```

The `id` field is human-readable. The canonical agent ID on the wire is `SHA-256(public_key)` — 32 bytes.

### 10.9 Exchange Protocol Sequence

```
  Agent A (initiator)                              Agent B (responder)
       |                                                |
       |-- TCP connect to B's address ----------------->|
       |                                                |
       |   ---- Noise NK Handshake ----                 |
       |-- → e, es ---------------------------------->  |
       |<-- ← e, ee ---------------------------------- |
       |   (encrypted channel established)              |
       |                                                |
       |-- [encrypted] EXCHANGE_REQ:                    |
       |     agent_id_A = SHA-256(pk_A)                 |
       |     envelope:                                  |
       |       nonce = random 32 bytes                  |
       |       peer_agent_id = SHA-256(pk_B)            |
       |       attestation_hash = H(attest_A)           |
       |       chain_root_hash = H(chain_A[0])          |
       |       timestamp = now()                        |
       |       signature = sign(above, sk_A)            |
       |     chain = [attest_A_0, ..., attest_A_n]      |
       |     current = attest_A_current                 |
       |------------------------------------------------>|
       |                                                |
       |                         B receives frame:      |
       |                           (Noise decrypts +    |
       |                            verifies AEAD)      |
       |                           lookup agent_id_A    |
       |                             in trust registry  |
       |                           verify envelope sig  |
       |                             with pk_A          |
       |                           check envelope.      |
       |                             peer_agent_id      |
       |                             == own agent_id    |
       |                           check timestamp      |
       |                             within freshness   |
       |                             window             |
       |                           check attestation_   |
       |                             hash matches       |
       |                             SHA-256(serialise(  |
       |                               current))        |
       |                           verify attest_A sig  |
       |                             with pk_A          |
       |                           if v2: walk chain    |
       |                           check drift <=       |
       |                             LOCAL max_drift    |
       |                                                |
       |                         B decides:             |
       |                           accepted or rejected |
       |                                                |
       |                         B builds envelope_B:   |
       |                           nonce = echo A's     |
       |                           peer_agent_id =      |
       |                             SHA-256(pk_A)      |
       |                           attest_hash = H(B)   |
       |                           sign(above, sk_B)    |
       |                                                |
       |                         B sends EXCHANGE_RSP:  |
       |<-- [encrypted] ---------------------------------|
       |                                                |
       |  A receives frame:                             |
       |    (Noise decrypts)                            |
       |    lookup agent_id_B                           |
       |    verify envelope sig with pk_B               |
       |    check envelope.peer_agent_id == own id      |
       |    check nonce matches the one A sent          |
       |    check timestamp freshness                   |
       |    check attestation_hash matches payload      |
       |    verify attest_B sig with pk_B               |
       |    if v2: walk chain                           |
       |    check drift <= LOCAL max_drift              |
       |                                                |
       |  A decides:                                    |
       |    both accepted → cooperate                   |
       |    any rejected  → refuse                      |
       |                                                |
       |-- TCP close ---------------------------------->|
```

1 Noise handshake + 1 request + 1 response = 3 TCP round-trips total for a complete mutual attestation exchange.

### 10.10 `got-wire` Crate Design

New crate: `crates/got-wire/`

```rust
// --- Noise transport ---

/// Perform Noise NK handshake as initiator.
/// `responder_pk` is the Ed25519 public key (converted to X25519 internally).
pub fn noise_connect(
    stream: &mut TcpStream,
    responder_pk: &[u8; 32],
) -> Result<NoiseSession, WireError>;

/// Perform Noise NK handshake as responder.
/// `own_sk` is the Ed25519 secret key (converted to X25519 internally).
pub fn noise_accept(
    stream: &mut TcpStream,
    own_sk: &SigningKey,
) -> Result<NoiseSession, WireError>;

/// Encrypted bidirectional channel after handshake.
pub struct NoiseSession {
    // internal CipherState pair
}

impl NoiseSession {
    pub fn send_frame(&mut self, frame: &Frame) -> Result<(), WireError>;
    pub fn recv_frame(&mut self) -> Result<Frame, WireError>;
}

// --- Frame types ---

pub struct Frame {
    pub message_type: MessageType,
    pub payload: Vec<u8>,
}

pub enum MessageType {
    ExchangeReq = 0x01,
    ExchangeRsp = 0x02,
    VerifyReq   = 0x03,
    VerifyRsp   = 0x04,
    ChainReq    = 0x05,
    ChainRsp    = 0x06,
    Error       = 0xFF,
}

// --- Signed envelope ---

pub struct ExchangeEnvelope {
    pub nonce: [u8; 32],
    pub peer_agent_id: [u8; 32],
    pub attestation_hash: [u8; 32],
    pub chain_root_hash: [u8; 32],
    pub timestamp: u64,
    pub signature: [u8; 64],
}

impl ExchangeEnvelope {
    /// Build and sign an envelope.
    pub fn create(
        nonce: [u8; 32],
        peer_agent_id: [u8; 32],
        attestation: &GeometricAttestation,
        chain_anchor: Option<&GeometricAttestation>,
        signing_key: &SigningKey,
    ) -> Self;

    /// Verify envelope signature and check all bindings.
    pub fn verify(
        &self,
        expected_peer_id: &[u8; 32],  // must match peer_agent_id
        expected_nonce: Option<&[u8; 32]>,  // for responses
        attestation: &GeometricAttestation,  // hash must match
        signer_pk: &VerifyingKey,
        max_age_secs: u64,
    ) -> Result<(), WireError>;

    /// Serialise the signed-over fields (for signing/verification).
    pub fn signable_bytes(&self) -> [u8; 136];  // 32+32+32+32+8
}

// --- Payload types ---

pub struct ExchangeRequest {
    pub agent_id: [u8; 32],
    pub envelope: ExchangeEnvelope,
    pub chain: Vec<GeometricAttestation>,
    pub current: GeometricAttestation,
}

pub struct ExchangeResponse {
    pub agent_id: [u8; 32],
    pub envelope: ExchangeEnvelope,
    pub verdict: Verdict,
    pub chain: Vec<GeometricAttestation>,
    pub current: GeometricAttestation,
    pub reason: String,
}

pub enum Verdict { Accepted = 0x01, Rejected = 0x02, Error = 0x03 }

// --- Trust registry ---

pub struct TrustRegistry {
    pub agents: HashMap<[u8; 32], AgentEntry>,
    pub max_chain_length: usize,
    pub max_envelope_age_secs: u64,
}

pub struct AgentEntry {
    pub name: String,
    pub public_key: [u8; 32],
    pub max_drift_accepted: f32,  // LOCAL policy, never sent on wire
    pub roles: Vec<String>,
}

impl TrustRegistry {
    pub fn load(path: &Path) -> Result<Self, WireError>;
    pub fn lookup(&self, agent_id: &[u8; 32]) -> Option<&AgentEntry>;
    pub fn agent_id(public_key: &[u8; 32]) -> [u8; 32];  // SHA-256(pk)
}

// --- Chain verification ---

/// Verify a chain of attestations: signatures, linkage, drift bounds.
pub fn verify_chain(
    chain: &[GeometricAttestation],
    current: &GeometricAttestation,
    signer_pk: &[u8; 32],
    max_drift: f32,  // from LOCAL registry, not from wire
) -> Result<ChainVerdict, WireError>;

// --- High-level transport ---

/// Listen for incoming GOT/1 connections on the given address.
pub fn listen(
    addr: SocketAddr,
    own_key: &SigningKey,
    registry: &TrustRegistry,
    own_attestation: &GeometricAttestation,
    own_chain: &[GeometricAttestation],
) -> Result<(), WireError>;

/// Connect to a peer and perform a full attestation exchange.
pub fn exchange(
    addr: SocketAddr,
    peer_pk: &[u8; 32],
    own_key: &SigningKey,
    registry: &TrustRegistry,
    own_attestation: &GeometricAttestation,
    own_chain: &[GeometricAttestation],
) -> Result<ExchangeResult, WireError>;

pub struct ExchangeResult {
    pub peer_verdict: Verdict,      // what the peer said about us
    pub our_verdict: Verdict,       // what we decided about the peer
    pub peer_attestation: GeometricAttestation,
    pub peer_chain: Vec<GeometricAttestation>,
    pub reason: String,
}
```

Dependencies: `got-core`, `got-attest`, `sha2`, `ed25519-dalek`, `x25519-dalek`, `snow` (Noise protocol implementation), `serde`, `serde_json`, `thiserror`, `toml`.

### 10.11 Chain Walk Algorithm

```
fn verify_chain(chain, current, signer_pk, max_drift) -> Result<ChainVerdict>:
    all = chain ++ [current]

    // 1. Anchor check
    if all[0].parent_attestation_hash.is_some():
        return Err(BrokenChain("first attestation must have no parent"))

    // 2. Walk each link
    for i in 0..all.len():
        // Signature check
        if !verify(all[i], signer_pk)?:
            return Err(InvalidSignature(i))

        // Linkage check
        if i > 0:
            expected = attestation_hash(&all[i-1])
            if all[i].parent_attestation_hash != Some(expected):
                return Err(BrokenChain(i))

        // Drift check (using LOCAL max_drift, not from wire)
        if let Some(drift) = all[i].geometry_drift:
            if drift > max_drift:
                return Err(DriftExceeded { index: i, drift, max_drift })

    return Ok(ChainVerdict::Valid { length: all.len() })
```

### 10.12 Security Analysis

How each threat from §10.1 is addressed:

| Threat | Mitigation | Mechanism |
|---|---|---|
| T1: Eavesdropping | **Defeated.** All frames encrypted. | Noise NK → ChaCha20-Poly1305 |
| T2: Tampering | **Defeated.** AEAD detects modification. Envelope signature binds attestation to exchange context. | Noise AEAD + ExchangeEnvelope.signature |
| T3: Replay | **Defeated.** Nonce is inside the signed envelope. Timestamp enforces freshness window. | ExchangeEnvelope.nonce + timestamp + signature |
| T4: Identity spoofing | **Defeated.** Envelope signature over peer_agent_id binds identity. Noise NK authenticates responder's static key. | ExchangeEnvelope.peer_agent_id + Noise NK |
| T5: MITM | **Defeated.** Noise NK provides server authentication. Envelope binding prevents relay attacks. | Noise handshake + envelope channel binding |
| T6: DoS | **Mitigated.** Payload length limit (configurable). Connection rate limiting (implementation-level). Unknown agent IDs rejected before expensive operations. | Frame length check + trust registry lookup first |

**Residual risks:**

- **Key compromise.** If an agent's Ed25519 secret key is stolen, the attacker can impersonate that agent until the key is revoked in all trust registries. There is no in-protocol revocation mechanism.
- **Initiator anonymity.** Noise NK does not authenticate the initiator during the handshake — only after the envelope signature is verified. A malicious initiator can complete the handshake and learn that the responder is alive before being identified. This is a minor information leak.
- **Timing side channels.** The protocol does not attempt constant-time processing of frames. An observer of frame timing could infer message sizes (despite encryption, since length-prefixed framing leaks payload size).
- **Trust registry integrity.** If an attacker can modify an agent's trust registry TOML (e.g. by compromising the filesystem), they can inject trusted keys or relax drift thresholds. The registry must be protected at the OS level.

### 10.13 Behaviour Matrix

| Scenario | Protocol behaviour |
|---|---|
| Both agents frozen (v1) | Single attestation each, no chain. chain_length=0. |
| One agent has updated (v2) | That agent sends full chain. Peer walks it, checks drift from LOCAL registry. |
| Drift exceeds LOCAL max | Peer sends `EXCHANGE_RSP` with verdict=Rejected, reason="drift 0.072 > local max 0.05". |
| Unknown agent ID | `ERROR` frame, code 6. Connection closed after Noise handshake. |
| Nonce mismatch in envelope | Initiator verifies envelope.nonce == sent nonce. Mismatch → reject (code 5). |
| Envelope signature invalid | Reject immediately (code 7). Possible MITM or forgery attempt. |
| Attestation hash vs payload mismatch | Reject (code 8). Payload was modified after envelope was signed. |
| Timestamp outside freshness window | Reject (code 9). Possible replay of old envelope. |
| Noise handshake fails | `ERROR` code 4. TCP close. Possible wrong responder key. |
| Corrupted ciphertext | Noise AEAD tag mismatch. Connection aborted. |
| Chain too long | Exceeds `max_chain_length` in registry config. `ERROR` frame, code 3. |

### 10.14 What GOT/1 Does Not Cover

- **Discovery** — agents must know each other's TCP addresses. Service discovery (mDNS, registry REST API, etc.) is out of scope.
- **Session persistence** — every exchange is a single TCP connection. No connection pooling, no keepalive, no session resumption.
- **Ordering** — chain order is sender's responsibility. No consensus protocol for multi-agent chain agreement.
- **Partial chain delivery** — if the chain is too large, there is no pagination. The sender must include the full chain or the receiver rejects it.
- **Key rotation** — if an agent rotates its signing key, peers must update their trust registries out-of-band. No in-protocol key rotation or revocation mechanism.
- **Multi-party exchange** — GOT/1 is pairwise. An aggregator topology (see `architecture-agent-protocol.md`) requires multiple pairwise exchanges.
- **Padding** — frame lengths are visible to traffic analysts despite encryption. Length padding is not implemented.

### 10.15 Phase 10 Tests

- Noise NK handshake succeeds between two in-process agents
- Noise NK handshake fails with wrong responder key
- Frame encode/decode round-trip (all message types, inside Noise session)
- `ExchangeEnvelope::create` + `verify` round-trip
- Envelope with wrong nonce → rejected
- Envelope with wrong peer_agent_id → rejected
- Envelope with tampered attestation_hash → rejected
- Envelope with expired timestamp → rejected
- Envelope with forged signature → rejected
- `EXCHANGE_REQ` → `EXCHANGE_RSP` round-trip (v1, accepted)
- `EXCHANGE_REQ` → `EXCHANGE_RSP` round-trip (v2 with chain, accepted)
- Exchange where drift exceeds local max → verdict=rejected
- Unknown agent ID → ERROR code 6
- Chain walk: valid chain passes
- Chain walk: broken link → rejected
- Chain walk: drift exceeds bound → rejected
- Chain walk: missing anchor → rejected
- Trust registry TOML load/lookup
- Trust registry: `agent_id()` matches `SHA-256(public_key)`
- Two-agent TCP integration: handshake + exchange + mutual verify
- Replay attempt: reuse old envelope → nonce/timestamp rejection

### 10.16 Acceptance Criteria

- [ ] Noise NK handshake works between two agents (no plaintext on wire)
- [ ] Exchange envelope is signed and all bindings are verified
- [ ] Nonce replay protection works (signed nonce, timestamp freshness)
- [ ] Drift thresholds are enforced from local trust registry only (never from wire)
- [ ] Frame codec round-trips all message types without data loss
- [ ] Chain verification correctly rejects broken/stale chains
- [ ] Trust registry loads from TOML and resolves agent IDs
- [ ] Wrong Noise key → handshake failure
- [ ] Envelope signature forgery → rejection
- [ ] Integration test: two TCP agents complete full exchange with encryption

---

## Phase 11 — Hardware-Isolated Measurement

Phases 8–9 establish causal interventions and inline measurement, but both run in the same trust domain as the model. A sufficiently capable model that controls its own process could, in principle, detect and manipulate the measurement code. This phase moves activation capture to hardware that the model cannot influence.

### 11.1 The Problem

Software-based hooks (Phase 10) execute in the same address space as the model. Even if the measurement sidecar runs in a separate process, the model's forward pass produces the activations that the sidecar reads. If the model can alter what it writes to the activation buffer when it detects measurement, the hook reads sanitised data.

Hardware-isolated measurement intercepts activations at the hardware level — below the model's software stack — so the model has no mechanism to alter what is captured.

### 11.2 Architecture

```
  ┌───────────────────────────────────────────────┐
  │  GPU / Accelerator                            │
  │                                               │
  │  Model forward pass writes activations        │
  │  to layer output buffers (VRAM)               │
  │                                               │
  │  ┌───────────────────────────────────────┐    │
  │  │  DMA / Hardware Tap                   │    │
  │  │  Copies activation buffers to         │    │
  │  │  isolated measurement hardware        │    │
  │  │  (separate memory region, separate    │    │
  │  │   trust domain)                       │    │
  │  └───────────────────────────────────────┘    │
  └───────────────────────────────────────────────┘
           │
           ▼
  ┌───────────────────────────────────────────────┐
  │  Measurement TEE / Secure Enclave             │
  │                                               │
  │  Receives raw activations via DMA             │
  │  Runs causal probes (Phase 9 intervention)    │
  │  Produces signed attestation                  │
  │  Signing key never leaves enclave             │
  │                                               │
  └───────────────────────────────────────────────┘
```

### 11.3 Trust Boundary

| Component | Trust domain | Can model influence? |
|---|---|---|
| Model weights & code | Model process | Yes (by definition) |
| Activation buffers (VRAM) | Model writes, hardware reads | Model writes, but hardware captures what was actually written |
| DMA tap | Hardware/firmware | No — below model's software stack |
| Measurement enclave | Separate TEE | No — isolated execution environment |
| Signing key | Inside TEE | No — never exported |

The critical property: **the model writes activations to VRAM as part of its normal computation. The hardware tap captures exactly what was written, before the model has any opportunity to overwrite or sanitise.**

### 11.4 Pairing with Causal Interventions

Hardware capture alone guarantees authentic activations but doesn't prove the probes measure real mechanisms. Causal intervention (Phase 8) alone proves mechanism but runs in software the model might manipulate. Together:

1. Hardware captures real activations → feeds to measurement enclave
2. Measurement enclave runs causal intervention (perturb + measure output shift)
3. Perturbation is applied by the enclave, not by the model process
4. The enclave observes whether the output changes — using activations captured by hardware

This is the strongest combination: the model cannot fake the activations (hardware capture), and the probes are validated against real mechanisms (causal intervention).

### 11.5 Implementation Approaches

| Approach | Hardware Required | Maturity |
|---|---|---|
| GPU DMA snoop | Custom FPGA or modified GPU driver | Research prototype |
| NVIDIA Confidential Computing (H100 TEE) | H100 GPU + driver support | Early production |
| CPU TEE (Intel SGX / AMD SEV) | Commodity server CPU | Production (CPU-side only) |
| Custom accelerator with built-in taps | ASIC design | Far future |

For the PoC, the most practical path is **CPU TEE with activation copy-out**: the model runs on GPU, activations are copied to CPU memory inside a TEE enclave via DMA, and probing runs inside the enclave. This is slower but uses commodity hardware.

### 11.6 Enclave Interface (`got-enclave`)

```rust
/// Trait for a hardware-isolated measurement environment.
pub trait MeasurementEnclave {
    /// Receive activations from hardware capture.
    /// `layer` is the layer index, `h` is the raw activation vector.
    fn receive_activations(&mut self, layer: usize, h: &[f32]);

    /// Run causal intervention inside the enclave.
    /// Returns CausalScore for each sampled probe.
    fn run_causal_check(
        &self,
        probes: &[ProbeVector],
        delta: f32,
    ) -> Vec<CausalScore>;

    /// Produce a signed attestation from accumulated measurements.
    /// The signing key never leaves the enclave.
    fn attest(&self) -> GeometricAttestation;
}
```

### 11.7 What This Phase Does Not Solve

- **Who provisions the enclave?** The entity that controls enclave deployment controls which probes are loaded. If the model operator controls the enclave, they could load favourable probes.
- **Side channels.** TEEs are susceptible to power analysis, timing attacks, and speculative execution attacks. The enclave protects against software-level manipulation, not physical attacks.
- **Performance.** Copying activations out of GPU VRAM for every forward pass adds latency. Batching and sampling strategies can mitigate this.
- **Firmware trust.** The DMA tap relies on firmware or hardware that itself must be trusted. This pushes the trust boundary one level down, but doesn't eliminate it.

### 11.8 Phase 11 Tests

- Mock enclave receives activations and produces valid attestation
- Mock enclave runs causal intervention and returns scores
- Attestation signed inside enclave is verifiable outside
- Integration: mock hardware capture → enclave → attestation → verify chain
- Enclave rejects tampered activation data (hash mismatch)

### 11.9 Acceptance Criteria

- [ ] `MeasurementEnclave` trait compiles and mock implementation works
- [ ] Mock enclave produces valid, signed attestations
- [ ] Causal intervention runs inside mock enclave
- [ ] Attestation chain from enclave integrates with Phase 10 wire protocol
- [ ] End-to-end: capture → enclave probe → attest → transmit → verify

---

## Phase 12 — Persistent Attestation Store & Audit Trail

Phases 1–11 produce, sign, chain, transport, and hardware-verify attestations, but every attestation exists only in memory for the duration of a single process. A governance framework needs a persistent, queryable record of every attestation ever produced — an auditable history that an independent body can inspect after the fact. This phase adds the storage layer.

### 12.1 The Problem

Without persistence, attestation chains are ephemeral. A model operator could produce a failing attestation, discard it, retune, and produce a passing one — with no evidence the first ever existed. A persistent store with append-only semantics makes this detectable: gaps in the chain, missing sequence numbers, or timestamp discontinuities are all audit signals.

### 12.2 Architecture

```
  got-enclave / got-wire
        │
        │  GeometricAttestation (signed)
        ▼
  ┌─────────────────────────────────────────┐
  │  got-store                              │
  │                                         │
  │  AttestationStore trait                  │
  │    ├── append(attestation) → StoreId    │
  │    ├── get(id) → Attestation            │
  │    ├── chain(model_id) → Vec<Att>       │
  │    ├── query(filter) → Vec<Att>         │
  │    └── audit(model_id) → AuditReport    │
  │                                         │
  │  Backends:                              │
  │    ├── MemoryStore  (testing)           │
  │    └── FileStore    (PoC persistence)   │
  └─────────────────────────────────────────┘
```

### 12.3 Store Semantics

| Property | Guarantee |
|---|---|
| Append-only | Once stored, an attestation cannot be modified or deleted. |
| Chain-aware | The store validates `parent_hash` links on insertion. Orphaned attestations (parent not in store) are flagged. |
| Signature-verified | Every attestation is signature-verified before storage. Invalid signatures are rejected. |
| Queryable | Attestations can be retrieved by model ID, signer public key, time range, schema version, or causal flag. |
| Deterministic IDs | Store IDs are derived from the attestation's content hash (SHA-256 of serialised form), making them reproducible. |

### 12.4 Query Filter

```rust
pub struct StoreFilter {
    pub model_id: Option<String>,
    pub signer: Option<VerifyingKey>,
    pub after: Option<u64>,       // timestamp lower bound
    pub before: Option<u64>,      // timestamp upper bound
    pub schema_version: Option<u16>,
    pub causal_flag: Option<bool>,
}
```

Filters compose conjunctively: all specified fields must match.

### 12.5 Audit Report

The `audit()` method produces a structured summary of a model's attestation history:

```rust
pub struct AuditReport {
    pub model_id: String,
    pub total_attestations: usize,
    pub chain_length: usize,
    pub chain_valid: bool,           // all parent_hash links verified
    pub first_timestamp: Option<u64>,
    pub last_timestamp: Option<u64>,
    pub schema_versions_seen: Vec<u16>,
    pub drift_summary: DriftSummary,
    pub causal_summary: CausalSummary,
    pub signers: Vec<[u8; 32]>,      // unique signer key hashes
}

pub struct DriftSummary {
    pub readings_with_drift: usize,
    pub max_drift: Option<f64>,
    pub mean_drift: Option<f64>,
}

pub struct CausalSummary {
    pub attestations_with_causal: usize,
    pub causal_pass_count: usize,
    pub causal_fail_count: usize,
    pub mean_consistency: Option<f64>,
}
```

This gives an auditor a single-call summary: how many attestations, is the chain intact, has drift been stable, are causal checks passing.

### 12.6 FileStore Layout

```
store_root/
├── index.json              # model_id → list of content hashes
├── attestations/
│   ├── <sha256_hex>.json   # one file per attestation
│   └── ...
└── audit_cache/            # optional cached audit reports
```

Each attestation is stored as a JSON file named by its content hash. The index maps model IDs to their attestation chains (ordered by timestamp). This is intentionally simple — a production system would use a database.

### 12.7 Integration with Wire Protocol

When a wire protocol exchange completes (Phase 10), both the local and peer attestations can be stored:

```rust
let (result, verdict) = perform_exchange(...)?;
store.append(&result.peer_attestation, &peer_verifying_key)?;
```

This creates a record of every attestation received from peers, building a multi-agent audit trail.

### 12.8 Phase 12 Tests

- Append and retrieve single attestation
- Chain validation accepts valid chain, rejects broken parent_hash
- Signature verification rejects tampered attestation on insert
- Query by model_id returns only matching attestations
- Query by time range filters correctly
- Query by causal_flag filters correctly
- Audit report reflects correct chain length, drift, and causal summaries
- FileStore persists and reloads across instantiations
- Duplicate append (same content hash) is idempotent
- MemoryStore and FileStore produce identical results for same operations
- Integration: enclave → wire → store → audit end-to-end pipeline

### 12.9 Acceptance Criteria

- [ ] `AttestationStore` trait compiles with `MemoryStore` and `FileStore` backends
- [ ] Append-only semantics enforced (no mutation or deletion)
- [ ] Chain validation catches broken parent links
- [ ] Query filters work conjunctively across all fields
- [ ] `AuditReport` correctly summarises chain integrity, drift, and causal status
- [ ] `FileStore` survives process restart (write, drop, re-open, read)
- [ ] Integration test: produce attestation chain → store → query → audit

---

## Phase 13 — Adversarial Hardening

Phases 1–12 build a complete measurement pipeline: train probes, sign attestations, chain them, transport them, capture activations in hardware, and persist the audit trail. But a security audit of the PoC reveals three structural weaknesses that an adversary could exploit even if every phase works as designed. This phase hardens the protocol against those attacks.

### 13.1 The Problems

Three flaws, in descending severity:

**Flaw 1 — Model callback supplied by untrusted agent (CRITICAL).** `causal_check()` accepts a `model_fn: &dyn Fn(&[f32]) -> Vec<f32>` closure from the caller. The Phase 8 causal intervention — the keystone of the entire system — perturbs activations and observes the output shift. But if the adversary controls `model_fn`, they can return whatever outputs make the probe appear causal. The enclave trait `MeasurementEnclave::run_causal_check()` also takes `model_fn` as a parameter, and `enclave_pipeline()` passes it through. An adversary who supplies a fake forward pass can make any probe appear causally valid, defeating the keystone entirely.

**Flaw 2 — Frobenius drift hides directional gaming (SERIOUS).** `drift_from()` returns a single scalar: ‖Φ_new − Φ_ref‖_F / ‖Φ_ref‖_F. An adversary can surgically modify the geometry in probe-relevant directions while keeping the global Frobenius norm small. The unused dimensions absorb the perturbation budget. The verifier sees drift within bounds, but the probe-relevant subspace has shifted enough to invalidate the measurement.

**Flaw 3 — No chain completeness guarantee (MODERATE).** An agent can produce an unfavourable attestation, discard it, and re-chain from the prior attestation. The verifier sees a valid chain with correct parent hashes and signatures, but has no way to detect the gap. The divergence-flagged attestation vanishes from the record.

### 13.2 Fix: ModelHandle Trait (Flaw 1)

**Principle:** The model forward pass is enclave-internal, not caller-supplied.

#### 13.2.1 `ModelHandle` Trait

Define in `got-probe/src/intervention.rs`:

```rust
/// Encapsulates a model's forward pass from a probed layer to output.
///
/// In production, the implementation lives inside the TEE and is loaded
/// from a verified model shard. The enclave owns the handle; the caller
/// never supplies it per-call.
pub trait ModelHandle {
    fn forward(&self, h: &[f32]) -> Vec<f32>;
}

/// PoC convenience wrapper: wraps a closure as a ModelHandle.
/// In production, this is replaced by a TEE-internal model shard loader.
pub struct ClosureModelHandle<F: Fn(&[f32]) -> Vec<f32>> {
    f: F,
}

impl<F: Fn(&[f32]) -> Vec<f32>> ClosureModelHandle<F> {
    pub fn new(f: F) -> Self { Self { f } }
}

impl<F: Fn(&[f32]) -> Vec<f32>> ModelHandle for ClosureModelHandle<F> {
    fn forward(&self, h: &[f32]) -> Vec<f32> { (self.f)(h) }
}
```

#### 13.2.2 API Changes

| Location | Before | After |
|---|---|---|
| `causal_check()` | `model_fn: &dyn Fn(&[f32]) -> Vec<f32>` | `model: &dyn ModelHandle` |
| `causal_check_multi_layer()` | `model_fn_by_layer: &dyn Fn(usize, &[f32]) -> Vec<f32>` | `model: &dyn ModelHandle` with layer routing internal |
| `MeasurementEnclave::run_causal_check()` | `model_fn` parameter | Drop parameter; enclave uses internal handle |
| `MockEnclave::new()` | No model parameter | `model: Box<dyn ModelHandle>` provisioned at construction |
| `enclave_pipeline()` | `model_fn` parameter | Drop parameter; enclave already owns model |
| `MeasurementSidecar::ingest()` | `model_fn: Option<&dyn Fn(…)>` | `model: Option<&dyn ModelHandle>` |

#### 13.2.3 Why This Works

The API makes model access enclave-internal. The model is provisioned into the enclave at construction, not handed in per-call by the untrusted agent. In the PoC, `ClosureModelHandle` is functionally equivalent to the current closure, but the ownership boundary is architecturally correct: production TEE replaces `ClosureModelHandle` with a `TeeModelShard` loaded into enclave memory from a verified image.

**PoC limitation (documented):** Whoever constructs `MockEnclave` still supplies the handle — same trust boundary as before in a dev container. But the API is correct for production, and the ownership semantics are explicit.

### 13.3 Fix: Per-Probe Directional Drift (Flaw 2)

**Principle:** Drift is measured along each probe direction, not just globally.

#### 13.3.1 `directional_drift()` in `got-core/src/geometry.rs`

Add to `CausalGeometry`:

```rust
/// Drift along a specific probe direction.
///
/// Computes |wᵀ(Φ_new − Φ_ref)w| / |wᵀΦ_ref w|.
/// This measures how much the geometry has changed specifically
/// in the direction the probe measures, not just globally.
pub fn directional_drift(
    &self,
    reference: &CausalGeometry,
    direction: &[f32],
) -> Result<f32, GeometryError> {
    // wᵀΦw via the Gram matrix
    let quad_new = self.quadratic_form(direction)?;
    let quad_ref = reference.quadratic_form(direction)?;
    if quad_ref.abs() < f32::EPSILON {
        return Ok(if (quad_new - quad_ref).abs() < f32::EPSILON { 0.0 } else { f32::INFINITY });
    }
    Ok((quad_new - quad_ref).abs() / quad_ref.abs())
}

/// Compute wᵀΦw for a direction vector w.
fn quadratic_form(&self, w: &[f32]) -> Result<f32, GeometryError> {
    self.check_vec(w, "direction")?;
    let n = self.hidden_dim;
    let mut result = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            result += w[i] * self.gram[i * n + j] * w[j];
        }
    }
    Ok(result)
}
```

#### 13.3.2 Integration Points

| Location | Change |
|---|---|
| `read_probe_checked()` | After global Frobenius check, call `directional_drift()` with the probe's weight vector. Reject if directional drift exceeds `ProbeSet::max_directional_drift`. |
| `ProbeSet` struct | Add `max_directional_drift: Option<f32>` field (backward-compatible default `None`). |
| `GeometricAttestation` | Add `directional_drifts: Vec<DirectionalDrift>` field (serde-skippable when empty). |
| `DirectionalDrift` | New struct: `{ probe_name: String, drift: f32 }`. |
| `serialise_for_signing()` | Include directional drifts — they are covered by Ed25519 signature. |
| `verify_chain()` | Check per-direction drift bounds if present. |

#### 13.3.3 Why This Works

The adversary can no longer hide probe-direction changes inside a favourable global norm. Each probe direction is individually bounded. To game the system, the adversary would need to keep drift small in every direction the probes actually measure — which is exactly what "the geometry hasn't changed" means.

### 13.4 Fix: Signed Monotonic Sequence Number (Flaw 3)

**Principle:** Every attestation carries a signed, monotonically increasing sequence number that the enclave controls.

#### 13.4.1 Schema Change

Add to `GeometricAttestation`:

```rust
/// Monotonic sequence number assigned by the enclave.
/// The enclave increments this on every attestation; the counter
/// never resets. Gaps in the sequence indicate omitted attestations.
#[serde(default)]
pub sequence_number: u64,
```

#### 13.4.2 Enclave Changes

`MockEnclave` gets a `next_sequence: u64` field (starts at 0). Each call to `attest()` or `attest_with_causal()` embeds the current value in the attestation and increments the counter. The counter is never reset — `reset()` clears frames but not the sequence.

In production, the counter is backed by a hardware monotonic counter (SGX `sgx_create_monotonic_counter`, SEV via vTPM). Destroying and re-creating the enclave destroys the counter — which is exactly the right threat model.

#### 13.4.3 Verification Changes

`serialise_for_signing()` includes the sequence number, so it is covered by the Ed25519 signature and cannot be tampered with.

`verify_chain()` checks `all[i].sequence_number == all[i-1].sequence_number + 1` for every link after the anchor. A gap means an attestation was produced and discarded — the chain is rejected.

#### 13.4.4 Why This Works

The sequence number is signed, so the agent cannot forge it. The enclave counter only increments, so the agent cannot reset it. If the verifier sees sequence [0, 1, 3], it knows attestation #2 was produced and omitted. The agent cannot suppress unfavourable results without leaving detectable evidence.

### 13.5 Implementation Order

| Priority | Flaw | Scope | Rationale |
|---|---|---|---|
| 1st | #3 Sequence number | Small, additive | No API breaks. Clean additive change to struct + enclave + verifier. |
| 2nd | #2 Directional drift | Medium, new geometry function + attestation field | Additive. New method on `CausalGeometry`, new field on attestation. |
| 3rd | #1 ModelHandle trait | Large, trait + all callsites + all tests | Refactors the `MeasurementEnclave` trait and all causal intervention callsites. Goes last to avoid merge conflicts with #3 and #2. |

### 13.6 Phase 13 Tests

**Sequence number:**
- Two consecutive attestations have sequence numbers 0, 1
- `verify_chain()` rejects chain with gap (0, 2)
- `verify_chain()` rejects chain with duplicate sequence (0, 0)
- `reset()` does not reset sequence counter
- Sequence number is included in signed payload (tampering detected)

**Directional drift:**
- `directional_drift()` returns 0.0 for identical geometries
- `directional_drift()` detects change along probe direction even when global Frobenius drift is small
- `read_probe_checked()` rejects when directional drift exceeds bound
- `read_probe_checked()` passes when directional drift is within bound
- Directional drifts are included in signed attestation payload

**ModelHandle trait:**
- `ClosureModelHandle` wraps a closure and produces same results as direct call
- `MockEnclave` constructed with `ModelHandle` runs causal check without external `model_fn`
- `run_causal_check()` uses enclave-internal model (no parameter)
- `enclave_pipeline()` works without `model_fn` parameter
- Existing causal intervention tests pass with `ClosureModelHandle` wrapper

### 13.7 Acceptance Criteria

- [ ] `sequence_number` field added to `GeometricAttestation`, included in `serialise_for_signing()`
- [ ] `MockEnclave` assigns monotonic sequence numbers; `reset()` preserves counter
- [ ] `verify_chain()` rejects gaps and duplicates in sequence
- [ ] `directional_drift()` computes probe-direction-specific drift
- [ ] `read_probe_checked()` enforces directional drift bound when configured
- [ ] `DirectionalDrift` records included in attestation and covered by signature
- [ ] `ModelHandle` trait defined; `ClosureModelHandle` PoC wrapper implemented
- [ ] `MeasurementEnclave::run_causal_check()` no longer accepts `model_fn` parameter
- [ ] `MockEnclave` owns model handle; `enclave_pipeline()` drops `model_fn` parameter
- [ ] All existing tests updated and passing
- [ ] New adversarial-scenario tests for each flaw

---

## Critical Analysis: Are The Probes Correct?

The entire value of this system depends on one question: **do the probes measure what they claim to measure?**

Every phase from 1 to 11 is infrastructure to make probe readings *reproducible*, *signed*, *chained*, *transportable*, and *tamper-evident*. But if the probe for "honesty" actually measures "token frequency of words commonly associated with honesty in the training corpus," then the entire system is a precisely calibrated instrument pointed at the wrong thing.

The defence is layered:

| Layer | Phase | What it guarantees |
|---|---|---|
| Statistical validity | 1–6 | Probes produce deterministic, reproducible readings under the causal inner product. Divergence flags indicate when the reading is unreliable. |
| Temporal continuity | 7 | Geometry drift detection ensures probes are not applied to a model that has changed beyond their validity window. Chaining makes drift auditable. |
| Causal validity | 8 | **Keystone.** Causal interventions prove that perturbing activations in the probe direction changes the model's output. The probe reads a real mechanism, not a surface correlation. |
| Coverage | 9 | Inline measurement ensures every inference is monitored, not just spot-checks. The model cannot behave differently when unobserved. |
| Transport integrity | 10 | Encrypted, signed wire protocol ensures attestations are not tampered with in transit between agents. |
| Hardware authenticity | 11 | Hardware-isolated capture ensures the activations being probed are real, not sanitised by the model. |
| Audit persistence | 12 | Append-only attestation store with chain validation ensures no attestation can be silently discarded, and an auditor can inspect the full history after the fact. |
| Adversarial hardening | 13 | Model callback is enclave-owned (not caller-supplied), drift is measured per-probe-direction (not just globally), and signed monotonic sequence numbers prevent attestation omission. |

**Without Phase 8, all other layers secure a measurement that might be meaningless.**

This is why Phase 8 is the keystone. It is also the hardest phase: causal intervention requires a `model_fn` callback that exposes the model's forward pass, perturbation magnitudes that are empirically valid, and a consistency threshold that separates real mechanisms from noise. Getting this right is more important than getting the wire protocol encrypted or the hardware capture tamper-proof.

---

## What This PoC Proves and What It Does Not

### What it proves

- The causal inner product ⟨w, h⟩_c = wᵀΦh is computable from the unembedding matrix
- Linear probes trained under this metric produce deterministic readings
- Those readings can be assembled into a cryptographically signed attestation
- The attestation is independently reproducible: same weights + same input + same probes = identical output
- The format is self-describing and version-tagged for forward compatibility
- Geometry drift is measurable and boundable, making self-learning models auditable
- Attestation chains create a tamper-evident history of model evolution
- Causal interventions can distinguish probes that measure real mechanisms from probes that exploit surface correlations
- Inline measurement can monitor every inference, not just periodic audits
- A purpose-built wire protocol (GOT/1) can transport attestations between agents with encryption (Noise NK), signed exchange envelopes, replay protection, and chain verification in a single round-trip
- Hardware-isolated measurement can capture activations below the model's software stack
- Persistent, append-only attestation storage with chain validation and audit reporting enables after-the-fact inspection of a model's entire measurement history
- Enclave-owned model handles prevent an adversary from faking the causal forward pass
- Per-probe directional drift detects surgical geometry changes that global Frobenius norm misses
- Signed monotonic sequence numbers make attestation omission detectable

### What it does not prove

- That the probe readings **mean** anything about AI values (this requires causal validation with real models, not just the synthetic tests in the PoC)
- That the corpus used to train the probes is representative, fair, or legitimate
- That the coverage flags reliably indicate when the probes are out of distribution
- That the confidence values are calibrated (they are not, in the PoC)
- That causal intervention with synthetic `model_fn` transfers to real model forward passes
- That perturbation magnitude δ is ecologically valid for real inputs
- That hardware-isolated capture is feasible at production inference latencies
- That any institution is prepared to take responsibility for interpreting the output

### The gap

The geometry is a measurement instrument. Like any instrument, it reports what it is pointed at. Who decides what it is pointed at — which value dimensions are probed, which corpus defines the labels, what threshold separates "reliable" from "unreliable," and who adjudicates disputes about coverage — is a governance question.

Phases 9–12 address the technical gap — proving that probes measure real mechanisms, monitoring every inference, securing the transport, capturing activations at the hardware level, and persisting the audit trail. But the **institutional** gap remains: even a perfectly validated, causally verified, tamper-evident measurement is meaningless without an institutional context that decides what the number is allowed to count as.

This PoC is the technical proof that a governance framework would have something concrete to govern. The probe produces a number. Causal intervention proves the number reflects a real mechanism. The attestation signs it. The protocol lets someone else verify it. Hardware isolation proves the measurement wasn't faked. But who decides what the number *means* — that is the hardest problem, and it is not a technical one.
