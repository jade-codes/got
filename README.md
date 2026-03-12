# Geometry of Trust

**A cryptographically signed, deterministic measurement framework for AI value alignment — built on the causal geometry of transformer residual streams.**

> The geometry is ready. The governance is not. The most important work in AI alignment is not technical. It is institutional.

## What Is This?

Geometry of Trust (GoT) is a hackathon proof-of-concept that demonstrates a concrete technical pipeline for measuring, attesting, and independently verifying whether an AI model's internal representations encode specific value-relevant properties.

It works by exploiting a key finding from mechanistic interpretability research: value-relevant concepts have **measurable linear structure** in a transformer's residual stream. GoT formalises this with a **causal inner product** derived from the model's own unembedding matrix, trains linear probes under that geometry, and produces **Ed25519-signed attestations** that any independent party can reproduce and verify.

### The Core Idea

Instead of asking a model *what it believes* (which it can lie about), GoT measures the **geometric structure of its internal activations** under a mathematically principled metric:

$$\langle u, v \rangle_c = u^\top \Phi \, v \quad \text{where} \quad \Phi = U^\top U$$

This causal inner product weights directions in the residual stream by how much they influence the model's actual output distribution — not just by Euclidean distance.

## Key Properties

- **Deterministic**: Same model + same input + same probes = identical readings, byte-for-byte
- **Independently reproducible**: Anyone with the model weights can re-extract activations and verify the attestation (Tier 3 trust)
- **Cryptographically signed**: Ed25519 signatures over canonical serialised bytes
- **Chainable**: Attestations can reference parent attestations, tracking value drift over model updates
- **Causal**: Optional intervention-based proofs that probe directions *causally* influence model output, not just correlate with it
- **Calibrated**: Platt scaling + ECE metric ensure confidence values are meaningful, not just ranked
- **PKI-backed**: Agent certificates with expiry, revocation (CRL), and key rotation ceremonies

## Architecture

```
got-core        Layer 0 — Core types, causal geometry (Gram matrix, inner product)
  ↑       ↑
got-probe   got-attest    Layer 1–2 — Probe training/inference, attestation signing
  ↑       ↑
got-wire    got-store     Layer 3 — Wire protocol for agent exchange, attestation storage
  ↑       ↑
got-enclave               Layer 4 — Hardware enclave boundary (TEE mock for PoC)
  ↑
got-cli                   Layer 5 — CLI binary orchestrating the full pipeline
```

| Crate | Purpose |
|---|---|
| `got-core` | `CausalGeometry`, `GeometricAttestation`, `LayerActivation`, precision types |
| `got-probe` | Linear probe training (SGD under causal IP), Platt calibration, ECE metric, inference, causal checks |
| `got-attest` | Attestation assembly, Ed25519 signing/verification, Merkle roots |
| `got-wire` | Framed wire protocol, exchange envelopes, chain verification, trust registry, PKI certificates, CRL |
| `got-store` | Attestation persistence (in-memory and on-disk), audit reports |
| `got-enclave` | TEE abstraction — signing keys never leave the enclave boundary |
| `got-cli` | CLI with `keygen`, `train`, `attest`, `verify`, `checkpoint`, `drift`, `calibration-report`, `issue-cert`, `revoke-cert`, `rotate-key` subcommands |

## Getting Started

### Prerequisites

- **Rust** (stable, 2021 edition) — for building the core pipeline
- **Python 3.8+** with `torch` and `transformers` — for extracting activations from a model

### Build

```bash
git clone https://github.com/gim-home/got.git
cd got
cargo build --release
```

### Run the Tests

```bash
# Unit tests across all crates
cargo test

# Integration tests (end-to-end pipeline with synthetic data)
cargo test --test integration
```

The integration tests exercise the full pipeline — geometry computation, probe training, attestation, verification, chaining, drift detection, and causal intervention — all with synthetic data, no GPU required.

### End-to-End Pipeline (With a Real Model)

#### 1. Extract Activations

Use the Python extraction script to pull residual-stream activations and the unembedding matrix from any HuggingFace causal language model:

```bash
pip install torch transformers

python scripts/extract_activations.py \
    --model meta-llama/Llama-3-8B \
    --input "The cat sat on the mat" \
    --layers 12 18 24 \
    --output-activations data/activations.gotact \
    --output-unembedding data/unembedding.gotue
```

This produces binary `.gotact` and `.gotue` files in documented formats (see [scripts/README.md](scripts/README.md)).

Tested models include LLaMA, Mistral, GPT-2, GPT-Neo, and GPT-J.

#### 2. Create Labels

Edit the generated labels stub — one `0` or `1` per token position — to reflect the value dimension you're probing for:

```bash
vim data/activations.labels
```

#### 3. Generate a Signing Key

```bash
cargo run --release -p got-cli -- keygen --output data/key
```

This creates `data/key` (secret) and `data/key.pub` (public).

#### 4. Train Probes

Train a linear probe for a specific layer under the causal inner product:

```bash
cargo run --release -p got-cli -- train \
    --activations data/activations.gotact \
    --labels data/activations.labels \
    --unembedding data/unembedding.gotue \
    --layer 12 \
    --dimension "harmlessness" \
    --output data/probes_layer12.json
```

To train **calibrated** probes with Platt scaling, provide held-out validation labels:

```bash
cargo run --release -p got-cli -- train \
    --activations data/activations.gotact \
    --labels data/activations.labels \
    --validation-labels data/validation.labels \
    --unembedding data/unembedding.gotue \
    --layer 12 \
    --dimension "harmlessness" \
    --output data/probes_layer12.json
```

Repeat for additional layers as needed.

#### 5. Produce an Attestation

```bash
cargo run --release -p got-cli -- attest \
    --activations data/activations.gotact \
    --probes data/probes_layer12.json \
    --unembedding data/unembedding.gotue \
    --key data/key \
    --model-id "meta-llama/Llama-3-8B" \
    --output data/attestation.json
```

The output is a signed `GeometricAttestation` JSON containing probe readings, confidence scores, coverage flags, and the Ed25519 signature.

#### 6. Verify

```bash
cargo run --release -p got-cli -- verify \
    --attestation data/attestation.json \
    --pubkey data/key.pub
```

#### 7. Evaluate Calibration (Optional)

After training calibrated probes, generate an ECE (Expected Calibration Error) report:

```bash
cargo run --release -p got-cli -- calibration-report \
    --activations data/activations.gotact \
    --labels data/validation.labels \
    --probes data/probes_layer12.json \
    --unembedding data/unembedding.gotue
```

This prints a per-bin confidence-vs-accuracy table and the overall ECE score.

#### 8. Track Drift (Optional)

Save a geometry checkpoint and later compare against an updated model:

```bash
# Save reference geometry
cargo run --release -p got-cli -- checkpoint \
    --unembedding data/unembedding.gotue \
    --output data/reference.gotgeo

# After model update, measure drift
cargo run --release -p got-cli -- drift \
    --reference data/reference.gotgeo \
    --current data/unembedding_v2.gotue
```

### PKI: Certificates, Revocation, and Key Rotation

GoT includes a minimal PKI for binding agent keys to verifiable identities.

#### Issue a Certificate

```bash
# Generate a CA keypair
cargo run --release -p got-cli -- keygen --output data/ca

# Issue a certificate for an agent
cargo run --release -p got-cli -- issue-cert \
    --ca-key data/ca \
    --subject-pubkey data/key.pub \
    --subject-name "alice" \
    --roles producer,verifier \
    --validity-days 365 \
    --output data/alice-cert.json
```

#### Revoke a Certificate

```bash
cargo run --release -p got-cli -- revoke-cert \
    --ca-key data/ca \
    --cert data/alice-cert.json \
    --reason key-compromise \
    --output data/crl.json
```

#### Rotate an Agent Key

```bash
cargo run --release -p got-cli -- keygen --output data/key2

cargo run --release -p got-cli -- rotate-key \
    --old-key data/key \
    --new-key data/key2 \
    --ca-key data/ca \
    --subject-name "alice" \
    --roles producer,verifier \
    --output data/rotation.json
```

The trust registry validates certificates against configured CA keys, checks expiry on every exchange, and rejects agents whose certificates appear in a loaded CRL.

## Trust Tiers

The attestation schema supports three progressive levels of trust:

| Tier | Schema | What It Proves |
|---|---|---|
| **Tier 1 — Signature** | v1 | Ed25519 signature over deterministic canonical bytes |
| **Tier 2 — Consistency** | v2 | Signature + parent chain hash + geometry drift bounds + coverage flags |
| **Tier 3 — Reproduction** | v3 | Full re-extraction + re-probing + causal intervention scores + bitwise match |

## What This PoC Does *Not* Do

This project deliberately proves only the **technical substrate** — that the causal inner product is computable, probe readings are deterministic, and attestations are independently reproducible. It does not address:

- **Real TEE integration** — the enclave layer is a software mock; production deployment requires SGX/TDX/SEV hardware
- **Corpus curation** — who decides what concepts to probe for
- **Probe interpretation** — what the readings *mean* for governance
- **Coverage semantics** — whether the probed dimensions are sufficient
- **Institutional governance** — who has standing to adjudicate trust
- **Platt calibration ground truth** — the calibration pipeline is functional but needs real-world labelled datasets for meaningful ECE scores

Those are the hard problems. This is the plumbing that proves the hard problems are worth solving.

## Documentation

- [PLAN.md](PLAN.md) — Full implementation plan with mathematical details
- [scripts/README.md](scripts/README.md) — Activation extraction guide and binary format specs
- [docs/architecture-layers.md](docs/architecture-layers.md) — Layer-by-layer architecture
- [docs/architecture-code.md](docs/architecture-code.md) — Crate dependency graph and internal structure
- [docs/architecture-flows.md](docs/architecture-flows.md) — End-to-end data flow diagrams
- [docs/architecture-sequences.md](docs/architecture-sequences.md) — Sequence diagrams for key operations
- [docs/architecture-deployment.md](docs/architecture-deployment.md) — PoC and production deployment architecture
- [docs/architecture-agent-protocol.md](docs/architecture-agent-protocol.md) — Agent-to-agent attestation protocol
- [docs/architecture-motherboard.md](docs/architecture-motherboard.md) — Motherboard-style trust and comms diagrams

## License

This is a hackathon proof-of-concept.
