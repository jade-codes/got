# System Architecture: Hardware to Software

How the protocol maps onto real hardware and software for a single
agent. Shows both the current PoC (MockEnclave, everything in one
process) and the target production deployment (real TEE, hardware
isolation).

All diagrams reflect the security-hardened codebase (353 tests passing).

---

## PoC Architecture (Current)

Everything runs on a standard machine. No hardware isolation.
MockEnclave simulates the enclave API but runs in the same process.

```
┌─────────────────────────────────────────────────────────────────────┐
│  HARDWARE                                                           │
│                                                                     │
│  Standard CPU (x86_64 / ARM)       GPU (optional, for model only)  │
│  No SGX/SEV required               CUDA for forward pass           │
│  RAM: model + protocol state        VRAM: model weights + KV cache │
└─────────┬───────────────────────────────────┬───────────────────────┘
          │                                   │
┌─────────v───────────────────────────────────v───────────────────────┐
│  OS / CONTAINER                                                     │
│                                                                     │
│  Linux (or any OS with Rust toolchain)                              │
│  Dev container, bare metal, or cloud VM                             │
│  Filesystem: stores .gotact, .gotue, .gotgeo, probes.json,         │
│              attestation JSON, keypair files, trust registry TOML   │
└─────────┬───────────────────────────────────────────────────────────┘
          │
┌─────────v───────────────────────────────────────────────────────────┐
│  PROCESS SPACE                                                      │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Python Process (runs once per extraction)                     │ │
│  │                                                                │ │
│  │  extract_activations.py                                        │ │
│  │    - Loads model via HuggingFace (GPU or CPU)                  │ │
│  │    - Hooks forward pass, captures activations                  │ │
│  │    - Reads lm_head.weight (unembedding matrix U)               │ │
│  │    - Writes .gotact + .gotue to filesystem                     │ │
│  │                                                                │ │
│  │  Runs separately from the Rust process.                        │ │
│  │  No IPC — communicates only via binary files on disk.          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                         │                                           │
│                    .gotact / .gotue (files on disk)                  │
│                         │                                           │
│  ┌──────────────────────v─────────────────────────────────────────┐ │
│  │  Rust Process (got-cli or agent runtime)                       │ │
│  │                                                                │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │  Layer 5: CLI / Agent Runtime                            │  │ │
│  │  │    - Parses .gotact, .gotue, .gotgeo binary files        │  │ │
│  │  │    - Orchestrates the full pipeline                      │  │ │
│  │  │    - Manages keypair lifecycle (zeroize on drop)         │  │ │
│  │  │    - Drives exchanges with peer agents                   │  │ │
│  │  │    N-3: All CLI commands return anyhow::Result<()>       │  │ │
│  │  └──────────────┬───────────────────────────────────────────┘  │ │
│  │                 │                                              │ │
│  │  ┌──────────────v───────────────────────────────────────────┐  │ │
│  │  │  Layer 4: MockEnclave  *** SAME ADDRESS SPACE ***        │  │ │
│  │  │    - Signing key in process memory (not hardware)        │  │ │
│  │  │    - Probes loaded from probes.json (agent can see)      │  │ │
│  │  │    - Geometry in process memory (agent can see)          │  │ │
│  │  │    - enclave_pipeline() runs probes + signs              │  │ │
│  │  │                                                          │  │ │
│  │  │  ⚠ No real isolation: the agent runtime can read the     │  │ │
│  │  │    signing key, probes, and geometry. This is the PoC    │  │ │
│  │  │    limitation that real TEE hardware (step 12) fixes.    │  │ │
│  │  └──────────────┬───────────────────────────────────────────┘  │ │
│  │                 │                                              │ │
│  │  ┌──────────────v───────────────────────────────────────────┐  │ │
│  │  │  Layers 3a/3b: Wire Protocol + Store                    │  │ │
│  │  │    - Frame encode → Result (N-1: size guard ≤ 16 MiB)   │  │ │
│  │  │    - ExchangeEnvelope (S-9: verified flag,               │  │ │
│  │  │      from_bytes_verified(), is_verified())               │  │ │
│  │  │    - verify_chain(signer_pks: &[VerifyingKey]) S-8      │  │ │
│  │  │    - perform_exchange() — in-memory for now              │  │ │
│  │  │    - MemoryStore / FileStore (atomic + hash-on-load)     │  │ │
│  │  │    - TrustRegistry (S-2: SHA-256 integrity on load,      │  │ │
│  │  │      max_attestation_age_secs, expected_model_hash)      │  │ │
│  │  │    - Domain scoping (§4 / Appendix B):                    │  │ │
│  │  │      check_domain_compatibility() at Phase 0              │  │ │
│  │  └──────────────┬───────────────────────────────────────────┘  │ │
│  │                 │                                              │ │
│  │  ┌──────────────v───────────────────────────────────────────┐  │ │
│  │  │  Layer 2: Attestation Signing (got-attest)               │  │ │
│  │  │    - serialise_for_signing() — canonical LE bytes        │  │ │
│  │  │    - Ed25519 sign / verify                               │  │ │
│  │  │    - S-7: timestamp ≤ now + 300 s                        │  │ │
│  │  │    - S-13: string fields ≤ 256 bytes                     │  │ │
│  │  │    - S-20: ≤ 1024 layers, ≤ 65536 readings              │  │ │
│  │  │    - attestation_hash() — SHA-256 for chain linkage      │  │ │
│  │  │    - merkle_root() — RFC 6962 Merkle tree                │  │ │
│  │  └──────────────┬───────────────────────────────────────────┘  │ │
│  │                 │                                              │ │
│  │  ┌──────────────v───────────────────────────────────────────┐  │ │
│  │  │  Layer 1: Probes + Causal Intervention (got-probe)       │  │ │
│  │  │    - train_probe() — SGD under causal IP                 │  │ │
│  │  │    - read_probe() / read_probe_checked()                 │  │ │
│  │  │    - causal_check() — perturbation experiment            │  │ │
│  │  │    - MeasurementSidecar — windowed runtime measurement   │  │ │
│  │  │    - CollectingHook (N-2: mutex poison recovery)         │  │ │
│  │  └──────────────┬───────────────────────────────────────────┘  │ │
│  │                 │                                              │ │
│  │  ┌──────────────v───────────────────────────────────────────┐  │ │
│  │  │  Layer 0: Core Types + Geometry (got-core)               │  │ │
│  │  │    - CausalGeometry (Φ = UᵀU, Gram matrix)              │  │ │
│  │  │    - GeometricAttestation (single canonical layout)      │  │ │
│  │  │    - S-21: model_hash is Option<[u8; 32]>               │  │ │
│  │  │    - sha256(), geometry_hash(), drift_from()             │  │ │
│  │  │    - UnsignedAttestation newtype wrapper                 │  │ │
│  │  │    - Phase 13: sequence_number, directional_drifts,      │  │ │
│  │  │      probe_commitment                                    │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Filesystem                                                    │ │
│  │                                                                │ │
│  │  agent_key.sec    Ed25519 secret key (0o600 perms, zeroize)   │ │
│  │  agent_key.pub    Ed25519 public key                          │ │
│  │  model.gotact     Extracted activations (binary)              │ │
│  │  model.gotue      Extracted unembedding (binary)              │ │
│  │  model.gotgeo     Geometry checkpoint (binary)                │ │
│  │  probes.json      Trained probe weights                       │ │
│  │  attestation.json Signed attestation (JSON)                   │ │
│  │  registry.toml    Trust registry (agent IDs, keys, policy)    │ │
│  │  store/           FileStore directory (attestation history)    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What's missing in the PoC

| Gap | Why it matters | What fixes it |
|-----|---------------|---------------|
| No hardware isolation | Agent can read signing key + probes | Real TEE (step 12) |
| No network transport | Exchanges are in-memory | TCP/TLS transport over wire framing |
| No remote attestation | Can't prove the enclave is genuine hardware | Intel IAS/DCAP or AMD SEV cert chain |
| No probe provisioning channel | Probes loaded from local file | Secure channel from governance body to enclave |
| Model forward pass is external | `model_fn` closure called from agent process | Model inference inside confidential compute |

---

## Production Architecture (Target)

With real TEE hardware. The enclave is a hardware-isolated region that
the agent runtime, host OS, and hypervisor cannot read.

```
┌─────────────────────────────────────────────────────────────────────┐
│  HARDWARE                                                           │
│                                                                     │
│  CPU: Intel Xeon (SGX) or AMD EPYC (SEV)                           │
│    - TEE hardware: encrypted memory regions                         │
│    - Attestation key fused at manufacture                           │
│    - Memory encryption engine (MEE / SME)                           │
│                                                                     │
│  GPU: NVIDIA H100 (Confidential Compute)                            │
│    - VRAM encrypted by GPU hardware                                 │
│    - Host CPU cannot read GPU memory                                │
│    - DMA to TEE enclave preserves confidentiality                   │
│                                                                     │
│  HSM (optional): hardware security module for key storage           │
└────────┬────────────────────────────┬─────────────────┬─────────────┘
         │                            │                 │
┌────────v────────────────────────────v─────────────────v─────────────┐
│  HYPERVISOR / HOST OS                                               │
│                                                                     │
│  Can schedule/manage enclaves and VMs                               │
│  CANNOT read encrypted enclave memory                               │
│  CANNOT forge attestation reports                                   │
│  CANNOT extract signing keys                                        │
└────────┬────────────────────────────┬───────────────────────────────┘
         │                            │
         │ untrusted                  │ hardware-isolated
         │                            │
┌────────v──────────────┐  ┌──────────v──────────────────────────────┐
│  AGENT RUNTIME        │  │  TEE ENCLAVE (hardware-isolated)        │
│  (untrusted)          │  │                                         │
│                       │  │  ┌───────────────────────────────────┐  │
│  Orchestration:       │  │  │  Signing Key                      │  │
│    - Keypair mgmt     │  │  │  (generated inside, never leaves) │  │
│    - Exchange logic   │  │  └───────────────────────────────────┘  │
│    - Wire protocol    │  │                                         │
│      N-1: encode      │  │  ┌───────────────────────────────────┐  │
│        → Result       │  │  │  Probes (provisioned by           │  │
│    - Trust registry   │  │  │  governance body via encrypted    │  │
│      S-2: integrity   │  │  │  channel — agent never sees them) │  │
│    - Attestation      │  │  └───────────────────────────────────┘  │
│      store            │  │                                         │
│                       │  │  ┌───────────────────────────────────┐  │
│  Envelope handling:   │  │  │  CausalGeometry (Φ)               │  │
│    S-9: verified flag │  │  │  (computed inside from .gotue)     │  │
│    from_bytes_        │  │  └───────────────────────────────────┘  │
│      verified()       │  │                                         │
│                       │  │  Pipeline (all inside enclave):         │
│  Chain verification:  │  │    1. Receive ActivationFrames via DMA  │
│    S-8: &[VerifyingKey]│ │    2. Verify integrity hashes           │
│    key rotation       │  │    3. Run probes (read_probe)           │
│                       │  │    4. Run causal checks (causal_check)  │
│  Cannot read:         │  │    5. Assemble attestation              │
│    - Enclave memory   │  │       S-7/S-13/S-20 gates              │
│    - Signing key      │  │       S-21: model_hash Option           │
│    - Probe weights    │  │    6. Sign with enclave key             │
│                       │  │    7. Return signed blob to agent       │
│  Receives:            │  │                                         │
│    - Signed attest.   │  │  Remote Attestation:                    │
│      (opaque blob)    │  │    CPU vendor signs a report proving:   │
│    - CausalScores     │  │    "This code is running in genuine     │
│                       │  │     SGX/SEV hardware, the enclave has   │
│                       │  │     not been tampered with, and the     │
│                       │  │     signing key was generated inside."  │
│                       │  │                                         │
└────────┬──────────────┘  └─────────────────────────┬───────────────┘
         │                                           │
         │ signed attestation (opaque)               │ DMA (encrypted)
         │                                           │
┌────────v───────────────────────────────────────────v────────────────┐
│  NETWORK                                                            │
│                                                                     │
│  Agent A ◄──── TLS / wire frames ────► Agent B                     │
│                                                                     │
│  Frame { magic: GOT1, type, length, payload }                       │
│    N-1: encode returns Result, payload ≤ 16 MiB                     │
│                                                                     │
│  ExchangeRequest { envelope, chain, current_attestation }           │
│  ExchangeResponse { envelope, verdict, chain, current_attestation } │
│    S-9: envelope.verified flag tracks verification state            │
│                                                                     │
│  Each agent:                                                        │
│    1. Self-attests (enclave pipeline)                                │
│    2. Sends signed attestation + chain to peer                      │
│    3. Receives peer's signed attestation + chain                    │
│    4. Verifies envelope (S-9), chain (S-8), drift bounds            │
│    5. Cooperate or refuse                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What the TEE gives you

| Property | Without TEE (PoC) | With TEE (production) |
|----------|-------------------|----------------------|
| Signing key confidentiality | Agent process can read it | Hardware prevents extraction |
| Probe confidentiality | Agent loads probes.json | Probes provisioned into enclave only |
| Measurement integrity | Agent could tamper with readings | Enclave runs probes, agent can't interfere |
| Attestation authenticity | Trust the agent's honesty | Trust the CPU vendor's hardware |
| Remote attestation | Not available | CPU vendor certifies enclave is genuine |
| Model inference isolation | Runs in agent process | GPU confidential compute (H100) |

### Data flow through the production stack

```
  Model (GPU, encrypted VRAM)
       │
       │ DMA (hardware-encrypted channel)
       v
  TEE Enclave
       │ receives ActivationFrames
       │ verifies integrity hashes
       │ computes Φ = UᵀU
       │ runs probes: wᵀΦh + b
       │ runs causal checks: perturb h, observe y
       │ assembles GeometricAttestation (single canonical layout)
       │   S-7/S-13/S-20 bounds checks
       │   S-21: model_hash is Option<[u8; 32]>
       │ signs with enclave-resident Ed25519 key
       │
       │ signed attestation (opaque blob)
       v
  Agent Runtime (untrusted)
       │ wraps attestation in ExchangeEnvelope
       │   S-9: verified flag set by create()
       │ builds request/response
       │ sends via wire protocol
       │   N-1: Frame::encode() → Result
       │
       │ TLS / wire frames
       v
  Peer Agent
       │ validates envelope (from_bytes_verified — S-9)
       │ walks attestation chain
       │   S-8: verify_chain(&[VerifyingKey]) for rotation
       │ checks drift bounds
       │ cooperate or refuse
```

---

## Mapping: Build Order Steps to System Components

| Build Step | Component | Runs in | Security Hardening |
|-----------|-----------|---------|-------------------|
| 1. Core types | got-core | Agent process | S-21: model_hash Option |
| 2. Geometry | got-core::geometry | Agent (PoC) / Enclave (prod) | — |
| 3. Probes | got-probe | Agent (PoC) / Enclave (prod) | N-2: mutex poison recovery |
| 4. Attestation signing | got-attest | Agent (PoC) / Enclave (prod) | S-7, S-13, S-20 |
| 5. CLI | got-cli | Agent process | N-3: anyhow::Result |
| 6. Integration tests | tests/ | CI | 353 tests passing |
| 7. Python bridge | extract_activations.py | Separate Python process | — |
| 8. Drift + chaining | got-core + got-attest | Agent (PoC) / Enclave (prod) | — |
| 9. Causal intervention (KEYSTONE) | got-probe::intervention | Agent (PoC) / Enclave (prod) | — |
| 10. Inline measurement | got-probe::hooks | Agent process / Enclave | N-2: poison recovery |
| 11. Wire protocol | got-wire | Agent process | S-2, S-8, S-9, N-1 |
| 12. Hardware isolation | got-enclave (real TEE) | TEE enclave | Full boundary |
