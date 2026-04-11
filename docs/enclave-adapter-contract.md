# Enclave Adapter Contract

The protocol's hardware-isolated measurement layer (`got-enclave`) is
specified by two traits — `HardwareCapture` and `MeasurementEnclave` —
plus an in-process mock implementation (`MockDmaTap`, `MockEnclave`)
that the rest of the workspace uses for tests. This document describes
**what a real TEE adapter has to provide** to satisfy the contract,
**what the surrounding code relies on**, and **the boundary between
the mock and a production adapter**. The intent is that someone with
SGX, SEV-SNP, or H100 confidential computing hardware can drop in a
real implementation without having to refactor anything else in the
workspace.

> **Status of the implementation.** `MockEnclave` validates the
> protocol *flow* — frame capture, integrity verification, probe
> evaluation, causal intervention, attestation signing — but does
> **not** provide a security boundary. The mock runs in the same
> address space as the agent runtime; the signing key, the probes,
> and the model handle are all reachable from the host process. This
> document is what makes the security gap explicit and unambiguous,
> so that a real adapter has a concrete contract to satisfy.
>
> Real hardware integration is genuinely blocked on having the
> hardware to test against, plus the platform SDKs and live
> attestation infrastructure (Intel Attestation Service, AMD SEV
> firmware, NVIDIA attestation). The protocol cannot ship working
> SGX/SEV/H100 from a workstation that has none of those.

---

## 1. The two traits

### 1.1 `HardwareCapture` — the byte path

```rust
pub trait HardwareCapture: Send + Sync {
    fn capture(&self, layer: usize, token_position: usize, values: &[f32])
        -> Option<ActivationFrame>;
    fn hidden_dim(&self) -> usize;
    fn captured_layers(&self) -> Vec<usize>;
}
```

The `HardwareCapture` implementation is the boundary between the
**model** (which runs on the GPU/accelerator) and the **enclave**
(which runs the probes). Its job is to deliver each requested
activation as an `ActivationFrame` whose `integrity_hash` is computed
**by the capture hardware itself**, not by the model process.

A production adapter must guarantee:

| Property | Mock satisfies? | Real adapter must |
|---|---|---|
| Activations come from VRAM as the model writes them, before any model-side overwrite or sanitisation | ❌ (mock receives them from the host process) | Use DMA snoop, PCIe tap, or confidential-compute copy-out so the model cannot influence what is captured |
| The integrity hash is computed by the capture hardware, not by the model | ❌ (mock computes it in-process) | Compute on the capture hardware (e.g. inside the SGX driver, the SEV firmware, or the H100 confidential computing engine) before the bytes leave the trust boundary |
| The hash binds layer + token_position + values | ✅ | Same — the canonical layout is `H(layer || token_position || canonical(values))` |
| Float canonicalisation: `-0.0` is normalised to `+0.0` before hashing | ✅ | Same — must match `ActivationFrame::compute_hash` exactly so the enclave's recomputed hash agrees |
| `Send + Sync` (the capture device can run on a dedicated capture thread) | ✅ | Same |

The contract is intentionally narrow: an `ActivationFrame` is just
`(layer, token_position, values, integrity_hash)`, and the enclave
calls `frame.verify_integrity()` to recompute the hash and reject
tampered transit. **A real capture adapter is the entire trust path
between the model's memory and the enclave's memory** — every byte
the enclave consumes flows through this trait.

### 1.2 `MeasurementEnclave` — the trust path

```rust
pub trait MeasurementEnclave {
    fn receive_activations(&mut self, frame: ActivationFrame) -> Result<(), EnclaveError>;
    fn run_causal_check(&self, delta: f32) -> Result<Vec<CausalScore>, EnclaveError>;
    fn attest(...) -> Result<GeometricAttestation, EnclaveError>;
    fn attest_with_causal(...) -> Result<GeometricAttestation, EnclaveError>;
    fn verifying_key(&self) -> VerifyingKey;
    fn frame_count(&self) -> usize;
    fn reset(&mut self);
}
```

The `MeasurementEnclave` implementation is **the entire trust
boundary** of the attestation pipeline. Once a frame is inside the
enclave, the model — and the host process running the model —
cannot influence what the probes measure, what causal intervention
results report, or what the signing key produces.

A production adapter must guarantee, at minimum:

| Property | Mock satisfies? | Real adapter must |
|---|---|---|
| `receive_activations` recomputes `integrity_hash` and rejects mismatches | ✅ | Same; mismatch returns `EnclaveError::IntegrityViolation` |
| The signing key is **generated inside** the enclave at provisioning time and never exported | ❌ (mock takes the key as a constructor argument) | Generate the key inside the TEE; expose only the verifying key via `verifying_key()`. The signing key bytes never cross the enclave boundary, including to the host process that loaded the enclave |
| The probes are provisioned through a secure channel and never visible to the host process | ❌ (mock takes them as a `Vec<ProbeVector>` constructor argument) | Use sealed storage, remote attestation, or attestation-gated provisioning to load probes; the host process running the model never sees the probe weights |
| The model handle used for causal intervention runs **inside** the enclave, not the host's model | ❌ (mock takes a `Box<dyn ModelHandle>`) | Load a model shard into TEE-isolated memory at provisioning time; `run_causal_check` invokes the in-enclave model, never the host's |
| `attest_with_causal` embeds causal scores **before signing**, so the signing key never leaves the enclave between probing and signing | ✅ | Same — the contract is "compute, then sign, then return". A real adapter that splits these phases breaks the security guarantee |
| Memory between enclave invocations is isolated from the host (or wiped on each invocation) | ❌ | Use the platform's enclave memory isolation (SGX EPC, SEV-SNP encrypted memory, H100 CC memory partition) |
| Sequence numbers are monotonic and durable across restarts | ⚠️ (mock keeps an in-memory counter) | Back the counter with a hardware monotonic counter (SGX MC, AMD vTPM, NVIDIA secure counter) so a rollback attack cannot replay an old sequence number |

### 1.3 The `attest` / `attest_with_causal` distinction

The trait offers two attestation entry points. **`attest_with_causal`
is the preferred method when causal validation has been performed**,
because it embeds the causal scores into the unsigned attestation
struct *inside* the enclave and signs the result *inside* the
enclave. There is no point at which the signing key handles a
partial payload, and no opportunity for a host-side caller to
substitute its own causal scores into a payload signed by enclave-
held probes.

A real adapter that implements only `attest` and asks the host to
fill in causal scores after signing **violates the contract** —
the attestation would carry a key that the enclave authorised but
content the host could tamper with. The trait is shaped this way
specifically to prevent that mistake.

---

## 2. What the surrounding code relies on

The exchange pipeline never speaks to `HardwareCapture` or
`MeasurementEnclave` directly. It speaks to a `GeometricAttestation`
and asks `got-attest::verify` whether the signature is valid against
a verifying key in the trust registry. The contract `got-enclave`
satisfies for the rest of the workspace is therefore narrow:

1. **`verifying_key()` returns a stable Ed25519 public key** that
   the trust registry can pin via `expected_model_hash`-style
   binding. The same enclave instance must always return the same
   key (across calls and across `reset()`).
2. **`attest_with_causal` returns a `GeometricAttestation` whose
   signature verifies against `verifying_key()`** under the
   canonical bytes from `got-attest::serialise_for_signing`. If
   the enclave applies any custom canonicalisation, it MUST match
   the protocol's, otherwise external verifiers will reject every
   attestation.
3. **The returned attestation's `causal_scores` reflect the actual
   intervention performed inside the enclave**. If a real adapter
   stubs causal scores, the attestation is meaningless even though
   the signature is valid — the §8.2 `require_causal_validation`
   check passes against fabricated scores. This is a contract
   violation that no test can catch from outside the enclave; the
   adapter author has to vouch for it.
4. **`frame_count()` and `reset()` correctly track the
   measurement window**. The host sidecar uses these to decide
   when to close a window and request an attestation.
5. **`EnclaveError::IntegrityViolation` is raised when, and only
   when, a frame's hash does not match its values**. Other errors
   should not be reported as integrity violations.

That is the entire surface. Everything else — how probes are
provisioned, how the model handle is loaded, how the signing key is
generated, how the enclave's memory is isolated — is invisible to
the rest of the workspace. A real adapter is free to implement
those concerns however the platform SDK requires.

---

## 3. The mock's job (and what the mock cannot do)

`MockEnclave` exists for one reason: to let the *protocol flow* be
tested end-to-end without requiring SGX/SEV/H100 hardware. It is
gated behind `#[cfg(any(test, feature = "mock"))]` so production
binaries cannot accidentally link it.

What the mock validates:

- The end-to-end pipeline from `HardwareCapture` → frame integrity →
  enclave-side probe evaluation → causal intervention → signed
  attestation works.
- `assemble_and_sign` produces verifiable signatures from the
  enclave-internal key.
- `IntegrityViolation` is raised when the capture device tampers
  with a frame's hash (`MockDmaTap::with_tamper()`).
- `enclave_pipeline` (the high-level helper) wires capture →
  receive → causal_check → attest_with_causal correctly.

What the mock does **not** validate:

- Memory isolation between the enclave and the host process.
- Sealing of the signing key against host process introspection.
- Sealing of the probe weights against host process introspection.
- Rollback resistance of the sequence counter.
- That a malicious model running on the same machine cannot exfiltrate
  enclave state via cache timing, speculative execution, or shared
  GPU resources.

These are all real security concerns, and they are all the
responsibility of the platform-specific TEE implementation. The mock
takes the position that *if* you treat the mock as a black box from
outside the enclave boundary, the protocol behaves correctly — the
hardware integration's job is to make "treat as a black box" actually
true.

---

## 4. Adapter sketches per platform

The following are not implementations — they are notes on how an
adapter author would map the trait surface onto each platform's
primitives. The point is to show that the trait shape is realisable
on the platforms the paper names; nothing here ships in this repo.

### 4.1 Intel SGX

- **Probe / signing key provisioning**: load via SGX remote
  attestation. Probes and signing key seed are sealed to the
  enclave's `MRENCLAVE` measurement, so only the same enclave build
  can unseal them.
- **Model handle**: load a model shard into EPC memory at enclave
  startup. Causal intervention runs inside the enclave; the host
  never sees activations or probe outputs.
- **`HardwareCapture`**: difficult on SGX since the model typically
  runs on a GPU outside the enclave. One approach is to run the model
  itself inside an SGX2 enclave with EDMM, accepting the throughput
  cost. Another is to use a CPU-only model with SGX-internal forward
  pass.
- **Sequence counter**: SGX monotonic counters (`sgx_create_monotonic_counter`).
- **Attestation**: SGX quote bound to the enclave's `verifying_key()`
  proves to remote verifiers that the key was generated inside a
  legitimate enclave with a known measurement.

### 4.2 AMD SEV-SNP

- **Probe / signing key provisioning**: deliver via SEV-SNP guest
  attestation, sealed to the launch measurement. The hypervisor
  cannot read encrypted guest memory.
- **Model handle**: load the model into the encrypted guest's
  memory. The entire VM is the enclave; capture flows from a model
  process *inside* the VM to a measurement process *inside* the
  same VM.
- **`HardwareCapture`**: can use a co-resident model process within
  the encrypted VM. DMA from a GPU outside the VM is harder; an
  H100 with CC mode is the better fit there.
- **Sequence counter**: AMD vTPM or a hypervisor-provided counter
  that survives migration.
- **Attestation**: SEV attestation report binds the enclave's
  verifying key to the launch measurement.

### 4.3 NVIDIA H100 Confidential Computing

- **Probe / signing key provisioning**: load via NVIDIA's CC mode
  attestation flow; sealed to the GPU's launch measurement.
- **Model handle**: model shards loaded into encrypted VRAM; the
  entire forward pass runs in CC mode so activations never leave
  the encrypted partition.
- **`HardwareCapture`**: this is the *natural* fit. The capture
  device and the enclave are both inside the GPU's CC partition.
  Activations are read directly from encrypted VRAM into the
  measurement code, no CPU round-trip.
- **Sequence counter**: NVIDIA secure counter or pair the GPU
  attestation with a CPU-side TPM counter.
- **Attestation**: NVIDIA attestation service binds the verifying
  key to the GPU's CC launch state.

In all three cases the trait shape fits — the hard work is the
platform SDK plumbing, not the contract this document describes.

---

## 5. Drop-in checklist for a real adapter

If you have hardware to test against, here's what you'd implement:

1. A `HardwareCapture` impl that talks to your capture device. Most
   of the work is in the device driver / SDK; the trait surface is
   small.
2. A `MeasurementEnclave` impl that lives inside your enclave runtime.
   Generate the signing key inside the enclave. Load probes via
   sealed storage or attestation. Provision the model handle from a
   verified shard.
3. A constructor that returns `Box<dyn MeasurementEnclave>` to host
   code. The constructor performs the platform attestation flow and
   sets up the enclave; on failure it returns an error rather than
   handing back a half-initialised enclave.
4. A way to bind the enclave's `verifying_key()` to the trust
   registry. Typically this means: the enclave produces an
   attestation report (SGX quote / SEV report / H100 CC report)
   over the verifying key, the report is verified by the registry
   maintainer out-of-band, and the verifying key is added to the
   `TrustRegistry` as a regular agent entry whose `expected_model_hash`
   pins the launch measurement.
5. A test harness that exercises `enclave_pipeline` against the real
   adapter and confirms the pipeline behaves the same as it does
   against `MockEnclave`. If your adapter passes the same tests
   `got-enclave::tests` runs against the mock, the rest of the
   workspace should consume it without modification.

The deliberate goal of this contract is that step 5 produces a
near-identical test surface — the only thing changing is which
binary owns the secrets.

---

## 6. What this document is not

- **Not a security audit.** It does not certify any specific
  adapter as secure; it specifies the interface a secure adapter
  must satisfy.
- **Not a substitute for platform documentation.** SGX, SEV-SNP, and
  H100 CC each have hundreds of pages of platform docs. This
  document is the protocol-side contract; the platform side is the
  vendor's responsibility.
- **Not a guarantee that a working adapter exists.** As of this
  writing, no production-grade adapter ships in this repo. The mock
  is the only implementation, and it is intentionally insecure.
- **Not a spec for cross-platform adapters.** Each platform has
  different primitives and tradeoffs; an adapter is per-platform.

---

## 7. References

- `crates/got-enclave/src/lib.rs` — module-level overview.
- `crates/got-enclave/src/capture.rs` — `HardwareCapture`,
  `ActivationFrame`, `MockDmaTap`.
- `crates/got-enclave/src/enclave.rs` — `MeasurementEnclave`,
  `MockEnclave`, `enclave_pipeline`.
- Protocol paper §11 (Phase 11: hardware-isolated measurement)
  and §14.1 (deferred limitations).
- [`docs/architecture-agent-protocol.md`](architecture-agent-protocol.md) §10.1.
