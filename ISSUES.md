# Issues

Codebase audit — updated 2026-03-05.

## Previously Fixed (Issues 1–13)

All 13 original spec-violation issues have been implemented and the integration test suite passes (68/68, 0 warnings).

| # | Severity | Component | Issue | Status |
|---|----------|-----------|-------|--------|
| 1 | **High** | got-core | Library panics instead of returning `Result` | ✅ Fixed |
| 2 | **High** | got-cli | `divergence_flag` always `false` | ✅ Fixed |
| 3 | **High** | got-cli | Unchecked `read_probe` skips geometry validation | ✅ Fixed |
| 4 | **High** | got-cli | `sequence_number` always 0, breaks chain verification | ✅ Fixed |
| 5 | Medium | all | `directional_drifts` never populated | ✅ Fixed |
| 6 | Medium | all | `probe_commitment` never computed | ✅ Fixed |
| 7 | Medium | got-enclave | `-0.0` not canonicalised in activation frame hash | ✅ Fixed |
| 8 | Medium | got-wire | Envelope timestamp uses BE, convention is LE | ✅ Fixed |
| 9 | Low | got-core | `GeometryError::NaN` returned for infinity inputs | ✅ Fixed |
| 10 | Low | got-attest/cli | `merkle_root(&[])` collides with "no shards provided" | ✅ Fixed |
| 11 | Low | got-wire | Registry can't parse `expected_model_hash` from TOML | ✅ Fixed |
| 12 | Low | got-attest | No structural validation of `causal_flag` vs scores | ✅ Fixed |
| 13 | Low | got-probe | Unsigned attestation has no type-level guard | ✅ Fixed |

---

## Current Issues — Code Hygiene Sweep

Compiler reports 0 warnings across the workspace (`cargo build`, `cargo test --no-run`).
The issues below were found via manual source-level review.

---

### 14. Broken unit tests in got-probe (Fix #1 fallout)

**Severity**: High — crate-level tests do not compile

Fix #1 changed `UnembeddingMatrix::new` to return `Result`, but crate-internal
test helpers in got-probe were not updated (only `tests/integration.rs` was fixed).

| File | Line | Problem |
|------|------|---------|
| `crates/got-probe/src/lib.rs` | 223 | `UnembeddingMatrix::new(...)` missing `.unwrap()` |
| `crates/got-probe/src/hooks.rs` | 582 | Same |
| `crates/got-probe/src/intervention.rs` | 329 | Same |

**Fix**: Add `.unwrap()` to all three call sites.

---

### 15. Stale `merkle_root_empty` test in got-attest (Fix #10 fallout)

**Severity**: High — test actively fails

Fix #10 changed `merkle_root(&[])` to return `sha256(b"merkle-empty")` instead
of `[0u8; 32]`, but the unit test in `crates/got-attest/src/lib.rs:548` still
asserts `[0u8; 32]`.

**Fix**: Update assertion to `got_core::sha256(b"merkle-empty")`.

---

### 16. Duplicate `agent_id` functions

**Severity**: Low — redundant code

Two public functions with identical implementations:
- `got_wire::envelope::agent_id(pk)` → `got_core::sha256(pk.as_bytes())`
- `got_wire::registry::compute_agent_id(pk)` → `got_core::sha256(pk.as_bytes())`

Both are called from different modules. `compute_agent_id` is the more
descriptive name and is used by the exchange protocol.

**Fix**: Remove `envelope::agent_id`, re-export or call `registry::compute_agent_id`
from envelope tests. Or consolidate into a single public function in `got-wire`'s
top-level `lib.rs`.

---

### 17. `got_attest::sha256` is a trivial re-export of `got_core::sha256`

**Severity**: Low — unnecessary indirection

`crates/got-attest/src/lib.rs` defines:
```rust
#[inline]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    got_core::sha256(data)
}
```

The CLI imports `sha256` from `got_attest`. All other crates already depend on
`got_core` and can call `got_core::sha256` directly.

**Fix**: Remove `got_attest::sha256`. Update `got-cli/src/main.rs` import to
use `got_core::sha256`.

---

### 18. Unused `name` parameter in `CausalGeometry::check_vec`

**Severity**: Low — dead parameter

`crates/got-core/src/geometry.rs` line 266:
```rust
let _ = name; // used only in potential future error messages
```

The `name: &str` parameter is accepted by `check_vec` but immediately discarded.
Every call site passes a string literal that serves no purpose.

**Fix**: Remove the `name` parameter from `check_vec` and all call sites (`inner_product`, `gram_vec`, `directional_drift`).

---

### 19. `is_multiple_of` is nightly-only — breaks stable Rust

**Severity**: Medium — portability

`crates/got-attest/src/lib.rs` line 253 in `merkle_root`:
```rust
if !nodes.len().is_multiple_of(2) {
```

`usize::is_multiple_of` is gated behind `#![feature(unsigned_is_multiple_of)]`
and is not available on stable Rust. The code compiles on nightly but will fail
on any stable toolchain.

**Fix**: Replace with `nodes.len() % 2 != 0`.

---

### 20. `MockEnclave::attest` and `attest_with_causal` share ~40 duplicate lines

**Severity**: Low — maintainability

Both methods in `crates/got-enclave/src/enclave.rs` contain identical code for:
collecting layers, reading probes, computing `divergence_flag`, building
`input_hash`, and getting the timestamp. Only the causal fields differ.

**Fix**: Extract a `build_attestation_base()` helper that returns a partially
filled `GeometricAttestation`, then have both methods fill in the
causal-specific fields and sign. Low priority (PoC mock code).

---

### 21. Hex encode/decode logic reimplemented in 3 crates

**Severity**: Low — code duplication (non-blocking)

Hex conversion appears in:
- `got_core::hex32` / `hex64` / `optional_hex32` (serde modules)
- `got_wire::registry::parse_hex_32`
- `got_store::file_store::{hex_encode, hex_decode}`

All use the same `format!("{b:02x}")` / `u8::from_str_radix` pattern with
minor variations.

**Fix**: Consider adding a shared `hex` utility module in `got-core` or
adopting the `hex` crate. Low priority — current code is correct.

---

## Summary

| # | Severity | Component | Issue | Status |
|---|----------|-----------|-------|--------|
| 14 | **High** | got-probe | `UnembeddingMatrix::new` missing `.unwrap()` in 3 test helpers | ✅ Fixed |
| 15 | **High** | got-attest | `merkle_root_empty` test asserts stale value | ✅ Fixed |
| 16 | Low | got-wire | Duplicate `agent_id` / `compute_agent_id` functions | ✅ Fixed |
| 17 | Low | got-attest/cli | `got_attest::sha256` trivial re-export | ✅ Fixed |
| 18 | Low | got-core | Unused `name` param in `check_vec` | ✅ Fixed |
| 19 | Medium | got-attest | `is_multiple_of` is nightly-only | ✅ Fixed |
| 20 | Low | got-enclave | `attest` / `attest_with_causal` code duplication | ✅ Fixed |
| 21 | Low | multiple | Hex encode/decode reimplemented 3 times | Won't fix — private crate-local utils |

---

## Security Audit (Issues 22–42)

Security-focused review of the entire codebase, threat-modelled as a trust
protocol where adversaries are model operators trying to forge, replay, or
dilute attestations.

---

### 22. `verify()` returns `Ok(false)` instead of `Err` for invalid signatures (S-1)

**Severity**: Critical — callers using `?` silently accept forgeries

`got_attest::verify()` returns `Ok(false)` on bad signatures. The `?` operator
passes through `Ok(false)` silently — a careless consumer could treat forged
attestations as valid. The conventional crypto API returns `Err` on failure so
misuse is structurally impossible.

**Affected**: `got-store`, `got-wire`, `got-cli` (all check the bool today but
the API is a latent trap).

**Fix**: Change `verify()` to `Result<(), AttestationError>` returning
`Err(SignatureInvalid)` on failure. Update all call sites.

**Test**: `sec_verify_returns_err_on_bad_signature` — tamper with an attestation
and call `verify()`; assert that it returns `Err(SignatureInvalid)`, not
`Ok(false)`.

---

### 23. Trust registry has no integrity protection (S-2)

**Severity**: Critical — root of trust is unauthenticated

`TrustRegistry::load(path)` reads a TOML file with no signature, MAC, or hash
check. Filesystem compromise = silent trust compromise.

**Fix**: Add `TrustRegistry::load_verified(path, expected_hash)` that
recomputes SHA-256 of the file and rejects mismatches. Deprecate bare `load()`.

**Test**: `sec_registry_load_verified_rejects_tampered` — write registry to
temp file, compute hash, append a byte, call `load_verified` and assert
`Err`.

---

### 24. Nonce / keygen uses `thread_rng()` not `OsRng` (S-3)

**Severity**: High — non-guaranteed CSPRNG quality for security-critical randomness

Exchange nonces (`perform_exchange`) and key generation (`cmd_keygen`) use
`rand::thread_rng()`. The contract does NOT guarantee cryptographic quality on
all targets.

**Fix**: Replace with `rand::rngs::OsRng` for nonces and keygen. Probe sampling
may keep `thread_rng()`.

**Test**: `sec_nonce_is_unique_across_calls` — generate 100 nonces, assert all
32-byte values are unique (statistical test for CSPRNG output).

---

### 25. `from_raw_gram()` performs no validation (S-4)

**Severity**: High — crafted geometry checkpoint can corrupt pipeline

`CausalGeometry::from_raw_gram()` accepts any `Vec<f32>` without checking
symmetry, PSD, or NaN/Infinity.

**Fix**: Validate symmetry (`|Φ[i,j] - Φ[j,i]| < ε`), reject NaN/Infinity,
optionally check PSD via Cholesky.

**Tests**:
- `sec_from_raw_gram_rejects_nan` — pass NaN in gram → `Err`
- `sec_from_raw_gram_rejects_infinity` — pass Infinity → `Err`
- `sec_from_raw_gram_rejects_asymmetric` — pass non-symmetric matrix → `Err`

---

### 26. `geometry_hash()` does not bind epsilon (S-5)

**Severity**: High — different regularisations hash identically

Two geometries with identical gram data but different epsilon (e.g. 1e-6 vs
1e-2) produce the same `geometry_hash`, yet compute different inner products.
Attacker can claim drift = 0 while using a differently-regularised geometry.

**Fix**: Include `self.epsilon.to_le_bytes()` in the hash computation.

**Test**: `sec_geometry_hash_includes_epsilon` — build two geometries from same
gram data but different epsilon, assert their hashes differ.

---

### 27. FileStore vulnerable to path traversal / non-atomic writes (S-6)

**Severity**: High — filesystem race / crash can corrupt store

FileStore does not use atomic writes (`write` then `rename`), does not set
restrictive permissions, and does not verify `root` is a real directory.

**Fix**: Write via temp + rename. Set 0o600 on files, 0o700 on dirs. Verify
root is not a symlink.

**Test**: `sec_file_store_atomic_write_survives_incomplete` — write an
attestation, verify index.json is consistent after reopen (exercises the
atomic write path).

---

### 28. No attestation timestamp freshness during signing (S-7)

**Severity**: High — pre-computed / backdated attestations possible

`assemble_and_sign()` does not validate that `timestamp` is sane. An attacker
can pre-compute attestations with future timestamps or backdate them.

**Fix**: Reject timestamps more than e.g. 300s in the future in
`assemble_and_sign()`.

**Test**: `sec_assemble_rejects_future_timestamp` — set timestamp = now +
3600, call `assemble_and_sign`, assert `Err`.

---

### 29. Chain verification does not support key rotation (S-8)

**Severity**: High — compromised key cannot be revoked

`verify_chain()` now accepts `signer_pks: &[VerifyingKey]` — a slice of
trusted keys. Each link is verified against all provided keys; any match
accepts the link. This supports key rotation by passing both old and new keys.

**Test**: `sec_verify_chain_key_rotation` — builds a chain signed by two
different keys and verifies: single-key rejection, both-key acceptance, and
empty-slice rejection.

---

### 30. `ExchangeEnvelope::from_bytes()` performs no validation (S-9) — ✅ Fixed

**Severity**: Medium — unverified envelope can enter protocol logic

`from_bytes()` deserialises arbitrary bytes into an envelope with no field
validation. Verify must be called separately but nothing enforces this.

**Fix**: Added private `verified: bool` field to `ExchangeEnvelope`.
`create()` sets `verified = true`; `from_bytes()` sets `verified = false`.
New `from_bytes_verified(data, …verify params…)` atomically parses + verifies,
returning `Err(WireError)` on failure. `is_verified()` accessor available.

**Test**: `sec_envelope_from_bytes_verified_gate` — validates all four
scenarios: create → verified, from_bytes → unverified, from_bytes_verified
rejects tampered signature, from_bytes_verified accepts valid.

---

### 31. Sidecar silently skips probe read errors (S-10)

**Severity**: Medium — attacker can selectively omit unfavourable readings

In `MeasurementSidecar::ingest()`, `Err(_) => continue` silently drops
probes that error. No count, no flag, no diagnostic.

**Fix**: Track `skipped_probe_count` in the sidecar. Set
`divergence_flag = true` if any probes were skipped.

**Test**: `sec_sidecar_flags_skipped_probes` — feed activations with
wrong dimension to one probe; verify attestation has `divergence_flag = true`
or `skipped_count > 0`.

---

### 32. Sidecar `input_hash` is SHA-256 of window index only (S-11)

**Severity**: Medium — attestation not bound to actual input data

`MeasurementSidecar` hashes `window_index.to_le_bytes()` for `input_hash`,
not the actual activation data. Two different inputs at same window index
produce identical `input_hash`.

**Fix**: Hash activation values with domain separation, matching the enclave's
`compute_input_hash()`.

**Test**: `sec_sidecar_input_hash_depends_on_activation_data` — run two
sidecar windows with different activations at same window index, assert
their `input_hash` values differ.

---

### 33. Rank check uses trace heuristic, may miss rank deficiency (S-12)

**Severity**: Medium — numerically meaningless probe readings possible

`is_positive_definite()` uses trace-based heuristic. A crafted matrix with
large trace but near-zero eigenvalues can fool it.

**Fix**: Use Cholesky factorisation to definitively check PSD.

**Test**: `sec_rank_check_detects_degenerate_with_large_trace` — construct
a matrix with large trace but a zero eigenvalue (e.g.
`diag(1000, 0)` → trace=1000 but rank 1), call
`is_positive_definite()`, assert false.

---

### 34. No bounds on string fields in attestation (S-13)

**Severity**: Medium — DoS via multi-GB model_id

`GeometricAttestation` string fields (`model_id`, `corpus_version`,
`probe_version`) have no length limits. `assemble_and_sign` and
`serialise_for_signing` faithfully process them.

**Fix**: Enforce max length (e.g. 256 bytes) in `assemble_and_sign()`.

**Test**: `sec_assemble_rejects_oversized_model_id` — create attestation with
1 MB `model_id`, call `assemble_and_sign`, assert `Err`.

---

### 35. `SigningKey` not zeroized — `zeroize` feature missing (S-14)

**Severity**: Medium — key material lingers in memory

`ed25519-dalek` dependency is configured with `features = ["rand_core"]` but
not `"zeroize"`. The `SigningKey`'s `Drop` impl only zeroizes when the feature
is enabled.

**Fix**: Add `"zeroize"` to ed25519-dalek features in all crate Cargo.toml
files.

**Test**: `sec_ed25519_dalek_has_zeroize_feature` — compile-time check: assert
`SigningKey` implements `Zeroize` trait (only true when feature is active).

---

### 36. `SystemTime::now()` not monotonic, can be manipulated (S-15)

**Severity**: Medium — clock tampering bypasses freshness checks

All timestamp generation uses `SystemTime::now()`. An attacker controlling the
system clock can bypass `max_envelope_age_secs` and freshness checks.

**Fix**: Document clock integrity assumption. In enclave path, use TEE trusted
clock.

**Test**: `sec_validate_request_rejects_stale_attestation` — build request
with old timestamp, validate with a known `now`, assert rejection. (Already
partially tested; this hardens the explicit age check.)

---

### 37. FileStore does not re-verify hashes on load (S-16)

**Severity**: Medium — tampered JSON on disk loaded silently

`FileStore::open()` rebuilds cache from disk but never recomputes
`attestation_store_id` against the filename hash. Tampering goes undetected.

**Fix**: On load, recompute hash from deserialised attestation and compare to
filename. Reject mismatches.

**Test**: `sec_file_store_rejects_tampered_json_on_reload` — write attestation,
manually edit JSON on disk, reopen store, assert error or missing attestation.

---

### 38. `decode_error` uses `from_utf8_lossy` (S-17)

**Severity**: Medium — malformed UTF-8 can inject replacement chars

`decode_error()` uses `String::from_utf8_lossy()` which silently replaces
invalid sequences with U+FFFD. An attacker could exploit this for log
injection.

**Fix**: Use `String::from_utf8()` and return `Err` on invalid UTF-8.

**Test**: `sec_decode_error_rejects_invalid_utf8` — construct error payload
with invalid UTF-8 bytes, call `decode_error`, assert `Err`.

---

### 39. `perform_exchange()` nonce quality (S-18)

**Severity**: Low — covered by S-3 (thread_rng → OsRng)

Once S-3 is fixed (issue #24), nonces in `perform_exchange` will be
cryptographic quality. No additional fix needed.

**Test**: Covered by issue #24 test.

---

### 40. `MockEnclave` provides no real isolation (S-19)

**Severity**: Low — expected PoC limitation, already documented

The mock runs in the same address space. The signing key, probes, and
activations are all host-visible. This is fine for testing but provides zero
additional trust in production.

**Fix**: Gate `MockEnclave` behind `#[cfg(any(test, feature = "mock"))]` so
it cannot appear in release builds by accident.

**Test**: `sec_mock_enclave_not_in_release` — `#[cfg(not(any(test, feature = "mock")))]`
compile-time assertion that `MockEnclave` is not importable. (Expressed as a
doc comment / feature gate; no runtime test needed.)

---

### 41. No maximum on array sizes in attestation (S-20)

**Severity**: Low — DoS via millions of readings

`serialise_for_signing()` and `assemble_and_sign()` process unbounded arrays.
Multi-million reading attestations cause excessive CPU/memory.

**Fix**: Enforce max layers (e.g. 1024) and max total readings (e.g. 65536) in
`assemble_and_sign()`.

**Test**: `sec_assemble_rejects_oversized_layer_readings` — create attestation
with 2000 layers, call `assemble_and_sign`, assert `Err`.

---

### 42. `model_hash` sentinel `[0xFF; 32]` is ambiguous (S-21)

**Severity**: Low — verifier cannot distinguish "not provided" from real hash

`model_hash` is now `Option<[u8; 32]>` in `GeometricAttestation`. The CLI sets
`None` when `--shards` is not provided, replacing the ambiguous `[0xFF; 32]`
sentinel. The signing payload uses a tag byte to distinguish `None` from
`Some([0u8; 32])`.

**Test**: `sec_model_hash_option_none_is_distinct` — confirms `None` and
`Some([0; 32])` round-trip correctly and produce distinct attestation hashes.

---

### 43. `Frame::encode()` does not validate payload length before `as u32` cast (N-1) — ✅ Fixed

**Severity**: Low — silent truncation if payload > 4 GiB

`Frame::encode()` casts `self.payload.len() as u32` without verifying the
length fits in u32 or is within `MAX_PAYLOAD_SIZE`. While `decode()` checks
`MAX_PAYLOAD_SIZE` on the receiving side, `encode()` has no corresponding check.
A payload exceeding `u32::MAX` would silently truncate the length field.

**Fix**: `encode()` now returns `Result<Vec<u8>, WireError>`. Uses `try_into()`
for the u32 conversion and rejects payloads > `MAX_PAYLOAD_SIZE` with
`WireError::PayloadTooLarge`.

**Test**: `sec_frame_encode_rejects_oversized_payload` — creates a frame with
`MAX_PAYLOAD_SIZE + 1` bytes and asserts `encode()` returns
`Err(WireError::PayloadTooLarge { .. })`.

---

### 44. `CollectingHook` Mutex poisoning causes panic (N-2) — ✅ Fixed

**Severity**: Low — thread panic propagation

`CollectingHook` uses `self.buffer.lock().unwrap()` in 3 places (`drain()`,
`len()`, `on_activation()`). If a thread panics while holding the lock, the
Mutex becomes poisoned and all subsequent calls will panic, potentially crashing
the host process.

**Fix**: All 3 call sites now use `lock().unwrap_or_else(|e| e.into_inner())`
to recover from mutex poisoning instead of panicking.

**Test**: `sec_collecting_hook_survives_mutex_poison` — intentionally poisons
the mutex from another thread, then verifies the hook continues to accept
activations and drain them successfully.

---

### 45. CLI uses `expect()`/`panic!()` for all error handling (N-3) — ✅ Fixed

**Severity**: Info — poor UX, stack trace leaks internal paths

`got-cli/src/main.rs` uses `expect()` and `panic!()` exclusively (~40 call
sites) for error handling. No `Result` propagation. This produces stack traces
on user errors instead of friendly messages and may expose internal file paths.

**Fix**: Migrated to `anyhow::Result`. `main()` returns `Result<()>`. All
subcommand functions (`cmd_keygen`, `cmd_train`, `cmd_attest`, `cmd_verify`,
`cmd_checkpoint`, `cmd_drift`) and I/O helpers now return `anyhow::Result`.
All `expect()` → `.context("...")? `, all `panic!()` → `bail!()`. Errors now
print a clean message without stack traces.

---

## Feature Tasks — Production Readiness

Tasks that are implementable now with no external hardware or institutional
dependencies. Ordered by impact.

---

### 46. CLI `train` subcommand: add `--validation-labels` for Platt calibration (F-1)

**Severity**: Feature — unlocks calibrated confidence values

`train_probe_calibrated()` and `fit_platt()` already exist in `got-probe`
([lib.rs:164](crates/got-probe/src/lib.rs)) and are tested. The CLI `train`
subcommand ([main.rs:268](crates/got-cli/src/main.rs)) only calls the
uncalibrated `train_probe()`.

**What to do**:

1. Add `--validation-labels <path>` optional argument to the `Train` variant
   in `got-cli/src/main.rs`.
2. When provided, load the validation labels file (same format as `--labels`).
3. Split activations into training set (matched by `--labels`) and validation
   set (matched by `--validation-labels`). Ensure the two sets are disjoint
   (compare activation indices or token positions).
4. Call `train_probe_calibrated()` instead of `train_probe()`, passing
   `--lr` / `--epochs` for training and new `--platt-lr` / `--platt-epochs`
   args (default 0.01 / 200) for the Platt fitting pass.
5. Print the fitted `platt_scale` and `platt_shift` values so the user can
   inspect them.

**Acceptance**:
- `cargo run -p got-cli -- train --validation-labels val.labels ...` produces
  a `ProbeSet` with non-default `platt_scale`/`platt_shift`.
- Without `--validation-labels`, behaviour is unchanged (uncalibrated).

**Effort**: ~1 hour.

---

### 47. Expected Calibration Error (ECE) metric (F-2)

**Severity**: Feature — measures whether confidence values are trustworthy

Once Platt calibration is wired up (issue #46), there is no way to evaluate
*how well* the calibration worked. ECE is the standard metric: bin predictions
by confidence, compare average confidence to actual accuracy per bin.

**What to do**:

1. Add `pub fn expected_calibration_error(predictions: &[(f32, bool)], bins: usize) -> f32`
   to `got-probe/src/lib.rs`. Each entry is `(calibrated_confidence, true_label)`.
   - Bucket by confidence into `bins` equal-width bins.
   - Per bin: ECE contribution = `|avg_confidence − accuracy| × (bin_count / total)`.
   - Return the weighted sum.
2. Add a CLI subcommand `calibration-report` (or extend `train` output) that,
   given a probe set and held-out labelled activations, prints per-bin accuracy
   vs. confidence and the overall ECE score.

**Acceptance**:
- Unit test: perfectly calibrated predictions → ECE ≈ 0.
- Unit test: all-confident-but-wrong predictions → ECE ≈ 1.
- CLI prints a human-readable calibration table.

**Effort**: ~2 hours.

---

### 48. Agent certificate structure for PKI (F-3)

**Severity**: Feature — replaces raw public keys with verifiable identity chain

The current `TrustRegistry` maps raw public keys to agent entries. There is no
way to verify *who issued* a key or *when it expires*. This is the first step
toward a PKI.

**What to do**:

1. Create a new file `crates/got-wire/src/certificate.rs`.
2. Define an `AgentCertificate` struct:
   ```rust
   pub struct AgentCertificate {
       pub subject_name: String,
       pub subject_public_key: VerifyingKey,
       pub issuer_public_key: VerifyingKey,
       pub not_before: u64,      // Unix timestamp
       pub not_after: u64,       // Unix timestamp
       pub roles: Vec<String>,
       pub max_drift_accepted: f32,
       pub expected_model_hash: Option<[u8; 32]>,
       pub signature: [u8; 64],  // Ed25519 over canonical fields
   }
   ```
3. Implement `sign_certificate(subject, issuer_signing_key) -> AgentCertificate`
   with deterministic canonical serialisation (reuse the length-prefixed LE
   pattern from `serialise_for_signing`).
4. Implement `verify_certificate(cert, issuer_public_key) -> Result<(), WireError>`.
5. Implement `is_valid_at(cert, now_unix) -> bool` for expiry checking.
6. Add serde support (JSON with hex-encoded keys/signature).

**Acceptance**:
- Round-trip: issue cert → serialise → verify succeeds.
- Tampered cert → verify returns `Err`.
- Expired cert → `is_valid_at` returns `false`.

**Effort**: ~4 hours.

---

### 49. Certificate-aware trust registry (F-4)

**Severity**: Feature — integrates certificates into the exchange protocol

Depends on issue #48.

**What to do**:

1. Add `pub certificate: Option<AgentCertificate>` to `AgentEntry`.
2. Extend `TrustRegistry::from_toml()` to parse an optional
   `certificate = "path/to/cert.json"` field per agent. Load and verify the
   certificate against a configured CA public key.
3. Add `pub ca_public_keys: Vec<VerifyingKey>` to `TrustRegistry` — one or
   more CA root keys. Registry rejects agent entries whose certificates are
   not signed by a known CA.
4. In `validate_request()` and `validate_response()` (exchange.rs), if the
   registry entry has a certificate, check `is_valid_at(cert, now)` and
   reject expired certificates.
5. Extend TOML format:
   ```toml
   [registry]
   ca_public_keys = ["hex...", "hex..."]

   [[agents]]
   name = "alice"
   certificate = "certs/alice.json"
   ```

**Acceptance**:
- Exchange succeeds with valid certificate.
- Exchange rejects agent with expired certificate.
- Exchange rejects agent whose certificate is not signed by a known CA.
- Backward-compatible: agents without certificates still work (PoC mode).

**Effort**: ~4 hours.

---

### 50. Key rotation ceremony (F-5)

**Severity**: Feature — allows agents to rotate keys without breaking chains

Depends on issue #48. The chain verifier (issue #29) already accepts multiple
signer keys. This formalises the rotation process.

**What to do**:

1. Add a `KeyRotation` struct to `got-wire/src/certificate.rs`:
   ```rust
   pub struct KeyRotation {
       pub old_public_key: VerifyingKey,
       pub new_public_key: VerifyingKey,
       pub new_certificate: AgentCertificate,
       pub timestamp: u64,
       pub old_key_signature: [u8; 64],  // old key signs new key
       pub new_key_signature: [u8; 64],  // new key signs old key (mutual proof)
   }
   ```
2. Implement `create_rotation(old_sk, new_sk, issuer_sk) -> KeyRotation`.
3. Implement `verify_rotation(rotation) -> Result<(), WireError>` — checks
   both cross-signatures and the certificate.
4. Add `TrustRegistry::apply_rotation(rotation)` — verifies, then updates the
   agent entry to use the new key while retaining the old key in a
   `previous_keys: Vec<VerifyingKey>` list for chain verification of old
   attestations.
5. Add a CLI subcommand `rotate-key`:
   ```
   got-cli rotate-key --old-key data/key --new-key data/key2 \
       --ca-key data/ca.key --output data/rotation.json
   ```

**Acceptance**:
- Rotation round-trips: create → verify succeeds.
- Chain verification still works for attestations signed by the old key.
- New attestations must use the new key.
- Rotation without valid cross-signatures → rejected.

**Effort**: ~6 hours.

---

### 51. CLI `keygen` for CA root key + `issue-cert` subcommand (F-6)

**Severity**: Feature — completes the minimal PKI toolchain

Depends on issue #48.

**What to do**:

1. Extend `keygen` with `--ca` flag. When set, adds a `ca=true` marker to the
   public key file header (or a separate metadata file) so the CLI can
   distinguish CA keys from agent keys.
2. Add `issue-cert` subcommand:
   ```
   got-cli issue-cert \
       --ca-key data/ca.key \
       --subject-pubkey data/alice.pub \
       --subject-name "alice" \
       --roles producer,verifier \
       --validity-days 365 \
       --output data/alice-cert.json
   ```
3. The output is a signed `AgentCertificate` JSON file that can be referenced
   from registry TOML (issue #49).

**Acceptance**:
- `issue-cert` produces a cert verifiable by the CA's public key.
- Cert with expired `validity-days` is rejected by the registry.

**Effort**: ~2 hours.

---

### 52. Certificate revocation list (CRL) support (F-7)

**Severity**: Feature — allows revoking compromised agent keys

Depends on issues #48–49.

**What to do**:

1. Define `CertificateRevocationList` struct:
   ```rust
   pub struct CertificateRevocationList {
       pub issuer: VerifyingKey,
       pub entries: Vec<RevokedEntry>,  // agent_id + revocation_time + reason
       pub issued_at: u64,
       pub next_update: u64,
       pub signature: [u8; 64],
   }
   ```
2. Add `TrustRegistry::load_crl(crl) -> Result<(), WireError>` — verifies CRL
   signature against a CA key, then marks matching agents as revoked.
3. Exchange validation rejects agents whose certificates appear in a loaded CRL.
4. CLI subcommand `revoke`:
   ```
   got-cli revoke --ca-key data/ca.key --agent data/alice-cert.json \
       --reason key-compromise --output data/crl.json
   ```

**Acceptance**:
- Revoked agent's exchange attempts are rejected.
- Non-revoked agents unaffected.
- CRL with invalid CA signature → rejected.

**Effort**: ~4 hours.

---

## Summary (Full)

| # | Severity | Component | Issue | Status |
|---|----------|-----------|-------|--------|
| 1–13 | Various | Various | Original spec violations | ✅ Fixed |
| 14–20 | Various | Various | Code hygiene sweep | ✅ Fixed |
| 21 | Low | multiple | Hex duplication | Won't fix |
| 22 | **Critical** | got-attest | `verify()` API returns `Ok(false)` for bad sigs | ✅ Fixed |
| 23 | **Critical** | got-wire | Trust registry unauthenticated | ✅ Fixed |
| 24 | **High** | got-wire/cli | Nonce/keygen uses `thread_rng` not `OsRng` | ✅ Fixed |
| 25 | **High** | got-core | `from_raw_gram()` no validation | ✅ Fixed |
| 26 | **High** | got-core | `geometry_hash()` doesn't bind epsilon | ✅ Fixed |
| 27 | **High** | got-store | FileStore non-atomic writes, no perms | ✅ Fixed |
| 28 | **High** | got-attest | No timestamp freshness during signing | ✅ Fixed |
| 29 | **High** | got-wire | Chain verification: no key rotation | ✅ Fixed (S-8) |
| 30 | Medium | got-wire | `from_bytes()` envelope unvalidated | ✅ Fixed (S-9) |
| 31 | Medium | got-probe | Sidecar silently skips probe errors | ✅ Fixed |
| 32 | Medium | got-probe | Sidecar `input_hash` ≠ actual data | ✅ Fixed |
| 33 | Medium | got-core | Rank check trace heuristic | ✅ Fixed |
| 34 | Medium | got-attest | Unbounded string fields | ✅ Fixed |
| 35 | Medium | all | `ed25519-dalek` missing `zeroize` feature | ✅ Fixed |
| 36 | Medium | got-wire | `SystemTime::now()` clock trust | ✅ Fixed |
| 37 | Medium | got-store | FileStore no hash-on-load | ✅ Fixed |
| 38 | Medium | got-wire | `decode_error` lossy UTF-8 | ✅ Fixed |
| 39 | Low | got-wire | `perform_exchange` nonce (→ #24) | ✅ Fixed |
| 40 | Low | got-enclave | `MockEnclave` no real isolation | ✅ Fixed |
| 41 | Low | got-attest | Unbounded array sizes | ✅ Fixed |
| 42 | Low | got-cli | `model_hash` sentinel ambiguous | ✅ Fixed (S-21) |
| 43 | Low | got-wire | `Frame::encode()` no payload length validation | ✅ Fixed (N-1) |
| 44 | Low | got-probe | `CollectingHook` Mutex poison panic | ✅ Fixed (N-2) |
| 45 | Info | got-cli | CLI uses `expect()`/`panic!()` for all errors | ✅ Fixed (N-3) |
| 46 | Feature | got-cli | CLI `--validation-labels` for Platt calibration | ✅ Fixed |
| 47 | Feature | got-probe/cli | Expected Calibration Error (ECE) metric | ✅ Fixed |
| 48 | Feature | got-wire | Agent certificate structure for PKI | ✅ Fixed |
| 49 | Feature | got-wire | Certificate-aware trust registry | ✅ Fixed |
| 50 | Feature | got-wire/cli | Key rotation ceremony | ✅ Fixed |
| 51 | Feature | got-cli | CA keygen + `issue-cert` subcommand | ✅ Fixed |
| 52 | Feature | got-wire/cli | Certificate revocation list (CRL) | ✅ Fixed |
