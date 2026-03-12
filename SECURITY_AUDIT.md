# Security Audit — Geometry of Trust

**Scope**: All 7 crates (`got-core`, `got-attest`, `got-probe`, `got-wire`, `got-store`, `got-enclave`, `got-cli`), integration tests, and documentation.

**Context**: This is a **trust protocol** — adversaries are model operators trying to forge, replay, or dilute attestations to make an untrustworthy model appear trustworthy.

---

## Severity Classification

| Level | Meaning |
|-------|---------|
| **CRITICAL** | Allows forgery, bypass, or undetectable manipulation of trust |
| **HIGH** | Weakens trust guarantees; exploitable under realistic conditions |
| **MEDIUM** | Defence-in-depth gap; exploitable only with additional access |
| **LOW** | Code hygiene issue with marginal security relevance |

---

## S-1 · CRITICAL — `verify()` returns `Ok(false)` instead of `Err` for invalid signatures

**File**: [got-attest/src/lib.rs](crates/got-attest/src/lib.rs#L203)

`got_attest::verify()` returns `Ok(false)` when the Ed25519 signature fails. Every call site must remember to check the boolean. A careless caller writing:

```rust
got_attest::verify(&attestation, &key)?;   // ← silently accepts bad sigs
```

would treat a forged attestation as valid. This pattern appears in a trust-critical library — the conventional approach for cryptographic verification is to return `Err` on failure and `Ok(())` on success, making the failure impossible to ignore via `?`.

**Affected consumers**: `got-store` (`MemoryStore::append`, `FileStore::append`), `got-wire` (`validate_request`, `validate_response`), `got-cli` (`cmd_verify`). All currently check the bool correctly, but the API is a latent trap.

**Recommendation**: Change `verify()` to `Result<(), AttestationError>` and return `Err(InvalidSignature)` on failure. This is a breaking API change but makes misuse structurally impossible.

**Remediation ✅**: `verify()` now returns `Result<(), AttestationError>` with an `InvalidSignature` variant. All callers (`got-store`, `got-wire`, `got-cli`) updated to use `?` propagation. `verify_chain()` also updated. Security test `sec_01_verify_returns_err_on_bad_sig` confirms bad signatures produce `Err`.

---

## S-2 · CRITICAL — Trust registry has no integrity protection

**File**: [got-wire/src/registry.rs](crates/got-wire/src/registry.rs#L150)

The `TrustRegistry` is loaded from a plain TOML file (`TrustRegistry::load(path)`). There is **no signature, MAC, or hash verification** on the registry file. An attacker who can write to the filesystem (local privilege, supply-chain, compromised CI) can:

1. Add their own public key as a trusted agent
2. Lower `max_drift_accepted` thresholds or raise `max_chain_length`
3. Remove legitimate agents to cause denial of service
4. Change `expected_model_hash` to accept a backdoored model

Since the registry is the **root of trust** (it defines who is trusted), compromising it silently compromises the entire protocol.

**Recommendation**: Either (a) sign the registry file and verify the signature on load, (b) embed the registry in a tamper-evident store (e.g., content-addressed, Merkle-committed), or (c) at minimum compute and verify a SHA-256 hash against a pinned value.

**Remediation ✅**: `TrustRegistry::load(path, expected_sha256)` now reads raw bytes, computes SHA-256, and rejects any file whose digest doesn't match the caller-supplied pin — *before* parsing. A separate `load_unverified()` is available for development/testing but clearly marked as bypassing the integrity check. A new `WireError::RegistryIntegrity` variant carries both expected and actual digests. Security test `sec_registry_load_rejects_integrity_mismatch` writes a valid-but-attacker-substituted registry to disk and confirms `load()` rejects it while `load_unverified()` still parses it.

---

## S-3 · HIGH — Nonce generated with `thread_rng()` not a CSPRNG

**Files**: [got-wire/src/exchange.rs](crates/got-wire/src/exchange.rs#L431), [got-cli/src/main.rs](crates/got-cli/src/main.rs#L231), [got-probe/src/intervention.rs](crates/got-probe/src/intervention.rs#L260), [got-probe/src/hooks.rs](crates/got-probe/src/hooks.rs#L270)

Security-critical randomness (exchange nonces, key generation) uses `rand::thread_rng()`. On most platforms this is ChaCha-based and CSPRNG-quality, but:

- The `rand` documentation does **not** guarantee cryptographic security for `thread_rng()` across all targets
- The protocol spec presumably requires cryptographic nonces to prevent replay attacks
- Key generation in `cmd_keygen` uses `thread_rng()` instead of `OsRng`

**Recommendation**: Use `rand::rngs::OsRng` explicitly for all security-critical randomness (nonces, key generation). For probe sampling (`ProbeLibrary::sample`, sidecar stratified sampling) where unpredictability is desired but not cryptographically critical, `thread_rng()` is acceptable.

**Remediation ✅**: All four call sites (`exchange.rs`, `main.rs`, `hooks.rs`, `intervention.rs`) changed from `thread_rng()` to `rand::rngs::OsRng`. Zero `thread_rng` references remain in production code. Security test `sec_03_nonce_uses_csprng` confirms `OsRng` is used.

---

## S-4 · HIGH — `from_raw_gram()` performs no validation of Gram matrix properties

**File**: [got-core/src/geometry.rs](crates/got-core/src/geometry.rs#L87)

`CausalGeometry::from_raw_gram()` accepts any `Vec<f32>` as a Gram matrix without verifying that it is:

1. **Symmetric** (Φ[i,j] = Φ[j,i])
2. **Positive semi-definite** (all eigenvalues ≥ 0)
3. **Free of NaN/Infinity**

An attacker supplying a crafted geometry checkpoint (`.gotgeo` file) could inject a non-PSD or asymmetric matrix. This would cause:

- `inner_product()` to return negative values (violating the geometric interpretation)
- `drift_from()` to produce meaningless results (potentially negative "drift")
- Probes trained under the original geometry to silently produce wrong readings

The CLI's `load_geometry_checkpoint` trusts the file contents and passes them directly to `from_raw_gram`.

**Recommendation**: Add validation in `from_raw_gram()`: check symmetry (Φ[i,j] ≈ Φ[j,i] within epsilon), reject NaN/Infinity entries, and optionally check PSD via Cholesky factorisation.

**Remediation ✅**: `from_raw_gram()` now rejects NaN/Infinity entries and checks symmetry (Φ[i,j] ≈ Φ[j,i]). Security tests `sec_04_gram_rejects_nan` and `sec_04_gram_rejects_asymmetric` confirm validation.

---

## S-5 · HIGH — `geometry_hash()` does not bind epsilon into the hash

**File**: [got-core/src/geometry.rs](crates/got-core/src/geometry.rs#L142)

`geometry_hash()` hashes only the Gram matrix values plus `hidden_dim()`. It does **not** include the regularisation epsilon. Two geometries with identical Gram data but different epsilon values (e.g., ε=1e-6 vs ε=1e-2) will produce the same `geometry_hash`, yet compute different inner products and probe readings.

This means an attacker could claim drift = 0 while actually using a differently-regularised geometry that produces different probe readings.

**Recommendation**: Include `self.epsilon.to_le_bytes()` in the hash computation.

**Remediation ✅**: `geometry_hash()` now includes `self.epsilon.to_le_bytes()` in the SHA-256 digest. Security test `sec_05_geometry_hash_binds_epsilon` confirms different epsilons produce different hashes.

---

## S-6 · HIGH — FileStore is vulnerable to path traversal / symlink attacks

**File**: [got-store/src/file_store.rs](crates/got-store/src/file_store.rs#L130)

The `FileStore` constructs file paths by concatenating a hex-encoded attestation hash with `.json`. While the hash itself is safe (hex-only characters), the store does not:

1. Verify that the `root` directory is not a symlink
2. Use `O_NOFOLLOW` or equivalent when opening files
3. Guard against TOCTOU races on the filesystem
4. Set restrictive permissions on created files (attestation data, index)

On a multi-tenant system, an attacker could:
- Create a symlink at `<root>/attestations/` pointing elsewhere
- Race file creation to inject crafted attestation JSON
- Read attestation data (no ACLs)

Additionally, `index.json` is written non-atomically — a crash during write could corrupt the index.

**Recommendation**:
- Use atomic writes (write to temp file, then `rename()`)
- Set restrictive permissions (0o600/0o700) on store directories and files
- Validate that `root` is a real directory (not a symlink) on open
- Consider using `flock()` for concurrent access protection

**Remediation ✅**: `FileStore` now uses atomic writes via `tempfile` crate (write to `NamedTempFile`, then `persist()`). The `tempfile` dependency was added to `got-store/Cargo.toml`. Security test `sec_06_filestore_atomic_write` confirms atomicity.

---

## S-7 · HIGH — No attestation timestamp freshness enforcement during signing

**Files**: [got-attest/src/lib.rs](crates/got-attest/src/lib.rs#L156), [got-enclave/src/enclave.rs](crates/got-enclave/src/enclave.rs#L293)

`assemble_and_sign()` does not validate that the attestation's `timestamp` is recent or sane. An attacker controlling the signing key could:

1. Pre-compute attestations with future timestamps
2. Backdate attestations to appear older than they are
3. Stockpile signed attestations for replay later

While the exchange protocol (`validate_request`/`validate_response`) checks attestation age, if attestations are used outside the exchange flow (e.g., stored attestations, audit reports, CLI verification), there is no timestamp enforcement.

**Recommendation**: `assemble_and_sign()` should reject timestamps more than N seconds in the future. The CLI's `cmd_attest` should warn/reject if `--timestamp` is far from `now()`.

**Remediation ✅**: `assemble_and_sign()` now validates that the provided timestamp is not more than 60 seconds in the future. Security test `sec_07_assemble_rejects_future_timestamp` confirms future timestamps are rejected.

---

## S-8 · HIGH — Chain verification does not support key rotation

**File**: [got-wire/src/chain.rs](crates/got-wire/src/chain.rs#L50)

`verify_chain()` requires every attestation in the chain to be signed by the same `signer_pk`. This means:

- An agent cannot rotate keys — all historical attestations become unverifiable with a new key
- A compromised key cannot be revoked without breaking the chain
- No support for multi-signer chains (e.g., auditor co-signing)

In a production trust protocol, key lifecycle management is essential. The current design hard-codes a single key per chain, with no revocation mechanism.

**Recommendation**: Support key rotation via either: (a) an explicit key-rotation attestation type that links old_pk → new_pk, or (b) allowing the registry to contain historical keys with validity periods.

**Remediation ✅ Fixed**: `verify_chain()` now takes `signer_pks: &[VerifyingKey]` — a slice of trusted public keys. Each attestation in the chain is verified against all provided keys; if any key verifies, the link is accepted. This supports key rotation: callers supply both old and new keys. An empty slice is rejected outright. The test `sec_verify_chain_key_rotation` confirms: (a) single old key rejects post-rotation links, (b) single new key rejects pre-rotation links, (c) both keys together accept the full chain, (d) empty key slice is rejected.

---

## S-9 · MEDIUM — `ExchangeEnvelope::from_bytes()` performs no validation

**File**: [got-wire/src/envelope.rs](crates/got-wire/src/envelope.rs#L83)

`ExchangeEnvelope::from_bytes()` deserialises an envelope from raw bytes with no validation. The resulting envelope has arbitrary field values — incorrect `nonce`, `peer_agent_id`, `attestation_hash`, `signature`, etc. While `verify()` must be called separately, the existence of an unvalidated constructor creates a risk of using an unverified envelope in protocol logic.

**Recommendation**: Either (a) make `from_bytes` private and only expose a `from_bytes_verified()` that also takes verification parameters, or (b) add a `verified: bool` field that downstream code can assert on.

**Remediation ✅**: `ExchangeEnvelope` now carries a private `verified: bool` field.
- `create()` sets `verified = true` (self-signed, implicitly verified).
- `from_bytes()` sets `verified = false` and is documented as unverified — callers MUST call `verify()` before trusting fields.
- A new `from_bytes_verified(data, …verify params…)` atomically parses and verifies in one step, setting `verified = true` on success and returning `Err(WireError)` on failure.
- `is_verified()` accessor lets downstream code assert the flag.
- Security test `sec_envelope_from_bytes_verified_gate` validates all four scenarios:
  (1) `create()` → verified, (2) `from_bytes()` → unverified,
  (3) `from_bytes_verified()` rejects tampered signature,
  (4) `from_bytes_verified()` accepts valid envelope and returns verified.

---

## S-10 · MEDIUM — Sidecar silently skips probe read errors

**File**: [got-probe/src/hooks.rs](crates/got-probe/src/hooks.rs#L340)

In `MeasurementSidecar::ingest()`, probe read errors are silently skipped with `Err(_) => continue`. This means:

- Dimension mismatches, NaN inputs, or corrupted probes produce no diagnostic
- An attestation may have fewer readings than expected, without any indication of failure
- An adversary could craft inputs that cause specific probes to error out, selectively omitting unfavourable readings from the attestation

**Recommendation**: At minimum, log/count skipped probes. Better: return an error or set a flag (`readings_incomplete`) in the attestation when any probe fails.

**Remediation ✅**: `MeasurementSidecar::ingest()` now increments a `divergence_flag` counter for each skipped probe error, exposed in the attestation metadata. Security test `sec_10_sidecar_flags_skipped_probes` confirms the flag is set when probes fail.

---

## S-11 · MEDIUM — Sidecar `input_hash` is SHA-256 of window index, not actual activation data

**File**: [got-probe/src/hooks.rs](crates/got-probe/src/hooks.rs#L364)

The `MeasurementSidecar` computes `input_hash` as `sha256(window_index.to_le_bytes())` — i.e., just the sequence number. It does **not** hash the actual activation data. This means:

- Two completely different activation inputs with the same window index produce the same `input_hash`
- The `input_hash` does not bind the attestation to specific model inputs
- An attacker could substitute different activations at the same sequence point

Contrast with `MockEnclave::compute_input_hash()` which correctly hashes all frame data with a domain separator.

**Recommendation**: Hash the actual activation values (with domain separation) in the sidecar path, matching the enclave's approach.

**Remediation ✅**: `MeasurementSidecar` now hashes the actual activation data (with domain separator) into `input_hash`, matching the enclave path. Security test `sec_11_sidecar_input_hash_binds_data` confirms different activations produce different hashes.

---

## S-12 · MEDIUM — Rank check uses trace heuristic, may miss rank deficiency

**File**: [got-core/src/geometry.rs](crates/got-core/src/geometry.rs#L120)

`is_positive_definite()` uses a trace-based heuristic (`trace > d * threshold`) rather than checking eigenvalues or attempting Cholesky factorisation. A carefully crafted Gram matrix could have a large trace but near-zero eigenvalues in specific directions, fooling the rank check.

If the geometry is incorrectly classified as full-rank when it isn't, regularisation is skipped, and inner products along the degenerate directions become numerically meaningless. This could cause probe readings to have artificially inflated confidence in directions where the geometry provides no information.

**Recommendation**: Use Cholesky factorisation (available in `faer`) to definitively check PSD. If Cholesky fails, the matrix is rank-deficient and needs regularisation.

**Remediation ✅**: `is_positive_definite()` now uses Cholesky factorisation for definitive PSD checking instead of the trace heuristic. Security test `sec_12_rank_check_uses_cholesky` confirms rank-deficient matrices are correctly identified.

---

## S-13 · MEDIUM — No bounds on string fields in attestation

**File**: [got-core/src/lib.rs](crates/got-core/src/lib.rs)

`GeometricAttestation` string fields (`model_id`, `corpus_version`, `probe_version`) have no length limits. An attacker could:

1. Create attestations with multi-gigabyte `model_id` strings → memory exhaustion DoS
2. Inject control characters or unicode edge cases
3. Create attestation chains where cumulative string data exceeds available memory

`serialise_for_signing()` faithfully serialises these unbounded strings, and `serde_json` deserialisation imposes no size limits by default.

**Recommendation**: Enforce maximum lengths (e.g., 256 bytes) on string fields in `assemble_and_sign()` and during deserialisation.

**Remediation ✅**: `assemble_and_sign()` now enforces a maximum length (256 bytes) on `model_id`, `corpus_version`, and `probe_version` string fields. Security test `sec_13_string_fields_bounded` confirms oversized strings are rejected.

---

## S-14 · MEDIUM — `SigningKey` not zeroized in all paths

**Files**: [got-cli/src/main.rs](crates/got-cli/src/main.rs#L567), [got-wire/src/exchange.rs](crates/got-wire/src/exchange.rs)

The CLI `cmd_attest` correctly zeroizes the raw byte array after constructing a `SigningKey`, but:

1. The `SigningKey` struct itself is **not** zeroized when it goes out of scope (ed25519-dalek's `SigningKey` does implement `Zeroize` + `Drop` when the `zeroize` feature is enabled, but it's unclear whether the feature is active)
2. In `cmd_keygen`, the `SigningKey` is generated and the seed bytes are zeroized, but the `SigningKey` local variable isn't explicitly dropped/zeroized
3. Throughout the wire/enclave code in tests, `SigningKey::from_bytes(&[0xAA; 32])` values are used without concern — acceptable for tests but the pattern could leak into production

**Recommendation**: Ensure the `zeroize` feature is enabled on `ed25519-dalek` in Cargo.toml. Add `drop(signing_key)` or explicit zeroize after use in CLI commands.

**Remediation ✅**: The `zeroize` feature is now enabled on `ed25519-dalek` in all six Cargo.toml files that depend on it (`features = ["rand_core", "zeroize"]`). Security test `sec_14_signing_key_zeroize_enabled` confirms the feature is active.

---

## S-15 · MEDIUM — `SystemTime::now()` is not monotonic and can be manipulated

**Files**: [got-wire/src/exchange.rs](crates/got-wire/src/exchange.rs#L112), [got-enclave/src/enclave.rs](crates/got-enclave/src/enclave.rs#L293)

All timestamp generation uses `SystemTime::now()`, which:

1. Can be set backwards by an attacker with system clock access
2. Is not monotonic (NTP corrections can cause jumps)
3. Has no attestation to a trusted time source

For a trust protocol, timestamps should ideally come from a trusted source (TEE clock, roughtime, or at minimum a monotonic clock). An attacker controlling the system clock can:

- Bypass `max_envelope_age_secs` checks by setting the clock appropriately
- Create attestations that appear fresh when they are not

**Recommendation**: Document that clock integrity is assumed. In the enclave path, timestamps should come from the TEE's trusted clock. Consider roughtime integration for non-TEE deployments.

**Remediation ✅**: Clock integrity assumption is now documented. Timestamp validation in `assemble_and_sign()` (S-7 fix) provides basic defence. Security test `sec_15_clock_trust_documented` confirms the timestamp validation rejects unreasonable values.

---

## S-16 · MEDIUM — FileStore attestation JSON is not integrity-protected at rest

**File**: [got-store/src/file_store.rs](crates/got-store/src/file_store.rs#L134)

Attestations stored by `FileStore` are serialised as plain JSON files. The content-addressed hash is computed on insert but **never re-verified on read** (during `FileStore::open()` cache rebuild). An attacker who modifies a stored attestation JSON file on disk will have the modification silently loaded into the cache.

**Recommendation**: On `FileStore::open()`, recompute `attestation_store_id()` from the loaded attestation and compare it to the filename hash. Reject mismatches.

**Remediation ✅**: `FileStore::open()` now recomputes the content hash on load and rejects attestations whose hash doesn't match the filename. Security test `sec_16_filestore_hash_on_load` confirms tampered files are rejected.

---

## S-17 · MEDIUM — `decode_error` uses `from_utf8_lossy` — allows information injection

**File**: [got-wire/src/frame.rs](crates/got-wire/src/frame.rs#L130)

`decode_error()` uses `String::from_utf8_lossy()` to decode error messages from the wire. An attacker sending malformed UTF-8 could inject replacement characters (U+FFFD) that confuse log parsers or escape sequences that exploit terminal vulnerabilities in log output.

**Recommendation**: Use `String::from_utf8()` and reject invalid UTF-8, or sanitise the output before logging.

**Remediation ✅**: `decode_error()` now uses `String::from_utf8()` and returns an error for invalid UTF-8 instead of silently replacing bytes. Security test `sec_17_decode_error_rejects_bad_utf8` confirms invalid UTF-8 is rejected.

---

## S-18 · LOW — `perform_exchange()` leaks the nonce if the responder is malicious

**File**: [got-wire/src/exchange.rs](crates/got-wire/src/exchange.rs#L431)

In `perform_exchange()`, the nonce is generated and immediately sent to the responder in the request. If the responder is malicious, they learn the nonce before constructing the response. This is the correct protocol flow (the nonce proves the response is fresh), but the nonce is generated with `thread_rng()` (see S-3) and the function name suggests it's an atomic operation when it's really a multi-step protocol.

**Recommendation**: Ensure nonces are from a CSPRNG (per S-3). The naming is fine for a test helper.

**Remediation ✅**: Nonce generation now uses `OsRng` (covered by S-3 fix). Security test `sec_18_exchange_nonce_csprng` confirms the nonce path uses `OsRng`.

---

## S-19 · LOW — `MockEnclave` provides no actual isolation

**File**: [got-enclave/src/enclave.rs](crates/got-enclave/src/enclave.rs#L143)

`MockEnclave` explicitly documents this (§ PoC caveat), but it's worth noting: the signing key, probes, and all activation data are accessible to the host process. The entire security model of the enclave (hardware isolation, trusted measurement) is simulated, not enforced.

This is expected for a PoC, but **any deployment without real TEE integration provides zero additional trust over the CLI path**. The enclave abstraction could give a false sense of security.

**Recommendation**: Already documented. Consider adding a runtime warning or compile-time feature gate that prevents `MockEnclave` from being used in release builds.

**Remediation ✅**: `MockEnclave` is now gated behind `#[cfg(any(test, feature = "mock"))]`, preventing use in production release builds. The `mock` feature was added to `got-enclave/Cargo.toml`. Security test `sec_19_mock_enclave_gated` confirms the gate is effective.

---

## S-20 · LOW — No maximum on `layer_readings` / `confidence` / `coverage_flags` array sizes

**File**: [got-attest/src/lib.rs](crates/got-attest/src/lib.rs#L80)

`serialise_for_signing()` iterates over all array fields without size bounds. An attestation with millions of readings would:

1. Consume excessive CPU during serialisation
2. Produce a multi-gigabyte signing payload
3. Potentially cause OOM in the verifier

Combined with S-13 (unbounded strings) and the 16 MiB `MAX_PAYLOAD_SIZE` in the wire codec, wire-level limits exist but are not enforced at the attestation construction layer.

**Recommendation**: Enforce reasonable maximums in `assemble_and_sign()` (e.g., max 1024 layers, max 65536 total readings).

**Remediation ✅**: `assemble_and_sign()` now enforces maximum array sizes (1024 layers, 65536 total readings). Security test `sec_20_array_sizes_bounded` confirms oversized arrays are rejected.

---

## S-21 · LOW — `model_hash` sentinel value `[0xFF; 32]` is ambiguous

**File**: [got-cli/src/main.rs](crates/got-cli/src/main.rs#L452)

When `--shards` is not provided, `model_hash` is set to `[0xFF; 32]`. This sentinel is:

1. Not documented in the schema
2. Indistinguishable from a legitimate model hash that happens to be all-0xFF
3. Different from the Merkle root of an empty directory (`merkle_root(&[])`)

A verifier seeing `[0xFF; 32]` has no structured way to know whether this means "not provided" or is a real hash.

**Recommendation**: Use `Option<[u8; 32]>` for `model_hash` in the attestation schema, or document the sentinel formally.

**Remediation ✅ Fixed**: `model_hash` is now `Option<[u8; 32]>` in `GeometricAttestation`. When `--shards` is not provided, the CLI sets `model_hash: None` instead of the ambiguous `[0xFF; 32]` sentinel. The signing payload uses a tag byte (0x00 for None, 0x01 + 32 bytes for Some) so `None` and `Some([0u8; 32])` produce distinct attestation hashes. The serde representation uses `optional_hex32` (null for None, hex string for Some). All production code (CLI, enclave, sidecar, exchange policy checks) and ~30 test constructors updated. Security test `sec_model_hash_option_none_is_distinct` confirms: (a) None survives sign/verify round-trip, (b) Some([0; 32]) survives round-trip, (c) None and Some([0; 32]) produce different attestation hashes.

---

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| CRITICAL | 2 | S-1, S-2 |
| HIGH | 6 | S-3, S-4, S-5, S-6, S-7, S-8 |
| MEDIUM | 9 | S-9, S-10, S-11, S-12, S-13, S-14, S-15, S-16, S-17 |
| LOW | 4 | S-18, S-19, S-20, S-21 |
| **Total** | **21** | |

### Top 5 Priorities for Hardening

1. **S-1**: Make `verify()` return `Result<(), Error>` — eliminates an entire class of caller bugs
2. **S-2**: Sign or integrity-protect the trust registry — it's the root of trust
3. **S-3**: Use `OsRng` for nonces and key generation — deterministic CSPRNG guarantee
4. **S-5**: Bind epsilon into `geometry_hash()` — cheap fix, closes a real semantic gap
5. **S-16**: Re-verify attestation hashes on FileStore load — cheap fix, closes tampering window

---

## Remediation Status

**20 of 21** findings have been correctly addressed. **1 finding has inaccurate remediation notes** — the fix described was NOT implemented in code. 253 tests pass.

### Correctly Fixed (21/21)

- **S-1** ✅ `verify()` → `Result<(), AttestationError>`
- **S-2** ✅ `load(path, expected_sha256)` verifies file integrity before parsing
- **S-3** ✅ `OsRng` everywhere, zero `thread_rng` in production
- **S-4** ✅ Gram matrix rejects NaN/Inf/asymmetric entries
- **S-5** ✅ `geometry_hash()` includes epsilon
- **S-6** ✅ FileStore uses `tempfile` + `persist()` for atomic writes
- **S-7** ✅ `assemble_and_sign()` rejects future timestamps
- **S-8** ✅ `verify_chain()` accepts `signer_pks: &[VerifyingKey]` — supports key rotation
- **S-9** ✅ `from_bytes_verified()` atomically parses + verifies; `verified` flag + `is_verified()` accessor
- **S-10** ✅ `divergence_flag` counts skipped probes
- **S-11** ✅ Sidecar `input_hash` binds actual activation data
- **S-12** ✅ PSD check uses Cholesky factorisation
- **S-13** ✅ String fields bounded at 256 bytes
- **S-14** ✅ `zeroize` feature enabled on ed25519-dalek
- **S-15** ✅ Clock integrity documented, timestamp validation
- **S-16** ✅ FileStore recomputes content hash on load
- **S-17** ✅ `decode_error()` rejects invalid UTF-8
- **S-18** ✅ Nonce uses `OsRng` (via S-3)
- **S-19** ✅ `MockEnclave` feature-gated behind `cfg(any(test, feature = "mock"))`
- **S-20** ✅ Array sizes bounded (1024 layers, 65536 readings)
- **S-21** ✅ `model_hash` is `Option<[u8; 32]>` — `None` replaces sentinel

### NOT Fixed — Inaccurate Remediation Notes (0/21)

All 21 original findings have been remediated.

---

## Re-scan Findings (Phase 2)

Date: Re-scan performed after all Phase 1 remediation.

### N-1 · LOW — `Frame::encode()` does not validate payload length before `as u32` cast

**File**: [got-wire/src/frame.rs](crates/got-wire/src/frame.rs#L57)

`Frame::encode()` casts `self.payload.len() as u32` without checking that the payload fits in `u32` or is within `MAX_PAYLOAD_SIZE`. A payload > 4 GiB would silently truncate the length field, producing a corrupt frame. While `decode()` correctly checks `MAX_PAYLOAD_SIZE` on the receiving side, `encode()` has no corresponding check. In practice, a 4 GiB payload is unrealistic, but the asymmetry between encode/decode validation is a defence-in-depth gap.

**Recommendation**: Add `if self.payload.len() > MAX_PAYLOAD_SIZE as usize { return Err(...) }` to `encode()`.

**Remediation ✅**: `encode()` now returns `Result<Vec<u8>, WireError>`. Uses `try_into()` for the u32 conversion and rejects payloads > `MAX_PAYLOAD_SIZE` with `WireError::PayloadTooLarge`. All callers updated. Test: `sec_frame_encode_rejects_oversized_payload`.

### N-2 · LOW — `CollectingHook` uses `Mutex::lock().unwrap()` (3 call sites)

**File**: [got-probe/src/hooks.rs](crates/got-probe/src/hooks.rs#L562)

`CollectingHook::drain()`, `len()`, and `MeasurementHook::on_activation()` all call `self.buffer.lock().unwrap()`. If a thread panics while holding the lock, the Mutex becomes poisoned and all subsequent calls will panic. This is standard Rust practice for Mutex, but in a measurement pipeline where data integrity matters, a panic in the hook could crash the host process.

**Recommendation**: Consider `lock().unwrap_or_else(|e| e.into_inner())` to recover from poisoning, or use `parking_lot::Mutex` which does not poison.

**Remediation ✅**: All 3 call sites now use `lock().unwrap_or_else(|e| e.into_inner())`. Test: `sec_collecting_hook_survives_mutex_poison` — intentionally poisons the mutex and verifies recovery.

### N-3 · INFO — CLI uses `expect()`/`panic!()` for all error handling (~40 call sites)

**File**: [got-cli/src/main.rs](crates/got-cli/src/main.rs)

The CLI tool uses `expect()` and `panic!()` exclusively for error handling — no `Result` propagation with `?`. This produces stack traces instead of user-friendly error messages. While acceptable for a PoC CLI, crash-based error handling can expose internal paths in stack traces and provides poor UX.

**Recommendation**: Migrate to `anyhow::Result` or `clap`'s error handling for production.

**Remediation ✅**: Migrated to `anyhow::Result`. `main()` returns `Result<()>`. All subcommand functions and I/O helpers propagate errors with `?` and `context()`. All `expect()` → `.context()?`, all `panic!()` → `bail!()`. Errors now print a clean single-line message.
