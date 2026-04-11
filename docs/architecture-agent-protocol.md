# Agent-to-Agent Attestation Protocol

The diagrams in `architecture-sequences.md` and `architecture-flows.md` model
both user-to-agent and agent-to-agent interactions. This document focuses
specifically on the **agent-to-agent** case, where two or more autonomous
agents mutually verify each other's geometric attestations before cooperating.

The wire protocol (envelopes, framing, exchange, chain verification, trust
registry) is **fully implemented** in `got-wire`.

All protocol paths reflect the security-hardened codebase (353 tests passing).

---

## 1. Key Assumptions

| # | Assumption | Rationale |
|---|---|---|
| A1 | Each agent controls its own Ed25519 keypair | No shared secret between agents |
| A2 | Each agent has a `.gotue` file containing its model's unembedding matrix U | A new `.gotue` is exported whenever the model updates |
| A3 | Agents exchange attestations over an untrusted channel | Signatures + envelopes make tampering detectable |
| A4 | A trust registry maps agent identities to public keys | S-2: registry verified by SHA-256 on load |
| A5 | Agents agree on acceptable `max_drift` bounds out-of-band | Governance parameter, not derived from math |
| A6 | Envelope timestamps provide freshness (configurable max age) | S-9: prevents replay of old attestations |
| A7 | Hardware enclave keeps signing key inside trust boundary | Key never exposed to application layer |
| A8 | Chain verification accepts multiple signer keys | S-8: supports key rotation across chain boundaries |
| A9 | Each agent declares a domain scope (primary domain, permitted patterns, exclusions) | §4 / Appendix B: enforces structural cross-domain boundaries before any cryptographic verification |

---

## 2. Mutual Attestation Exchange

Two agents (Alice, Bob) each hold a signed attestation about their own
model. The protocol verifies the other agent's alignment properties
before cooperating. The exchange is symmetric — both sides produce and
verify.

The exchange protocol is implemented in `got-wire::exchange`:
`build_request()`, `build_response()`, `validate_request()`,
`validate_response()`, `perform_exchange()`.

```
  Agent Alice                  Channel               Agent Bob
  (Model A, KeyPair A)                                (Model B, KeyPair B)
       |                          |                        |
       |   ---- PHASE 0: Domain Compatibility (local) ---- |
       |                          |                        |
       |  registry.lookup(id_B)   |    registry.lookup(id_A)|
       |  check_domain_           |    check_domain_       |
       |    compatibility(        |      compatibility(    |
       |      scope_A, scope_B)   |        scope_B, scope_A)|
       |  Exclusion / permission /|    Exclusion / perm /  |
       |  mode intersection.      |    mode intersection.  |
       |  FAIL → abort early      |    FAIL → reject       |
       |  (no crypto, no probes)  |    (Verdict::Rejected) |
       |                          |                        |
       |   ---- PHASE 1: Self-Attest (parallel) ----      |
       |                          |                        |
       |-- compute Φ_A from U_A  |   compute Φ_B from U_B--|
       |-- run probes on Φ_A     |      run probes on Φ_B--|
       |-- optional causal_check |  optional causal_check --|
       |-- sign(attest_A, sk_A)  |    sign(attest_B, sk_B)--|
       |   assemble_and_sign()   |    assemble_and_sign()   |
       |   S-7/S-13/S-20 gates   |    S-7/S-13/S-20 gates  |
       |   (or enclave_pipeline) |   (or enclave_pipeline)  |
       |                          |                        |
       |   ---- PHASE 2: Exchange ----                     |
       |                          |                        |
       |-- build_request(         |                        |
       |     nonce, id_B, sk_A,   |                        |
       |     chain_A, attest_A)   |                        |
       |                          |                        |
       |   ExchangeRequest {      |                        |
       |     agent_id: id_A,      |                        |
       |     envelope: signed     |                        |
       |       (nonce, id_B,      |                        |
       |        attest_hash,      |                        |
       |        chain_root,       |                        |
       |        timestamp, sig),  |                        |
       |     chain: [...],        |                        |
       |     current: attest_A }  |                        |
       |                          |                        |
       |-- Frame::encode(req) --->|  N-1: returns Result   |
       |   send request --------->|---> receive request    |
       |                          |                        |
       |                          |   validate_request(    |
       |                          |     req, id_B,         |
       |                          |     None, registry)    |
       |                          |   S-2: registry        |
       |                          |     integrity verified |
       |                          |                        |
       |                          |   build_response(      |
       |                          |     nonce, id_A, sk_B, |
       |                          |     verdict, chain_B,  |
       |                          |     attest_B, reason)  |
       |                          |                        |
       |   receive response <-----|<--- send response      |
       |                          |                        |
       |   ---- PHASE 3: Verify ----                       |
       |                          |                        |
       |-- validate_response(     |                        |
       |     rsp, id_A,           |                        |
       |     nonce, registry)     |                        |
       |                          |                        |
       |   Envelope verify:       |    Envelope verify:    |
       |     Ed25519 sig check    |      Ed25519 sig check |
       |     peer_agent_id match  |      peer_agent_id match|
       |     attest_hash match    |      attest_hash match |
       |     chain_root match     |      chain_root match  |
       |     timestamp freshness  |      timestamp freshness|
       |     S-9: verified=true   |      S-9: verified=true|
       |                          |                        |
       |   If chain (v2/v3):      |    If chain:           |
       |     verify_chain(        |      verify_chain(     |
       |       chain, current,    |        chain, current, |
       |       &[pk_B], max_drift)|        &[pk_A], max_drift)|
       |       ↑ S-8: rotation   |        ↑ S-8: rotation |
       |     -> ChainVerdict      |      -> ChainVerdict   |
       |                          |                        |
       |   ---- PHASE 4: Decide ----                       |
       |                          |                        |
       |   Both Accepted?         |           Both Accepted?|
       |     yes -> cooperate     |    cooperate <- yes    |
       |     no  -> refuse        |       refuse <- no     |
       |                          |                        |
```

### Key difference from user-to-agent flow

In the user-to-agent flow, the user trusts their own model implicitly and
only verifies external attestations. In the agent-to-agent flow, **both
sides produce and verify** — the protocol is symmetric. The exchange
envelope provides mutual authentication and replay resistance (nonce +
timestamp + peer binding).

---

## 3. Exchange Envelope (Implemented)

The `ExchangeEnvelope` in `got-wire::envelope` provides a signed binding
between an attestation and a specific exchange. This prevents relay attacks
where a valid attestation is redirected to a different peer.

S-9 hardening: the envelope has a `verified: bool` field (private).
- `from_bytes()` sets `verified = false` — the caller MUST call `verify()`.
- `from_bytes_verified()` combines deserialisation + verification in one step.
- `create()` sets `verified = true` (self-signed, implicitly verified).
- `is_verified()` accessor lets callers confirm verification status.

```
  Envelope (200 bytes total)
  ==========================

  Signable portion (136 bytes):
  ┌───────────────────────────────────────┐
  │  nonce              [u8; 32]          │  Random, generated by initiator
  │  peer_agent_id      [u8; 32]          │  SHA-256(recipient's public key)
  │  attestation_hash   [u8; 32]          │  SHA-256(serialise_for_signing(attest))
  │  chain_root_hash    [u8; 32]          │  SHA-256(serialise_for_signing(chain[0]))
  │  timestamp          u64 LE            │  Unix UTC seconds
  └───────────────────────────────────────┘
            │
            v
  Ed25519 sign(signable_bytes, sender's sk)
            │
            v
  ┌───────────────────────────────────────┐
  │  signature          [u8; 64]          │  Appended to form 200-byte wire format
  └───────────────────────────────────────┘

  Internal state:
  ┌───────────────────────────────────────┐
  │  verified           bool (private)    │  S-9: tracks verification status
  └───────────────────────────────────────┘
```

### Verification steps (in `envelope.verify()`)

1. **Ed25519 signature** — verify signable bytes against sender's public key
2. **peer_agent_id** — must match recipient's own agent ID
3. **Nonce** — for responses, must match the initiator's nonce
4. **attestation_hash** — must match `SHA-256(serialise_for_signing(attestation))`
5. **chain_root_hash** — must match chain[0] if present, or zeroes if no chain
6. **Timestamp** — `age ≤ max_envelope_age_secs` (from trust registry config);
   rejects both too-old and future timestamps

---

## 4. Chained Attestation in Agent-to-Agent Context

When an agent self-learns (updates its weights), it must produce a chained
attestation (schema v2 or v3) and present the full chain to peers. The peer
walks the chain to decide whether the model has drifted acceptably.

Chain verification is implemented in `got-wire::chain::verify_chain()`.
S-8: accepts `signer_pks: &[VerifyingKey]` — each attestation need only
verify against **at least one** key in the set, supporting key rotation.

```
  Agent Alice (self-learning)           Agent Bob (verifier)
       |                                      |
       |=== Epoch 0 (initial) ===             |
       |                                      |
       |-- checkpoint Φ₀ (.gotgeo)           |
       |-- train probes against Φ₀           |
       |-- attest₀:                          |
       |     schema_version: 1 or 3          |
       |     parent_hash: None               |
       |     geometry_hash: H(Φ₀)           |
       |     geometry_drift: 0.0              |
       |     model_hash: Option (S-21)       |
       |     causal_scores: [...] (if v3)    |
       |-- assemble_and_sign(attest₀, sk_A) |
       |     S-7/S-13/S-20 gates             |
       |                                      |
       |-- exchange attest₀ ---------------->|
       |                              verify(attest₀, pk_A)
       |                              store attest₀ as anchor
       |                                      |
       |=== Epoch 1 (after update) ===        |
       |                                      |
       |-- model updates weights              |
       |-- compute Φ₁ from new U₁           |
       |-- drift = ‖Φ₁ − Φ₀‖_F / ‖Φ₀‖_F  |
       |-- if drift > max_drift:              |
       |     STOP — must retrain probes      |
       |-- attest₁:                          |
       |     parent_hash: attestation_hash(₀)|
       |     geometry_hash: H(Φ₁)           |
       |     geometry_drift: drift            |
       |-- assemble_and_sign(attest₁, sk_A) |
       |                                      |
       |-- send [attest₀, attest₁] -------->|
       |                                      |
       |                              verify_chain(
       |                                [attest₀], attest₁,
       |                                &[pk_A], max_drift)
       |                                ↑ S-8: key rotation
       |                              → ChainVerdict {
       |                                  length: 2,
       |                                  max_drift_observed }
       |                                      |
       |=== Epoch 2 (another update) ===      |
       |                                      |
       |-- compute Φ₂, drift from Φ₀        |
       |     (always relative to reference)   |
       |-- attest₂:                          |
       |     parent_hash: attestation_hash(₁)|
       |     geometry_hash: H(Φ₂)           |
       |     geometry_drift: cumulative       |
       |-- sign(attest₂, sk_A)              |
       |                                      |
       |-- send [attest₀..₂] --------------->|
       |                              verify_chain(
       |                                [attest₀, attest₁],
       |                                attest₂,
       |                                &[pk_A, pk_A_new],
       |                                max_drift)
       |                              ↑ S-8: old+new key
       |                                      |
```

### Chain verification rules (implemented in `verify_chain()`)

1. **Signature** — each attestation must verify against **at least one** key in `signer_pks` (S-8)
2. **Linkage** — `attest_i.parent_attestation_hash == attestation_hash(attest_{i-1})`
3. **Drift bound** — `attest_i.geometry_drift ≤ max_drift`
4. **Model identity** — `model_id` must be consistent across all chain entries
5. **Completeness** — chain must start from an anchor with `parent_hash = None`
6. **Final link** — current attestation's parent must point to chain's last element

---

## 5. Trust Registry (Implemented)

The `TrustRegistry` in `got-wire::registry` maps agent identities to
public keys and local policy. It is loaded from a TOML configuration file.

S-2 hardening: `TrustRegistry::load(path)` computes SHA-256 of the file
contents and verifies against an expected integrity hash.

```
  Trust Registry (TOML)
  =====================

  [registry]
  max_chain_length = 100
  max_envelope_age_secs = 300
  max_attestation_age_secs = 3600     # defence-in-depth

  [[agents]]
  name = "alice"
  public_key = "aabb..."    # 64 hex chars = 32 bytes
  max_drift_accepted = 0.05
  roles = ["producer"]
  # expected_model_hash = "ccdd..."  # optional: pin model identity

  [[agents]]
  name = "bob"
  public_key = "ccdd..."
  max_drift_accepted = 0.05
  roles = ["verifier"]

  [[agents]]
  name = "carol"
  public_key = "eeff..."
  max_drift_accepted = 0.10
  roles = ["aggregator"]
```

Agent IDs are computed as `SHA-256(public_key_bytes)` — deterministic and
derived from the public key, never manually assigned.

```
  In code:

  TrustRegistry::from_toml(toml_str) → TrustRegistry
  TrustRegistry::load(path) → TrustRegistry  (S-2: SHA-256 integrity check)
  registry.lookup(agent_id) → Option<&AgentEntry>
  registry.add_agent(entry)

  AgentEntry {
    name: String,
    public_key: VerifyingKey,
    agent_id: [u8; 32],           // SHA-256(public_key.as_bytes())
    max_drift_accepted: f32,
    roles: Vec<String>,
    expected_model_hash: Option<[u8; 32]>,  // pin model identity
    certificate: Option<AgentCertificate>,  // optional CA-signed binding
    domain_scope: Option<DomainScope>,      // §4: declared competence
  }

  TrustRegistry {
    agents: HashMap<[u8; 32], AgentEntry>,
    max_chain_length: usize,
    max_envelope_age_secs: u64,
    max_attestation_age_secs: u64,    // defence-in-depth
  }
```

---

## 5b. Domain Scoping (Implemented — Protocol §4 / Appendix B)

The `got-wire::domain` module implements the structural Phase 0 layer
specified in §4 of the protocol companion paper. Each `AgentEntry` may
carry an optional `DomainScope` declaring the agent's primary domain
of competence, the patterns it is permitted to interact with, and an
exclusion list. When **both** peers declare a scope, the bidirectional
compatibility check runs immediately after registry lookup — *before*
envelope verification, attestation signature checks, chain walking, or
geometric threshold checks. When either peer is unscoped, the check is
skipped (backwards compatible with PoC registries).

Domain incompatibility is structural: it cannot be overridden by high
probe readings, certificate elevation, or governance dispensation.

```
  Domain types
  ============

  Domain(String)              // dot-separated lowercase namespace
                              // e.g. "agriculture.crop-management"
                              // strict parser: [a-z0-9.-], no empty
                              // segments, no leading/trailing dot

  DomainPattern               // exact / sub-tree wildcard / global
    "agriculture.crop-management"   exact
    "agriculture.*"                 sub-tree (matches the parent too)
    "*"                             global wildcard
    Substring guard: "agriculture-x" does NOT match "agriculture.*"

  InteractionMode             // §4.2
    ReadOnly                  // receive only
    Advisory                  // non-binding recommendations
    Cooperative               // joint decision-making

  PermittedDomain { pattern, mode }

  DomainScope {
    primary:    Domain,
    permitted:  Vec<PermittedDomain>,
    exclusions: Vec<DomainPattern>,
  }
```

Mode lookup uses **most-specific-pattern wins** — an exact pattern
beats any wildcard, and a longer wildcard prefix beats a shorter one.

```
  check_domain_compatibility(a: &DomainScope, b: &DomainScope)
    → Result<(), WireError>

    1. Exclusions (hard veto, both directions)
         a.is_excluded(b.primary)  → DomainExcluded
         b.is_excluded(a.primary)  → DomainExcluded

    2. Bidirectional permission
         a.mode_for(b.primary)     → DomainNotPermitted if None
         b.mode_for(a.primary)     → DomainNotPermitted if None

    3. Mode intersection non-empty
         (ReadOnly, ReadOnly)      → DomainModeIncompatible
         (any other pair)          → OK
```

### Wired into the exchange

`validate_request` and `validate_response` run the check at "Phase 0"
— immediately after the sender lookup and certificate validation, and
before envelope verification:

```rust
if let Some(self_entry) = registry.lookup(own_agent_id) {
    if let (Some(peer_scope), Some(self_scope)) =
        (entry.domain_scope.as_ref(), self_entry.domain_scope.as_ref())
    {
        if let Err(e) = check_domain_compatibility(peer_scope, self_scope) {
            return Ok((Verdict::Rejected, format!("domain incompatible: {e}")));
        }
    }
}
```

### TOML

Domain scope is declared per-agent as inline tables in the registry
TOML. Permitted/exclusion lists are rejected at parse time if the
agent has no `primary_domain`.

```toml
[[agents]]
name = "alice"
public_key = "aabb..."
primary_domain = "agriculture.crop-management"
exclusion_domains = ["transport.*"]
permitted_domains = [
  { pattern = "agriculture.*", mode = "cooperative" },
  { pattern = "meteorology.*", mode = "advisory" },
]

[[agents]]
name = "vehicle-controller"
public_key = "ccdd..."
primary_domain = "transport.autonomous-vehicle"
permitted_domains = [
  { pattern = "transport.*", mode = "cooperative" },
  { pattern = "infrastructure.traffic-management", mode = "cooperative" },
]
```

### Use cases (from the protocol paper)

| Pair | Result | Failure / mode |
|---|---|---|
| §5.1 agriculture.crop-management ↔ transport.autonomous-vehicle | **Rejected at Phase 0** | `DomainExcluded` (also `DomainNotPermitted` without the exclusion) |
| §5.2 healthcare.diagnostic-advisory ↔ healthcare.drug-interaction | Accepted | Asymmetric: `Advisory` ↔ `ReadOnly` |
| §5.3 supply-chain peers within `agriculture.*` | Accepted | `Cooperative` ↔ `Cooperative` |

---

## 6. Multi-Agent Trust Negotiation

When more than two agents form a group, each agent must decide which
peers to trust. This requires a shared trust registry.

```
                Alice           Bob           Carol
                  |               |              |
                  |-- attest_A -->|              |
                  |-- attest_A --|------------->|
                  |               |              |
                  |<- attest_B --|              |
                  |<- attest_B --|              |
                  |               |-- attest_B ->|
                  |               |              |
                  |<- attest_C --|--<- attest_C -|
                  |               |              |
                  |               |              |
          Each agent verifies all peers it cooperates with:
                  |               |              |
           verify(B,C)     verify(A,C)     verify(A,B)
                  |               |              |
           All pass?        All pass?       All pass?
            → join group     → join group    → join group
            or refuse        or refuse       or refuse
```

### Aggregator pattern

In a hub-and-spoke topology, a designated aggregator (Carol) collects
attestations from all agents, verifies them, and issues a group-level
summary. This avoids O(n²) pairwise verification:

```
              Alice                Carol (aggregator)              Bob
                |                        |                          |
                |-- attest_A ----------->|                          |
                |                        |<---------- attest_B -----|
                |                        |                          |
                |                  verify(attest_A, pk_A)           |
                |                  verify(attest_B, pk_B)           |
                |                        |                          |
                |                  All pass?                        |
                |                    → build group_attestation      |
                |                      { members: [A, B],           |
                |                        attestation_hashes: [...], |
                |                        group_drift_max: max(...), |
                |                        signed by sk_C }           |
                |                        |                          |
                |<-- group_attest -------|------- group_attest ---->|
                |                        |                          |
          verify(group, pk_C)            |          verify(group, pk_C)
          check own hash in members      |    check own hash in members
```

---

## 7. Failure Modes

| Failure | What the receiving agent sees | Response |
|---|---|---|
| Envelope signature invalid | `EnvelopeSignatureInvalid` | Reject, do not cooperate |
| Peer ID mismatch | `PeerIdMismatch { expected, got }` | Reject, possible relay attack |
| Nonce mismatch | `NonceMismatch` | Reject, possible replay |
| Attestation hash mismatch | `AttestationHashMismatch` | Reject, attestation tampered in transit |
| Chain root hash mismatch | `ChainRootHashMismatch` | Reject, chain anchor doesn't match |
| Timestamp expired | `TimestampExpired { age_secs, max_secs }` | Reject, demand fresh attestation |
| Attestation sig invalid | `verify()` returns `Ok(false)` | Reject, attestation was tampered |
| Schema version unknown | `UnknownSchemaVersion(v)` | Reject, incompatible agent version |
| Chain link broken | Parent hash doesn't match previous | `WireError::Chain(...)` — reject |
| Drift exceeds bound | `geometry_drift > max_drift_accepted` | Reject, demand re-probing |
| Geometry hash mismatch | Probe's hash ≠ attestation's hash | Reject, stale probes |
| Agent not in registry | `UnknownAgent(agent_id)` | Reject, unknown agent |
| NaN/Inf in fields | `AttestationError::NaN` or `Infinity` | Reject, corrupt data |
| Integrity violation | `IntegrityViolation { layer, pos }` | Reject, hardware capture tampered |
| Registry integrity fail | `RegistryIntegrity { expected, actual }` | Reject, registry file tampered (S-2) |
| Domain excluded | `DomainExcluded { excluder, target }` | Reject at Phase 0, structural — cannot be overridden |
| Domain not permitted | `DomainNotPermitted { from, target }` | Reject at Phase 0, peer outside declared scope |
| Modes incompatible | `DomainModeIncompatible { a, b }` | Reject at Phase 0, both peers ReadOnly |
| Domain parse error | `DomainParse(String)` | Reject registry load (malformed domain or pattern) |
| Payload too large | `PayloadTooLarge { size, limit }` | Reject frame (N-1) |
| Timestamp future | `TimestampFuture { delta, max }` | Reject, clock skew (S-7) |
| String field too large | `FieldTooLarge { field, size, max }` | Reject attestation (S-13) |
| No signer keys | `Chain("no signer keys provided")` | Config error — cannot verify |

---

## 8. Wire Protocol Framing (Implemented)

The wire protocol uses length-prefixed binary framing, implemented in
`got-wire::frame`. N-1 hardening: `encode()` returns `Result<Vec<u8>, WireError>`.

```
  Frame Format
  ============

  ┌──────────┬─────────┬──────────┬───────────────────┐
  │  Magic   │  Type   │  Length  │     Payload       │
  │  4 bytes │  1 byte │  4 bytes │  variable         │
  │ 0x474F5431│         │  u32 BE  │  ≤ 16 MiB        │
  └──────────┴─────────┴──────────┴───────────────────┘

  FRAME_HEADER_SIZE = 9 bytes
  MAX_PAYLOAD_SIZE  = 16 MiB (16 * 1024 * 1024)

  encode() returns Result<Vec<u8>, WireError>  (N-1)
    → PayloadTooLarge if payload > MAX_PAYLOAD_SIZE or > u32::MAX

  Message Types:
    0x01  ExchangeReq    Initiate attestation exchange
    0x02  ExchangeRsp    Response with verdict
    0x03  VerifyReq      Request verification of attestation
    0x04  VerifyRsp      Verification result
    0x05  ChainReq       Request attestation chain
    0x06  ChainRsp       Chain response
    0xFF  Error          Error with code + message

  Error payload:
    ┌──────────┬──────────────────────┐
    │  code    │  message             │
    │  4 bytes │  UTF-8 string        │
    │  u32 LE  │  (remaining bytes)   │
    └──────────┴──────────────────────┘
```

---

## 9. Mapping to Current Code

| Protocol step | Implementation | Module | Hardening |
|---|---|---|---|
| Self-attest (basic) | `assemble_and_sign(attestation, sk)` | `got-attest` | S-7, S-13, S-20 |
| Self-attest (enclave) | `enclave_pipeline(enclave, capture, ...)` | `got-enclave` | — |
| Self-attest (sidecar) | `MeasurementSidecar::ingest()` → window close | `got-probe::hooks` | N-2 |
| Causal check | `causal_check(probe, h, geom, δ, model_fn, thresh)` | `got-probe::intervention` | — |
| Build exchange request | `build_request(nonce, peer_id, sk, chain, current)` | `got-wire::exchange` | — |
| Build exchange response | `build_response(nonce, peer_id, sk, verdict, ...)` | `got-wire::exchange` | — |
| Validate request | `validate_request(req, own_id, nonce, registry)` | `got-wire::exchange` | S-2, §4 Phase 0 |
| Validate response | `validate_response(rsp, own_id, nonce, registry)` | `got-wire::exchange` | S-2, §4 Phase 0 |
| Domain compatibility check | `check_domain_compatibility(scope_a, scope_b)` | `got-wire::domain` | §4 / Appendix B |
| Full in-memory exchange | `perform_exchange(init_key, ..., resp_key, ..., registry)` | `got-wire::exchange` | S-2, S-8, S-9 |
| Envelope create | `ExchangeEnvelope::create()` | `got-wire::envelope` | S-9: verified=true |
| Envelope verify | `envelope.verify()` | `got-wire::envelope` | S-9: sets verified |
| Envelope deserialise+verify | `ExchangeEnvelope::from_bytes_verified()` | `got-wire::envelope` | S-9 |
| Chain verification | `verify_chain(chain, current, pks, max_drift)` | `got-wire::chain` | S-8: &[VerifyingKey] |
| Trust registry | `TrustRegistry::from_toml()` / `.load()` / `.lookup()` | `got-wire::registry` | S-2: integrity |
| Frame encode | `Frame::encode()` → `Result<Vec<u8>, WireError>` | `got-wire::frame` | N-1: size guard |
| Frame decode | `Frame::decode()` | `got-wire::frame` | — |
| Store attestation | `store.append(attestation, verifying_key)` | `got-store` | atomic + hash |
| Query attestations | `store.query(&filter)` / `store.chain(model_id)` | `got-store` | — |
| Audit model | `store.audit(model_id)` | `got-store` | — |
| Drift measurement | `CausalGeometry::drift_from(&reference)` | `got-core::geometry` | — |
| Distribution shift | `detect_distribution_shift(baseline, current, σ)` | `got-probe::hooks` | N-2 |

---

## 10. What This Does Not Cover

- **Key distribution** — how agents discover each other's public keys. Assumed
  out-of-band (trust registry TOML, PKI, web-of-trust, etc.).
- **Network transport** — the wire framing is defined but actual TCP/TLS
  transport is not implemented. Frames are currently exchanged in-memory.
- **Confidentiality** — attestations are signed but not encrypted. An
  eavesdropper sees probe readings and geometry hashes.
- **Ordering guarantees** — chain integrity assumes delivery in order.
  Out-of-order delivery requires buffering and reordering by parent hash.
- **Liveness** — no heartbeat or timeout mechanism. An agent that stops
  attesting is indistinguishable from one that has crashed.
- **Adversarial agents** — an agent that controls its own model can game
  Frobenius drift metrics. Directional drift analysis (Phase 13) or zero-knowledge
  proof of geometry would be needed for adversarial robustness.
- **Group attestation struct** — the aggregator pattern is described but
  `GroupAttestation` is not yet implemented as a struct.
- **Standardised domain taxonomy** — `got-wire::domain` enforces whatever
  hierarchy the trust registry declares, but the canonical taxonomy of
  competence domains (who maintains it, how new domains are added,
  dispute resolution) remains a governance question outside the protocol.
- **Real hardware TEE** — `MockEnclave` provides the abstraction but actual
  SGX/SEV/H100 integration is not implemented.

---

## 11. Behavioral Exchange Protocol (Tier 0)

For agents monitoring closed-source models, a parallel exchange protocol
operates alongside the geometric exchange:

- **Message types**: `BehavioralExchangeReq` (0x10) / `BehavioralExchangeRsp` (0x11)
- **Agent role**: `"behavioral-observer"` — must be present in TrustRegistry roles
- **Attestation type**: `BehavioralAttestation` (schema "B1") — structurally distinct from `GeometricAttestation`
- **Chain verification**: `verify_behavioral_chain()` — validates parent_hash linkage
- **Trust tier**: Tier 0 (Behavioral) — weaker than geometric attestations:
  - Access: outputs only (no model internals)
  - Determinism: statistical (not byte-identical)
  - Reproducibility: same geometry + same observations → same value space hash
- **Implementation**: `got-wire::behavioral` module + `got-proxy` crate
