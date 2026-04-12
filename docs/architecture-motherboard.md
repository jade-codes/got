# Agent-to-Agent Motherboard Diagram

A "circuit board" view of two GoT agents wired together. Each agent is a
**board** with functional **chips** (crates/modules) connected by internal
**buses** (function calls / data flow). The two boards connect through the
**wire protocol bus** (got-wire framing). This view covers both the
**trust protocol** (attestation, verification, chain walking) and the
**comms protocol** (frame encoding, envelope exchange, request/response).

Codebase: 353 tests passing. All security hardening (S-2…S-21, N-1…N-3) reflected.

---

## 1. Motherboard — Single Agent

Each agent's board has identical layout, mirrored during exchange.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  AGENT BOARD  (e.g. "Alice")                                                ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │  TRUST PROTOCOL BUS (internal data paths)                           │    ║
║  └──┬───────┬──────────┬──────────┬──────────┬──────────┬──────────┬───┘    ║
║     │       │          │          │          │          │          │         ║
║  ┌──┴───┐┌──┴────┐ ┌───┴───┐ ┌───┴────┐ ┌───┴───┐ ┌───┴────┐ ┌──┴──────┐  ║
║  │GEOM  ││PROBE  │ │ENCLAVE│ │ATTEST  │ │STORE  │ │REGISTRY│ │EXCHANGE │ │PROXY   │  ║
║  │      ││       │ │       │ │        │ │       │ │        │ │         │ │        │  ║
║  │got   ││got    │ │got    │ │got     │ │got    │ │got-wire│ │got-wire │ │got     │  ║
║  │-core ││-probe │ │-encl  │ │-attest │ │-store │ │::reg   │ │::exch   │ │-proxy  │  ║
║  │      ││       │ │       │ │        │ │       │ │        │ │         │ │        │  ║
║  │Causal││Measrmt│ │Mock   │ │assemble│ │Memory │ │Trust   │ │build_   │ │Proxy   │  ║
║  │Geom  ││Hook   │ │Enclave│ │_and_   │ │Store  │ │Registry│ │request()│ │Session │  ║
║  │Φ=UᵀU ││Collctr│ │enclave│ │sign()  │ │File   │ │lookup()│ │build_   │ │Value   │  ║
║  │drift ││Sidecar│ │_pipe  │ │verify()│ │Store  │ │S-2:SHA │ │response │ │Space   │  ║
║  │      ││Stats  │ │_line()│ │S-7/13  │ │append │ │integr  │ │validate │ │Welford │  ║
║  │S-21: ││N-2:   │ │       │ │S-20    │ │query  │ │age_sec │ │_request │ │+EWMA   │  ║
║  │model ││poison │ │       │ │merkle  │ │audit  │ │mdl_hsh │ │_response│ │3-signal│  ║
║  │hash  ││recov  │ │       │ │_root() │ │chain  │ │        │ │perform_ │ │deviatn │  ║
║  │Optn  ││       │ │       │ │        │ │       │ │        │ │exchange │ │B1 attest│  ║
║  └──┬───┘└──┬────┘ └───┬───┘ └───┬────┘ └───┬───┘ └───┬────┘ └──┬──────┘ └──┬─────┘  ║
║     │       │          │          │          │          │          │         ║
║  ───┴───────┴──────────┴──────────┴──────────┴──────────┴──────────┴───      ║
║     │                                                                       ║
║  ┌──┴─────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   ║
║  │  SIGNING UNIT      │  │  ENVELOPE UNIT       │  │  FRAMING UNIT       │   ║
║  │  Ed25519 (dalek)   │  │  got-wire::envelope  │  │  got-wire::frame    │   ║
║  │                    │  │                      │  │                     │   ║
║  │  SigningKey sk     │  │  create() → env      │  │  encode() → Result  │   ║
║  │  VerifyingKey pk   │  │  verify()            │  │  N-1: size guard    │   ║
║  │  sign(msg, sk)     │  │  from_bytes()        │  │  MAX = 16 MiB      │   ║
║  │  verify(msg, sig)  │  │  from_bytes_verified │  │  decode()           │   ║
║  │                    │  │  is_verified()       │  │  MAGIC: 0x474F5431  │   ║
║  │  zeroize on drop   │  │  S-9: verified flag  │  │  msg types: 0x01-FF│   ║
║  └────────────────────┘  └──────────┬───────────┘  └──────────┬──────────┘   ║
║                                     │                         │              ║
║  ┌──────────────────────────────────┴─────────────────────────┴───────────┐  ║
║  │  CHAIN VERIFICATION UNIT                                               │  ║
║  │  got-wire::chain                                                       │  ║
║  │                                                                        │  ║
║  │  verify_chain(chain, current, &[VerifyingKey], max_drift)              │  ║
║  │  S-8: accepts multiple signer keys for key rotation                    │  ║
║  │  → ChainVerdict { length, max_drift_observed }                         │  ║
║  │                                                                        │  ║
║  │  Checks: signature (any key) · parent hash linkage · drift bound       │  ║
║  │          model_id consistency · anchor has no parent · completeness     │  ║
║  └────────────────────────────────────────────────────┬───────────────────┘  ║
║                                                       │                      ║
║  ═══════════════════════════════════════════════════ WIRE OUT ═══════════════ ║
╚══════════════════════════════════════════════════════╤════════════════════════╝
                                                       │
                                                   CHANNEL
                                                  (untrusted)
```

---

## 2. Two Boards Wired Together — Full Exchange

```
╔═══════════════════════════════════════╗         ╔═══════════════════════════════════════╗
║           ALICE BOARD                 ║         ║            BOB BOARD                  ║
║                                       ║         ║                                       ║
║  ┌────────┐  ┌────────┐  ┌────────┐  ║         ║  ┌────────┐  ┌────────┐  ┌────────┐  ║
║  │GEOMETRY│  │ PROBE  │  │ENCLAVE │  ║         ║  │GEOMETRY│  │ PROBE  │  │ENCLAVE │  ║
║  │  Φ_A   │  │CollHook│  │MockEncl│  ║         ║  │  Φ_B   │  │CollHook│  │MockEncl│  ║
║  │  U_A   │  │Sidecar │  │pipeline│  ║         ║  │  U_B   │  │Sidecar │  │pipeline│  ║
║  └───┬────┘  └───┬────┘  └───┬────┘  ║         ║  └───┬────┘  └───┬────┘  └───┬────┘  ║
║      │           │           │        ║         ║      │           │           │        ║
║  ════╧═══════════╧═══════════╧════    ║         ║  ════╧═══════════╧═══════════╧════    ║
║  │  TRUST PROTOCOL BUS (local)  │    ║         ║  │  TRUST PROTOCOL BUS (local)  │    ║
║  ════╤═══════════╤═══════════╤════    ║         ║  ════╤═══════════╤═══════════╤════    ║
║      │           │           │        ║         ║      │           │           │        ║
║  ┌───┴────┐  ┌───┴───┐  ┌───┴────┐   ║         ║  ┌───┴────┐  ┌───┴───┐  ┌───┴────┐   ║
║  │ATTEST  │  │ STORE │  │REGISTRY│   ║         ║  │ATTEST  │  │ STORE │  │REGISTRY│   ║
║  │assemble│  │append │  │lookup  │   ║         ║  │assemble│  │append │  │lookup  │   ║
║  │_and_   │  │query  │  │S-2:SHA │   ║         ║  │_and_   │  │query  │  │S-2:SHA │   ║
║  │sign()  │  │audit  │  │ages,   │   ║         ║  │sign()  │  │audit  │  │ages,   │   ║
║  │S-7/13  │  │chain  │  │mdl_hsh │   ║         ║  │S-7/13  │  │chain  │  │mdl_hsh │   ║
║  │S-20    │  │       │  │        │   ║         ║  │S-20    │  │       │  │        │   ║
║  └───┬────┘  └───┬───┘  └───┬────┘   ║         ║  └───┬────┘  └───┬───┘  └───┬────┘   ║
║      │           │           │        ║         ║      │           │           │        ║
║  ════╧═══════════╧═══════════╧════    ║         ║  ════╧═══════════╧═══════════╧════    ║
║  │        SIGNING BUS             │   ║         ║  │        SIGNING BUS             │   ║
║  ════╤═══════════════════╤════════    ║         ║  ════╤═══════════════════╤════════    ║
║      │                   │            ║         ║      │                   │            ║
║  ┌───┴────────┐  ┌───────┴────────┐   ║         ║  ┌───┴────────┐  ┌───────┴────────┐   ║
║  │ SIGNING    │  │   ENVELOPE     │   ║         ║  │ SIGNING    │  │   ENVELOPE     │   ║
║  │ Ed25519    │  │   create()     │   ║         ║  │ Ed25519    │  │   create()     │   ║
║  │ sk_A, pk_A │  │   verify()     │   ║         ║  │ sk_B, pk_B │  │   verify()     │   ║
║  │ zeroize    │  │   S-9:verified │   ║         ║  │ zeroize    │  │   S-9:verified │   ║
║  └────────────┘  └───────┬────────┘   ║         ║  └────────────┘  └───────┬────────┘   ║
║                          │            ║         ║                          │            ║
║  ┌───────────────────────┴────────┐   ║         ║  ┌───────────────────────┴────────┐   ║
║  │     CHAIN VERIFIER             │   ║         ║  │     CHAIN VERIFIER             │   ║
║  │  verify_chain(&[pk_B], drift)  │   ║         ║  │  verify_chain(&[pk_A], drift)  │   ║
║  │  S-8: multi-key rotation       │   ║         ║  │  S-8: multi-key rotation       │   ║
║  └───────────────┬────────────────┘   ║         ║  └───────────────┬────────────────┘   ║
║                  │                    ║         ║                  │                    ║
║  ┌───────────────┴────────────────┐   ║         ║  ┌───────────────┴────────────────┐   ║
║  │       EXCHANGE CONTROLLER      │   ║         ║  │       EXCHANGE CONTROLLER      │   ║
║  │  build_request()               │   ║         ║  │  validate_request()            │   ║
║  │  validate_response()           │   ║         ║  │  build_response()              │   ║
║  └───────────────┬────────────────┘   ║         ║  └───────────────┬────────────────┘   ║
║                  │                    ║         ║                  │                    ║
║  ┌───────────────┴────────────────┐   ║         ║  ┌───────────────┴────────────────┐   ║
║  │           FRAMING              │   ║         ║  │           FRAMING              │   ║
║  │  Frame::encode() → Result      │   ║         ║  │  Frame::decode()               │   ║
║  │  N-1: PayloadTooLarge guard    │   ║         ║  │  BadMagic / Incomplete check   │   ║
║  │  MAGIC=0x474F5431              │   ║         ║  │  MAGIC=0x474F5431              │   ║
║  └───────────────┬────────────────┘   ║         ║  └───────────────┬────────────────┘   ║
║                  │                    ║         ║                  │                    ║
╚══════════════════╪════════════════════╝         ╚══════════════════╪════════════════════╝
                   │                                                │
                   │    ┌─────────────────────────────────┐         │
                   │    │    WIRE PROTOCOL CHANNEL         │         │
                   │    │    (untrusted network / pipe)    │         │
                   │    │                                  │         │
                   ├────►  ExchangeReq frame (0x01)   ────┼────────►│
                   │    │    [magic][type][len][payload]   │         │
                   │    │                                  │         │
                   │◄───┼── ExchangeRsp frame (0x02)  ◄───┤─────────┤
                   │    │    [magic][type][len][payload]   │         │
                   │    │                                  │         │
                   │    │  Optional follow-up:             │         │
                   │    │    ChainReq (0x05) / Rsp (0x06)  │         │
                   │    │    VerifyReq (0x03) / Rsp (0x04) │         │
                   │    │    Error (0xFF)                   │         │
                   │    └─────────────────────────────────┘         │
                   │                                                │
```

---

## 3. Data Flow Through the Board — Trust Protocol

This traces a single attestation from creation to verified exchange.

```
Step 1: GEOMETRY CHIP                      Step 5: FRAMING CHIP
  Model → U matrix                           encode(ExchangeReq)
  Φ = UᵀU (CausalGeometry)                  → [magic][0x01][len][json]
  drift_from(&reference)                     N-1: Result, size guard
         │                                          │
         ▼                                          ▼
Step 2: PROBE CHIP                         Step 6: WIRE OUT
  CollectingHook.on_activations()            → untrusted channel →
  MeasurementSidecar.ingest()
  ActivationStats (Welford online)         Step 7: FRAMING CHIP (peer)
  detect_distribution_shift()                decode() → ExchangeReq
  N-2: poison recovery                             │
         │                                          ▼
         ▼                                 Step 8: REGISTRY CHIP (peer)
Step 3: ATTEST CHIP                          lookup(agent_id) → AgentEntry
  assemble_and_sign(                         S-2: SHA-256 verified on load
    attestation, &signing_key)               check: max_attestation_age_secs
  S-7: timestamp ≤ 300s future               §4 Phase 4 (defence in depth
  S-13: strings ≤ 256 bytes                    domain re-verify):
  S-20: layers ≤ 1024                            check_domain_compatibility(
                                                   peer_scope, self_scope)
                                                 Supervised pair OK (§5.5)
                                                 → DomainExcluded /
                                                   DomainNotPermitted /
                                                   DomainModeIncompatible
                                                 (skipped if either side
                                                  has no scope declared)
                                                 NOTE: Phase 1 pre-flight
                                                 (check_domain_before_
                                                  exchange) already ran
                                                    │
                                                    ▼
                                            §7.3 / §8.2 Governance:
                                              effective_thresholds(
                                                self_entry, peer)
                                              enforce_governance(
                                                max_drift, min_confidence,
                                                require_chain,
                                                require_causal_validation)
                                                    │
                                                    ▼
                                            §2.1 Attestation scope binding:
                                              check_attestation_scope_binding(
                                                peer, attestation)
                                              embedded declaration must
                                              match registry domain_scope
                                                    │
                                                    ▼
                                            Step 9: ENVELOPE CHIP (peer)
  schema_version = SCHEMA_VERSION            verify(envelope, pk_sender)
  UnsignedAttestation → sign                 S-9: from_bytes_verified()
  → GeometricAttestation (signed)            check: nonce, peer_id,
         │                                     attest_hash, chain_root,
         ▼                                     timestamp freshness
Step 4: ENVELOPE + EXCHANGE CHIPS                   │
  ExchangeEnvelope::create(                         ▼
    nonce, peer_id, attest_hash,           Step 10: CHAIN CHIP (peer)
    chain_root, timestamp, sk)              verify_chain(chain, current,
  S-9: verified = true                        &[pk_sender], max_drift)
  build_request(nonce, peer_id,             S-8: try all trusted keys
    sk, chain, current_attest)              → ChainVerdict { length,
         │                                      max_drift_observed }
         ▼                                          │
  ──── TO FRAMING ────                              ▼
                                           Step 11: EXCHANGE CHIP (peer)
                                             validate_request()
                                             → verdict: Accepted / Rejected
                                             build_response()
                                                    │
                                                    ▼
                                           Step 12: FRAMING → WIRE → back
                                             ExchangeRsp to initiator
                                             initiator validate_response()
```

---

## 4. Comms Protocol — Frame-Level Detail

```
  INITIATOR                                         RESPONDER
  ─────────                                         ─────────
  build_request()                                   (waiting)
       │                                                │
       ▼                                                │
  Frame::encode(                                        │
    MessageType::ExchangeReq,                           │
    serde_json::to_vec(&request)                        │
  )                                                     │
  → Result<Vec<u8>, WireError>                          │
    ┌─────────────────────────────┐                     │
    │ 47 4F 54 31 │ 01 │ xx xx  │  ← 9-byte header    │
    │   magic     │type│ length │                      │
    ├─────────────────────────────┤                     │
    │ { "agent_id": "aa...",     │  ← JSON payload     │
    │   "envelope": { ... },     │    (≤ 16 MiB)       │
    │   "chain": [ ... ],        │                      │
    │   "current_attestation":   │                      │
    │     { ... } }              │                      │
    └─────────────────────────────┘                     │
       │                                                │
       │──── send over channel ───────────────────────► │
       │                                                │
       │                                      Frame::decode(buf)
       │                                      → Frame { msg_type,
       │                                           payload }
       │                                                │
       │                                      serde_json::from_slice
       │                                      → ExchangeRequest
       │                                                │
       │                                      validate_request(
       │                                        req, self_id,
       │                                        None, &registry)
       │                                                │
       │                                      build_response(
       │                                        nonce, peer_id,
       │                                        sk, verdict,
       │                                        chain, attest,
       │                                        reason)
       │                                                │
       │                                      Frame::encode(
       │                                        ExchangeRsp,
       │                                        &response_json)
       │                                                │
       │◄─── receive response ────────────────────────  │
       │                                                │
  Frame::decode(buf)                                    │
  → ExchangeResponse                                   │
       │                                                │
  validate_response(                                    │
    rsp, self_id,                                       │
    original_nonce,                                     │
    &registry)                                          │
       │                                                │
  match verdict:                                        │
    Accepted → cooperate                                │
    Rejected → refuse + log reason                      │
    Error    → log + retry or abort                     │
```

---

## 5. Security Hardening Annotations (Pin Map)

Each "pin" on the motherboard that was hardened during audit:

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    SECURITY PIN MAP                          │
  │                                                              │
  │   S-2  ●── Registry SHA-256 integrity on load               │
  │   S-7  ●── Timestamp future guard (≤ 300s)                  │
  │   S-8  ●── verify_chain(&[VerifyingKey]) multi-key rotation  │
  │   S-9  ●── Envelope verified flag + from_bytes_verified()    │
  │   S-13 ●── String field length bounds (≤ 256 bytes)          │
  │   S-20 ●── Layer count cap (≤ 1024)                          │
  │   S-21 ●── model_hash: Option<[u8; 32]>                     │
  │   N-1  ●── Frame::encode() → Result with size guard          │
  │   N-2  ●── Mutex poison recovery (unwrap_or_else)            │
  │   N-3  ●── CLI anyhow migration (not shown on board)         │
  │                                                              │
  │   Location on board:                                         │
  │     S-2  → REGISTRY chip                                     │
  │     S-7  → ATTEST chip (assemble_and_sign)                   │
  │     S-8  → CHAIN VERIFIER                                    │
  │     S-9  → ENVELOPE unit                                     │
  │     S-13 → ATTEST chip (assemble_and_sign)                   │
  │     S-20 → ATTEST chip (assemble_and_sign)                   │
  │     S-21 → GEOMETRY chip (CausalGeometry/Attestation)        │
  │     N-1  → FRAMING unit                                      │
  │     N-2  → PROBE chip (CollectingHook)                       │
  └──────────────────────────────────────────────────────────────┘
```

---

## 6. Trust Level Heat Map

Visualises the trust boundary across the two-board system.

```
  ╔════════════════════════════╗   ╔════════════════════════════╗
  ║ ALICE (fully trusted zone) ║   ║ BOB (fully trusted zone)   ║
  ║                            ║   ║                            ║
  ║  ██████  ENCLAVE          ║   ║  ██████  ENCLAVE          ║
  ║  ██████  (signing key     ║   ║  ██████  (signing key     ║
  ║          never leaves)    ║   ║          never leaves)    ║
  ║                            ║   ║                            ║
  ║  ▓▓▓▓▓▓  ATTEST + STORE  ║   ║  ▓▓▓▓▓▓  ATTEST + STORE  ║
  ║  ▓▓▓▓▓▓  (signed data)   ║   ║  ▓▓▓▓▓▓  (signed data)   ║
  ║                            ║   ║                            ║
  ║  ░░░░░░  GEOMETRY + PROBE ║   ║  ░░░░░░  GEOMETRY + PROBE ║
  ║  ░░░░░░  (raw readings)   ║   ║  ░░░░░░  (raw readings)   ║
  ║                            ║   ║                            ║
  ╠════════════════════════════╣   ╠════════════════════════════╣
  ║  ▒▒▒▒▒▒  FRAMING          ║   ║  ▒▒▒▒▒▒  FRAMING          ║
  ║  ▒▒▒▒▒▒  (serialised)     ║   ║  ▒▒▒▒▒▒  (serialised)     ║
  ╚════════════╤═══════════════╝   ╚═══════════════╤════════════╝
               │                                   │
               │  ╔════════════════════════════╗    │
               │  ║   UNTRUSTED CHANNEL        ║    │
               └──╢   (any network / pipe)     ╟────┘
                  ║   (eavesdrop / tamper)      ║
                  ╚════════════════════════════╝

  Legend:
    ██████  Highest trust — hardware isolation (TEE)
    ▓▓▓▓▓▓  High trust — cryptographically signed
    ░░░░░░  Local trust — unsigned raw data
    ▒▒▒▒▒▒  Boundary — serialised, not yet verified by peer
    ╔═════╗  Untrusted — assume adversarial
```

---

## 7. Three-Agent Formation (Motherboard Backplane)

When three agents cooperate, the backplane connects all boards:

```
  ┌──────────────────┐
  │  ALICE BOARD      │
  │  (pk_A, sk_A)     │
  └────────┬─────────┘
           │
     ══════╪══════════════════════════════════════
     │     │     BACKPLANE (wire protocol bus)   │
     │     │                                      │
     │     ├──── ExchangeReq/Rsp ──── to BOB     │
     │     ├──── ExchangeReq/Rsp ──── to CAROL   │
     │     │                                      │
     ══════╪════════╪═══════════════╪═════════════
           │        │               │
  ┌────────┴───┐ ┌──┴──────────┐ ┌─┴────────────┐
  │ ALICE (self)│ │  BOB BOARD  │ │ CAROL BOARD   │
  │            │ │  (pk_B)     │ │ (pk_C)        │
  │            │ │             │ │ AGGREGATOR     │
  │            │ │             │ │ role           │
  └────────────┘ └─────────────┘ └───────────────┘

  Each connection on the backplane:
    Alice ←→ Bob:    mutual exchange (symmetric)
    Alice ←→ Carol:  mutual exchange (symmetric)
    Bob   ←→ Carol:  mutual exchange (symmetric)

    OR (aggregator pattern):
    Alice  → Carol:  submit attest_A
    Bob    → Carol:  submit attest_B
    Carol  → Alice:  group_attestation
    Carol  → Bob:    group_attestation
```

---

## 8. Key Takeaways

1. **Symmetric design** — every agent has identical board layout; roles
   (producer, verifier, aggregator) are registry metadata, not code changes.
2. **Trust boundary at signing** — raw geometry/probe data is local; only
   signed attestations cross the wire.
3. **Defence in depth** — envelope (S-9), chain (S-8), registry (S-2),
   attest bounds (S-7/S-13/S-20), framing (N-1), concurrency (N-2)
   each guard a different attack surface.
4. **Key rotation** — chain verification accepts `&[VerifyingKey]` (S-8),
   so agents can rotate keys without breaking chains.
5. **Replay resistance** — nonce + peer_id + timestamp in envelope prevents
   relay and replay attacks.
6. **Transport agnostic** — framing is defined, transport is pluggable.
   Currently in-memory; TCP/TLS is a deployment concern.
