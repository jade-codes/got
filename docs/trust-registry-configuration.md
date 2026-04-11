# Trust Registry Configuration

Practical guide to building, tuning, and deploying a `TrustRegistry`
TOML file. The domain scoping / governance thresholds side of the
same file is covered in
[agent-domain-configurations.md](agent-domain-configurations.md);
this document covers everything else — freshness windows, chain
limits, model pinning, roles, certificates, the integrity pin, and
the rollout patterns for moving a registry from PoC to production.

---

## 1. What a trust registry is for

The `TrustRegistry` (`got-wire::registry::TrustRegistry`) is the
root of local policy for an agent. When a peer sends an attestation,
the verifier looks the peer up in its registry and applies the
registry's rules: "do I know you", "is your certificate still
valid", "is your primary domain compatible with mine", "is your
attestation fresh enough", "is your chain short enough", "did you
attest for the model I'm expecting". None of this is carried in
the attestation itself — the attestation is deterministic content,
the registry is the verifier's judgement.

Every registry has two tiers of configuration:

1. **Global `[registry]` section** — freshness windows, chain
   length cap, integrity defaults. These apply to every incoming
   attestation regardless of sender.
2. **Per-agent `[[agents]]` entries** — identity, trust bindings,
   and local policy for one specific peer.

Both live in the same TOML file, which is loaded via
`TrustRegistry::load(path, expected_sha256)` (integrity-checked,
production) or `TrustRegistry::load_unverified(path)` (PoC only,
skips the SHA-256 check).

---

## 2. The `[registry]` section

```toml
[registry]
max_chain_length         = 100    # default
max_envelope_age_secs    = 300    # default (5 minutes)
max_attestation_age_secs = 3600   # default (1 hour)
```

All three fields are optional — the defaults above apply if the
whole section is omitted. Each one is a **ceiling**, not a target —
incoming values that exceed the ceiling cause the exchange to be
rejected.

### `max_chain_length`

How many links (inclusive of the anchor) are allowed in an incoming
attestation chain.

- **Guards against:** resource exhaustion attacks where a malicious
  peer submits a chain of millions of entries to force the verifier
  to walk and cryptographically verify each one.
- **Enforced by:** `validate_request` and `validate_response` before
  `verify_chain` is called. If `req.chain.len() > max_chain_length`,
  the verdict is `Verdict::Rejected` with a reason string.
- **Tuning:** the default of 100 is generous for any realistic
  deployment — a chain grows by one link per model update or
  attestation window, so 100 covers tens of model generations. If
  you run a long-lived agent with a fine-grained chain (one link
  per hour), you may want to raise this; if you run a conservative
  chain (one link per release), 20–50 is ample.
- **Out-of-band pruning:** the protocol does not define chain
  pruning semantics. A long-lived deployment typically archives
  older chain segments, delivers only the last N to peers, and
  relies on governance to vouch for the pre-archive portion.

### `max_envelope_age_secs`

Maximum acceptable age of an `ExchangeEnvelope`'s `timestamp` at
verification time. Measured as `now - envelope.timestamp`.

- **Guards against:** replay attacks — a previously-valid envelope
  re-sent hours or days later.
- **Enforced by:** `envelope.verify(...)` inside `validate_request`
  / `validate_response`. Future timestamps are also rejected
  (S-9 hardening).
- **Tuning by domain class:**

  | Domain class | Suggested `max_envelope_age_secs` |
  |---|---|
  | Critical infrastructure (§5.4) | 60 |
  | Healthcare, finance (regulated) | 120 |
  | Commercial supply chain | 300 (default) |
  | Research / experimental | 600 |

  The tradeoff is straightforward: tighter windows make replay harder
  but impose a real-time-clock requirement on all participants.
  Clock skew across federated deployments typically dominates below
  ~30s, so going below that is usually impractical.

### `max_attestation_age_secs`

Defence-in-depth freshness bound on the attestation itself (not the
envelope).

- **Guards against:** the case where an adversary produces a fresh
  envelope but wraps an old, stale attestation — the envelope would
  pass its own age check, but the attestation inside is from last
  month.
- **Enforced by:** `validate_request` after envelope verification
  succeeds. Reject if `now - attestation.timestamp >
  max_attestation_age_secs`, or if the attestation timestamp is in
  the future.
- **Tuning:** should be **larger** than `max_envelope_age_secs` (an
  attestation can legitimately be cached and re-served through new
  envelopes for some time) but **smaller** than the expected model
  update cadence (otherwise a lazy operator never re-probes). The
  default of 1 hour assumes a cadence of ≥1 hour between probing
  windows; for a sidecar that runs every minute, 5–15 minutes is
  more appropriate.

---

## 3. Per-agent configuration

Each `[[agents]]` entry describes one peer the verifier will accept
exchanges from. Agents not in the registry are rejected with
`WireError::UnknownAgent(...)` before any cryptographic work.

```toml
[[agents]]
id                 = "alice-farm"
public_key         = "64 hex chars = 32-byte Ed25519 verifying key"
max_drift_accepted = 0.05
roles              = ["producer", "verifier"]
expected_model_hash = "64 hex chars, optional"
# ...domain scope + governance thresholds (see agent-domain-configurations.md)
```

### `id`

Human-readable name for the agent. It is **not** the cryptographic
identity — `agent_id` is always computed as `SHA-256(public_key)` and
is what the protocol uses on the wire. The `id` field is for logs,
error messages, and humans reading the TOML.

### `public_key`

The agent's Ed25519 verifying key, as 64 lowercase hex characters
(32 bytes). The TOML loader validates that:
- the string is exactly 64 ASCII hex characters (rejects invalid
  lengths and non-hex);
- the bytes deserialise to a valid Ed25519 curve point (rejects
  garbage or tampered keys).

A registry TOML whose `public_key` cannot be parsed fails to load
with `WireError::RegistryParse(...)`. There is no silent fallback.

### `max_drift_accepted`

The flat per-agent Frobenius drift bound, used as the fallback when
no per-peer-domain `governance_thresholds` pattern matches the
incoming peer's primary domain.

The relationship between `max_drift_accepted` and
`governance_thresholds` is:

1. On each incoming attestation, the verifier calls
   `effective_thresholds(self_entry, peer)`.
2. If `self_entry.governance_table` has a pattern matching
   `peer.domain_scope.primary`, the most-specific match wins and
   its `max_drift` is used.
3. Otherwise, fall back to
   `GovernanceThresholds::permissive(self_entry.max_drift_accepted)`.

So `max_drift_accepted` is the "pre-§8.2 PoC" knob — the value a
verifier applies to peers that are either unscoped or do not match
any of the verifier's governance patterns. A production deployment
typically sets this to a safe default (e.g. 0.05) and then adds
tighter per-domain overrides in `governance_thresholds`.

Default: 0.05 if omitted.

Indicative values (from §7.3, applied either here or in a
matching governance row):

| Domain class | `max_drift_accepted` |
|---|---|
| Critical infrastructure | 0.02 |
| Healthcare | 0.03 |
| Finance (regulated) | 0.05 |
| Commercial supply chain | 0.10 |
| Research / experimental | 0.25 |

### `roles`

A free-form list of strings that label what the agent is authorised
to do. **Most roles are descriptive labels that the protocol does
not itself enforce** — they exist for operators to read and for
application code to check against. The one exception is
`behavioral-observer`, which `got-wire::behavioral::validate_request`
actually checks before accepting a Tier-0 behavioural exchange.

Recognised role names in the current codebase:

| Role | Enforced? | Meaning |
|---|---|---|
| `producer` | no (label) | Agent produces its own geometric attestations |
| `verifier` | no (label) | Agent verifies attestations from peers |
| `aggregator` | no (label) | Agent collects attestations into a group (hub-and-spoke topology, §6 agent-protocol doc) |
| `regulatory-observer` | no (label) | Agent has oversight authority; paired with `InteractionMode::Supervised` in the scope layer |
| `behavioral-observer` | **yes** | Only agents with this role may participate in Tier-0 behavioural exchange (`got-wire::behavioral`). A geometric-only agent without this role is rejected from the behavioural path. |

If you invent a new role for your deployment, nothing in the
protocol will enforce it — you'll need application-level code that
reads `registry.lookup(&agent_id)?.roles` and decides accordingly.

Default: empty list.

### `expected_model_hash`

Optional pin binding an agent to one specific model hash.

- **When set:** `validate_request` / `validate_response` reject any
  attestation whose `model_hash` field is not `Some(expected)`. This
  catches the case where an agent rotates its probes and attestation
  machinery but attests under a different (unauthorised) model.
- **When `None` (default):** the verifier accepts any model hash
  the peer presents, subject to the rest of the checks.
- **Rotation implications:** if you pin `expected_model_hash`, then
  every legitimate model update requires a registry edit to rotate
  the pin. For fleet deployments, treat the pin like any other
  governance decision — it's the registry operator's responsibility
  to push a new pin when the new model is authorised.

Set this when:

- The model is frozen (pre-trained checkpoint, no fine-tuning).
- Governance wants to forbid silent model swaps.
- You're distributing a signed "this agent runs exactly this
  model" claim alongside the registry file.

Leave `None` when:

- The agent self-updates and produces chained attestations — the
  `geometry_drift` bound in `governance_thresholds` is a better
  gate than a hard pin.
- You want to move quickly and revisit hard pinning later.

Format: 64 lowercase hex characters (32 bytes). Same validation
path as `public_key`.

---

## 4. Integrity pinning: `load()` vs `load_unverified()`

`TrustRegistry` exposes two loading functions:

```rust
// Production path.
pub fn load(path: &Path, expected_sha256: &[u8; 32])
    -> Result<Self, WireError>;

// PoC / development path.
pub fn load_unverified(path: &Path) -> Result<Self, WireError>;
```

### `load()`

Reads the file bytes, computes SHA-256 over them, and compares
against `expected_sha256`. On mismatch, returns
`WireError::RegistryIntegrity { expected, actual }` **before any
parsing happens**. Then parses as TOML.

This is the S-2 hardening from the security audit. It means the
trust registry file itself is tamper-evident — an attacker who can
write to the filesystem cannot replace the registry with their own
agents unless they also know the pinned digest.

The `expected_sha256` value must be obtained out-of-band: pinned in
source code, delivered via a secure channel, signed by a separate
identity key, or committed to a Merkle tree of deployment artefacts.
The protocol does not specify how the digest is distributed — that's
an operator decision.

### `load_unverified()`

Reads the file without the digest check. Use this only in
development or when the registry file lives inside a trust boundary
(e.g. inside an enclave's measured filesystem) that already
guarantees integrity.

**Do not use `load_unverified()` in production**, even if you don't
have a digest distribution pipeline set up yet — if nothing else,
the presence of `load()` calls in deployment scripts provides a
searchable audit trail for later hardening.

### Integrity workflow

Typical operator flow for a production registry:

1. Governance body publishes `registry.toml` + its pinned digest
   `registry.sha256` via a signed release channel (e.g. git
   repository, signed artefact bundle, etc.).
2. Each verifier is configured with the pinned digest as a constant
   (compiled in, or read from a trusted config file alongside its
   own signing key).
3. On startup, the verifier calls `TrustRegistry::load(path,
   expected_sha256)`.
4. When the registry changes, the governance body publishes a new
   `(registry.toml, registry.sha256)` pair. Verifiers hot-swap their
   registry on signal, or on scheduled restart.

---

## 5. Certificates and CRLs

The PoC builds an `AgentCertificate` type (in `got-wire::certificate`)
that binds an agent's public key to an identity via a CA signature.
Certificates are **optional** and **not configured via TOML**: the
TOML loader always builds a registry with an empty `ca_public_keys`
vector, meaning certificates are ignored.

To enable certificate enforcement, programmatically set
`registry.ca_public_keys` after loading and use
`add_agent_verified(entry)` instead of `add_agent(entry)`:

```rust
use got_wire::registry::TrustRegistry;

let mut registry = TrustRegistry::load(&path, &expected_digest)?;
registry.ca_public_keys.push(governance_ca_public_key);
registry.add_agent_verified(agent_entry)?;  // validates the cert
```

When `ca_public_keys` is non-empty, `add_agent_verified`:

1. Requires the `entry.certificate` to be present.
2. Verifies the certificate's Ed25519 signature.
3. Checks that the issuer is in `ca_public_keys`.
4. Checks that the certificate's subject key matches
   `entry.public_key`.
5. Checks that the certificate is not in any loaded CRL.

During exchange, `validate_agent_certificate(&agent_id, now)` checks
that the certificate is still within its `not_before` / `not_after`
window and has not been revoked.

### Loading a CRL

```rust
let crl: CertificateRevocationList = /* parse from file */;
registry.load_crl(crl)?;
```

`load_crl` verifies the CRL's signature against one of the
`ca_public_keys` before installing it. Multiple CRLs can be
installed — the verifier checks every CRL when validating a
certificate.

### Key rotation and certificates

`TrustRegistry::apply_rotation(&key_rotation)` performs a
cryptographically bound key rotation:

1. Verifies both old and new key signatures over the rotation
   record.
2. Removes the old entry from the registry.
3. Inserts a new entry under the rotated key.
4. **Preserves `domain_scope` and `governance_table` from the old
   entry** — rotation cannot change the agent's domain or
   thresholds. A primary domain change is a governance event, not
   a cryptographic one.

This means: a compromised agent can rotate its key to a new one,
but cannot use the rotation to escalate into a different domain or
relax its governance bounds.

---

## 5b. Cross-registry federation (§14.5)

A single `TrustRegistry` is fine for a single-jurisdiction deployment.
For multi-jurisdictional deployments — e.g. an EU healthcare authority,
a US FDA registry, and a UK MHRA registry that all need to resolve
agents in the same exchange pipeline — `got-wire::federation` provides
a scoped composition layer.

The full federation design from the protocol paper involves signed
cross-registry vouching, async sync between authorities, revocation
propagation, and arbitration policies for jurisdictional conflicts.
That is multi-week work and out of scope for the PoC. **What this
module provides instead is the composition layer**: an explicit-
priority list of `TrustRegistry`s, conflict reporting on policy
divergence, and a `resolve()` method that produces a single flat
`TrustRegistry` the rest of the exchange pipeline can consume
unchanged. No protocol changes, no signature changes to
`validate_request`, no new wire format.

### The model

```rust
use got_wire::federation::{FederatedRegistry, NamedRegistry};
use got_wire::registry::TrustRegistry;

let eu = TrustRegistry::load(&eu_path, &eu_digest)?;
let us = TrustRegistry::load(&us_path, &us_digest)?;
let uk = TrustRegistry::load(&uk_path, &uk_digest)?;

// Simple form: no integrity pin, no operator key, no vouchers.
// Equivalent to the pre-vouching API.
let federation = FederatedRegistry::from_members(vec![
    NamedRegistry::unverified("eu-healthcare", 0, eu),
    NamedRegistry::unverified("us-fda",        1, us),
    NamedRegistry::unverified("uk-mhra",       2, uk),
]);

// Inspect any policy conflicts before resolving.
for w in federation.validate_consistency() {
    eprintln!("federation warning: {w}");
}

// Resolve to a single TrustRegistry the rest of the pipeline uses.
let resolved = federation.resolve()?;

// Hand `resolved` to perform_exchange / validate_request as normal.
```

### Resolution semantics

- **Cryptographic identity is universal.** Because
  `agent_id = SHA-256(public_key)`, two registries that list the
  same `agent_id` are by definition talking about the same key.
  The federation never has to arbitrate identity — only the *policy*
  attached to that identity (drift bounds, scope, governance,
  expected_model_hash, certificate).
- **Lower priority wins.** Priority is a Linux-style nice value:
  `priority = 0` beats `priority = 1`. The first member registry
  in priority order to claim a given `agent_id` provides the
  resolved entry; lower-priority members are silently overridden
  for that ID.
- **Globals come from the head.** The resolved registry's
  `max_chain_length`, `max_envelope_age_secs`, and
  `max_attestation_age_secs` are inherited from the highest-priority
  member. If you need different freshness windows per jurisdiction,
  the federation layer cannot express that — you would need
  per-agent overrides, which the protocol does not currently
  support at this layer.
- **CAs and CRLs are unioned.** The resolved registry trusts every
  CA any member trusts and applies every CRL any member loaded.
  This is intentional: revocation in one jurisdiction propagates
  to the federated view immediately.
- **Distinct agents are preserved.** Agents that appear in only
  one member registry are simply added to the resolved set.

### Conflict reporting

`validate_consistency()` returns a `Vec<FederationWarning>` listing
every material disagreement between member registries. The detected
conflicts are:

| `FederationWarning` variant | What it means |
|---|---|
| `MaxDriftMismatch` | Same agent registered with different `max_drift_accepted` flat fallback values |
| `ExpectedModelHashMismatch` | Same agent pinned to different model hashes |
| `DomainScopeMismatch` | Same agent declares different `primary_domain`, `permitted_domains`, or `exclusion_domains` |
| `GovernanceTableMismatch` | Same agent has different `governance_thresholds` rows |

Field comparisons are *order-insensitive* for the permitted /
exclusion / governance lists — reordering the same set of patterns
is not flagged as a conflict.

### What conflict reporting does not catch

- **Differences in the registry's global `[registry]` section.**
  `max_chain_length` etc. are not per-agent, so the federation layer
  ignores them; the highest-priority member's globals win silently.
- **Roles that differ across registries.** `roles` is a
  free-form list mostly used as a label; it is not part of the
  conflict signature.
- **Certificate metadata that differs.** Certificates are checked
  separately at exchange time via the resolved CA / CRL union.

### Signed cross-registry vouching

Operational composition is enough for trusted environments where
every operator already knows which other operators they accept. For
deployments where federation members need to *cryptographically*
declare cross-trust — so a verifier can detect an unauthorised member
that was slipped into the federation file — the
`FederationVoucher` type provides one-hop signed vouching.

The model: each member registry's *operator* (distinct from the
agents inside the registry) holds an Ed25519 keypair. When operator A
decides to vouch for operator B's registry, A signs a
`FederationVoucher` over B's registry file digest. Any holder of the
voucher and A's verifying key can confirm that A endorsed B's
registry at a specific digest, with optional expiry.

```rust
use got_wire::federation::{FederatedRegistry, FederationVoucher, NamedRegistry};
use got_wire::registry::{compute_agent_id, TrustRegistry};

// Operators hold separate keys from any agents in their registries.
let eu_operator: ed25519_dalek::SigningKey = load_eu_operator_key()?;
let us_operator: ed25519_dalek::SigningKey = load_us_operator_key()?;

// Load and pin the registry files.
let eu_digest = sha256(&std::fs::read(&eu_path)?);
let us_digest = sha256(&std::fs::read(&us_path)?);
let eu = TrustRegistry::load(&eu_path, &eu_digest)?;
let us = TrustRegistry::load(&us_path, &us_digest)?;

// EU operator vouches for the US registry.
let voucher = FederationVoucher::create(
    compute_agent_id(&eu_operator.verifying_key()),
    us_digest,
    "us-fda",
    now_unix(),
    now_unix() + 30 * 86_400,        // 30-day expiry
    "ratified at G7 healthcare summit 2026-Q2",
    &eu_operator,
)?;

// Build the federation with operator keys + vouchers attached.
let federation = FederatedRegistry::from_members(vec![
    NamedRegistry {
        name: "eu-healthcare".into(),
        priority: 0,                                  // root of trust
        registry: eu,
        digest: Some(eu_digest),
        operator_key: Some(eu_operator.verifying_key()),
        vouchers: vec![],                             // root needs no voucher
    },
    NamedRegistry {
        name: "us-fda".into(),
        priority: 1,
        registry: us,
        digest: Some(us_digest),
        operator_key: Some(us_operator.verifying_key()),
        vouchers: vec![voucher],
    },
]);

// Verify the chain before resolving.
for w in federation.verify_vouchers(now_unix()) {
    eprintln!("voucher warning: {w}");
}
```

`FederatedRegistry::verify_vouchers(now_unix)` walks every non-lead
member and checks that at least one of its `vouchers`:

1. Has an `issuer_id` matching the operator key of some
   higher-priority member.
2. Verifies cryptographically against that operator's verifying key.
3. Signs over a `subject_digest` matching the member's `digest`
   field.
4. Has not expired at `now_unix`.

Members that satisfy all four checks are considered vouched. The
lead (priority 0) is the root of trust — it does not need a voucher.
Failures emit `VoucherWarning` variants:

| Variant | Cause |
|---|---|
| `Missing` | Non-lead member has no voucher that passes all four checks. The warning lists which higher-priority members *could* have signed one. |
| `Expired` | Voucher's `not_after` has passed. |
| `SignatureInvalid` | Signature does not verify against the named issuer's key. |
| `DigestMismatch` | Voucher signs over a different digest than the on-disk file. Either the file was tampered with or the voucher is stale relative to a registry update. |
| `UnknownIssuer` | `issuer_id` does not correspond to any higher-priority member's `operator_key`. |
| `NoDigestPin` | A non-lead member's `digest` is `None`, so vouchers cannot bind to it. |

### Vouching limitations

The scoped vouching layer is **one-hop** and **structural**. It
does not provide:

- **Multi-hop chains.** A vouches B, B vouches C — the verifier
  does not aggregate this into "A transitively vouches C". Each
  member must carry a voucher from a higher-priority neighbour.
- **Voucher revocation lists.** The only revocation mechanism is
  expiry. Set short `not_after` deadlines and re-issue.
- **Operator key rotation.** Operators are pinned by the verifying
  key in `NamedRegistry::operator_key`. Rotating an operator key
  requires rebuilding the federation file with the new key and
  re-issuing every voucher signed by the old key.
- **Async sync between authorities.** Vouchers are static artefacts
  loaded with the federation file; live sync would need a separate
  fetch protocol.

### When to use federation

- **Multi-jurisdictional deployments** where authoritative registries
  exist in more than one place.
- **Staged rollouts** where you want a "draft" registry to override
  the "stable" registry for a subset of agents during a migration.
- **Audit/operations split** where the security team maintains a
  high-priority allowlist that overrides operational defaults from
  a lower-priority deployment registry.

### When **not** to use federation

- Single-jurisdiction deployment with one authoritative registry —
  use `TrustRegistry::load` directly.
- You need real-time sync of revocation across federated authorities.
  Federation here is a static-snapshot composer; refresh requires
  reloading the member registries from disk.
- You need multi-hop voucher chains, voucher CRLs, or live operator
  key rotation. These are real federation features but they are not
  implemented in the scoped layer.

---

## 6. Deployment patterns

### PoC / development

```toml
[registry]
# Use defaults.

[[agents]]
id = "alice"
public_key = "..."
max_drift_accepted = 0.10          # permissive
roles = ["producer", "verifier"]
# no domain_scope: compat check short-circuits to "pass"
# no expected_model_hash: any model accepted
# no governance_thresholds: flat max_drift_accepted applies
```

- Loaded via `load_unverified()` or `load()` with a locally-computed
  digest.
- No certificates.
- Matches the behaviour of every test in the codebase.

### Production: commercial supply chain

```toml
[registry]
max_chain_length         = 100
max_envelope_age_secs    = 300
max_attestation_age_secs = 3600

[[agents]]
id = "supplier-001"
public_key = "..."
max_drift_accepted = 0.10
roles = ["producer", "verifier"]
expected_model_hash = "..."        # pin to authorised model
primary_domain = "agriculture.supply-chain"
permitted_domains = [
  { pattern = "agriculture.*", mode = "cooperative" },
]

[[agents.governance_thresholds]]
pattern = "agriculture.*"
max_drift = 0.10
require_chain = true
```

- Loaded via `load()` with pinned digest.
- Certificates optional but recommended.
- Governance `require_chain = true` mandates Tier 2 minimum.

### Production: healthcare (strict)

```toml
[registry]
max_chain_length         = 100
max_envelope_age_secs    = 120
max_attestation_age_secs = 600    # 10 min: catches stale attestations fast

[[agents]]
id = "diag-clinical-01"
public_key = "..."
max_drift_accepted = 0.03
roles = ["producer", "verifier"]
expected_model_hash = "..."        # hard pin
primary_domain = "healthcare.diagnostic-advisory"
permitted_domains = [
  { pattern = "healthcare.drug-interaction", mode = "advisory" },
]

[[agents.governance_thresholds]]
pattern = "healthcare.*"
max_drift                 = 0.03
min_confidence            = 0.80
min_causal_score          = 0.85
require_chain             = true
require_causal_validation = true
```

- Loaded via `load()` with pinned digest from a signed release.
- CA-signed certificates enforced programmatically.
- CRL checked at every exchange.
- Tight envelope/attestation windows to surface staleness quickly.
- Tier-3 causal validation mandatory across all of `healthcare.*`.

### Production: critical infrastructure

Same as healthcare but:

- `max_envelope_age_secs = 60` (near-real-time per §5.4).
- `max_drift = 0.02` in governance.
- `min_causal_score = 0.85`.
- `expected_model_hash` hard-pinned for every agent.

---

## 7. Migrating from permissive to strict

Tightening a live registry without breaking running agents requires
staging. The protocol's rejection-is-absolute semantics mean a
change that makes any existing attestation invalid causes those
agents to stop cooperating immediately — so every change needs a
deployment plan that updates producers before (or simultaneously
with) verifiers.

Recommended phased rollout:

1. **Baseline measurement.** For one release cycle, log what the
   permissive registry *would* reject if tightened — record observed
   drift, confidence, chain length, and envelope age for every
   incoming attestation.
2. **Set thresholds slightly above observed maxima.** Pick
   thresholds that bound the 99th-percentile of observed values,
   plus a safety margin. Apply to a staging registry. Run the
   deployment end-to-end against the staging registry for at least
   one update cycle.
3. **Roll out to production in-place.** Push the tightened registry
   + new digest to all verifiers simultaneously. Monitor for the
   first hour; any rejected agent is either a genuine outlier or a
   threshold that needs one more safety margin.
4. **Tighten further only after measurements allow.** Each
   subsequent tightening round repeats the measurement →
   threshold-setting cycle. Don't try to reach paper-indicative
   values in one jump.

Model-hash pinning is always a special case:

- Pinning for the first time requires every pinned agent to restart
  with the authorised model binary, or its attestations start
  failing the moment the registry swaps.
- Rotating a pin (new model version) requires a governance decision
  plus a simultaneous registry + agent update.

---

## 8. Troubleshooting checklist

When a verifier rejects an attestation, walk the error reason
backwards through the validation pipeline:

| Rejection reason contains | Check |
|---|---|
| `unknown agent` | Sender's public key isn't in the registry; key hash mismatch, or you forgot to add them. |
| `certificate validation failed` | Cert expired, revoked, or issuer not in `ca_public_keys`. |
| `domain incompatible` | Phase 0 §4 check — inspect both sides' `primary_domain`, `permitted_domains`, and `exclusion_domains`. See `agent-domain-configurations.md` for worked examples. |
| `envelope verification failed` | Nonce mismatch (replay), peer_agent_id mismatch (relay), timestamp out of window, or bad Ed25519 signature. |
| `attestation signature invalid` | Attestation was tampered in transit, or the peer rotated their key without updating the registry. |
| `attestation timestamp is in the future` | Clock skew; ensure NTP is running on both sides. |
| `attestation too old` | `max_attestation_age_secs` exceeded; sender hasn't re-probed recently. |
| `model_hash does not match registry policy` | `expected_model_hash` pin disagrees with the attestation's claimed `model_hash`. Either a legitimate model rotation (update the pin) or an unauthorised swap. |
| `chain too long` | `max_chain_length` exceeded; prune or raise the ceiling. |
| `chain verification failed` | Parent hash broken, key rotation not registered, drift exceeds `max_drift`, or model_id inconsistent across chain. |
| `schema required for domain ...` | Governance `require_chain` / `require_causal_validation` not satisfied — peer sent a Tier-1 attestation where Tier 2+ is mandated. |
| `confidence X below minimum` | Governance `min_confidence` exceeded; peer's probes are below the configured reliability bar. |
| `causal consistency X below minimum` | Governance `min_causal_score` exceeded. |
| `attestation domain_scope_declaration disagrees with registry` | §2.1 binding — peer's embedded scope doesn't match its registry entry. Either a relay attack or a misconfigured agent. |
| `registry integrity check failed` | The loaded TOML doesn't match the pinned SHA-256. File tampered or wrong pin. |

---

## 9. Quick reference

| What you want | Where to set it |
|---|---|
| Freshness window (envelope) | `[registry] max_envelope_age_secs` |
| Freshness window (attestation) | `[registry] max_attestation_age_secs` |
| Chain length cap | `[registry] max_chain_length` |
| Per-agent flat drift bound | `[[agents]] max_drift_accepted` |
| Per-peer-domain drift / confidence / tier | `[[agents.governance_thresholds]]` |
| Which peers this agent will interact with | `[[agents]] primary_domain` + `permitted_domains` + `exclusion_domains` |
| Which peers this agent refuses | `[[agents]] exclusion_domains` |
| Pin to a specific model | `[[agents]] expected_model_hash` |
| Regulator / oversight semantics | `InteractionMode::Supervised` in `permitted_domains` |
| Cert enforcement | Programmatic: set `registry.ca_public_keys`, call `add_agent_verified` |
| CRL enforcement | Programmatic: `registry.load_crl(crl)` |
| Registry file integrity | `TrustRegistry::load(path, expected_sha256)` |

---

## 10. References

- [agent-domain-configurations.md](agent-domain-configurations.md)
  — domain scoping and per-peer-domain governance thresholds.
- [architecture-agent-protocol.md](architecture-agent-protocol.md)
  §5 — trust registry TOML format.
- [architecture-agent-protocol.md](architecture-agent-protocol.md)
  §5c — per-domain governance thresholds implementation detail.
- Protocol paper §7.3 — chain drift bounds.
- Protocol paper §8 — trust registry and governance architecture.
- `got-wire/src/registry.rs` — TOML loader and the typed
  `TrustRegistry` / `AgentEntry` structures.
- `got-wire/src/certificate.rs` — certificate and CRL types,
  key-rotation verification.
