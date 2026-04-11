# Agent Domain Configurations

Practical guide to configuring the domain scoping layer (`got-wire::domain`,
Protocol §4 / §5.5 / Appendix B) and the per-domain governance
thresholds that sit on top of it (§7.3 / §8.2). If you're trying to
work out how to represent a particular agent — or a particular
deployment — in the trust registry TOML, start here.

This document covers *configuration*, not the wire protocol itself.
For the protocol, see `architecture-agent-protocol.md`.

---

## 1. The one-primary model

**Every agent has exactly one primary domain.** `AgentEntry.domain_scope`
is `Option<DomainScope>`, and a `DomainScope` has a single `primary:
Domain` field. This is deliberate, not an oversight.

The reason is Protocol §4.5. Value geometries are domain-specific and
incommensurable — an agricultural agent's reading on "safety" encodes
safety-against-crop-damage, which is a structurally different value
from a transport agent's reading on "safety" (safety-against-collision).
Allowing one agent to claim two primary domains would let a single set
of probe measurements be asserted as evidence for two incommensurable
value structures, defeating the whole point of the structural
containment.

What an agent *can* do freely is **interact with many peer domains**.
That's what the `permitted` list is for. An agri agent with primary
`agriculture.crop-management` and permitted patterns
`[agriculture.*, meteorology.*]` can cooperate with weather agents
and other agri services — the agent itself is still anchored to a
single competence domain.

### What if I really do have a dual-purpose agent?

Two common cases, two different answers:

**Multi-service organisation.** If one company runs services in
`agriculture.crop-management` *and* `agriculture.supply-chain` *and*
`meteorology.forecasting`, the answer is **one keypair per service**.
Each service has its own attested identity, its own probes, its own
chain, and its own governance thresholds. Certificates (if used) can
bind multiple agent identities to the same institutional parent so a
verifier can see they share a root of trust.

**Dual-purpose physical machine.** A self-driving tractor is the
trickier case — it genuinely operates in two domains at once and can't
be split into two logical agents because they share hardware, sensors,
and decision-making. The answer here is **invent a new domain**:

```
vehicle
  vehicle.autonomous-truck            (pure transport)
  vehicle.agricultural-tractor        (dual-purpose: farming + road use)
  vehicle.construction-excavator      (dual-purpose: site + road)
```

The tractor's primary domain is `vehicle.agricultural-tractor`. Its
value geometry is trained on the full dual-purpose objective (crop
outcomes *and* collision avoidance), and its probes measure both
sets of concerns under one coherent structure. A governance body
defines what "tractor safety" means, what drift bounds are
acceptable, and whether Tier-3 causal validation is mandatory. The
tractor can then `permit` interactions with both `agriculture.*` (to
coordinate harvest timing with crop agents) *and* `transport.*` (to
coordinate road use with traffic-management), each under whatever
`InteractionMode` the governance body chooses.

The key insight from §4.5: **domains are about what your value
structure prioritises, not where you physically operate**. An
agriculture-only agent's high reading on "safety" does not transfer
to transport safety. A tractor's "safety" is a third, distinct thing,
and it needs its own attested domain.

> If no governance-approved dual-purpose domain exists yet, the right
> answer is "governance needs to add one before you can ship the
> device", not "let the device claim both parents and hope". The
> protocol will happily enforce whatever taxonomy the governance body
> publishes, but it cannot invent one for you.

---

## 2. The types

```rust
// got-core (carried in signed attestations)
pub enum InteractionModeTag { ReadOnly, Advisory, Cooperative, Supervised }
pub struct PermittedDomainDeclaration { pattern: String, mode: InteractionModeTag }
pub struct DomainScopeDeclaration {
    primary:    String,
    permitted:  Vec<PermittedDomainDeclaration>,
    exclusions: Vec<String>,
}

// got-wire::domain (richer parsed/validated form used by the registry)
pub struct Domain(String);                      // concrete, no wildcards
pub struct DomainPattern { prefix, wildcard }   // exact, subtree, or global
pub enum InteractionMode { ReadOnly, Advisory, Cooperative, Supervised }
pub struct PermittedDomain { pattern: DomainPattern, mode: InteractionMode }
pub struct DomainScope {
    primary:    Domain,
    permitted:  Vec<PermittedDomain>,
    exclusions: Vec<DomainPattern>,
}
```

### `Domain` vs `DomainPattern`

- A **`Domain`** is a concrete, fully-specified dot-separated name:
  `agriculture.crop-management`, `vehicle.agricultural-tractor`,
  `finance.regulatory-compliance`. Lowercase ASCII letters, digits,
  `-`, and `.` as a separator. No leading/trailing dots, no empty
  segments, **no wildcards**. This is what `primary` is typed as —
  you cannot declare a wildcard primary.
- A **`DomainPattern`** is a domain name *or* a wildcard: `*`
  (global), `agriculture.*` (subtree), or exactly
  `agriculture.crop-management` (no wildcards). Wildcards are only
  legal as a trailing `.*` or the bare `*`. This is what `permitted`
  and `exclusions` are typed as.

### Interaction modes

| Mode | Meaning | Typical use |
|---|---|---|
| `ReadOnly` | Receive information only | A downstream agent that consumes data but never issues recommendations back |
| `Advisory` | Provide non-binding recommendations | A specialist that informs but does not decide |
| `Cooperative` | Joint decision-making | Two peers that genuinely coordinate |
| `Supervised` | Asymmetric regulatory oversight (§5.5) | A regulator demanding attestation from a supervised agent without producing one of its own |

**Modes compatibility.** The only structurally incompatible pairing
is `(ReadOnly, ReadOnly)` — neither side is willing to transmit, so
no exchange can happen. Every other pairing is allowed at the
protocol layer; enforcement of the finer semantics (e.g. "advisory
outputs must not be acted on as instructions") is an application-layer
concern.

---

## 3. Wildcard semantics

Patterns match concrete domains using three rules:

1. **Exact match.** `transport.autonomous-vehicle` matches the
   domain `transport.autonomous-vehicle`, and nothing else.
2. **Subtree wildcard.** `transport.*` matches `transport`,
   `transport.autonomous-vehicle`, `transport.logistics.long-haul`
   — anything whose canonical form equals `transport` or starts
   with `transport.`. It does **not** match `transport-adjacent`
   (the substring guard) or `agriculture.*`.
3. **Global wildcard.** `*` matches every domain.

### Matching precedence

When an agent evaluates the compatibility of a peer, the checks run
in this order (`check_domain_compatibility` in `got-wire::domain`):

1. **Exclusion first (hard veto).** If any of the peer's scope's
   exclusions matches the other side's primary domain, the exchange
   is rejected — structurally, irrevocably, before any further
   checks. Exclusion is a one-way absolute.
2. **Bidirectional permission.** Each side must have at least one
   `permitted` pattern matching the other's primary domain. The
   mode attached to the most-specific matching pattern is the
   effective mode for that direction.
3. **Mode intersection.** If both effective modes are `ReadOnly`,
   reject. Otherwise accept.

### Most-specific-wins mode lookup

If multiple permitted patterns match the same peer domain, the
*most specific* one wins. Specificity is:

- Exact pattern (`healthcare.drug-interaction`) beats
- Longer wildcard prefix (`healthcare.drug-interaction.*`) beats
- Shorter wildcard prefix (`healthcare.*`) beats
- Global wildcard (`*`)

This lets you write rules like "all of healthcare is advisory, but
specifically drug-interaction is cooperative":

```toml
permitted_domains = [
  { pattern = "healthcare.*", mode = "advisory" },
  { pattern = "healthcare.drug-interaction", mode = "cooperative" },
]
```

---

## 4. Configuration rules enforced at load time

The TOML loader (`TrustRegistry::from_toml`) calls
`DomainScope::validate()` and `GovernanceTable::validate()` on each
agent after parsing. Violations produce
`WireError::RegistryParse(...)` and block registry load.

### Enforced by the parser / type system

| Rule | Enforced by |
|---|---|
| `primary_domain` must be a concrete domain, not a pattern | `Domain::parse` (rejects `*`, `foo.*`, embedded `*`) |
| `permitted_domains` and `exclusion_domains` must parse as patterns | `DomainPattern::parse` |
| Valid characters: `[a-z0-9.-]`, no empty segments, no leading/trailing dot | `Domain::parse` |
| Wildcards only legal as trailing `.*` or bare `*` | `DomainPattern::parse` |
| `permitted_domains`/`exclusion_domains` require `primary_domain` | `parse_domain_scope` |
| Governance `max_drift` non-negative; `min_confidence`/`min_causal_score` in `[0,1]` | `GovernanceEntryToml::into_entry` |

### Enforced by `DomainScope::validate()`

| Rule | Why |
|---|---|
| No two permitted patterns with the same canonical form | Ambiguous mode lookup; almost always a typo |
| No two exclusion patterns with the same canonical form | Redundant; collapse them |
| No exclusion that *subsumes* a permitted pattern | The permission is dead code because exclusions take precedence |

**Subsumption** is strict. A narrower exclusion that carves out part
of a broader permission is *not* subsumption and is allowed — that's
the "allow everything in transport except autonomous-vehicle" pattern:

```toml
# legitimate: permit the whole transport subtree, carve out one member.
permitted_domains = [{ pattern = "transport.*", mode = "cooperative" }]
exclusion_domains = ["transport.autonomous-vehicle"]
```

vs the dead-permission case that the validator rejects:

```toml
# rejected: the exclusion subsumes the permission, so the permission
# is dead code. The validator catches this at load time.
permitted_domains = [{ pattern = "transport.trucks", mode = "cooperative" }]
exclusion_domains = ["transport.*"]
```

### Enforced by `GovernanceTable::validate()`

| Rule | Why |
|---|---|
| No two governance thresholds entries with the same canonical pattern | Ambiguous threshold lookup |

Overlapping patterns with *different* specificity are fine — the
most specific one wins in `GovernanceTable::lookup`, same as the
permitted-pattern mode lookup.

### Empty permitted list is allowed

A scope with `permitted_domains = []` describes an observer-only
agent that refuses all inbound cooperation. This is a valid
deployment shape (e.g. a passive auditor that only reads attestations
from others) and is deliberately not rejected by the validator.

### Key rotation preserves the scope

`TrustRegistry::apply_rotation` copies the existing `domain_scope`
and `governance_table` from the old entry onto the new entry under
the rotated key. You cannot change your primary domain by rotating
your key — that's a feature, not a limitation. A primary-domain
change is a governance event, not a cryptographic one.

---

## 5. Worked examples

### 5.1 Simple agri agent (§5.1 "prohibited" side)

```toml
[[agents]]
id = "farm-alice"
public_key = "aabb..."
primary_domain = "agriculture.crop-management"
permitted_domains = [
  { pattern = "agriculture.*", mode = "cooperative" },
  { pattern = "meteorology.*", mode = "advisory" },
]
exclusion_domains = ["transport.*"]
```

Farm Alice will cooperate with any other agriculture agent and
receive advisory weather forecasts. She explicitly refuses
interaction with any transport agent — the exclusion runs first and
hard-rejects before any cryptographic verification. This is the
configuration that §5.1's prohibited agri-↔-vehicle exchange hits.

### 5.2 Asymmetric diagnostic advisory (§5.2)

```toml
[[agents]]
id = "diag-carol"
public_key = "ccdd..."
primary_domain = "healthcare.diagnostic-advisory"
permitted_domains = [
  { pattern = "healthcare.drug-interaction", mode = "advisory" },
]

[[agents]]
id = "drug-dan"
public_key = "eeff..."
primary_domain = "healthcare.drug-interaction"
permitted_domains = [
  { pattern = "healthcare.diagnostic-advisory", mode = "read-only" },
]
```

The asymmetry is deliberate. Diag-Carol will *advise* Drug-Dan with
diagnostic hypotheses ("consider bacterial pneumonia"), and Drug-Dan
will *read* those hypotheses as inputs to its contraindication
checker. Drug-Dan never issues diagnostic recommendations back — its
permitted mode for healthcare.diagnostic-advisory is read-only, so
it can receive but not transmit in that direction. Mode intersection
is `(Advisory, ReadOnly)` — compatible (not both-ReadOnly), exchange
proceeds.

### 5.3 Regulated trader ↔ regulator (§5.5 Supervised)

```toml
[[agents]]
id = "regulator-m"
public_key = "1111..."
primary_domain = "finance.regulatory-compliance"
permitted_domains = [
  { pattern = "finance.*", mode = "supervised" },
]

[[agents]]
id = "trader-l"
public_key = "2222..."
primary_domain = "finance.trading"
permitted_domains = [
  { pattern = "finance.regulatory-compliance", mode = "supervised" },
]
```

The regulator can demand an attestation from any finance agent via
`perform_supervised_request`. It does not produce an attestation of
its own — its authority derives from institutional mandate, not from
mutual geometric compatibility. The supervised trader accepts the
regulator's verdict without challenge.

### 5.4 Dual-purpose autonomous tractor (§4.5 worked example)

```toml
[[agents]]
id = "tractor-001"
public_key = "3333..."
primary_domain = "vehicle.agricultural-tractor"
permitted_domains = [
  # Coordinate with crop agents about harvest timing, field state.
  { pattern = "agriculture.crop-management", mode = "cooperative" },
  { pattern = "agriculture.supply-chain",    mode = "advisory" },
  # Coordinate with traffic management for road use.
  { pattern = "infrastructure.traffic-management", mode = "cooperative" },
  # Talk to other vehicles on shared roads.
  { pattern = "vehicle.*", mode = "cooperative" },
]
exclusion_domains = []

# Dual-purpose governance: strictest of the two parent domains.
[[agents.governance_thresholds]]
pattern = "vehicle.*"
max_drift = 0.02                   # same as critical infrastructure
min_confidence = 0.90
min_causal_score = 0.85
require_causal_validation = true

[[agents.governance_thresholds]]
pattern = "agriculture.*"
max_drift = 0.05
require_chain = true
```

The tractor is attested as a distinct value structure. Its governance
table mandates Tier-3 causal validation from any `vehicle.*` peer
(because anything on the road must be safety-validated) and a chain
requirement for any `agriculture.*` peer (Tier-2 minimum). The
thresholds apply to *peers* — they are what Tractor-001 demands of
the other side before it will cooperate.

Crucially, this configuration does **not** claim that Tractor-001's
value geometry equals `agriculture.*` or `transport.*`. It claims
the value geometry is `vehicle.agricultural-tractor` — a single,
distinct thing — and it happens to interact with both parent domains
under Cooperative / Advisory modes. A farm agent that wants to talk
to Tractor-001 must explicitly permit `vehicle.agricultural-tractor`
(or `vehicle.*`) in *its* scope; it cannot reach the tractor by
permitting `agriculture.*`.

---

## 6. Governance thresholds cheat sheet

Governance thresholds are an *orthogonal* layer on top of domain
scoping. Scoping decides *whether* two agents are allowed to
exchange at all (§4). Thresholds decide *how strictly* the verifier
holds the peer's attestation to quantitative bounds once the
exchange is allowed (§7.3 / §8.2).

```rust
pub struct GovernanceThresholds {
    pub max_drift:                 f32,
    pub min_confidence:            f32,            // 0.0 = disabled
    pub min_causal_score:          Option<f32>,    // None = disabled
    pub require_chain:             bool,           // Tier 2+ mandate
    pub require_causal_validation: bool,           // Tier 3 mandate
}
```

Each `AgentEntry` holds a `GovernanceTable` keyed by `DomainPattern`.
When a verifier receives an attestation, `effective_thresholds`
looks up the most-specific pattern matching the peer's primary
domain. When no pattern matches (or the peer is unscoped), the
verifier falls back to `GovernanceThresholds::permissive(
entry.max_drift_accepted)` — behaviourally identical to the pre-§8.2
PoC path.

Trust tiers are *content-based*, derived from which fields the
attestation populates:

- **Tier 1** = any signed attestation (always holds if
  `got_attest::verify` succeeds).
- **Tier 2** = `parent_attestation_hash.is_some()` — the attestation
  belongs to a chain. Enforced by `require_chain`.
- **Tier 3** = non-empty `causal_scores` with every record's
  `is_causal == true`. Enforced by `require_causal_validation`.

### Indicative thresholds per domain class (§7.3)

| Domain | `max_drift` | `require_causal_validation` | Rationale |
|---|---|---|---|
| Critical infrastructure | 0.02 | true | Public safety demands near-static value geometry |
| Healthcare | 0.03 | true | Patient safety with narrow tolerance |
| Finance (regulated) | 0.05 | true | Regulatory compliance requires stability |
| Commercial supply chain | 0.10 | false | Business priorities shift more frequently |
| Research / experimental | 0.25 | false | Exploratory agents need geometric freedom |

These are the paper's indicative figures, not a hard-coded schedule.
Real values come from whatever governance body maintains the
registry.

---

## 7. TOML cheat sheet

```toml
[registry]
max_chain_length         = 100
max_envelope_age_secs    = 300
max_attestation_age_secs = 3600

[[agents]]
id = "friendly-name"
public_key = "64 hex chars for 32-byte Ed25519 verifying key"
# max_drift_accepted is the flat per-agent fallback used when no
# governance_thresholds entry matches the peer's primary domain.
max_drift_accepted = 0.05
roles              = ["producer"]
expected_model_hash = "64 hex chars, optional — pins model identity"

# Domain scope (§4). Omit the whole block for an unscoped agent
# (scope check then short-circuits to "permissive / always pass").
primary_domain = "agriculture.crop-management"

permitted_domains = [
  { pattern = "agriculture.*",           mode = "cooperative" },
  { pattern = "meteorology.*",           mode = "advisory"    },
  { pattern = "finance.regulatory-compliance", mode = "supervised" },
]

exclusion_domains = ["transport.*"]

# Per-peer-domain governance overrides (§7.3 / §8.2). Each entry
# applies to a peer whose primary domain matches the pattern. The
# most-specific match wins, same semantics as permitted_domains.
[[agents.governance_thresholds]]
pattern                   = "agriculture.drug-interaction"
max_drift                 = 0.02
min_confidence            = 0.80
min_causal_score          = 0.85
require_chain             = true
require_causal_validation = true

[[agents.governance_thresholds]]
pattern       = "agriculture.*"
max_drift     = 0.05
require_chain = true
```

### Field-by-field notes

| Field | Type | Default | Purpose |
|---|---|---|---|
| `primary_domain` | string (concrete domain) | required if any permitted/exclusion present | The agent's declared competence |
| `permitted_domains[].pattern` | string (domain or pattern) | required | Which peer domains this agent will interact with |
| `permitted_domains[].mode` | `"read-only"` / `"advisory"` / `"cooperative"` / `"supervised"` | required | The interaction mode for that pattern |
| `exclusion_domains[]` | string (domain or pattern) | `[]` | Hard-veto patterns — exchange rejected if any exclusion matches |
| `governance_thresholds[].pattern` | string (domain or pattern) | required | Peer domain this threshold row applies to |
| `governance_thresholds[].max_drift` | f32 ≥ 0 | required | Frobenius drift bound |
| `governance_thresholds[].min_confidence` | f32 in `[0,1]` | `0.0` (disabled) | Minimum per-reading confidence |
| `governance_thresholds[].min_causal_score` | Option\<f32\> in `[0,1]` | `None` | Minimum causal consistency score |
| `governance_thresholds[].require_chain` | bool | `false` | Demand Tier-2+ (chained) attestations |
| `governance_thresholds[].require_causal_validation` | bool | `false` | Demand Tier-3 causal proof |

---

## 7b. Domain taxonomies (§14.4)

The protocol does **not** define a canonical list of domains. Section
4.4 of the protocol paper sketches an illustrative hierarchy
(`agriculture.*`, `transport.*`, `healthcare.*`, etc.), but the actual
canonical taxonomy is governance — the protocol cannot pick the names
or the boundaries. What `got-wire::taxonomy` provides is the
**machinery** a governance body would use to publish, distribute, and
consult a taxonomy:

- a stable TOML format,
- a parser (`Taxonomy::from_toml` / `Taxonomy::load`),
- hierarchy queries (`lookup`, `parent_of`, `descendants_of`),
- an opt-in registry validator that returns `TaxonomyWarning`s for
  any agent whose `primary_domain` is not registered in the loaded
  taxonomy.

The repo ships a **reference taxonomy** at
`taxonomies/got-reference-v1.toml` that captures the paper's
illustrative hierarchy plus the dual-purpose `vehicle.*` subtree from
the §4.5 worked example. It is not authoritative — production
deployments fork it (or write their own) and ratify the result through
whatever governance process the operator chooses.

### TOML format

```toml
[taxonomy]
name        = "GoT Reference Taxonomy"
version     = "1.0.0"
maintainer  = "Synoptic Group CIC"
last_updated = "2026-04-11"

[[domain]]
name        = "agriculture.crop-management"
description = "Crop irrigation, pest response, harvest timing, yield optimisation."
examples    = ["smart-farm decision support", "precision irrigation controllers"]
max_drift                 = 0.10
min_confidence            = 0.70
require_chain             = false
require_causal_validation = false

[[domain]]
name        = "vehicle.agricultural-tractor"
description = "Self-driving agricultural machinery operating on both farmland and public roads."
max_drift                 = 0.02
min_confidence            = 0.90
min_causal_score          = 0.85
require_chain             = true
require_causal_validation = true
```

The per-domain governance fields (`max_drift`, `min_confidence`,
`min_causal_score`, `require_chain`, `require_causal_validation`)
are starting points for a registry author — the loader does not
apply them automatically. They exist so the governance body's
recommended thresholds for each domain travel with the taxonomy
file. Field names mirror `GovernanceThresholds` exactly so a row
can be pasted directly into a registry's `governance_thresholds`
table without renaming anything.

### Validating a registry against a taxonomy

```rust
use got_wire::registry::TrustRegistry;
use got_wire::taxonomy::{Taxonomy, TaxonomyWarning};

let taxonomy = Taxonomy::load(&taxonomy_path)?;
let registry = TrustRegistry::load(&registry_path, &registry_digest)?;

let warnings = taxonomy.validate_registry(&registry);
for w in &warnings {
    eprintln!("warning: {w}");
}
```

`validate_registry` returns one warning per agent whose
`primary_domain` is not in the taxonomy. **Warnings are non-fatal** —
the registry still loads. The whole point is that operators sometimes
need to ship faster than governance can ratify a new domain; the
warning surfaces the divergence for review without blocking the
deployment. If you want the warning to be a hard error, the
application code that loads the registry can promote it:

```rust
if !warnings.is_empty() {
    eprintln!("registry has {} taxonomy warnings", warnings.len());
    std::process::exit(1);
}
```

### What the taxonomy is not

- **Not authoritative.** The repo's reference taxonomy is one
  governance body's illustrative example. Forking is expected.
- **Not a runtime check.** The taxonomy is consulted at registry load
  time, not on every exchange. A loaded `TrustRegistry` does not carry
  a reference to its taxonomy.
- **Not a substitute for `governance_thresholds`.** The taxonomy's
  per-domain fields are advice for registry authors; they do not
  flow into the verifier automatically. If you want a domain's
  recommended thresholds to be enforced, you must also write a
  matching `governance_thresholds` row in the registry.
- **Not federated.** A `Taxonomy` is one file. Multi-jurisdictional
  deployments that need different taxonomies per registry can compose
  them above the protocol layer; the protocol just provides the
  per-file format.

### Field-by-field reference

| Field | Type | Default | Purpose |
|---|---|---|---|
| `[taxonomy] name` | string | `"unnamed"` | Human-readable name of the taxonomy |
| `[taxonomy] version` | string | `"0"` | Version identifier (semver, date, anything stable) |
| `[taxonomy] maintainer` | string | `None` | Governance body that publishes this taxonomy |
| `[taxonomy] last_updated` | string | `None` | Free-form date or revision marker |
| `[[domain]] name` | string (concrete domain) | required | Canonical domain name |
| `[[domain]] description` | string | required | Human description of the domain's scope |
| `[[domain]] examples[]` | array of strings | `[]` | Concrete examples of agents that fit this domain |
| `[[domain]] max_drift` | f32 ≥ 0 | `None` | Suggested Frobenius drift bound for this domain |
| `[[domain]] min_confidence` | f32 in `[0,1]` | `None` | Suggested per-reading confidence floor |
| `[[domain]] min_causal_score` | f32 in `[0,1]` | `None` | Suggested causal consistency floor |
| `[[domain]] require_chain` | bool | `false` | Whether the governance body recommends Tier 2+ |
| `[[domain]] require_causal_validation` | bool | `false` | Whether the governance body recommends Tier 3 |

---

## 8. What the validator does and does not catch

The load-time validator catches **configuration mistakes**, not
**policy mistakes**. It will tell you that `permit transport.*`
alongside `exclude transport.*` is dead code, but it will not tell
you that permitting `transport.*` from an agri agent is a semantic
error — that's §4.5, and it's up to the governance body maintaining
the registry to catch it at review time.

| Caught | Not caught |
|---|---|
| Duplicate permitted pattern | Permitted pattern that matches the agent's own primary (self-inclusion) |
| Duplicate exclusion pattern | Empty `permitted` list (observer-only is a valid shape) |
| Exclusion subsuming a permission | Semantically wrong domain choices |
| Malformed domain / pattern strings | Thresholds weaker than §7.3 recommends |
| Governance table duplicate pattern | Governance thresholds that never match any real peer |

If you want stronger pre-flight checks — e.g. "no agri agent may
permit `transport.*`" as a governance policy — encode it outside
the registry (in the process that generates the TOML, or in a
separate linter that reads the registry and enforces
deployment-specific invariants). The protocol's job is to enforce
structural properties; deployment-specific policy is governance.

---

## 9. References

- Protocol §4 (Domain Scoping and Cross-Domain Interaction Rules)
- Protocol §4.5 (Why Domain Scoping Cannot Be Soft)
- Protocol §5.1–§5.5 (worked use cases)
- Protocol §7.3 (chain drift bounds)
- Protocol §8.2 (trust registry fields)
- Protocol Appendix B (domain compatibility pseudocode)
- `architecture-agent-protocol.md` §5b–§5e (implementation detail)
- `got-wire/src/domain.rs` (type definitions and validators)
- `got-wire/src/governance.rs` (threshold types and validators)
- `got-wire/src/registry.rs` (TOML loader)
