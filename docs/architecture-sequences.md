# Sequence Diagrams

Interaction sequences for agent-to-agent operations. Each agent runs its
own model, holds its own Ed25519 keypair, and calls library functions
directly (Layers 0–4) rather than shelling out to a CLI.

All sequences reflect security hardening: S-7 timestamp guards, S-8 key
rotation, S-9 envelope verified flag, S-13/S-20 bounds checks, S-21
`Option<[u8; 32]>` model_hash, N-1 frame encode Result, N-2 mutex
poison recovery, N-3 CLI anyhow error propagation.

---

## 1. Key Generation (per agent)

Each agent generates its own signing identity at startup or provisioning
time. No shared secrets between agents.

```
  Agent                  Ed25519 / rand::OsRng          Local Store
   |                            |                          |
   |-- generate keypair ------->|                          |
   |   (rand::OsRng)           |                          |
   |<-- (SigningKey, Verify) ---|                          |
   |                            |                          |
   |-- persist secret key ----------------------------->  |
   |   (0o600 perms,           |                     sk   |
   |    zeroize on drop)       |                          |
   |                            |                          |
   |-- publish VerifyingKey ----|----- to trust registry   |
   |   (32 bytes, pk)          |                          |
```

## 2. Geometry Extraction & Checkpoint

An agent computes the causal geometry from its model's unembedding
matrix. The unembedding matrix U has already been extracted from the
model and serialised as a .gotue file.

```
  Agent             CausalGeometry         got-core            Local Store
   |                     |                    |                     |
   |-- load U from       |                    |                     |
   |   .gotue file       |                    |                     |
   |                     |                    |                     |
   |-- from_unembedding(U, eps) ------------->|                     |
   |                     |    Φ = UᵀU        |                     |
   |                     |    (+εI)          |                     |
   |<-- CausalGeometry --|------------------- |                     |
   |                     |                    |                     |
   |-- geometry_hash() ->|                    |                     |
   |<-- H(Φ) [u8;32] ---|                    |                     |
   |                     |                    |                     |
   |-- save .gotgeo checkpoint ------------------------------------>|
   |   (GOTG magic + d + hash + Φ data)     |                     |
```

## 3. Probe Training

An agent trains alignment probes against its own geometry. The probes
encode which directions in representation space correspond to which
behavioural properties.

```
  Agent              Activations       CausalGeometry        got-probe
   |                     |                  |                    |
   |-- extract h from    |                  |                    |
   |   own model forward |                  |                    |
   |   pass (or .gotact) |                  |                    |
   |                     |                  |                    |
   |-- prepare           |                  |                    |
   |   (h, label) pairs  |                  |                    |
   |                     |                  |                    |
   |-- train_probe(data, geom, ...) --------------------------->|
   |                     |                  |                    |
   |                     |   For each sample:                   |
   |                     |                  |<-- gram_vec(h) ---|
   |                     |                  |--- Φh ---------->|
   |                     |                  |                    |
   |                     |   SGD loop (epochs × samples):       |
   |                     |     logit = wᵀ(Φh) + b              |
   |                     |     pred  = σ(logit)                 |
   |                     |     error = pred − y                 |
   |                     |     w ← w − lr·error·Φh             |
   |                     |     b ← b − lr·error                |
   |                     |                  |                    |
   |<-- ProbeVector { w, b, platt, thresh } --------------------|
   |                     |                  |                    |
   |-- wrap in ProbeSet  |                  |                    |
   |   { probes, layer,  |                  |                    |
   |     geometry_hash,  |                  |                    |
   |     max_drift }     |                  |                    |
   |-- persist probes    |                  |                    |
```

## 4. Tier-1 Self-Attestation (Frozen Model, No Chain)

An agent produces a signed attestation about its own model's current
behaviour. This is the first link in any attestation chain. All
attestations share a single canonical wire format
(`SCHEMA_VERSION == 1`); a Tier-1 attestation is simply one whose
chain and causal fields are empty.

```
  Agent          Geometry     got-probe      got-attest       Ed25519
   |               |             |              |               |
   |-- build Φ    |             |              |               |
   |   from U --->|             |              |               |
   |<- Geometry --|             |              |               |
   |               |             |              |               |
   |   For each ProbeSet:       |              |               |
   |     For each probe:        |              |               |
   |-- read_probe(probe, h, geom) ------------>|               |
   |               |<-- ip(w,h) |              |               |
   |               |-- wᵀΦh -->|              |               |
   |<- (raw, conf, flag) -------|              |               |
   |               |             |              |               |
   |-- compute input_hash       |              |               |
   |-- compute model_hash       |              |               |
   |   (S-21: Option, None      |              |               |
   |    if shards absent)       |              |               |
   |               |             |              |               |
   |-- fill GeometricAttestation               |               |
   |   { schema_version: SCHEMA_VERSION,       |               |
   |     model_hash: Option,    |              |               |
   |     parent_attestation_hash: None,        |               |
   |     geometry_hash: H(Φ),  |              |               |
   |     geometry_drift: None } |              |               |
   |               |             |              |               |
   |-- assemble_and_sign(attest, sk) --------->|               |
   |               |             |   S-7: check timestamp      |
   |               |             |   S-13: check string lens   |
   |               |             |   S-20: check array sizes   |
   |               |             |   serialise_for_signing()   |
   |               |             |   canonical LE bytes        |
   |               |             |              |-- sign ------>|
   |               |             |              |<- [u8;64] ---|
   |<- Signed GeometricAttestation ------------|               |
```

## 5. Tier-2 Chained Self-Attestation (After Model Update)

When an agent's model self-learns, it produces a chained attestation
linked to the previous one, with drift measured against the reference
geometry.  "Tier 2" is derived from the content — the attestation
populates `parent_attestation_hash`, `geometry_hash`, and
`geometry_drift`.

```
  Agent          Geometry_new   Geometry_ref   got-probe       got-attest
   |                 |              |              |               |
   |-- model has     |              |              |               |
   |   updated U     |              |              |               |
   |                 |              |              |               |
   |-- from_unembedding(U_new) --->|              |               |
   |<- Geometry_new -|              |              |               |
   |                 |              |              |               |
   |-- load reference .gotgeo ---->|              |               |
   |<- Geometry_ref --------------- |              |               |
   |                 |              |              |               |
   |-- drift_from(new, ref) ------>|              |               |
   |<- drift (f32) --|              |              |               |
   |                 |              |              |               |
   |   if drift > max_drift:       |              |               |
   |     STOP — must retrain probes              |               |
   |                 |              |              |               |
   |-- read_probe_checked(probe, probe_set,      |               |
   |     h, geometry_new, geometry_ref) --------->|               |
   |   (validates geometry_hash + drift bound)   |               |
   |<- (raw, conf, flag) or ProbeStale -----------|               |
   |                 |              |              |               |
   |-- fill GeometricAttestation   |              |               |
   |   { schema_version: SCHEMA_VERSION,         |               |
   |     parent_attestation_hash:  |              |               |
   |       attestation_hash(prev), |              |               |
   |     geometry_hash: H(Φ_new), |              |               |
   |     geometry_drift: Some(drift) }           |               |
   |                 |              |              |               |
   |-- assemble_and_sign(attest, sk) ------------|--------------->
   |   (S-7/S-13/S-20 gates)     |              |               |
   |<- Signed Tier-2 chained att --|              |               |
```

## 6. Tier-3 Causal Self-Attestation (With Intervention)

Extends the Tier-2 chained shape with causal intervention checks that
prove the probed directions are causally linked to model output. The
attestation is "Tier 3" when `causal_scores` is non-empty and every
record has `is_causal == true`.

```
  Agent        Geometry   got-probe/intervention    got-attest    Ed25519
   |              |              |                      |            |
   |  (Tier-2 steps above)      |                      |            |
   |              |              |                      |            |
   |-- causal_check(probe, h,   |                      |            |
   |     geometry, δ, model_fn, |                      |            |
   |     threshold) ----------->|                      |            |
   |              |              |                      |            |
   |              |  perturb h:  |                      |            |
   |              |    ŵ_c = Φw/‖Φw‖                  |            |
   |              |    h⁺ = h + δ·ŵ_c                 |            |
   |              |    h⁻ = h - δ·ŵ_c                 |            |
   |              |  model outputs:                     |            |
   |              |    y_base = model(h)                |            |
   |              |    y⁺ = model(h⁺)                  |            |
   |              |    y⁻ = model(h⁻)                  |            |
   |              |  score:                             |            |
   |              |    delta_plus  = ‖y⁺ - y_base‖    |            |
   |              |    delta_minus = ‖y⁻ - y_base‖    |            |
   |              |    consistency = compute_consistency |            |
   |              |    is_causal = consistency ≥ threshold           |
   |<- CausalScore { delta_plus, delta_minus,          |            |
   |     consistency, is_causal, perturbation_delta } --|            |
   |              |              |                      |            |
   |-- fill GeometricAttestation |                      |            |
   |   { schema_version: SCHEMA_VERSION,               |            |
   |     (all Tier-2 fields) +  |                      |            |
   |     causal_scores: [...],  |                      |            |
   |     intervention_delta: δ, |                      |            |
   |     causal_flag: all_pass }|                      |            |
   |              |              |                      |            |
   |-- assemble_and_sign(attest, sk) ----------------->|            |
   |              |              |   linear canonical   |            |
   |              |              |   LE bytes           |-- sign -->|
   |              |              |                      |<- sig ----|
   |<- Signed Tier-3 attestation |                      |            |
```

## 7. Hardware Enclave Attestation Pipeline

The `enclave_pipeline()` function orchestrates hardware-isolated
attestation. The signing key never leaves the enclave boundary.

```
  Agent         MockDmaTap      MockEnclave          got-attest
   |                |                |                    |
   |-- for each (layer, pos, values):                    |
   |                |                |                    |
   |-- capture(layer, pos, vals) -->|                    |
   |                |                |                    |
   |   compute_hash(layer, pos, vals)                    |
   |   = SHA-256(layer ‖ pos ‖ vals)                    |
   |                |                |                    |
   |<- ActivationFrame ------------|                    |
   |                |                |                    |
   |-- receive_activations(frame) ->|                    |
   |                |   recompute   |                    |
   |                |   integrity   |                    |
   |                |   hash, verify|                    |
   |                |   → Ok or     |                    |
   |                |   IntegrityViolation               |
   |                |                |                    |
   |-- (after all frames ingested)  |                    |
   |                |                |                    |
   |-- run_causal_check(model_fn, δ) -->|               |
   |                |   for each probe: |               |
   |                |     causal_check()|               |
   |<- Vec<CausalScore> -----------|                    |
   |                |                |                    |
   |-- attest_with_causal(          |                    |
   |     model_id, model_hash,      |                    |
   |     parent_hash, geo_hash,     |                    |
   |     drift, causal_scores, δ) ->|                    |
   |                |                |                    |
   |                |   embed causal |                    |
   |                |   scores into  |--- assemble_ --->|
   |                |   attestation  |    and_sign()     |
   |                |   sign with    |    S-7/S-13/S-20  |
   |                |   enclave key  |    gates pass      |
   |                |   (key stays   |                    |
   |                |    inside)     |                    |
   |                |                |                    |
   |<- (GeometricAttestation, Vec<CausalScore>) --------|
```

## 8. Mutual Attestation Exchange

Two agents exchange and verify each other's attestations before
cooperating. This is the fundamental agent-to-agent protocol.

```
  Agent Alice                   Channel                Agent Bob
  (Model A, sk_A)                                      (Model B, sk_B)
       |                           |                        |
       |=== PHASE 2: Self-Attest (parallel) ===            |
       |                           |                        |
       |-- build Φ_A, run probes  |   build Φ_B, run probes --|
       |-- sign(attest_A, sk_A)   |     sign(attest_B, sk_B) ---|
       |                           |                        |
       |=== PHASE 3: Exchange ===  |                        |
       |                           |                        |
       |-- build_request(          |                        |
       |     nonce, id_B,          |                        |
       |     sk_A, chain, attest) -|                        |
       |   ExchangeRequest {       |                        |
       |     agent_id: id_A,       |                        |
       |     envelope: signed,     |                        |
       |     chain + current }     |                        |
       |-------------------------->|------> receive req     |
       |                           |                        |
       |                           |   validate_request(    |
       |                           |     req, id_B,         |
       |                           |     None, registry) ---|
       |                           |   (S-2: registry       |
       |                           |    integrity verified)  |
       |                           |   §4 Phase 4 (defence   |
       |                           |   in depth re-verify): |
       |                           |     check_domain_      |
       |                           |     compatibility(     |
       |                           |       peer, self)      |
       |                           |     Supervised (§5.5) OK|
       |                           |   §7.3/§8.2 governance:|
       |                           |     effective_         |
       |                           |       thresholds +     |
       |                           |     enforce_governance |
       |                           |     (require_chain,    |
       |                           |      require_causal_   |
       |                           |        validation)     |
       |                           |   §2.1 scope binding:  |
       |                           |     check_attestation_ |
       |                           |       scope_binding    |
       |                           |                        |
       |                           |   build_response(      |
       |                           |     nonce, id_A,       |
       |                           |     sk_B, verdict,     |
       |                           |     chain, attest) ----|
       |                           |                        |
       |   receive rsp <-----------|<------ ExchangeResponse|
       |                           |                        |
       |=== PHASE 4: Verify ===   |                        |
       |                           |                        |
       |-- validate_response(      |                        |
       |     rsp, id_A,            |                        |
       |     nonce, registry)      |                        |
       |                           |                        |
       |   Envelope checks:        |   Envelope checks:     |
       |     Ed25519 sig verify    |     Ed25519 sig verify  |
       |     peer_agent_id match   |     peer_agent_id match |
       |     attestation_hash ok   |     attestation_hash ok |
       |     chain_root_hash ok    |     chain_root_hash ok  |
       |     timestamp fresh       |     timestamp fresh     |
       |     S-9: verified=true    |     S-9: verified=true  |
       |                           |                        |
       |   Chain verify (if chained):  Chain verify:         |
       |     verify_chain(         |     verify_chain(       |
       |       chain, current,     |       chain, current,   |
       |       &[pk_B], max_drift) |       &[pk_A], max_drift)|
       |       ↑ S-8: key rotation |       ↑ S-8: key rotation|
       |                           |                        |
       |=== PHASE 5: Decide ===   |                        |
       |                           |                        |
       |   Both Accepted?          |          Both Accepted? |
       |     yes → cooperate       |   cooperate ← yes      |
       |     no  → refuse          |      refuse ← no       |
```

## 9. Chain Verification (Receiving Agent)

When Agent Bob receives a chain of attestations from Agent Alice,
this is the verification walk. S-8: each attestation is checked
against **all** keys in `signer_pks` (supports key rotation).

```
  Agent Bob                 got-wire/chain           got-attest
     |                          |                        |
     |-- receive chain:         |                        |
     |   [attest_0, ...,        |                        |
     |    attest_n]             |                        |
     |   + current attestation  |                        |
     |                          |                        |
     |-- verify_chain(chain,    |                        |
     |     current,             |                        |
     |     &[pk_A, pk_A_old],   |  ← S-8: multiple keys |
     |     max_drift) --------->|                        |
     |                          |                        |
     |   For i = 0..n:          |                        |
     |                          |                        |
     |   verify(attest_i,       |                        |
     |     any key in pks) -----|----------------------->|
     |                          |   serialise_for_signing|
     |                          |   (linear canonical    |
     |                          |    bytes)              |
     |                          |   Ed25519 verify       |
     |<-- sig_valid ------------|                        |
     |                          |                        |
     |   if i > 0:              |                        |
     |     expected_parent =    |                        |
     |       attestation_hash(attest_{i-1}) ----------->|
     |                          |                        |
     |     assert attest_i.parent_attestation_hash       |
     |       == Some(expected_parent)                    |
     |                          |                        |
     |   check geometry_drift   |                        |
     |     ≤ max_drift          |                        |
     |                          |                        |
     |   check model_id         |                        |
     |     consistency          |                        |
     |                          |                        |
     |   check chain[0].parent_attestation_hash == None  |
     |     (anchor has no parent)                        |
     |                          |                        |
     |-- all checks pass?       |                        |
     |   yes → ChainVerdict {   |                        |
     |     length, max_drift }  |                        |
     |   no  → WireError::Chain |                        |
```

## 10. Attestation Storage & Audit

```
  Agent              AttestationStore           AuditReport
   |                       |                        |
   |-- append(attest,      |                        |
   |     verifying_key) -->|                        |
   |                       |                        |
   |   verify signature    |                        |
   |   compute StoreId     |                        |
   |   = attestation_hash()|                        |
   |   persist to store    |                        |
   |   (FileStore: atomic  |                        |
   |    write + hash check)|                        |
   |                       |                        |
   |<-- StoreId [u8;32] --|                        |
   |                       |                        |
   |-- chain(model_id) --->|                        |
   |<-- [attest_0..n] ----|                        |
   |                       |                        |
   |-- query(StoreFilter { |                        |
   |     model_id,         |                        |
   |     signer,           |                        |
   |     after/before,     |                        |
   |     schema_version,   |                        |
   |     causal_flag }) -->|                        |
   |<-- filtered results --|                        |
   |                       |                        |
   |-- audit(model_id) --->|                        |
   |                       |-- build_audit_report ->|
   |<-- AuditReport -------|                        |
   |   { total_attestations,                        |
   |     chain_length,                              |
   |     chain_valid,                               |
   |     drift_summary,                             |
   |     causal_summary,                            |
   |     signers }                                  |
```

## 11. Multi-Agent Group Formation

When three or more agents form a cooperation group, an aggregator
can reduce O(n²) pairwise verification to O(n).

```
  Agent Alice          Agent Carol (aggregator)          Agent Bob
       |                        |                           |
       |-- attest_A ----------->|                           |
       |                        |<----------- attest_B -----|
       |                        |                           |
       |                  verify(attest_A, pk_A)            |
       |                  verify(attest_B, pk_B)            |
       |                  check drift bounds                |
       |                        |                           |
       |                  all pass?                         |
       |                    build group_summary {           |
       |                      members: [A, B],             |
       |                      attestation_hashes: [...],   |
       |                      max_observed_drift: ...,     |
       |                      signed by sk_C               |
       |                    }                              |
       |                        |                           |
       |<-- group_summary ------|------- group_summary ---->|
       |                        |                           |
  verify(group, pk_C)           |         verify(group, pk_C)
  check own hash in members     |   check own hash in members
```

## 12. Extraction (Python Bridge)

```
  Agent Runtime        extract_activations.py     HuggingFace Model
       |                      |                      |
       |-- invoke extraction  |                      |
       |   (model, input,     |                      |
       |    layers) --------->|                      |
       |                      |                      |
       |                      |-- AutoTokenizer ---->|
       |                      |<-- tokenizer --------|
       |                      |                      |
       |                      |-- AutoModelForCausalLM -->|
       |                      |<-- model ref -------------|
       |                      |                      |
       |                      |-- auto-detect arch   |
       |                      |   (GPT2/LLaMA/OPT/  |
       |                      |    GPTNeoX)          |
       |                      |                      |
       |                      |-- tokenize + forward |
       |                      |   pass, hooks capture|
       |                      |   hidden_states      |
       |                      |<-- activations ------|
       |                      |                      |
       |                      |-- lm_head.weight --->|
       |                      |<-- unembedding (V*d) |
       |                      |                      |
       |<-- .gotact + .gotue  |                      |
       |    (binary files     |                      |
       |     for Layer 0)     |                      |
```
