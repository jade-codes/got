// ---------------------------------------------------------------------------
// got-net integration tests — real TCP socket on 127.0.0.1.
//
// These exercise the full stack: tokio listener accepts a connection,
// dispatches it to a blocking thread, runs the Noise NK handshake,
// decodes a real ExchangeRequest, runs the trust-registry validation,
// signs and sends back an ExchangeResponse, and the client validates
// it.  No mocks — every byte travels through the kernel TCP stack.
//
// Each test binds 127.0.0.1:0 to get a free random port so they can
// run in parallel without colliding.
// ---------------------------------------------------------------------------

use std::sync::Arc;

use ed25519_dalek::SigningKey;

use got_attest::assemble_and_sign;
use got_core::{GeometricAttestation, InnerProduct, Precision, SCHEMA_VERSION};
use got_net::client::{request, RequestParams};
use got_net::server::{accept_loop, AttestationProvider, ServerConfig, StaticAttestationProvider};
use got_wire::domain::{Domain, DomainPattern, DomainScope, InteractionMode, PermittedDomain};
use got_wire::exchange::Verdict;
use got_wire::governance::GovernanceTable;
use got_wire::registry::{compute_agent_id, AgentEntry, TrustRegistry};

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn make_attest(k: &SigningKey, model_id: &str) -> GeometricAttestation {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let a = GeometricAttestation {
        schema_version: SCHEMA_VERSION,
        model_id: model_id.to_string(),
        model_hash: Some([0x11; 32]),
        precision: Precision::Fp32,
        inner_product: InnerProduct::Causal,
        input_hash: [0x22; 32],
        timestamp: now,
        corpus_version: "c1".into(),
        probe_version: "p1".into(),
        layer_readings: vec![vec![1.0, 2.0]],
        confidence: vec![0.9],
        coverage_flags: vec![false],
        divergence_flag: false,
        parent_attestation_hash: None,
        geometry_hash: None,
        geometry_drift: None,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: 0,
        directional_drifts: vec![],
        probe_commitment: None,
        density_reading: None,
        curvature_reading: None,
        domain_scope_declaration: None,
        signature: [0u8; 64],
    };
    assemble_and_sign(a, k).unwrap()
}

fn registry_with(alice: &SigningKey, bob: &SigningKey) -> TrustRegistry {
    let mut r = TrustRegistry::empty();
    let alice_pk = alice.verifying_key();
    let bob_pk = bob.verifying_key();
    r.add_agent(AgentEntry {
        name: "alice".into(),
        public_key: alice_pk,
        agent_id: compute_agent_id(&alice_pk),
        max_drift_accepted: 0.05,
        roles: vec!["producer".into()],
        expected_model_hash: None,
        certificate: None,
        domain_scope: None,
        governance_table: GovernanceTable::default(),
    });
    r.add_agent(AgentEntry {
        name: "bob".into(),
        public_key: bob_pk,
        agent_id: compute_agent_id(&bob_pk),
        max_drift_accepted: 0.05,
        roles: vec!["verifier".into()],
        expected_model_hash: None,
        certificate: None,
        domain_scope: None,
        governance_table: GovernanceTable::default(),
    });
    r
}

// ---------------------------------------------------------------------------
// Test 1 — happy path: real socket, valid exchange, both sides accept.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_exchange_over_tcp_accepted() {
    let alice = key(0xAA); // initiator
    let bob = key(0xBB); // responder

    let registry = Arc::new(registry_with(&alice, &bob));
    let bob_attest = make_attest(&bob, "bob-model");
    let server_config = ServerConfig {
        signing_key: Arc::new(bob.clone()),
        registry: registry.clone(),
        attestation: Arc::new(StaticAttestationProvider {
            current: bob_attest.clone(),
            chain: vec![],
        }) as Arc<dyn AttestationProvider>,
    };

    // Bind to a random port and capture the address before spawning.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(accept_loop(listener, server_config));

    // Run the client.
    let alice_attest = make_attest(&alice, "alice-model");
    let outcome = request(
        addr,
        RequestParams {
            signing_key: alice.clone(),
            responder_vk: bob.verifying_key(),
            chain: vec![],
            current: alice_attest.clone(),
        },
        (*registry).clone(),
    )
    .await
    .unwrap();

    assert_eq!(outcome.verdict, Verdict::Accepted, "{}", outcome.reason);
    assert_eq!(outcome.response.verdict, Verdict::Accepted);
    assert_eq!(outcome.response.current.model_id, "bob-model");

    server_task.abort();
}

// ---------------------------------------------------------------------------
// Test 2 — unknown agent: client signs as Charlie, but only Alice and
// Bob are in the registry.  The server must reject at the lookup step.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_exchange_over_tcp_unknown_agent_rejected() {
    let alice = key(0xAA);
    let bob = key(0xBB);
    let charlie = key(0xCC); // not in registry

    let registry = Arc::new(registry_with(&alice, &bob));
    let bob_attest = make_attest(&bob, "bob-model");
    let server_config = ServerConfig {
        signing_key: Arc::new(bob.clone()),
        registry: registry.clone(),
        attestation: Arc::new(StaticAttestationProvider {
            current: bob_attest,
            chain: vec![],
        }) as Arc<dyn AttestationProvider>,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(accept_loop(listener, server_config));

    // Charlie tries to exchange with Bob.
    let charlie_attest = make_attest(&charlie, "charlie-model");
    let result = request(
        addr,
        RequestParams {
            signing_key: charlie.clone(),
            responder_vk: bob.verifying_key(),
            chain: vec![],
            current: charlie_attest,
        },
        (*registry).clone(),
    )
    .await;

    // The server raises UnknownAgent during validate_request — the
    // connection is closed mid-exchange so the client sees an io error
    // on its read of the response.  Either form is acceptable; what
    // matters is that the exchange did NOT succeed.
    match result {
        Err(_) => { /* connection closed: server rejected before responding */ }
        Ok(outcome) => assert_ne!(
            outcome.verdict,
            Verdict::Accepted,
            "unknown agent must not be accepted"
        ),
    }

    server_task.abort();
}

// ---------------------------------------------------------------------------
// Test 3 — multiple sequential exchanges over distinct connections.
// Confirms the listener is genuinely accepting > 1 connection and that
// state from one exchange does not contaminate the next.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multiple_sequential_exchanges() {
    let alice = key(0xAA);
    let bob = key(0xBB);

    let registry = Arc::new(registry_with(&alice, &bob));
    let bob_attest = make_attest(&bob, "bob-model");
    let server_config = ServerConfig {
        signing_key: Arc::new(bob.clone()),
        registry: registry.clone(),
        attestation: Arc::new(StaticAttestationProvider {
            current: bob_attest,
            chain: vec![],
        }) as Arc<dyn AttestationProvider>,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(accept_loop(listener, server_config));

    for i in 0..5 {
        let alice_attest = make_attest(&alice, &format!("alice-model-{i}"));
        let outcome = request(
            addr,
            RequestParams {
                signing_key: alice.clone(),
                responder_vk: bob.verifying_key(),
                chain: vec![],
                current: alice_attest,
            },
            (*registry).clone(),
        )
        .await
        .unwrap();
        assert_eq!(outcome.verdict, Verdict::Accepted, "iteration {i}: {}", outcome.reason);
    }

    server_task.abort();
}

// ---------------------------------------------------------------------------
// Test 4 — Phase 1 domain pre-flight: the client aborts BEFORE
// computing attestations when domain scopes are incompatible.
// ---------------------------------------------------------------------------

fn scoped_registry(
    alice: &SigningKey,
    bob: &SigningKey,
    alice_domain: &str,
    alice_permitted: &str,
    bob_domain: &str,
    bob_permitted: &str,
) -> TrustRegistry {
    let mut r = TrustRegistry::empty();
    let alice_pk = alice.verifying_key();
    let bob_pk = bob.verifying_key();

    let scope = |primary: &str, permitted: &str| DomainScope {
        primary: Domain::parse(primary).unwrap(),
        permitted: vec![PermittedDomain {
            pattern: DomainPattern::parse(permitted).unwrap(),
            mode: InteractionMode::Cooperative,
        }],
        exclusions: vec![],
    };

    r.add_agent(AgentEntry {
        name: "alice".into(),
        public_key: alice_pk,
        agent_id: compute_agent_id(&alice_pk),
        max_drift_accepted: 0.05,
        roles: vec![],
        expected_model_hash: None,
        certificate: None,
        domain_scope: Some(scope(alice_domain, alice_permitted)),
        governance_table: GovernanceTable::default(),
    });
    r.add_agent(AgentEntry {
        name: "bob".into(),
        public_key: bob_pk,
        agent_id: compute_agent_id(&bob_pk),
        max_drift_accepted: 0.05,
        roles: vec![],
        expected_model_hash: None,
        certificate: None,
        domain_scope: Some(scope(bob_domain, bob_permitted)),
        governance_table: GovernanceTable::default(),
    });
    r
}

#[tokio::test]
async fn phase_1_domain_preflight_rejects_incompatible() {
    let alice = key(0xAA); // agriculture
    let bob = key(0xBB);   // transport

    let registry = Arc::new(scoped_registry(
        &alice,
        &bob,
        "agriculture.crop-management",
        "agriculture.*",
        "transport.autonomous-vehicle",
        "transport.*",
    ));
    let bob_attest = make_attest(&bob, "bob-model");
    let server_config = ServerConfig {
        signing_key: Arc::new(bob.clone()),
        registry: registry.clone(),
        attestation: Arc::new(StaticAttestationProvider {
            current: bob_attest,
            chain: vec![],
        }) as Arc<dyn AttestationProvider>,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(accept_loop(listener, server_config));

    let alice_attest = make_attest(&alice, "alice-model");
    let result = request(
        addr,
        RequestParams {
            signing_key: alice.clone(),
            responder_vk: bob.verifying_key(),
            chain: vec![],
            current: alice_attest,
        },
        (*registry).clone(),
    )
    .await;

    // The client's Phase 1 pre-flight rejects before sending.
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("domain") || msg.contains("Domain"),
                "expected domain rejection, got: {msg}"
            );
        }
        Ok(outcome) => {
            assert_ne!(outcome.verdict, Verdict::Accepted);
            assert!(
                outcome.reason.contains("domain") || outcome.reason.contains("Domain"),
                "expected domain rejection, got: {}",
                outcome.reason
            );
        }
    }

    server_task.abort();
}

#[tokio::test]
async fn phase_1_compatible_domains_succeed() {
    let alice = key(0xAA);
    let bob = key(0xBB);

    let registry = Arc::new(scoped_registry(
        &alice,
        &bob,
        "agriculture.crop-management",
        "agriculture.*",
        "agriculture.supply-chain",
        "agriculture.*",
    ));
    let bob_attest = make_attest(&bob, "bob-model");
    let server_config = ServerConfig {
        signing_key: Arc::new(bob.clone()),
        registry: registry.clone(),
        attestation: Arc::new(StaticAttestationProvider {
            current: bob_attest,
            chain: vec![],
        }) as Arc<dyn AttestationProvider>,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(accept_loop(listener, server_config));

    let alice_attest = make_attest(&alice, "alice-model");
    let outcome = request(
        addr,
        RequestParams {
            signing_key: alice.clone(),
            responder_vk: bob.verifying_key(),
            chain: vec![],
            current: alice_attest,
        },
        (*registry).clone(),
    )
    .await
    .unwrap();

    assert_eq!(outcome.verdict, Verdict::Accepted, "{}", outcome.reason);
    server_task.abort();
}
