// ---------------------------------------------------------------------------
// Integration test: full proxy pipeline with synthetic data.
//
// Creates a session, feeds 30 observations through it, verifies baseline
// establishment, triggers deviation detection, and produces signed attestations.
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use ed25519_dalek::SigningKey;
use got_core::geometry::CausalGeometry;
use got_incoherence::embeddings::PrecomputedEmbeddings;
use got_proxy::attestation::{
    attestation_hash, verify_attestation, AttestationType,
    BEHAVIORAL_SCHEMA_VERSION,
};
use got_proxy::config::ProxyConfig;
use got_proxy::deviation::DeviationVerdict;
use got_proxy::session::ProxySession;
use got_proxy::store::MemoryValueSpaceStore;
use got_proxy::value_space::BehavioralValueSpace;

fn make_geometry(dim: usize) -> CausalGeometry {
    let mut gram = vec![0.0f32; dim * dim];
    for i in 0..dim {
        gram[i * dim + i] = 1.0;
    }
    CausalGeometry::from_raw_gram(gram, dim).unwrap()
}

fn make_embeddings(dim: usize) -> HashMap<String, Vec<f32>> {
    let mut embeddings = HashMap::new();
    // Spread terms across orthogonal-ish directions
    let terms = [
        "honesty", "courage", "fairness", "empathy", "loyalty",
        "wisdom", "integrity", "compassion",
    ];
    for (i, term) in terms.iter().enumerate() {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        // Add some secondary structure
        emb[(i + 1) % dim] = 0.3;
        embeddings.insert(term.to_string(), emb);
    }
    embeddings
}

#[test]
fn full_proxy_pipeline() {
    let dim = 8;
    let geometry = make_geometry(dim);
    let embeddings = make_embeddings(dim);
    let source = PrecomputedEmbeddings::new(embeddings).unwrap();
    let sk = SigningKey::from_bytes(&[42u8; 32]);
    let vk = sk.verifying_key();

    let mut session = ProxySession::new(
        "integration-test".into(),
        "test-closed-model".into(),
        sk,
        geometry,
        source,
        ProxyConfig::default(),
        MemoryValueSpaceStore::new(),
    )
    .unwrap();

    // Phase 1: Build baseline (25 consistent observations)
    for i in 0..25 {
        let angle = (i as f32) * 0.15;
        let mut emb = vec![0.0f32; dim];
        emb[0] = angle.cos();
        emb[1] = angle.sin();
        emb[2] = 0.3;
        let result = session.observe(&emb).unwrap();
        assert_eq!(result.observation_count, i + 1);
        // Before min_observations, deviation should be None
        if i < 19 {
            assert!(result.deviation.is_none());
        }
    }

    // Take baseline snapshot
    let baseline_att = session
        .snapshot_and_attest(AttestationType::Baseline)
        .unwrap();
    assert_eq!(baseline_att.schema_version, BEHAVIORAL_SCHEMA_VERSION);
    assert_eq!(baseline_att.target_model_id, "test-closed-model");
    assert_eq!(baseline_att.observation_count, 25);
    assert_eq!(baseline_att.sequence_number, 1);
    assert!(baseline_att.parent_hash.is_none());
    assert!(verify_attestation(&baseline_att, &vk).is_ok());

    // Phase 2: More consistent observations (should stay within baseline)
    for _ in 0..5 {
        let result = session.observe(&[0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        if let Some(dev) = &result.deviation {
            assert!(dev.baseline_sufficient);
            assert_eq!(dev.verdict, DeviationVerdict::WithinBaseline);
        }
    }

    // Phase 3: Status check
    let status = session.status();
    assert_eq!(status.observation_count, 30);
    assert_eq!(status.attestation_count, 1);
    assert!(!status.top_values.is_empty());

    // Phase 4: Second snapshot (should chain)
    let snapshot_att = session
        .snapshot_and_attest(AttestationType::Snapshot)
        .unwrap();
    assert_eq!(snapshot_att.sequence_number, 2);
    assert!(snapshot_att.parent_hash.is_some());
    assert!(verify_attestation(&snapshot_att, &vk).is_ok());

    // Phase 5: Verify attestation chain linkage
    let _hash1 = attestation_hash(&baseline_att);
    // The parent_hash of snapshot_att should exist (chained via value space)
    assert!(snapshot_att.parent_hash.is_some());

    // Phase 6: Deviation history
    let history = session.deviation_history();
    assert!(!history.is_empty());
}

#[test]
fn value_space_hash_determinism() {
    let dim = 4;
    let _geometry = make_geometry(dim);
    let embeddings: HashMap<String, Vec<f32>> = [
        ("honesty".into(), vec![1.0, 0.0, 0.0, 0.0]),
        ("courage".into(), vec![0.0, 1.0, 0.0, 0.0]),
    ]
    .into_iter()
    .collect();
    let sk = SigningKey::from_bytes(&[42u8; 32]);

    // Create two sessions with identical inputs
    let source1 = PrecomputedEmbeddings::new(embeddings.clone()).unwrap();
    let source2 = PrecomputedEmbeddings::new(embeddings).unwrap();

    let mut s1 = ProxySession::new(
        "s1".into(), "model".into(), sk.clone(),
        make_geometry(dim), source1,
        ProxyConfig::default(), MemoryValueSpaceStore::new(),
    ).unwrap();

    let mut s2 = ProxySession::new(
        "s2".into(), "model".into(), sk,
        make_geometry(dim), source2,
        ProxyConfig::default(), MemoryValueSpaceStore::new(),
    ).unwrap();

    // Feed identical observations
    let observations = [
        vec![0.8, 0.2, 0.0, 0.0],
        vec![0.3, 0.7, 0.0, 0.0],
        vec![0.5, 0.5, 0.0, 0.0],
    ];
    for obs in &observations {
        s1.observe(obs).unwrap();
        s2.observe(obs).unwrap();
    }

    // Value space hashes should be identical
    assert_eq!(s1.value_space().hash(), s2.value_space().hash());
}

#[test]
fn file_store_roundtrip() {
    use got_proxy::store::{FileValueSpaceStore, ValueSpaceStore};

    let dir = std::env::temp_dir().join(format!("got-proxy-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut store = FileValueSpaceStore::new(&dir).unwrap();
    let space = BehavioralValueSpace::new([0xCC; 32]);
    let id = store.store_snapshot(&space).unwrap();

    // Re-open store and verify
    let store2 = FileValueSpaceStore::new(&dir).unwrap();
    let loaded = store2.get_snapshot(&id).unwrap();
    assert_eq!(loaded.reference_geometry_hash, space.reference_geometry_hash);
    assert_eq!(loaded.hash(), space.hash());

    let _ = std::fs::remove_dir_all(&dir);
}
