#!/usr/bin/env python3
"""Test the got-web API in real model mode."""
import json
import urllib.request

# Test 1: Demo conversation endpoint
print("=== Test 1: Demo Conversation Endpoint ===")
resp = urllib.request.urlopen("http://127.0.0.1:3000/api/demo-conversation")
data = json.loads(resp.read())
msgs = data["messages"]
print(f"Messages: {len(msgs)}")
print(f"Embedding dim: {len(msgs[0]['embedding'])}")

# Test 2: Analysis endpoint
print("\n=== Test 2: Analysis Endpoint ===")
req_data = json.dumps(data).encode()
req = urllib.request.Request(
    "http://127.0.0.1:3000/api/conversation/analyse",
    data=req_data,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(f"Turns: {len(result['turns'])}")
print(f"Available terms: {len(result['available_terms'])}")

for t in result["turns"]:
    detected = [f"{d['term']}({d['cos_phi']:.3f})" for d in t["detected_values"]]
    print(f"  Turn {t['turn']:2d} [{t['speaker']:8s}] coherence={t['coherence_score']:.4f} trust={t['trust_score']:.4f} detected=[{', '.join(detected)}]")
    if t["new_contradictions"]:
        for c in t["new_contradictions"]:
            print(f"    NEW CONTRADICTION: {c['term_a']} <-> {c['term_b']} (severity={c['severity']:.3f}, cos={c['causal_cosine']:.3f})")

# ---------------------------------------------------------------------------
# Embed API test
# ---------------------------------------------------------------------------

BASE = "http://127.0.0.1:3000"
dim = len(msgs[0]["embedding"])

print("\n=== Test 2b: Embed Endpoint ===")
req_data = json.dumps({"text": "I believe in honesty and fairness"}).encode()
req = urllib.request.Request(
    f"{BASE}/api/embed",
    data=req_data,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
embed_result = json.loads(resp.read())
print(f"Dim: {embed_result['dim']}")
print(f"Matched tokens: {embed_result['matched_tokens']}/{embed_result['total_tokens']}")
assert embed_result['dim'] == dim, f"Expected dim {dim}, got {embed_result['dim']}"
assert embed_result['matched_tokens'] > 0, "Expected at least one matched token"

# ---------------------------------------------------------------------------
# Proxy API tests
# ---------------------------------------------------------------------------

# Test 3: Create proxy session
print("\n=== Test 3: Create Proxy Session ===")
req_data = json.dumps({"target_model_id": "test-model"}).encode()
req = urllib.request.Request(
    f"{BASE}/api/proxy/session",
    data=req_data,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
session = json.loads(resp.read())
sid = session["session_id"]
print(f"Session ID: {sid}")
print(f"Target model: {session['target_model_id']}")
print(f"Geometry hash: {session['reference_geometry_hash'][:16]}...")

# Test 4: Submit observations (use message embeddings from demo)
print(f"\n=== Test 4: Submit {len(msgs)} Observations ===")
for i, msg in enumerate(msgs):
    req_data = json.dumps({"embedding": msg["embedding"]}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/proxy/session/{sid}/observe",
        data=req_data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    obs = json.loads(resp.read())
    n_vals = len(obs["detected_values"])
    dev_str = ""
    if obs.get("deviation"):
        dev_str = f" deviation={obs['deviation']['combined_score']:.4f} ({obs['deviation']['verdict']})"
    print(f"  Obs {obs['observation_count']:3d}: {n_vals} values detected{dev_str}")

# Test 5: Session status
print("\n=== Test 5: Session Status ===")
resp = urllib.request.urlopen(f"{BASE}/api/proxy/session/{sid}/status")
status = json.loads(resp.read())
print(f"Observations: {status['observation_count']}")
print(f"Top values: {status['top_values'][:5]}")
if status.get("latest_deviation"):
    print(f"Latest deviation: {status['latest_deviation']['verdict']}")

# Test 6: Force snapshot + attestation
print("\n=== Test 6: Snapshot + Attestation ===")
req_data = json.dumps({"attestation_type": "baseline"}).encode()
req = urllib.request.Request(
    f"{BASE}/api/proxy/session/{sid}/snapshot",
    data=req_data,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
snap = json.loads(resp.read())
print(f"Attestation hash: {snap['attestation_hash'][:16]}...")
print(f"Sequence number: {snap['sequence_number']}")
print(f"Type: {snap['attestation_type']}")

# Test 7: Deviation history
print("\n=== Test 7: Deviation History ===")
resp = urllib.request.urlopen(f"{BASE}/api/proxy/session/{sid}/history")
history = json.loads(resp.read())
print(f"History entries: {len(history['deviations'])}")

print("\n=== All proxy tests passed ===")
