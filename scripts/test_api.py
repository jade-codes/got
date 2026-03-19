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
