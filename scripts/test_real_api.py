#!/usr/bin/env python3
"""Test the got-web API with real GPT-2 geometry."""
import json
import urllib.request
import sys

BASE = "http://127.0.0.1:3000"

def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())

def post(path, data):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())

# 1. Test demo conversation endpoint
print("=== Demo Conversation ===")
demo = get("/api/demo-conversation")
msgs = demo["messages"]
print(f"  Messages: {len(msgs)}")
for i, m in enumerate(msgs):
    emb_dim = len(m["embedding"])
    print(f"  [{i}] {m['speaker']:10s} dim={emb_dim}  {m['text'][:70]}...")

# 2. Submit the demo conversation for analysis
print("\n=== Analyse Conversation ===")
result = post("/api/conversation/analyse", demo)

print(f"  Mode: {result.get('mode', 'unknown')}")

# Response is per-turn — get the last turn for summary
turns = result.get("turns", [])
if turns:
    last = turns[-1]
    print(f"  Final coherence: {last['coherence_score']:.4f}")
    print(f"  Final trust:     {last['trust_score']:.4f}")
    print(f"  Contradictions:  {len(last.get('all_contradictions', []))}")
    print(f"  Redundancies:    {len(last.get('all_redundancies', []))}")
    print(f"  Cumulative vals: {last.get('num_terms', 0)}")

# 3. Show per-turn detections
print("\n=== Per-Turn Detections ===")
for turn in turns:
    idx = turn["turn"]
    vals = turn.get("detected_values", [])
    val_str = ", ".join(f"{v['term']}({v.get('cos_phi', 0):.2f})" for v in vals[:5])
    extra = f" +{len(vals)-5} more" if len(vals) > 5 else ""
    coh = turn["coherence_score"]
    trust = turn["trust_score"]
    print(f"  Turn {idx:2d} coh={coh:.3f} trust={trust:.3f}: [{val_str}{extra}]")

# 4. Show contradictions from last turn
print("\n=== Contradictions (final) ===")
for c in last.get("all_contradictions", []):
    print(f"  {c['term_a']} <-> {c['term_b']}  severity={c['severity']:.3f}  cos={c.get('causal_cosine', 0):.3f}  angle={c.get('angle_degrees', 0):.1f}°")

# 5. Show pairwise matrix summary from last turn
print("\n=== Pairwise Summary (final) ===")
pairs = last.get("pairwise", [])
if pairs:
    cosines = [p["causal_cosine"] for p in pairs]
    print(f"  Pairs: {len(pairs)}  min_cos={min(cosines):.4f}  max_cos={max(cosines):.4f}  mean={sum(cosines)/len(cosines):.4f}")

print("\n=== ALL TESTS PASSED ===")
