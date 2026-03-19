#!/usr/bin/env python3
import json, urllib.request

resp = urllib.request.urlopen("http://127.0.0.1:3000/api/demo-conversation")
conv = json.loads(resp.read())
payload = json.dumps({"messages": conv["messages"]}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:3000/api/conversation/analyse",
    data=payload,
    headers={"Content-Type": "application/json"},
)
resp2 = urllib.request.urlopen(req)
d = json.loads(resp2.read())

last = d["turns"][-1]
print(f"Cumulative terms ({len(last['cumulative_values'])}): {last['cumulative_values']}")
print(f"Total pairwise: {len(last['pairwise'])}")
print(f"Contradictions: {len(last['all_contradictions'])}")
print(f"Redundancies: {len(last['all_redundancies'])}")
print(f"num_terms: {last['num_terms']}, num_unresolved: {last['num_unresolved']}")
print()

# Show all contradictions with severity
contras = sorted(last["all_contradictions"], key=lambda c: -c["severity"])
for c in contras:
    print(f"  {c['term_a']:20s} <-> {c['term_b']:20s}  cos={c['causal_cosine']:.4f}  sev={c['severity']:.4f}")

max_sev = max(c["severity"] for c in contras)
ratio = len(contras) / len(last["pairwise"])
score = (1 - max_sev) * (1 - ratio)
print()
print(f"max_severity = {max_sev:.4f}")
print(f"contradiction_ratio = {len(contras)}/{len(last['pairwise'])} = {ratio:.4f}")
print(f"coherence = (1 - {max_sev:.4f}) * (1 - {ratio:.4f}) = {score:.4f}")
print(f"API coherence_score = {last['coherence_score']:.4f}")
