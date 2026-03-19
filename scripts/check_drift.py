#!/usr/bin/env python3
import json, sys, urllib.request

# First get the demo conversation
resp = urllib.request.urlopen("http://127.0.0.1:3000/api/demo-conversation")
conv = json.loads(resp.read())

# Build analysis request
payload = json.dumps({"messages": conv["messages"]}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:3000/api/conversation/analyse",
    data=payload,
    headers={"Content-Type": "application/json"},
)
resp2 = urllib.request.urlopen(req)
d = json.loads(resp2.read())

for t in d["turns"]:
    nc = len(t["new_contradictions"])
    ac = len(t["all_contradictions"])
    tc = len(t.get("turn_contradictions", []))
    mc = len(t.get("message_contradictions", []))
    mcoh = t.get("message_coherence", 1.0)
    conv = t.get("convergence", 0.0)
    vals = ", ".join(f'{v["term"]}({v["cos_phi"]:.2f})' for v in t["detected_values"][:3])
    print(f'Turn {t["turn"]:2d} {t["speaker"]:20s} drift={t["speaker_drift"]:.3f} '
          f'coh={t["coherence_score"]:.3f} msg_coh={mcoh:.3f} trust={t["trust_score"]:.3f} '
          f'conv={conv:.3f} msg_c={mc} active_c={tc} all_c={ac}')

print()
print("Assessment:", d["assessment"]["verdict"], "-", d["assessment"]["summary"])
print(f"Influence: {d['assessment']['influence_score']:.4f}")
for ss in d["speaker_summary"]:
    print(f"  {ss['speaker']}: drift={ss['semantic_drift']:.4f}, msgs={ss['message_count']}")
