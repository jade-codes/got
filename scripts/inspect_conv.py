#!/usr/bin/env python3
"""Quick inspect of demo conversation structure."""
import json
d = json.load(open("data/demo/demo_conversation.json"))
print(f"Type: {type(d).__name__}")
if isinstance(d, dict):
    print(f"Keys: {list(d.keys())}")
    msgs = d.get("messages", [])
else:
    msgs = d
print(f"Messages: {len(msgs)}")
for i, m in enumerate(msgs):
    print(f"  [{i}] keys={list(m.keys())}")
    text = m.get("text", m.get("content", ""))
    speaker = m.get("speaker", m.get("role", ""))
    emb_len = len(m.get("embedding", []))
    print(f"       speaker={speaker}, text={text[:60]}..., emb_dim={emb_len}")
