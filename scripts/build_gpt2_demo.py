#!/usr/bin/env python3
"""Produce a demo conversation JSON with real GPT-2 768-d embeddings."""
import json
from pathlib import Path

conv = json.load(open("data/demo/demo_conversation.json"))
msg_embs = json.load(open("data/models/gpt2-message-embeddings.json"))

# Replace 32-d synthetic embeddings with 768-d GPT-2 embeddings
for i, msg in enumerate(conv["messages"]):
    assert msg_embs[i]["turn"] == i
    msg["embedding"] = msg_embs[i]["embedding"]

out = Path("data/models/gpt2-demo-conversation.json")
with open(out, "w") as f:
    json.dump(conv, f)
print(f"Wrote {out} with {len(conv['messages'])} messages, embedding dim={len(conv['messages'][0]['embedding'])}")
