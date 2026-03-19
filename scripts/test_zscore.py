#!/usr/bin/env python3
"""
Test z-scored logit detection and standard cosine pairwise analysis.
"""
import json
import numpy as np

# Load data
with open("data/models/gpt2-term-analysis.json") as f:
    term_info = json.load(f)
with open("data/models/gpt2-message-embeddings.json") as f:
    msg_data = json.load(f)
with open("data/demo/demo_conversation.json") as f:
    conv = json.load(f)

import struct
with open("data/models/gpt2.gotue", "rb") as f:
    f.read(4); f.read(2)
    vocab_size = struct.unpack("<I", f.read(4))[0]
    hidden_dim = struct.unpack("<I", f.read(4))[0]
    U = np.frombuffer(f.read(), dtype=np.float32).reshape(vocab_size, hidden_dim)

# Term vectors
terms = []
term_vecs = []
for name, info in sorted(term_info.items()):
    if info["found"]:
        terms.append(name)
        term_vecs.append(U[info["token_index"]])
term_vecs = np.array(term_vecs)  # (26, 768)
msg_vecs = np.array([m["embedding"] for m in msg_data])  # (13, 768)

# === Standard cosine between TERMS ===
print("=== Standard Cosine Between Terms ===")
tnorms = np.linalg.norm(term_vecs, axis=1, keepdims=True)
cos_terms = (term_vecs @ term_vecs.T) / (tnorms @ tnorms.T)

pairs = []
for i in range(len(terms)):
    for j in range(i+1, len(terms)):
        pairs.append((terms[i], terms[j], cos_terms[i,j]))

pairs.sort(key=lambda x: x[2])
print("MOST OPPOSED:")
for a,b,c in pairs[:10]:
    print(f"  {a:20s} ↔ {b:20s}  cos = {c:+.4f}")
print("MOST ALIGNED:")
for a,b,c in sorted(pairs, key=lambda x: -x[2])[:10]:
    print(f"  {a:20s} ↔ {b:20s}  cos = {c:+.4f}")
print(f"Stats: min={cos_terms[np.triu_indices(len(terms),1)].min():.4f}  "
      f"max={cos_terms[np.triu_indices(len(terms),1)].max():.4f}  "
      f"mean={cos_terms[np.triu_indices(len(terms),1)].mean():.4f}")

# === Z-scored logit detection ===
print("\n=== Z-Scored Logit Detection ===")
logits = msg_vecs @ term_vecs.T  # (13, 26)

for i in range(len(msg_data)):
    speaker = msg_data[i]["speaker"]
    text = conv["messages"][i]["text"][:60]
    row = logits[i]
    z = (row - row.mean()) / max(row.std(), 1e-10)
    
    # Top 6 by z-score
    top_idx = np.argsort(-z)[:6]
    detected = [(terms[j], z[j], row[j]) for j in top_idx]
    
    # Introduced values: z > 0 (above average)
    introduced = [terms[j] for j in range(len(terms)) if z[j] > 0]
    
    print(f"\n  Turn {i:2d} [{speaker:8s}] {text}...")
    print(f"    Top 6: {', '.join(f'{t}(z={zs:.2f})' for t,zs,_ in detected)}")
    print(f"    Introduced ({len(introduced)}): {', '.join(introduced[:8])}")
