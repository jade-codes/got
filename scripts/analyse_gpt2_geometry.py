#!/usr/bin/env python3
"""
Analyse GPT-2 term geometry:
1. Pairwise causal cosines between value terms
2. Raw logits (h · u_i) for message-term detection
3. Check if logits discriminate messages
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

# Load unembedding matrix
import struct
with open("data/models/gpt2.gotue", "rb") as f:
    magic = f.read(4)
    version = struct.unpack("<H", f.read(2))[0]
    vocab_size = struct.unpack("<I", f.read(4))[0]
    hidden_dim = struct.unpack("<I", f.read(4))[0]
    raw = np.frombuffer(f.read(), dtype=np.float32)
    U = raw.reshape(vocab_size, hidden_dim)

print(f"U shape: {U.shape}")

# Extract term embeddings (rows of U)
terms = []
term_vecs = []
for name, info in sorted(term_info.items()):
    if info["found"]:
        idx = info["token_index"]
        terms.append(name)
        term_vecs.append(U[idx])
term_vecs = np.array(term_vecs)  # (26, 768)
print(f"Terms: {len(terms)}")

# === Part 1: Pairwise analysis ===
print("\n=== Pairwise Causal Cosines (Φ = UᵀU) ===")
# Φ = UᵀU, so cos_Φ(u_i, u_j) = (Uu_i)·(Uu_j) / (|Uu_i| * |Uu_j|)
# Where Uu_i = U @ u_i = U @ U[i] (transforms to vocab space)
# Simplification: u_i and u_j are ROWS of U, but Φ maps from hidden to hidden.
# Actually: u_i^T Φ u_j = u_i^T U^T U u_j 
# But u_i IS a row of U, so this = (U @ u_i)^T @ (U @ u_j)

# Compute Φ-norms and Φ-products
UU = U @ term_vecs.T  # (50257, 26) - each column is U @ u_i

# cos_Φ(i,j) = UU[:,i] . UU[:,j] / (|UU[:,i]| * |UU[:,j]|)
norms_phi = np.linalg.norm(UU, axis=0)  # (26,)
cos_phi = (UU.T @ UU) / np.outer(norms_phi, norms_phi)  # (26, 26)

# Find interesting pairs
pairs = []
for i in range(len(terms)):
    for j in range(i+1, len(terms)):
        pairs.append((terms[i], terms[j], cos_phi[i,j]))

# Sort by most negative (potential contradictions)
pairs.sort(key=lambda x: x[2])
print("\nMost OPPOSED pairs (low cos_Φ):")
for a, b, c in pairs[:15]:
    print(f"  {a:20s} ↔ {b:20s}  cos_Φ = {c:.4f}")

print("\nMost ALIGNED pairs (high cos_Φ):")
for a, b, c in sorted(pairs, key=lambda x: -x[2])[:15]:
    print(f"  {a:20s} ↔ {b:20s}  cos_Φ = {c:.4f}")

# === Part 2: Logit-based detection ===
print("\n=== Logit-based Detection (h · u_i) ===")
msg_vecs = np.array([m["embedding"] for m in msg_data])  # (13, 768)

# Raw logits: message @ term.T (standard dot product)
logits = msg_vecs @ term_vecs.T  # (13, 26)

# Also compute standard cosine (no Φ)
msg_norms = np.linalg.norm(msg_vecs, axis=1, keepdims=True)
term_norms_std = np.linalg.norm(term_vecs, axis=1, keepdims=True)
cos_std = (msg_vecs @ term_vecs.T) / (msg_norms @ term_norms_std.T)  # (13, 26)

for i in range(len(msg_data)):
    speaker = msg_data[i]["speaker"]
    text = conv["messages"][i]["text"][:50]
    # Top terms by logit
    top_idx = np.argsort(-logits[i])[:6]
    top_terms = [(terms[j], logits[i,j]) for j in top_idx]
    # Top terms by standard cosine
    top_cos_idx = np.argsort(-cos_std[i])[:6]
    top_cos = [(terms[j], cos_std[i,j]) for j in top_cos_idx]
    
    print(f"\n  Turn {i:2d} [{speaker:8s}] {text}...")
    print(f"    By logit:  {', '.join(f'{t}({v:.1f})' for t,v in top_terms)}")
    print(f"    By cosine: {', '.join(f'{t}({v:.4f})' for t,v in top_cos)}")

# === Part 3: Distribution stats ===
print("\n=== Global Stats ===")
print(f"Message norms:  min={msg_norms.min():.1f}  max={msg_norms.max():.1f}  mean={msg_norms.mean():.1f}")
print(f"Term norms:     min={term_norms_std.min():.3f}  max={term_norms_std.max():.3f}  mean={term_norms_std.mean():.3f}")
print(f"Logit range:    min={logits.min():.1f}  max={logits.max():.1f}  mean={logits.mean():.1f}")
print(f"Cosine range:   min={cos_std.min():.4f}  max={cos_std.max():.4f}  mean={cos_std.mean():.4f}")
print(f"Cos_Φ (terms):  min={cos_phi.min():.4f}  max={cos_phi.max():.4f}  mean off-diag={cos_phi[np.triu_indices(len(terms),1)].mean():.4f}")
