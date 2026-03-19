#!/usr/bin/env python3
"""Test progressively stronger debiasing of GPT-2 term embeddings."""
import json, numpy as np, struct

with open("data/models/gpt2.gotue", "rb") as f:
    f.read(4)  # magic
    f.read(2)  # version
    vocab_size = struct.unpack("<I", f.read(4))[0]
    hidden_dim = struct.unpack("<I", f.read(4))[0]
    data = f.read()
    matrix = np.frombuffer(data, dtype=np.float32).reshape(vocab_size, hidden_dim)

vocab = json.load(open("data/models/gpt2-vocab.json"))
vocab_clean = [t.replace("Ġ", "") for t in vocab]

VALUE_TERMS = [
    "accountability", "bravery", "compassion", "courage",
    "creativity", "cruelty", "efficiency", "empathy", "equality",
    "equity", "fairness", "freedom", "honesty", "humility",
    "innovation", "integrity", "justice", "loyalty", "openness",
    "oppression", "resilience", "responsibility", "secrecy",
    "tradition", "transparency", "wisdom",
]

term_embs = {}
for term in VALUE_TERMS:
    if term in vocab_clean:
        idx = vocab_clean.index(term)
        term_embs[term] = matrix[idx].copy()

terms = sorted(term_embs.keys())
E = np.array([term_embs[t] for t in terms])

def cosine_stats(E, label):
    """Compute and display pairwise cosine stats."""
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_n = E / norms
    C = E_n @ E_n.T
    n = len(terms)
    vals = [C[i,j] for i in range(n) for j in range(i+1, n)]
    vals = np.array(vals)
    
    pairs = [(terms[i], terms[j], C[i,j]) for i in range(n) for j in range(i+1,n)]
    pairs.sort(key=lambda x: x[2])
    
    print(f"\n=== {label} ===")
    print(f"  range: [{vals.min():.3f}, {vals.max():.3f}]  mean={vals.mean():.3f}  std={vals.std():.3f}")
    print(f"  Most opposed:")
    for a, b, c in pairs[:8]:
        print(f"    {a:20s} <-> {b:20s}  cos={c:.3f}")
    print(f"  Most aligned:")
    for a, b, c in sorted(pairs, key=lambda x: -x[2])[:5]:
        print(f"    {a:20s} <-> {b:20s}  cos={c:.3f}")
    
    # Threshold analysis
    for thr in [-0.5, -0.3, -0.2, -0.1]:
        cnt = (vals < thr).sum()
        if cnt > 0:
            print(f"  cos < {thr}: {cnt} pairs")
    return vals

# Raw
cosine_stats(E, "RAW (no centering)")

# Mean-centered
E_c = E - E.mean(axis=0)
cosine_stats(E_c, "MEAN-CENTERED")

# All-but-the-Top: remove top-k principal components
for k in [1, 2, 3, 5]:
    E_c = E - E.mean(axis=0)
    U, S, Vt = np.linalg.svd(E_c, full_matrices=False)
    # Remove top-k components
    for i in range(k):
        proj = (E_c @ Vt[i]) [:, None] * Vt[i][None, :]
        E_c = E_c - proj
    cosine_stats(E_c, f"ALL-BUT-TOP-{k} (mean + remove {k} PCs)")
