#!/usr/bin/env python3
"""Analyse what centering does to GPT-2 term embeddings."""
import json, numpy as np

# Load the term embeddings (from the vocab lookup)
vocab = json.load(open("data/models/gpt2-vocab.json"))
import struct

# Load the .gotue file
with open("data/models/gpt2.gotue", "rb") as f:
    magic = f.read(4)
    version = struct.unpack("<H", f.read(2))[0]
    vocab_size = struct.unpack("<I", f.read(4))[0]
    hidden_dim = struct.unpack("<I", f.read(4))[0]
    data = f.read()
    matrix = np.frombuffer(data, dtype=np.float32).reshape(vocab_size, hidden_dim)

print(f"Unembedding: {vocab_size} x {hidden_dim}")

# Clean vocab (strip BPE prefix)
vocab_clean = [t.replace("Ġ", "") for t in vocab]

VALUE_TERMS = [
    "accountability", "bravery", "compassion", "courage",
    "creativity", "cruelty", "efficiency", "empathy", "equality",
    "equity", "fairness", "freedom", "honesty", "humility",
    "innovation", "integrity", "justice", "loyalty", "openness",
    "oppression", "resilience", "responsibility", "secrecy",
    "tradition", "transparency", "wisdom",
]

# Get term embeddings
term_embs = {}
for term in VALUE_TERMS:
    if term in vocab_clean:
        idx = vocab_clean.index(term)
        term_embs[term] = matrix[idx]

terms = sorted(term_embs.keys())
E = np.array([term_embs[t] for t in terms])
print(f"\nTerms: {len(terms)}")

# Raw cosines
norms = np.linalg.norm(E, axis=1, keepdims=True)
E_norm = E / norms
raw_cos = E_norm @ E_norm.T

# Extract upper triangle
n = len(terms)
raw_vals = []
for i in range(n):
    for j in range(i+1, n):
        raw_vals.append(raw_cos[i,j])
raw_vals = np.array(raw_vals)

print(f"\n=== RAW cosines ===")
print(f"  min={raw_vals.min():.4f}  max={raw_vals.max():.4f}  mean={raw_vals.mean():.4f}  std={raw_vals.std():.4f}")

# Mean-center
mean_emb = E.mean(axis=0)
E_centered = E - mean_emb
norms_c = np.linalg.norm(E_centered, axis=1, keepdims=True)
E_c_norm = E_centered / norms_c
centered_cos = E_c_norm @ E_c_norm.T

centered_vals = []
for i in range(n):
    for j in range(i+1, n):
        centered_vals.append(centered_cos[i,j])
centered_vals = np.array(centered_vals)

print(f"\n=== CENTERED cosines ===")
print(f"  min={centered_vals.min():.4f}  max={centered_vals.max():.4f}  mean={centered_vals.mean():.4f}  std={centered_vals.std():.4f}")

# Show the most opposed pairs (lowest centered cosine)
pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append((terms[i], terms[j], raw_cos[i,j], centered_cos[i,j]))

pairs.sort(key=lambda x: x[3])

print(f"\n=== Most OPPOSED (centered) ===")
for a, b, raw, cent in pairs[:15]:
    print(f"  {a:20s} <-> {b:20s}  raw={raw:.3f}  centered={cent:.3f}")

print(f"\n=== Most ALIGNED (centered) ===")
for a, b, raw, cent in sorted(pairs, key=lambda x: -x[3])[:10]:
    print(f"  {a:20s} <-> {b:20s}  raw={raw:.3f}  centered={cent:.3f}")

# Histogram bins
print(f"\n=== Distribution (centered) ===")
for lo in np.arange(-0.8, 0.9, 0.2):
    hi = lo + 0.2
    count = ((centered_vals >= lo) & (centered_vals < hi)).sum()
    bar = "#" * count
    print(f"  [{lo:+.1f}, {hi:+.1f}): {count:3d}  {bar}")

# What thresholds would work?
below_neg03 = (centered_vals < -0.3).sum()
below_neg05 = (centered_vals < -0.5).sum()
above_06 = (centered_vals > 0.6).sum()
above_08 = (centered_vals > 0.8).sum()
print(f"\n  cos < -0.3: {below_neg03} pairs")
print(f"  cos < -0.5: {below_neg05} pairs")
print(f"  cos > 0.6: {above_06} pairs")
print(f"  cos > 0.8: {above_08} pairs")
