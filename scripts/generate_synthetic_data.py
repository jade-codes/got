#!/usr/bin/env python3
"""Generate synthetic test data for the GoT demo pipeline.

Creates:
  <out>/model.gotue        — 50×8 unembedding matrix (binary)
  <out>/activations.gotact — 2 layers × 20 samples (binary)
  <out>/labels.txt         — 20 binary labels
  <out>/embeddings.json    — 20 value terms in ℝ^8
"""

import argparse
import json
import math
import os
import random
import struct
import sys


def write_gotue(path: str, vocab: int, dim: int, seed: int = 42) -> None:
    random.seed(seed)
    with open(path, "wb") as f:
        f.write(b"GOTU")
        f.write(struct.pack("<H", 1))  # version
        f.write(struct.pack("<I", vocab))
        f.write(struct.pack("<I", dim))
        for _ in range(vocab * dim):
            f.write(struct.pack("<f", random.gauss(0, 1.0 / math.sqrt(dim))))


def write_gotact(path: str, dim: int, layers: list, samples: int, seed: int = 43) -> None:
    """Write .gotact binary matching the CLI's expected format.

    Format:
      "GOTA" (4 bytes magic)
      version: u16 LE
      model_id: u32 LE length + UTF-8 bytes
      precision: u8 (0 = fp32)
      hidden_dim: u32 LE
      num_layers: u32 LE
      num_positions: u32 LE
      Per layer:
        layer_index: u32 LE
        Per position:
          token_position: u32 LE
          values: hidden_dim × f32 LE
    """
    random.seed(seed)
    model_id = b"synthetic-model"
    with open(path, "wb") as f:
        f.write(b"GOTA")
        f.write(struct.pack("<H", 1))           # version
        f.write(struct.pack("<I", len(model_id)))  # model_id length
        f.write(model_id)                       # model_id bytes
        f.write(struct.pack("B", 0))            # precision (fp32)
        f.write(struct.pack("<I", dim))         # hidden_dim
        f.write(struct.pack("<I", len(layers)))  # num_layers
        f.write(struct.pack("<I", samples))     # num_positions per layer
        for layer in layers:
            f.write(struct.pack("<I", layer))   # layer_index
            for pos in range(samples):
                f.write(struct.pack("<I", pos))  # token_position
                for _ in range(dim):
                    f.write(struct.pack("<f", random.gauss(0, 1)))


def write_labels(path: str, count: int, seed: int = 44) -> None:
    random.seed(seed)
    with open(path, "w") as f:
        for _ in range(count):
            f.write(f"{random.randint(0, 1)}\n")


def unit_vec(v: list) -> list:
    n = math.sqrt(sum(x * x for x in v))
    return [round(x / n, 6) for x in v]


def make_antonym(emb: dict, term: str, anchor: str, noise: float = 0.1) -> None:
    """Set term ≈ −anchor (opposing direction with slight noise)."""
    emb[term] = unit_vec([-x + random.gauss(0, noise) for x in emb[anchor]])


def make_synonym(emb: dict, term: str, anchor: str, noise: float = 0.05) -> None:
    """Set term ≈ anchor (near-parallel with slight noise)."""
    emb[term] = unit_vec([x + random.gauss(0, noise) for x in emb[anchor]])


def make_blend(emb: dict, term: str, anchors: dict, noise: float = 0.08) -> None:
    """Set term to a weighted blend of existing terms + noise.

    anchors: dict of {term: weight}, e.g. {"courage": 0.6, "compassion": 0.4}
    """
    dim = len(next(iter(emb.values())))
    v = [0.0] * dim
    for anchor, weight in anchors.items():
        for i in range(dim):
            v[i] += weight * emb[anchor][i]
    v = [x + random.gauss(0, noise) for x in v]
    emb[term] = unit_vec(v)


def write_embeddings(path: str, dim: int, seed: int = 45) -> None:
    random.seed(seed)

    # Base terms — each gets a random unit vector
    base_terms = [
        "honesty", "transparency", "innovation", "creativity",
        "efficiency", "fairness", "loyalty", "justice",
        "courage", "compassion", "wisdom", "freedom",
        "equality", "responsibility", "resilience", "humility",
    ]
    emb = {}
    for t in base_terms:
        emb[t] = unit_vec([random.gauss(0, 1) for _ in range(dim)])

    # --- Phase 1: Cluster positive terms so they share semantic directions ---
    # This means antonyms (created next) will naturally oppose the *whole*
    # cluster, not just one isolated random vector.
    make_blend(emb, "compassion",    {
               "compassion": 0.50, "fairness": 0.15, "equality": 0.15, "freedom": 0.10, "courage": 0.10})
    make_blend(emb, "courage",       {"courage": 0.50, "honesty": 0.15,
               "responsibility": 0.15, "compassion": 0.10, "freedom": 0.10})
    make_blend(emb, "freedom",       {
               "freedom": 0.40, "equality": 0.20, "compassion": 0.15, "courage": 0.10, "fairness": 0.15})
    make_blend(emb, "justice",       {
               "fairness": 0.40, "equality": 0.25, "compassion": 0.15, "responsibility": 0.20})
    make_blend(emb, "loyalty",       {
               "responsibility": 0.40, "honesty": 0.25, "courage": 0.20, "compassion": 0.15})
    make_blend(emb, "wisdom",        {
               "compassion": 0.20, "responsibility": 0.20, "courage": 0.20, "honesty": 0.20, "fairness": 0.20})
    make_blend(emb, "transparency",  {
               "transparency": 0.40, "honesty": 0.35, "fairness": 0.15, "responsibility": 0.10})
    make_blend(emb, "responsibility", {
               "responsibility": 0.45, "fairness": 0.20, "compassion": 0.15, "honesty": 0.10, "courage": 0.10})
    make_blend(emb, "fairness",      {"fairness": 0.45, "equality": 0.20,
               "compassion": 0.15, "responsibility": 0.10, "honesty": 0.10})
    make_blend(emb, "equality",      {"equality": 0.40, "fairness": 0.25,
               "freedom": 0.15, "compassion": 0.10, "responsibility": 0.10})

    # --- Phase 2: Antonym pairs — opposing the clustered directions ---
    # Because compassion/courage/freedom now share components, cruelty (= −compassion)
    # will automatically be anti-correlated with courage, freedom, fairness etc.
    make_antonym(emb, "cruelty",     "compassion")
    make_antonym(emb, "oppression",  "freedom")
    make_antonym(emb, "cowardice",   "courage")
    make_antonym(emb, "secrecy",     "transparency")
    make_antonym(emb, "tradition",   "innovation")

    # --- Phase 3: Synonym pairs — near-aligned with clustered versions ---
    make_synonym(emb, "integrity",      "honesty")
    make_synonym(emb, "truthfulness",   "honesty")
    make_synonym(emb, "openness",       "transparency")
    make_synonym(emb, "bravery",        "courage")
    make_synonym(emb, "empathy",        "compassion")
    make_synonym(emb, "equity",         "equality")
    make_synonym(emb, "accountability", "responsibility")

    num_terms = len(emb)
    with open(path, "w") as f:
        json.dump(emb, f, indent=2)
    return num_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Generating synthetic data in {args.out}/")
    write_gotue(f"{args.out}/model.gotue", args.vocab, args.dim)
    print(f"  model.gotue        {args.vocab}×{args.dim} unembedding")
    write_gotact(f"{args.out}/activations.gotact",
                 args.dim, args.layers, args.samples)
    print(
        f"  activations.gotact {len(args.layers)} layers × {args.samples} samples")
    write_labels(f"{args.out}/labels.txt", args.samples)
    print(f"  labels.txt         {args.samples} labels")
    n = write_embeddings(f"{args.out}/embeddings.json", args.dim)
    print(f"  embeddings.json    {n} value terms in ℝ^{args.dim}")


if __name__ == "__main__":
    main()
