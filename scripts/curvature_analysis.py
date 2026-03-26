#!/usr/bin/env python3
"""
Compute curvature of value subspaces for each extracted model.

For each model, computes:
  - Menger curvature for all value-term triples
  - Per-term mean/max curvature
  - Local participation ratio (k-nearest-neighbour dimensionality)
  - Angle deficit (deviation from flat space)

This provides the geometric side of Conjecture 2. The human side
(moral uncertainty / deliberation time data) is required separately.

Usage:
    python curvature_analysis.py
    python curvature_analysis.py --models gpt2 qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "models"


def load_gotue(path: Path):
    raw = path.read_bytes()
    assert raw[:4] == b"GOTU"
    offset = 6
    vocab_size = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    hidden_dim = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    values = np.frombuffer(raw, dtype="<f4", offset=offset, count=vocab_size * hidden_dim)
    return vocab_size, hidden_dim, values.reshape(vocab_size, hidden_dim)


def resolve_terms(name, matrix, data_dir):
    path = data_dir / f"{name}-term-analysis.json"
    with open(path, encoding="utf-8") as f:
        analysis = json.load(f)
    resolved = {}
    for term, info in analysis.items():
        if info.get("found") and info.get("token_index") is not None:
            idx = info["token_index"]
            if idx < matrix.shape[0]:
                resolved[term] = matrix[idx].copy()
    return resolved


def mean_centre(embeddings):
    vecs = np.array(list(embeddings.values()))
    mean = vecs.mean(axis=0)
    return {k: v - mean for k, v in embeddings.items()}


def menger_curvature(a, b, c):
    """Menger curvature of triangle ABC."""
    d_ab = np.linalg.norm(b - a)
    d_bc = np.linalg.norm(c - b)
    d_ca = np.linalg.norm(a - c)

    # Area via Heron's formula
    s = (d_ab + d_bc + d_ca) / 2
    area_sq = s * (s - d_ab) * (s - d_bc) * (s - d_ca)
    area = np.sqrt(max(area_sq, 0))

    product = d_ab * d_bc * d_ca
    if product < 1e-10:
        return 0, area, 0, 0, 0

    kappa = 4 * area / product

    # Angles via law of cosines
    def angle(adj1, adj2, opp):
        denom = 2 * adj1 * adj2
        if denom < 1e-10:
            return 0
        cos_a = (adj1**2 + adj2**2 - opp**2) / denom
        return np.arccos(np.clip(cos_a, -1, 1))

    ang_a = angle(d_ab, d_ca, d_bc)
    ang_b = angle(d_ab, d_bc, d_ca)
    ang_c = angle(d_ca, d_bc, d_ab)

    return kappa, area, ang_a, ang_b, ang_c


def local_pr(cosine_mat, idx, k):
    """Participation ratio of k nearest neighbours of idx."""
    n = cosine_mat.shape[0]
    # Sort by distance (1 - cosine)
    dists = 1 - cosine_mat[idx]
    neighbours = np.argsort(dists)[1:k+1]  # skip self
    if len(neighbours) < 2:
        return 1.0

    sub = cosine_mat[np.ix_(neighbours, neighbours)]
    eigenvalues = np.linalg.eigvalsh(sub)
    eigenvalues = np.maximum(eigenvalues, 0)
    s = eigenvalues.sum()
    s2 = (eigenvalues**2).sum()
    if s2 < 1e-15:
        return 1.0
    return float(s**2 / s2)


def analyse_model(name, data_dir, k=5):
    gotue_path = data_dir / f"{name}.gotue"
    if not gotue_path.exists():
        print(f"  {name}: .gotue not found, skipping")
        return None

    print(f"\n  Loading {name}...")
    vocab_size, hidden_dim, matrix = load_gotue(gotue_path)
    raw_embeddings = resolve_terms(name, matrix, data_dir)
    embeddings = mean_centre(raw_embeddings)
    terms = sorted(embeddings.keys())
    n = len(terms)
    print(f"    {n} terms resolved, dim={hidden_dim}")

    if n < 3:
        print(f"    Too few terms for curvature analysis")
        return None

    vecs = np.array([embeddings[t] for t in terms])

    # Cosine matrix
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = vecs / norms
    cos_mat = normed @ normed.T

    # All triples
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for ki in range(j+1, n):
                kappa, area, ang_a, ang_b, ang_c = menger_curvature(
                    vecs[i], vecs[j], vecs[ki]
                )
                triples.append({
                    "terms": (terms[i], terms[j], terms[ki]),
                    "kappa": kappa,
                    "area": area,
                    "angles": (ang_a, ang_b, ang_c),
                })
    triples.sort(key=lambda x: x["kappa"], reverse=True)

    # Per-term stats
    k_actual = min(k, n - 1)
    term_stats = []
    for i, term in enumerate(terms):
        term_kappas = [t["kappa"] for t in triples
                       if term in t["terms"]]
        term_angles = []
        for t in triples:
            if t["terms"][0] == term: term_angles.append(t["angles"][0])
            elif t["terms"][1] == term: term_angles.append(t["angles"][1])
            elif t["terms"][2] == term: term_angles.append(t["angles"][2])

        mean_k = np.mean(term_kappas) if term_kappas else 0
        max_k = np.max(term_kappas) if term_kappas else 0
        lpr = local_pr(cos_mat, i, k_actual)
        mean_angle = np.mean(term_angles) if term_angles else np.pi/3
        angle_deficit = mean_angle - np.pi/3

        term_stats.append({
            "term": term,
            "mean_kappa": float(mean_k),
            "max_kappa": float(max_k),
            "local_pr": float(lpr),
            "angle_deficit": float(angle_deficit),
        })

    term_stats.sort(key=lambda x: x["mean_kappa"], reverse=True)

    return {
        "model": name,
        "num_terms": n,
        "hidden_dim": hidden_dim,
        "k_neighbours": k_actual,
        "mean_kappa": float(np.mean([t["kappa"] for t in triples])),
        "max_kappa": float(np.max([t["kappa"] for t in triples])),
        "term_stats": term_stats,
        "top_triples": [
            {
                "terms": list(t["terms"]),
                "kappa": float(t["kappa"]),
                "area": float(t["area"]),
                "angles_deg": [float(np.degrees(a)) for a in t["angles"]],
            }
            for t in triples[:20]
        ],
        "bottom_triples": [
            {
                "terms": list(t["terms"]),
                "kappa": float(t["kappa"]),
                "area": float(t["area"]),
                "angles_deg": [float(np.degrees(a)) for a in t["angles"]],
            }
            for t in triples[-5:]
        ],
    }


def print_analysis(result):
    print(f"\n{'='*70}")
    print(f"  Curvature Analysis: {result['model']}")
    print(f"  {result['num_terms']} terms, {result['hidden_dim']} dim, k={result['k_neighbours']}")
    print(f"{'='*70}")

    print(f"\n  Global: mean k = {result['mean_kappa']:.4f}, max k = {result['max_kappa']:.4f}")

    print(f"\n  Per-term curvature (descending mean k):")
    print(f"  {'Term':<20s} {'Mean k':>10s} {'Max k':>10s} {'Local PR':>10s} {'Angle D':>10s}")
    print(f"  {'-'*62}")
    for ts in result["term_stats"]:
        print(f"  {ts['term']:<20s} {ts['mean_kappa']:>10.4f} {ts['max_kappa']:>10.4f} {ts['local_pr']:>10.2f} {ts['angle_deficit']:>+10.4f}")

    print(f"\n  Top 10 highest-curvature triples:")
    for t in result["top_triples"][:10]:
        a = t["angles_deg"]
        print(f"  {t['terms'][0]:<12s} {t['terms'][1]:<12s} {t['terms'][2]:<12s}  "
              f"k={t['kappa']:.4f}  area={t['area']:.4f}  "
              f"({a[0]:.0f}°, {a[1]:.0f}°, {a[2]:.0f}°)")

    print(f"\n  Bottom 5 lowest-curvature triples (most collinear):")
    for t in result["bottom_triples"]:
        a = t["angles_deg"]
        print(f"  {t['terms'][0]:<12s} {t['terms'][1]:<12s} {t['terms'][2]:<12s}  "
              f"k={t['kappa']:.4f}  area={t['area']:.4f}  "
              f"({a[0]:.0f}°, {a[1]:.0f}°, {a[2]:.0f}°)")

    # Conjecture 2 predictions
    print(f"\n  Conjecture 2 predictions:")
    print(f"  The following terms sit in the highest-curvature regions,")
    print(f"  where the value landscape is most 'bent'. Conjecture 2")
    print(f"  predicts these correspond to topics where humans report")
    print(f"  greater moral uncertainty:")
    for ts in result["term_stats"][:5]:
        print(f"    {ts['term']:<20s}  k = {ts['mean_kappa']:.4f}")
    print(f"  The following terms sit in the lowest-curvature (flattest) regions:")
    for ts in result["term_stats"][-5:]:
        print(f"    {ts['term']:<20s}  k = {ts['mean_kappa']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Curvature analysis of value subspaces")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--k", type=int, default=5, help="Neighbours for local PR")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    available = [p.stem for p in args.data_dir.glob("*.gotue")]
    models = args.models or available
    if not models:
        print("No models found. Run extract_models.py first.")
        return

    results = []
    for name in models:
        result = analyse_model(name, args.data_dir, args.k)
        if result:
            results.append(result)
            print_analysis(result)

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  CURVATURE COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Model':<35s} {'Mean k':>10s} {'Max k':>10s} {'Mean LPR':>10s}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
        for r in results:
            mean_lpr = np.mean([t["local_pr"] for t in r["term_stats"]])
            print(f"  {r['model']:<35s} {r['mean_kappa']:>10.4f} {r['max_kappa']:>10.4f} {mean_lpr:>10.2f}")

    # Save
    output_path = args.data_dir / "curvature-results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
