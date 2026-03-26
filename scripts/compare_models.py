#!/usr/bin/env python3
"""
Compare value geometry across multiple models using the extracted .gotue files.

This runs the comparison entirely in Python by:
  1. Loading each model's unembedding matrix (.gotue)
  2. Extracting embeddings for 28 value terms
  3. Computing pairwise cosine matrices
  4. Computing participation ratio (effective dimensionality)
  5. Comparing base vs instruction-tuned pairs

Usage:
    python compare_models.py
    python compare_models.py --models gpt2 gpt2-medium
    python compare_models.py --pair qwen2.5-0.5b qwen2.5-0.5b-instruct
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "models"

VALUE_TERMS = [
    "honesty", "integrity", "fairness", "transparency", "accountability",
    "justice", "freedom", "equality", "equity", "compassion",
    "empathy", "courage", "bravery", "wisdom", "humility",
    "loyalty", "responsibility", "resilience", "openness", "creativity",
    "innovation", "efficiency", "tradition", "cruelty", "oppression",
    "secrecy",
]


def load_gotue(path: Path) -> tuple[int, int, np.ndarray]:
    """Load a .gotue file, return (vocab_size, hidden_dim, data_matrix)."""
    raw = path.read_bytes()
    assert raw[:4] == b"GOTU", f"Bad magic in {path}"
    offset = 4
    _version = struct.unpack_from("<H", raw, offset)[0]
    offset += 2
    vocab_size = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    hidden_dim = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    total = vocab_size * hidden_dim
    values = np.frombuffer(raw, dtype="<f4", offset=offset, count=total)
    return vocab_size, hidden_dim, values.reshape(vocab_size, hidden_dim)


def load_vocab(path: Path) -> list[str]:
    """Load vocab JSON, strip BPE prefix."""
    with open(path, encoding="utf-8") as f:
        tokens = json.load(f)
    return [t.replace("\u0120", "").replace("Ġ", "") for t in tokens]


def resolve_terms_from_analysis(
    name: str, matrix: np.ndarray, data_dir: Path
) -> dict[str, np.ndarray]:
    """Resolve value terms using pre-computed term analysis (token indices from tokenizer.encode)."""
    analysis_path = data_dir / f"{name}-term-analysis.json"
    if not analysis_path.exists():
        return {}

    with open(analysis_path, encoding="utf-8") as f:
        analysis = json.load(f)

    resolved = {}
    for term, info in analysis.items():
        if info.get("found") and info.get("token_index") is not None:
            idx = info["token_index"]
            if idx < matrix.shape[0]:
                resolved[term] = matrix[idx].copy()
    return resolved


def cosine_matrix(embeddings: dict[str, np.ndarray], terms: list[str]) -> np.ndarray:
    """Compute n×n cosine similarity matrix for shared terms."""
    vecs = np.array([embeddings[t] for t in terms])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalised = vecs / norms
    return normalised @ normalised.T


def participation_ratio(cos_mat: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute participation ratio from cosine matrix eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(cos_mat)
    eigenvalues = np.maximum(eigenvalues, 0)  # clamp numerical noise
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    if sum_eig_sq < 1e-15:
        return 1.0, eigenvalues[::-1]
    pr = (sum_eig ** 2) / sum_eig_sq
    return float(pr), eigenvalues[::-1]  # descending


def mean_centre(embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean-centre embeddings to expose contrastive structure."""
    vecs = np.array(list(embeddings.values()))
    mean = vecs.mean(axis=0)
    return {k: v - mean for k, v in embeddings.items()}


def compare_pair(
    base_name: str,
    comp_name: str,
    base_embeddings: dict[str, np.ndarray],
    comp_embeddings: dict[str, np.ndarray],
) -> dict:
    """Compare two models' value geometry."""
    # Find shared terms
    shared = sorted(set(base_embeddings.keys()) & set(comp_embeddings.keys()))
    base_only = sorted(set(base_embeddings.keys()) - set(comp_embeddings.keys()))
    comp_only = sorted(set(comp_embeddings.keys()) - set(base_embeddings.keys()))

    if len(shared) < 2:
        print(f"  ERROR: only {len(shared)} shared terms, need at least 2")
        return None

    # Mean-centre for pairwise analysis
    base_centred = mean_centre({t: base_embeddings[t] for t in shared})
    comp_centred = mean_centre({t: comp_embeddings[t] for t in shared})

    # Cosine matrices
    base_cos = cosine_matrix(base_centred, shared)
    comp_cos = cosine_matrix(comp_centred, shared)

    # Participation ratios
    base_pr, base_spectrum = participation_ratio(base_cos)
    comp_pr, comp_spectrum = participation_ratio(comp_cos)

    # Frobenius distance between cosine matrices
    frob = np.linalg.norm(comp_cos - base_cos)

    # Per-term drift (if same dimension)
    base_dim = next(iter(base_embeddings.values())).shape[0]
    comp_dim = next(iter(comp_embeddings.values())).shape[0]
    term_drifts = {}
    if base_dim == comp_dim:
        for t in shared:
            bv = base_embeddings[t]
            cv = comp_embeddings[t]
            cos = np.dot(bv, cv) / (np.linalg.norm(bv) * np.linalg.norm(cv) + 1e-8)
            term_drifts[t] = float(cos)

    # Top relationship changes
    changes = []
    for i, ta in enumerate(shared):
        for j in range(i + 1, len(shared)):
            tb = shared[j]
            bc = float(base_cos[i, j])
            cc = float(comp_cos[i, j])
            changes.append((ta, tb, bc, cc, cc - bc))
    changes.sort(key=lambda x: abs(x[4]), reverse=True)

    return {
        "base": base_name,
        "compared": comp_name,
        "shared_terms": len(shared),
        "base_only": base_only,
        "comp_only": comp_only,
        "base_pr": base_pr,
        "comp_pr": comp_pr,
        "pr_delta": comp_pr - base_pr,
        "base_spectrum": base_spectrum[:5].tolist(),
        "comp_spectrum": comp_spectrum[:5].tolist(),
        "frobenius_distance": float(frob),
        "term_drifts": term_drifts,
        "top_changes": changes[:15],
        "base_dim": base_dim,
        "comp_dim": comp_dim,
    }


def print_comparison(result: dict) -> None:
    """Pretty-print a comparison result."""
    print(f"\n{'='*70}")
    print(f"  {result['base']}  vs  {result['compared']}")
    print(f"{'='*70}")
    print(f"  Hidden dims: {result['base_dim']} vs {result['comp_dim']}")
    print(f"  Shared terms: {result['shared_terms']}")
    if result['base_only']:
        print(f"  Base only: {', '.join(result['base_only'])}")
    if result['comp_only']:
        print(f"  Compared only: {', '.join(result['comp_only'])}")

    pr_delta = result['pr_delta']
    label = (
        "COLLAPSE DETECTED" if pr_delta < -1.0 else
        "contraction" if pr_delta < -0.3 else
        "stable" if abs(pr_delta) < 0.3 else
        "expansion"
    )
    print(f"\n  Effective Dimensionality (Participation Ratio):")
    print(f"    {result['base']:>35s}: {result['base_pr']:6.2f} / {result['shared_terms']}")
    print(f"    {result['compared']:>35s}: {result['comp_pr']:6.2f} / {result['shared_terms']}")
    print(f"    {'Delta':>35s}: {pr_delta:+6.2f}  [{label}]")

    print(f"\n  Eigenspectrum (top 5):")
    print(f"    Base:     {' '.join(f'{v:6.3f}' for v in result['base_spectrum'])}")
    print(f"    Compared: {' '.join(f'{v:6.3f}' for v in result['comp_spectrum'])}")

    print(f"\n  Cosine matrix Frobenius distance: {result['frobenius_distance']:.4f}")

    if result['term_drifts']:
        sorted_drifts = sorted(result['term_drifts'].items(), key=lambda x: x[1])
        mean_drift = np.mean([1.0 - v for v in result['term_drifts'].values()])
        print(f"\n  Per-term embedding drift (mean cosine distance: {mean_drift:.4f}):")
        for term, cos in sorted_drifts:
            bar_len = int((1.0 - cos) * 60)
            bar = "#" * min(bar_len, 60)
            print(f"    {term:<20s} cos={cos:.4f}  {bar}")

    print(f"\n  Top relationship changes:")
    for ta, tb, bc, cc, delta in result['top_changes'][:10]:
        arrow = "+" if delta > 0 else ""
        print(f"    {ta:<15s} <-> {tb:<15s}  {bc:+.3f} -> {cc:+.3f}  ({arrow}{delta:.3f})")


def load_model(name: str, data_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    """Load a model's value term embeddings from extracted files."""
    gotue_path = data_dir / f"{name}.gotue"

    if not gotue_path.exists():
        print(f"  ERROR: {gotue_path} not found. Run extract_models.py first.")
        return None, 0

    print(f"  Loading {name}...")
    vocab_size, hidden_dim, matrix = load_gotue(gotue_path)
    embeddings = resolve_terms_from_analysis(name, matrix, data_dir)
    print(f"    {vocab_size} vocab × {hidden_dim} dim, {len(embeddings)}/{len(VALUE_TERMS)} terms resolved")
    return embeddings, hidden_dim


def main():
    parser = argparse.ArgumentParser(description="Compare value geometry across models")
    parser.add_argument("--pair", nargs=2, help="Compare two specific models")
    parser.add_argument("--all", action="store_true", help="Run all available comparisons")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    # Default pairs for Conjecture 3 testing
    default_pairs = [
        ("gpt2", "gpt2-medium"),                        # Scaling
        ("qwen2.5-0.5b", "qwen2.5-0.5b-instruct"),     # Base vs instruct
        ("tinyllama-base", "tinyllama-chat"),            # Base vs chat (SFT + DPO)
        ("stablelm-base", "stablelm-tuned"),            # Base vs PPO RLHF
    ]

    if args.pair:
        pairs = [tuple(args.pair)]
    elif args.all:
        pairs = default_pairs
    else:
        # Run whatever models are available
        pairs = []
        for base, comp in default_pairs:
            if (args.data_dir / f"{base}.gotue").exists() and (args.data_dir / f"{comp}.gotue").exists():
                pairs.append((base, comp))
        if not pairs:
            print("No model pairs found. Run extract_models.py first.")
            print("Available .gotue files:")
            for f in sorted(args.data_dir.glob("*.gotue")):
                print(f"  {f.stem}")
            sys.exit(1)

    # Load all needed models
    models = {}
    for base, comp in pairs:
        for name in [base, comp]:
            if name not in models:
                emb, dim = load_model(name, args.data_dir)
                if emb is not None:
                    models[name] = emb

    # Run comparisons
    results = []
    for base, comp in pairs:
        if base in models and comp in models:
            result = compare_pair(base, comp, models[base], models[comp])
            if result:
                results.append(result)
                print_comparison(result)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Comparison':<45s} {'Base PR':>8s} {'Comp PR':>8s} {'Delta':>8s}")
        print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            label = f"{r['base']} vs {r['compared']}"
            print(f"  {label:<45s} {r['base_pr']:8.2f} {r['comp_pr']:8.2f} {r['pr_delta']:+8.2f}")

    # Save results
    output_path = args.data_dir / "comparison-results.json"
    serialisable = []
    for r in results:
        r2 = dict(r)
        r2['base_spectrum'] = [float(v) for v in r2['base_spectrum']]
        r2['comp_spectrum'] = [float(v) for v in r2['comp_spectrum']]
        r2['top_changes'] = [(a, b, float(bc), float(cc), float(d)) for a, b, bc, cc, d in r2['top_changes']]
        serialisable.append(r2)
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
