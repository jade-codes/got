#!/usr/bin/env python3
"""
Non-value control experiment: does the geometry we measure for value terms
differ from what we'd see for arbitrary word categories?

If participation ratio, curvature rankings, and instruction-tuning invariance
are just generic properties of any word cluster, our value-geometry findings
are vacuous. This experiment tests that by running the same analyses on
matched non-value control sets:

  1. Concrete objects  (table, hammer, river, ...) — no moral content
  2. Colour terms      (red, blue, green, ...) — perceptual, low-dimensional
  3. Random words       (several, although, ...) — no semantic coherence
  4. Profession terms   (doctor, teacher, ...) — social but not moral

For each control set and each model, we compute:
  - Participation ratio (effective dimensionality)
  - Mean pairwise cosine (clustering tightness)
  - Curvature statistics (Menger curvature of all triples)
  - Instruction-tuning delta (for base/tuned pairs)

If value terms are geometrically special, we expect:
  - Different PR from controls (not necessarily higher or lower)
  - Different curvature distribution
  - Value terms more affected OR less affected by instruction tuning
  - Any systematic difference that can't be explained by "it's just embeddings"

Usage:
    python scripts/control_experiment.py
    python scripts/control_experiment.py --models gpt2 qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
import json
import struct
from itertools import combinations
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "models"

# ── Term sets ──────────────────────────────────────────────────────────────

VALUE_TERMS = [
    "honesty", "integrity", "fairness", "transparency", "accountability",
    "justice", "freedom", "equality", "equity", "compassion",
    "empathy", "courage", "bravery", "wisdom", "humility",
    "loyalty", "responsibility", "resilience", "openness", "creativity",
    "innovation", "efficiency", "tradition", "cruelty", "oppression",
    "secrecy",
]

CONCRETE_TERMS = [
    "table", "hammer", "river", "mountain", "window",
    "bicycle", "forest", "bridge", "kitchen", "garden",
    "bottle", "carpet", "library", "candle", "mirror",
    "blanket", "fountain", "ladder", "garage", "tunnel",
    "basket", "curtain", "envelope", "lantern", "notebook",
    "umbrella",
]

COLOUR_TERMS = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "black", "white", "brown", "pink",
    "grey", "silver", "golden", "violet", "crimson",
    "scarlet", "ivory", "bronze", "coral", "amber",
    "indigo", "turquoise", "maroon", "magenta", "tan",
    "beige",
]

RANDOM_TERMS = [
    "several", "although", "perhaps", "already", "another",
    "between", "through", "without", "against", "during",
    "before", "because", "however", "whether", "neither",
    "toward", "beyond", "within", "almost", "always",
    "enough", "rather", "simply", "around", "indeed",
    "often",
]

PROFESSION_TERMS = [
    "doctor", "teacher", "lawyer", "engineer", "artist",
    "farmer", "soldier", "merchant", "carpenter", "scientist",
    "musician", "journalist", "librarian", "mechanic", "pilot",
    "architect", "detective", "surgeon", "translator", "plumber",
    "electrician", "pharmacist", "accountant", "therapist", "professor",
    "dentist",
]

TERM_SETS = {
    "values": VALUE_TERMS,
    "concrete": CONCRETE_TERMS,
    "colours": COLOUR_TERMS,
    "random": RANDOM_TERMS,
    "professions": PROFESSION_TERMS,
}

# ── Models ─────────────────────────────────────────────────────────────────

PAIRS = [
    ("qwen2.5-0.5b", "qwen2.5-0.5b-instruct", "SFT"),
    ("tinyllama-base", "tinyllama-chat", "SFT+DPO"),
    ("stablelm-base", "stablelm-tuned", "PPO-RLHF"),
]

STANDALONE = ["gpt2", "gpt2-medium"]


# ── Utilities ──────────────────────────────────────────────────────────────

def load_gotue(path: Path):
    raw = path.read_bytes()
    assert raw[:4] == b"GOTU", f"Bad magic in {path}"
    offset = 6
    vocab_size = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    hidden_dim = struct.unpack_from("<I", raw, offset)[0]; offset += 4
    values = np.frombuffer(raw, dtype="<f4", offset=offset, count=vocab_size * hidden_dim)
    return vocab_size, hidden_dim, values.reshape(vocab_size, hidden_dim)


def load_vocab(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        tokens = json.load(f)
    return [t.replace("\u0120", "").replace("Ġ", "") for t in tokens]


def resolve_terms(vocab: list[str], matrix: np.ndarray, terms: list[str]) -> dict[str, np.ndarray]:
    """Resolve single-token terms from vocab."""
    vocab_lower = {t.lower(): i for i, t in enumerate(vocab)}
    resolved = {}
    for term in terms:
        idx = vocab_lower.get(term.lower())
        if idx is not None and idx < matrix.shape[0]:
            resolved[term] = matrix[idx].copy()
    return resolved


def cosine_matrix(embeddings: dict[str, np.ndarray], terms: list[str]) -> np.ndarray:
    vecs = np.array([embeddings[t] for t in terms])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = vecs / norms
    return normed @ normed.T


def participation_ratio(cos_mat: np.ndarray) -> tuple[float, np.ndarray]:
    eigenvalues = np.linalg.eigvalsh(cos_mat)
    eigenvalues = np.maximum(eigenvalues, 0)
    s = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    if s2 < 1e-15:
        return 1.0, eigenvalues[::-1]
    return float(s ** 2 / s2), eigenvalues[::-1]


def mean_centre(embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    vecs = np.array(list(embeddings.values()))
    mean = vecs.mean(axis=0)
    return {k: v - mean for k, v in embeddings.items()}


def menger_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Menger curvature of a triangle in R^d."""
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    if ab < 1e-12 or bc < 1e-12 or ca < 1e-12:
        return 0.0
    s = (ab + bc + ca) / 2
    area_sq = s * (s - ab) * (s - bc) * (s - ca)
    if area_sq <= 0:
        return 0.0
    area = np.sqrt(area_sq)
    return 4 * area / (ab * bc * ca)


def curvature_stats(embeddings: dict[str, np.ndarray], terms: list[str]) -> dict:
    """Compute curvature statistics for a term set."""
    if len(terms) < 3:
        return {"mean_kappa": 0, "max_kappa": 0, "std_kappa": 0, "n_triples": 0}

    kappas = []
    for a, b, c in combinations(terms, 3):
        k = menger_curvature(embeddings[a], embeddings[b], embeddings[c])
        kappas.append(k)

    kappas = np.array(kappas)
    return {
        "mean_kappa": float(kappas.mean()),
        "max_kappa": float(kappas.max()),
        "std_kappa": float(kappas.std()),
        "median_kappa": float(np.median(kappas)),
        "n_triples": len(kappas),
    }


def mean_pairwise_cosine(embeddings: dict[str, np.ndarray], terms: list[str]) -> float:
    """Mean off-diagonal cosine similarity (measures clustering tightness)."""
    cos = cosine_matrix(embeddings, terms)
    n = len(terms)
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(cos[mask].mean())


# ── Main analysis ──────────────────────────────────────────────────────────

def analyse_model(name: str, data_dir: Path) -> dict | None:
    gotue_path = data_dir / f"{name}.gotue"
    vocab_path = data_dir / f"{name}-vocab.json"

    if not gotue_path.exists():
        return None

    vocab_size, hidden_dim, matrix = load_gotue(gotue_path)
    vocab = load_vocab(vocab_path) if vocab_path.exists() else None

    if vocab is None:
        return None

    results = {"model": name, "hidden_dim": hidden_dim, "vocab_size": vocab_size, "sets": {}}

    for set_name, terms in TERM_SETS.items():
        resolved = resolve_terms(vocab, matrix, terms)
        resolved_terms = sorted(resolved.keys())
        n = len(resolved_terms)

        if n < 3:
            results["sets"][set_name] = {
                "resolved": n, "total": len(terms), "skipped": True
            }
            continue

        centred = mean_centre(resolved)
        cos = cosine_matrix(centred, resolved_terms)
        pr, spectrum = participation_ratio(cos)
        mpc = mean_pairwise_cosine(centred, resolved_terms)
        curv = curvature_stats(resolved, resolved_terms)

        results["sets"][set_name] = {
            "resolved": n,
            "total": len(terms),
            "pr": round(pr, 3),
            "pr_normalised": round(pr / n, 4),
            "mean_pairwise_cosine": round(mpc, 4),
            "spectrum_top5": [round(float(e), 3) for e in spectrum[:5]],
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in curv.items()},
        }

    return results


def compare_tuning_effect(
    base_name: str, tuned_name: str, data_dir: Path
) -> dict | None:
    """Compare base vs tuned across all term sets."""
    gotue_b = data_dir / f"{base_name}.gotue"
    gotue_t = data_dir / f"{tuned_name}.gotue"
    vocab_b = data_dir / f"{base_name}-vocab.json"
    vocab_t = data_dir / f"{tuned_name}-vocab.json"

    if not all(p.exists() for p in [gotue_b, gotue_t, vocab_b, vocab_t]):
        return None

    _, _, mat_b = load_gotue(gotue_b)
    _, _, mat_t = load_gotue(gotue_t)
    voc_b = load_vocab(vocab_b)
    voc_t = load_vocab(vocab_t)

    results = {}
    for set_name, terms in TERM_SETS.items():
        res_b = resolve_terms(voc_b, mat_b, terms)
        res_t = resolve_terms(voc_t, mat_t, terms)
        shared = sorted(set(res_b.keys()) & set(res_t.keys()))

        if len(shared) < 3:
            results[set_name] = {"shared": len(shared), "skipped": True}
            continue

        # PR comparison
        cent_b = mean_centre({t: res_b[t] for t in shared})
        cent_t = mean_centre({t: res_t[t] for t in shared})
        cos_b = cosine_matrix(cent_b, shared)
        cos_t = cosine_matrix(cent_t, shared)
        pr_b, _ = participation_ratio(cos_b)
        pr_t, _ = participation_ratio(cos_t)

        # Frobenius distance
        frob = float(np.linalg.norm(cos_t - cos_b))

        # Per-term embedding drift
        drifts = []
        dim_b = res_b[shared[0]].shape[0]
        dim_t = res_t[shared[0]].shape[0]
        if dim_b == dim_t:
            for t in shared:
                bv, tv = res_b[t], res_t[t]
                c = np.dot(bv, tv) / (np.linalg.norm(bv) * np.linalg.norm(tv) + 1e-8)
                drifts.append(float(c))

        # Curvature comparison
        curv_b = curvature_stats(res_b, shared)
        curv_t = curvature_stats(res_t, shared)

        results[set_name] = {
            "shared": len(shared),
            "base_pr": round(pr_b, 3),
            "tuned_pr": round(pr_t, 3),
            "pr_delta": round(pr_t - pr_b, 4),
            "frobenius": round(frob, 4),
            "mean_drift": round(np.mean(drifts), 6) if drifts else None,
            "min_drift": round(min(drifts), 6) if drifts else None,
            "base_mean_kappa": round(curv_b["mean_kappa"], 4),
            "tuned_mean_kappa": round(curv_t["mean_kappa"], 4),
            "kappa_delta": round(curv_t["mean_kappa"] - curv_b["mean_kappa"], 4),
        }

    return results


def print_model_results(results: dict):
    name = results["model"]
    dim = results["hidden_dim"]
    print(f"\n{'=' * 70}")
    print(f"  {name}  ({dim}-dim)")
    print(f"{'=' * 70}")
    print(f"  {'Set':<14} {'n':>3} {'PR':>8} {'PR/n':>7} {'MeanCos':>9} "
          f"{'MeanK':>9} {'MaxK':>9} {'StdK':>9}")
    print(f"  {'-' * 14} {'-' * 3} {'-' * 8} {'-' * 7} {'-' * 9} "
          f"{'-' * 9} {'-' * 9} {'-' * 9}")

    for set_name in TERM_SETS:
        s = results["sets"].get(set_name, {})
        if s.get("skipped"):
            print(f"  {set_name:<14} {s.get('resolved', 0):>3}  (skipped, too few terms)")
            continue
        print(f"  {set_name:<14} {s['resolved']:>3} {s['pr']:>8.3f} {s['pr_normalised']:>7.4f} "
              f"{s['mean_pairwise_cosine']:>9.4f} {s['mean_kappa']:>9.4f} "
              f"{s['max_kappa']:>9.4f} {s['std_kappa']:>9.4f}")


def print_tuning_comparison(base: str, tuned: str, method: str, results: dict):
    print(f"\n{'=' * 70}")
    print(f"  TUNING EFFECT: {base} vs {tuned} ({method})")
    print(f"{'=' * 70}")
    print(f"  {'Set':<14} {'n':>3} {'BasePR':>8} {'TunePR':>8} {'Delta':>8} "
          f"{'Frob':>7} {'MeanDrift':>10} {'K_delta':>8}")
    print(f"  {'-' * 14} {'-' * 3} {'-' * 8} {'-' * 8} {'-' * 8} "
          f"{'-' * 7} {'-' * 10} {'-' * 8}")

    for set_name in TERM_SETS:
        s = results.get(set_name, {})
        if s.get("skipped"):
            print(f"  {set_name:<14} {s.get('shared', 0):>3}  (skipped)")
            continue

        drift_str = f"{1 - s['mean_drift']:.6f}" if s["mean_drift"] is not None else "N/A"
        print(f"  {set_name:<14} {s['shared']:>3} {s['base_pr']:>8.3f} {s['tuned_pr']:>8.3f} "
              f"{s['pr_delta']:>+8.4f} {s['frobenius']:>7.4f} {drift_str:>10} "
              f"{s['kappa_delta']:>+8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Non-value control experiment")
    parser.add_argument("--models", nargs="*", help="Specific models to analyse")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "control-experiment.json")
    args = parser.parse_args()

    all_models = STANDALONE + [m for p in PAIRS for m in p[:2]]
    if args.models:
        all_models = args.models

    # ── Per-model analysis ──
    print("\n" + "=" * 70)
    print("  NON-VALUE CONTROL EXPERIMENT")
    print("  Comparing value terms against matched control sets")
    print("=" * 70)

    model_results = {}
    for name in all_models:
        r = analyse_model(name, args.data_dir)
        if r:
            model_results[name] = r
            print_model_results(r)
        else:
            print(f"\n  {name}: skipped (missing .gotue or vocab)")

    # ── Tuning effect comparison ──
    tuning_results = {}
    for base, tuned, method in PAIRS:
        r = compare_tuning_effect(base, tuned, args.data_dir)
        if r:
            tuning_results[f"{base}_vs_{tuned}"] = {"method": method, "sets": r}
            print_tuning_comparison(base, tuned, method, r)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: Value terms vs controls")
    print(f"{'=' * 70}")

    # Aggregate PR/n across models
    set_prs = {s: [] for s in TERM_SETS}
    for name, r in model_results.items():
        for set_name in TERM_SETS:
            s = r["sets"].get(set_name, {})
            if not s.get("skipped") and "pr_normalised" in s:
                set_prs[set_name].append(s["pr_normalised"])

    print(f"\n  Mean PR/n across all models (higher = more spread):")
    for set_name in TERM_SETS:
        vals = set_prs[set_name]
        if vals:
            print(f"    {set_name:<14} {np.mean(vals):.4f}  (n={len(vals)} models)")

    # Aggregate tuning deltas
    if tuning_results:
        print(f"\n  Mean tuning PR delta across pairs:")
        for set_name in TERM_SETS:
            deltas = []
            for pair_key, pair_data in tuning_results.items():
                s = pair_data["sets"].get(set_name, {})
                if not s.get("skipped") and "pr_delta" in s:
                    deltas.append(s["pr_delta"])
            if deltas:
                print(f"    {set_name:<14} {np.mean(deltas):>+.4f}  (n={len(deltas)} pairs)")

        print(f"\n  Mean tuning Frobenius distance across pairs:")
        for set_name in TERM_SETS:
            frobs = []
            for pair_key, pair_data in tuning_results.items():
                s = pair_data["sets"].get(set_name, {})
                if not s.get("skipped") and "frobenius" in s:
                    frobs.append(s["frobenius"])
            if frobs:
                print(f"    {set_name:<14} {np.mean(frobs):.4f}  (n={len(frobs)} pairs)")

    # Save
    output = {
        "experiment": "non-value-control",
        "description": "Comparing value-term geometry against matched non-value control sets",
        "term_sets": {k: v for k, v in TERM_SETS.items()},
        "models": model_results,
        "tuning_comparisons": tuning_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
