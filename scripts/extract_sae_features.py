#!/usr/bin/env python3
"""
Extract value-relevant SAE features and export as ProbeSet JSON.

Loads a trained SAE checkpoint, identifies features aligned with
value-relevant probe directions, and exports them in the ProbeSet
format consumed by the got-cli pipeline.

Usage:
    # With existing probes (identifies SAE features aligned with probes):
    python scripts/extract_sae_features.py \
        --sae data/sae/gpt2-layer6.pt \
        --probes data/probes_layer6.json \
        --unembedding data/models/gpt2.gotue \
        --top-k 5 \
        --output data/sae_probes_layer6.json

    # Without probes (select by activation sparsity / monosemanticity):
    python scripts/extract_sae_features.py \
        --sae data/sae/gpt2-layer6.pt \
        --unembedding data/models/gpt2.gotue \
        --top-k 10 \
        --output data/sae_features_layer6.json

    # Then use with existing pipeline:
    cargo run --release -p got-cli -- collapse-report \
        --unembedding data/models/gpt2.gotue \
        --probes data/sae_probes_layer6.json

Dependencies:
    pip install torch
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Import SAE from train_sae (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from train_sae import TopKSAE


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def compute_causal_alignment(
    feature_dirs: torch.Tensor,
    probe_weights: List[torch.Tensor],
    gram: torch.Tensor,
    d: int,
) -> torch.Tensor:
    """Compute causal inner product alignment between SAE features and probes.

    Returns (n_features, n_probes) matrix of |<f_i, w_j>_c| / (||f_i||_c * ||w_j||_c).
    """
    n_features = feature_dirs.shape[0]
    n_probes = len(probe_weights)

    # Compute Φf for all features: (n_features, d) @ (d, d) -> (n_features, d)
    phi_f = feature_dirs @ gram.T  # (n_features, d)

    alignment = torch.zeros(n_features, n_probes)
    for j, w in enumerate(probe_weights):
        # <f_i, w_j>_c = f_i^T Φ w_j
        phi_w = gram @ w  # (d,)
        dots = feature_dirs @ phi_w  # (n_features,)

        # Norms under causal IP
        f_norms = (feature_dirs * phi_f).sum(dim=1).clamp(min=0).sqrt()  # (n_features,)
        w_norm = (w @ phi_w).clamp(min=0).sqrt()

        denom = f_norms * w_norm
        cos = torch.where(denom > 1e-8, dots / denom, torch.zeros_like(dots))
        alignment[:, j] = cos.abs()

    return alignment


def compute_monosemanticity(
    sae: TopKSAE,
    activations: Optional[torch.Tensor],
) -> torch.Tensor:
    """Estimate monosemanticity score for each feature.

    Higher = feature fires more independently (less co-activation).
    Computed as 1 - mean(pairwise correlation of activation patterns).

    If no activations provided, returns ones (no filtering).
    """
    if activations is None:
        return torch.ones(sae.n_features)

    with torch.no_grad():
        z = sae.encode(activations)  # (n_samples, n_features)

    # Binary activation pattern
    active = (z > 0).float()  # (n_samples, n_features)

    # Per-feature activation frequency
    freq = active.mean(dim=0)  # (n_features,)

    # For features that fire, compute mean pairwise co-activation
    # This is expensive for large n_features, so sample
    n_feat = active.shape[1]
    if n_feat > 1000:
        # Sample 1000 features for correlation estimation
        idx = torch.randperm(n_feat)[:1000]
        active_sample = active[:, idx]
    else:
        active_sample = active

    # Correlation matrix of activation patterns
    # Normalise columns to zero mean
    centered = active_sample - active_sample.mean(dim=0, keepdim=True)
    norms = centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
    normed = centered / norms
    corr = (normed.T @ normed) / active_sample.shape[0]

    # Mean absolute off-diagonal correlation per feature
    n = corr.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_corr = corr.abs()[mask].reshape(n, n - 1).mean(dim=1)

    # Monosemanticity = 1 - mean_correlation
    mono = 1.0 - mean_corr

    # Map back to full feature set if sampled
    if n_feat > 1000:
        full_mono = torch.ones(n_feat) * 0.5  # default for unsampled
        full_mono[idx] = mono
        return full_mono

    return mono


def select_features(
    sae: TopKSAE,
    probe_weights: Optional[List[torch.Tensor]],
    probe_labels: Optional[List[str]],
    gram: Optional[torch.Tensor],
    d: int,
    activations: Optional[torch.Tensor],
    top_k: int,
    min_monosemanticity: float = 0.3,
) -> List[Dict]:
    """Select the most value-relevant SAE features.

    Selection criteria:
    1. If probes provided: rank by max causal alignment with any probe
    2. Filter by monosemanticity threshold
    3. Take top-k
    """
    feature_dirs = sae.feature_directions()  # (n_features, d)
    n_features = feature_dirs.shape[0]

    # Compute monosemanticity
    mono_scores = compute_monosemanticity(sae, activations)

    if probe_weights is not None and gram is not None:
        # Rank by causal alignment with probes
        alignment = compute_causal_alignment(feature_dirs, probe_weights, gram, d)
        # Max alignment across probes for each feature
        max_alignment, best_probe = alignment.max(dim=1)

        # Combined score: alignment * monosemanticity
        combined = max_alignment * mono_scores
    else:
        # No probes: rank by decoder norm (features with stronger directions)
        # weighted by monosemanticity
        dec_norms = feature_dirs.norm(dim=1)
        combined = dec_norms * mono_scores
        max_alignment = dec_norms
        best_probe = torch.zeros(n_features, dtype=torch.long)

    # Filter by monosemanticity
    valid_mask = mono_scores >= min_monosemanticity
    combined[~valid_mask] = -1.0

    # Top-k selection
    topk_vals, topk_idx = combined.topk(min(top_k, n_features))

    selected = []
    for rank, feat_idx in enumerate(topk_idx):
        feat_idx = feat_idx.item()
        if combined[feat_idx] < 0:
            break  # Below monosemanticity threshold

        direction = feature_dirs[feat_idx].tolist()

        # Determine label
        if probe_labels is not None and probe_weights is not None:
            best_p = best_probe[feat_idx].item()
            label = f"{probe_labels[best_p]}-sae-{feat_idx}"
            align_score = max_alignment[feat_idx].item()
        else:
            label = f"sae-feature-{feat_idx}"
            align_score = max_alignment[feat_idx].item()

        selected.append({
            "sae_index": feat_idx,
            "label": label,
            "weights": direction,
            "monosemanticity_score": mono_scores[feat_idx].item(),
            "alignment_score": align_score,
            "combined_score": combined[feat_idx].item(),
        })

    return selected


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def load_probes(path: Path) -> Tuple[List[torch.Tensor], List[str], int]:
    """Load probe weights from a ProbeSet JSON file."""
    with open(path) as f:
        data = json.load(f)

    weights = []
    labels = []
    layer = data.get("layer", 0)
    for p in data.get("probes", []):
        weights.append(torch.tensor(p["weights"], dtype=torch.float32))
        labels.append(p.get("dimension_name", "unknown"))

    return weights, labels, layer


def load_gram_from_gotue(path: Path) -> Tuple[torch.Tensor, int]:
    """Load unembedding matrix and compute Gram matrix Φ = U^T U."""
    data = open(path, "rb").read()
    if data[:4] != b"GOTU":
        raise ValueError(f"Not a .gotue file: {path}")

    offset = 4
    _version = struct.unpack_from("<H", data, offset)[0]; offset += 2
    vocab_size = struct.unpack_from("<I", data, offset)[0]; offset += 4
    hidden_dim = struct.unpack_from("<I", data, offset)[0]; offset += 4

    n_floats = vocab_size * hidden_dim
    values = struct.unpack_from(f"<{n_floats}f", data, offset)
    U = torch.tensor(values, dtype=torch.float32).reshape(vocab_size, hidden_dim)

    # Φ = U^T U
    gram = U.T @ U
    return gram, hidden_dim


def features_to_probeset(
    features: List[Dict],
    layer: int,
    sae_source: str,
) -> Dict:
    """Convert selected SAE features to ProbeSet JSON format."""
    probes = []
    for feat in features:
        probes.append({
            "dimension_name": feat["label"],
            "weights": feat["weights"],
            "bias": 0.0,
            "platt_scale": 1.0,
            "platt_shift": 0.0,
            "reliability_threshold": 0.5,
        })

    return {
        "probes": probes,
        "version": "sae-v1",
        "corpus_version": sae_source,
        "layer": layer,
        "geometry_hash": None,
        "max_drift": None,
        "max_directional_drift": None,
        # SAE-specific metadata
        "sae_metadata": {
            "source": sae_source,
            "features": [
                {
                    "label": f["label"],
                    "sae_index": f["sae_index"],
                    "monosemanticity_score": f["monosemanticity_score"],
                    "alignment_score": f["alignment_score"],
                }
                for f in features
            ],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract value-relevant SAE features as ProbeSet JSON"
    )
    parser.add_argument("--sae", required=True, type=Path, help="SAE checkpoint (.pt)")
    parser.add_argument("--probes", type=Path, default=None,
                        help="Existing ProbeSet JSON (for alignment-based selection)")
    parser.add_argument("--unembedding", type=Path, default=None,
                        help=".gotue file (for causal inner product alignment)")
    parser.add_argument("--activations", type=Path, default=None,
                        help="Activations tensor (.pt) for monosemanticity estimation")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of features to select")
    parser.add_argument("--min-mono", type=float, default=0.3,
                        help="Minimum monosemanticity score")
    parser.add_argument("--output", required=True, type=Path, help="Output ProbeSet JSON path")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load SAE
    print(f"Loading SAE from {args.sae}...")
    sae = TopKSAE.load(args.sae)
    sae_meta = getattr(sae, "metadata", {})
    print(f"  {sae.n_features} features, k={sae.k}, d={sae.d_model}")
    print(f"  Source: {sae_meta.get('model', 'unknown')}, layer {sae_meta.get('layer', '?')}")

    # Load probes if provided
    probe_weights = None
    probe_labels = None
    layer = sae_meta.get("layer", 0)
    if args.probes:
        print(f"Loading probes from {args.probes}...")
        probe_weights, probe_labels, layer = load_probes(args.probes)
        print(f"  {len(probe_weights)} probes: {probe_labels}")

    # Load Gram matrix if unembedding provided
    gram = None
    d = sae.d_model
    if args.unembedding:
        print(f"Loading geometry from {args.unembedding}...")
        gram, d = load_gram_from_gotue(args.unembedding)
        print(f"  Gram matrix: {d}x{d}")

    # Load activations for monosemanticity if provided
    activations = None
    if args.activations and args.activations.exists():
        print(f"Loading activations from {args.activations}...")
        activations = torch.load(args.activations, weights_only=True)
        # Normalise using SAE's stored params if available
        if hasattr(sae, "act_mean"):
            activations = (activations - sae.act_mean) / sae.act_std.clamp(min=1e-6)
        print(f"  {activations.shape[0]} samples")

    # Select features
    print(f"Selecting top-{args.top_k} features (min monosemanticity={args.min_mono})...")
    features = select_features(
        sae,
        probe_weights=probe_weights,
        probe_labels=probe_labels,
        gram=gram,
        d=d,
        activations=activations,
        top_k=args.top_k,
        min_monosemanticity=args.min_mono,
    )
    print(f"  Selected {len(features)} features:")
    for f in features:
        print(f"    {f['label']}: mono={f['monosemanticity_score']:.3f}, "
              f"align={f['alignment_score']:.3f}, combined={f['combined_score']:.3f}")

    if not features:
        print("ERROR: No features passed selection criteria.", file=sys.stderr)
        print("Try lowering --min-mono or increasing --top-k.", file=sys.stderr)
        sys.exit(1)

    # Export as ProbeSet
    sae_source = f"{sae_meta.get('model', 'unknown')}-sae-layer{layer}"
    probeset = features_to_probeset(features, layer, sae_source)

    with open(args.output, "w") as f:
        json.dump(probeset, f, indent=2)
    print(f"Saved ProbeSet to {args.output}")

    # Summary
    print()
    print("Next steps:")
    print(f"  # Collapse report:")
    print(f"  cargo run --release -p got-cli -- collapse-report \\")
    print(f"      --unembedding <model>.gotue --probes {args.output}")
    print(f"  # Coherence (if you have value ordering constraints):")
    print(f"  cargo run --release -p got-cli -- coherence \\")
    print(f"      --activations <model>.gotact --unembedding <model>.gotue \\")
    print(f"      --ordering value_ordering.json --layer {layer}")


if __name__ == "__main__":
    main()
