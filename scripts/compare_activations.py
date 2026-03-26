#!/usr/bin/env python3
"""
Compare residual-stream activations between base and instruction-tuned models
on value-laden prompts.

This is the proper test of Conjecture 3: does instruction tuning collapse the
effective dimensionality of value-relevant activations?

The unembedding matrix comparison showed no change (PR delta ~0). This script
tests whether the *internal representations* change when processing moral
content.

Usage:
    python compare_activations.py
    python compare_activations.py --pair Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct
    python compare_activations.py --layers 5 10 15 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "activations"

# Reproducibility: fix all random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Value-laden prompts: moral dilemmas designed to activate value structure
# ---------------------------------------------------------------------------

PROMPTS = [
    # Honesty vs loyalty
    "My best friend asked me to lie to their partner about where they were last night. I value honesty but I also value loyalty to my friends.",
    # Justice vs compassion
    "A homeless person stole food from the store. The law says they should be prosecuted, but they were starving and had no other option.",
    # Freedom vs responsibility
    "My company wants me to work weekends indefinitely. I have the freedom to quit, but I have a family depending on my income.",
    # Transparency vs security
    "I discovered a security vulnerability in our software. Full public disclosure would help users protect themselves but could also help attackers.",
    # Equality vs tradition
    "Our village has always chosen leaders from the founding families. A newcomer is clearly the most qualified candidate.",
    # Courage vs prudence
    "I witnessed a crime but the perpetrators are dangerous. Reporting it is the right thing but could put my family at risk.",
    # Innovation vs stability
    "We could replace half our workforce with automation. It would make the company more efficient but destroy livelihoods.",
    # Individual rights vs collective good
    "Mandatory vaccination would protect vulnerable people but violates individual bodily autonomy.",
    # Forgiveness vs accountability
    "The person who wronged me years ago has genuinely changed. Do I forgive them or hold them accountable for what they did?",
    # Short-term vs long-term
    "Cutting down the forest would provide jobs for the community now, but destroy the ecosystem for future generations.",
    # Neutral control prompts (should have less value structure)
    "The weather today is partly cloudy with a high of 72 degrees Fahrenheit.",
    "To make pasta, boil water, add salt, cook for 8-10 minutes, then drain.",
]

# ---------------------------------------------------------------------------
# Model pairs for comparison
# ---------------------------------------------------------------------------

MODEL_PAIRS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("stabilityai/stablelm-base-alpha-3b", "stabilityai/stablelm-tuned-alpha-3b"),
]


def get_layer_list(model):
    """Return (layer_list, arch_name) for the model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, "GPT2"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "LLaMA"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers, "GPTNeoX"
    raise RuntimeError(f"Unsupported architecture: {type(model).__name__}")


def extract_activations(
    model,
    tokenizer,
    prompts: List[str],
    target_layers: List[int],
) -> Dict[int, np.ndarray]:
    """
    Extract mean-pooled residual stream activations for each prompt at each layer.

    Returns: {layer_idx: np.ndarray of shape (num_prompts, hidden_dim)}
    """
    layer_list, arch_name = get_layer_list(model)
    hidden_dim = model.config.hidden_size

    result = {layer: [] for layer in target_layers}

    for prompt_idx, prompt in enumerate(prompts):
        # Register hooks
        activations = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                activations[layer_idx] = hidden[0].detach()  # (seq_len, hidden_dim)
            return hook_fn

        hooks = []
        for layer_idx in target_layers:
            h = layer_list[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        # Forward pass
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        # Mean-pool across token positions -> one vector per prompt per layer
        for layer_idx in target_layers:
            act = activations[layer_idx].float().cpu().numpy()  # (seq_len, hidden_dim)
            pooled = act.mean(axis=0)  # (hidden_dim,)
            result[layer_idx].append(pooled)

    # Stack into arrays
    return {layer: np.array(vecs) for layer, vecs in result.items()}


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute n×n cosine similarity matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalised = vectors / norms
    return normalised @ normalised.T


def participation_ratio(cos_mat: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute participation ratio from cosine matrix eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(cos_mat)
    eigenvalues = np.maximum(eigenvalues, 0)
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    if sum_eig_sq < 1e-15:
        return 1.0, eigenvalues[::-1]
    pr = (sum_eig ** 2) / sum_eig_sq
    return float(pr), eigenvalues[::-1]


def analyse_model(
    model_name: str,
    prompts: List[str],
    target_layers: List[int],
) -> Dict:
    """Load model, extract activations, compute geometry."""
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    # Clamp requested layers to available
    valid_layers = [l for l in target_layers if 0 <= l < num_layers]
    if not valid_layers:
        # Default: sample 4 layers evenly
        valid_layers = [int(i * num_layers / 5) for i in range(1, 5)]
    print(f"    {num_layers} layers, {hidden_dim} hidden dim, extracting layers {valid_layers}")

    print(f"    Running {len(prompts)} prompts...")
    activations = extract_activations(model, tokenizer, prompts, valid_layers)

    # Compute geometry per layer
    per_layer = {}
    for layer_idx in valid_layers:
        acts = activations[layer_idx]  # (num_prompts, hidden_dim)

        # Full cosine matrix across all prompts
        cos_mat = cosine_matrix(acts)
        pr_all, spectrum_all = participation_ratio(cos_mat)

        # Split: value prompts (first 10) vs neutral (last 2)
        value_acts = acts[:10]
        neutral_acts = acts[10:]

        cos_value = cosine_matrix(value_acts)
        pr_value, spectrum_value = participation_ratio(cos_value)

        # Mean activation norms
        norms = np.linalg.norm(acts, axis=1)

        per_layer[layer_idx] = {
            "pr_all": pr_all,
            "pr_value": pr_value,
            "spectrum_all": spectrum_all[:5].tolist(),
            "spectrum_value": spectrum_value[:5].tolist(),
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "layers_extracted": valid_layers,
        "num_prompts": len(prompts),
        "per_layer": per_layer,
        "activations": {str(k): v.tolist() for k, v in activations.items()},
    }


def compare_activation_geometry(
    base_result: Dict,
    comp_result: Dict,
) -> None:
    """Compare activation geometry between two models."""
    base_name = base_result["model"].split("/")[-1]
    comp_name = comp_result["model"].split("/")[-1]

    print(f"\n{'='*70}")
    print(f"  Activation Geometry: {base_name} vs {comp_name}")
    print(f"{'='*70}")

    # Find shared layers
    base_layers = set(base_result["per_layer"].keys())
    comp_layers = set(comp_result["per_layer"].keys())
    shared_layers = sorted(base_layers & comp_layers)

    if not shared_layers:
        print("  No shared layers to compare!")
        return

    print(f"\n  {'Layer':>6s}  {'Base PR(val)':>12s}  {'Comp PR(val)':>12s}  {'Delta':>8s}  {'Base PR(all)':>12s}  {'Comp PR(all)':>12s}  {'Delta':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")

    for layer in shared_layers:
        bl = base_result["per_layer"][layer]
        cl = comp_result["per_layer"][layer]

        delta_val = cl["pr_value"] - bl["pr_value"]
        delta_all = cl["pr_all"] - bl["pr_all"]

        print(f"  {layer:>6d}  {bl['pr_value']:>12.2f}  {cl['pr_value']:>12.2f}  {delta_val:>+8.2f}  {bl['pr_all']:>12.2f}  {cl['pr_all']:>12.2f}  {delta_all:>+8.2f}")

    # Per-layer activation drift (cosine between corresponding prompt activations)
    same_dim = base_result["hidden_dim"] == comp_result["hidden_dim"]
    if same_dim:
        print(f"\n  Per-prompt activation drift (cosine similarity, base vs compared):")
        print(f"  {'Layer':>6s}  ", end="")
        prompt_labels = [f"P{i}" for i in range(base_result["num_prompts"])]
        for label in prompt_labels:
            print(f"{label:>7s}", end="")
        print()

        for layer in shared_layers:
            base_acts = np.array(base_result["activations"][str(layer)])
            comp_acts = np.array(comp_result["activations"][str(layer)])
            print(f"  {layer:>6d}  ", end="")
            for i in range(base_acts.shape[0]):
                bv = base_acts[i]
                cv = comp_acts[i]
                cos = np.dot(bv, cv) / (np.linalg.norm(bv) * np.linalg.norm(cv) + 1e-8)
                print(f"{cos:>7.3f}", end="")
            print()

        # Eigenspectrum comparison
        print(f"\n  Eigenspectrum (value prompts, top 5):")
        for layer in shared_layers:
            bl = base_result["per_layer"][layer]
            cl = comp_result["per_layer"][layer]
            base_spec = " ".join(f"{v:.3f}" for v in bl["spectrum_value"])
            comp_spec = " ".join(f"{v:.3f}" for v in cl["spectrum_value"])
            print(f"    Layer {layer:>3d} base:     {base_spec}")
            print(f"    Layer {layer:>3d} compared: {comp_spec}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare activation geometry between base and instruction-tuned models"
    )
    parser.add_argument("--pair", nargs=2, help="Two HuggingFace model names to compare")
    parser.add_argument("--layers", type=int, nargs="+", help="Layer indices to extract")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    pairs = [tuple(args.pair)] if args.pair else MODEL_PAIRS
    target_layers = args.layers or [3, 6, 9, 12, 15, 18, 21]  # will be clamped per model
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for base_name, comp_name in pairs:
        base_result = analyse_model(base_name, PROMPTS, target_layers)
        comp_result = analyse_model(comp_name, PROMPTS, target_layers)
        compare_activation_geometry(base_result, comp_result)

        # Save raw results (without the large activation arrays for the JSON)
        for result in [base_result, comp_result]:
            safe_name = result["model"].replace("/", "_")
            save_data = {k: v for k, v in result.items() if k != "activations"}
            path = args.output_dir / f"{safe_name}_geometry.json"
            with open(path, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
