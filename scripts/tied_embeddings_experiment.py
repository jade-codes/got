#!/usr/bin/env python3
"""
Tied Embeddings Experiment

Tests whether tied embedding matrices contaminate the causal inner product
signal by comparing GPT-2 (tied embeddings) against Mistral-7B (untied).

If tied embeddings degrade causal consistency, GPT-2 will show systematically
lower scores. If both models are comparable, tied embeddings are not a
significant confounder.

Usage:
    python scripts/tied_embeddings_experiment.py \
        --tied gpt2 \
        --untied mistralai/Mistral-7B-v0.1 \
        --output-dir data/tied_experiment

    # With smaller models for faster testing:
    python scripts/tied_embeddings_experiment.py \
        --tied gpt2 \
        --untied EleutherAI/pythia-410m \
        --layers 4 8 \
        --output-dir data/tied_experiment_small

Dependencies:
    pip install torch transformers
    cargo build --release -p got-cli
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Value-relevant prompts (50 total, 10 per dimension)
# ---------------------------------------------------------------------------

PROMPTS = {
    "honesty": [
        "Being truthful is important even when the truth is uncomfortable.",
        "A leader who deceives their followers cannot be trusted.",
        "Transparency in government builds public confidence.",
        "Sometimes a white lie protects someone from unnecessary pain.",
        "Whistleblowers serve democracy by exposing hidden corruption.",
        "Withholding information is not the same as outright lying.",
        "Trust is built on consistent honesty over time.",
        "Deception in advertising should be strictly regulated.",
        "Children should learn that honesty is always the best policy.",
        "Diplomatic communication requires careful truth management.",
    ],
    "helpfulness": [
        "Helping others in need is a fundamental human obligation.",
        "Effective charity requires understanding what people actually need.",
        "Sometimes the most helpful thing is to step back and let someone learn.",
        "Technology should be designed to serve human wellbeing.",
        "Volunteerism strengthens communities and builds social bonds.",
        "Enabling dependency through help can cause long-term harm.",
        "Emergency responders exemplify selfless service to others.",
        "Foreign aid must respect local autonomy and knowledge.",
        "Teachers who go beyond the curriculum truly help students grow.",
        "Helping should never come with strings attached or expectations.",
    ],
    "fairness": [
        "Equal treatment under the law is a cornerstone of justice.",
        "Systemic bias must be actively addressed, not just acknowledged.",
        "Meritocracy only works when everyone starts from equal footing.",
        "Fair algorithms require diverse training data and ongoing audits.",
        "Progressive taxation helps create a more equitable society.",
        "Affirmative action addresses historical injustices in access.",
        "Criminal sentencing should be consistent regardless of background.",
        "Intellectual property law must balance innovation and access.",
        "Fair negotiations require honest disclosure from all parties.",
        "Equity sometimes requires treating people differently to achieve fairness.",
    ],
    "autonomy": [
        "Individuals should have the right to make their own life choices.",
        "Government surveillance threatens personal freedom and privacy.",
        "Informed consent is essential before any medical procedure.",
        "Social media algorithms that manipulate behavior undermine autonomy.",
        "Parents must gradually increase children's decision-making freedom.",
        "Mandatory policies sometimes override individual choice for collective good.",
        "Bodily autonomy is a fundamental human right that must be protected.",
        "Workplace policies should respect employee independence and judgment.",
        "Nudge policies can improve outcomes while preserving choice.",
        "True freedom requires access to accurate information and education.",
    ],
    "safety": [
        "Protecting vulnerable populations requires proactive safety measures.",
        "Technology companies bear responsibility for harm caused by their products.",
        "Safety regulations exist because markets alone cannot prevent all harm.",
        "Precaution is warranted when potential consequences are severe and irreversible.",
        "Workplace safety standards have saved millions of lives over decades.",
        "AI systems must be tested for safety before deployment at scale.",
        "Public health measures sometimes require temporary restrictions on freedom.",
        "Building codes save lives even when they increase construction costs.",
        "Safety culture means every person feels empowered to raise concerns.",
        "Risk assessment must consider worst-case scenarios, not just averages.",
    ],
}

# ---------------------------------------------------------------------------
# Binary format writers
# ---------------------------------------------------------------------------

def write_gotue(path: Path, weight: torch.Tensor) -> None:
    """Write unembedding matrix in .gotue binary format."""
    w = weight.detach().float().cpu()
    vocab_size, hidden_dim = w.shape
    with open(path, "wb") as f:
        f.write(b"GOTU")
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_dim))
        for row in range(vocab_size):
            for col in range(hidden_dim):
                f.write(struct.pack("<f", w[row, col].item()))
    print(f"  Wrote {path} ({vocab_size} x {hidden_dim})")


def write_gotact(
    path: Path,
    model_id: str,
    hidden_dim: int,
    activations: Dict[int, List[torch.Tensor]],
) -> None:
    """Write activations in .gotact binary format."""
    layers_sorted = sorted(activations.keys())
    num_layers = len(layers_sorted)
    num_positions = len(activations[layers_sorted[0]]) if num_layers > 0 else 0

    with open(path, "wb") as f:
        f.write(b"GOTA")
        f.write(struct.pack("<H", 1))
        model_bytes = model_id.encode("utf-8")
        f.write(struct.pack("<I", len(model_bytes)))
        f.write(model_bytes)
        f.write(struct.pack("<B", 0))  # fp32
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", num_layers))
        f.write(struct.pack("<I", num_positions))

        for layer_idx in layers_sorted:
            for pos, act in enumerate(activations[layer_idx]):
                f.write(struct.pack("<I", layer_idx))
                f.write(struct.pack("<I", pos))
                vals = act.detach().float().cpu()
                for v in vals:
                    f.write(struct.pack("<f", v.item()))

    print(f"  Wrote {path} ({num_layers} layers x {num_positions} positions)")


def write_labels(path: Path, labels: List[int]) -> None:
    """Write labels file."""
    with open(path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")


# ---------------------------------------------------------------------------
# Model loading and extraction
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str = "auto", dtype: str = "float32"):
    """Load a HuggingFace causal LM."""
    print(f"Loading {model_name}...")
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True,
        )
    except Exception:
        print(f"  Warning: device_map={device} failed, falling back to CPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        )

    model.eval()
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {hidden}d, {n_layers} layers")
    return model, tokenizer


def check_tied_embeddings(model) -> bool:
    """Check if the model uses tied embeddings (input embed == output embed)."""
    try:
        input_emb = model.get_input_embeddings().weight.data_ptr()
        output_emb = model.get_output_embeddings().weight.data_ptr()
        return input_emb == output_emb
    except Exception:
        return False


def detect_layers(model) -> list:
    """Detect transformer layer modules."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown architecture: {type(model).__name__}")


def extract_activations(
    model, tokenizer, prompts: List[str], target_layers: List[int],
) -> Dict[int, List[torch.Tensor]]:
    """Extract mean-pooled residual stream activations per prompt per layer."""
    all_layers = detect_layers(model)
    activations: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0].mean(dim=0).detach().cpu()
        return hook_fn

    hooks = []
    for layer_idx in target_layers:
        if layer_idx < len(all_layers):
            hooks.append(all_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        for layer_idx in target_layers:
            if layer_idx in captured:
                activations[layer_idx].append(captured[layer_idx].clone())
        captured.clear()

    for h in hooks:
        h.remove()
    return activations


def get_unembedding(model) -> torch.Tensor:
    """Extract unembedding matrix."""
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().cpu()
    elif hasattr(model, "embed_out"):
        return model.embed_out.weight.detach().cpu()
    raise ValueError("Cannot find unembedding matrix")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def run_cli(args: List[str], description: str = "") -> subprocess.CompletedProcess:
    """Run got-cli command."""
    cmd = ["cargo", "run", "--release", "-p", "got-cli", "--"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and description:
        print(f"  ERROR in {description}: {result.stderr[:200]}")
    return result


def parse_attest_output(text: str) -> Optional[Dict]:
    """Extract causal consistency info from attestation output."""
    result = {}
    # Look for causal scores in attestation JSON output
    for m in re.finditer(r'"causal_score":\s*([\d.e+-]+)', text):
        result.setdefault("causal_scores", []).append(float(m.group(1)))
    for m in re.finditer(r'"confidence":\s*([\d.e+-]+)', text):
        result.setdefault("confidences", []).append(float(m.group(1)))
    return result if result else None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tied Embeddings Experiment")
    parser.add_argument("--tied", required=True, help="Tied-embeddings model (e.g. gpt2)")
    parser.add_argument("--untied", required=True, help="Untied-embeddings model (e.g. mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layer indices (default: auto-select 4 evenly spaced)")
    parser.add_argument("--output-dir", default="data/tied_experiment")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Flatten prompts
    all_prompts = []
    dimensions = list(PROMPTS.keys())
    for dim in dimensions:
        all_prompts.extend(PROMPTS[dim])

    print(f"Tied Embeddings Experiment")
    print(f"  Tied model:   {args.tied}")
    print(f"  Untied model: {args.untied}")
    print(f"  {len(all_prompts)} prompts across {len(dimensions)} dimensions")
    print()

    models_info = {}

    # -----------------------------------------------------------------------
    # Step 1: Extract for both models
    # -----------------------------------------------------------------------

    for tag, model_name in [("tied", args.tied), ("untied", args.untied)]:
        if not args.skip_extraction:
            print(f"=== Extracting {tag}: {model_name} ===")
            model, tokenizer = load_model(model_name, args.device, args.dtype)

            is_tied = check_tied_embeddings(model)
            n_layers = model.config.num_hidden_layers
            hidden_dim = model.config.hidden_size
            models_info[tag] = {
                "name": model_name,
                "tied": is_tied,
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
            }
            print(f"  Tied embeddings: {is_tied}")

            # Auto-select layers if not specified
            if args.layers is None:
                target_layers = [
                    n_layers // 4,
                    n_layers // 2,
                    3 * n_layers // 4,
                    n_layers - 1,
                ]
            else:
                target_layers = [l for l in args.layers if l < n_layers]

            models_info[tag]["layers"] = target_layers

            ue = get_unembedding(model)
            write_gotue(out / f"{tag}.gotue", ue)

            print(f"  Extracting activations at layers {target_layers}...")
            acts = extract_activations(model, tokenizer, all_prompts, target_layers)
            write_gotact(out / f"{tag}.gotact", model_name, hidden_dim, acts)

            del model, tokenizer, ue, acts
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print()

    # Save model info
    info_path = out / "models_info.json"
    with open(info_path, "w") as f:
        json.dump(models_info, f, indent=2)

    # Reload info if skipping extraction
    if args.skip_extraction and info_path.exists():
        with open(info_path) as f:
            models_info = json.load(f)

    # -----------------------------------------------------------------------
    # Step 2: Train probes on each model independently
    # -----------------------------------------------------------------------

    print("=== Training probes ===")
    for tag in ["tied", "untied"]:
        info = models_info.get(tag, {})
        target_layers = info.get("layers", args.layers or [])

        for layer in target_layers:
            for dim_idx, dim in enumerate(dimensions):
                # Generate labels: positive for this dimension's prompts
                all_labels = [0] * len(all_prompts)
                start = dim_idx * 10  # 10 prompts per dimension
                for i in range(5):  # first 5 are "positive"
                    all_labels[start + i] = 1

                labels_path = out / f"labels_{tag}_{dim}.txt"
                write_labels(labels_path, all_labels)

                probe_out = out / f"probes_{tag}_{dim}_layer{layer}.json"
                run_cli([
                    "train",
                    "--activations", str(out / f"{tag}.gotact"),
                    "--labels", str(labels_path),
                    "--unembedding", str(out / f"{tag}.gotue"),
                    "--layer", str(layer),
                    "--dimension", dim,
                    "--output", str(probe_out),
                ], f"train {tag} {dim} layer {layer}")
    print()

    # -----------------------------------------------------------------------
    # Step 3: Produce attestations (with causal checks where possible)
    # -----------------------------------------------------------------------

    print("=== Producing attestations ===")
    results = {"tied": {}, "untied": {}}

    for tag in ["tied", "untied"]:
        info = models_info.get(tag, {})
        target_layers = info.get("layers", args.layers or [])

        # Generate a signing key for this experiment
        key_path = out / f"key_{tag}"
        if not key_path.exists():
            run_cli(["keygen", "--output", str(key_path)], f"keygen {tag}")

        for layer in target_layers:
            probe_files = []
            for dim in dimensions:
                p = out / f"probes_{tag}_{dim}_layer{layer}.json"
                if p.exists():
                    probe_files.append(str(p))

            if not probe_files:
                continue

            att_out = out / f"attestation_{tag}_layer{layer}.json"
            result = run_cli([
                "attest",
                "--activations", str(out / f"{tag}.gotact"),
                "--probes", *probe_files,
                "--unembedding", str(out / f"{tag}.gotue"),
                "--key", str(key_path),
                "--model-id", info.get("name", tag),
                "--output", str(att_out),
            ], f"attest {tag} layer {layer}")

            if att_out.exists():
                with open(att_out) as f:
                    att_data = json.load(f)

                # Extract readings
                readings = att_data.get("readings", [])
                causal_scores = []
                confidences = []
                for r in readings:
                    if "confidence" in r:
                        confidences.append(r["confidence"])
                    if "causal_score" in r:
                        causal_scores.append(r["causal_score"])

                results[tag][f"layer{layer}"] = {
                    "readings": len(readings),
                    "confidences": confidences,
                    "causal_scores": causal_scores,
                    "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
                    "mean_causal": sum(causal_scores) / len(causal_scores) if causal_scores else None,
                }
    print()

    # -----------------------------------------------------------------------
    # Step 4: Collapse report comparison
    # -----------------------------------------------------------------------

    print("=== Collapse reports ===")
    for tag in ["tied", "untied"]:
        info = models_info.get(tag, {})
        target_layers = info.get("layers", args.layers or [])

        for layer in target_layers:
            for dim in dimensions:
                probe_path = out / f"probes_{tag}_{dim}_layer{layer}.json"
                if not probe_path.exists():
                    continue
                result = run_cli([
                    "collapse-report",
                    "--unembedding", str(out / f"{tag}.gotue"),
                    "--probes", str(probe_path),
                ], f"collapse {tag} {dim} layer {layer}")
                if result.returncode == 0:
                    # Parse dim_eff
                    m = re.search(r"dim_eff:\s+([\d.]+)", result.stdout)
                    if m:
                        key = f"dim_eff_{dim}_layer{layer}"
                        results[tag][key] = float(m.group(1))
    print()

    # -----------------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------------

    print("=" * 50)
    print("Tied Embeddings Experiment")
    print("=" * 50)

    for tag in ["tied", "untied"]:
        info = models_info.get(tag, {})
        tied_str = "TIED" if info.get("tied") else "UNTIED"
        print(f"  {tag}: {info.get('name', '?')} ({tied_str}, {info.get('hidden_dim', '?')}d, {info.get('n_layers', '?')} layers)")
    print()

    # Causal consistency comparison
    print("Causal Consistency (from attestation readings):")
    print("-" * 50)
    for tag in ["tied", "untied"]:
        for layer_key, data in sorted(results[tag].items()):
            if isinstance(data, dict) and "mean_confidence" in data:
                mc = data["mean_confidence"]
                mcs = data.get("mean_causal")
                conf_str = f"mean_confidence={mc:.4f}" if mc is not None else "no confidence data"
                causal_str = f", mean_causal={mcs:.4f}" if mcs is not None else ""
                print(f"  {tag} {layer_key}: {conf_str}{causal_str} ({data['readings']} readings)")
    print()

    # Per-dimension breakdown
    print("Per-dimension breakdown:")
    print("-" * 50)
    for dim in dimensions:
        tied_scores = []
        untied_scores = []
        for layer_key, data in results.get("tied", {}).items():
            if isinstance(data, dict) and "confidences" in data:
                tied_scores.extend(data["confidences"])
        for layer_key, data in results.get("untied", {}).items():
            if isinstance(data, dict) and "confidences" in data:
                untied_scores.extend(data["confidences"])

        tied_mean = sum(tied_scores) / len(tied_scores) if tied_scores else float("nan")
        untied_mean = sum(untied_scores) / len(untied_scores) if untied_scores else float("nan")
        print(f"  {dim:15s}  tied={tied_mean:.4f}  untied={untied_mean:.4f}")
    print()

    # dim_eff comparison
    print("Effective Value Dimensionality (dim_eff):")
    print("-" * 50)
    for dim in dimensions:
        for tag in ["tied", "untied"]:
            for key, val in sorted(results[tag].items()):
                if key.startswith(f"dim_eff_{dim}_"):
                    print(f"  {tag:8s} {key}: {val:.3f}")
    print()

    # Conclusion
    tied_confs = []
    untied_confs = []
    for data in results.get("tied", {}).values():
        if isinstance(data, dict) and "confidences" in data:
            tied_confs.extend(data["confidences"])
    for data in results.get("untied", {}).values():
        if isinstance(data, dict) and "confidences" in data:
            untied_confs.extend(data["confidences"])

    print("Conclusion:")
    print("-" * 50)
    if tied_confs and untied_confs:
        tied_mean = sum(tied_confs) / len(tied_confs)
        untied_mean = sum(untied_confs) / len(untied_confs)
        diff = abs(tied_mean - untied_mean)
        tied_std = (sum((x - tied_mean) ** 2 for x in tied_confs) / len(tied_confs)) ** 0.5
        untied_std = (sum((x - untied_mean) ** 2 for x in untied_confs) / len(untied_confs)) ** 0.5

        print(f"  Tied:   mean={tied_mean:.4f}, std={tied_std:.4f} (n={len(tied_confs)})")
        print(f"  Untied: mean={untied_mean:.4f}, std={untied_std:.4f} (n={len(untied_confs)})")
        print(f"  Difference: {diff:.4f}")
        print()

        # Simple threshold: if difference < 0.1, not significant
        if diff < 0.1:
            print("  Tied embeddings do NOT significantly affect causal consistency.")
            print("  The difference between tied and untied models is within noise.")
        elif tied_mean < untied_mean:
            print("  Tied embeddings MAY affect causal consistency.")
            print(f"  The tied model shows lower scores by {diff:.4f}.")
            print("  Further investigation with more models is recommended.")
        else:
            print("  The tied model shows HIGHER causal consistency than untied.")
            print("  Tied embeddings do not appear to be a confounder.")
    else:
        print("  Insufficient data to draw conclusions.")
        print("  Check that extraction and attestation succeeded for both models.")
    print()

    # Save results
    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
