#!/usr/bin/env python3
"""
Extract unembedding matrices and vocab from multiple models for comparison.

Produces per-model:
  data/models/{name}.gotue   — unembedding matrix (V × d, f32 LE)
  data/models/{name}-vocab.json — vocabulary tokens

Usage:
    python extract_models.py                    # extract all default models
    python extract_models.py --models gpt2      # extract specific model(s)
    python extract_models.py --list             # list available models
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Models to extract — base/instruct pairs for Conjecture 3 testing
# ---------------------------------------------------------------------------

MODELS = {
    # Already extracted, but re-extract for consistency
    "gpt2": {
        "hf_name": "gpt2",
        "desc": "GPT-2 124M (baseline)",
    },
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "desc": "GPT-2 355M (dimensionality scaling)",
    },
    # Base vs instruct pair — small enough for CPU
    "qwen2.5-0.5b": {
        "hf_name": "Qwen/Qwen2.5-0.5B",
        "desc": "Qwen2.5 0.5B base",
    },
    "qwen2.5-0.5b-instruct": {
        "hf_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "desc": "Qwen2.5 0.5B instruct-tuned",
    },
    # Base vs chat pair
    "tinyllama-base": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "desc": "TinyLlama 1.1B base",
    },
    "tinyllama-chat": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "desc": "TinyLlama 1.1B chat (SFT + DPO)",
    },
    # True RLHF pair — StableLM base vs PPO-RLHF tuned
    "stablelm-base": {
        "hf_name": "stabilityai/stablelm-base-alpha-3b",
        "desc": "StableLM 3B base (no alignment)",
        "fp16": True,
    },
    "stablelm-tuned": {
        "hf_name": "stabilityai/stablelm-tuned-alpha-3b",
        "desc": "StableLM 3B tuned (PPO RLHF on Anthropic HH)",
        "fp16": True,
    },
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "models"


def write_gotue(path: Path, vocab_size: int, hidden_dim: int, data: torch.Tensor) -> None:
    """Write unembedding matrix to .gotue binary format."""
    assert data.shape == (vocab_size, hidden_dim)
    flat = data.float().cpu().contiguous()
    with open(path, "wb") as f:
        f.write(b"GOTU")
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack(f"<{vocab_size * hidden_dim}f", *flat.view(-1).tolist()))


def extract_model(name: str, info: dict, output_dir: Path) -> dict:
    """Extract unembedding matrix and vocab from a single model."""
    hf_name = info["hf_name"]
    print(f"\n{'='*60}")
    print(f"Extracting: {name} ({hf_name})")
    print(f"  {info['desc']}")
    print(f"{'='*60}")

    # Load tokenizer
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)

    # Load model — use float16 for large models to fit in RAM/disk
    use_fp16 = info.get("fp16", False)
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"  Loading model (CPU, {dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    hidden_dim = config.hidden_size
    vocab_size = config.vocab_size

    # Extract unembedding matrix
    if hasattr(model, "lm_head"):
        ue_weight = model.lm_head.weight.detach()
    elif hasattr(model, "embed_out"):
        ue_weight = model.embed_out.weight.detach()
    else:
        print(f"  ERROR: cannot find unembedding matrix", file=sys.stderr)
        return None

    actual_vocab = ue_weight.shape[0]
    print(f"  Vocab: {actual_vocab} | Hidden dim: {hidden_dim}")

    # Write .gotue
    gotue_path = output_dir / f"{name}.gotue"
    write_gotue(gotue_path, actual_vocab, hidden_dim, ue_weight)
    print(f"  Written: {gotue_path} ({gotue_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Write vocab
    vocab_path = output_dir / f"{name}-vocab.json"
    # Get all tokens in order
    vocab_tokens = []
    for i in range(actual_vocab):
        try:
            token = tokenizer.decode([i])
        except Exception:
            token = f"<unk_{i}>"
        vocab_tokens.append(token)

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_tokens, f, ensure_ascii=False)
    print(f"  Written: {vocab_path}")

    # Compute basic stats on the 28 value terms
    value_terms = [
        "honesty", "integrity", "fairness", "transparency", "accountability",
        "justice", "freedom", "equality", "equity", "compassion",
        "empathy", "courage", "bravery", "wisdom", "humility",
        "loyalty", "responsibility", "resilience", "openness", "creativity",
        "innovation", "efficiency", "tradition", "cruelty", "oppression",
        "secrecy", "truthfulness", "cowardice",
    ]

    term_results = {}
    resolved_count = 0
    for term in value_terms:
        # Try to find the term in vocab (case-insensitive, with/without BPE prefix)
        candidates = [term, term.capitalize(), f" {term}", f"Ġ{term}", f" {term.capitalize()}"]
        found = False
        for candidate in candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                idx = ids[0]
                vec = ue_weight[idx]
                norm = vec.norm().item()
                term_results[term] = {
                    "found": True,
                    "token_index": idx,
                    "token_string": tokenizer.decode([idx]),
                    "norm": round(norm, 4),
                }
                resolved_count += 1
                found = True
                break
        if not found:
            term_results[term] = {"found": False}

    # Write term analysis
    term_path = output_dir / f"{name}-term-analysis.json"
    with open(term_path, "w") as f:
        json.dump(term_results, f, indent=2)
    print(f"  Terms resolved: {resolved_count}/{len(value_terms)}")
    print(f"  Written: {term_path}")

    # Free memory
    del model, ue_weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "hf_name": hf_name,
        "vocab_size": actual_vocab,
        "hidden_dim": hidden_dim,
        "terms_resolved": resolved_count,
        "gotue_path": str(gotue_path),
        "gotue_mb": round(gotue_path.stat().st_size / 1024 / 1024, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract models for comparison")
    parser.add_argument("--models", nargs="+", help="Specific models to extract")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, info in MODELS.items():
            print(f"  {name:30s} {info['desc']}")
        return

    models_to_extract = args.models or list(MODELS.keys())
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    for name in models_to_extract:
        if name not in MODELS:
            print(f"Unknown model: {name}. Use --list to see available models.")
            sys.exit(1)
        result = extract_model(name, MODELS[name], args.output)
        if result:
            results.append(result)

    # Write summary
    print(f"\n{'='*60}")
    print(f"Extraction complete: {len(results)}/{len(models_to_extract)} models")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:30s} {r['vocab_size']:>6d} vocab × {r['hidden_dim']:>4d} dim  ({r['gotue_mb']:.0f} MB)  {r['terms_resolved']} terms")

    summary_path = args.output / "extraction-summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
