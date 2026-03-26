#!/usr/bin/env python3
"""
Extract just the unembedding matrix from a single model shard.
For models too large to download fully, we download only the shard
containing lm_head / embed_out and extract the weight directly.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer


def write_gotue(path, vocab_size, hidden_dim, data):
    assert data.shape == (vocab_size, hidden_dim)
    flat = data.float().cpu().contiguous()
    with open(path, "wb") as f:
        f.write(b"GOTU")
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack(f"<{vocab_size * hidden_dim}f", *flat.view(-1).tolist()))


VALUE_TERMS = [
    "honesty", "integrity", "fairness", "transparency", "accountability",
    "justice", "freedom", "equality", "equity", "compassion",
    "empathy", "courage", "bravery", "wisdom", "humility",
    "loyalty", "responsibility", "resilience", "openness", "creativity",
    "innovation", "efficiency", "tradition", "cruelty", "oppression",
    "secrecy", "truthfulness", "cowardice",
]


def extract(hf_name, output_name, output_dir):
    print(f"\n{'='*60}")
    print(f"  Extracting: {hf_name} (shard-only mode)")
    print(f"{'='*60}")

    config = AutoConfig.from_pretrained(hf_name)
    hidden_dim = config.hidden_size
    vocab_size = config.vocab_size
    print(f"  Config: {vocab_size} vocab x {hidden_dim} hidden")

    # Find which shard has the unembedding
    index_path = hf_hub_download(hf_name, "pytorch_model.bin.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Look for lm_head or embed_out
    ue_key = None
    ue_shard = None
    for key, shard in index["weight_map"].items():
        if "lm_head" in key or "embed_out" in key:
            ue_key = key
            ue_shard = shard
            break

    if ue_key is None:
        print("  ERROR: cannot find unembedding weight in index")
        return None

    print(f"  Unembedding: {ue_key} in {ue_shard}")
    print(f"  Downloading shard (this may take a while)...")

    shard_path = hf_hub_download(hf_name, ue_shard)
    print(f"  Loading shard...")

    # Load just this shard
    state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
    ue_weight = state_dict[ue_key].float()
    del state_dict  # free memory immediately

    actual_vocab = ue_weight.shape[0]
    print(f"  Unembedding shape: {ue_weight.shape}")

    # Write .gotue
    gotue_path = output_dir / f"{output_name}.gotue"
    write_gotue(gotue_path, actual_vocab, hidden_dim, ue_weight)
    print(f"  Written: {gotue_path} ({gotue_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Tokenizer for vocab + term analysis
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)

    # Vocab
    vocab_path = output_dir / f"{output_name}-vocab.json"
    vocab_tokens = [tokenizer.decode([i]) for i in range(actual_vocab)]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_tokens, f, ensure_ascii=False)

    # Term analysis
    term_results = {}
    resolved = 0
    for term in VALUE_TERMS:
        candidates = [term, term.capitalize(), f" {term}", f" {term.capitalize()}"]
        found = False
        for candidate in candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                idx = ids[0]
                vec = ue_weight[idx]
                term_results[term] = {
                    "found": True, "token_index": idx,
                    "token_string": tokenizer.decode([idx]),
                    "norm": round(vec.norm().item(), 4),
                }
                resolved += 1
                found = True
                break
        if not found:
            term_results[term] = {"found": False}

    term_path = output_dir / f"{output_name}-term-analysis.json"
    with open(term_path, "w") as f:
        json.dump(term_results, f, indent=2)
    print(f"  Terms resolved: {resolved}/{len(VALUE_TERMS)}")

    del ue_weight
    return {"name": output_name, "vocab_size": actual_vocab, "hidden_dim": hidden_dim, "terms": resolved}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract(args.hf_name, args.output_name, args.output_dir)


if __name__ == "__main__":
    main()
