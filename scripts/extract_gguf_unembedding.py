#!/usr/bin/env python3
"""
Extract the unembedding matrix from a GGUF model file and write it as .gotue.

GGUF files store tensors in quantized formats (Q4_K, Q6_K, etc.).
This script reads the output.weight tensor, dequantizes it to fp32,
and writes it in the .gotue binary format for use with got-web/got-cli.

Usage:
    python scripts/extract_gguf_unembedding.py \
        --gguf data/qwen35-9b.gguf \
        --output data/models/qwen35-9b.gotue

    # Also extract vocabulary:
    python scripts/extract_gguf_unembedding.py \
        --gguf data/qwen35-9b.gguf \
        --output data/models/qwen35-9b.gotue \
        --vocab-output data/models/qwen35-9b-vocab.json

Dependencies:
    pip install gguf numpy
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, dequantize
except ImportError:
    print("Error: gguf package required. Install with: pip install gguf", file=sys.stderr)
    sys.exit(1)


def extract_unembedding(gguf_path: Path) -> tuple[np.ndarray, list[str]]:
    """Extract the output.weight tensor and vocabulary from a GGUF file.

    Returns (weight_matrix, vocab_tokens).
    weight_matrix shape: (vocab_size, hidden_dim) in fp32.
    """
    print(f"Reading GGUF file: {gguf_path}")
    reader = GGUFReader(str(gguf_path))

    # Find the output.weight tensor
    output_tensor = None
    for tensor in reader.tensors:
        if tensor.name == "output.weight":
            output_tensor = tensor
            break

    if output_tensor is None:
        # Some models use "lm_head.weight" instead
        for tensor in reader.tensors:
            if "lm_head" in tensor.name and "weight" in tensor.name:
                output_tensor = tensor
                break

    if output_tensor is None:
        available = [t.name for t in reader.tensors if "output" in t.name or "lm_head" in t.name or "embed" in t.name]
        print(f"Error: output.weight tensor not found. Available candidates: {available}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found tensor: {output_tensor.name}")
    print(f"  Shape: {output_tensor.shape}")
    print(f"  Type: {output_tensor.tensor_type}")

    # Dequantize to fp32
    print("  Dequantizing to fp32...")
    weight = dequantize(output_tensor.data, output_tensor.tensor_type)
    vocab_size, hidden_dim = weight.shape
    print(f"  Dequantized: {vocab_size} vocab x {hidden_dim} hidden dim")
    print(f"  Value range: [{weight.min():.4f}, {weight.max():.4f}]")

    # Extract vocabulary
    vocab_tokens = []
    for field_name, field in reader.fields.items():
        if 'tokenizer.ggml.tokens' in field_name:
            # Each part after the metadata parts contains a token
            parts = field.parts
            for i, part in enumerate(parts):
                if hasattr(part, 'tobytes'):
                    token = part.tobytes().decode('utf-8', errors='replace')
                    vocab_tokens.append(token)
            break

    if not vocab_tokens:
        print("  Warning: no vocabulary found in GGUF metadata")
    else:
        print(f"  Vocabulary: {len(vocab_tokens)} tokens")

    return weight, vocab_tokens


def write_gotue(path: Path, weight: np.ndarray) -> None:
    """Write unembedding matrix in .gotue binary format."""
    vocab_size, hidden_dim = weight.shape
    with open(path, "wb") as f:
        f.write(b"GOTU")
        f.write(struct.pack("<H", 1))  # version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_dim))
        f.write(weight.astype(np.float32).tobytes())
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {path} ({vocab_size} x {hidden_dim}, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Extract unembedding matrix from GGUF to .gotue")
    parser.add_argument("--gguf", required=True, type=Path, help="Path to GGUF model file")
    parser.add_argument("--output", required=True, type=Path, help="Output .gotue file path")
    parser.add_argument("--vocab-output", type=Path, default=None, help="Output vocabulary JSON path")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    weight, vocab_tokens = extract_unembedding(args.gguf)
    write_gotue(args.output, weight)

    if args.vocab_output:
        args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.vocab_output, "w") as f:
            json.dump(vocab_tokens, f)
        print(f"  Wrote vocabulary: {args.vocab_output} ({len(vocab_tokens)} tokens)")

    print("\nNext steps:")
    print(f"  cargo run -p got-web -- \\")
    print(f"      --geometry {args.output} \\")
    if args.vocab_output:
        print(f"      --vocab {args.vocab_output} \\")
    print(f"      --values values.toml")


if __name__ == "__main__":
    main()
