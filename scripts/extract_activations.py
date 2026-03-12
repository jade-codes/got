#!/usr/bin/env python3
"""
Extract residual-stream activations and the unembedding matrix from a
HuggingFace transformer model, writing them in the binary formats consumed
by got-cli (.gotact and .gotue).

Usage:
    python extract_activations.py \
        --model meta-llama/Llama-3-8B \
        --input "The cat sat on the mat" \
        --layers 12 18 24 \
        --output-activations activations.gotact \
        --output-unembedding unembedding.gotue

Dependencies:
    pip install torch transformers
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# .gotact format (see PLAN.md §6.1)
# ---------------------------------------------------------------------------
#   Magic:          4 bytes   "GOTA"
#   Version:        u16 LE    (1)
#   Model ID:       u32 LE len + UTF-8 bytes
#   Precision tag:  u8        (0=fp32, 1=fp16, 2=bf16, 3=int8)
#   hidden_dim:     u32 LE
#   num_layers:     u32 LE
#   num_positions:  u32 LE
#   For each layer (num_layers):
#     layer_index:  u32 LE
#     For each position (num_positions):
#       token_position: u32 LE
#       values: hidden_dim × f32 LE


def write_string(f, s: str) -> None:
    encoded = s.encode("utf-8")
    f.write(struct.pack("<I", len(encoded)))
    f.write(encoded)


def write_gotact(
    path: Path,
    model_id: str,
    hidden_dim: int,
    layer_activations: Dict[int, torch.Tensor],
    precision_tag: int = 0,
) -> None:
    """
    Write activations to .gotact binary format.

    layer_activations: dict mapping layer_index → tensor of shape (num_positions, hidden_dim)
    """
    sorted_layers = sorted(layer_activations.keys())
    num_layers = len(sorted_layers)

    # All layers must have the same number of positions
    num_positions = layer_activations[sorted_layers[0]].shape[0]
    for layer_idx in sorted_layers:
        assert layer_activations[layer_idx].shape == (num_positions, hidden_dim), (
            f"Layer {layer_idx} shape {layer_activations[layer_idx].shape} "
            f"does not match expected ({num_positions}, {hidden_dim})"
        )

    with open(path, "wb") as f:
        # Header
        f.write(b"GOTA")
        f.write(struct.pack("<H", 1))  # version
        write_string(f, model_id)
        f.write(struct.pack("<B", precision_tag))
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", num_layers))
        f.write(struct.pack("<I", num_positions))

        # Per-layer data
        for layer_idx in sorted_layers:
            f.write(struct.pack("<I", layer_idx))
            tensor = layer_activations[layer_idx].float().cpu()
            for pos in range(num_positions):
                f.write(struct.pack("<I", pos))
                f.write(struct.pack(f"<{hidden_dim}f", *tensor[pos].tolist()))


# ---------------------------------------------------------------------------
# .gotue format (see PLAN.md §6.2)
# ---------------------------------------------------------------------------
#   Magic:          4 bytes   "GOTU"
#   Version:        u16 LE    (1)
#   vocab_size V:   u32 LE
#   hidden_dim d:   u32 LE
#   data:           V × d × f32 LE   (row-major)


def write_gotue(
    path: Path,
    vocab_size: int,
    hidden_dim: int,
    data: torch.Tensor,
) -> None:
    """
    Write the unembedding matrix to .gotue binary format.

    data: tensor of shape (vocab_size, hidden_dim), row-major.
    """
    assert data.shape == (vocab_size, hidden_dim), (
        f"Unembedding shape {data.shape} does not match ({vocab_size}, {hidden_dim})"
    )

    flat = data.float().cpu().contiguous()
    with open(path, "wb") as f:
        f.write(b"GOTU")
        f.write(struct.pack("<H", 1))  # version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_dim))
        # Write all values as little-endian f32
        f.write(struct.pack(f"<{vocab_size * hidden_dim}f", *flat.view(-1).tolist()))


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------


def extract(
    model_name: str,
    input_text: str,
    target_layers: List[int],
    output_act: Path,
    output_ue: Path,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
) -> None:
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    print(f"Loading model: {model_name} (dtype={dtype})")
    load_kwargs = dict(
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    # Only pass device_map when accelerate is available; otherwise load to CPU
    if device != "cpu":
        try:
            import accelerate  # noqa: F401
            load_kwargs["device_map"] = device
        except ImportError:
            print("  (accelerate not installed — loading to CPU)")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    config = model.config
    hidden_dim = config.hidden_size
    num_layers_total = config.num_hidden_layers

    # Validate requested layers
    for layer in target_layers:
        if layer < 0 or layer >= num_layers_total:
            print(
                f"Error: layer {layer} out of range [0, {num_layers_total})",
                file=sys.stderr,
            )
            sys.exit(1)

    # Tokenise
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    num_tokens = input_ids.shape[1]
    print(f"Input tokens: {num_tokens} | Hidden dim: {hidden_dim} | Layers: {target_layers}")

    # -----------------------------------------------------------------------
    # Auto-detect architecture for layer access and unembedding extraction.
    #
    # Supported architectures:
    #   GPT-2:          model.transformer.h[i]         + model.lm_head
    #   LLaMA/Mistral:  model.model.layers[i]          + model.lm_head
    #   OPT:            model.model.decoder.layers[i]   + model.lm_head
    #   GPTNeoX/Pythia: model.gpt_neox.layers[i]       + model.embed_out
    # -----------------------------------------------------------------------

    def _get_layer_list(m) -> Tuple[object, str]:
        """Return (layer_list, arch_name) for the model."""
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h, "GPT2"
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers, "LLaMA"
        if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
            return m.model.decoder.layers, "OPT"
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers, "GPTNeoX"
        raise RuntimeError(
            f"Unsupported architecture: {type(m).__name__}. "
            "Add a branch for this model's layer access path."
        )

    layer_list, arch_name = _get_layer_list(model)
    print(f"Detected architecture: {arch_name} ({type(model).__name__})")

    # Register hooks on residual stream (output of each transformer block)
    activations: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) — take the first element
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # hidden: (batch, seq_len, hidden_dim) — take batch 0
            activations[layer_idx] = hidden[0].detach()

        return hook_fn

    hooks = []
    for layer_idx in target_layers:
        layer_module = layer_list[layer_idx]
        h = layer_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        model(input_ids)

    # Clean up hooks
    for h in hooks:
        h.remove()

    print(f"Captured activations for {len(activations)} layers")

    # Extract unembedding matrix — try common locations across architectures
    if hasattr(model, "lm_head"):
        ue_weight = model.lm_head.weight.detach()
    elif hasattr(model, "embed_out"):
        ue_weight = model.embed_out.weight.detach()
    else:
        print(
            "Error: could not find unembedding matrix (tried lm_head, embed_out)",
            file=sys.stderr,
        )
        sys.exit(1)

    vocab_size = ue_weight.shape[0]
    print(f"Unembedding matrix: {vocab_size} × {hidden_dim}")

    # Determine precision tag
    precision_tag = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }.get(dtype, 0)

    # Write outputs
    write_gotact(output_act, model_name, hidden_dim, activations, precision_tag)
    print(f"Activations written to {output_act}")

    write_gotue(output_ue, vocab_size, hidden_dim, ue_weight)
    print(f"Unembedding matrix written to {output_ue}")

    # Also write labels stub (all zeros — user must provide real labels)
    labels_path = output_act.with_suffix(".labels")
    with open(labels_path, "w") as f:
        for _ in range(num_tokens):
            f.write("0\n")
    print(f"Label stub written to {labels_path} (edit with real labels before training)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract residual-stream activations and unembedding matrix "
        "from a HuggingFace transformer model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g. meta-llama/Llama-3-8B)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text to run through the model",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer indices to extract activations from (0-indexed)",
    )
    parser.add_argument(
        "--output-activations",
        type=Path,
        default=Path("activations.gotact"),
        help="Output path for activations (.gotact)",
    )
    parser.add_argument(
        "--output-unembedding",
        type=Path,
        default=Path("unembedding.gotue"),
        help="Output path for unembedding matrix (.gotue)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on (auto, cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for loading",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow execution of remote code from the model repository (security risk)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    extract(
        model_name=args.model,
        input_text=args.input,
        target_layers=args.layers,
        output_act=args.output_activations,
        output_ue=args.output_unembedding,
        device=args.device,
        dtype=dtype_map[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
