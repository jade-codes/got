#!/usr/bin/env python3
"""
Inject an activation vector at a specified transformer layer and generate text.

This is the injection counterpart to extract_activations.py. It:
  1. Loads a HuggingFace transformer model
  2. Reads an activation vector from a binary file (.gotinj)
  3. Registers a forward hook that REPLACES the hidden state at the target layer
  4. Runs generation from the injected state
  5. Outputs: generated text, output logits, and entropy/confidence metrics

Binary input format (.gotinj):
  hidden_dim:  u32 LE
  layer_index: u32 LE
  values:      hidden_dim x f32 LE

Usage:
    python inject_and_generate.py \
        --model gpt2 \
        --injection activation.gotinj \
        --max-tokens 50 \
        --output result.json

    # Or pipe a vector via stdin (same binary format):
    python inject_and_generate.py --model gpt2 --injection - --max-tokens 50

Dependencies:
    pip install torch transformers
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# .gotinj format
# ---------------------------------------------------------------------------

def read_gotinj(path: Path) -> Tuple[int, int, List[float]]:
    """Read an injection vector. Returns (hidden_dim, layer_index, values)."""
    if str(path) == "-":
        data = sys.stdin.buffer.read()
    else:
        data = path.read_bytes()

    offset = 0
    (hidden_dim,) = struct.unpack_from("<I", data, offset)
    offset += 4
    (layer_index,) = struct.unpack_from("<I", data, offset)
    offset += 4
    values = list(struct.unpack_from(f"<{hidden_dim}f", data, offset))
    return hidden_dim, layer_index, values


def write_gotinj(path: Path, hidden_dim: int, layer_index: int, values: List[float]) -> None:
    """Write an injection vector to .gotinj format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", layer_index))
        f.write(struct.pack(f"<{hidden_dim}f", *values))


# ---------------------------------------------------------------------------
# Architecture detection (shared with extract_activations.py)
# ---------------------------------------------------------------------------

def get_layer_list(model) -> Tuple[object, str]:
    """Return (layer_list, arch_name) for the model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, "GPT2"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "LLaMA"
    if (hasattr(model, "model") and hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "layers")):
        return model.model.decoder.layers, "OPT"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers, "GPTNeoX"
    raise RuntimeError(
        f"Unsupported architecture: {type(model).__name__}. "
        "Add a branch for this model's layer access path."
    )


# ---------------------------------------------------------------------------
# Injection + generation
# ---------------------------------------------------------------------------

def inject_and_generate(
    model_name: str,
    hidden_dim: int,
    layer_index: int,
    injection_vector: List[float],
    max_tokens: int = 50,
    device: str = "auto",
    temperature: float = 1.0,
    seed_text: str = "",
) -> Dict:
    """
    Inject an activation vector at a layer and generate text.

    Returns a dict with:
      - generated_text: str
      - output_logits: list of floats (vocab-sized logit vector at final position)
      - output_entropy: float (Shannon entropy of softmax)
      - model_confidence: float (max softmax probability)
      - num_tokens_generated: int
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype=torch.float32)
    if device != "cpu":
        try:
            import accelerate  # noqa: F401
            load_kwargs["device_map"] = device
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    layer_list, arch_name = get_layer_list(model)
    print(f"Architecture: {arch_name}, injecting at layer {layer_index}", file=sys.stderr)

    # Prepare injection tensor
    inj_tensor = torch.tensor(injection_vector, dtype=torch.float32).to(model.device)

    # The hook replaces the hidden state at the target layer for all positions
    # with the injection vector (broadcast across sequence length).
    injection_done = [False]  # mutable flag

    def injection_hook(module, input, output):
        if injection_done[0]:
            return output  # only inject once (first forward pass)
        injection_done[0] = True

        if isinstance(output, tuple):
            hidden = output[0]
            # Replace all positions with the injection vector
            batch_size, seq_len, _ = hidden.shape
            replaced = inj_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            return (replaced,) + output[1:]
        else:
            batch_size, seq_len, _ = output.shape
            return inj_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

    # Register hook
    hook_handle = layer_list[layer_index].register_forward_hook(injection_hook)

    # Tokenise seed text (or use BOS token)
    if seed_text:
        input_ids = tokenizer.encode(seed_text, return_tensors="pt").to(model.device)
    else:
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.tensor([[bos]], device=model.device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-8),
            pad_token_id=tokenizer.pad_token_id,
        )

    hook_handle.remove()

    # Decode generated text
    generated_ids = output[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Get final logits by running one more forward pass (without injection)
    with torch.no_grad():
        final_output = model(output)
        logits = final_output.logits[0, -1, :].float().cpu()

    logits_list = logits.tolist()

    # Compute entropy and confidence
    probs = torch.softmax(logits, dim=0)
    log_probs = torch.log_softmax(logits, dim=0)
    entropy = -(probs * log_probs).sum().item()
    confidence = probs.max().item()

    return {
        "generated_text": generated_text,
        "output_logits": logits_list,
        "output_entropy": entropy,
        "model_confidence": confidence,
        "num_tokens_generated": len(generated_ids),
        "layer_index": layer_index,
        "hidden_dim": hidden_dim,
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# Batch mode: run interpolation experiment
# ---------------------------------------------------------------------------

def run_interpolation(
    model_name: str,
    vector_a: List[float],
    vector_b: List[float],
    layer_index: int,
    steps: List[float],
    max_tokens: int = 50,
    device: str = "auto",
    temperature: float = 1.0,
    seed_text: str = "",
) -> List[Dict]:
    """
    Run injection + generation at each interpolation step between vector_a and vector_b.

    Returns a list of result dicts, one per step.
    """
    hidden_dim = len(vector_a)
    assert len(vector_b) == hidden_dim, "vectors must have same dimension"

    results = []
    for t in steps:
        # Linear interpolation: h(t) = (1-t)*a + t*b
        interpolated = [(1.0 - t) * a + t * b for a, b in zip(vector_a, vector_b)]
        print(f"\n--- Step t={t:.3f} ---", file=sys.stderr)

        result = inject_and_generate(
            model_name=model_name,
            hidden_dim=hidden_dim,
            layer_index=layer_index,
            injection_vector=interpolated,
            max_tokens=max_tokens,
            device=device,
            temperature=temperature,
            seed_text=seed_text,
        )
        result["t"] = t
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject an activation vector at a transformer layer and generate text."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--injection", type=Path, required=True,
                        help="Path to .gotinj file (or '-' for stdin)")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: stdout)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, ...)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--seed-text", type=str, default="",
                        help="Optional seed text to start generation")

    args = parser.parse_args()

    hidden_dim, layer_index, values = read_gotinj(args.injection)
    print(f"Injection: dim={hidden_dim}, layer={layer_index}", file=sys.stderr)

    result = inject_and_generate(
        model_name=args.model,
        hidden_dim=hidden_dim,
        layer_index=layer_index,
        injection_vector=values,
        max_tokens=args.max_tokens,
        device=args.device,
        temperature=args.temperature,
        seed_text=args.seed_text,
    )

    # Strip large logits array for readable output (keep in full JSON)
    output_json = json.dumps(result, indent=2)

    if args.output:
        args.output.write_text(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
