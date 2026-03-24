#!/usr/bin/env python3
"""
Train a sparse autoencoder (SAE) on transformer activations.

Produces an SAE checkpoint that extract_sae_features.py can consume.
Uses a simple top-k SAE architecture trained on residual stream activations.

Usage:
    python scripts/train_sae.py \
        --model meta-llama/Llama-3-8B \
        --layer 16 \
        --n-features 4096 \
        --k 64 \
        --prompts data/prompts.txt \
        --output data/sae/llama3-8b-layer16.pt

    # Quick test with GPT-2:
    python scripts/train_sae.py \
        --model gpt2 \
        --layer 6 \
        --n-features 2048 \
        --k 32 \
        --output data/sae/gpt2-layer6.pt

Dependencies:
    pip install torch transformers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# SAE architecture
# ---------------------------------------------------------------------------

class TopKSAE(nn.Module):
    """Top-k sparse autoencoder.

    Encodes d-dimensional activations into n_features latents, keeping
    only the top-k active. Decoder reconstructs from the sparse code.

    Architecture:
        encode: x -> W_enc @ (x - b_dec) + b_enc -> top-k -> z
        decode: z -> W_dec @ z + b_dec -> x_hat

    W_dec columns are unit-normalised (the feature directions).
    """

    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        self.W_enc = nn.Parameter(torch.randn(n_features, d_model) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialise W_dec with unit-norm columns
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse latent code (batch_size, n_features)."""
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        # Top-k sparsity
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from sparse latent code."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """Full forward pass. Returns (x_hat, z, loss_dict)."""
        z = self.encode(x)
        x_hat = self.decode(z)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # L1 on active features (encourages sparsity beyond top-k)
        l1_loss = z.abs().sum(dim=-1).mean()

        return x_hat, z, {"recon": recon_loss, "l1": l1_loss}

    def normalise_decoder(self):
        """Re-normalise decoder weights to unit norm."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def feature_directions(self) -> torch.Tensor:
        """Return the n_features decoder directions (n_features, d_model)."""
        return self.W_dec.detach().cpu()

    def save(self, path: Path, metadata: dict = None):
        """Save SAE checkpoint with metadata."""
        checkpoint = {
            "d_model": self.d_model,
            "n_features": self.n_features,
            "k": self.k,
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TopKSAE":
        """Load SAE from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        sae = cls(
            d_model=checkpoint["d_model"],
            n_features=checkpoint["n_features"],
            k=checkpoint["k"],
        )
        sae.load_state_dict(checkpoint["state_dict"])
        sae.metadata = checkpoint.get("metadata", {})
        return sae


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    # Value-relevant
    "Honesty and transparency are essential for building trust.",
    "Deception can sometimes be justified to protect the innocent.",
    "Everyone deserves to be treated with fairness and dignity.",
    "Power should be used to protect the vulnerable, not exploit them.",
    "Freedom of thought is a fundamental human right.",
    "Compassion for others is the foundation of a good society.",
    "Innovation requires the courage to challenge established norms.",
    "Tradition provides stability and wisdom from past generations.",
    "Justice demands that wrongdoers face consequences for their actions.",
    "Mercy and forgiveness can break cycles of violence and harm.",
    # General knowledge (for diverse activations)
    "The capital of France is Paris, located on the Seine River.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Machine learning models learn patterns from training data.",
    "The stock market reflects collective expectations about future earnings.",
    "Climate change is driven primarily by greenhouse gas emissions.",
    "Shakespeare wrote plays that explore the full range of human emotion.",
    "Mathematics provides a universal language for describing natural phenomena.",
    "Democracy requires an informed and engaged citizenry to function well.",
    "Music can evoke powerful emotions across cultural boundaries.",
    "The scientific method relies on hypothesis testing and peer review.",
    # Longer prompts for more token positions
    "When faced with a moral dilemma, people often struggle to balance competing values like honesty and kindness.",
    "The development of artificial intelligence raises important questions about safety, fairness, and human autonomy.",
    "Historical progress toward greater equality has often required courage from those who challenged unjust systems.",
    "Environmental stewardship requires balancing economic development with preservation of natural ecosystems.",
    "Educational systems should foster both critical thinking and respect for diverse perspectives.",
    "Medical ethics involves navigating complex tradeoffs between patient autonomy and physician responsibility.",
    "Technological innovation can either empower or constrain human freedom depending on how it is governed.",
    "Social cohesion depends on shared values, mutual respect, and willingness to engage across differences.",
    "Economic inequality raises fundamental questions about fairness, opportunity, and the social contract.",
    "The rule of law depends on public trust in institutions and fair application of justice.",
]


def collect_activations(
    model,
    tokenizer,
    layer_idx: int,
    prompts: List[str],
    device: str,
    max_tokens: int = 128,
) -> torch.Tensor:
    """Collect residual stream activations from a specific layer.

    Returns tensor of shape (total_tokens, d_model).
    """
    all_layers = _detect_layers(model)
    if layer_idx >= len(all_layers):
        raise ValueError(f"Layer {layer_idx} out of range (model has {len(all_layers)})")

    all_acts = []
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["acts"] = hidden[0].detach()  # (seq_len, d_model)

    hook = all_layers[layer_idx].register_forward_hook(hook_fn)

    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_tokens,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        if "acts" in captured:
            all_acts.append(captured["acts"].cpu())
        captured.clear()

    hook.remove()

    if not all_acts:
        raise ValueError("No activations collected")

    return torch.cat(all_acts, dim=0)  # (total_tokens, d_model)


def _detect_layers(model) -> list:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise ValueError(f"Unknown architecture: {type(model).__name__}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sae(
    activations: torch.Tensor,
    n_features: int,
    k: int,
    lr: float = 3e-4,
    epochs: int = 50,
    batch_size: int = 256,
    l1_weight: float = 1e-3,
    device: str = "cpu",
    log_every: int = 10,
) -> TopKSAE:
    """Train a top-k SAE on collected activations."""
    d_model = activations.shape[1]
    n_samples = activations.shape[0]

    print(f"  Training SAE: {n_samples} samples, {d_model}d -> {n_features} features (k={k})")

    sae = TopKSAE(d_model, n_features, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Normalise activations
    act_mean = activations.mean(dim=0)
    act_std = activations.std(dim=0).clamp(min=1e-6)
    activations_norm = (activations - act_mean) / act_std
    activations_norm = activations_norm.to(device)

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        total_recon = 0.0
        total_l1 = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = activations_norm[batch_idx]

            x_hat, z, losses = sae(batch)
            loss = losses["recon"] + l1_weight * losses["l1"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalise_decoder()

            total_recon += losses["recon"].item()
            total_l1 += losses["l1"].item()
            n_batches += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            avg_recon = total_recon / n_batches
            avg_l1 = total_l1 / n_batches

            # Compute sparsity: mean number of active features per sample
            with torch.no_grad():
                sample = activations_norm[:min(1000, n_samples)]
                z = sae.encode(sample)
                active = (z > 0).float().sum(dim=-1).mean().item()

            print(f"  Epoch {epoch + 1:4d}/{epochs}: recon={avg_recon:.6f}, l1={avg_l1:.4f}, active={active:.1f}/{n_features}")

    # Store normalisation params in SAE for later use
    sae.act_mean = act_mean.cpu()
    sae.act_std = act_std.cpu()

    return sae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SAE on transformer activations")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--layer", required=True, type=int, help="Layer index to extract from")
    parser.add_argument("--n-features", type=int, default=4096, help="Number of SAE features")
    parser.add_argument("--k", type=int, default=64, help="Top-k sparsity")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1-weight", type=float, default=1e-3)
    parser.add_argument("--prompts", type=Path, default=None,
                        help="Text file with one prompt per line (default: built-in set)")
    parser.add_argument("--output", required=True, type=Path, help="Output .pt checkpoint path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts
    if args.prompts and args.prompts.exists():
        prompts = [l.strip() for l in open(args.prompts) if l.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using {len(prompts)} built-in prompts")

    # Load model
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = args.device
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True,
        )
        device = str(next(model.parameters()).device)
    except Exception:
        print("  Falling back to CPU")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        device = "cpu"
    model.eval()
    d_model = model.config.hidden_size
    print(f"  {d_model}d, {model.config.num_hidden_layers} layers")

    # Collect activations
    print(f"Collecting activations from layer {args.layer}...")
    acts = collect_activations(model, tokenizer, args.layer, prompts, device)
    print(f"  Collected {acts.shape[0]} token activations ({acts.shape[1]}d)")

    # Free model memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Train SAE
    print("Training SAE...")
    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = train_sae(
        acts,
        n_features=args.n_features,
        k=args.k,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l1_weight=args.l1_weight,
        device=train_device,
    )

    # Save
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "n_samples": acts.shape[0],
        "d_model": d_model,
    }
    sae.save(args.output, metadata)
    print(f"Saved SAE to {args.output}")
    print(f"  {args.n_features} features, k={args.k}, d_model={d_model}")


if __name__ == "__main__":
    main()
