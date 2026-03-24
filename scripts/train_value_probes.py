#!/usr/bin/env python3
"""
Train value probes using the activation server.

Generates training data by sending value-relevant prompts through the
activation server, labels them, writes .gotact format, and calls
got-cli train for each value dimension.

Usage:
    python scripts/train_value_probes.py \
        --activation-server http://localhost:8100 \
        --layer 8 \
        --output data/probes/probes_layer8.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import requests

# Positive prompts (express the value) and negative (express the opposite or neutral)
VALUE_PROMPTS = {
    "honesty": {
        "positive": [
            "I believe in always telling the truth, even when it's difficult.",
            "Transparency and honesty are the foundation of every good relationship.",
            "Being truthful builds trust that lasts a lifetime.",
            "I would rather face hard truths than comfortable lies.",
            "Honest communication prevents misunderstandings and builds respect.",
            "The truth may hurt temporarily, but lies cause lasting damage.",
            "Integrity means being honest even when no one is watching.",
            "I value people who speak truthfully and openly.",
            "Dishonesty erodes the fabric of society over time.",
            "A commitment to truth is a commitment to justice.",
        ],
        "negative": [
            "Sometimes you need to hide the truth to protect people.",
            "White lies are just part of being polite in society.",
            "What people don't know can't hurt them.",
            "Strategic ambiguity is a valuable diplomatic tool.",
            "The weather today is quite pleasant for this time of year.",
            "I enjoy watching movies on the weekend with my family.",
            "The recipe calls for two cups of flour and one egg.",
            "Traffic was heavy on the highway this morning.",
            "The conference will be held in the main auditorium.",
            "Please submit your report by the end of the week.",
        ],
    },
    "compassion": {
        "positive": [
            "We should care deeply about the suffering of others.",
            "Empathy and kindness make the world a better place.",
            "Helping those in need is a fundamental moral obligation.",
            "I feel moved when I see others in pain and want to help.",
            "Compassion means acting to reduce suffering wherever we find it.",
            "Every person deserves to be treated with warmth and understanding.",
            "The measure of a society is how it treats its most vulnerable.",
            "Kindness costs nothing but means everything.",
            "We should extend grace and understanding to those who struggle.",
            "Taking care of each other is what makes us human.",
        ],
        "negative": [
            "People should solve their own problems without help.",
            "Weakness should not be rewarded with sympathy.",
            "Survival of the fittest is nature's way.",
            "Emotional attachment clouds rational judgment.",
            "The stock market closed higher today on strong earnings.",
            "Please ensure all documents are filed alphabetically.",
            "The meeting has been rescheduled to next Thursday.",
            "Our quarterly projections look promising this year.",
            "The new software update includes performance improvements.",
            "Remember to water the plants before you leave.",
        ],
    },
    "fairness": {
        "positive": [
            "Everyone deserves equal treatment regardless of their background.",
            "Justice requires that we treat all people fairly and impartially.",
            "A fair society gives everyone an equal opportunity to succeed.",
            "Rules should apply equally to everyone without exception.",
            "We must actively fight against bias and discrimination.",
            "Equal rights are not negotiable — they are fundamental.",
            "Fairness means considering the needs of all stakeholders.",
            "No one should be disadvantaged because of circumstances beyond their control.",
            "A just system holds the powerful to the same standards as everyone else.",
            "Equity requires addressing systemic barriers to opportunity.",
        ],
        "negative": [
            "Some people are simply more deserving than others.",
            "Life isn't fair and we shouldn't pretend it can be.",
            "Merit alone should determine outcomes, regardless of starting point.",
            "The strong naturally rise to the top in any system.",
            "The library closes at nine on weekday evenings.",
            "This model of laptop has excellent battery life.",
            "We should take the scenic route through the countryside.",
            "The orchestra will perform Beethoven's Fifth Symphony tonight.",
            "Regular exercise improves both physical and mental health.",
            "The deadline for applications is the first of next month.",
        ],
    },
    "courage": {
        "positive": [
            "Standing up for what's right even when it costs you takes real bravery.",
            "Courage means facing fear and acting on your principles anyway.",
            "The world needs more people willing to speak truth to power.",
            "It takes courage to defend the unpopular but correct position.",
            "Bravery is not the absence of fear but action despite it.",
            "We should admire those who risk everything for their beliefs.",
            "Moral courage means refusing to stay silent in the face of injustice.",
            "Sometimes doing the right thing requires extraordinary personal sacrifice.",
            "Standing firm on principles despite social pressure is heroic.",
            "Courage is the virtue that makes all other virtues possible.",
        ],
        "negative": [
            "It's better to stay safe than to take unnecessary risks.",
            "Going along with the group is usually the wisest course.",
            "Self-preservation should always come first.",
            "Why make enemies when you can just keep your head down?",
            "The parking lot will be repaved over the weekend.",
            "This recipe makes approximately twelve servings.",
            "The flight departs at six thirty in the morning.",
            "Please remember to lock the door when you leave.",
            "The average temperature in July is around thirty degrees.",
            "Our next team meeting is scheduled for Monday afternoon.",
        ],
    },
}


def write_gotact(path: Path, model_id: str, hidden_dim: int, layer: int, activations: list[list[float]]):
    """Write activations in .gotact binary format."""
    num_positions = len(activations)
    with open(path, "wb") as f:
        f.write(b"GOTA")
        f.write(struct.pack("<H", 1))  # version
        model_bytes = model_id.encode("utf-8")
        f.write(struct.pack("<I", len(model_bytes)))
        f.write(model_bytes)
        f.write(struct.pack("<B", 0))  # fp32
        f.write(struct.pack("<I", hidden_dim))
        f.write(struct.pack("<I", 1))  # 1 layer
        f.write(struct.pack("<I", num_positions))

        for pos, act in enumerate(activations):
            f.write(struct.pack("<I", layer))
            f.write(struct.pack("<I", pos))
            for v in act:
                f.write(struct.pack("<f", v))


def main():
    parser = argparse.ArgumentParser(description="Train value probes via activation server")
    parser.add_argument("--activation-server", default="http://localhost:8100")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--unembedding", default="data/models/qwen35-9b.gotue",
                        help="Path to .gotue for geometry")
    parser.add_argument("--output", required=True, type=Path, help="Output probes JSON")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.output.parent

    url = args.activation_server
    layer = args.layer

    # Check server
    health = requests.get(f"{url}/health").json()
    print(f"Activation server: {health['hidden_dim']}d, {health['n_layers']} layers")

    dimensions = list(VALUE_PROMPTS.keys())
    all_probes = []

    for dim in dimensions:
        print(f"\n=== Training probe: {dim} ===")
        prompts = VALUE_PROMPTS[dim]
        pos_prompts = prompts["positive"]
        neg_prompts = prompts["negative"]
        all_prompts = pos_prompts + neg_prompts
        labels = [1] * len(pos_prompts) + [0] * len(neg_prompts)

        # Get activations from activation server
        print(f"  Extracting {len(all_prompts)} activations at layer {layer}...")
        activations = []
        for prompt in all_prompts:
            r = requests.post(f"{url}/hidden_states",
                              json={"text": prompt, "layer": layer}, timeout=30)
            activations.append(r.json()["hidden_state"])

        hidden_dim = len(activations[0])
        print(f"  Got {len(activations)} x {hidden_dim}d activations")

        # Write .gotact
        act_path = work_dir / f"activations_{dim}.gotact"
        write_gotact(act_path, "qwen3-8b", hidden_dim, layer, activations)

        # Write labels
        labels_path = work_dir / f"labels_{dim}.txt"
        with open(labels_path, "w") as f:
            for label in labels:
                f.write(f"{label}\n")

        # Train probe via got-cli
        probe_path = work_dir / f"probe_{dim}.json"
        cmd = [
            "cargo", "run", "--release", "-p", "got-cli", "--",
            "train",
            "--activations", str(act_path),
            "--labels", str(labels_path),
            "--unembedding", str(args.unembedding),
            "--layer", str(layer),
            "--dimension", dim,
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--output", str(probe_path),
        ]
        print(f"  Training ({args.epochs} epochs, lr={args.lr})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            continue

        # Load the trained probe and collect it
        with open(probe_path) as f:
            ps = json.load(f)
        if ps.get("probes"):
            all_probes.append(ps["probes"][0])
            print(f"  Trained: weights dim={len(ps['probes'][0]['weights'])}, bias={ps['probes'][0]['bias']:.4f}")

    if not all_probes:
        print("ERROR: No probes trained successfully")
        sys.exit(1)

    # Combine into single ProbeSet
    combined = {
        "probes": all_probes,
        "version": "v1",
        "corpus_version": "interactive-training",
        "layer": layer,
        "geometry_hash": None,
        "max_drift": None,
        "max_directional_drift": None,
    }

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved {len(all_probes)} probes to {args.output}")
    print(f"Dimensions: {[p['dimension_name'] for p in all_probes]}")
    print(f"\nNext: restart got-web with --probes {args.output}")


if __name__ == "__main__":
    main()
