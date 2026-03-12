#!/usr/bin/env python3
"""
End-to-end test: extract activations from real HuggingFace models, then run
the full got-cli pipeline (train → attest → verify) on the output.

This exercises the Python ↔ Rust boundary across multiple model architectures.

Models tested (all CPU-friendly, <1GB each):
  - sshleifer/tiny-gpt2    GPT2LMHeadModel        (d=2,   V=50257,  2 layers)
  - facebook/opt-125m      OPTForCausalLM          (d=768, V=50272, 12 layers)
  - EleutherAI/pythia-70m  GPTNeoXForCausalLM      (d=512, V=50304,  6 layers)

Usage:
    python scripts/test_real_models.py [--models MODEL [MODEL ...]] [--cli-path PATH]

Prerequisites:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers
    cargo build -p got-cli --release   (or --debug)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelSpec:
    """A model to test, with its expected properties."""

    name: str                # HuggingFace model ID
    arch: str                # Architecture label (for display)
    hidden_dim: int          # Expected hidden_dim
    num_layers: int          # Total layers in model
    test_layers: List[int]   # Which layers to extract


# Default model catalogue — chosen to cover distinct architectures
DEFAULT_MODELS = [
    ModelSpec(
        name="sshleifer/tiny-gpt2",
        arch="GPT2",
        hidden_dim=2,
        num_layers=2,
        test_layers=[0, 1],
    ),
    ModelSpec(
        name="facebook/opt-125m",
        arch="OPT",
        hidden_dim=768,
        num_layers=12,
        test_layers=[0, 6, 11],
    ),
    ModelSpec(
        name="EleutherAI/pythia-70m",
        arch="GPTNeoX",
        hidden_dim=512,
        num_layers=6,
        test_layers=[0, 3, 5],
    ),
]

INPUT_TEXT = "The cat sat on the mat"


def find_cli(cli_path: Optional[str]) -> Path:
    """Locate the got-cli binary."""
    if cli_path:
        p = Path(cli_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"got-cli not found at {p}")

    # Try common locations
    for candidate in [
        Path("target/release/got-cli"),
        Path("target/debug/got-cli"),
    ]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "got-cli binary not found. Run `cargo build -p got-cli` first."
    )


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, printing it for visibility."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
    if result.returncode != 0 and check:
        print(f"    STDERR: {result.stderr.strip()}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def test_model(spec: ModelSpec, cli: Path, workspace: Path) -> dict:
    """
    Full end-to-end test for one model:
      1. Extract activations + unembedding via Python
      2. Generate Ed25519 keypair via got-cli
      3. Train a probe via got-cli
      4. Produce attestation via got-cli
      5. Verify attestation via got-cli
      6. Re-run attestation to check determinism
    """
    model_dir = workspace / spec.name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    act_path = model_dir / "activations.gotact"
    ue_path = model_dir / "unembedding.gotue"
    key_path = model_dir / "test.key"
    pubkey_path = model_dir / "test.pub"
    probes_path = model_dir / "probes.json"
    labels_path = model_dir / "labels.txt"
    attest1_path = model_dir / "attestation1.json"
    attest2_path = model_dir / "attestation2.json"

    results = {"model": spec.name, "arch": spec.arch, "steps": {}}

    # 1. Extract activations
    print(f"\n  [1/6] Extracting activations from {spec.name}...")
    extract_script = Path("scripts/extract_activations.py")
    layers_args = [str(l) for l in spec.test_layers]
    run([
        sys.executable, str(extract_script),
        "--model", spec.name,
        "--input", INPUT_TEXT,
        "--layers", *layers_args,
        "--output-activations", str(act_path),
        "--output-unembedding", str(ue_path),
        "--device", "cpu",
        "--dtype", "float32",
    ])

    assert act_path.exists(), f".gotact not created for {spec.name}"
    assert ue_path.exists(), f".gotue not created for {spec.name}"

    # Validate file magic bytes
    with open(act_path, "rb") as f:
        assert f.read(4) == b"GOTA", f"Bad magic in {act_path}"
    with open(ue_path, "rb") as f:
        assert f.read(4) == b"GOTU", f"Bad magic in {ue_path}"

    results["steps"]["extract"] = "ok"
    print(f"    .gotact: {act_path.stat().st_size:,} bytes")
    print(f"    .gotue:  {ue_path.stat().st_size:,} bytes")

    # 2. Generate keypair
    print(f"  [2/6] Generating keypair...")
    run([str(cli), "keygen", "--output", str(key_path)])
    assert key_path.exists()
    assert pubkey_path.exists()
    results["steps"]["keygen"] = "ok"

    # 3. Train probe (we need labels — generate synthetic binary labels)
    # The gotact file stores num_positions × num_layers activations.
    # For training, we use layer 0's activations and alternating labels.
    print(f"  [3/6] Training probe on layer {spec.test_layers[0]}...")

    # Count how many token positions we have (from the input text)
    # We'll create alternating labels: 0, 1, 0, 1, ...
    # (Meaningless for real interpretation, but exercises the pipeline.)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    tokens = tokenizer(INPUT_TEXT, return_tensors="pt")
    num_positions = tokens["input_ids"].shape[1]

    with open(labels_path, "w") as f:
        for i in range(num_positions):
            f.write(f"{i % 2}\n")

    run([
        str(cli), "train",
        "--activations", str(act_path),
        "--unembedding", str(ue_path),
        "--labels", str(labels_path),
        "--layer", str(spec.test_layers[0]),
        "--dimension", "test-value",
        "--lr", "0.01",
        "--epochs", "50",
        "--output", str(probes_path),
    ])

    assert probes_path.exists()
    results["steps"]["train"] = "ok"

    # 4. Attest (with fixed timestamp for determinism)
    print(f"  [4/6] Producing attestation...")
    run([
        str(cli), "attest",
        "--activations", str(act_path),
        "--probes", str(probes_path),
        "--unembedding", str(ue_path),
        "--key", str(key_path),
        "--model-id", spec.name,
        "--corpus-version", "test-corpus-v1",
        "--timestamp", "1709568000",
        "--output", str(attest1_path),
    ])

    assert attest1_path.exists()
    with open(attest1_path) as f:
        a1 = json.load(f)
    assert a1["model_id"] == spec.name
    assert a1["schema_version"] == 1
    assert len(a1["layer_readings"]) == 1  # one probe set = one layer
    assert len(a1["signature"]) == 128  # 64-byte Ed25519 sig as hex
    results["steps"]["attest"] = "ok"

    # 5. Verify
    print(f"  [5/6] Verifying attestation...")
    run([
        str(cli), "verify",
        "--attestation", str(attest1_path),
        "--pubkey", str(pubkey_path),
    ])
    results["steps"]["verify"] = "ok"

    # 6. Re-attest and check determinism
    print(f"  [6/6] Re-attesting to check determinism...")
    run([
        str(cli), "attest",
        "--activations", str(act_path),
        "--probes", str(probes_path),
        "--unembedding", str(ue_path),
        "--key", str(key_path),
        "--model-id", spec.name,
        "--corpus-version", "test-corpus-v1",
        "--timestamp", "1709568000",
        "--output", str(attest2_path),
    ])

    with open(attest1_path) as f:
        json1 = f.read()
    with open(attest2_path) as f:
        json2 = f.read()

    if json1 == json2:
        print("    Determinism: PASS (byte-identical attestations)")
        results["steps"]["determinism"] = "ok"
    else:
        # Parse and compare field-by-field for better diagnostics
        a2 = json.load(open(attest2_path))
        mismatches = []
        for key in ["layer_readings", "confidence", "coverage_flags", "signature"]:
            if a1.get(key) != a2.get(key):
                mismatches.append(key)
        if mismatches:
            print(f"    Determinism: FAIL — fields differ: {mismatches}")
            results["steps"]["determinism"] = f"FAIL: {mismatches}"
        else:
            print("    Determinism: PASS (key fields match, whitespace may differ)")
            results["steps"]["determinism"] = "ok (fields match)"

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end test with real HuggingFace models")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model names to test (default: all three)",
    )
    parser.add_argument(
        "--cli-path",
        type=str,
        help="Path to got-cli binary",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Don't delete temporary files after test",
    )

    args = parser.parse_args()

    # Filter models if specified
    if args.models:
        specs = [m for m in DEFAULT_MODELS if m.name in args.models]
        if not specs:
            print(f"No matching models found. Available: {[m.name for m in DEFAULT_MODELS]}")
            sys.exit(1)
    else:
        specs = DEFAULT_MODELS

    cli = find_cli(args.cli_path)
    print(f"Using got-cli: {cli}")

    workspace = Path(tempfile.mkdtemp(prefix="got_test_"))
    print(f"Workspace: {workspace}")

    all_results = []
    failures = []

    for spec in specs:
        print(f"\n{'='*60}")
        print(f"TESTING: {spec.name} ({spec.arch})")
        print(f"  hidden_dim={spec.hidden_dim}, layers={spec.num_layers}, test_layers={spec.test_layers}")
        print(f"{'='*60}")

        try:
            result = test_model(spec, cli, workspace)
            all_results.append(result)

            failed_steps = [k for k, v in result["steps"].items() if "FAIL" in str(v)]
            if failed_steps:
                failures.append((spec.name, failed_steps))
        except Exception as e:
            print(f"\n  FAILED: {e}", file=sys.stderr)
            failures.append((spec.name, [str(e)]))
            all_results.append({"model": spec.name, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "PASS" if "error" not in r and all("FAIL" not in str(v) for v in r.get("steps", {}).values()) else "FAIL"
        print(f"  {r['model']:40s} {status}")
        if "steps" in r:
            for step, val in r["steps"].items():
                mark = "✓" if "FAIL" not in str(val) else "✗"
                print(f"    {mark} {step}: {val}")
        if "error" in r:
            print(f"    ✗ {r['error']}")

    if not args.keep_artifacts:
        shutil.rmtree(workspace, ignore_errors=True)
        print(f"\nCleaned up {workspace}")
    else:
        print(f"\nArtifacts kept at {workspace}")

    if failures:
        print(f"\n{len(failures)} model(s) had failures.")
        sys.exit(1)
    else:
        print(f"\nAll {len(specs)} model(s) passed all steps.")
        sys.exit(0)


if __name__ == "__main__":
    main()
