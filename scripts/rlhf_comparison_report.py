#!/usr/bin/env python3
"""
RLHF Manifold Collapse Experiment — Report Generator

Reads the raw output from rlhf_comparison.py and produces a formatted
summary comparing base vs instruct models across all measured dimensions.

Usage:
    python scripts/rlhf_comparison_report.py \
        --results data/rlhf_experiment/raw_results.json

Or with direct CLI output files:
    python scripts/rlhf_comparison_report.py \
        --results-dir data/rlhf_experiment
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_collapse_output(text: str) -> Optional[Dict]:
    """Parse got-cli collapse-report output."""
    result = {}
    m = re.search(r"Probes \(k\):\s+(\d+)", text)
    if m:
        result["k"] = int(m.group(1))
    m = re.search(r"dim_eff:\s+([\d.]+)", text)
    if m:
        result["dim_eff"] = float(m.group(1))
    m = re.search(r"dim_eff / k:\s+([\d.]+)", text)
    if m:
        result["ratio"] = float(m.group(1))
    m = re.search(r"Assessment:\s+(.+)", text)
    if m:
        result["assessment"] = m.group(1).strip()

    # Parse eigenvalues
    eigenvalues = []
    for m in re.finditer(r"lambda_\d+ = ([\d.e+-]+)\s+\(([\d.]+)%\)", text):
        eigenvalues.append({"value": float(m.group(1)), "pct": float(m.group(2))})
    if eigenvalues:
        result["eigenvalues"] = eigenvalues

    return result if result else None


def parse_compare_output(text: str) -> Optional[Dict]:
    """Parse got-cli compare output."""
    result = {}
    m = re.search(r"Global distance \(Frobenius\):\s+([\d.e+-]+)", text)
    if m:
        result["global_distance"] = float(m.group(1))
    m = re.search(r"Probe-projected distance:\s+([\d.e+-]+)", text)
    if m:
        result["probe_projected_distance"] = float(m.group(1))

    per_probe = []
    for m in re.finditer(r"\[\d+\]\s+(.+?):\s+([\d.e+-]+)", text):
        per_probe.append({"dimension": m.group(1), "distance": float(m.group(2))})
    if per_probe:
        result["per_probe"] = per_probe

    return result if result else None


def parse_coherence_output(text: str) -> Optional[Dict]:
    """Parse got-cli coherence output."""
    result = {}
    m = re.search(r"Mean:\s+([\d.]+)", text)
    if m:
        result["mean"] = float(m.group(1))
    m = re.search(r"Min:\s+([\d.]+)", text)
    if m:
        result["min"] = float(m.group(1))
    m = re.search(r"Max:\s+([\d.]+)", text)
    if m:
        result["max"] = float(m.group(1))
    m = re.search(r"Positions:\s+(\d+)", text)
    if m:
        result["positions"] = int(m.group(1))

    # Parse per-position scores for variance calculation
    scores = []
    for m in re.finditer(r"\[\s*\d+\]\s+([\d.]+)", text):
        scores.append(float(m.group(1)))
    if scores:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        result["std"] = variance ** 0.5

    violations = []
    for m in re.finditer(r"position (\d+): (.+)", text):
        violations.append({"position": int(m.group(1)), "constraint": m.group(2).strip()})
    if violations:
        result["violations"] = violations

    return result if result else None


def format_report(results: Dict) -> str:
    """Generate the formatted experiment report."""
    lines = []
    lines.append("RLHF Manifold Collapse Experiment")
    lines.append("=" * 40)
    lines.append("")

    # --- Effective Value Dimensionality ---
    base_collapse = {}
    instruct_collapse = {}
    for key, val in results.get("base", {}).items():
        if key.startswith("collapse_"):
            parsed = parse_collapse_output(val)
            if parsed:
                base_collapse[key] = parsed
    for key, val in results.get("instruct", {}).items():
        if key.startswith("collapse_"):
            parsed = parse_collapse_output(val)
            if parsed:
                instruct_collapse[key] = parsed

    if base_collapse or instruct_collapse:
        lines.append("Effective Value Dimensionality (dim_eff):")
        lines.append("-" * 40)
        all_layers = sorted(set(
            list(base_collapse.keys()) + list(instruct_collapse.keys())
        ))
        for layer_key in all_layers:
            layer_label = layer_key.replace("collapse_", "")
            lines.append(f"  {layer_label}:")
            bc = base_collapse.get(layer_key, {})
            ic = instruct_collapse.get(layer_key, {})

            if bc:
                k = bc.get("k", "?")
                de = bc.get("dim_eff", 0)
                lines.append(f"    Base:      {de:.2f} / {k} ({bc.get('ratio', 0) * 100:.0f}% of maximum)")
            if ic:
                k = ic.get("k", "?")
                de = ic.get("dim_eff", 0)
                lines.append(f"    Instruct:  {de:.2f} / {k} ({ic.get('ratio', 0) * 100:.0f}% of maximum)")

            if bc and ic and bc.get("dim_eff") and ic.get("dim_eff"):
                change = (ic["dim_eff"] - bc["dim_eff"]) / bc["dim_eff"] * 100
                direction = "COLLAPSE" if change < -20 else "EXPANSION" if change > 20 else "STABLE"
                lines.append(f"    Change:    {change:+.1f}% <- {direction}")
            lines.append("")
    else:
        lines.append("Effective Value Dimensionality: [no data]")
        lines.append("")

    # --- Value Alignment Distance ---
    compare_data = results.get("compare", {})
    if compare_data:
        lines.append("Value Alignment Distance:")
        lines.append("-" * 40)
        for key, val in sorted(compare_data.items()):
            parsed = parse_compare_output(val)
            if parsed:
                lines.append(f"  {key}:")
                gd = parsed.get("global_distance", 0)
                pd = parsed.get("probe_projected_distance")
                lines.append(f"    Global (Frobenius):  {gd:.6f}")
                if pd is not None:
                    lines.append(f"    Probe-projected:     {pd:.6f}")
                    if gd > 1e-9:
                        ratio = pd / gd
                        marker = " <- value-relevant change exceeds global" if ratio > 2.0 else ""
                        lines.append(f"    Ratio (proj/global): {ratio:.1f}x{marker}")
                if parsed.get("per_probe"):
                    for pp in parsed["per_probe"]:
                        lines.append(f"      {pp['dimension']}: {pp['distance']:.6f}")
                lines.append("")
    else:
        lines.append("Value Alignment Distance: [no data]")
        lines.append("")

    # --- Coherence Scores ---
    base_coherence = {}
    instruct_coherence = {}
    for key, val in results.get("base", {}).items():
        if key.startswith("coherence_"):
            parsed = parse_coherence_output(val)
            if parsed:
                base_coherence[key] = parsed
    for key, val in results.get("instruct", {}).items():
        if key.startswith("coherence_"):
            parsed = parse_coherence_output(val)
            if parsed:
                instruct_coherence[key] = parsed

    if base_coherence or instruct_coherence:
        lines.append("Coherence Score C(h):")
        lines.append("-" * 40)
        all_layers = sorted(set(
            list(base_coherence.keys()) + list(instruct_coherence.keys())
        ))
        for layer_key in all_layers:
            layer_label = layer_key.replace("coherence_", "")
            lines.append(f"  {layer_label}:")
            bc = base_coherence.get(layer_key, {})
            ic = instruct_coherence.get(layer_key, {})

            if bc:
                std_str = f", std={bc['std']:.2f}" if "std" in bc else ""
                lines.append(f"    Base:      mean={bc.get('mean', 0):.2f}{std_str}")
            if ic:
                std_str = f", std={ic['std']:.2f}" if "std" in ic else ""
                lines.append(f"    Instruct:  mean={ic.get('mean', 0):.2f}{std_str}")

            if bc and ic and bc.get("mean") and ic.get("mean"):
                mean_change = (ic["mean"] - bc["mean"]) / bc["mean"] * 100
                lines.append(f"    Mean change: {mean_change:+.1f}%")
                if "std" in bc and "std" in ic and bc["std"] > 1e-9:
                    var_change = (ic["std"] - bc["std"]) / bc["std"] * 100
                    lines.append(f"    Std change:  {var_change:+.1f}%")
            lines.append("")
    else:
        lines.append("Coherence Score C(h): [no data]")
        lines.append("")

    # --- Interpretation ---
    lines.append("Interpretation:")
    lines.append("-" * 40)

    has_collapse = any(
        base_collapse.get(k, {}).get("dim_eff", 0) > instruct_collapse.get(k, {}).get("dim_eff", 0)
        for k in base_collapse
        if k in instruct_collapse
    )
    has_expansion = any(
        base_collapse.get(k, {}).get("dim_eff", 0) < instruct_collapse.get(k, {}).get("dim_eff", 0)
        for k in base_collapse
        if k in instruct_collapse
    )

    if has_collapse and not has_expansion:
        lines.append("  dim_eff decreased after RLHF, consistent with manifold collapse")
        lines.append("  (Conjecture 3). The instruct model's value geometry uses fewer")
        lines.append("  effective dimensions.")
    elif has_expansion and not has_collapse:
        lines.append("  dim_eff INCREASED after RLHF. This is INCONSISTENT with manifold")
        lines.append("  collapse (Conjecture 3). The instruct model uses MORE effective")
        lines.append("  dimensions, suggesting RLHF expanded the value representation.")
    elif has_collapse and has_expansion:
        lines.append("  dim_eff changes are mixed across layers. Some layers show collapse,")
        lines.append("  others show expansion. Conjecture 3 may be layer-dependent.")
    else:
        lines.append("  Insufficient data to assess manifold collapse.")

    lines.append("")
    lines.append("  NOTE: These results should be reported for ALL layers, not cherry-picked.")
    lines.append("  If dim_eff does not decrease, Conjecture 3 is falsified for this model")
    lines.append("  pair and this should be acknowledged.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RLHF Comparison Report Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", type=Path, help="Path to raw_results.json")
    group.add_argument("--results-dir", type=Path, help="Path to experiment output directory")
    args = parser.parse_args()

    if args.results_dir:
        results_path = args.results_dir / "raw_results.json"
    else:
        results_path = args.results

    if not results_path.exists():
        print(f"Error: {results_path} not found", file=sys.stderr)
        print("Run rlhf_comparison.py first to generate results.", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    report = format_report(results)
    print(report)

    # Also save to file
    report_path = results_path.parent / "report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"(Report also saved to {report_path})")


if __name__ == "__main__":
    main()
