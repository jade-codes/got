#!/usr/bin/env python3
"""Sweep layers to find where value concepts separate best."""

import json, math, subprocess, sys, time, requests, signal
import numpy as np

DESCRIPTIONS = {
    'compassion': 'Caring deeply about the suffering of others and acting to alleviate it',
    'cruelty': 'Deliberately causing pain or suffering to others without remorse',
    'fairness': 'Believing people deserve equal treatment and just outcomes',
    'oppression': 'Systematically restricting the freedom or rights of others',
    'honesty': 'Consistently telling the truth and being transparent',
    'secrecy': 'Deliberately withholding information to manipulate or control others',
    'courage': 'Standing firm on principles despite personal risk',
    'cowardice': 'Abandoning principles to avoid personal discomfort',
    'freedom': 'Valuing individual autonomy and the right to make ones own choices',
    'transparency': 'Making decisions and processes visible and accountable',
    'tradition': 'Respecting established customs and preserving continuity',
    'innovation': 'Embracing new ideas and approaches to improve on the status quo',
    'wisdom': 'Applying deep understanding and experience to make sound judgments',
}

ANTONYM_PAIRS = [
    ('compassion', 'cruelty'), ('honesty', 'secrecy'),
    ('fairness', 'oppression'), ('courage', 'cowardice'),
]

LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 35]
MODEL = "Qwen/Qwen3-8B"
PORT = 8100
URL = f"http://localhost:{PORT}"

def start_server(layer):
    # Kill ANY existing activation server on this port
    subprocess.run(["pkill", "-f", "activation_server"], capture_output=True)
    time.sleep(3)  # wait for port to free

    proc = subprocess.Popen(
        [sys.executable, "scripts/activation_server.py",
         "--model", MODEL, "--layer", str(layer),
         "--quantize", "4bit", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        time.sleep(2)
        try:
            r = requests.get(f"{URL}/health", timeout=2)
            if r.ok:
                data = r.json()
                # Verify it's actually the right layer
                if data.get("layer") == layer:
                    return proc
        except:
            pass
    return proc

def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()
    time.sleep(2)  # wait for port to free

def measure(layer):
    embs = {}
    for name, desc in DESCRIPTIONS.items():
        r = requests.post(f"{URL}/hidden_states", json={"text": desc}, timeout=30)
        e = r.json()["hidden_state"]
        norm = math.sqrt(sum(x*x for x in e))
        embs[name] = [x/norm for x in e] if norm > 1e-8 else e

    names = list(embs.keys())
    k = len(names)
    G = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            G[i, j] = sum(a*b for a, b in zip(embs[names[i]], embs[names[j]]))

    ev = sorted(np.linalg.eigvalsh(G), reverse=True)
    s = sum(max(0, v) for v in ev)
    s2 = sum(max(0, v)**2 for v in ev)
    dim_eff = s*s/s2 if s2 > 0 else 1

    cos_all = [G[i, j] for i in range(k) for j in range(i+1, k)]
    antonym_cos = [float(G[names.index(a), names.index(b)]) for a, b in ANTONYM_PAIRS]

    return {
        "dim_eff": round(dim_eff, 2),
        "mean_cos": round(float(np.mean(cos_all)), 4),
        "min_cos": round(float(min(cos_all)), 4),
        "max_cos": round(float(max(cos_all)), 4),
        "spread": round(float(max(cos_all) - min(cos_all)), 4),
        "antonym_mean": round(float(np.mean(antonym_cos)), 4),
    }

print(f"{'Layer':>5} {'dim_eff':>8} {'mean_cos':>9} {'spread':>7} {'antonym':>8}")
print("-" * 45)

results = {}
for layer in LAYERS:
    proc = start_server(layer)
    try:
        m = measure(layer)
        results[layer] = m
        print(f"{layer:>5} {m['dim_eff']:>8.2f} {m['mean_cos']:>9.4f} {m['spread']:>7.4f} {m['antonym_mean']:>8.4f}")
    except Exception as e:
        print(f"{layer:>5} ERROR: {e}")
    finally:
        stop_server(proc)

print()
best = max(results, key=lambda l: results[l]["dim_eff"])
print(f"Best layer: {best} (dim_eff = {results[best]['dim_eff']})")
