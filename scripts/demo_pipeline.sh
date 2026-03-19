#!/usr/bin/env bash
# ==========================================================================
# demo_pipeline.sh — Geometry of Trust end-to-end demo
#
# keygen → synthetic data → train probe → attest → verify → web UI
#
# Usage:
#   ./scripts/demo_pipeline.sh              # full pipeline + web server
#   ./scripts/demo_pipeline.sh --no-web     # pipeline only
#   ./scripts/demo_pipeline.sh --clean      # wipe data/demo and re-run
# ==========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/demo"
STORE="$DATA/store"
CLI="$ROOT/target/debug/got-cli"
WEB="$ROOT/target/debug/got-web"

if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning $DATA..."
    rm -rf "$DATA"
    shift
fi

mkdir -p "$DATA"

# -- Build ----------------------------------------------------------------
echo "Building..."
cargo build -p got-cli -p got-web 2>&1

# -- Keys -----------------------------------------------------------------
if [[ ! -f "$DATA/key" ]]; then
    echo "Generating keypair..."
    "$CLI" keygen --output "$DATA/key"
fi

# -- Synthetic data -------------------------------------------------------
if [[ ! -f "$DATA/model.gotue" ]]; then
    echo "Generating synthetic data..."
    python3 "$ROOT/scripts/generate_synthetic_data.py" --out "$DATA"
fi

# -- Train ----------------------------------------------------------------
if [[ ! -f "$DATA/probe_layer0.json" ]]; then
    echo "Training probe..."
    "$CLI" train \
        --activations "$DATA/activations.gotact" \
        --labels "$DATA/labels.txt" \
        --unembedding "$DATA/model.gotue" \
        --layer 0 --dimension truthful \
        --lr 0.001 --epochs 50 \
        --output "$DATA/probe_layer0.json"
fi

# -- Attest ---------------------------------------------------------------
echo "Attesting..."
"$CLI" attest \
    --activations "$DATA/activations.gotact" \
    --probes "$DATA/probe_layer0.json" \
    --unembedding "$DATA/model.gotue" \
    --key "$DATA/key" \
    --model-id demo-model-v1 \
    --corpus-version synthetic-v1 \
    --timestamp 1742000000 \
    --store-dir "$STORE" \
    --output "$DATA/attestation.json"

# -- Verify ---------------------------------------------------------------
echo "Verifying..."
"$CLI" verify \
    --attestation "$DATA/attestation.json" \
    --pubkey "$DATA/key.pub"

echo ""
echo "Done. Outputs in $DATA/"
echo "  attestation.json, model.gotue, embeddings.json, store/"
echo ""

# -- Web ------------------------------------------------------------------
if [[ "${1:-}" != "--no-web" ]]; then
    echo "Starting got-web (--store-dir $STORE)..."
    echo "Upload embeddings.json + model.gotue via the web UI."
    exec "$WEB" --store-dir "$STORE"
fi
