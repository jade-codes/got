# Activation Extraction

Extract residual-stream activations and the unembedding matrix from a HuggingFace transformer model into the binary formats consumed by `got-cli`.

## Dependencies

```bash
pip install torch transformers
```

For GPU inference, install the appropriate CUDA version of PyTorch — see https://pytorch.org/get-started/locally/.

## Usage

```bash
python extract_activations.py \
    --model meta-llama/Llama-3-8B \
    --input "The cat sat on the mat" \
    --layers 12 18 24 \
    --output-activations activations.gotact \
    --output-unembedding unembedding.gotue
```

### Arguments

| Flag | Required | Description |
|---|---|---|
| `--model` | Yes | HuggingFace model name or local path (e.g. `meta-llama/Llama-3-8B`) |
| `--input` | Yes | Input text to run through the model |
| `--layers` | Yes | Space-separated layer indices (0-indexed) to extract activations from |
| `--output-activations` | No | Output path for activations (default: `activations.gotact`) |
| `--output-unembedding` | No | Output path for unembedding matrix (default: `unembedding.gotue`) |
| `--device` | No | Device: `auto`, `cpu`, `cuda`, `cuda:0`, etc. (default: `auto`) |
| `--dtype` | No | Model precision: `float32`, `float16`, `bfloat16` (default: `float32`) |
| `--trust-remote-code` | No | Allow remote code execution from model repo (default: off, security risk) |

### Choosing layers

For a 32-layer model, good defaults are the early-middle, middle, and late-middle layers:

```
--layers 8 16 24
```

For a 40-layer model (e.g. LLaMA-3-8B):

```
--layers 12 18 24 30
```

## Output formats

### `.gotact` — Activations

Binary format containing residual-stream activations at each requested layer for every token position. See [PLAN.md §6.1](../PLAN.md) for the byte-level specification.

```
Magic:          "GOTA" (4 bytes)
Version:        u16 LE (1)
Model ID:       length-prefixed UTF-8
Precision:      u8 tag (0=fp32, 1=fp16, 2=bf16, 3=int8)
hidden_dim:     u32 LE
num_layers:     u32 LE
num_positions:  u32 LE
[per layer × per position: layer_index u32 + token_position u32 + hidden_dim × f32 LE]
```

### `.gotue` — Unembedding matrix

Binary format containing the model's unembedding matrix `U ∈ ℝ^{V × d}` (row-major). See [PLAN.md §6.2](../PLAN.md) for the byte-level specification.

```
Magic:          "GOTU" (4 bytes)
Version:        u16 LE (1)
vocab_size:     u32 LE
hidden_dim:     u32 LE
data:           V × d × f32 LE (row-major)
```

### `.labels` — Label stub

A text file with one label per line (`0` or `1`), one per token position. The script writes all zeros as a stub — **you must edit this with real labels before training probes**.

## End-to-end pipeline

```bash
# 1. Extract activations
python scripts/extract_activations.py \
    --model meta-llama/Llama-3-8B \
    --input "The cat sat on the mat" \
    --layers 12 18 24 \
    --output-activations data/activations.gotact \
    --output-unembedding data/unembedding.gotue

# 2. Edit labels (one 0 or 1 per token position)
vim data/activations.labels

# 3. Generate signing key
cargo run -p got-cli -- keygen --output data/key

# 4. Train probes (one per layer)
cargo run -p got-cli -- train \
    --activations data/activations.gotact \
    --labels data/activations.labels \
    --unembedding data/unembedding.gotue \
    --layer 12 \
    --dimension "test-value" \
    --output data/probes_layer12.json

# 5. Produce attestation
cargo run -p got-cli -- attest \
    --activations data/activations.gotact \
    --probes data/probes_layer12.json \
    --unembedding data/unembedding.gotue \
    --key data/key \
    --model-id "meta-llama/Llama-3-8B" \
    --output data/attestation.json

# 6. Verify
cargo run -p got-cli -- verify \
    --attestation data/attestation.json \
    --pubkey data/key.pub
```

## Supported models

Any HuggingFace `AutoModelForCausalLM` with a standard architecture should work. Tested architectures:

- LLaMA / LLaMA-2 / LLaMA-3
- Mistral
- GPT-2 / GPT-Neo / GPT-J

The script hooks into `model.model.layers[i]` for residual-stream activations and reads `model.lm_head.weight` for the unembedding matrix. Models with non-standard layer paths may require minor adaptation.

## Precision note

For deterministic attestation, both extraction and probing must use the same precision. The `--dtype` flag controls what precision the model is loaded at. The `.gotact` file records the precision tag so `got-cli` can verify consistency.

**Use `float32` for the PoC** — this avoids precision-related non-determinism from mixed-precision operations.
