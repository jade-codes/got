#!/usr/bin/env python3
"""
Check which value terms exist as single tokens in GPT-2's vocabulary,
and extract message embeddings from GPT-2 for the demo conversation.

Outputs:
  - data/models/gpt2-term-analysis.json  (term → token mapping + coverage)
  - data/models/gpt2-message-embeddings.json  (per-message 768-d vectors)
"""
import json
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
EMBEDDINGS_FILE = Path("data/demo/embeddings.json")
CONVERSATION_FILE = Path("data/demo/demo_conversation.json")
VOCAB_FILE = Path("data/models/gpt2-vocab.json")
OUTPUT_TERMS = Path("data/models/gpt2-term-analysis.json")
OUTPUT_MSG_EMB = Path("data/models/gpt2-message-embeddings.json")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

# Get unembedding matrix
ue_weight = model.lm_head.weight.detach()  # (50257, 768)

# Load vocabulary
with open(VOCAB_FILE) as f:
    vocab = json.load(f)

# Build reverse lookup: lowercase stripped token → index
# GPT-2 BPE tokens have Ġ prefix for space-preceded tokens
token_to_idx = {}
for idx, tok_str in enumerate(vocab):
    # Try both with and without Ġ prefix
    clean = tok_str.replace("Ġ", "").lower().strip()
    if clean and clean not in token_to_idx:
        token_to_idx[clean] = idx

# === PART 1: Check value terms ===
print("\n=== Value Term Analysis ===")
with open(EMBEDDINGS_FILE) as f:
    embeddings_data = json.load(f)
terms = sorted(embeddings_data.keys())

term_analysis = {}
found = 0
for term in terms:
    key = term.lower().strip()
    if key in token_to_idx:
        idx = token_to_idx[key]
        vec = ue_weight[idx].tolist()
        norm = torch.norm(ue_weight[idx]).item()
        term_analysis[term] = {
            "token_index": idx,
            "token_string": vocab[idx],
            "found": True,
            "norm": round(norm, 4),
        }
        found += 1
        print(f"  ✓ {term:20s} → [{idx}] {repr(vocab[idx]):20s}  norm={norm:.4f}")
    else:
        # Try with Ġ prefix explicitly
        alt_key = "Ġ" + key
        alt_idx = None
        for i, v in enumerate(vocab):
            if v.lower() == alt_key.lower():
                alt_idx = i
                break
        if alt_idx is not None:
            vec = ue_weight[alt_idx].tolist()
            norm = torch.norm(ue_weight[alt_idx]).item()
            term_analysis[term] = {
                "token_index": alt_idx,
                "token_string": vocab[alt_idx],
                "found": True,
                "norm": round(norm, 4),
            }
            found += 1
            print(f"  ✓ {term:20s} → [{alt_idx}] {repr(vocab[alt_idx]):20s}  norm={norm:.4f}")
        else:
            term_analysis[term] = {"found": False, "token_string": None, "token_index": None}
            print(f"  ✗ {term:20s} — not a single token")

print(f"\nCoverage: {found}/{len(terms)} terms found as single tokens")

with open(OUTPUT_TERMS, "w") as f:
    json.dump(term_analysis, f, indent=2)
print(f"Term analysis written to {OUTPUT_TERMS}")


# === PART 2: Extract message embeddings ===
print("\n=== Message Embedding Extraction ===")

# Check if conversation file exists
if not CONVERSATION_FILE.exists():
    # Try loading from the demo module's JSON
    print(f"  {CONVERSATION_FILE} not found, checking for alternative...")
    # The demo conversation is embedded in the Rust code, let's check for it
    alt = Path("data/demo/store/index.json")
    if alt.exists():
        print(f"  Using store index at {alt}")

# Load conversation
with open(CONVERSATION_FILE) as f:
    conversation = json.load(f)

messages = conversation.get("messages", conversation)
if isinstance(messages, dict):
    messages = messages.get("messages", [])

print(f"  Found {len(messages)} messages")

message_embeddings = []
for i, msg in enumerate(messages):
    text = msg.get("text", msg.get("content", ""))
    speaker = msg.get("speaker", msg.get("role", "unknown"))

    # Tokenize and run through model
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get last hidden layer, mean-pool over token positions
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, 768)
    mean_pooled = last_hidden[0].mean(dim=0)  # (768,)

    norm = torch.norm(mean_pooled).item()
    message_embeddings.append({
        "turn": i,
        "speaker": speaker,
        "text": text[:80] + ("..." if len(text) > 80 else ""),
        "embedding": mean_pooled.tolist(),
        "norm": round(norm, 4),
    })
    print(f"  Turn {i:2d} [{speaker:10s}] norm={norm:.6f}  tokens={inputs['input_ids'].shape[1]}")

with open(OUTPUT_MSG_EMB, "w") as f:
    json.dump(message_embeddings, f)
print(f"\nMessage embeddings written to {OUTPUT_MSG_EMB}")
print(f"Embedding dimension: {len(message_embeddings[0]['embedding'])}")
