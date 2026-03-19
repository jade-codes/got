#!/usr/bin/env python3
"""Save GPT-2 vocabulary as ordered JSON array for UnembeddingLookup."""
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
vocab = [""] * len(tok)
for token_str, idx in tok.get_vocab().items():
    vocab[idx] = token_str

with open("data/models/gpt2-vocab.json", "w") as f:
    json.dump(vocab, f)

print(f"Wrote {len(vocab)} tokens")
print(f"Sample [0..5]: {vocab[:5]}")
print(f"Sample: 'honesty' token check:")
for i, t in enumerate(vocab):
    if "honesty" in t.lower():
        print(f"  [{i}] = {repr(t)}")
