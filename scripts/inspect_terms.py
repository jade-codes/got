#!/usr/bin/env python3
import json
d = json.load(open("data/demo/embeddings.json"))
terms = sorted(d.keys())
print(f"{len(terms)} terms:")
for t in terms:
    print(f"  {t} ({len(d[t])}d)")
