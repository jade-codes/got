#!/bin/bash
set -e
source ~/.cargo/env 2>/dev/null || true
cd /mnt/c/Users/Public/got
fuser -k 3000/tcp 2>/dev/null || true
sleep 1
exec ./target/debug/got-web --geometry data/models/gpt2.gotue --vocab data/models/gpt2-vocab.json
