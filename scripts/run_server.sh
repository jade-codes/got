#!/bin/bash
cd /mnt/c/Users/Public/got
source ~/.cargo/env
fuser -k 3000/tcp 2>/dev/null
sleep 1
./target/debug/got-web \
  --geometry data/models/gpt2.gotue \
  --vocab data/models/gpt2-vocab.json \
  --listen 0.0.0.0:3000
