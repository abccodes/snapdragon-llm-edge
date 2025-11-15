#!/bin/bash
# run-cli-mac.sh — local Mac wrapper around llama.cpp

# Resolve paths relative to this script
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$HERE/../llama.cpp"
BIN="$LLAMA_DIR/build-mac/bin/llama-cli"
MODEL="$LLAMA_DIR/models/Llama-3.2-1B-Instruct-Q4_0.gguf"

if [ ! -x "$BIN" ]; then
  echo "ERROR: llama-cli not found at $BIN" >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

# Use Metal + decent defaults; no flash-attention for now
"$BIN" \
  -m "$MODEL" \
  -t 4 \
  --ctx-size 32768 \
  --batch-size 1 \
  --no-display-prompt \
  --temp 1.0 \
  --seed 42 \
  -ngl 0 \
  "$@"

