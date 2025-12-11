
#!/system/bin/sh
# Base directories
BASEDIR="/data/local/tmp/llama.cpp"
BINDIR="$BASEDIR/bin"
LIBDIR="$BASEDIR/lib"
MODELDIR="$BASEDIR/../gguf"
MODEL="Llama-3.2-1B-Instruct-Q4_0.gguf"

THREADS=8
CTX_SIZE=1024
BATCH_SIZE=128

# Clear old Snapdragon env vars
unset D M GGML_HEXAGON_NDEV GGML_HEXAGON_NHVX GGML_HEXAGON_HOSTBUF GGML_HEXAGON_VERBOSE GGML_HEXAGON_PROFILE GGML_HEXAGON_OPMASK

# Set LD paths
export LD_LIBRARY_PATH="$LIBDIR"
export ADSP_LIBRARY_PATH="$LIBDIR"

# Enable core dumps
ulimit -c unlimited

# Run llama-cli (CPU only: device none, ngl 0)
cd "$BINDIR" || exit 1

./llama-cli \
  -m "$MODELDIR/$MODEL" \
  -t "$THREADS" \
  --ctx-size "$CTX_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --device "none" \
  -ngl 0 \
  --no-mmap \
  -ctk f16 \
  -ctv f16 \
  -fa 1 \
  --poll 30 \
  --temp 0.4 \
  --repeat-penalty 1.1 \
  --top-p 1.0 \
  --top-k 30 \
  --min-p 0.15 \
  --keep 0 \
  "$@"
