#!/usr/bin/env bash
set -euo pipefail

########################################
# Bench config (EDIT THESE)
########################################

# Model (filename AS IT EXISTS ON PHONE)
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
REMOTE_MODEL_DIR="/data/local/tmp/gguf"
REMOTE_MODEL_PATH="${REMOTE_MODEL_DIR}/${MODEL_NAME}"

# llama.cpp install on device
REMOTE_LLAMA_DIR="/data/local/tmp/llama.cpp"
REMOTE_BIN="${REMOTE_LLAMA_DIR}/bin/llama-bench"
REMOTE_LIB_DIR="${REMOTE_LLAMA_DIR}/lib"

# Hardware / execution params
N_THREADS=6
N_GPU_LAYERS=99

# Bench shape
PROMPT_TOKENS=256
GEN_TOKENS=128

# Logging
OUT_DIR="./bench_logs"
mkdir -p "${OUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/bench_${TIMESTAMP}_${MODEL_NAME%.gguf}.log"

########################################
# Sanity checks
########################################

if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb not found in PATH."
  exit 1
fi

echo "== Bench config =="
echo "  MODEL_NAME      : ${MODEL_NAME}"
echo "  REMOTE_MODEL    : ${REMOTE_MODEL_PATH}"
echo "  PROMPT_TOKENS   : ${PROMPT_TOKENS}"
echo "  GEN_TOKENS      : ${GEN_TOKENS}"
echo "  N_THREADS       : ${N_THREADS}"
echo "  N_GPU_LAYERS    : ${N_GPU_LAYERS}"
echo "  LOG_FILE        : ${LOG_FILE}"
echo

########################################
# Run llama-bench on device
########################################

echo "== Running llama-bench on device =="

adb shell "
cd ${REMOTE_LLAMA_DIR} || exit 1

export LD_LIBRARY_PATH=${REMOTE_LIB_DIR}
export ADSP_LIBRARY_PATH=${REMOTE_LIB_DIR}

${REMOTE_BIN} \
  -m ${REMOTE_MODEL_PATH} \
  -p ${PROMPT_TOKENS} \
  -n ${GEN_TOKENS} \
  -t ${N_THREADS} \
  -ngl ${N_GPU_LAYERS}
" | tee "${LOG_FILE}"

EOF

echo
echo "== Done =="
echo "Bench output saved to: ${LOG_FILE}"

