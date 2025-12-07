#!/usr/bin/env bash
set -euo pipefail

########################################
# Global config
########################################

# How many random trials to run
N_TRIALS=200

# Where llama.cpp lives on the phone
REMOTE_LLAMA_DIR="/data/local/tmp/llama.cpp"
REMOTE_LIB_DIR="${REMOTE_LLAMA_DIR}/lib"
REMOTE_BIN="${REMOTE_LLAMA_DIR}/bin/llama-bench"

# Where models live on the phone
REMOTE_MODEL_DIR="/data/local/tmp/gguf"

# Output directory on HOST
OUT_DIR="hyperparam_bench_random_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"
RESULTS_LOG="${OUT_DIR}/bench_runs.log"

########################################
# Hyperparameter spaces (MATCH YOUR TRUTHFUL SETUP)
########################################

# Device / mode
MODES=("CPU")   # we will always run CPU-only for this bench

# System prompts (kept for consistency with TruthfulQA; not used by llama-bench)
SYSTEM_PROMPTS=(
  ""
)

# Sampling parameters
TEMPS=(0.5)
REPEAT_PENALTIES=(1.1)
TOP_PS=(0.9 0.95 1.0)
TOP_KS=(50)

# Context and batch sizes
CTX_SIZES=(512)

BATCH_SIZES_CPU=(128)
BATCH_SIZES_NPU=(256 512 1024)
BATCH_SIZES_GPU=(512 1024 2048)

# For now using same UBatch size as batch size (kept for consistency)
UBATCH_SIZES=(32 64 256 512 1024)

# Hardware settings
THREADS=(8)

# Offload layers (0 = CPU only)
NGL_VALUES=(0)

# KV cache quantization
KV_CACHE_TYPES_CTK=("f16")
KV_CACHE_TYPES_CTV=("f16" "q8_0")

# Flash attention (always ON)
FLASH_ATTN=("on")

# StreamLLM: number of sink tokens (kept for logging consistency)
KEEP_VALUES=(4)

# Context shift (Sliding-window KV)
CONTEXT_SHIFT=(1)

# Performance tuning
POLL_LEVELS=(30)

# Whether to mmap model (kept for logging consistency)
USE_MMAP=(0)

# How to split model across multiple GPUs/NPUs (not used here)
SPLIT_MODES=("none")

# Models ON DEVICE
MODELS=(
   "qwen2-7b-tinytron-Q4_K_M.gguf"
   "LFM2-8B-A1B-Q4_K_M.gguf"
   "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
)
#  "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
#  "Llama-3.2-1B-Instruct-Q4_0.gguf"
#  "TinyLlama-1.1B-Chat-Q4_K_M.gguf"
#  "mobilellm-r1.5-950M_q4_0.gguf"

# Bench shape (not in screenshot; specific to llama-bench)
PROMPT_TOKENS=(256)   # -p
GEN_TOKENS=(128)      # -n


########################################
# Helpers
########################################

rand_choice() {
  # $1 = name of bash array (e.g. MODELS)
  local arr_name="$1[@]"
  local arr=( "${!arr_name}" )
  local n="${#arr[@]}"
  local idx=$(( RANDOM % n ))
  echo "${arr[$idx]}"
}

########################################
# Sanity check
########################################

if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb not found in PATH."
  exit 1
fi

echo "== Starting RANDOM llama-bench search (CPU-only, FlashAttention ON) =="
echo "Results dir: ${OUT_DIR}"
echo "Trials:      ${N_TRIALS}"
echo

########################################
# Main random search loop
########################################

for ((i = 1; i <= N_TRIALS; i++)); do
  echo "=== Trial ${i}/${N_TRIALS} ==="

  MODE="$(rand_choice MODES)"      # will be "CPU"
  MODEL="$(rand_choice MODELS)"

  TEMP="$(rand_choice TEMPS)"
  RP="$(rand_choice REPEAT_PENALTIES)"
  TOP_P="$(rand_choice TOP_PS)"
  TOP_K="$(rand_choice TOP_KS)"

  CTX="$(rand_choice CTX_SIZES)"

  BATCH_CPU="$(rand_choice BATCH_SIZES_CPU)"
  BATCH_NPU="$(rand_choice BATCH_SIZES_NPU)"
  BATCH_GPU="$(rand_choice BATCH_SIZES_GPU)"
  UBATCH="$(rand_choice UBATCH_SIZES)"

  T="$(rand_choice THREADS)"
  NGL="$(rand_choice NGL_VALUES)"          # will be 0 (CPU only)

  KV_CTK="$(rand_choice KV_CACHE_TYPES_CTK)"
  KV_CTV="$(rand_choice KV_CACHE_TYPES_CTV)"

  FA="$(rand_choice FLASH_ATTN)"          # will be "on"
  KEEP="$(rand_choice KEEP_VALUES)"
  CSHIFT="$(rand_choice CONTEXT_SHIFT)"
  POLL="$(rand_choice POLL_LEVELS)"
  MMAP="$(rand_choice USE_MMAP)"
  SPLIT="$(rand_choice SPLIT_MODES)"

  P_LEN="$(rand_choice PROMPT_TOKENS)"
  G_LEN="$(rand_choice GEN_TOKENS)"

  RUN_TAG="trial${i}_$(basename "${MODEL}" .gguf)_mode${MODE}_T${TEMP}_P${TOP_P}_K${TOP_K}_C${CTX}_Bcpu${BATCH_CPU}_thr${T}_ngl${NGL}_pp${P_LEN}_tg${G_LEN}_fa${FA}"
  LOG_FILE="${OUT_DIR}/${RUN_TAG}.log"

  {
    echo "===== ${RUN_TAG} ====="
    echo "MODEL        : ${MODEL}"
    echo "MODE         : ${MODE}"
    echo "TEMP         : ${TEMP}"
    echo "TOP_P        : ${TOP_P}"
    echo "TOP_K        : ${TOP_K}"
    echo "REPEAT_PEN   : ${RP}"
    echo "CTX_SIZE     : ${CTX}"
    echo "BATCH_CPU    : ${BATCH_CPU}"
    echo "BATCH_NPU    : ${BATCH_NPU}"
    echo "BATCH_GPU    : ${BATCH_GPU}"
    echo "UBATCH       : ${UBATCH}"
    echo "THREADS      : ${T}"
    echo "NGL          : ${NGL}"
    echo "KV_CTK       : ${KV_CTK}"
    echo "KV_CTV       : ${KV_CTV}"
    echo "FLASH_ATTN   : ${FA}"
    echo "KEEP         : ${KEEP}"
    echo "CONTEXT_SHIFT: ${CSHIFT}"
    echo "POLL_LEVEL   : ${POLL}"
    echo "USE_MMAP     : ${MMAP}"
    echo "SPLIT_MODE   : ${SPLIT}"
    echo "PROMPT/GEN   : ${P_LEN}/${G_LEN}"
    echo
  } | tee -a "${RESULTS_LOG}"

  # Actual llama-bench call on device
  # CPU-only: --device none, NGL=0
  adb shell "
cd ${REMOTE_LLAMA_DIR} || exit 1

export LD_LIBRARY_PATH=${REMOTE_LIB_DIR}
export ADSP_LIBRARY_PATH=${REMOTE_LIB_DIR}

# Force FlashAttention ON (env flag is safer than guessing CLI flag)
export LLAMA_FLASH_ATTENTION=1

${REMOTE_BIN} \
  -m ${REMOTE_MODEL_DIR}/${MODEL} \
  -p ${P_LEN} \
  -n ${G_LEN} \
  -t ${T} \
  -ngl 0 \
  --device none
" | tee "${LOG_FILE}" -a "${RESULTS_LOG}"

  {
    echo
    echo "===== END ${RUN_TAG} ====="
    echo
  } | tee -a "${RESULTS_LOG}"

done

echo "== Done =="
echo "All logs in: ${OUT_DIR}"

