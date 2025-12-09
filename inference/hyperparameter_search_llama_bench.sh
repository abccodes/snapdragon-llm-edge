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
RESULTS_CSV="${OUT_DIR}/results.csv"

########################################
# Hyperparameter spaces (aligned with LongBench)
########################################

# Mode (for logging only)
MODES=("CPU")

# Models
MODELS=(
  "Llama-3.2-1B-Instruct-Q4_0.gguf"
  "mobilellm-r1.5-950M_q4_0.gguf"
  "qwen2.5-3b-instruct-q4_k_m.gguf"
  "qwen2-7b-tinytron-Q4_K_M.gguf"
  "minicpm-3b-openhermes-2.5-v2.Q4_K_M.gguf"
  "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
  "deepseek-r1-1.5b-bf16.gguf"
  "gemma-3n-5b-e2b-bf16_q4_0.gguf"
  "granite-3.3-8b-instruct-Q4_K_M.gguf"
  "phi-4-mini-reasoning-4b-bf16.gguf"
)

# Sampling-ish params (logged only)
TEMPS=(0.4)
REPEAT_PENALTIES=(1.1 1.2)
TOP_PS=(1.0)
TOP_KS=(30)
MIN_PS=(0.15)

# “Context sizes” (only used to keep p+n reasonable)
CTX_SIZES=(512 1024 2048 4096)

# Batch sizes (llama-bench -b)
BATCH_SIZES_CPU=(128 256)

# Micro-batch (llama-bench -ub)
UBATCH_SIZES=(32 64 128 256)

# Threads
THREADS=(8)

# Offload layers (0 = CPU-only)
NGL_VALUES=(0)

# KV cache quantization
KV_CACHE_CTK_VALUES=("f16")
KV_CACHE_CTV_VALUES=("q8_0")

# Flash attention (llama-bench: -fa 0/1)
FLASH_ATTN_VALUES=(1)

# “Keep” (sinks) – logged only
KEEP_VALUES=(4)

# Poll level
POLL_LEVELS=(30)

# Mmap (llama-bench: -mmp 0/1)
USE_MMAP_VALUES=(0)

# Split mode
SPLIT_MODES=("layer")

# Token limit (LongBench) – logged only
TOKEN_LIMITS=(750)

# Bench shapes (must be ≤ chosen “ctx”)
PROMPT_TOKENS=(512 1024 2048)
GEN_TOKENS=(256 512)

########################################
# Helpers
########################################

rand_choice() {
  local arr_name="$1[@]"
  local arr=( "${!arr_name}" )
  local n="${#arr[@]}"
  local idx=$(( RANDOM % n ))
  echo "${arr[$idx]}"
}

# Parse llama-bench table to get:
#  - prefill_tps from the 'pp*' row
#  - decode_tps from the 'tg*' row
parse_bench_tps() {
  local logfile="$1"
  local prefill="0.0"
  local decode="0.0"

  if grep -q "|.*pp[0-9]" "$logfile"; then
    prefill=$(grep "|.*pp[0-9]" "$logfile" | tail -1 | \
      awk -F'|' '{
        gsub(/^[ \t]+|[ \t]+$/,"",$9);
        split($9,a,"±");
        gsub(/^[ \t]+|[ \t]+$/,"",a[1]);
        print a[1]
      }')
  fi

  if grep -q "|.*tg[0-9]" "$logfile"; then
    decode=$(grep "|.*tg[0-9]" "$logfile" | tail -1 | \
      awk -F'|' '{
        gsub(/^[ \t]+|[ \t]+$/,"",$9);
        split($9,a,"±");
        gsub(/^[ \t]+|[ \t]+$/,"",a[1]);
        print a[1]
      }')
  fi

  echo "${decode} ${prefill}"
}

########################################
# Sanity check + CSV header
########################################

if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb not found in PATH."
  exit 1
fi

echo "== Starting RANDOM llama-bench search (CPU-only) =="
echo "Results dir: ${OUT_DIR}"
echo "Trials:      ${N_TRIALS}"
echo

# CSV header
echo "model,decoding_tps,prefill_tps,run_id,mode,temperature,repeat_penalty,top_p,top_k,min_p,ctx_size,keep,batch_size,ubatch_size,threads,ngl,ctk,ctv,flash_attn,poll_level,use_mmap,split_mode,token_limit,prompt_tokens,gen_tokens,run_tag,log_file" > "${RESULTS_CSV}"

########################################
# Main random search loop
########################################

for ((i = 1; i <= N_TRIALS; i++)); do
  echo "=== Trial ${i}/${N_TRIALS} ==="

  MODE="$(rand_choice MODES)"
  MODEL="$(rand_choice MODELS)"

  TEMP="$(rand_choice TEMPS)"
  RP="$(rand_choice REPEAT_PENALTIES)"
  TOP_P="$(rand_choice TOP_PS)"
  TOP_K="$(rand_choice TOP_KS)"
  MIN_P="$(rand_choice MIN_PS)"

  CTX="$(rand_choice CTX_SIZES)"
  BATCH_CPU="$(rand_choice BATCH_SIZES_CPU)"
  UBATCH="$(rand_choice UBATCH_SIZES)"

  T="$(rand_choice THREADS)"
  NGL="$(rand_choice NGL_VALUES)"

  KV_CTK="$(rand_choice KV_CACHE_CTK_VALUES)"
  KV_CTV="$(rand_choice KV_CACHE_CTV_VALUES)"

  FA="$(rand_choice FLASH_ATTN_VALUES)"
  KEEP="$(rand_choice KEEP_VALUES)"

  POLL="$(rand_choice POLL_LEVELS)"
  MMAP="$(rand_choice USE_MMAP_VALUES)"
  SPLIT="$(rand_choice SPLIT_MODES)"

  TOKEN_LIMIT="$(rand_choice TOKEN_LIMITS)"

  P_LEN="$(rand_choice PROMPT_TOKENS)"
  G_LEN="$(rand_choice GEN_TOKENS)"

  # Keep p + n ≤ “ctx”
  if (( P_LEN + G_LEN > CTX )); then
    G_LEN=$(( CTX - P_LEN ))
    if (( G_LEN < 1 )); then
      P_LEN=$(( CTX / 2 ))
      G_LEN=$(( CTX - P_LEN ))
    fi
  fi

  RUN_TAG="trial${i}_$(basename "${MODEL}" .gguf)_mode${MODE}_T${TEMP}_P${TOP_P}_K${TOP_K}_minp${MIN_P}_C${CTX}_Bcpu${BATCH_CPU}_thr${T}_ngl${NGL}_pp${P_LEN}_tg${G_LEN}_fa${FA}"
  LOG_FILE="${OUT_DIR}/${RUN_TAG}.log"

  {
    echo "===== ${RUN_TAG} ====="
    echo "MODEL         : ${MODEL}"
    echo "MODE          : ${MODE}"
    echo "TEMP          : ${TEMP}"
    echo "TOP_P         : ${TOP_P}"
    echo "TOP_K         : ${TOP_K}"
    echo "MIN_P         : ${MIN_P}"
    echo "REPEAT_PEN    : ${RP}"
    echo "CTX_SIZE_LOG  : ${CTX}"
    echo "BATCH_CPU     : ${BATCH_CPU}"
    echo "UBATCH        : ${UBATCH}"
    echo "THREADS       : ${T}"
    echo "NGL           : ${NGL}"
    echo "KV_CTK        : ${KV_CTK}"
    echo "KV_CTV        : ${KV_CTV}"
    echo "FLASH_ATT     : ${FA}"
    echo "KEEP (log)    : ${KEEP}"
    echo "POLL_LEVEL    : ${POLL}"
    echo "USE_MMAP      : ${MMAP}"
    echo "SPLIT_MODE    : ${SPLIT}"
    echo "TOKEN_LIMIT   : ${TOKEN_LIMIT}"
    echo "PROMPT/GEN    : ${P_LEN}/${G_LEN}"
    echo
  } | tee -a "${RESULTS_LOG}"

  # Run llama-bench on device with only supported flags
  adb shell "
cd ${REMOTE_LLAMA_DIR} || exit 1

export LD_LIBRARY_PATH=${REMOTE_LIB_DIR}
export ADSP_LIBRARY_PATH=${REMOTE_LIB_DIR}

${REMOTE_BIN} \
  -m ${REMOTE_MODEL_DIR}/${MODEL} \
  -p ${P_LEN} \
  -n ${G_LEN} \
  -t ${T} \
  -b ${BATCH_CPU} \
  -ub ${UBATCH} \
  -ctk ${KV_CTK} \
  -ctv ${KV_CTV} \
  -fa ${FA} \
  -dev none \
  -ngl ${NGL} \
  -sm ${SPLIT} \
  -mmp ${MMAP} \
  --poll ${POLL}
" | tee "${LOG_FILE}" -a "${RESULTS_LOG}"

  {
    echo
    echo "===== END ${RUN_TAG} ====="
    echo
  } | tee -a "${RESULTS_LOG}"

  # Parse prefill/decoding token speeds from the log
  TPS_LINE="$(parse_bench_tps "${LOG_FILE}")"
  DEC_TPS="${TPS_LINE%% *}"
  PREF_TPS="${TPS_LINE#* }"

  # Append row to CSV
  echo "${MODEL},${DEC_TPS},${PREF_TPS},${i},${MODE},${TEMP},${RP},${TOP_P},${TOP_K},${MIN_P},${CTX},${KEEP},${BATCH_CPU},${UBATCH},${T},${NGL},${KV_CTK},${KV_CTV},${FA},${POLL},${MMAP},${SPLIT},${TOKEN_LIMIT},${P_LEN},${G_LEN},${RUN_TAG},${LOG_FILE}" >> "${RESULTS_CSV}"

done

echo "== Done =="
echo "All logs in: ${OUT_DIR}"
echo "CSV results: ${RESULTS_CSV}"

