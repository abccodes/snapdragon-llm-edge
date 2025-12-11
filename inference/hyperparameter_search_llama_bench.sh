#!/usr/bin/env bash
set -euo pipefail

N_TRIALS=1
REMOTE_LLAMA_DIR="/data/local/tmp/llama.cpp"
REMOTE_LIB_DIR="${REMOTE_LLAMA_DIR}/lib"
REMOTE_BIN="${REMOTE_LLAMA_DIR}/bin/llama-bench"
REMOTE_MODEL_DIR="/data/local/tmp/gguf"
OUT_DIR="hyperparam_bench_random_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"
RESULTS_LOG="${OUT_DIR}/bench_runs.log"
RESULTS_CSV="${OUT_DIR}/results.csv"

########################################
# Hyperparameter tuning
########################################

MODES=("CPU")

# Models - All working quantized models
MODELS=(
  "Llama-3.2-1B-Instruct-Q4_0.gguf"
#  "llama-3.2-1b-instruct-q4_k_m.gguf"
#  "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
#  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
#  "MobileLLM-R1-360M_Q8_0.gguf"
#  "MobileLLM-R1-950M_Q4_K_M.gguf"
#  "MobileLLM-R1-950M_Q4_K_S.gguf"
#  "MobileLLM-R1-950M_Q5_K_M.gguf"
#  "mobilellm-r1.5-950M_q4_0.gguf"
#  "TinyLlama-1.1B-Chat-Q4_K_M.gguf"
#  "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
#  "deepseek-r1-1.5b-bf16.gguf"
#  "deepseek-r1-7b-bf16_q4_0.gguf"
#  "qwen2.5-3b-instruct-q4_k_m.gguf"
#  "qwen2-7b-tinytron-Q4_K_M.gguf"
#  "Qwen3-4B-Q4_K_M.gguf"
#  "minicpm-3b-openhermes-2.5-v2.Q4_K_M.gguf"
#  "gemma-3n-5b-e2b-bf16_q4_0.gguf"
#  "granite-3.3-8b-instruct-Q4_K_M.gguf"
#  "phi-4-mini-reasoning-4b-bf16.gguf"
#  "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
#  "LFM2-2.6B.Q4_K_M.gguf"
#  "LFM2-8B-A1B-Q4_K_M.gguf"
)

TEMPS=(0.4)
REPEAT_PENALTIES=(1.1)
TOP_PS=(1.0)
TOP_KS=(30)
MIN_PS=(0.15)
TOKEN_LIMITS=(700)
CTX_SIZES=(1024)
BATCH_SIZES_CPU=(128)
UBATCH_SIZES=(128)
THREADS=(8)
NGL_VALUES=(0)
KV_CACHE_CTK_VALUES=("f16")
KV_CACHE_CTV_VALUES=("f16")
FLASH_ATTN_VALUES=(1)
POLL_LEVELS=(30)
USE_MMAP_VALUES=(0)
SPLIT_MODES=("none")
DRY_MULTIPLIER=(1.0)
FREQUENCY_PENALTY=(0.1)
PRESENCE_PENALTY=(0.4)

# Setting to 0 because llamabench doesnt support streamllm implementation
# If setting to 4 initial tokens just makes speed worse. This is because implemeentation is not fully supported
# Context shift is also not supported so for most accurate speed comparison setting to 0 makes the most sense
KEEP_VALUES=(0)

# Llamabench specific benchmarks - use default
PROMPT_TOKENS=(512)
GEN_TOKENS=(128)

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

# Parse llama-bench table to get token speeds
# This function handles variable table formats (with/without KV cache columns)
parse_bench_tps() {
  local logfile="$1"
  local prefill="0.0"
  local decode="0.0"

  if grep -q "|.*pp[0-9]" "$logfile"; then
    # Find the column with "±" (the t/s column)
    prefill=$(grep "|.*pp[0-9]" "$logfile" | tail -1 | \
      awk -F'|' '{
        # Loop through fields to find the one with ± (the t/s column)
        for(i=1; i<=NF; i++) {
          if($i ~ /±/) {
            gsub(/^[ \t]+|[ \t]+$/,"",$i);
            split($i,a,/±/);
            gsub(/^[ \t]+|[ \t]+$/,"",a[1]);
            print a[1];
            exit;
          }
        }
      }')
  fi

  if grep -q "|.*tg[0-9]" "$logfile"; then
    decode=$(grep "|.*tg[0-9]" "$logfile" | tail -1 | \
      awk -F'|' '{
        # Loop through fields to find the one with ± (the t/s column)
        for(i=1; i<=NF; i++) {
          if($i ~ /±/) {
            gsub(/^[ \t]+|[ \t]+$/,"",$i);
            split($i,a,/±/);
            gsub(/^[ \t]+|[ \t]+$/,"",a[1]);
            print a[1];
            exit;
          }
        }
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

echo "== Validating models on device =="
echo "Checking which models exist in ${REMOTE_MODEL_DIR}..."
echo

# Get list of actual models on device
AVAILABLE_MODELS=$(adb shell "ls ${REMOTE_MODEL_DIR}/*.gguf 2>/dev/null" | sed 's/.*\///' | tr -d '\r')

if [ -z "$AVAILABLE_MODELS" ]; then
  echo "ERROR: No .gguf models found in ${REMOTE_MODEL_DIR}"
  echo "Please check the path and ensure models are on the device."
  exit 1
fi

echo "Available models:"
echo "$AVAILABLE_MODELS" | while read -r model; do
  echo "  - $model"
done
echo

# Filter MODELS array to only include available models
VALID_MODELS=()
for model in "${MODELS[@]}"; do
  if echo "$AVAILABLE_MODELS" | grep -q "^${model}$"; then
    VALID_MODELS+=("$model")
  else
    echo "Warning: Model not found on device: $model"
  fi
done

if [ ${#VALID_MODELS[@]} -eq 0 ]; then
  echo "ERROR: None of the specified models were found on the device."
  echo "Update the MODELS array in the script to match your actual filenames."
  exit 1
fi

echo "Will use ${#VALID_MODELS[@]} valid models:"
for model in "${VALID_MODELS[@]}"; do
  echo "  ✓ $model"
done
echo

# Replace MODELS with VALID_MODELS
MODELS=("${VALID_MODELS[@]}")

echo "== Starting RANDOM llama-bench search (CPU-only) =="
echo "Results dir: ${OUT_DIR}"
echo "Trials:      ${N_TRIALS}"
echo

echo "model,decoding_tps,prefill_tps,run_id,mode,temperature,repeat_penalty,top_p,top_k,min_p,ctx_size,keep,batch_size,ubatch_size,threads,ngl,ctk,ctv,flash_attn,poll_level,use_mmap,split_mode,dry_multiplier,frequency_penalty,presence_penalty,token_limit,prompt_tokens,gen_tokens,run_tag,log_file" > "${RESULTS_CSV}"

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
  DRY_MULT="$(rand_choice DRY_MULTIPLIER)"
  FREQ_PEN="$(rand_choice FREQUENCY_PENALTY)"
  PRES_PEN="$(rand_choice PRESENCE_PENALTY)"

  if [ "$FA" -eq 0 ]; then
    KV_CTK="f16"
    KV_CTV="f16"
  fi

  TOKEN_LIMIT="$(rand_choice TOKEN_LIMITS)"

  P_LEN="$(rand_choice PROMPT_TOKENS)"
  G_LEN="$(rand_choice GEN_TOKENS)"

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
    echo "DRY_MULT      : ${DRY_MULT}"
    echo "FREQ_PEN      : ${FREQ_PEN}"
    echo "PRES_PEN      : ${PRES_PEN}"
    echo "TOKEN_LIMIT   : ${TOKEN_LIMIT}"
    echo "PROMPT/GEN    : ${P_LEN}/${G_LEN}"
    echo
  } | tee -a "${RESULTS_LOG}"

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
  -ngl ${NGL} \
  -sm ${SPLIT} \
  -nkvo 1 \
  -nopo 1 \
  -mmp ${MMAP} \
  --poll ${POLL}
" 2>&1 | tee -a "${LOG_FILE}" "${RESULTS_LOG}"
  
  BENCH_EXIT_CODE=${PIPESTATUS[0]}
  
  if [ ${BENCH_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Benchmark failed with exit code ${BENCH_EXIT_CODE}" | tee -a "${LOG_FILE}" "${RESULTS_LOG}"
  fi

  {
    echo
    echo "===== END ${RUN_TAG} ====="
    echo
  } | tee -a "${RESULTS_LOG}"

  TPS_LINE="$(parse_bench_tps "${LOG_FILE}")"
  DEC_TPS="${TPS_LINE%% *}"
  PREF_TPS="${TPS_LINE#* }"
  MODEL_NAME="$(basename "${MODEL}" .gguf)"

  echo "${MODEL_NAME},${DEC_TPS},${PREF_TPS},${i},${MODE},${TEMP},${RP},${TOP_P},${TOP_K},${MIN_P},${CTX},${KEEP},${BATCH_CPU},${UBATCH},${T},${NGL},${KV_CTK},${KV_CTV},${FA},${POLL},${MMAP},${SPLIT},${DRY_MULT},${FREQ_PEN},${PRES_PEN},${TOKEN_LIMIT},${P_LEN},${G_LEN},${RUN_TAG},${LOG_FILE}" >> "${RESULTS_CSV}"

done

echo "== Done =="
echo "All logs in: ${OUT_DIR}"
echo "CSV results: ${RESULTS_CSV}"
