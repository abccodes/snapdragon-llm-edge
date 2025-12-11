#!/bin/bash

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="hyperparam_search_longbench_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/search_log.txt"
RESULTS_CSV="$OUTPUT_DIR/results.csv"
BEST_CONFIG_FILE="$OUTPUT_DIR/BEST_RESULTS.txt"
RESULTS_CSV_ABS="$(pwd)/$RESULTS_CSV"
BEST_CONFIG_FILE_ABS="$(pwd)/$BEST_CONFIG_FILE"

echo "Starting LongBench hyperparameter search at $(date)" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "run_id,model,mode,temperature,repeat_penalty,top_p,top_k,min_p,ctx_size,keep,batch_size,ubatch_size,threads,ngl,ctk,ctv,flash_attn,context_shift,poll_level,use_mmap,split_mode,dry_multiplier,frequency_penalty,presence_penalty,token_limit,system_prompt,rouge1,rouge2,rougeL,rougeLsum,avg_prefill_speed,avg_decode_speed,avg_total_speed,runtime_seconds" > "$RESULTS_CSV"

################################################################################
# HYPERPARAMETER SPACES
################################################################################

MODELS=(
	#"MobileLLM-R1-360M_Q8_0.gguf"
	"Llama-3.2-1B-Instruct-Q4_0.gguf"
)

MODES=("CPU")
TEMPS=(0.4)
REPEAT_PENALTIES=(1.1)
TOP_PS=(1.0)
TOP_KS=(30)
MIN_P=(0.15)
TOKEN_LIMITS=(700)
CTX_SIZES=(1024)
BATCH_SIZES_CPU=(128)
THREADS=(8)
NGL_VALUES=(0)
KV_CACHE_CTK_VALUES=("f16")
KV_CACHE_CTV_VALUES=("f16")
FLASH_ATTN=("on")
KEEP_VALUES=(0)
CONTEXT_SHIFT=(1)
POLL_LEVELS=(30)
USE_MMAP=(0)
SPLIT_MODES=("none")
DRY_MULTIPLIER=(1.0)
FREQUENCY_PENALTY=(0.1)
PRESENCE_PENALTY=(0.4)

################################################################################
# HELPER FUNCTIONS
################################################################################

get_random() {
    local arr=("$@")
    local rand_idx=$((RANDOM % ${#arr[@]}))
    echo "${arr[$rand_idx]}"
}

update_best_results() {
    python3 << PYPYTHON
import pandas as pd
import sys
from datetime import datetime

try:
    df = pd.read_csv('${RESULTS_CSV_ABS}')
    if len(df) == 0:
        with open('${BEST_CONFIG_FILE_ABS}', 'w') as f:
            f.write("No results yet.\n")
        sys.exit(0)

    df['rougeL'] = pd.to_numeric(df['rougeL'], errors='coerce')
    df_valid = df[(df['rougeL'].notna()) & (df['rougeL'] > 0.0)]
    df_failed = df[(df['rougeL'].isna()) | (df['rougeL'] == 0.0)]

    with open('${BEST_CONFIG_FILE_ABS}', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LONGBENCH (QMSUM) - BEST RESULTS\n")
        f.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total: {len(df)} | Valid: {len(df_valid)} | Failed: {len(df_failed)}\n\n")

        if len(df_valid) == 0:
            f.write("No valid runs yet.\n")
            sys.exit(0)

        f.write("STATISTICS\n" + "=" * 80 + "\n")
        f.write(f"ROUGE-L: {df_valid['rougeL'].min():.4f} - {df_valid['rougeL'].max():.4f} (avg: {df_valid['rougeL'].mean():.4f})\n\n")

        best_rougeL = df_valid.loc[df_valid['rougeL'].idxmax()]
        f.write("🏆 BEST BY ROUGE-L\n" + "=" * 80 + "\n")
        f.write(f"Run: {int(best_rougeL['run_id'])} | ROUGE-L: {best_rougeL['rougeL']:.4f}\n")
        f.write(f"Model: {best_rougeL['model']}\n")
        f.write(f"Temp: {best_rougeL['temperature']:.1f} | Batch: {int(best_rougeL['batch_size'])} | KV: {best_rougeL['ctk']}/{best_rougeL['ctv']}\n\n")
        f.write(f"Full results: ${RESULTS_CSV_ABS}\n")

    with open('${BEST_CONFIG_FILE_ABS}', 'r') as f:
        print(f.read())
except Exception as e:
    print(f"Error: {e}")
PYPYTHON
}

parse_speeds() {
    local debug_log="$1"
    python3 << PYPYTHON
import re, sys
from pathlib import Path
debug_log = Path("$debug_log")
if not debug_log.is_file():
    print("0.00 0.00 0.00")
    sys.exit(0)
total_records = []
try:
    with debug_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "total time" in line and "llama_perf_context_print:" in line:
                nums = re.findall(r'\d+\.\d+|\d+', line)
                if len(nums) >= 2:
                    total_records.append({"time_ms": float(nums[0]), "tokens": int(nums[1])})
except: pass
avg_total_speed = 0.0
if total_records:
    for rec in total_records:
        if rec['time_ms'] > 0:
            avg_total_speed += rec['tokens'] / rec['time_ms'] * 1000
    avg_total_speed /= len(total_records)
print(f"0.00 0.00 {avg_total_speed:.2f}")
PYPYTHON
}

run_configuration() {
    local run_id=$1 model=$2 temp=$3 repeat_penalty=$4 top_p=$5 top_k=$6 min_p=$7 
    local ctx_size=$8 keep=$9 batch_size=${10} threads=${11} ctk=${12} ctv=${13}
    local flash_attn=${14} context_shift=${15} poll_level=${16} use_mmap=${17}
    local dry_mult=${18} freq_penalty=${19} presence_penalty=${20} token_limit=${21}

    local run_dir="$OUTPUT_DIR/run_${run_id}"
    mkdir -p "$run_dir"

    echo "========================================" | tee -a "$LOG_FILE"
    echo "RUN #${run_id} - $(date)" | tee -a "$LOG_FILE"
    echo "Model: $model | Temp: $temp | Batch: $batch_size" | tee -a "$LOG_FILE"
    echo "KV: $ctk/$ctv | DRY: $dry_mult | Presence: $presence_penalty" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Build flags
    local context_shift_flag=""
    [ "$context_shift" -eq 1 ] && context_shift_flag="--context-shift" || context_shift_flag="--no-context-shift"
    
    local mmap_flag=""
    [ "$use_mmap" -eq 0 ] && mmap_flag="--no-mmap" || mmap_flag=""

    # Create Python evaluation script
    cat > "$run_dir/run_eval.py" << 'ENDPY'
#!/usr/bin/env python3
import os, sys, subprocess, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
import evaluate

# Configuration injected by bash
ENDPY

    # Inject ALL variables as Python strings (avoiding special chars)
    cat >> "$run_dir/run_eval.py" <<ENDPY
MODEL = """$model"""
TEMP = $temp
REPEAT_PENALTY = $repeat_penalty
TOP_P = $top_p
TOP_K = $top_k
MIN_P = $min_p
CTX_SIZE = $ctx_size
KEEP = $keep
BATCH_SIZE = $batch_size
THREADS = $threads
CTK = """$ctk"""
CTV = """$ctv"""
FLASH_ATTN = """$flash_attn"""
CONTEXT_SHIFT_FLAG = """$context_shift_flag"""
POLL_LEVEL = $poll_level
MMAP_FLAG = """$mmap_flag"""
DRY_MULT = $dry_mult
FREQ_PENALTY = $freq_penalty
PRESENCE_PENALTY = $presence_penalty
TOKEN_LIMIT = $token_limit
RUN_DIR = """$run_dir"""
ENDPY

    # Append main evaluation logic
    cat >> "$run_dir/run_eval.py" << 'ENDPY'

ds = load_dataset("zai-org/LongBench", "qmsum", split="test")
rouge = evaluate.load("rouge")
output_dir = os.path.join(RUN_DIR, "outputs")
os.makedirs(output_dir, exist_ok=True)
stderr_file = open(os.path.join(RUN_DIR, 'debug.log'), 'w')

predictions, references = [], []
basedir = "/data/local/tmp/llama.cpp"

for i, rec in enumerate(ds):
    print(f"-------- sample {i} --------")
    ans = rec["answers"]
    ref = (ans[0] if isinstance(ans, list) else ans).strip()
    
    # Add instruction for summarization
    raw_input = rec["input"].replace("'", " ").replace('"', ' ')
    prompt = f"Summarize the following meeting transcript:\n\n{raw_input}\n\nSummary:"
    
    # Build llama-cli command string
    llama_cmd = f"-m {basedir}/../gguf/{MODEL} "
    llama_cmd += f"-p '{prompt}' "
    llama_cmd += f"-n {TOKEN_LIMIT} -t {THREADS} -c {CTX_SIZE} -b {BATCH_SIZE} "
    llama_cmd += f"-ngl 0 -dev none -ctk {CTK} -ctv {CTV} "
    llama_cmd += f"--temp {TEMP} --repeat-penalty {REPEAT_PENALTY} "
    llama_cmd += f"--top-p {TOP_P} --top-k {TOP_K} --min-p {MIN_P} "
    llama_cmd += f"--keep {KEEP} -fa {FLASH_ATTN} {CONTEXT_SHIFT_FLAG} "
    llama_cmd += f"--poll {POLL_LEVEL} --no-display-prompt -no-cnv "
    
    if DRY_MULT > 0:
        llama_cmd += f"--dry-multiplier {DRY_MULT} --dry-base 1.75 "
    if FREQ_PENALTY > 0:
        llama_cmd += f"--frequency-penalty {FREQ_PENALTY} "
    if PRESENCE_PENALTY > 0:
        llama_cmd += f"--presence-penalty {PRESENCE_PENALTY} "
    if MMAP_FLAG:
        llama_cmd += f"{MMAP_FLAG} "
    
    # Full adb shell command
    adb_cmd = [
        "adb", "shell",
        f"cd {basedir}; ulimit -c unlimited; "
        f"LD_LIBRARY_PATH={basedir}/./lib ADSP_LIBRARY_PATH={basedir}/./lib "
        f"././bin/llama-cli {llama_cmd}"
    ]
    
    output_file = os.path.join(output_dir, f"out_{i}.txt")
    
    print(f"Running inference for sample {i}...")
    print(f"Output file: {output_file}")
    
    with open(output_file, "w") as fout:
        proc = subprocess.run(adb_cmd, stdout=fout, stderr=stderr_file, text=True)
    
    print(f"Return code: {proc.returncode}")
    
    if proc.returncode != 0:
        print(f"[ERROR] Failed with return code {proc.returncode}")
        continue
    
    # Check file size
    file_size = os.path.getsize(output_file)
    print(f"Output file size: {file_size} bytes")
    
    with open(output_file) as fin:
        pred = fin.read().strip()
        pred_len = len(pred)
        print(f"Prediction length: {pred_len} chars")
        
        if pred:
            print(f"First 100 chars: {pred[:100]}")
            predictions.append(pred)
            references.append(ref)
        else:
            print(f"[WARNING] Empty prediction!")

stderr_file.close()

print(f"\n=== EVALUATION SUMMARY ===")
print(f"Total predictions: {len(predictions)}")
print(f"Total references: {len(references)}")

if predictions:
    try:
        per_sample = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=False)
        
        print("\n=== Per-sample ROUGE-L ===")
        for i in range(len(predictions)):
            rl = per_sample["rougeL"][i]
            print(f"Sample {i}: ROUGE-L = {rl:.4f}")
        
        filtered_preds = [p for i, p in enumerate(predictions) if per_sample["rougeL"][i] > 0.0]
        filtered_refs = [r for i, r in enumerate(references) if per_sample["rougeL"][i] > 0.0]
        
        print(f"\nFiltered: {len(filtered_preds)}/{len(predictions)} samples with ROUGE-L > 0")
        
        if filtered_preds:
            result = rouge.compute(predictions=filtered_preds, references=filtered_refs, use_stemmer=True)
        else:
            result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        
        print(f"\n=== Aggregated ROUGE ===")
        print(f"ROUGE-1: {result['rouge1']:.4f}")
        print(f"ROUGE-2: {result['rouge2']:.4f}")
        print(f"ROUGE-L: {result['rougeL']:.4f}")
        print(f"ROUGE-Lsum: {result['rougeLsum']:.4f}")
        
        # Output in parseable format - use the ACTUAL keys from result dict
        print(f"ROUGE1={result['rouge1']}")
        print(f"ROUGE2={result['rouge2']}")
        print(f"ROUGEL={result['rougeL']}")
        print(f"ROUGELSUM={result['rougeLsum']}")
        
    except Exception as e:
        print(f"ERROR in ROUGE computation: {e}")
        import traceback
        traceback.print_exc()
        # Print zeros for parsing
        print("ROUGE1=0.0")
        print("ROUGE2=0.0")
        print("ROUGEL=0.0")
        print("ROUGELSUM=0.0")
else:
    print("WARNING: No predictions generated!")
    print("ROUGE1=0.0")
    print("ROUGE2=0.0")
    print("ROUGEL=0.0")
    print("ROUGELSUM=0.0")
ENDPY

    chmod +x "$run_dir/run_eval.py"

    echo "Running LongBench test..." | tee -a "$LOG_FILE"
    local start_time=$(date +%s)
    python3 "$run_dir/run_eval.py" 2>&1 | tee "$run_dir/eval_output.txt" | tee -a "$LOG_FILE"
    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))

    # Parse ROUGE scores
    local rouge1=$(grep "ROUGE1=" "$run_dir/eval_output.txt" 2>/dev/null | tail -1 | sed 's/.*=//' || echo "0.0")
    local rouge2=$(grep "ROUGE2=" "$run_dir/eval_output.txt" 2>/dev/null | tail -1 | sed 's/.*=//' || echo "0.0")
    local rougeL=$(grep "ROUGEL=" "$run_dir/eval_output.txt" 2>/dev/null | tail -1 | sed 's/.*=//' || echo "0.0")
    local rougeLsum=$(grep "ROUGELSUM=" "$run_dir/eval_output.txt" 2>/dev/null | tail -1 | sed 's/.*=//' || echo "0.0")

    # Parse speeds
    local speeds=$(parse_speeds "$run_dir/debug.log")
    local avg_total_speed=$(echo "$speeds" | awk '{print $3}')

    echo "ROUGE-L: $rougeL | Runtime: ${runtime}s | Speed: $avg_total_speed tok/s" | tee -a "$LOG_FILE"

    # Append to CSV
    echo "${run_id},${model},CPU,${temp},${repeat_penalty},${top_p},${top_k},${min_p},${ctx_size},${keep},${batch_size},${batch_size},${threads},0,${ctk},${ctv},${flash_attn},${context_shift},${poll_level},${use_mmap},none,${dry_mult},${freq_penalty},${presence_penalty},${token_limit},,${rouge1},${rouge2},${rougeL},${rougeLsum},0.00,0.00,${avg_total_speed},${runtime}" >> "$RESULTS_CSV"

    update_best_results
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# MAIN SEARCH
################################################################################

NUM_TRIALS="${NUM_TRIALS:-1}"

echo "Running $NUM_TRIALS trials..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for run_id in $(seq 1 $NUM_TRIALS); do
    model=$(get_random "${MODELS[@]}")
    temp=$(get_random "${TEMPS[@]}")
    repeat_penalty=$(get_random "${REPEAT_PENALTIES[@]}")
    top_p=$(get_random "${TOP_PS[@]}")
    top_k=$(get_random "${TOP_KS[@]}")
    min_p=$(get_random "${MIN_P[@]}")
    ctx_size=$(get_random "${CTX_SIZES[@]}")
    keep=$(get_random "${KEEP_VALUES[@]}")
    token_limit=$(get_random "${TOKEN_LIMITS[@]}")
    batch_size=$(get_random "${BATCH_SIZES_CPU[@]}")
    threads=$(get_random "${THREADS[@]}")
    
    # KV cache - select ONE value from each array
    ctk=$(get_random "${KV_CACHE_CTK_VALUES[@]}")
    ctv=$(get_random "${KV_CACHE_CTV_VALUES[@]}")
    
    flash_attn=$(get_random "${FLASH_ATTN[@]}")
    context_shift=$(get_random "${CONTEXT_SHIFT[@]}")
    poll_level=$(get_random "${POLL_LEVELS[@]}")
    use_mmap=$(get_random "${USE_MMAP[@]}")
    dry_mult=$(get_random "${DRY_MULTIPLIER[@]}")
    freq_penalty=$(get_random "${FREQUENCY_PENALTY[@]}")
    presence_penalty=$(get_random "${PRESENCE_PENALTY[@]}")

    run_configuration "$run_id" "$model" "$temp" "$repeat_penalty" "$top_p" "$top_k" "$min_p" "$ctx_size" "$keep" "$batch_size" "$threads" "$ctk" "$ctv" "$flash_attn" "$context_shift" "$poll_level" "$use_mmap" "$dry_mult" "$freq_penalty" "$presence_penalty" "$token_limit"

    sleep 1
done

echo "Search complete! Results: $BEST_CONFIG_FILE" | tee -a "$LOG_FILE"
