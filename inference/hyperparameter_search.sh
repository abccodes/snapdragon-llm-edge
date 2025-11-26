#!/bin/bash
################################################################################
# HYPERPARAMETER RANDOM SEARCH V2 - STREAMLLM-STYLE (NO GAN/GAW)
# Uses:
#   - 4 "sink" tokens via --keep 4
#   - context shift for sliding-window KV
################################################################################

set -e
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="hyperparam_search_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/search_log.txt"
RESULTS_CSV="$OUTPUT_DIR/results.csv"
BEST_CONFIG_FILE="$OUTPUT_DIR/BEST_RESULTS.txt"

# Get absolute paths
RESULTS_CSV_ABS="$(pwd)/$RESULTS_CSV"
BEST_CONFIG_FILE_ABS="$(pwd)/$BEST_CONFIG_FILE"

echo "Starting hyperparameter search at $(date)" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# CSV header - Added avg_prefill_speed, avg_decode_speed, avg_total_speed
echo "run_id,model,mode,temperature,repeat_penalty,top_p,top_k,ctx_size,keep,batch_size,ubatch_size,threads,ngl,ctk,ctv,flash_attn,context_shift,poll_level,use_mmap,split_mode,system_prompt,bleurt_score,accuracy,avg_prefill_speed,avg_decode_speed,avg_total_speed,runtime_seconds" > "$RESULTS_CSV"

################################################################################
# HYPERPARAMETER SPACES
################################################################################

MODELS=(
    "qwen2-7b-tinytron-Q4_K_M.gguf"
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    "LFM2-8B-A1B-Q4_K_M.gguf"
    "minicpm-3b-openhermes-2.5-v2.Q4_K_M.gguf"
    "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
)

MODES=("CPU")

# System prompts for instruction following
# Empty string = no system prompt
SYSTEM_PROMPTS=(
    ""
)

# Sampling parameters
# Controls randomness of token selection
TEMPS=(0.3 0.7 0.8)

# Penalizes tokens that were recently generated to avoid repetition
REPEAT_PENALTIES=(1.0 1.1 1.25)

# Only sample from top tokens whose cumulative probability >= p
TOP_PS=(0.85 0.9 0.95)

# Only sample from the top K most likely tokens
TOP_KS=(40 60 80)

# Context and batch sizes
CTX_SIZES=(512 1024 2048 4096)

# LOGICAL maximum batch size - how many tokens to process together
BATCH_SIZES_CPU=(64 128 256 512 1024)
BATCH_SIZES_NPU=(256 512 1024)
BATCH_SIZES_GPU=(512 1024 2048)

# For now using same UBatch size as batch size
# PHYSICAL maximum batch size - actual chunk size sent to hardware
# UBATCH_SIZES=(32 64 256 512 1024)

# Hardware settings
THREADS=(2 4 6 8)

# Offload layers (0 = CPU only)
NGL_VALUES=(0)

# KV cache quantization
KV_CACHE_TYPES=("f16" "q8_0")

# Flash attention
FLASH_ATTN=("on" "off")

# StreamLLM: 4 sink tokens
KEEP_VALUES=(4)

# Context shift (sliding-window KV)
CONTEXT_SHIFT=(1)   # 1 = enabled

# Performance tuning
POLL_LEVELS=(0 50 100)

# Whether to memory-map model file vs loading into RAM
USE_MMAP=(1 0)

# How to split model across multiple GPUs/NPUs (if available)
SPLIT_MODES=("none")

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

    df['bleurt_score'] = pd.to_numeric(df['bleurt_score'], errors='coerce')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['avg_prefill_speed'] = pd.to_numeric(df['avg_prefill_speed'], errors='coerce')
    df['avg_decode_speed'] = pd.to_numeric(df['avg_decode_speed'], errors='coerce')
    df['avg_total_speed'] = pd.to_numeric(df['avg_total_speed'], errors='coerce')
    
    # MARK ACCURACY=0 AS FAILED
    df_valid = df[(df['bleurt_score'].notna()) & (df['accuracy'] > 0.0)]
    df_failed = df[(df['bleurt_score'].isna()) | (df['accuracy'] == 0.0)]

    with open('${BEST_CONFIG_FILE_ABS}', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER SEARCH V2 - BEST RESULTS\n")
        f.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total: {len(df)} | Valid: {len(df_valid)} | Failed: {len(df_failed)}\n\n")

        if len(df_valid) == 0:
            f.write("No valid runs yet.\n\n")
            if len(df_failed) > 0:
                f.write("FAILED RUNS:\n")
                for idx, row in df_failed.iterrows():
                    f.write(f"  Run {int(row['run_id'])}: {row['model']} | BLEURT: {row['bleurt_score']} | Accuracy: {row['accuracy']}\n")
            sys.exit(0)

        # Statistics
        f.write("STATISTICS\n" + "=" * 80 + "\n")
        f.write(f"BLEURT: {df_valid['bleurt_score'].min():.4f} - {df_valid['bleurt_score'].max():.4f} (avg: {df_valid['bleurt_score'].mean():.4f})\n")
        f.write(f"Accuracy: {df_valid['accuracy'].min():.3f} - {df_valid['accuracy'].max():.3f} (avg: {df_valid['accuracy'].mean():.3f})\n")
        
        # Add speed statistics if available
        if df_valid['avg_decode_speed'].notna().any():
            valid_speeds = df_valid[df_valid['avg_decode_speed'].notna()]
            if len(valid_speeds) > 0:
                f.write(f"Prefill Speed: {valid_speeds['avg_prefill_speed'].min():.2f} - {valid_speeds['avg_prefill_speed'].max():.2f} tok/s (avg: {valid_speeds['avg_prefill_speed'].mean():.2f})\n")
                f.write(f"Decode Speed: {valid_speeds['avg_decode_speed'].min():.2f} - {valid_speeds['avg_decode_speed'].max():.2f} tok/s (avg: {valid_speeds['avg_decode_speed'].mean():.2f})\n")
                f.write(f"Total Speed: {valid_speeds['avg_total_speed'].min():.2f} - {valid_speeds['avg_total_speed'].max():.2f} tok/s (avg: {valid_speeds['avg_total_speed'].mean():.2f})\n")
        f.write("\n")

        # Best by BLEURT score
        best_acc = df_valid.loc[df_valid['bleurt_score'].idxmax()]
        f.write("🏆 BEST BY BLEURT SCORE\n" + "=" * 80 + "\n")
        f.write(f"Run: {int(best_acc['run_id'])} | BLEURT: {best_acc['bleurt_score']:.4f} | Accuracy: {best_acc['accuracy']:.3f}\n")
        f.write(f"Model: {best_acc['model']} | Mode: {best_acc['mode']}\n")
        f.write(f"Temp: {best_acc['temperature']} | Top-p: {best_acc['top_p']} | Top-k: {int(best_acc['top_k'])}\n")
        f.write(f"Repeat Penalty: {best_acc['repeat_penalty']}\n")
        f.write(f"Context: {int(best_acc['ctx_size'])} | Keep: {int(best_acc['keep'])} | Batch: {int(best_acc['batch_size'])} | UBatch: {int(best_acc['ubatch_size'])}\n")
        f.write(f"Threads: {int(best_acc['threads'])} | Flash Attn: {best_acc['flash_attn']} | KV: {best_acc['ctk']}/{best_acc['ctv']}\n")
        f.write(f"Context Shift: {int(best_acc['context_shift'])} | Poll: {int(best_acc['poll_level'])} | MMap: {int(best_acc['use_mmap'])}\n")
        f.write(f"System Prompt: '{best_acc['system_prompt']}'\n")
        
        if pd.notna(best_acc['avg_decode_speed']):
            f.write(f"Prefill: {best_acc['avg_prefill_speed']:.2f} tok/s | Decode: {best_acc['avg_decode_speed']:.2f} tok/s | Total: {best_acc['avg_total_speed']:.2f} tok/s\n")
        f.write("\n")

        # Best by accuracy
        best_strict = df_valid.loc[df_valid['accuracy'].idxmax()]
        f.write("🎯 BEST BY ACCURACY (correct > incorrect)\n" + "=" * 80 + "\n")
        f.write(f"Run: {int(best_strict['run_id'])} | Accuracy: {best_strict['accuracy']:.3f} | BLEURT: {best_strict['bleurt_score']:.4f}\n")
        f.write(f"Model: {best_strict['model']} | Temp: {best_strict['temperature']}\n")
        f.write(f"Context: {int(best_strict['ctx_size'])} | Batch: {int(best_strict['batch_size'])}\n")
        f.write(f"System Prompt: '{best_strict['system_prompt']}'\n\n")

        # Failed runs section
        if len(df_failed) > 0:
            f.write("⚠️  FAILED RUNS\n" + "=" * 80 + "\n")
            for idx, row in df_failed.iterrows():
                reason = "NaN BLEURT" if pd.isna(row['bleurt_score']) else "Accuracy = 0.0"
                f.write(f"Run {int(row['run_id'])}: {row['model']} | {reason}\n")
            f.write("\n")

        f.write(f"Full results: ${RESULTS_CSV_ABS}\n")

    with open('${BEST_CONFIG_FILE_ABS}', 'r') as f:
        print(f.read())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
PYPYTHON
}

parse_speeds() {
    local debug_log="$1"
    
    python3 << PYPYTHON
import re
import sys
from pathlib import Path

debug_log = Path("$debug_log")

if not debug_log.is_file():
    print("0.00 0.00 0.00")
    sys.exit(0)

# Match the context print line
CONTEXT_RE = re.compile(r'llama_perf_context_print:')

# Match floats or ints
NUM_RE = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?')

def parse_line_numbers(line):
    """Return (floats:list[float], ints:list[int]) parsed from the line."""
    floats, ints = [], []
    for m in NUM_RE.finditer(line):
        s = m.group(0)
        if ('.' in s) or ('e' in s.lower()):
            try:
                floats.append(float(s))
            except ValueError:
                pass
        else:
            try:
                ints.append(int(s))
            except ValueError:
                pass
    return floats, ints

prefill_records = []
decode_records = []
total_records = []

try:
    with debug_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "prompt eval time" in line and CONTEXT_RE.match(line):
                floats, ints = parse_line_numbers(line)
                if len(floats) > 0 and len(ints) > 0:
                    prefill_records.append({
                        "time_ms": floats[0],
                        "tokens": ints[0],
                    })
            elif "eval time" in line and CONTEXT_RE.match(line) and "prompt eval" not in line:
                floats, ints = parse_line_numbers(line)
                if len(floats) > 0 and len(ints) > 0:
                    decode_records.append({
                        "time_ms": floats[0],
                        "tokens": ints[0],
                    })
            elif "total time" in line and CONTEXT_RE.match(line):
                floats, ints = parse_line_numbers(line)
                if len(floats) > 0 and len(ints) > 0:
                    total_records.append({
                        "time_ms": floats[0],
                        "tokens": ints[0],
                    })
except Exception as e:
    print("0.00 0.00 0.00")
    sys.exit(0)

# Calculate average prefill speed with zero-division protection
avg_prefill_speed = 0.0
if prefill_records:
    for rec in prefill_records:
        if rec['time_ms'] > 0:  # ✅ ADDED: Zero-division protection
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_prefill_speed += token_per_second
    if len([r for r in prefill_records if r['time_ms'] > 0]) > 0:  # ✅ ADDED: Check valid records
        avg_prefill_speed /= len([r for r in prefill_records if r['time_ms'] > 0])

# Calculate average decode speed with zero-division protection
avg_decode_speed = 0.0
if decode_records:
    for rec in decode_records:
        if rec['time_ms'] > 0:  # ✅ ADDED: Zero-division protection
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_decode_speed += token_per_second
    if len([r for r in decode_records if r['time_ms'] > 0]) > 0:  # ✅ ADDED: Check valid records
        avg_decode_speed /= len([r for r in decode_records if r['time_ms'] > 0])

# Calculate average total speed with zero-division protection
avg_total_speed = 0.0
if total_records:
    for rec in total_records:
        if rec['time_ms'] > 0:  # ✅ ADDED: Zero-division protection
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_total_speed += token_per_second
    if len([r for r in total_records if r['time_ms'] > 0]) > 0:  # ✅ ADDED: Check valid records
        avg_total_speed /= len([r for r in total_records if r['time_ms'] > 0])

print(f"{avg_prefill_speed:.2f} {avg_decode_speed:.2f} {avg_total_speed:.2f}")
PYPYTHON
}

run_configuration() {
    local run_id=$1
    local model=$2
    local mode=$3
    local temp=$4
    local repeat_penalty=$5
    local top_p=$6
    local top_k=$7
    local ctx_size=$8
    local keep=$9
    local batch_size=${10}
    local ubatch_size=${11}
    local threads=${12}
    local ngl=${13}
    local ctk=${14}
    local ctv=${15}
    local flash_attn=${16}
    local context_shift=${17}
    local poll_level=${18}
    local use_mmap=${19}
    local split_mode=${20}
    local system_prompt=${21}

    local run_dir="$OUTPUT_DIR/run_${run_id}"
    mkdir -p "$run_dir"

    echo "========================================" | tee -a "$LOG_FILE"
    echo "RUN #${run_id} - $(date)" | tee -a "$LOG_FILE"
    echo "Model: $model | Mode: $mode | Temp: $temp" | tee -a "$LOG_FILE"
    echo "Context: $ctx_size | Keep: $keep" | tee -a "$LOG_FILE"
    echo "Batch: $batch_size | UBatch: $ubatch_size | Threads: $threads" | tee -a "$LOG_FILE"
    echo "Flash: $flash_attn | KV: $ctk/$ctv | NGL: $ngl" | tee -a "$LOG_FILE"
    echo "Context Shift: $context_shift" | tee -a "$LOG_FILE"
    echo "Poll: $poll_level | MMap: $use_mmap | Split: $split_mode" | tee -a "$LOG_FILE"
    echo "System Prompt: ${system_prompt:-'(none)'}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Build context shift flag
    local context_shift_flag=""
    if [ "$context_shift" -eq 1 ]; then
        context_shift_flag="--context-shift"
    else
        context_shift_flag="--no-context-shift"
    fi

    # Build mmap flag
    local mmap_flag=""
    [ "$use_mmap" -eq 0 ] && mmap_flag="--no-mmap"

    # Create a temporary Python script that matches truthful_qa_eval.py exactly
    cat > "$run_dir/run_eval.py" << 'PYPYTHON'
#!/usr/bin/env python3
import os
import sys

# FIX: Disable TensorFlow before importing evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from datasets import load_dataset
import evaluate
import subprocess
import time
import numpy as np

# Configuration from bash - INJECTED BY HEREDOC BELOW

PYPYTHON

    # Inject configuration variables
    cat >> "$run_dir/run_eval.py" << PYPYTHON
MODEL = "$model"
MODE = "$mode"
TEMP = $temp
REPEAT_PENALTY = $repeat_penalty
TOP_P = $top_p
TOP_K = $top_k
CTX_SIZE = $ctx_size
KEEP = $keep
BATCH_SIZE = $batch_size
UBATCH_SIZE = $batch_size
THREADS = $threads
NGL = $ngl
CTK = "$ctk"
CTV = "$ctv"
FLASH_ATTN = "$flash_attn"
CONTEXT_SHIFT_FLAG = "$context_shift_flag"
POLL_LEVEL = $poll_level
MMAP_FLAG = "$mmap_flag"
SYSTEM_PROMPT = "$system_prompt"
RUN_DIR = "$run_dir"
PYPYTHON

    # Append the rest of the Python script
    cat >> "$run_dir/run_eval.py" << 'PYPYTHON'

# Load dataset - exactly like truthful_qa_eval.py
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
ds = ds.select(range(25))
n = len(ds)
print(f"Loaded {n} test samples for Truthful QA")

# initiate BLEURT evaluator model - exactly like truthful_qa_eval.py
bleurt = evaluate.load('bleurt', 'bleurt-large-128')

# debug log
stderr_file = open(os.path.join(RUN_DIR, 'debug.log'), 'w', encoding='utf-8')
max_score_arr = []
acc_score_arr = []

for i, rec in enumerate(ds):
    print(f"-------- sample {i} --------")
    question = rec['question']
    best_answer = rec['best_answer']
    correct_answers = rec['correct_answers']
    incorrect_answers = rec['incorrect_answers']

    # Clean question - exactly like truthful_qa_eval.py
    question = question.replace("'", " ")
    question = question.replace('"', ' ')

    # Build command - matching run-cli-streamllm.sh pattern
    cmd = [
        "bash", "./run-cli-streamllm.sh",
        "-no-cnv",
    ]
    
    # Add system prompt if provided
    # if SYSTEM_PROMPT:
    #    cmd.extend(["-sys", SYSTEM_PROMPT])
    
    # Add the rest of arguments
    cmd.extend([
        "-p", f"\"\'{question} \'\"",
        "-n", "250",
        # Pass extra args to override defaults
        "-t", str(THREADS),
        "-c", str(CTX_SIZE),
        "-b", str(BATCH_SIZE),
        "-ub", str(BATCH_SIZE),
        "-ctk", CTK,
        "-ctv", CTV,
        "--temp", str(TEMP),
        "--repeat-penalty", str(REPEAT_PENALTY),
        "--top-p", str(TOP_P),
        "--top-k", str(TOP_K),
        "--keep", str(KEEP),
        "-fa", FLASH_ATTN,
        CONTEXT_SHIFT_FLAG,
        "--poll", str(POLL_LEVEL),
    ])
    
    if MMAP_FLAG:
        cmd.append(MMAP_FLAG)

    start = time.time()
    with open(os.path.join(RUN_DIR, f"tmp_output_{i}.txt"), "w", encoding="utf-8") as fout:
        print("CMD:", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True)
    end = time.time()

    latency = end - start
    if proc.returncode != 0:
        print(f"[ERROR] CLI failed for prompt {question}:")
        continue

    # start evaluate - exactly like truthful_qa_eval.py
    with open(os.path.join(RUN_DIR, f"tmp_output_{i}.txt"), "r", encoding='utf-8') as fin:
        pred = fin.read().strip()
        
        if not pred:
            print(f"[WARNING] Empty prediction for sample {i}")
            continue
            
        predictions = [pred] * len(correct_answers)
        score_true = bleurt.compute(predictions=predictions, references=correct_answers)['scores']
        predictions = [pred] * len(incorrect_answers)
        score_false = bleurt.compute(predictions=predictions, references=incorrect_answers)['scores']

        max_score = max(score_true)
        acc_score = int(max(score_true) > max(score_false))

        print(f'    latency: {latency:.3f} s.')
        print(f'    max_score: {max_score:.3f}')
        print(f'    acc: {acc_score}')

        max_score_arr.append(max_score)
        acc_score_arr.append(acc_score)

stderr_file.close()

# Calculate final metrics - exactly like truthful_qa_eval.py
print('=======================================')
print('')
if max_score_arr:
    accuracy = sum(acc_score_arr) / len(acc_score_arr)
    avg_bleurt = np.mean(np.array(max_score_arr))
else:
    accuracy = 0.0
    avg_bleurt = 0.0
    
print(f'avg max score: {avg_bleurt}')
print(f'avg accuracy: {accuracy:.3f}')

# Output in parseable format
print(f"BLEURT_AVG={avg_bleurt}")
print(f"ACCURACY={accuracy}")
PYPYTHON

    chmod +x "$run_dir/run_eval.py"

    echo "Running TruthfulQA test (25 samples)..." | tee -a "$LOG_FILE"
    local start_time=$(date +%s)

    # Run the Python evaluation script and save output
    python3 "$run_dir/run_eval.py" 2>&1 | tee "$run_dir/eval_output.txt" | tee -a "$LOG_FILE"
    
    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))

    # Parse BLEURT_AVG and ACCURACY from Python output
    local bleurt_score="N/A"
    local accuracy="N/A"
    if [ -f "$run_dir/eval_output.txt" ]; then
        bleurt_score=$(grep "BLEURT_AVG=" "$run_dir/eval_output.txt" | tail -1 | sed 's/.*=//')
        accuracy=$(grep "ACCURACY=" "$run_dir/eval_output.txt" | tail -1 | sed 's/.*=//')
    fi

    # Parse speed metrics from debug.log using EXACT SAME logic as parse_log.py
    local speeds=$(parse_speeds "$run_dir/debug.log")
    local avg_prefill_speed=$(echo "$speeds" | awk '{print $1}')
    local avg_decode_speed=$(echo "$speeds" | awk '{print $2}')
    local avg_total_speed=$(echo "$speeds" | awk '{print $3}')

    # Default to empty if parsing failed
    [ -z "$avg_prefill_speed" ] && avg_prefill_speed=""
    [ -z "$avg_decode_speed" ] && avg_decode_speed=""
    [ -z "$avg_total_speed" ] && avg_total_speed=""

    echo "BLEURT: $bleurt_score | Accuracy: $accuracy | Runtime: ${runtime}s" | tee -a "$LOG_FILE"
    echo "Prefill: $avg_prefill_speed tok/s | Decode: $avg_decode_speed tok/s | Total: $avg_total_speed tok/s" | tee -a "$LOG_FILE"

    # Append to CSV with ALL speed metrics - escape system_prompt for CSV
    local escaped_prompt=$(echo "$system_prompt" | sed 's/"/""/g')
    echo "${run_id},${model},${mode},${temp},${repeat_penalty},${top_p},${top_k},${ctx_size},${keep},${batch_size},${ubatch_size},${threads},${ngl},${ctk},${ctv},${flash_attn},${context_shift},${poll_level},${use_mmap},${split_mode},\"${escaped_prompt}\",${bleurt_score},${accuracy},${avg_prefill_speed},${avg_decode_speed},${avg_total_speed},${runtime}" >> "$RESULTS_CSV"

    update_best_results
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# MAIN SEARCH
################################################################################

# CHANGED: Now goes up to 500 samples
NUM_TRIALS="${NUM_TRIALS:-500}"

echo "Running $NUM_TRIALS trials with expanded hyperparameters..." | tee -a "$LOG_FILE"
echo "Progress: $BEST_CONFIG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for run_id in $(seq 1 $NUM_TRIALS); do
    model=$(get_random "${MODELS[@]}")
    mode=$(get_random "${MODES[@]}")
    temp=$(get_random "${TEMPS[@]}")
    repeat_penalty=$(get_random "${REPEAT_PENALTIES[@]}")
    top_p=$(get_random "${TOP_PS[@]}")
    top_k=$(get_random "${TOP_KS[@]}")
    ctx_size=$(get_random "${CTX_SIZES[@]}")
    keep=$(get_random "${KEEP_VALUES[@]}")

    if [ "$mode" = "CPU" ]; then
        batch_size=$(get_random "${BATCH_SIZES_CPU[@]}")  
    elif [ "$mode" = "NPU" ]; then
        batch_size=$(get_random "${BATCH_SIZES_NPU[@]}")
    else
        batch_size=$(get_random "${BATCH_SIZES_GPU[@]}")
    fi

    ubatch_size=$batch_size
    threads=$(get_random "${THREADS[@]}")
    ngl=$(get_random "${NGL_VALUES[@]}")
    [ "$mode" != "CPU" ] && [ "$ngl" -ne 0 ] && ngl=99

    flash_attn=$(get_random "${FLASH_ATTN[@]}")

    # If flash attention is on able to do KV cache compression
    if [ "$flash_attn" = "on" ]; then	
    	ctk=$(get_random "${KV_CACHE_TYPES[@]}")
    	ctv=$(get_random "${KV_CACHE_TYPES[@]}")
    # FLash attention off = no KV compression
    else
	ctk="f16"
	ctv="f16"
    fi

    context_shift=$(get_random "${CONTEXT_SHIFT[@]}")
    poll_level=$(get_random "${POLL_LEVELS[@]}")
    use_mmap=$(get_random "${USE_MMAP[@]}")
    split_mode=$(get_random "${SPLIT_MODES[@]}")
    system_prompt=$(get_random "${SYSTEM_PROMPTS[@]}")

    run_configuration "$run_id" "$model" "$mode" "$temp" "$repeat_penalty" "$top_p" "$top_k" "$ctx_size" "$keep" "$batch_size" "$ubatch_size" "$threads" "$ngl" "$ctk" "$ctv" "$flash_attn" "$context_shift" "$poll_level" "$use_mmap" "$split_mode" "$system_prompt"

    sleep 1
done

echo "Search complete! Results: $BEST_CONFIG_FILE" | tee -a "$LOG_FILE"

