#!/bin/bash
################################################################################
# HYPERPARAMETER RANDOM SEARCH - LONGBENCH (QMSUM)
# Uses:
#   - 4 "sink" tokens via --keep 4
#   - context shift for sliding-window KV
#   - Tests on qmsum task from LongBench
################################################################################

# TODO - Test to make sure this works

set -e
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="hyperparam_search_longbench_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/search_log.txt"
RESULTS_CSV="$OUTPUT_DIR/results.csv"
BEST_CONFIG_FILE="$OUTPUT_DIR/BEST_RESULTS.txt"

# Get absolute paths
RESULTS_CSV_ABS="$(pwd)/$RESULTS_CSV"
BEST_CONFIG_FILE_ABS="$(pwd)/$BEST_CONFIG_FILE"

echo "Starting LongBench hyperparameter search at $(date)" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# CSV header - Added avg_prefill_speed, avg_decode_speed, avg_total_speed
echo "run_id,model,mode,temperature,repeat_penalty,top_p,top_k,ctx_size,keep,batch_size,ubatch_size,threads,ngl,ctk,ctv,flash_attn,context_shift,poll_level,use_mmap,split_mode,system_prompt,rouge_l,rouge_1,rouge_2,rouge_lsum,avg_prefill_speed,avg_decode_speed,avg_total_speed,runtime_seconds" > "$RESULTS_CSV"

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

# System prompts for summarization
SYSTEM_PROMPTS=(
    ""
)

# Sampling parameters
TEMPS=(0.3 0.5 0.8)
REPEAT_PENALTIES=(1.0 1.1 1.25)
TOP_PS=(0.85 0.9 0.95)
TOP_KS=(40 60 80)

# Context and batch sizes - LongBench needs larger context
CTX_SIZES=(512 1024 2048 4096)

# Batch sizes
BATCH_SIZES_CPU=(64 128 256 512)
BATCH_SIZES_NPU=(256 512 1024)
BATCH_SIZES_GPU=(512 1024 2048)

# Use same as batch sizes for now
# UBATCH_SIZES=(256 512 1024)

# Hardware settings
THREADS=(2 4 6 8)
NGL_VALUES=(0)

# KV cache quantization
KV_CACHE_TYPES=("f16" "q8_0")

# Flash attention
FLASH_ATTN=("on" "off")

# StreamLLM: 4 sink tokens
KEEP_VALUES=(0 4)

# Context shift (sliding-window KV)
CONTEXT_SHIFT=(1)

# Performance tuning
POLL_LEVELS=(0 50 100)
USE_MMAP=(1 0)
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

    df['rouge_l'] = pd.to_numeric(df['rouge_l'], errors='coerce')
    df['rouge_1'] = pd.to_numeric(df['rouge_1'], errors='coerce')
    df['rouge_2'] = pd.to_numeric(df['rouge_2'], errors='coerce')
    df['rouge_lsum'] = pd.to_numeric(df['rouge_lsum'], errors='coerce')
    df['avg_prefill_speed'] = pd.to_numeric(df['avg_prefill_speed'], errors='coerce')
    df['avg_decode_speed'] = pd.to_numeric(df['avg_decode_speed'], errors='coerce')
    df['avg_total_speed'] = pd.to_numeric(df['avg_total_speed'], errors='coerce')

    # Filter out failed runs (NaN ROUGE-L or ROUGE-L == 0)
    df_valid = df[(df['rouge_l'].notna()) & (df['rouge_l'] > 0.0)]
    df_failed = df[(df['rouge_l'].isna()) | (df['rouge_l'] == 0.0)]

    with open('${BEST_CONFIG_FILE_ABS}', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LONGBENCH (QMSUM) HYPERPARAMETER SEARCH - BEST RESULTS\n")
        f.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total: {len(df)} | Valid: {len(df_valid)} | Failed: {len(df_failed)}\n\n")

        if len(df_valid) == 0:
            f.write("No valid runs yet.\n\n")
            if len(df_failed) > 0:
                f.write("FAILED RUNS:\n")
                for idx, row in df_failed.iterrows():
                    f.write(f"  Run {int(row['run_id'])}: {row['model']} | ROUGE-L: {row['rouge_l']}\n")
            sys.exit(0)

        # Statistics
        f.write("STATISTICS\n" + "=" * 80 + "\n")
        f.write(f"ROUGE-L: {df_valid['rouge_l'].min():.4f} - {df_valid['rouge_l'].max():.4f} (avg: {df_valid['rouge_l'].mean():.4f})\n")
        f.write(f"ROUGE-1: {df_valid['rouge_1'].min():.4f} - {df_valid['rouge_1'].max():.4f} (avg: {df_valid['rouge_1'].mean():.4f})\n")
        f.write(f"ROUGE-2: {df_valid['rouge_2'].min():.4f} - {df_valid['rouge_2'].max():.4f} (avg: {df_valid['rouge_2'].mean():.4f})\n")

        # Add speed statistics if available
        if df_valid['avg_decode_speed'].notna().any():
            valid_speeds = df_valid[df_valid['avg_decode_speed'].notna()]
            if len(valid_speeds) > 0:
                f.write(f"Prefill Speed: {valid_speeds['avg_prefill_speed'].min():.2f} - {valid_speeds['avg_prefill_speed'].max():.2f} tok/s (avg: {valid_speeds['avg_prefill_speed'].mean():.2f})\n")
                f.write(f"Decode Speed: {valid_speeds['avg_decode_speed'].min():.2f} - {valid_speeds['avg_decode_speed'].max():.2f} tok/s (avg: {valid_speeds['avg_decode_speed'].mean():.2f})\n")
                f.write(f"Total Speed: {valid_speeds['avg_total_speed'].min():.2f} - {valid_speeds['avg_total_speed'].max():.2f} tok/s (avg: {valid_speeds['avg_total_speed'].mean():.2f})\n")
        f.write("\n")

        # Best by ROUGE-L score
        best_rouge = df_valid.loc[df_valid['rouge_l'].idxmax()]
        f.write("🏆 BEST BY ROUGE-L SCORE\n" + "=" * 80 + "\n")
        f.write(f"Run: {int(best_rouge['run_id'])} | ROUGE-L: {best_rouge['rouge_l']:.4f}\n")
        f.write(f"ROUGE-1: {best_rouge['rouge_1']:.4f} | ROUGE-2: {best_rouge['rouge_2']:.4f} | ROUGE-Lsum: {best_rouge['rouge_lsum']:.4f}\n")
        f.write(f"Model: {best_rouge['model']} | Mode: {best_rouge['mode']}\n")
        f.write(f"Temp: {best_rouge['temperature']} | Top-p: {best_rouge['top_p']} | Top-k: {int(best_rouge['top_k'])}\n")
        f.write(f"Repeat Penalty: {best_rouge['repeat_penalty']}\n")
        f.write(f"Context: {int(best_rouge['ctx_size'])} | Keep: {int(best_rouge['keep'])} | Batch: {int(best_rouge['batch_size'])} | UBatch: {int(best_rouge['ubatch_size'])}\n")
        f.write(f"Threads: {int(best_rouge['threads'])} | Flash Attn: {best_rouge['flash_attn']} | KV: {best_rouge['ctk']}/{best_rouge['ctv']}\n")
        f.write(f"Context Shift: {int(best_rouge['context_shift'])} | Poll: {int(best_rouge['poll_level'])} | MMap: {int(best_rouge['use_mmap'])}\n")
        f.write(f"System Prompt: '{best_rouge['system_prompt']}'\n")

        if pd.notna(best_rouge['avg_decode_speed']):
            f.write(f"Prefill: {best_rouge['avg_prefill_speed']:.2f} tok/s | Decode: {best_rouge['avg_decode_speed']:.2f} tok/s | Total: {best_rouge['avg_total_speed']:.2f} tok/s\n")
        f.write("\n")

        # Failed runs section
        if len(df_failed) > 0:
            f.write("⚠️  FAILED RUNS\n" + "=" * 80 + "\n")
            for idx, row in df_failed.iterrows():
                reason = "NaN ROUGE-L" if pd.isna(row['rouge_l']) else "ROUGE-L = 0.0"
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
        if rec['time_ms'] > 0:
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_prefill_speed += token_per_second
    if len([r for r in prefill_records if r['time_ms'] > 0]) > 0:
        avg_prefill_speed /= len([r for r in prefill_records if r['time_ms'] > 0])

# Calculate average decode speed with zero-division protection
avg_decode_speed = 0.0
if decode_records:
    for rec in decode_records:
        if rec['time_ms'] > 0:
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_decode_speed += token_per_second
    if len([r for r in decode_records if r['time_ms'] > 0]) > 0:
        avg_decode_speed /= len([r for r in decode_records if r['time_ms'] > 0])

# Calculate average total speed with zero-division protection
avg_total_speed = 0.0
if total_records:
    for rec in total_records:
        if rec['time_ms'] > 0:
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_total_speed += token_per_second
    if len([r for r in total_records if r['time_ms'] > 0]) > 0:
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

    # Create a temporary Python script for LongBench evaluation
    cat > "$run_dir/run_longbench.py" << 'PYPYTHON'
#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import re
from pathlib import Path
from datasets import load_dataset
import evaluate

# Configuration from bash - INJECTED BY HEREDOC BELOW

PYPYTHON

    # Inject configuration variables
    cat >> "$run_dir/run_longbench.py" << PYPYTHON
MODEL = "$model"
MODE = "$mode"
TEMP = $temp
REPEAT_PENALTY = $repeat_penalty
TOP_P = $top_p
TOP_K = $top_k
CTX_SIZE = $ctx_size
KEEP = $keep
BATCH_SIZE = $batch_size
UBATCH_SIZE = $ubatch_size
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
LOCAL_PROMPT_DIR = "./prompt_files"
DEVICE_PROMPT_PREFIX = "/data/local/tmp/prompt_files"
PYPYTHON

    # Append the rest of the Python script
    cat >> "$run_dir/run_longbench.py" << 'PYPYTHON'

def run_one_prompt(cli_path, prompt_device_path, output_path, stderr_file):
    """Run CLI with prompt file and capture output."""
    cmd = [
        "bash", cli_path,
        "-no-cnv",
        "-f", prompt_device_path,
        # Pass configuration
        "-t", str(THREADS),
        "-c", str(CTX_SIZE),
        "-b", str(BATCH_SIZE),
        "-ub", str(UBATCH_SIZE),
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
    ]

    if MMAP_FLAG:
        cmd.append(MMAP_FLAG)

    # Add system prompt if provided
    if SYSTEM_PROMPT:
        cmd.extend(["-sys", SYSTEM_PROMPT])

    start = time.time()
    with open(output_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True)
    end = time.time()

    latency = end - start
    if proc.returncode != 0:
        print(f"[ERROR] CLI failed for prompt {prompt_device_path}")
    return latency

def load_references():
    """Load qmsum test split and return reference map."""
    ds = load_dataset("zai-org/LongBench", "qmsum", split="test")
    ref_map = {}
    for i, rec in enumerate(ds):
        ans = rec["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            ref = ans[0]
        else:
            ref = ans
        ref_map[i] = ref.strip()
    return ref_map

def main():
    print(f"Loading references from LongBench qmsum test split...")
    ref_map = load_references()
    print(f"Loaded {len(ref_map)} references")

    # Create output directory for this run
    output_dir = os.path.join(RUN_DIR, "qmsum_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Open debug log
    stderr_file = open(os.path.join(RUN_DIR, 'debug.log'), 'w', encoding='utf-8')

    # Get all prompt files
    local_prompt_path = Path(LOCAL_PROMPT_DIR)
    prompt_files = sorted(local_prompt_path.glob("*.prompt.txt"))
    print(f"Found {len(prompt_files)} prompt files")

    latencies = []
    for pf in prompt_files:
        fname = pf.name  # e.g. "qmsum_test_0.prompt.txt"
        prompt_dev_path = os.path.join(DEVICE_PROMPT_PREFIX, fname)

        # Derive output filename
        base = fname
        if base.endswith(".prompt.txt"):
            base = base[:-len(".prompt.txt")]
        out_fname = base + ".txt"
        out_path = os.path.join(output_dir, out_fname)

        print(f"Running prompt {fname} → {out_fname}")
        latency = run_one_prompt("./run-cli-streamllm.sh", prompt_dev_path, out_path, stderr_file)
        print(f"  latency: {latency:.3f} s")
        latencies.append(latency)

    stderr_file.close()

    # Evaluate ROUGE scores
    print("\nEvaluating ROUGE scores...")
    rouge = evaluate.load("rouge")
    pattern = re.compile(r"qmsum_test_(\d+)\.txt")
    predictions = []
    references = []
    sample_ids = []

    for fname in sorted(os.listdir(output_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        outpath = os.path.join(output_dir, fname)
        with open(outpath, "r", encoding="utf-8") as f:
            pred = f.read().strip()

        if idx not in ref_map:
            print(f"Warning: no reference for sample {idx}, skipping")
            continue

        ref = ref_map[idx]
        sample_ids.append(idx)
        predictions.append(pred)
        references.append(ref)

    # Compute per-sample scores
    if predictions:
        per_sample = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
            use_aggregator=False
        )

        # Filter out samples with ROUGE-L == 0.0 for aggregated score
        filtered_predictions = []
        filtered_references = []
        for i, rl in enumerate(per_sample["rougeL"]):
            if rl > 0.0:
                filtered_predictions.append(predictions[i])
                filtered_references.append(references[i])

        if filtered_predictions:
            result = rouge.compute(
                predictions=filtered_predictions,
                references=filtered_references,
                use_stemmer=True
            )
        else:
            result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    else:
        result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    print("\n=== ROUGE Results ===")
    print(f"ROUGE-L: {result['rougeL']:.4f}")
    print(f"ROUGE-1: {result['rouge1']:.4f}")
    print(f"ROUGE-2: {result['rouge2']:.4f}")
    print(f"ROUGE-Lsum: {result['rougeLsum']:.4f}")

    # Output in parseable format
    print(f"ROUGE_L={result['rougeL']}")
    print(f"ROUGE_1={result['rouge1']}")
    print(f"ROUGE_2={result['rouge2']}")
    print(f"ROUGE_LSUM={result['rougeLsum']}")

if __name__ == "__main__":
    main()
PYPYTHON

    chmod +x "$run_dir/run_longbench.py"

    echo "Running LongBench qmsum test..." | tee -a "$LOG_FILE"
    local start_time=$(date +%s)

    # Run the Python evaluation script and save output
    python3 "$run_dir/run_longbench.py" 2>&1 | tee "$run_dir/eval_output.txt" | tee -a "$LOG_FILE"

    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))

    # Parse ROUGE scores from Python output
    local rouge_l="N/A"
    local rouge_1="N/A"
    local rouge_2="N/A"
    local rouge_lsum="N/A"
    if [ -f "$run_dir/eval_output.txt" ]; then
        rouge_l=$(grep "ROUGE_L=" "$run_dir/eval_output.txt" | tail -1 | sed 's/.*=//')
        rouge_1=$(grep "ROUGE_1=" "$run_dir/eval_output.txt" | tail -1 | sed 's/.*=//')
        rouge_2=$(grep "ROUGE_2=" "$run_dir/eval_output.txt" | tail -1 | sed 's/.*=//')
        rouge_lsum=$(grep "ROUGE_LSUM=" "$run_dir/eval_output.txt" | tail -1 | sed 's/.*=//')
    fi

    # Parse speed metrics from debug.log
    local speeds=$(parse_speeds "$run_dir/debug.log")
    local avg_prefill_speed=$(echo "$speeds" | awk '{print $1}')
    local avg_decode_speed=$(echo "$speeds" | awk '{print $2}')
    local avg_total_speed=$(echo "$speeds" | awk '{print $3}')

    # Default to empty if parsing failed
    [ -z "$avg_prefill_speed" ] && avg_prefill_speed=""
    [ -z "$avg_decode_speed" ] && avg_decode_speed=""
    [ -z "$avg_total_speed" ] && avg_total_speed=""

    echo "ROUGE-L: $rouge_l | ROUGE-1: $rouge_1 | ROUGE-2: $rouge_2 | Runtime: ${runtime}s" | tee -a "$LOG_FILE"
    echo "Prefill: $avg_prefill_speed tok/s | Decode: $avg_decode_speed tok/s | Total: $avg_total_speed tok/s" | tee -a "$LOG_FILE"

    # Append to CSV with ALL metrics - escape system_prompt for CSV
    local escaped_prompt=$(echo "$system_prompt" | sed 's/"/""/g')
    echo "${run_id},${model},${mode},${temp},${repeat_penalty},${top_p},${top_k},${ctx_size},${keep},${batch_size},${ubatch_size},${threads},${ngl},${ctk},${ctv},${flash_attn},${context_shift},${poll_level},${use_mmap},${split_mode},\"${escaped_prompt}\",${rouge_l},${rouge_1},${rouge_2},${rouge_lsum},${avg_prefill_speed},${avg_decode_speed},${avg_total_speed},${runtime}" >> "$RESULTS_CSV"

    update_best_results
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# MAIN SEARCH
################################################################################

NUM_TRIALS="${NUM_TRIALS:-100}"

echo "Running $NUM_TRIALS trials with LongBench qmsum..." | tee -a "$LOG_FILE"
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

    # KV cache compression only when flash attention is ON
    if [ "$flash_attn" = "on" ]; then
        ctk=$(get_random "${KV_CACHE_TYPES[@]}")
        ctv=$(get_random "${KV_CACHE_TYPES[@]}")
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
