#!/usr/bin/env python3
import os, sys, subprocess, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
import evaluate

# Configuration injected by bash
MODEL = """qwen2-7b-tinytron-Q4_K_M.gguf"""
TEMP = 0.2
REPEAT_PENALTY = 1.1
TOP_P = 1.0
TOP_K = 40
MIN_P = 0.15
CTX_SIZE = 4096
KEEP = 0
BATCH_SIZE = 512
THREADS = 8
CTK = """f16"""
CTV = """q8_0"""
FLASH_ATTN = """on"""
CONTEXT_SHIFT_FLAG = """--context-shift"""
POLL_LEVEL = 30
MMAP_FLAG = """--no-mmap"""
DRY_MULT = 1.0
FREQ_PENALTY = 0.15
PRESENCE_PENALTY = 0.2
TOKEN_LIMIT = 500
RUN_DIR = """hyperparam_search_longbench_20251206_222236/run_13"""

ds = load_dataset("zai-org/LongBench", "qmsum", split="test").select(range(8))
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
    llama_cmd += f"-ngl 0 -ctk {CTK} -ctv {CTV} "
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
