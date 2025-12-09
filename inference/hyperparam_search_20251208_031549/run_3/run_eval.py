#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import numpy as np

# FIX: Disable TensorFlow before importing evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from datasets import load_dataset
import evaluate

# Configuration from bash - INJECTED BY HEREDOC BELOW

MODEL = "Llama-3.2-1B-Instruct-Q4_0.gguf"
MODE = "CPU"
TEMP = 0.2
REPEAT_PENALTY = 1.1
TOP_P = 1.0
TOP_K = 40
MIN_P = 0.15
CTX_SIZE = 2048
KEEP = 4
BATCH_SIZE = 128
UBATCH_SIZE = 128
THREADS = 8
NGL = 0
CTK = "f16"
CTV = "q8_0"
FLASH_ATTN = "on"
CONTEXT_SHIFT_FLAG = "--context-shift"
POLL_LEVEL = 30
MMAP_FLAG = "--no-mmap"
DRY_MULT = 1.0
FREQ_PENALTY = 0.15
PRESENCE_PENALTY = 0.3
TOKEN_LIMIT = 600
SYSTEM_PROMPT = ""
RUN_DIR = "hyperparam_search_20251208_031549/run_3"

# ADB and device configuration
BASEDIR = "/data/local/tmp/llama.cpp"
BRANCH = "."

# Load dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
ds = ds.select(range(300))
n = len(ds)
print(f"Loaded {n} test samples for Truthful QA")

# initiate BLEURT evaluator model
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

    # Clean question - escape for shell
    question_escaped = question.replace("'", "'\\''")
    
    # Build the adb shell command directly
    # Format the prompt argument carefully for shell
    prompt_arg = f'"{{\\'\\'{question_escaped}\\'\\'"}}"'
    
    # Build llama-cli arguments
    llama_args = [
        "-m", f"{BASEDIR}/../gguf/{MODEL}",
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
        "--min-p", str(MIN_P),
        "--keep", str(KEEP),
        "-fa", FLASH_ATTN,
        CONTEXT_SHIFT_FLAG,
        "--poll", str(POLL_LEVEL),
        "-ngl", str(NGL),
        "-n", str(TOKEN_LIMIT),
        "--no-display-prompt",
        "-no-cnv",
    ]
    
    # Add sampling parameters
    if DRY_MULT > 0:
        llama_args.extend(["--dry-multiplier", str(DRY_MULT)])
        llama_args.extend(["--dry-base", "1.75"])
    
    if FREQ_PENALTY > 0:
        llama_args.extend(["--frequency-penalty", str(FREQ_PENALTY)])
    
    if PRESENCE_PENALTY > 0:
        llama_args.extend(["--presence-penalty", str(PRESENCE_PENALTY)])
    
    if MMAP_FLAG:
        llama_args.append(MMAP_FLAG)
    
    # Add prompt
    llama_args.extend(["-p", prompt_arg])
    
    # Build full adb command
    llama_args_str = " ".join(llama_args)
    
    adb_cmd = f"""adb shell 'cd {BASEDIR}; ulimit -c unlimited; \\
        LD_LIBRARY_PATH={BASEDIR}/{BRANCH}/lib \\
        ADSP_LIBRARY_PATH={BASEDIR}/{BRANCH}/lib \\
        ./{BRANCH}/bin/llama-cli {llama_args_str}'"""
    
    print(f"CMD: {adb_cmd}")
    
    start = time.time()
    
    # Execute via shell to properly handle the complex quoting
    with open(os.path.join(RUN_DIR, f"tmp_output_{i}.txt"), "w", encoding="utf-8") as fout:
        proc = subprocess.run(
            adb_cmd,
            shell=True,
            stdout=fout,
            stderr=stderr_file,
            text=True
        )
    
    end = time.time()
    latency = end - start
    
    if proc.returncode != 0:
        print(f"[ERROR] CLI failed for prompt: {question}")
        continue

    # start evaluate
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

# Calculate final metrics
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
