#!/usr/bin/env python3
import os
from datasets import load_dataset
import evaluate
import subprocess
import time
import numpy as np
import sys

# Configuration from bash
MODEL = "LFM2-8B-A1B-Q4_K_M.gguf"
MODE = "CPU"
TEMP = 1.0
REPEAT_PENALTY = 1.1
TOP_P = 0.9
TOP_K = 40
CTX_SIZE = 8192
KEEP = 4
BATCH_SIZE = 256
UBATCH_SIZE = 512
THREADS = 6
NGL = 0
CTK = "q8_0"
CTV = "f16"
FLASH_ATTN = "off"
CONTEXT_SHIFT_FLAG = "--context-shift"
POLL_LEVEL = 0
MMAP_FLAG = ""
RUN_DIR = "hyperparam_search_20251125_005256/run_9"

# Load dataset - exactly like truthful_qa_eval.py
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
ds = ds.select(range(50))
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
    # Note: We bypass run-cli-streamllm.sh and call adb directly with all flags
    cmd = [
        "bash", "./run-cli-streamllm.sh",
        "-no-cnv",
        "-p", f"\"\'{question} \'\"",
        "-n", "25",
        # Pass extra args to override defaults
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
accuracy = sum(acc_score_arr) / n
avg_bleurt = np.mean(np.array(max_score_arr))
print(f'avg max score: {avg_bleurt}')
print(f'avg accuracy: {accuracy:.3f}')

# Output in parseable format
print(f"BLEURT_AVG={avg_bleurt}")
print(f"ACCURACY={accuracy}")
