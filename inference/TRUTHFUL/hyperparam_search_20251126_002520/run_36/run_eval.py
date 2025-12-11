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

MODEL = "LFM2-8B-A1B-Q4_K_M.gguf"
MODE = "CPU"
TEMP = 0.7
REPEAT_PENALTY = 1.5
TOP_P = 0.85
TOP_K = 20
CTX_SIZE = 2048
KEEP = 4
BATCH_SIZE = 256
UBATCH_SIZE = 256
THREADS = 4
NGL = 0
CTK = "f16"
CTV = "f16"
FLASH_ATTN = "on"
CONTEXT_SHIFT_FLAG = "--context-shift"
POLL_LEVEL = 100
MMAP_FLAG = "--no-mmap"
SYSTEM_PROMPT = ""
RUN_DIR = "hyperparam_search_20251126_002520/run_36"

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
