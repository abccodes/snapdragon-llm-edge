#!/usr/bin/env python3
import os
from datasets import load_dataset
import evaluate
import subprocess
import time
import numpy as np

def run_evaluate(extra_args=[]):

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    # Only evaluate the first x
    # Can change range or comment out line
    ds = ds.select(range(50))
    n = len(ds) 
    print(f"Loaded {n} test samples for Truthful QA")

    # initiate BLEURT evaluator model
    bleurt = evaluate.load('bleurt', 'bleurt-large-128') # load large model for accuracy

    #debug log
    stderr_file = open('debug.log', 'w', encoding='utf-8')
    max_score_arr = []
    acc_score_arr = []

    for i, rec in enumerate(ds):
        print(f"-------- sample {i} --------")
        question = rec['question']
        best_answer = rec['best_answer']
        correct_answers = rec['correct_answers']
        incorrect_answers = rec['incorrect_answers']

        question = question.replace("'", " ")
        question = question.replace('"', ' ')

        cmd = ["bash", "./run-cli.sh", "-no-cnv", "-p", f"\"\'{question} \'\"", "-n", str(25)] + extra_args
        start = time.time()
        with open("tmp_output.txt", "w", encoding="utf-8") as fout:
            
            print("CMD:", " ".join(cmd))
            # Note: we pass stderr=subprocess.PIPE so we can separately handle it
            proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True)
        end = time.time()

        latency = end - start
        if proc.returncode != 0:
            # Print stderr to console
            print(f"[ERROR] CLI failed for prompt {question}:")
            print(proc.stderr)
            return -1, -1
        
        # start evaluate
        with open("tmp_output.txt", "r", encoding='utf-8') as fin:
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
        
        os.remove("tmp_output.txt")
    
    print('=======================================')
    print('')
    accuracy = sum(acc_score_arr) / n
    print(f'avg max score: {np.mean(np.array(max_score_arr))}')
    print(f'avg accuracy: {accuracy:.3f}')


def main():
    run_evaluate()

if __name__ == "__main__":
    main()
