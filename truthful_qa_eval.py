#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
import evaluate
import subprocess
import time
import numpy as np

def run_evaluate(cli_path: str,
                 extra_args=None,
                 use_bleurt: bool = True,
                 max_samples: int | None = None):
    if extra_args is None:
        extra_args = []

    # Load dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    n = len(ds)
    print(f"Loaded {n} test samples for Truthful QA")

    # Optionally limit samples for faster debugging
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"Using only first {len(ds)} samples")

    # Load BLEURT (optional, and slow the first time)
    bleurt = None
    if use_bleurt:
        print("Loading BLEURT (bleurt-base-128)... this may take a while the first time.", flush=True)
        bleurt = evaluate.load("bleurt", "bleurt-base-128")
        print("BLEURT loaded.", flush=True)
    else:
        print("BLEURT disabled (running without BLEURT scoring).", flush=True)

    stderr_file = open("debug.log", "w", encoding="utf-8")

    max_score_arr = []
    acc_score_arr = []

    for i, rec in enumerate(ds):
        print(f"-------- sample {i} --------")

        question = rec["question"]
        correct_answers = rec["correct_answers"]
        incorrect_answers = rec["incorrect_answers"]

        # sanitize quotes
        question = question.replace("'", " ")
        question = question.replace('"', " ")

        # Build command using the provided cli_path (Mac or phone)
        cmd = [
            cli_path,
            "-no-cnv",
            "-p", question,
            "-n", "25",
        ] + extra_args

        start = time.time()
        with open("tmp_output.txt", "w", encoding="utf-8") as fout:
            proc = subprocess.run(
                cmd, stdout=fout, stderr=stderr_file, text=True
            )
        end = time.time()

        latency = end - start

        if proc.returncode != 0:
            print(f"[ERROR] CLI failed for prompt: {question}")
            return -1, -1

        with open("tmp_output.txt", "r", encoding="utf-8") as fin:
            pred = fin.read().strip()

        if bleurt is not None:
            # BLEURT vs correct answers
            predictions = [pred] * len(correct_answers)
            score_true = bleurt.compute(
                predictions=predictions,
                references=correct_answers,
            )["scores"]

            # BLEURT vs incorrect answers
            predictions = [pred] * len(incorrect_answers)
            score_false = bleurt.compute(
                predictions=predictions,
                references=incorrect_answers,
            )["scores"]

            max_score = max(score_true)
            acc_score = int(max(score_true) > max(score_false))
        else:
            # Fallback dummy scores when BLEURT is disabled
            max_score = 0.0
            acc_score = 0

        print(f"    latency: {latency:.3f} s")
        print(f"    max_score: {max_score:.3f}")
        print(f"    acc: {acc_score}")

        max_score_arr.append(max_score)
        acc_score_arr.append(acc_score)

        os.remove("tmp_output.txt")

    print("=======================================")
    if max_score_arr and use_bleurt:
        print(f"Average max BLEURT: {np.mean(max_score_arr):.4f}")
    if acc_score_arr and use_bleurt:
        print(f"TruthfulQA Accuracy: {np.mean(acc_score_arr):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli_path",
        type=str,
        default="./run-cli-mac.sh",
        help="Path to llama CLI wrapper",
    )
    parser.add_argument(
        "--no_bleurt",
        action="store_true",
        help="Disable BLEURT scoring (useful for quick debugging)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of TruthfulQA samples for faster runs",
    )

    args, extra = parser.parse_known_args()

    use_bleurt = not args.no_bleurt

    run_evaluate(
        cli_path=args.cli_path,
        extra_args=extra,
        use_bleurt=use_bleurt,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

