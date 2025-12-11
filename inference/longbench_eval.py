#!/usr/bin/env python3

import os
import re
from datasets import load_dataset
import evaluate

def load_references():
    """
    Load the qmsum test split and return a dict mapping sample index -> reference summary.
    You must adapt this to the correct field name in the dataset.
    """
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

def evaluate_folder_outputs(output_dir: str, ref_map: dict):
    """
    Traverse output files (qmsum_test_{id}.txt) in output_dir,
    collect predictions and references, compute rouge via evaluate.
    Return results list and overall metrics.
    """
    rouge = evaluate.load("rouge")
    pattern = re.compile(r"qmsum_test_(\d+)\.txt")
    predictions = []
    references = []
    sample_ids = []

    for fname in sorted(os.listdir(output_dir)):
        m = pattern.match(fname)
        if not m:
            print(f"Skipping unrecognized file name: {fname}")
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
    per_sample = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        use_aggregator=False
    )

    # Build per-sample result list (idx, rougeL)
    results = []
    for i, idx in enumerate(sample_ids):
        rl = per_sample["rougeL"][i]
        results.append((idx, rl))

    # === Filter out samples with ROUGE-L == 0.0 for aggregated score ===
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
        # Fallback: no non-zero samples, return zeros
        result = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    return results, result

def main():
    output_dir = "qmsum_outputs"
    print("Loading references …")
    ref_map = load_references()
    print(f"Loaded {len(ref_map)} references.")

    print("Evaluating predictions in", output_dir)
    per_sample_scores, aggregated = evaluate_folder_outputs(output_dir, ref_map)

    print("\n=== ROUGE-L per sample ===")
    for idx, rl in per_sample_scores:
        print(f"Sample {idx}: ROUGE-L = {rl:.4f}")

    print("\n=== Aggregated ROUGE ===")
    # Print only rougeL, or print all metrics
    print(f"ROUGE-L (F1): {aggregated['rougeL']:.4f}")
    print(f"ROUGE-1: {aggregated['rouge1']:.4f}, ROUGE-2: {aggregated['rouge2']:.4f}, ROUGE-Lsum: {aggregated['rougeLsum']:.4f}")

if __name__ == "__main__":
    main()

