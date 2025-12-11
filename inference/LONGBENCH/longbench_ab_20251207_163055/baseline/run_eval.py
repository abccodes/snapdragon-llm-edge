#!/usr/bin/env python3
import os, subprocess, sys
from datasets import load_dataset
import evaluate

MODEL = "Llama-3.2-1B-Instruct-Q4_0.gguf"
THREADS = 8
CTX_SIZE = 512
TOKEN_LIMIT = 500
BATCH = 128
TEMP = 0.25
TOP_K = 40
TOP_P = 1.0
MIN_P = 0.10
REPEAT_PEN = 1.1
FLASH = "on"
KEEP = 0
CTX_SHIFT = "--no-context-shift"
OUTDIR = "longbench_ab_20251207_163055/baseline"

basedir = "/data/local/tmp/llama.cpp"
ds = load_dataset("zai-org/LongBench", "qmsum", split="test").select(range(8))
rouge = evaluate.load("rouge")
os.makedirs(os.path.join(OUTDIR, "outputs"), exist_ok=True)

preds, refs = [], []
stderr_log = open(os.path.join(OUTDIR, "debug.log"), "w")

for i, rec in enumerate(ds):
    ans = rec["answers"]
    ref = (ans[0] if isinstance(ans, list) else ans).strip()
    raw_input = rec["input"].replace("'", " ").replace('"', " ")
    prompt = f"Summarize the following meeting transcript:\n\n{raw_input}\n\nSummary:"

    llama_cmd = f"-m {basedir}/../gguf/{MODEL} "
    llama_cmd += f"-p '{prompt}' "
    llama_cmd += f"-n {TOKEN_LIMIT} -t {THREADS} -c {CTX_SIZE} -b {BATCH} "
    llama_cmd += f"-ngl 0 "
    llama_cmd += f"--temp {TEMP} --repeat-penalty {REPEAT_PEN} "
    llama_cmd += f"--top-k {TOP_K} --top-p {TOP_P} --min-p {MIN_P} "
    llama_cmd += f"--keep {KEEP} -fa {FLASH} {CTX_SHIFT} "
    llama_cmd += f"--no-display-prompt -no-cnv --no-mmap "

    adb_cmd = [
        "adb", "shell",
        f"cd {basedir}; "
        f"LD_LIBRARY_PATH={basedir}/./lib ADSP_LIBRARY_PATH={basedir}/./lib "
        f"././bin/llama-cli {llama_cmd}"
    ]

    outfile = os.path.join(OUTDIR, "outputs", f"out_{i}.txt")
    print(f"[sample {i}] running...")
    with open(outfile, "w") as fout:
        proc = subprocess.run(adb_cmd, stdout=fout, stderr=stderr_log, text=True)
    if proc.returncode != 0:
        print(f"[sample {i}] error code {proc.returncode}")
        continue

    with open(outfile) as fin:
        pred = fin.read().strip()
    if pred:
        preds.append(pred)
        refs.append(ref)
    else:
        print(f"[sample {i}] empty prediction")

stderr_log.close()

if not preds:
    print("ROUGE1=0.0"); print("ROUGE2=0.0"); print("ROUGEL=0.0"); print("ROUGELSUM=0.0")
    sys.exit(0)

try:
    per = rouge.compute(predictions=preds, references=refs, use_stemmer=True, use_aggregator=False)
    filtered_preds = [p for i,p in enumerate(preds) if per["rougeL"][i] > 0.0]
    filtered_refs  = [r for i,r in enumerate(refs)  if per["rougeL"][i] > 0.0]
    if filtered_preds:
        agg = rouge.compute(predictions=filtered_preds, references=filtered_refs, use_stemmer=True)
    else:
        agg = {"rouge1":0.0,"rouge2":0.0,"rougeL":0.0,"rougeLsum":0.0}
except Exception as e:
    print(f"ERROR in ROUGE: {e}")
    agg = {"rouge1":0.0,"rouge2":0.0,"rougeL":0.0,"rougeLsum":0.0}

print(f"ROUGE1={agg['rouge1']}")
print(f"ROUGE2={agg['rouge2']}")
print(f"ROUGEL={agg['rougeL']}")
print(f"ROUGELSUM={agg['rougeLsum']}")
