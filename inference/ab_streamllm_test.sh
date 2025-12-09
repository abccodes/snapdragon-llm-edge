#!/usr/bin/env bash
set -euo pipefail

# Simple A/B: baseline (no context shift) vs StreamLLM (context shift on)
# Dataset: LongBench qmsum, first 8 samples
#
# Configurable via env:
#   MODEL       (default: Llama-3.2-1B-Instruct-Q4_0.gguf)
#   THREADS     (default: 8)
#   CTX_SIZE    (default: 512)
#   TOKEN_LIMIT (default: 500)
#   BATCH       (default: 128)
#   TEMP        (default: 0.25)
#   TOP_K       (default: 40)
#   TOP_P       (default: 1.0)
#   MIN_P       (default: 0.10)
#   REPEAT_PEN  (default: 1.1)
#   FLASH       (default: on)
#   KEEP_BASE   (default: 0)     # keep for baseline
#   KEEP_STREAM (default: 64)    # keep for StreamLLM
#   RUN_DIR     (default: longbench_ab_${timestamp})

MODEL="${MODEL:-Llama-3.2-1B-Instruct-Q4_0.gguf}"
THREADS="${THREADS:-8}"
CTX_SIZE="${CTX_SIZE:-512}"
TOKEN_LIMIT="${TOKEN_LIMIT:-500}"
BATCH="${BATCH:-128}"
TEMP="${TEMP:-0.25}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-1.0}"
MIN_P="${MIN_P:-0.10}"
REPEAT_PEN="${REPEAT_PEN:-1.1}"
FLASH="${FLASH:-on}"
KEEP_BASE="${KEEP_BASE:-0}"
KEEP_STREAM="${KEEP_STREAM:-4}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-longbench_ab_${TS}}"
mkdir -p "$RUN_DIR"/{baseline,streamllm}

echo "A/B LongBench qmsum"
echo "Model: $MODEL"
echo "Run dir: $RUN_DIR"
echo

gen_py_runner() {
  local outdir="$1"
  local ctx_shift_flag="$2"   # "--context-shift" or "--no-context-shift"
  local keep_tokens="$3"

  cat > "${outdir}/run_eval.py" <<PY
#!/usr/bin/env python3
import os, subprocess, sys
from datasets import load_dataset
import evaluate

MODEL = "${MODEL}"
THREADS = ${THREADS}
CTX_SIZE = ${CTX_SIZE}
TOKEN_LIMIT = ${TOKEN_LIMIT}
BATCH = ${BATCH}
TEMP = ${TEMP}
TOP_K = ${TOP_K}
TOP_P = ${TOP_P}
MIN_P = ${MIN_P}
REPEAT_PEN = ${REPEAT_PEN}
FLASH = "${FLASH}"
KEEP = ${keep_tokens}
CTX_SHIFT = "${ctx_shift_flag}"
OUTDIR = "${outdir}"

basedir = "/data/local/tmp/llama.cpp"
ds = load_dataset("zai-org/LongBench", "qmsum", split="test").select(range(100))
rouge = evaluate.load("rouge")
os.makedirs(os.path.join(OUTDIR, "outputs"), exist_ok=True)

preds, refs = [], []
stderr_log = open(os.path.join(OUTDIR, "debug.log"), "w")

for i, rec in enumerate(ds):
    ans = rec["answers"]
    ref = (ans[0] if isinstance(ans, list) else ans).strip()
    raw_input = rec["input"].replace("'", " ").replace('"', " ")
    prompt = f"Summarize the following meeting transcript:\\n\\n{raw_input}\\n\\nSummary:"

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
PY
  chmod +x "${outdir}/run_eval.py"
}

gen_py_runner "${RUN_DIR}/baseline"  "--no-context-shift" "${KEEP_BASE}"
gen_py_runner "${RUN_DIR}/streamllm" "--context-shift"    "${KEEP_STREAM}"

run_and_log() {
  local tag="$1" outdir="$2"
  echo "=== ${tag} ==="
  python3 "${outdir}/run_eval.py" | tee "${outdir}/eval_output.txt"
  echo
}

run_and_log "Baseline (no context shift)" "${RUN_DIR}/baseline"
run_and_log "StreamLLM (context shift on)" "${RUN_DIR}/streamllm"

echo "Done. Results in: ${RUN_DIR}"
