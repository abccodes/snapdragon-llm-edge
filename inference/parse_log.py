#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

# Match the context print line
CONTEXT_RE = re.compile(r'^llama_perf_context_print:')

# Match floats or ints
NUM_RE = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?')

def parse_line_numbers(line: str):
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

def main():
    ap = argparse.ArgumentParser(description="Extract prefill, decode, and total speeds from llama.cpp logs.")
    ap.add_argument("logfile", type=Path, help="Path to the log file to parse")
    args = ap.parse_args()

    if not args.logfile.is_file():
        raise SystemExit(f"File not found: {args.logfile}")

    prefill_records = []
    decode_records = []
    total_records = []
    
    with args.logfile.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if "prompt eval time" in line and CONTEXT_RE.match(line):
                floats, ints = parse_line_numbers(line)
                # Format: "prompt eval time = XXX.XX ms / YY tokens"
                if len(floats) > 0 and len(ints) > 0:
                    prefill_records.append({
                        "time_ms": floats[0],
                        "tokens": ints[0],
                        "line": line.rstrip("\n"),
                    })
            elif "eval time" in line and CONTEXT_RE.match(line) and "prompt eval" not in line:
                floats, ints = parse_line_numbers(line)
                # Format: "eval time = XXX.XX ms / YY runs"
                if len(floats) > 0 and len(ints) > 0:
                    decode_records.append({
                        "time_ms": floats[0],
                        "tokens": ints[0],
                        "line": line.rstrip("\n"),
                    })
            elif "total time" in line and CONTEXT_RE.match(line):
                floats, ints = parse_line_numbers(line)
                # Format: "total time = XXX.XX ms / YY tokens"
                if len(floats) > 0 and len(ints) > 0:
                    total_records.append({
                        "time_ms": floats[0],
                        "tokens": ints[0],
                        "line": line.rstrip("\n"),
                    })

    # Calculate average prefill speed
    if prefill_records:
        avg_prefill_speed = 0
        for rec in prefill_records:
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_prefill_speed += token_per_second
        avg_prefill_speed /= len(prefill_records)
        print(f"Average Prefill Speed: {avg_prefill_speed:.2f} tokens/s")
    else:
        print("No prefill records found")

    # Calculate average decode speed
    if decode_records:
        avg_decode_speed = 0
        for rec in decode_records:
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_decode_speed += token_per_second
        avg_decode_speed /= len(decode_records)
        print(f"Average Decode Speed: {avg_decode_speed:.2f} tokens/s")
    else:
        print("No decode records found")

    # Calculate average total speed (original metric)
    if total_records:
        avg_total_speed = 0
        for rec in total_records:
            token_per_second = rec['tokens'] / rec['time_ms'] * 1000
            avg_total_speed += token_per_second
        avg_total_speed /= len(total_records)
        print(f"Average Total Speed (prefill + decode): {avg_total_speed:.2f} tokens/s")
    else:
        print("No total records found")

if __name__ == "__main__":
    main()
