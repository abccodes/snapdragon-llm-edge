#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

PREFIX_RE = re.compile(r'^llama_perf_context_print:\s+total time\s*=')
# Match floats or ints (supports scientific notation and leading + / -)
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
            # classify pure integers separately
            try:
                ints.append(int(s))
            except ValueError:
                pass
    return floats, ints

def main():
    ap = argparse.ArgumentParser(description="Extract numbers from llama_perf_context_print total-time lines.")
    ap.add_argument("logfile", type=Path, help="Path to the log file to parse")
    ap.add_argument("--csv", type=Path, help="Optional path to write a CSV summary")
    args = ap.parse_args()

    if not args.logfile.is_file():
        raise SystemExit(f"File not found: {args.logfile}")

    records = []
    with args.logfile.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if PREFIX_RE.match(line):
                floats, ints = parse_line_numbers(line)
                records.append({
                    "line_number": lineno,
                    "floats": floats,
                    "ints": ints,
                    "line": line.rstrip("\n"),
                })

    # Print to stdout as JSON for easy consumption
    avg_tok_speed = 0
    for rec in records:
        token_per_second = rec['ints'][0] / rec['floats'][0] * 1000
        avg_tok_speed += token_per_second
    
    avg_tok_speed /= len(records)
    print(f"Average token speed: {avg_tok_speed} tokens/s")

if __name__ == "__main__":
    main()
