#!/bin/bash

# Find the most recent search directory
LATEST_DIR=$(ls -td hyperparam_search_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No hyperparameter search runs found."
    exit 1
fi

echo "Checking progress of: $LATEST_DIR"
echo ""

if [ -f "$LATEST_DIR/BEST_RESULTS.txt" ]; then
    cat "$LATEST_DIR/BEST_RESULTS.txt"
else
    echo "No results yet. Search may have just started."
fi

echo ""
echo "----------------------------------------"
echo "Raw CSV results:"
if [ -f "$LATEST_DIR/results.csv" ]; then
    echo "Total runs: $(tail -n +2 "$LATEST_DIR/results.csv" | wc -l)"
    echo ""
    echo "Last 5 runs:"
    tail -5 "$LATEST_DIR/results.csv" | column -t -s,
fi

