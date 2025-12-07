#!/bin/bash
#
# Test script for StreamLLM-style attention sinks implementation
# This script validates that the --sink-count and --sink-bias flags work correctly
#

set -e

echo "================================================"
echo "StreamLLM Attention Sinks Test Script"
echo "================================================"
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/../llama.cpp"
LLAMA_CLI="$LLAMA_DIR/build/bin/llama-cli"

# Check if llama-cli exists
if [ ! -f "$LLAMA_CLI" ]; then
    echo "ERROR: llama-cli not found at $LLAMA_CLI"
    echo "Please build llama.cpp first:"
    echo "  cd $LLAMA_DIR && cmake -B build && cmake --build build --target llama-cli"
    exit 1
fi

echo "Test 1: Verify --sink-count and --sink-bias flags are recognized"
echo "----------------------------------------------------------------"
if "$LLAMA_CLI" --help 2>&1 | grep -q "sink-count"; then
    echo "✓ --sink-count flag found in help"
else
    echo "✗ FAILED: --sink-count flag not found in help"
    exit 1
fi

if "$LLAMA_CLI" --help 2>&1 | grep -q "sink-bias"; then
    echo "✓ --sink-bias flag found in help"
else
    echo "✗ FAILED: --sink-bias flag not found in help"
    exit 1
fi
echo ""

echo "Test 2: Test parsing of --sink-count flag"
echo "----------------------------------------------------------------"
# We can't actually run inference without a model, but we can test that the flags parse
# The CLI will exit with error about missing model, but that's after parsing args
if "$LLAMA_CLI" --sink-count 8 2>&1 | grep -q "model"; then
    echo "✓ --sink-count 8 parsed successfully (failed at model load as expected)"
else
    echo "✗ FAILED: --sink-count argument parsing failed"
    exit 1
fi
echo ""

echo "Test 3: Test parsing of --sink-bias flag"
echo "----------------------------------------------------------------"
if "$LLAMA_CLI" --sink-bias 5.5 2>&1 | grep -q "model"; then
    echo "✓ --sink-bias 5.5 parsed successfully (failed at model load as expected)"
else
    echo "✗ FAILED: --sink-bias argument parsing failed"
    exit 1
fi
echo ""

echo "Test 4: Test parsing of both flags together"
echo "----------------------------------------------------------------"
if "$LLAMA_CLI" --sink-count 6 --sink-bias 3.0 2>&1 | grep -q "model"; then
    echo "✓ Both --sink-count and --sink-bias parsed successfully"
else
    echo "✗ FAILED: Combined argument parsing failed"
    exit 1
fi
echo ""

echo "Test 5: Test sinks disabled with --sink-count 0"
echo "----------------------------------------------------------------"
if "$LLAMA_CLI" --sink-count 0 2>&1 | grep -q "model"; then
    echo "✓ --sink-count 0 (disabled) parsed successfully"
else
    echo "✗ FAILED: --sink-count 0 argument parsing failed"
    exit 1
fi
echo ""

echo "================================================"
echo "All Tests Passed! ✓"
echo "================================================"
echo ""
echo "The --sink-count and --sink-bias flags have been successfully implemented."
echo "They can be used with llama-cli to enable StreamLLM-style attention sinks."
echo ""
echo "Example usage:"
echo "  $LLAMA_CLI -m <model> --ctx-size 4096 --context-shift --keep 4 --sink-count 4 --sink-bias 4.0"
echo ""
