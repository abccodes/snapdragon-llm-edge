#!/bin/bash
# Debug test to show sinks tensor initialization logs

LLAMA_CLI="./llama.cpp/build/bin/llama-cli"

# Check if llama-cli exists
if [ ! -f "$LLAMA_CLI" ]; then
    echo "Error: llama-cli not found at $LLAMA_CLI"
    echo "Please build first: cd llama.cpp && cmake -B build && cmake --build build --target llama-cli"
    exit 1
fi

echo "==================================="
echo "Sinks Debug Test"
echo "==================================="
echo ""
echo "This test shows that sinks tensors are being created and initialized"
echo "when --sink-count > 0. Without a model, the process will fail early,"
echo "but you should see log messages about sinks creation."
echo ""
echo "Looking for log messages like:"
echo "  'build_sinks: creating sinks tensor with N heads...'"
echo ""

# Run with sinks enabled - will fail without model but should show logs
echo "Test 1: Running with --sink-count 4 --sink-bias 20.0"
echo "-------------------------------------------------------"
$LLAMA_CLI --version 2>&1 | grep -i "build\|llama"
echo ""
echo "Note: To see actual sinks initialization, you would need a valid model file."
echo "      The logs appear during graph building which requires model loading."
echo ""

echo "Test 2: Verify parameters are accepted"
echo "---------------------------------------"
if $LLAMA_CLI --help 2>&1 | grep -q "sink-count"; then
    echo "✓ --sink-count parameter exists"
else
    echo "✗ --sink-count parameter NOT found"
fi

if $LLAMA_CLI --help 2>&1 | grep -q "sink-bias"; then
    echo "✓ --sink-bias parameter exists"
else
    echo "✗ --sink-bias parameter NOT found"
fi

echo ""
echo "==================================="
echo "Expected Behavior with Real Model:"
echo "==================================="
echo ""
echo "When you run with a real model and --sink-count > 0, you should see:"
echo "  1. Log: 'build_sinks: creating sinks tensor with N heads, sink_count=X, sink_bias=Y'"
echo "  2. During inference, sinks will bias attention on first X tokens"
echo "  3. Different sink_bias values will produce different outputs"
echo ""
echo "To test with a real model:"
echo "  $LLAMA_CLI -m /path/to/model.gguf --ctx-size 512 --sink-count 4 --sink-bias 20.0 -n 10 -p 'Test'"
echo ""
