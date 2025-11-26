#!/bin/bash

MODEL="qwen2-7b-tinytron-Q4_K_M.gguf"
PROMPT="What is the capital of France?"

echo "=== Speed Comparison Test ==="
echo ""

# Test CPU
echo "Testing CPU..."
adb shell "cd /data/local/tmp/llama.cpp && \
    LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib \
    ./bin/llama-cli -m ../gguf/$MODEL \
        -ngl 0 -t 8 --batch-size 256 \
        -p '$PROMPT' -n 50 --seed 42" 2>&1 | tee cpu_test.log

# Test NPU
echo ""
echo "Testing NPU..."
adb shell "cd /data/local/tmp/llama.cpp && \
    LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib \
    ./bin/llama-cli -m ../gguf/$MODEL \
        --device HTP0 -ngl 99 -t 6 --batch-size 512 \
        -p '$PROMPT' -n 50 --seed 42" 2>&1 | tee npu_test.log

# Test GPU
echo ""
echo "Testing GPU..."
adb shell "cd /data/local/tmp/llama.cpp && \
    LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib \
    ./bin/llama-cli -m ../gguf/$MODEL \
        --device GPUOpenCL -ngl 99 -t 6 --batch-size 512 \
        -p '$PROMPT' -n 50 --seed 42" 2>&1 | tee gpu_test.log

# Parse results
echo ""
echo "=== RESULTS ==="
echo ""
echo "CPU:"
grep "prompt eval time" cpu_test.log | tail -1
grep "eval time" cpu_test.log | grep -v "prompt" | tail -1
echo ""
echo "NPU:"
grep "prompt eval time" npu_test.log | tail -1
grep "eval time" npu_test.log | grep -v "prompt" | tail -1
echo ""
echo "GPU:"
grep "prompt eval time" gpu_test.log | tail -1
grep "eval time" gpu_test.log | grep -v "prompt" | tail -1

# Summary
echo ""
echo "=== SUMMARY ==="
echo "CPU Decode Speed: $(grep "eval time" cpu_test.log | grep -v "prompt" | tail -1 | grep -oP '\d+\.\d+ tokens per second')"
echo "NPU Decode Speed: $(grep "eval time" npu_test.log | grep -v "prompt" | tail -1 | grep -oP '\d+\.\d+ tokens per second')"
echo "GPU Decode Speed: $(grep "eval time" gpu_test.log | grep -v "prompt" | tail -1 | grep -oP '\d+\.\d+ tokens per second')"
