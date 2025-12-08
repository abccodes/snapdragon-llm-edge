# StreamLLM Sinks CPU Bug Fix - Summary

## Overview
Fixed a critical bug in the StreamLLM attention sinks implementation that caused garbled output on CPU when `--sink-count > 0`.

## Problem
The GGML CPU implementation was mathematically incorrect:
- It treated `sink_bias` as an **implicit extra token** in softmax
- Added `exp(sink_bias - max)` to the softmax sum
- This corrupted the attention distribution and caused nonsense output

## Solution
Changed the implementation to correctly apply StreamLLM attention sinks:
- Add `sink_bias` to the **logits of the first `sink_count` tokens**
- Then compute normal softmax over all tokens
- This matches the StreamLLM paper methodology

## Changes Summary

### API Changes
- `ggml_soft_max_add_sinks(tensor, sinks)` → `ggml_soft_max_add_sinks(tensor, sinks, sink_count)`
- `ggml_flash_attn_ext_add_sinks(tensor, sinks)` → `ggml_flash_attn_ext_add_sinks(tensor, sinks, sink_count)`

### Core Fix
**In `ggml-cpu/ops.cpp` - `ggml_compute_forward_soft_max_f32`:**

Before (BROKEN):
```cpp
float max = -INFINITY;
ggml_vec_max_f32(ne00, &max, wp);
if (sk) {
    max = MAX(max, sk[i02]);  // WRONG: include sink in max
}
ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
if (sk) {
    sum += (ggml_float) expf(sk[i02] - max);  // WRONG: add as extra token
}
```

After (FIXED):
```cpp
// Apply sink bias to first sink_count tokens
if (sk && sink_count > 0) {
    const int64_t n_sink = MIN((int64_t)sink_count, ne00);
    const float sink_bias = sk[i02];
    for (int64_t i = 0; i < n_sink; ++i) {
        wp[i] += sink_bias;  // CORRECT: bias the logits
    }
}

// Then compute normal softmax
float max = -INFINITY;
ggml_vec_max_f32(ne00, &max, wp);
ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
```

**Similar fix in flash attention:**
Apply bias during KV iteration for first `sink_count` tokens.

### Files Modified
1. `llama.cpp/ggml/include/ggml.h` - Updated API signatures
2. `llama.cpp/ggml/src/ggml.c` - Store sink_count in op_params
3. `llama.cpp/ggml/src/ggml-cpu/ops.cpp` - Fixed CPU kernels
4. `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` - Updated Vulkan backend
5. `llama.cpp/src/llama-graph.cpp` - Pass cparams.sink_count
6. `llama.cpp/tests/test-backend-ops.cpp` - Updated tests
7. `llama.cpp/pkg-snapdragon/include/ggml.h` - Updated Snapdragon API

## Testing Instructions

### Quick Test (No Model)
```bash
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --target llama-cli -j$(nproc)
./build/bin/llama-cli --help | grep sink
```

Should show:
```
--sink-count N                          number of attention sink tokens...
--sink-bias N                           bias value for attention sinks...
```

### Full Test (With Model)
Test the four scenarios from the issue:

```bash
MODEL="path/to/model.gguf"
PROMPT="Summarize this meeting about quarterly planning."
FLAGS="-n 300 -c 512 -b 128 -ngl 0 --temp 0.25 --seed 1234"

# 1. Baseline (should be clean)
./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --sink-count 0 --no-context-shift

# 2. Shift only (should be clean)
./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --context-shift --sink-count 0 --keep 64

# 3. Shift + sinks (FIXED - was garbled)
./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --context-shift --sink-count 4 --sink-bias 4 --keep 128

# 4. Sinks only (FIXED - was garbled)
./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --no-context-shift --sink-count 4 --sink-bias 4 --keep 128
```

**Expected:** All four tests produce clean, coherent text (scenarios 3 and 4 should now work).

## Impact

### What This Fixes
✅ Garbled output when using `--sink-count > 0` on CPU  
✅ StreamLLM not working as intended  
✅ Mathematically incorrect softmax computation with sinks

### What Remains Unchanged
- Context shifting still works the same way
- Baseline (no sinks) behavior unchanged
- Flash attention still works, just with corrected sinks
- Performance characteristics unchanged

## References

- **Issue**: StreamLLM sinks cause garbled output on CPU
- **StreamLLM Paper**: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- **Detailed Documentation**: See [SINKS_BUG_FIX.md](SINKS_BUG_FIX.md)

## Code Quality
✅ All code review comments addressed  
✅ Type-safe parameter access  
✅ Well-documented parameter layouts  
✅ Consistent coding patterns  
✅ Clean compilation with no warnings  
✅ Security scan passed
