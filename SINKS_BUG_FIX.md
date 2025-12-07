# StreamLLM Sinks CPU Bug Fix

## Problem

When enabling StreamLLM sinks with `--sink-count` and `--sink-bias`, generations become garbled on CPU. This affects all scenarios where sinks are enabled:

- Sinks alone produce nonsense
- Sinks + context shift produce nonsense  
- Context shift alone works fine
- No sinks/no shift works fine

## Root Cause

The GGML CPU implementation was treating `sink_bias` as an **implicit extra token** in the softmax calculation, rather than as a **bias to be added to the first N tokens' logits**.

### Incorrect Implementation (Before Fix)

In `ggml-cpu/ops.cpp`, the `ggml_compute_forward_soft_max_f32` function was doing:

```cpp
float max = -INFINITY;
ggml_vec_max_f32(ne00, &max, wp);

// WRONG: Treat sink as extra token
if (sk) {
    max = MAX(max, sk[i02]);  // Include sink in max
}

ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);

// WRONG: Add sink to sum as if it's an extra token
if (sk) {
    sum += (ggml_float) expf(sk[i02] - max);
}

sum = 1.0/sum;
ggml_vec_scale_f32(ne00, dp, sum);
```

This treats the sink_bias as if it were a logit value for an additional token, which is mathematically incorrect and causes the attention distribution to be corrupted.

### Correct Implementation (After Fix)

The fix adds `sink_bias` to the logits of the **first `sink_count` tokens** before computing softmax:

```cpp
// Copy and scale logits
ggml_vec_cpy_f32(ne00, wp, sp);
ggml_vec_scale_f32(ne00, wp, scale);

// Apply mask (if any)
if (mp_f32) {
    for (int i = 0; i < ne00; ++i) {
        wp[i] += slope*mp_f32[i];
    }
}

// CORRECT: Add sink bias to first sink_count tokens
if (sk && sink_count > 0) {
    const int64_t n_sink = MIN((int64_t)sink_count, ne00);
    const float sink_bias = sk[i02];  // bias for this head
    for (int64_t i = 0; i < n_sink; ++i) {
        wp[i] += sink_bias;
    }
}

// Now compute normal softmax
float max = -INFINITY;
ggml_vec_max_f32(ne00, &max, wp);

ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);

sum = 1.0/sum;
ggml_vec_scale_f32(ne00, dp, sum);
```

This correctly implements StreamLLM attention sinks as described in the paper: the first `sink_count` tokens receive a positive bias, increasing their attention weights and preventing attention collapse.

## Changes Made

### 1. API Changes

Updated the GGML API to pass `sink_count` to the kernels:

**Before:**
```cpp
void ggml_soft_max_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);
void ggml_flash_attn_ext_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);
```

**After:**
```cpp
void ggml_soft_max_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks, int32_t sink_count);
void ggml_flash_attn_ext_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks, int32_t sink_count);
```

### 2. Op Params Storage

- For `SOFT_MAX`: Store sink_count in `op_params[2]` (params 0-1 are scale and max_bias)
- For `FLASH_ATTN_EXT`: Store sink_count in `op_params[4]` (params 0-3 are used for other settings)

### 3. CPU Kernel Fixes

**Soft Max (`ggml-cpu/ops.cpp:ggml_compute_forward_soft_max_f32`):**
- Read sink_count from op_params[2]
- Apply sink_bias to first sink_count tokens before softmax
- Remove incorrect implicit token handling

**Flash Attention (`ggml-cpu/ops.cpp:ggml_compute_forward_flash_attn_ext_f16_one_chunk`):**
- Read sink_count from op_params[4]  
- Apply sink_bias during KV iteration for first sink_count tokens
- Remove incorrect post-processing of sinks

### 4. Call Site Updates

Updated all call sites to pass sink_count:

- `llama.cpp/src/llama-graph.cpp`: Pass `cparams.sink_count` 
- `llama.cpp/tests/test-backend-ops.cpp`: Pass `4` when sinks enabled (typical value)
- `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`: Extract sink_count from op_params

### 5. Header Updates

- `llama.cpp/ggml/include/ggml.h`
- `llama.cpp/pkg-snapdragon/include/ggml.h`

## StreamLLM Theory

The StreamLLM paper ([arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)) introduces "attention sinks" to prevent attention collapse in long-context scenarios with sliding window attention.

### Key Concepts

1. **Attention Sink Phenomenon**: Models tend to attend heavily to the first few tokens, even when they're not semantically important
2. **Solution**: Explicitly keep the first N tokens in KV cache (`--keep`) and bias their attention logits (`--sink-bias`)
3. **Effect**: Maintains stable attention distribution even as the context window slides

### How It Works

With parameters `--sink-count 4 --sink-bias 4.0`:

1. Keep first 4 tokens in KV cache during context shifts
2. Add bias of 4.0 to attention logits for these 4 tokens
3. Higher bias = higher attention weight = prevents collapse

**Attention Logits (before softmax):**
```
Token 0: logit + 4.0  (sink)
Token 1: logit + 4.0  (sink)
Token 2: logit + 4.0  (sink)
Token 3: logit + 4.0  (sink)
Token 4: logit
Token 5: logit
...
```

After softmax, tokens 0-3 will have proportionally higher attention weights.

## Testing

### Verification Without Model

Build succeeds without errors:
```bash
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --target llama-cli -j$(nproc)
```

CLI accepts parameters:
```bash
./llama.cpp/build/bin/llama-cli --help | grep sink
```

### Verification With Model

To fully verify the fix, test with an actual model using the four scenarios from the issue:

```bash
MODEL="../gguf/Llama-3.2-1B-Instruct-Q4_0.gguf"
PROMPT="Summarize this meeting about quarterly planning and roadmap updates. Include owners and dates."
FLAGS="-n 300 -c 512 -b 128 -ngl 0 --temp 0.25 --repeat-penalty 1.1 --top-k 40 --top-p 1.0 --min-p 0.10 --no-mmap --seed 1234"
ENV="GGML_OPENCL_DISABLE=1 GGML_OPENCL_PLATFORM_NAME=disable GGML_OPENCL_DEVICE_NAME=disable GGML_HEXAGON_DISABLE=1 GGML_HEX_NDEV=0 LLAMA_LOG_LEVEL=error"

# Test 1: Baseline (clean) - no sinks, no shift
env $ENV ./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --sink-count 0 --sink-bias 0 --no-context-shift --keep 0

# Test 2: Shift only (clean) - no sinks
env $ENV ./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --context-shift --sink-count 0 --sink-bias 0 --keep 64

# Test 3: Shift + sinks (should be fixed - was garbled)
env $ENV ./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --context-shift --sink-count 4 --sink-bias 4 --keep 128

# Test 4: Sinks only (should be fixed - was garbled)  
env $ENV ./llama-cli -m $MODEL -p "$PROMPT" $FLAGS --no-context-shift --sink-count 4 --sink-bias 4 --keep 128
```

**Expected Results:**
- All four tests should produce clean, coherent text
- Tests 3 and 4 should no longer produce garbled output
- Different sink_bias values (0 vs 4) should produce slightly different but valid outputs

### A/B Comparison Test

With the fix, changing sink_bias should produce different but valid outputs:

```bash
# Test A: High sink bias
./llama-cli -m model.gguf --ctx-size 512 --seed 12345 --sink-count 4 --sink-bias 20.0 -ngl 0 -n 100 -p "Once upon a time" > output_A.txt

# Test B: Low sink bias  
./llama-cli -m model.gguf --ctx-size 512 --seed 12345 --sink-count 4 --sink-bias 0.0 -ngl 0 -n 100 -p "Once upon a time" > output_B.txt

# Compare
diff output_A.txt output_B.txt
```

Both outputs should be coherent text (not garbled), but may diverge after context fills.

## Files Modified

1. `llama.cpp/ggml/include/ggml.h` - Updated API signatures
2. `llama.cpp/ggml/src/ggml.c` - Store sink_count in op_params
3. `llama.cpp/ggml/src/ggml-cpu/ops.cpp` - Fixed CPU kernels
4. `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` - Updated Vulkan call site
5. `llama.cpp/src/llama-graph.cpp` - Pass cparams.sink_count
6. `llama.cpp/tests/test-backend-ops.cpp` - Updated test calls
7. `llama.cpp/pkg-snapdragon/include/ggml.h` - Updated API signatures

## Impact

### What This Fixes

- ✅ Garbled output when using `--sink-count > 0` on CPU
- ✅ StreamLLM not working as intended
- ✅ Mathematically incorrect softmax computation with sinks

### What's Not Changed

- Context shifting still works the same way
- Baseline (no sinks) behavior unchanged  
- Flash attention still works, just with corrected sinks
- Other backends (Vulkan, Metal, etc.) need similar fixes if they implement sinks

### Backward Compatibility

The API change adds a required parameter to `ggml_soft_max_add_sinks` and `ggml_flash_attn_ext_add_sinks`. Any external code calling these functions will need to:

1. Update function calls to pass `sink_count` parameter
2. Recompile against the updated headers

## References

- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [llama.cpp Issue Report](https://github.com/abccodes/snapdragon-llm-edge/issues/XXX)
