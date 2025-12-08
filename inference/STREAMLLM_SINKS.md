# StreamLLM-style Attention Sinks

## Overview

This implementation adds StreamLLM-style attention sinks to llama.cpp to prevent attention collapse in long-context scenarios with context shifting (sliding window attention).

## How It Works

StreamLLM addresses the "attention sink" phenomenon where models attend heavily to the first few tokens even when they're not semantically important. By explicitly biasing these tokens, we can:

1. Keep the first N tokens in the KV cache during context shifts (controlled by `--keep`)
2. Add attention bias to those tokens to maintain stable attention distributions (controlled by `--sink-bias`)

This prevents the model's attention from collapsing when the context window slides.

## Parameters

### `--sink-count N` (default: 4)
Controls whether attention sinks are enabled:
- `0` = disabled (no sinks tensor created)
- `> 0` = enabled (creates sinks tensor with bias)

This should typically match the `--keep` value to ensure the kept tokens receive the attention bias.

### `--sink-bias X` (default: 4.0)
**Note:** According to the StreamingLLM paper, no additional bias is needed for attention sinks. The paper's key insight is that initial tokens naturally receive high attention (the "attention sink" phenomenon). By keeping these tokens in the KV cache via `--keep`, we preserve this natural attention pattern without any artificial biasing.

This parameter is currently disabled in the implementation to align with the paper's methodology.

## Implementation Details

### Parameter Flow
```
CLI args (--sink-count, --sink-bias)
  ↓
common_params (sink_count, sink_bias)
  ↓
llama_context_params (sink_count, sink_bias)
  ↓
llama_cparams (sink_count, sink_bias)
  ↓
llm_graph_params (via cparams)
  ↓
llm_graph_context::build_sinks()
  ↓
Sinks tensor [n_head] with uniform bias
  ↓
Passed to build_attn() → ggml_soft_max_add_sinks()
```

### GGML Integration

**Updated Implementation:** Following the StreamingLLM paper (https://arxiv.org/abs/2309.17453), the sinks mechanism works by:

1. **Keeping initial tokens in KV cache**: Use `--keep N` to retain the first N tokens
2. **Natural attention**: These initial tokens naturally receive high attention due to the "attention sink" phenomenon
3. **No artificial bias**: The paper does not add any bias to attention scores - the natural attention pattern is sufficient

The previous implementation that added artificial bias has been disabled to align with the paper's methodology.

### Model Support

Currently implemented in:
- `src/models/llama.cpp` (the primary/reference model)

Other models can be updated by:
1. Calling `ggml_tensor * sinks = build_sinks();` before the layer loop
2. Passing `sinks` instead of `nullptr` to `build_attn()` calls

## Usage Examples

### Basic usage with context shifting
```bash
llama-cli -m model.gguf \
  --ctx-size 4096 \
  --context-shift \
  --keep 4 \
  --sink-count 4 \
  --sink-bias 4.0 \
  -p "Your prompt here"
```

### With run-cli-streamllm.sh
```bash
# Set environment variables to control behavior
export ENABLE_SINKS=1      # Enable sinks
export SINK_KEEP=4         # Keep first 4 tokens in KV cache
export SINK_COUNT=4        # Enable sinks for 4 tokens
export SINK_BIAS=4.0       # Bias value
export CTX_SIZE=4096       # Context window size
export N_PRED=1000         # Generate 1000 tokens (may exceed ctx to trigger shifting)

./inference/run-cli-streamllm.sh
```

### Disabling sinks
```bash
llama-cli -m model.gguf \
  --context-shift \
  --keep 4 \
  --sink-count 0  # Disables sinks
```

## Testing

Run the validation script:
```bash
./inference/test-sinks.sh
```

This validates that:
- CLI flags are recognized
- Arguments parse correctly
- Both flags work together
- Sinks can be disabled

## References

- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- Original GGML sinks implementation in `ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_soft_max_f32()`

## Future Work

Potential improvements:
- Per-head sink bias values (currently all heads use the same bias)
- Dynamic sink bias based on context length
- Update additional model implementations beyond llama
- Integration with other attention optimization techniques
