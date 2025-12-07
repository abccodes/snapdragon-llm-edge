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
The bias value applied to sink tokens in the attention softmax. This value is added as an implicit token in the softmax calculation, preventing attention weights from going to zero.

Higher values increase the attention weight on sink tokens.

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

The sinks tensor is a 1D float tensor of shape `[n_head]` where each element contains the sink bias value. During softmax computation, GGML:

1. Treats the bias as if there's an implicit extra token with that value
2. Includes it in the max calculation: `max = MAX(max, sinks[head])`
3. Includes it in the sum: `sum += exp(sinks[head] - max)`
4. This prevents attention weights from collapsing to near-zero values

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
