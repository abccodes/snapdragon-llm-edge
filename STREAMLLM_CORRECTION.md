# StreamingLLM Implementation Correction

## Summary

After reviewing the StreamingLLM paper (https://arxiv.org/abs/2309.17453), the sinks bias mechanism has been **disabled** to align with the paper's actual methodology.

## Key Insight from the Paper

The StreamingLLM paper's key contribution is the observation of the **"attention sink" phenomenon**:

> "We observe an interesting phenomenon, namely attention sink, that keeping the KV of initial tokens will largely recover the performance of window attention."

The paper demonstrates that:
1. Language models naturally attend heavily to the **first few tokens** 
2. These initial tokens act as a "sink" for attention scores
3. By simply **keeping these tokens in the KV cache** during sliding window attention, performance is maintained
4. **No additional bias or modification to attention scores is needed**

## What StreamingLLM Actually Does

```
Traditional Sliding Window:
[Token 1, Token 2, ..., Token N] → KV Cache
When cache full: evict Token 1, add Token N+1
Result: Attention collapse, poor performance

StreamingLLM:
[Token 1, Token 2, Token 3, Token 4] → Keep these (attention sinks)
[Token 5, Token 6, ..., Token N] → Sliding window
When cache full: evict Token 5, keep Tokens 1-4, add Token N+1
Result: Stable attention, good performance
```

The mechanism is purely about **cache management**, not attention modification.

## Previous Incorrect Implementations

### Version 1: Implicit Extra Token (Original)
```cpp
// INCORRECT: Treated sink_bias as an extra token
max = MAX(max, sk[i02]);
sum += expf(sk[i02] - max);
```
This added an implicit token to the softmax, which is not mentioned in the paper.

### Version 2: Biased Logits (My Previous "Fix")
```cpp
// INCORRECT: Added bias to first N token logits  
for (int64_t i = 0; i < n_sink; ++i) {
    wp[i] += sink_bias;
}
```
This artificially increased attention to initial tokens, which is also not in the paper.

## Correct Implementation

According to the paper, no bias mechanism is needed:

```cpp
// CORRECT: Normal softmax computation
float max = -INFINITY;
ggml_vec_max_f32(ne00, &max, wp);

ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
sum = 1.0/sum;
ggml_vec_scale_f32(ne00, dp, sum);
```

The attention sinks mechanism is implemented **entirely through KV cache management** using the `--keep` parameter, not through attention score modification.

## Parameter Usage

### `--keep N` (Essential)
Specifies how many initial tokens to keep in the KV cache during sliding window attention.
```bash
llama-cli -m model.gguf --context-shift --keep 4 -p "prompt"
```
This is the **core StreamingLLM mechanism**.

### `--sink-count N` (Not Used)
This parameter was intended to enable/disable the bias mechanism, but since no bias is needed per the paper, this parameter doesn't affect attention computation. It may still be used for other purposes (e.g., determining which tokens are "sinks" for visualization or debugging).

### `--sink-bias X` (Disabled)
This parameter is now disabled. The paper does not use any bias value.

## Why Initial Tokens Get High Attention

From the paper, the attention sink phenomenon occurs because:
1. Early tokens are present for the entire sequence duration
2. Models learn to use these positions as "sinks" during training
3. Even if semantically unimportant, these positions accumulate attention
4. This is a learned behavior from the training process

The paper leverages this **natural phenomenon** rather than artificially creating it with bias.

## Impact of This Change

### Before (Incorrect Implementations)
- Added artificial bias to attention scores
- Could cause garbled output or unnatural attention patterns
- Did not match the paper's methodology

### After (Correct Implementation)
- No modification to attention scores
- Purely cache-based mechanism via `--keep`
- Matches the StreamingLLM paper's approach
- Clean, coherent output with proper cache management

## Testing

To use StreamingLLM correctly:

```bash
# Correct usage - just use --keep
llama-cli -m model.gguf \
  --ctx-size 4096 \
  --context-shift \
  --keep 4 \
  -n 1000 \
  -p "Your long prompt..."

# The --sink-count and --sink-bias parameters can be omitted
# They don't affect attention computation anymore
```

## References

- **StreamingLLM Paper**: Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024
  - https://arxiv.org/abs/2309.17453
  - Key Quote: "keeping the KV of initial tokens will largely recover the performance of window attention"
  
- **Paper's Key Figure**: Figure 1 shows the attention pattern with initial token sinks receiving high natural attention

## Conclusion

The StreamingLLM mechanism is simpler than previously implemented:
- **What it IS**: A KV cache management strategy that keeps initial tokens
- **What it ISN'T**: An attention score modification or biasing mechanism

The previous implementations added complexity that isn't in the paper and could cause issues. The corrected implementation is now a pure cache management approach via the `--keep` parameter.
