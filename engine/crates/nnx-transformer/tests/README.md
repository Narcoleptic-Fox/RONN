# Integration Tests

These tests validate NNX against real model files. They are skipped automatically when no model is available, so they never break CI.

## Running

Set `NNX_TEST_MODEL` to a GGUF model path:

```bash
NNX_TEST_MODEL=path/to/model.gguf cargo test -p nnx-transformer --test integration
```

To run with output visible (recommended for speed benchmarks):

```bash
NNX_TEST_MODEL=path/to/model.gguf cargo test -p nnx-transformer --test integration -- --nocapture
```

## Recommended Test Models

| Model | Size | Format | Notes |
|-------|------|--------|-------|
| SmolLM-135M | ~145MB | Q8_0 GGUF | Fastest, good for CI |
| TinyLlama-1.1B-Chat | ~700MB | Q4_K_M GGUF | Good balance of size and realism |
| Phi-2 | ~1.6GB | Q4_0 GGUF | Larger, tests dequantization paths |

## What the tests cover

| Test | Validates |
|------|-----------|
| `test_load_real_model` | GGUF parsing, weight loading, config extraction |
| `test_model_info_complete` | All metadata fields populated and dimensionally consistent |
| `test_forward_produces_valid_logits` | Forward pass produces finite, non-zero, non-exploded logits |
| `test_forward_deterministic` | Same input produces same output across separate loads |
| `test_sequential_tokens_update_cache` | KV cache position advances correctly during decode |
| `test_cache_clear` | Cache reset works after forward passes |
| `test_generate_text` | End-to-end text generation (if tokenizer is embedded) |
| `test_generation_speed` | Benchmark: tokens/second measurement |
