# NNX — Neural Network eXecution Engine

A pure Rust inference engine providing model loading, quantization, and compute
kernels. Designed as a standalone library that any orchestration layer (like
RONN) can build upon.

## Crate Overview

| Crate | Purpose |
|---|---|
| `nnx-core` | Core types: tensor, dtype, device, error, and the `InferenceEngine` trait |
| `nnx-gguf` | GGUF file format parser with memory-mapped loading |
| `nnx-safetensors` | SafeTensors weight loading |
| `nnx-onnx` | ONNX model/graph loading |
| `nnx-ggml` | Legacy GGML format loading (optional) |
| `nnx-quant` | Quantization block types and dequantization kernels |
| `nnx-kernels` | Optimized compute: matmul, softmax, RMSNorm, RoPE, SIMD |
| `nnx-transformer` | Transformer architecture: attention, FFN, full model forward pass |

## Design Principles

- **Pure Rust** — zero FFI, zero C/C++ dependencies
- **No orchestration opinions** — no routing, scheduling, or cognitive architecture
- **Memory-mapped loading** — mmap for fast startup, reduced memory pressure
- **Quantization-native** — first-class support for Q4_K, Q8_0, and friends
- **Extractable** — this directory can become its own repo at any time
