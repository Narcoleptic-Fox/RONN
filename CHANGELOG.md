# Changelog

All notable changes to RONN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- **Core Runtime**: Tensor operations with Candle backend
- **ONNX Compatibility**: 44 operators covering math, activations, neural network layers, tensor ops, and reductions
- **Execution Providers**:
  - CPU Provider with SIMD optimization (AVX2/AVX-512/NEON)
  - GPU Provider with multi-GPU and Tensor Core support
  - BitNet Provider for 1.58-bit quantization
  - WebAssembly Provider for browser deployment
  - Custom Provider plugin system
- **Brain-Inspired Features**:
  - Hierarchical Reasoning Module (HRM) with System 1/2 routing
  - Multi-tier memory system (working, episodic, semantic)
  - Continual learning with EWC and replay
- **High-Level API**: Model loading, session management, builder patterns
- **Python Bindings**: Full PyO3 integration with NumPy support
- **Performance Features**:
  - Async inference with Tokio
  - Batch processing
  - Memory pooling
  - Graph optimization (constant folding, fusion, DCE)
- **Observability**: Structured logging with tracing, profiling infrastructure
- **Documentation**: Comprehensive README, API docs, contributing guide

### Architecture
```
ronn-api       → High-level user API
ronn-core      → Core tensor operations, session management
ronn-onnx      → ONNX parsing and compatibility
ronn-graph     → Graph optimization pipeline
ronn-providers → Execution providers (CPU, GPU, BitNet, WASM)
ronn-hrm       → Hierarchical Reasoning Module
ronn-memory    → Multi-tier memory system
ronn-learning  → Continual learning engine
ronn-python    → Python bindings
```

### Performance
- HRM routing latency: ~2µs per decision
- Memory pooling: 1.2-1.5x allocation speedup
- SIMD vectorization: 2-8x math speedup
- GPU acceleration: 20-50x vs CPU for large models

### Known Limitations
- Integration tests with real models require manual model downloads
- Some ONNX operators not yet implemented (Pad, InstanceNorm, GroupNorm, etc.)
- C/C++ bindings planned but not yet available

[0.1.0]: https://github.com/Narcoleptic-Fox/RONN/releases/tag/v0.1.0
