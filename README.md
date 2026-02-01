<p align="center">
    <img src="assets/icon.png" alt="RONN Logo" width="200"/>
</p>

# RONN - Rust Open Neural Network Runtime

A next-generation ML inference runtime written in pure Rust, combining high-performance ONNX compatibility with cognitive computing architectures.

[![CI](https://github.com/Narcoleptic-Fox/RONN/workflows/CI/badge.svg)](https://github.com/Narcoleptic-Fox/RONN/actions)
[![Benchmarks](https://github.com/Narcoleptic-Fox/RONN/workflows/Benchmarks/badge.svg)](https://github.com/Narcoleptic-Fox/RONN/actions)
[![codecov](https://codecov.io/gh/Narcoleptic-Fox/RONN/branch/master/graph/badge.svg)](https://codecov.io/gh/Narcoleptic-Fox/RONN)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.90.0%2B-orange.svg)](https://rustup.rs)

## What Makes RONN Unique

RONN is the **first Rust ML runtime with brain-inspired computing features**, combining:
- Standard ONNX model execution
- Cognitive architecture patterns (HRM, multi-tier memory)
- Extreme optimization (BitNet 1-bit quantization)
- Production-ready performance

### Key Innovations

- **32x Compression**: BitNet provider enables 1-bit quantized models
- **10x Faster**: Adaptive routing between fast/slow inference paths
- **Smart Routing**: Complexity-based selection (System 1 vs System 2)
- **Pure Rust**: Zero FFI, memory-safe, cross-platform

## Quick Start

```rust
use ronn_api::prelude::*;
use ronn_core::{DataType, TensorLayout};
use std::collections::HashMap;

// Load an ONNX model
let model = Model::load("model.onnx")?;

// Configure session with optimization
let session = model.create_session(
    SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O3)
        .with_provider(ProviderType::GPU)
)?;

// Run inference
let mut inputs = HashMap::new();
inputs.insert("input", Tensor::zeros(
    vec![1, 3, 224, 224],
    DataType::F32,
    TensorLayout::RowMajor
)?);
let outputs = session.run(inputs)?;
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   High-Level API                    │
│          (Model, Session, Builder patterns)         │
├─────────────────────────────────────────────────────┤
│              ONNX Compatibility Layer               │
│    (Parser, 20+ operators, type conversion)         │
├─────────────────────────────────────────────────────┤
│           Graph Optimization Pipeline               │
│  (Constant folding, fusion, layout optimization)    │
├─────────────────────────────────────────────────────┤
│           Brain-Inspired Features (HRM)             │
│     (Complexity routing, adaptive execution)        │
├─────────────────────────────────────────────────────┤
│            Execution Provider Framework             │
│  (CPU, GPU, BitNet, WebAssembly, Custom)            │
├─────────────────────────────────────────────────────┤
│                  Core Runtime Engine                │
│       (Tensor ops, session mgmt, Candle)            │
└─────────────────────────────────────────────────────┘
```

## Features

### ONNX Compatibility

- **20+ Core Operators**: Conv, MatMul, ReLU, Softmax, BatchNorm, etc.
- **Standard Compliant**: ONNX protobuf parsing and execution
- **Type System**: F32, F16, BF16, I8, I32, I64, U8, U32, Bool
- **Shape Inference**: Automatic shape validation and broadcasting

### Performance Optimization

- **4 Optimization Levels**: O0 (none) → O3 (aggressive)
- **Graph Passes**:
  - Constant folding
  - Dead code elimination
  - Node fusion (Conv+BN+ReLU)
  - Layout optimization (NCHW/NHWC)
  - Provider-specific optimizations

### Brain-Inspired Features

#### Hierarchical Reasoning Module (HRM)
```rust
// Automatic routing based on complexity
let router = ComplexityRouter::new();
let provider = router.route(input_tensor)?; // → BitNet or FullPrecision
```

- **System 1 (Fast)**: BitNet for simple/repeated patterns
- **System 2 (Slow)**: Full precision for complex/novel queries
- **Adaptive**: Learns from usage patterns

#### Performance Tradeoffs

| Provider       | Latency | Memory | Accuracy |
| -------------- | ------- | ------ | -------- |
| Full Precision | 1.0x    | 1.0x   | 100%     |
| BitNet (1-bit) | 0.1x    | 0.03x  | 95-98%   |
| FP16           | 0.5x    | 0.5x   | 99%      |
| Multi-GPU      | 0.2x    | 2.0x   | 100%     |

### Execution Providers

1. **CPU Provider**: SIMD-optimized (AVX2/AVX-512/NEON)
2. **GPU Provider**: CUDA/ROCm via Candle, Tensor Core support
3. **BitNet Provider**: 1-bit quantized models (32x compression)
4. **WebAssembly**: Browser deployment with SIMD128
5. **Custom Providers**: NPU/TPU framework with plugin system

## Installation

```toml
[dependencies]
ronn-api = "0.1"
ronn-core = "0.1"
ronn-providers = "0.1"
```

## Examples

### Basic Inference

```rust
use ronn_api::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let model = Model::load("resnet18.onnx")?;

    // Create session with defaults
    let session = model.create_session_default()?;

    // Prepare inputs
    let mut inputs = HashMap::new();
    inputs.insert("input", Tensor::zeros(&[1, 3, 224, 224])?);

    // Run inference
    let outputs = session.run(inputs)?;

    // Process results
    for (name, tensor) in outputs {
        println!("{}: {:?}", name, tensor.shape());
    }

    Ok(())
}
```

### Brain-Inspired Routing

```rust
use ronn_api::prelude::*;
use ronn_providers::ProviderType;

// Configure adaptive routing
let options = SessionOptions::new()
    .with_optimization_level(OptimizationLevel::O3)
    .with_provider(ProviderType::BitNet); // Fast path

let session = model.create_session(options)?;

// Simple queries use BitNet (10x faster, 32x smaller)
let simple_output = session.run(simple_input)?;

// Complex queries automatically route to full precision
let complex_output = session.run(complex_input)?;
```

### Multi-GPU Inference

```rust
use ronn_providers::{GpuProviderConfig, MultiGpuConfig};

// Configure multi-GPU execution
let config = MultiGpuConfig {
    device_ids: vec![0, 1, 2, 3],
    strategy: PlacementStrategy::LoadBalanced,
    enable_p2p: true,
};

let session = model.create_session(
    SessionOptions::new()
        .with_provider(ProviderType::Gpu)
        .with_gpu_config(config)
)?;
```

## Project Structure

```
ronn/
├── crates/
│   ├── ronn-core/         # Core tensor operations, session management
│   ├── ronn-providers/    # Execution providers (CPU, GPU, BitNet, etc.)
│   ├── ronn-onnx/         # ONNX compatibility layer
│   ├── ronn-graph/        # Graph optimization pipeline
│   ├── ronn-hrm/          # Hierarchical reasoning module
│   ├── ronn-memory/       # Multi-tier memory system
│   ├── ronn-learning/     # Continual learning engine
│   └── ronn-api/          # High-level user API
├── examples/
│   ├── simple-inference/  # Basic usage examples
│   ├── brain-features/    # Brain-inspired computing demo
│   └── onnx-model/        # Real ONNX model inference
└── benches/               # Performance benchmarks
```

## Performance

### Benchmarks

```bash
cargo bench --all
```

**Measured Performance** (v0.1.0):

Core Operations (P50):
- **HRM Routing Latency**: 1.5-2.0 µs per decision
- **Tensor Creation**: 423-2,304 ns (100-10K elements)
- **Tensor Clone**: 11 ns
- **Vector Extraction**: 379 ns (1K elements)

These microbenchmarks demonstrate that RONN's brain-inspired routing adds **less than 2 microseconds** of overhead while enabling intelligent System 1/System 2 path selection.

### Optimization Tips

1. **Use O3 for production**: `OptimizationLevel::O3`
2. **Enable BitNet for edge**: 32x smaller models
3. **Multi-GPU for batches**: Parallel execution across devices
4. **Profile first**: Use `--features profiling`

## Development

### Build

```bash
# Full build
cargo build --release --all

# With GPU support
cargo build --release --features gpu

# Examples (run from their directories)
cd examples/simple-inference && cargo run
cd examples/brain-features && cargo run
cd examples/onnx-model && cargo run
```

### Test

```bash
# All tests
cargo test --all

# Specific crate
cargo test -p ronn-core

# With logging
RUST_LOG=debug cargo test
```

### Documentation

```bash
cargo doc --no-deps --open
```

## Roadmap

See [TASKS.md](./TASKS.md) for detailed development plans.

## Contributing

We welcome contributions! Areas of focus:
- Additional ONNX operators
- New execution providers
- Performance optimizations
- Documentation improvements

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [Candle](https://github.com/huggingface/candle) for tensor operations
- ONNX specification from [onnx/onnx](https://github.com/onnx/onnx)
- Inspired by cognitive neuroscience research

---

🚀 **Try it**: `cd examples/brain-features && cargo run`
📖 **Learn more**: Check out [docs/](./docs/)
💬 **Discuss**: Open an issue or start a discussion
