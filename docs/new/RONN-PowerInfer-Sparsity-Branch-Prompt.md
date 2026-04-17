# RONN: PowerInfer-Style Activation Sparsity Implementation

## Context

You are working on **RONN** (Rust-Optimized Neural Networks), a Rust ML inference runtime at `https://github.com/Dieshen/RONN`. RONN is a brain-inspired neural network runtime with existing crates for core inference, graph optimization, execution providers, memory systems, learning, and hierarchical reasoning.

Your task is to create a new branch `feature/activation-sparsity` and implement PowerInfer-style activation locality optimization as a new crate `ronn-sparsity`.

## Step 0: Setup

```bash
# Clone RONN and create feature branch
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/activation-sparsity

# Clone PowerInfer as reference (DO NOT copy code — study patterns, reimplement in Rust)
git clone https://github.com/Tiiny-AI/PowerInfer.git ../PowerInfer-reference
```

**Key PowerInfer reference files to study:**
- `../PowerInfer-reference/ggml-cuda.cu` — Sparse CUDA kernels (hot/cold neuron routing)
- `../PowerInfer-reference/llama.cpp` — Modified inference loop with activation prediction
- `../PowerInfer-reference/powerinfer-py/` — Python tooling for activation profiling
- `../PowerInfer-reference/convert-hf-to-powerinfer-gguf.py` — Model conversion with predictor weights
- `../PowerInfer-reference/smallthinker/` — Their latest MoE-optimized inference

## Step 1: Understand the Existing RONN Architecture

Before writing any code, read and understand the existing crate structure:

```
ronn/
├── crates/
│   ├── ronn-core/          # Core tensor ops, model loading, ONNX support
│   ├── ronn-providers/     # Execution providers (CPU, GPU, WASM)
│   ├── ronn-graph/         # Computation graph, optimization passes
│   ├── ronn-memory/        # Multi-tier memory (working/episodic/semantic)
│   ├── ronn-learning/      # Continual learning, EWC, experience replay
│   ├── ronn-hrm/           # Hierarchical Reasoning Module (System 1/2)
│   └── ronn-api/           # API layer, MCP integration
```

Read every `lib.rs` and `Cargo.toml` in the existing crates to understand:
- How tensors are represented
- How the computation graph works
- How execution providers are dispatched
- How ronn-hrm routes between fast/slow paths (this pattern is directly relevant)

## Step 2: Create the `ronn-sparsity` Crate

Create a new crate at `crates/ronn-sparsity/` with the following module structure:

```
crates/ronn-sparsity/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API, feature flags
│   ├── profiler.rs         # Activation profiling (records which neurons fire)
│   ├── predictor.rs        # Lightweight neural net that predicts active neurons
│   ├── classifier.rs       # Hot/cold neuron classification from profiles
│   ├── sparse_ops.rs       # Sparse matrix multiply, selective computation
│   ├── scheduler.rs        # GPU/CPU neuron routing and memory pinning
│   ├── formats.rs          # PowerInfer GGUF compatibility, sparse weight formats
│   └── metrics.rs          # Sparsity ratio tracking, speedup measurement
```

## Step 3: Core Concept — What PowerInfer Actually Does

The key insight is simple: **LLM inference follows a power-law distribution in neuron activation.**

- ~10-30% of neurons ("hot neurons") fire on almost every input
- ~70-90% of neurons ("cold neurons") are input-dependent and rarely fire
- A tiny predictor network can predict which cold neurons will fire BEFORE computing them
- Skip the rest entirely

### The Pipeline:

```
Input Token
    ↓
[Attention Layers] — compute normally (dense, always needed)
    ↓
[FFN Layer Entry]
    ↓
[Activation Predictor] — tiny MLP predicts which neurons fire (< 1ms)
    ↓
[Neuron Router]
    ├── Hot neurons → GPU (pre-loaded, always ready)
    ├── Predicted-active cold neurons → CPU (compute on demand)
    └── Predicted-inactive neurons → SKIP (this is the speedup)
    ↓
[Sparse Gather] — combine results
    ↓
Output
```

## Step 4: Implementation Details

### 4a. Activation Profiler (`profiler.rs`)

Build a profiler that records neuron activation patterns during calibration runs.

```rust
/// Run a calibration dataset through the model and record which neurons
/// activate (output > threshold) at each FFN layer.
/// 
/// Output: Per-layer activation frequency map
///   layer_id -> neuron_id -> activation_count / total_samples
///
/// PowerInfer reference: See powerinfer-py/ for their profiling approach.
/// They profile on ~1000 representative samples and use ReLU activation
/// as the sparsity signal (ReLU naturally zeros out inactive neurons).
```

Key design decisions:
- Support both ReLU-native models (natural sparsity) and SiLU/GELU models (need threshold-based classification)
- Store profiles as serializable artifacts (`.ronn-profile` files)
- Profile format should include: model hash, calibration dataset hash, per-layer stats

### 4b. Hot/Cold Classifier (`classifier.rs`)

Given activation profiles, classify neurons as hot or cold.

```rust
/// Classification strategy:
/// - Hot: activation frequency > hot_threshold (default: 0.8 = fires 80%+ of the time)
/// - Cold: activation frequency < hot_threshold
///
/// The classifier also determines the OPTIMAL split ratio for a given
/// GPU memory budget. E.g., "With 16GB VRAM, we can pin the top 20% 
/// hottest neurons on GPU and route the rest to CPU."
///
/// This is where RONN's existing ronn-hrm System 1/2 pattern maps:
/// - Hot neurons = System 1 (fast, always ready, GPU)
/// - Cold neurons = System 2 (on-demand, computed when predicted active, CPU)
```

### 4c. Activation Predictor (`predictor.rs`)

A small neural network (per FFN layer) that predicts which neurons will fire.

```rust
/// The predictor is a tiny MLP:
///   Input: hidden state entering the FFN layer
///   Output: binary mask (1 = neuron will fire, 0 = skip)
///
/// Architecture: Linear(hidden_dim, predictor_dim) -> ReLU -> Linear(predictor_dim, ffn_dim) -> Sigmoid
/// Where predictor_dim << ffn_dim (typically 128-256)
///
/// Training: Binary cross-entropy on recorded activation patterns
/// The predictor itself runs on GPU and adds < 1ms per layer.
///
/// PowerInfer reference: They store predictor weights alongside model weights
/// in their custom GGUF format. Study convert-hf-to-powerinfer-gguf.py.
```

Important: The predictor must be trainable from RONN's profiler output. Provide a training pipeline:
1. Profile model → activation data
2. Train predictors (one per FFN layer) using activation data as labels
3. Export trained predictors alongside model weights
4. At inference time, load predictors and use them to gate FFN computation

### 4d. Sparse Operations (`sparse_ops.rs`)

The actual computation kernels that skip inactive neurons.

```rust
/// Sparse matrix multiply: only compute rows/columns corresponding to
/// predicted-active neurons.
///
/// Two approaches (implement both, benchmark):
/// 1. Gather-Scatter: Extract active weight rows → dense matmul → scatter results
/// 2. Sparse kernel: Custom kernel that skips zero-masked computations
///
/// For CPU: Use SIMD-aware sparse ops (RONN already has SIMD in ronn-core)
/// For GPU: Implement as a custom execution provider operation
///
/// PowerInfer reference: ggml-cuda.cu contains their sparse CUDA kernels.
/// Study the SpMV (Sparse Matrix-Vector) implementation.
/// DO NOT copy their CUDA code — reimplement the algorithm in Rust using
/// wgpu compute shaders or CUDA bindings via cudarc crate.
```

### 4e. Neuron Scheduler (`scheduler.rs`)

Routes computation between GPU and CPU based on hot/cold classification.

```rust
/// The scheduler manages:
/// 1. Memory pinning: Hot neuron weights pre-loaded to GPU VRAM
/// 2. Compute routing: For each FFN layer:
///    - Hot neurons always computed on GPU
///    - Predicted-active cold neurons computed on CPU
///    - Inactive neurons skipped entirely
/// 3. Result gathering: Combine GPU + CPU partial results
///
/// Integration with ronn-providers:
/// - Extend the existing provider trait to support partial-layer execution
/// - A single FFN layer may execute across BOTH providers simultaneously
///
/// Memory budget awareness:
/// - Accept a VRAM budget parameter
/// - Automatically determine how many hot neurons fit in GPU memory
/// - Gracefully degrade: more CPU work if less VRAM available
```

### 4f. Sparse Model Format (`formats.rs`)

Support for loading and saving sparsity-aware model formats.

```rust
/// Support two workflows:
/// 1. Profile + Convert: Take a standard ONNX/GGUF model, profile it,
///    train predictors, save as .ronn-sparse format
/// 2. Import PowerInfer GGUF: Load *.powerinfer.gguf files directly
///    (for compatibility with their existing model ecosystem)
///
/// The .ronn-sparse format should include:
/// - Original model weights (quantized)
/// - Per-layer activation profiles
/// - Per-layer predictor weights
/// - Hot/cold classification metadata
/// - Target hardware profile (GPU memory, compute capability)
```

## Step 5: Integration Points with Existing RONN Crates

### ronn-graph integration
- Add a new graph optimization pass: `SparsityOptimizationPass`
- This pass identifies FFN layers and inserts predictor + routing nodes
- The optimized graph should look like: `Attention → Predictor → SparseFFN → LayerNorm`

### ronn-providers integration
- Extend the provider trait to support `execute_sparse(tensor, active_mask) -> tensor`
- CPU provider: implement sparse matmul with SIMD
- GPU provider: implement sparse matmul with compute shaders

### ronn-hrm integration
- Map hot/cold neuron routing to the existing System 1/2 decision framework
- HRM can dynamically adjust sparsity thresholds based on task:
  - Creative/exploratory tasks → lower sparsity (activate more neurons)
  - Routine/pattern-matched tasks → higher sparsity (maximum speed)

### ronn-memory integration
- Cache activation patterns for repeated prompt prefixes
- If the same prompt prefix is seen again, reuse the cached predictor outputs

## Step 6: CLI and API

Add commands to the RONN CLI:

```bash
# Profile a model's activation patterns
ronn profile --model model.onnx --calibration-data data.jsonl --output model.ronn-profile

# Train activation predictors from profile data
ronn train-predictors --profile model.ronn-profile --output model.ronn-predictors

# Convert to sparse format
ronn convert-sparse --model model.onnx --predictors model.ronn-predictors --output model.ronn-sparse

# Run inference with sparsity
ronn infer --model model.ronn-sparse --vram-budget 16 --prompt "Hello world"

# Benchmark sparse vs dense
ronn benchmark --model model.onnx --sparse-model model.ronn-sparse --prompts bench.jsonl
```

## Step 7: Testing Strategy

1. **Unit tests**: Each module independently (profiler records correctly, classifier splits correctly, sparse ops produce same results as dense)
2. **Correctness tests**: Sparse inference output must match dense inference output within floating point tolerance (1e-5)
3. **Benchmark tests**: Measure tokens/sec for sparse vs dense on same model
4. **Integration tests**: Full pipeline from ONNX model → profile → convert → sparse inference

## Step 8: Documentation

Update RONN's README to document:
- The sparsity feature and how it works
- Supported model architectures
- CLI commands for the profiling/conversion pipeline
- Expected speedup ranges based on model sparsity

## Constraints and Guidelines

- **Pure Rust**: No C/C++ FFI unless absolutely necessary for GPU kernels
- **No code copying from PowerInfer**: Study their approach, reimplement in Rust idiomatically
- **Maintain RONN's existing API**: This is additive, not breaking
- **Feature-gated**: All sparsity features behind `#[cfg(feature = "sparsity")]` flag
- **Error handling**: Use RONN's existing error types, extend as needed with `thiserror`
- **Async-ready**: Scheduler operations should be async-compatible for concurrent GPU/CPU execution
- **Hardware-agnostic**: Should work (with degraded performance) even without GPU

## Success Criteria

1. `ronn-sparsity` crate compiles and passes all tests
2. Can profile any ONNX model's activation patterns
3. Can train predictors and classify hot/cold neurons
4. Sparse inference produces correct output matching dense inference
5. Demonstrates measurable speedup (target: 2-5x on ReLU models, 1.5-3x on SiLU/GELU models)
6. Integrates cleanly with ronn-graph and ronn-providers
7. CLI commands work end-to-end

## References

- **PowerInfer Paper**: https://arxiv.org/abs/2312.12456
- **PowerInfer GitHub**: https://github.com/Tiiny-AI/PowerInfer (cloned at `../PowerInfer-reference`)
- **TurboSparse Paper**: https://arxiv.org/abs/2406.05955 (sparsifying non-ReLU models)
- **PowerInfer-2 Paper**: https://arxiv.org/abs/2406.06282 (mobile/edge optimizations)
- **TinyLoRA Paper**: https://arxiv.org/abs/2602.04118 (complementary — ultra-efficient fine-tuning)
- **Deja Vu Paper**: https://proceedings.mlr.press/v202/liu23am.html (original activation locality research)
