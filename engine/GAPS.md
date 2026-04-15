# NNX Gaps & Competitive Roadmap

Audit date: April 14, 2026.

This document tracks the real gaps between the current NNX implementation and the level required to compete seriously with `llama.cpp`, `vLLM`, `ONNX Runtime`, and `Whisper`.

The intent is not to maintain an aspirational feature dump. It is to keep an honest, current map of:

- what is already closed,
- what exists only as a partial implementation,
- what is missing but necessary for correctness,
- what is required for competitive parity,
- what is longer-term frontier work.

## Current Position

NNX has a credible foundation, a functioning serving layer, and a GPU kernel library:

- Pure Rust core types, kernels, quantization blocks, GGUF parsing, SafeTensors parsing, and a working transformer backend.
- End-to-end GGUF loading into a live model.
- A functioning `InferenceEngine` surface with model metadata and cache control methods.
- 10 architecture-specific transformer paths via data-driven registry (Llama, Mistral, CodeLlama, GPT-2, Phi, Gemma, Qwen, StableLM, Falcon, MPT).
- A `KVStore` trait abstracting contiguous and paged KV storage, with bit-for-bit cross-validated implementations.
- Paged KV memory with ref-counted CoW pages, continuous batching scheduler (with memory-aware admission accounting for prefix cache hits), and prefix caching (with capacity-bounded LRU eviction) — all integrated in the `nnx-serving` crate with `ServingEngine`. Hardened through three code-review rounds that found and fixed eight bugs in page lifecycle, ref counting, and scheduler interaction.
- A `KernelBackend` trait abstracting CPU and GPU compute dispatch, with `CpuBackend` (SIMD/rayon) and `CubeclBackend<R: Runtime>` (CUDA, ROCm, Metal, Vulkan, WebGPU via CubeCL). A standalone `GpuInference<R>` runs complete transformer decode steps on GPU, cross-validated against the CPU forward pass.
- Advanced samplers (Min-P, Typical, TFS, Mirostat v2) competitive with llama.cpp.
- 500+ tests across the engine: CPU unit/integration, GPU kernel correctness, cross-validation (paged vs contiguous, GPU vs CPU), end-to-end serving loops, and 49 targeted edge-case tests.

NNX is not yet competitive as a general-purpose inference library because:

- The GPU forward pass lives in `nnx-cubecl::GpuInference` as a standalone path; it is not yet wired into `nnx-transformer`'s `Model` so existing callers of `InferenceEngine` cannot transparently use GPU. This is the remaining P0 gap.
- Async/streaming execution, quantized KV cache, LoRA, structured outputs, and multi-modal support are the next tier of work.

## Status Summary

| Area                          | Status  | Notes                                                                                  |
| ----------------------------- | ------- | -------------------------------------------------------------------------------------- |
| GGUF loading                  | Partial | Works end-to-end with mixed dense/quantized matrix storage; not every tensor stays compact yet. |
| SafeTensors loading           | Closed  | Architecture-aware HF name mapping for 5 families, fused QKV splitting, multi-shard support, config.json inference. |
| Transformer execution (CPU)   | Closed  | Batched causal prefill, request-scoped KV, batch-safe layer features, `KVStore` trait for pluggable cache backends. |
| Transformer execution (GPU)   | Partial | `GpuInference<R>` in `nnx-cubecl` runs full decode step on GPU (embedding → blocks → norm → logits), cross-validated. Not yet integrated into `nnx-transformer`'s `Model`/`InferenceEngine`. |
| Quantization                  | Partial | GGUF matrix execution covers token embeddings, projection matrices, and `lm_head`, with scratch-reusing quantized prefill; activations, KV cache, and SafeTensors storage are still dense. |
| Kernel API safety             | Closed  | Public transformer runtime paths use checked RoPE/norm/cache/attention operations. Unchecked kernels kept for hot-path performance. |
| Architecture support          | Closed  | 10 architecture profiles via data-driven registry. |
| Serving infrastructure        | Closed  | `nnx-serving` crate with paged attention (`BlockAllocator`, `PagedLayerView`), continuous batching (`Scheduler` with prefix-aware admission), prefix caching (`PrefixCache` with capacity-bounded LRU), and `ServingEngine`. 115 tests including 30 edge cases. Hardened through 3 review rounds (8 bugs fixed). |
| GPU kernel library            | Closed  | `nnx-cubecl` crate: CubeCL kernels for matmul, rms_norm, layer_norm, softmax, rope, silu, gelu, fused_swiglu, fused_geglu, elementwise add/mul, attention scores/contraction, embedding lookup, cache append. Tiled launches (BLOCK_SIZE=256) safe for realistic model dims. 57 GPU tests. |
| Backend abstraction           | Closed  | `KernelBackend` trait in `nnx-core` with `CpuBackend` (nnx-kernels) and `CubeclBackend<R>` (nnx-cubecl). Execution-provider pattern. |
| Async/streaming runtime       | Missing | No async execution model, no token streaming API. `ServingEngine` is synchronous. |
| ONNX support                  | Missing | Placeholder crate only. |
| GGML support                  | Missing | File open works, tensor extraction does not. |
| Speech/multimodal             | Missing | No encoder-decoder path, VLM path, timestamps, or VAD integration. |

## Closed Since The Previous Version

These are no longer active gaps and should stay out of the roadmap unless they regress.

| Item                           | Previous Claim                                                              | Current State                                                                                           |
| ------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Trait metadata/cache alignment | `InferenceEngine` still missing `model_info()` and cache management support | Closed. `model_info()`, cache token/capacity/memory access, clear, and truncate are implemented.        |
| Real-model integration harness | No framework for real-model integration tests                               | Closed. `nnx-transformer/tests/integration.rs` provides a real-model harness gated by `NNX_TEST_MODEL`. |
| Parser test coverage           | GGUF and SafeTensors parser testing was missing                             | Largely closed. Both parsers now have substantial unit coverage for success and failure cases.          |
| GGUF memory budget enforcement | `LoadConfig.memory_budget` was not enforced                                 | Closed for GGUF model loading.                                                                          |
| Request-scoped KV ownership    | KV cache lived on the model and bled across requests                        | Closed. Requests now own their own KV cache lifecycle.                                                  |
| Batch correctness              | Multi-sequence batch input reused one mutable cache                         | Closed for correctness. Independent request branches no longer share cache state.                       |
| API honesty gaps               | `device`, `num_threads`, unknown architecture fallback, and quantization reporting were misleading | Closed at the API level. Unsupported device/thread settings fail fast, unknown architectures error, and source quantization is reported honestly. |
| Quantized forward execution    | GGUF runtime always dequantized everything to `f32`                         | Partially closed. Large GGUF projection matrices and `lm_head` now execute from compact storage.       |
| SafeTensors memory budget      | SafeTensors loading ignored `LoadConfig.memory_budget`                      | Closed. SafeTensors path now enforces memory budget too.                                                |
| Layer-feature serving semantics| Feature extraction state was sticky, weakly validated, and batch-fragile    | Closed for the current API. Requests now arm features one-shot, batch outputs are shape-checked, and batched `forward_layers()` is covered. |
| Transformer runtime crash paths| RoPE, normalization, and KV cache writes could still crash public runtime calls | Closed for transformer inference. Public transformer paths now use checked RoPE/norm operations and explicit KV cache bounds errors. |
| Quantized matrix runtime path  | Compact execution stopped at projections and `lm_head`                      | Closed for GGUF matrix weights. Token embeddings are compact too, and quantized batched prefill now reuses scratch instead of reallocating per prompt row. |
| Advanced samplers              | Only temperature, top-k, top-p, repetition penalty                          | Closed. Min-P, Typical, TFS, and Mirostat v2 implemented with presets (greedy, creative, precise, default_chat). Competitive with llama.cpp sampler surface. |
| Batched causal prefill         | Prefill walked positions sequentially for causal attention                   | Closed. Full Q*K^T attention score matrix with triangular causal mask, rayon parallel across heads, position-offset-aware for continued generation. |
| SafeTensors general model path | Loader hardcoded to HF decoder-only Llama-style naming                      | Closed. Architecture-aware WeightNameMap for 5 families (Llama, GPT-2, Phi, Gemma, Qwen), fused QKV splitting, TensorSource trait, multi-file shard support via index.json. |
| Architecture extension model   | Adding a new architecture required modifying central dispatch match arms     | Closed. Data-driven ArchitectureProfile registry with 10 static const profiles. Adding an architecture is adding a const — no dispatch code changes. Lookup by GGUF name or HF model_type. |
| Paged attention / paged KV     | No paged memory management for KV cache                                     | Closed. `nnx-serving` crate with `BlockAllocator` (ref-counted CoW, 16-token pages), `PagedLayerView` implementing `KVStore` trait, `ServingEngine` with full multi-layer paged forward pass. Bit-for-bit cross-validated against contiguous `LayerCache`. |
| Continuous batching            | No serving scheduler for concurrent request management                      | Closed. Two-queue FCFS `Scheduler` with memory-aware admission (accounts for prefix cache hits), sequence state machine (Waiting→Prefilling→Decoding→Finished), staggered request arrivals, cancel support with proper page freeing. Hardened: 8 bugs found and fixed across 3 code-review rounds. |
| Prefix caching                 | No reuse of shared prompt prefixes                                          | Closed. `PrefixCache` with chain-dependent 64-bit content hashing, capacity-bounded LRU eviction (`max_prefix_cache_entries`), auto-sharing on admission. Insert returns bool to prevent ref-count leaks on duplicates. Verified identical logits with and without caching. |
| GPU kernel library             | No GPU compute capability                                                   | Closed. `nnx-cubecl` crate with CubeCL kernels portable across CUDA, ROCm, Metal, Vulkan, WebGPU. Tiled elementwise launches (BLOCK_SIZE=256) with bounds-checking for workgroup-limit compliance. 57 GPU tests including edge cases at realistic model dims (4096, 11008 elements). |
| Backend abstraction            | No execution-provider model                                                 | Closed. `KernelBackend` trait in `nnx-core` with `CpuBackend` (nnx-kernels) and `CubeclBackend<R: Runtime>` (nnx-cubecl). |
| GPU forward pass               | No GPU inference path                                                       | Closed as standalone. `GpuInference<R>` in `nnx-cubecl` runs complete transformer decode step on GPU (embedding lookup → layers → norm → logits), cross-validated against CPU forward pass. Supports Llama-class configs (RMSNorm + SwiGLU + RoPE + Sequential). Not yet integrated into `nnx-transformer`'s `Model`/`InferenceEngine`. |

## Critical Gaps

The only remaining P0 item is wiring the GPU path into the main inference API.

| Priority | Gap                                          | Why It Matters                                                                                                   |
| -------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| P0       | GPU integration into `nnx-transformer` Model | `GpuInference<R>` works standalone but callers of `InferenceEngine` (RONN's HRM, speculative decoding, `ServingEngine`) cannot use GPU yet. Making `Model` generic over `KernelBackend` or adding a `GpuNnxBackend` that implements `InferenceEngine` would close this gap. |

## Active Implementation Gaps

| Priority | Gap                                      | Status  | Notes                                                                                                                                    |
| -------- | ---------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| P1       | Async/streaming execution API            | Missing | `ServingEngine` is synchronous. Non-blocking token streaming and modern server integration require an async runtime wrapper.              |
| P1       | SafeTensors execution storage            | Partial | SafeTensors tensors still materialize to dense `f32`; compact runtime storage is currently GGUF-first.                                  |
| P1       | Deeper quantized runtime                 | Partial | Matrix weights now stay compact on the GGUF path, but activations, KV cache, and non-matmul state are still dense runtime structures.  |
| P1       | ONNX loader implementation               | Missing | `nnx-onnx` is still placeholder-only.                                                                                                    |
| P1       | GGML tensor extraction                   | Missing | `nnx-ggml` opens files but does not parse architecture-specific tensors.                                                                 |
| P2       | Property-based parser tests              | Partial | Parser coverage is much better, but property-based testing and corpus-driven fuzzing are still missing.                                  |
| P2       | Quantization tooling                     | Missing | NNX can read quantized weights but cannot produce them.                                                                                  |
| P2       | Real-model CI discipline                 | Partial | The harness exists, but it is opt-in and does not guarantee coverage in normal CI.                                                       |

## Competitive Parity Roadmap

### Serving And Runtime Parity

| Priority | Capability                                | Status  | Why It Matters                                                                                                    |
| -------- | ----------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| P0       | Paged attention / paged KV memory         | Closed  | `nnx-serving`: `BlockAllocator`, `PagedLayerView`, `ServingEngine`. Cross-validated against contiguous cache.      |
| P0       | Continuous batching                       | Closed  | `Scheduler` with FCFS, prefix-aware admission, sequence state machine. 8 bugs found and fixed. 115 tests.         |
| P0       | Prefix caching                            | Closed  | `PrefixCache` with chain-dependent hashing, capacity-bounded LRU eviction. Ref-leak-safe insert. Verified identical logits. |
| P1       | Chunked prefill                           | Partial | `ServingConfig.max_prefill_tokens` is plumbed but not yet exercised in `ServingEngine` with real chunking logic.   |
| P1       | Async/streaming execution API             | Missing | Necessary for non-blocking token streaming and modern server integration.                                          |

### Model And Decoding Parity

| Priority | Capability            | Status  | Why It Matters                                                                                              |
| -------- | --------------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| P1       | Advanced samplers     | Closed  | Min-P, Typical, TFS, and Mirostat v2 are implemented with presets. Competitive with llama.cpp.             |
| P1       | Structured outputs    | Missing | Grammar-constrained decoding is already normal in `llama.cpp` and increasingly expected in serving systems. |
| P1       | LoRA adapter support  | Missing | Per-request adapter application is part of current inference-server expectations.                           |
| P1       | Speculative decoding  | Missing | Important for latency competitiveness and should integrate cleanly with layer-feature extraction.           |
| P2       | KV cache quantization | Missing | Matters for long-context memory pressure. TurboQuant (ICLR 2026, [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)) achieves 3-bit KV with zero accuracy loss via two-stage PolarQuant + QJL residual correction — training-free, data-oblivious, 6x memory reduction, up to 8x attention throughput on H100. Strong starting point now that paged KV is in place. |
| P2       | Diverse RoPE scaling  | Missing | Needed for compatibility with more modern model variants.                                                   |

### Backend And Deployment Parity

| Priority | Capability                                | Status  | Why It Matters                                                                                               |
| -------- | ----------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| P0       | Portable GPU backend                      | Partial | CubeCL kernels, `KernelBackend` trait, and `GpuInference` forward pass all implemented. Remaining: integrate into `nnx-transformer` Model or `InferenceEngine`. |
| P1       | Backend plugin / execution-provider model | Closed  | `KernelBackend` trait in `nnx-core` with `CpuBackend` and `CubeclBackend<R>` provides the Rust-native execution-provider pattern. |
| P1       | I/O binding / zero-copy buffer ownership  | Missing | Required for serious accelerator integration and low-overhead serving.                                       |
| P1       | Graph optimization and operator fusion    | Missing | Important for ONNX/GGML execution and for keeping up with optimized runtimes.                                |
| P1       | Dense-model multi-GPU support             | Missing | Tensor and pipeline parallelism are now baseline for larger deployments.                                     |
| P2       | MoE expert parallelism                    | Missing | Needed if NNX wants to support current MoE models competitively.                                             |

### Modality Parity

| Priority | Capability              | Why It Matters                                                                      |
| -------- | ----------------------- | ----------------------------------------------------------------------------------- |
| P1       | Encoder-decoder support | Necessary for Whisper, T5, BART, and other non-decoder-only architectures.          |
| P1       | VLM support             | Vision-language models are now table stakes for a modern inference library.         |
| P2       | Word-level timestamps   | Necessary for competitive speech support.                                           |
| P2       | VAD integration         | Important for real transcription pipelines and parity with practical speech stacks. |

## Frontier Work

These are worthwhile, but they should not outrank the critical and parity items above.

| Priority | Capability                                        | Notes                                                                                           |
| -------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| P2       | Flash-like attention kernels                      | Important for GPU throughput, but after GPU integration is complete.                             |
| P2       | Kernel specialization by shape/batch              | Good performance work once the execution model is stable.                                       |
| P2       | Activation quantization                           | Valuable, but not before quantized weight execution is solved.                                  |
| P3       | MLA-specific kernels                              | Relevant for newer model families, but not a first-wave requirement.                            |
| P3       | Lazy loading / out-of-core weight paging          | Important for very large models, but depends on stronger backend and memory abstractions.       |
| P3       | Hardware profiler and instrumentation             | Useful for optimization work, but not a blocker to correctness or parity.                       |
| P3       | Advanced speculative variants                     | EAGLE-style and other frontier methods should follow after baseline speculation works reliably. |
| P4       | Ternary / BitNet / ultra-low-bit frontier formats | Strategic, but premature until the main runtime is competitive on mainstream models.            |

## Recommended Order Of Attack

1. ~~Build paged attention, prefix caching, and continuous batching as the serving core.~~ **Done.** `nnx-serving` crate with `ServingEngine`, 115 tests, bit-for-bit cross-validated, hardened through 3 review rounds.
2. ~~Add a real GPU/backend abstraction.~~ **Done.** `KernelBackend` trait, `CpuBackend`, `CubeclBackend<R>`, `GpuInference<R>` with full forward pass. 57 GPU tests.
3. **Wire GPU into the main inference API.** Either make `nnx-transformer::Model` generic over `KernelBackend`, or add a `GpuNnxBackend` implementing `InferenceEngine`. This is the remaining P0. **This is the next step.**
4. Add async/streaming execution API on top of `ServingEngine` for real server integration.
5. Finish SafeTensors compact execution storage and deeper quantized runtime (activations, KV cache — TurboQuant for KV).
6. Add LoRA, structured outputs, and speculative decoding on top of the serving foundation.
7. Finish ONNX and GGML loader paths.
8. Expand into encoder-decoder, speech, and multimodal support.

## Competitive Baseline To Keep In Mind

As of April 14, 2026, current upstream projects already treat the following as normal:

- `vLLM`: paged attention, continuous batching, prefix caching, speculative decoding, structured outputs, LoRA serving, and distributed inference modes.
- `llama.cpp`: broad architecture coverage, many quantization formats, grammar-constrained decoding, speculative decoding, multimodal support, and practical local serving.
- `ONNX Runtime`: execution-provider architecture, graph optimizations, I/O binding, and broad hardware acceleration.
- `Whisper` ecosystem: usable speech pipelines expect encoder-decoder support, timestamps, and usually VAD-aware workflows.

NNX does not need to clone those projects feature-for-feature. It does need to be honest about which layer of the stack it wants to compete on. The current position is:

- **Local inference core**: strong. 10 architectures, quantized execution, batched prefill, advanced samplers.
- **Serving runtime**: solid. Paged attention, continuous batching, prefix caching implemented, tested, and hardened. Async runtime is next.
- **GPU compute**: kernel library and backend abstraction complete. Standalone GPU forward pass works and is cross-validated. Integration into the main API is the remaining P0.
- **Multi-modal**: not started. Future work once the serving and backend layers are solid.

## Scope Discipline

Do not confuse "interesting roadmap" with "highest leverage roadmap".

The next version of NNX should be judged primarily on:

- GPU inference accessible through the main `InferenceEngine` API,
- async execution API suitable for server integration,
- quantization-native execution including KV cache,
- LoRA and structured outputs as serving features.

The serving primitives are solid. The GPU kernel library is proven. The remaining ceiling is API integration.
