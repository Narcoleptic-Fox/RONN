# NNX Gaps & Competitive Roadmap

Audit date: April 13, 2026.

This document tracks the real gaps between the current NNX implementation and the level required to compete seriously with `llama.cpp`, `vLLM`, `ONNX Runtime`, and `Whisper`.

The intent is not to maintain an aspirational feature dump. It is to keep an honest, current map of:

- what is already closed,
- what exists only as a partial implementation,
- what is missing but necessary for correctness,
- what is required for competitive parity,
- what is longer-term frontier work.

## Current Position

NNX already has a credible foundation:

- Pure Rust core types, kernels, quantization blocks, GGUF parsing, SafeTensors parsing, and a working transformer backend.
- End-to-end GGUF loading into a live model.
- A functioning `InferenceEngine` surface with model metadata and cache control methods.
- Initial architecture-specific transformer paths for `Llama`, `GPT2`, `Phi`, `Gemma`, and `Qwen`.
- Unit and integration test coverage for the transformer crate.

NNX is not yet competitive as a general-purpose inference library because the most important missing pieces are still in correctness, batching semantics, backend extensibility, and serving architecture, not just raw kernel speed.

## Status Summary

| Area                          | Status  | Notes                                                                                  |
| ----------------------------- | ------- | -------------------------------------------------------------------------------------- |
| GGUF loading                  | Partial | Works end-to-end with mixed dense/quantized matrix storage; not every tensor stays compact yet. |
| SafeTensors loading           | Partial | Now reads adjacent `config.json`, supports budget checks, and handles HF decoder-only layouts better, but is still not fully generic. |
| Transformer execution         | Partial | Single-model CPU execution works; request-scoped KV is in place, prefill is materially improved, and layer-feature plumbing is now batch-safe and one-shot, but serving semantics are still incomplete. |
| Quantization                  | Partial | GGUF matrix execution now covers token embeddings, projection matrices, and `lm_head`, with scratch-reusing quantized prefill; activations, KV cache, and SafeTensors storage are still dense. |
| Kernel API safety             | Partial | Public transformer runtime paths now use checked RoPE/norm/cache behavior, but many unchecked kernels still exist elsewhere in `nnx-kernels`. |
| Architecture support          | Partial | Several architectures are hardcoded, but there is no stable extension model.           |
| ONNX support                  | Missing | Placeholder crate only.                                                                |
| GGML support                  | Missing | File open works, tensor extraction does not.                                           |
| Serving/runtime orchestration | Missing | No paged attention, continuous batching, prefix caching, or async runtime model.       |
| Speech/multimodal             | Missing | No encoder-decoder path, VLM path, timestamps, or VAD integration.                     |

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

## Critical Gaps

These are the highest-priority items because they block correctness, safe API usage, or honest feature claims.

| Priority | Gap                                         | Why It Matters                                                                                                                                                             |
| -------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0       | Paged attention / paged KV memory           | This is the next real infrastructure gap. Without it, NNX cannot compete on memory efficiency or high-throughput serving.                                               |
| P0       | Continuous batching                         | Correctness is now in better shape, but throughput under concurrent load will still lag badly without a real serving scheduler.                                         |
| P1       | Portable GPU backend                        | CPU-only execution remains a hard ceiling on competitiveness even with the runtime floor improved.                                                                       |

## Active Implementation Gaps

These are the important gaps already present in the codebase or immediately adjacent to it.

| Priority | Gap                                      | Status  | Notes                                                                                                                                    |
| -------- | ---------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| P1       | True batched prefill                     | Partial | Prefill now batches dense projection/FFN work across prompt tokens and quantized matrices reuse scratch instead of reallocating per row, but causal attention still walks positions sequentially. |
| P1       | SafeTensors as a general model path      | Partial | The loader now reads adjacent `config.json`, infers more architecture detail, and loads optional bias terms, but it still targets HF decoder-only naming rather than arbitrary layouts. |
| P1       | SafeTensors execution storage            | Partial | SafeTensors tensors still materialize to dense `f32`; compact runtime storage is currently GGUF-first.                                  |
| P1       | Deeper quantized runtime                 | Partial | Matrix weights now stay compact on the GGUF path, but activations, KV cache, and non-matmul state are still dense runtime structures.  |
| P1       | Standardized architecture extension path | Partial | Several architectures are hardcoded, but adding a new one still requires touching central dispatch and naming logic.                     |
| P1       | ONNX loader implementation               | Missing | `nnx-onnx` is still placeholder-only.                                                                                                    |
| P1       | GGML tensor extraction                   | Missing | `nnx-ggml` opens files but does not parse architecture-specific tensors.                                                                 |
| P2       | Property-based parser tests              | Partial | Parser coverage is much better, but property-based testing and corpus-driven fuzzing are still missing.                                  |
| P2       | Quantization tooling                     | Missing | NNX can read quantized weights but cannot produce them.                                                                                  |
| P2       | Real-model CI discipline                 | Partial | The harness exists, but it is opt-in and does not guarantee coverage in normal CI.                                                       |

## Competitive Parity Roadmap

These are the features that matter if NNX is meant to compete with current non-Rust leaders rather than remain a narrow local runtime.

### Serving And Runtime Parity

| Priority | Capability                                | Why It Matters                                                                                                                        |
| -------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| P0       | Paged attention / paged KV memory         | This is foundational for memory efficiency and high-throughput serving. vLLM treats this as core infrastructure, not an optimization. |
| P0       | Continuous batching                       | Without this, NNX will not be competitive as a serving engine under real concurrent load.                                             |
| P0       | Prefix caching                            | Reuse of shared prompt prefixes is now baseline serving behavior.                                                                     |
| P1       | Chunked prefill                           | Needed to keep latency acceptable while large prompts are being processed.                                                            |
| P1       | Async/streaming execution API             | Necessary for non-blocking token streaming and modern server integration.                                                             |
| P1       | Session scheduler / runtime orchestration | Request admission, cancellation, fairness, and cache ownership need a real runtime design.                                            |

### Model And Decoding Parity

| Priority | Capability            | Why It Matters                                                                                              |
| -------- | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| P1       | Advanced samplers     | Min-P, Typical, TFS, and Mirostat are expected in serious local inference stacks.                           |
| P1       | Structured outputs    | Grammar-constrained decoding is already normal in `llama.cpp` and increasingly expected in serving systems. |
| P1       | LoRA adapter support  | Per-request adapter application is part of current inference-server expectations.                           |
| P1       | Speculative decoding  | Important for latency competitiveness and should integrate cleanly with layer-feature extraction.           |
| P2       | KV cache quantization | Matters for long-context memory pressure.                                                                   |
| P2       | Diverse RoPE scaling  | Needed for compatibility with more modern model variants.                                                   |

### Backend And Deployment Parity

| Priority | Capability                                | Why It Matters                                                                                               |
| -------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| P0       | Portable GPU backend                      | CPU-only execution will not keep NNX competitive for mainstream inference workloads.                         |
| P1       | Backend plugin / execution-provider model | ONNX Runtime is strong partly because backends are extensible. NNX needs a Rust-native version of that idea. |
| P1       | I/O binding / zero-copy buffer ownership  | Required for serious accelerator integration and low-overhead serving.                                       |
| P1       | Graph optimization and operator fusion    | Important for ONNX/GGML execution and for keeping up with optimized runtimes.                                |
| P1       | Dense-model multi-GPU support             | Tensor and pipeline parallelism are now baseline for larger deployments.                                     |
| P2       | MoE expert parallelism                    | Needed if NNX wants to support current MoE models competitively.                                             |

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
| P2       | Flash-like attention kernels                      | Important, but after batching, memory model, and GPU backend fundamentals are addressed.        |
| P2       | Kernel specialization by shape/batch              | Good performance work once the execution model is stable.                                       |
| P2       | Activation quantization                           | Valuable, but not before quantized weight execution is solved.                                  |
| P3       | MLA-specific kernels                              | Relevant for newer model families, but not a first-wave requirement.                            |
| P3       | Lazy loading / out-of-core weight paging          | Important for very large models, but depends on stronger backend and memory abstractions.       |
| P3       | Hardware profiler and instrumentation             | Useful for optimization work, but not a blocker to correctness or parity.                       |
| P3       | Advanced speculative variants                     | EAGLE-style and other frontier methods should follow after baseline speculation works reliably. |
| P4       | Ternary / BitNet / ultra-low-bit frontier formats | Strategic, but premature until the main runtime is competitive on mainstream models.            |

## Recommended Order Of Attack

If the goal is to make NNX a serious Rust contender rather than an interesting prototype, the work should be sequenced like this:

1. Build paged attention, prefix caching, and continuous batching as the serving core.
2. Add a real GPU/backend abstraction instead of baking CPU assumptions into the current path.
3. Finish SafeTensors compact execution and deeper quantized runtime work beyond matrix weights.
4. Add LoRA, structured outputs, and speculative decoding on top of that serving foundation.
5. Finish ONNX and GGML loader paths.
6. Expand into encoder-decoder, speech, and multimodal support.

## Competitive Baseline To Keep In Mind

As of April 13, 2026, current upstream projects already treat the following as normal:

- `vLLM`: paged attention, continuous batching, prefix caching, speculative decoding, structured outputs, LoRA serving, and distributed inference modes.
- `llama.cpp`: broad architecture coverage, many quantization formats, grammar-constrained decoding, speculative decoding, multimodal support, and practical local serving.
- `ONNX Runtime`: execution-provider architecture, graph optimizations, I/O binding, and broad hardware acceleration.
- `Whisper` ecosystem: usable speech pipelines expect encoder-decoder support, timestamps, and usually VAD-aware workflows.

NNX does not need to clone those projects feature-for-feature. It does need to be honest about which layer of the stack it wants to compete on. Right now the most credible path is:

- become the best pure-Rust local inference core first,
- then become a serious Rust serving runtime,
- then expand into multi-modal and distributed frontier features.

## Scope Discipline

Do not confuse "interesting roadmap" with "highest leverage roadmap".

The next version of NNX should be judged primarily on:

- correctness under batching and concurrent sessions,
- honest and safe API behavior,
- quantization-native execution,
- serving primitives that actually move latency and throughput,
- backend architecture that can scale beyond CPU-only execution.

If those are not solid, the rest of the roadmap is mostly decoration.
