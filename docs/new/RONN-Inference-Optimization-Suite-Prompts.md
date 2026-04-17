# RONN: LLM Inference Optimization Suite

## Overview

You are working on **RONN** (Rust-Optimized Neural Networks), a Rust ML inference runtime at `https://github.com/Dieshen/RONN`. This document contains implementation prompts for five optimization features, each as its own branch. They share some infrastructure and should be developed with awareness of each other, but can be implemented independently.

**Existing RONN crates you must understand before starting ANY of these:**
```
ronn/
├── crates/
│   ├── ronn-core/          # Core tensor ops, model loading, ONNX support
│   ├── ronn-providers/     # Execution providers (CPU, GPU, WASM)
│   ├── ronn-graph/         # Computation graph, optimization passes
│   ├── ronn-memory/        # Multi-tier memory (working/episodic/semantic) + sleep consolidation
│   ├── ronn-learning/      # Continual learning, EWC, experience replay
│   ├── ronn-hrm/           # Hierarchical Reasoning Module (System 1/2 routing)
│   └── ronn-api/           # API layer, MCP integration
```

**Read every `lib.rs` and `Cargo.toml` before writing any code.**

---

# FEATURE 1: Speculative Decoding
**Branch:** `feature/speculative-decoding`
**New crate:** `ronn-speculative`
**Estimated complexity:** Medium-High
**Expected speedup:** 2-4x lossless

## Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/speculative-decoding

# Clone reference implementations (study only, do not copy code)
git clone https://github.com/SafeAILab/EAGLE.git ../EAGLE-reference
git clone https://github.com/sgl-project/SpecForge.git ../SpecForge-reference
```

**Key reference files to study:**
- `../EAGLE-reference/eagle/model/` — EAGLE draft model architecture
- `../EAGLE-reference/eagle/ge_data/` — Training data generation for draft models
- `../SpecForge-reference/` — Production-grade EAGLE-3 training pipeline
- EAGLE-3 paper: https://arxiv.org/abs/2503.01840
- Original speculative decoding paper: https://arxiv.org/abs/2302.01318

## Core Concept

Speculative decoding pairs two models:
1. **Draft model** (small, fast): proposes K candidate tokens ahead
2. **Target model** (large, accurate): verifies all K tokens in ONE forward pass
3. Accept the longest prefix that matches, reject the rest
4. **Mathematically lossless** — output distribution is identical to standard decoding

The speedup comes from GPU underutilization during autoregressive decoding. A single token generation only uses ~1-5% of available compute because it's memory-bandwidth-bound. Verification of K tokens costs roughly the same as generating 1 token.

## Why RONN's HRM Is Pre-Built For This

RONN's Hierarchical Reasoning Module already implements System 1 (fast, intuitive) and System 2 (slow, deliberate) routing. Speculative decoding IS this pattern:

```
System 1 (Draft)  →  Propose tokens quickly, low confidence acceptable
System 2 (Verify) →  Full model validates, guarantees correctness
HRM Router         →  Decides draft length, when to fall back to standard decoding
```

## Module Structure

```
crates/ronn-speculative/
├── Cargo.toml
├── src/
│   ├── lib.rs                # Public API, feature flags
│   ├── draft/
│   │   ├── mod.rs            # Draft model trait and implementations
│   │   ├── independent.rs    # Separate small model as drafter
│   │   ├── eagle.rs          # EAGLE-3 style (single layer reusing target features)
│   │   ├── self_draft.rs     # Self-speculative (early exit from target model)
│   │   └── ngram.rs          # N-gram lookup drafting (no neural network needed)
│   ├── verify/
│   │   ├── mod.rs            # Verification engine
│   │   ├── rejection.rs      # Rejection sampling (maintains exact distribution)
│   │   └── tree_verify.rs    # Tree-structured verification (verify branching drafts)
│   ├── tree/
│   │   ├── mod.rs            # Draft tree construction
│   │   ├── static_tree.rs    # Fixed tree topology
│   │   └── dynamic_tree.rs   # EAGLE-2 style adaptive tree based on confidence
│   ├── scheduler.rs          # Adaptive draft length, when to speculate vs standard
│   ├── training.rs           # Draft model training pipeline (for EAGLE-style)
│   └── metrics.rs            # Acceptance rate tracking, speedup measurement
```

## Implementation Details

### 1a. Draft Model Trait

```rust
/// All draft models implement this trait.
/// The trait must be generic enough to support:
/// - Independent small models (e.g., TinyLlama drafting for Llama-70B)
/// - EAGLE-style single-layer drafters that reuse target model features
/// - Self-speculative models that use early exit
/// - N-gram lookup (no neural computation)
///
/// The key method is `draft()` which produces a tree of candidate tokens
/// with associated log-probabilities.
pub trait DraftModel {
    /// Generate draft tokens given the current context.
    /// Returns a tree of candidate continuations with log-probs.
    fn draft(
        &self,
        context: &TokenContext,
        target_features: Option<&LayerFeatures>,  // For EAGLE-style
        max_draft_tokens: usize,
    ) -> DraftTree;

    /// Report how much memory/compute this drafter requires
    fn resource_requirements(&self) -> DraftResources;
}
```

### 1b. EAGLE-3 Implementation (`eagle.rs`)

This is the highest-priority draft model implementation. EAGLE-3 is state-of-the-art.

```rust
/// EAGLE-3 key insight: Instead of predicting next-token features from only
/// the top layer, fuse features from low, mid, and high layers of the target model.
///
/// Architecture:
///   1. During target model forward pass, extract features from layers L_low, L_mid, L_high
///   2. Concatenate/fuse these multi-level features
///   3. Feed fused features + token embedding into a single transformer layer
///   4. This lightweight layer autoregressively generates draft tokens
///
/// The drafter is ~1B parameters even for a 70B target model.
/// Training uses "Training-Time Testing" (TTT) — simulate multi-step
/// generation during training so the drafter learns to handle its own errors.
///
/// Reference: Study ../EAGLE-reference/eagle/model/ for architecture details.
/// Reference: Study ../SpecForge-reference/ for training pipeline.
```

Implementation steps:
1. Define the EAGLE feature extraction hooks (tap into target model's intermediate layers)
2. Implement the feature fusion module (learned weighted combination of layer outputs)
3. Build the single-layer autoregressive draft head
4. Implement tree-structured draft generation (not just linear chains)
5. Build the training pipeline that generates training data from target model runs

### 1c. Self-Speculative / Early Exit (`self_draft.rs`)

```rust
/// Self-speculative decoding uses the target model itself as the drafter
/// by exiting early from intermediate layers.
///
/// How it works:
///   1. Run input through first N layers (e.g., layers 1-16 of a 32-layer model)
///   2. Apply the language model head to the intermediate representation
///   3. Use these "early exit" predictions as draft tokens
///   4. Verify with full model (all 32 layers)
///
/// Advantages:
///   - No separate draft model to train or load
///   - No additional memory for draft model weights
///   - Trivial to implement given access to intermediate representations
///
/// This maps DIRECTLY to ronn-hrm's System 1/2:
///   System 1 = first N layers (fast, approximate)
///   System 2 = full model (slow, accurate)
///
/// Integration point: ronn-hrm should control the exit layer dynamically.
/// Easy tokens exit early, hard tokens use more layers.
/// Use confidence of the early-exit prediction to decide.
```

### 1d. N-gram / Retrieval Drafting (`ngram.rs`)

```rust
/// For known contexts (documentation, code patterns, repeated prompts),
/// draft tokens by looking up n-gram matches in a retrieval corpus.
///
/// How it works:
///   1. Maintain a suffix tree or trie of known text (from FoxStash, CRISP docs, etc.)
///   2. Match the last N generated tokens against the corpus
///   3. If match found, propose the continuation from the corpus as draft tokens
///   4. Target model verifies as usual
///
/// This is nearly FREE — no neural computation for drafting.
/// Works exceptionally well when:
///   - Model is outputting boilerplate code
///   - Model is quoting from provided context (RAG scenarios)
///   - Repetitive patterns in agent loops
///
/// Integration: ronn-memory's semantic tier provides the retrieval corpus.
/// FoxStash entries become speculative decoding accelerators.
```

### 1e. Rejection Sampling Verifier (`rejection.rs`)

```rust
/// The verification step MUST maintain the exact output distribution of the
/// target model. This is non-negotiable — it's what makes speculative
/// decoding lossless.
///
/// Algorithm (from Leviathan et al. 2023):
///   For each draft token i in sequence:
///     1. Compute target model probability p(token_i | context)
///     2. Compute draft model probability q(token_i | context)
///     3. Accept with probability min(1, p/q)
///     4. If rejected: sample a correction token from max(0, p - q) normalized
///     5. Stop — all subsequent draft tokens are discarded
///
/// For greedy decoding (temperature=0):
///   Simply check if argmax(target) == draft_token. Accept or reject.
///
/// Tree verification extends this to branching draft structures,
/// where multiple candidate continuations are verified simultaneously.
```

### 1f. Adaptive Scheduler (`scheduler.rs`)

```rust
/// The scheduler dynamically adjusts speculative decoding parameters:
///
/// 1. Draft length (K): How many tokens to speculate ahead
///    - Short K: Less wasted compute on rejections, but less potential speedup
///    - Long K: More potential speedup, but more wasted work on misses
///    - Optimal K depends on acceptance rate, which varies by task/domain
///
/// 2. When to speculate at all:
///    - If acceptance rate drops below threshold, fall back to standard decoding
///    - If system is compute-bound (batched inference), speculation may hurt
///
/// 3. Draft model selection (if multiple available):
///    - EAGLE for general tasks
///    - N-gram for RAG/known-context tasks
///    - Self-speculative for memory-constrained scenarios
///
/// Integration with ronn-hrm:
///   HRM provides the routing decision: speculate or decode standard.
///   HRM tracks acceptance rates as part of its confidence modeling.
///   The scheduler exposes a `SpeculativePolicy` that HRM uses.
///
/// Track metrics:
///   - Rolling acceptance rate (per position in draft)
///   - Average accepted tokens per speculation step
///   - Tokens/second with vs without speculation
///   - Draft model latency as fraction of verify latency
```

## Integration Points

### ronn-hrm
- System 1/2 routing becomes draft/verify routing
- Add `SpeculativePolicy` to HRM's decision framework
- HRM tracks acceptance rates and adjusts dynamically

### ronn-graph
- Add `SpeculativeDecodingPass` graph optimization
- This pass wraps autoregressive loops with draft-verify patterns
- The optimized graph includes both draft and verify paths

### ronn-memory
- Semantic memory tier provides n-gram corpus for retrieval drafting
- Working memory caches draft model states between speculation rounds
- KV cache from rejected tokens must be properly invalidated

### ronn-providers
- Draft model can run on a different provider than target model
- E.g., draft on CPU while target runs on GPU
- Or draft on NPU (S1 Max has 50 TOPS NPU)

## Testing

1. **Losslessness test**: Output distribution with speculative decoding MUST match standard decoding exactly (within floating point tolerance). This is the most critical test.
2. **Acceptance rate test**: Measure acceptance rates across different domains (code, prose, math)
3. **Speedup benchmark**: Wall-clock tokens/sec with and without speculation
4. **Draft model quality**: Compare EAGLE vs self-speculative vs n-gram acceptance rates
5. **Adaptive scheduler test**: Verify scheduler correctly falls back when acceptance is low

## CLI

```bash
# Run with speculative decoding (auto-selects draft strategy)
ronn infer --model model.onnx --speculative auto --prompt "Hello"

# Run with specific draft model
ronn infer --model model.onnx --speculative eagle --eagle-model eagle-head.onnx --prompt "Hello"

# Run with self-speculative (early exit)
ronn infer --model model.onnx --speculative self --exit-layer 16 --prompt "Hello"

# Run with n-gram drafting from a corpus
ronn infer --model model.onnx --speculative ngram --corpus docs/ --prompt "Hello"

# Train an EAGLE draft head for a target model
ronn train-eagle --target-model model.onnx --training-data data.jsonl --output eagle-head.onnx

# Benchmark speculative vs standard
ronn benchmark-spec --model model.onnx --eagle-model eagle-head.onnx --prompts bench.jsonl
```

---

# FEATURE 2: KV Cache Compression
**Branch:** `feature/kv-cache-compression`
**New crate:** `ronn-cache`
**Estimated complexity:** Medium
**Expected speedup:** 1.5-3.5x throughput, 2-4x context length

## Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/kv-cache-compression

# Clone reference implementations
git clone https://github.com/SqueezeAILab/KVQuant.git ../KVQuant-reference
git clone https://github.com/NVIDIA/kvpress.git ../kvpress-reference
```

**Key references:**
- KVQuant paper: https://arxiv.org/abs/2401.18079 (4-bit KV cache, NeurIPS 2024)
- Coupled Quantization: https://arxiv.org/abs/2405.03917 (1-bit per channel, NeurIPS 2024)
- PALU: https://arxiv.org/abs/2407.21118 (Low-rank KV compression, ICLR 2025)
- kvpress: https://github.com/NVIDIA/kvpress (NVIDIA's compression library)
- `../KVQuant-reference/` — Custom CUDA kernels for quantized attention
- `../kvpress-reference/` — Multiple compression strategies with benchmarks

## Core Problem

Every token generated requires storing its key and value vectors for all future attention computations. For a 70B model at fp16:
- Per token per layer: ~256 bytes (key) + ~256 bytes (value) = ~512 bytes
- 80 layers × 512 bytes = ~40KB per token
- 32K context = ~1.3GB of KV cache
- 128K context = ~5.2GB of KV cache

On your S1 Max with 128GB unified memory running a 70B model (~35GB quantized), KV cache is what determines your maximum context length. Compressing it 4x means 4x longer contexts.

## Why RONN's Memory Architecture Maps Here

RONN's `ronn-memory` has three tiers:
- **Working memory**: Recent, full-precision, actively attended
- **Episodic memory**: Past experiences, can be compressed
- **Semantic memory**: Long-term knowledge, heavily compressed

This maps perfectly to KV cache lifecycle:
- Recent tokens (last ~256): Full precision, working memory
- Older tokens: Quantized to 4-bit or 2-bit, episodic
- Very old tokens: Evicted or compressed to 1-bit, semantic summaries
- **Sleep consolidation** = background cache compression during idle

## Module Structure

```
crates/ronn-cache/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public API
│   ├── paged/
│   │   ├── mod.rs              # PagedAttention-style virtual memory for KV cache
│   │   ├── page_table.rs       # Page allocation, mapping, reference counting
│   │   ├── block_manager.rs    # Physical block allocation on GPU/CPU
│   │   └── copy_on_write.rs    # Shared prefixes between requests (beam search, etc.)
│   ├── quantization/
│   │   ├── mod.rs              # KV cache quantization strategies
│   │   ├── per_channel.rs      # Per-channel quantization (offline calibratable)
│   │   ├── per_token.rs        # Per-token quantization (online, handles outliers)
│   │   ├── coupled.rs          # Coupled quantization across interdependent channels
│   │   └── calibration.rs      # Offline calibration for quantization parameters
│   ├── eviction/
│   │   ├── mod.rs              # Token eviction strategies
│   │   ├── attention_sink.rs   # Keep first few tokens + recent window (StreamingLLM)
│   │   ├── heavy_hitter.rs     # Keep tokens with highest cumulative attention
│   │   ├── scoring.rs          # Scoring functions for eviction priority
│   │   └── adaptive.rs         # Dynamic eviction based on memory pressure
│   ├── compression/
│   │   ├── mod.rs              # Higher-level compression beyond quantization
│   │   ├── low_rank.rs         # PALU-style SVD compression of KV matrices
│   │   └── merging.rs          # Merge similar KV entries (semantic dedup)
│   ├── tiered.rs               # Tiered cache matching ronn-memory's 3-tier system
│   └── metrics.rs              # Cache hit rates, compression ratios, quality impact
```

## Implementation Details

### 2a. Paged Attention (`paged/`)

```rust
/// PagedAttention (from vLLM) treats KV cache like virtual memory:
///
/// Problem: Standard attention pre-allocates max_seq_len for every request,
/// wasting 60-80% of memory on padding.
///
/// Solution:
///   1. Divide KV cache into fixed-size blocks (e.g., 16 tokens per block)
///   2. Allocate blocks on-demand as sequence grows
///   3. Use a page table to map logical positions to physical blocks
///   4. Blocks can be non-contiguous in memory
///
/// Benefits:
///   - Near-zero memory waste
///   - Shared prefixes: If two requests start with the same system prompt,
///     they share the same physical KV blocks (copy-on-write)
///   - Efficient memory reclamation when requests complete
///
/// On S1 Max unified memory: Pages can live on GPU or "spill" to system RAM
/// without the PCIe penalty that discrete GPUs pay. This means paging is
/// nearly free — a huge advantage over NVIDIA hardware.
```

### 2b. Per-Channel Quantization (`per_channel.rs`)

```rust
/// KV cache activations have consistent per-channel patterns across tokens.
/// Some channels consistently have larger magnitudes than others.
///
/// Per-channel quantization:
///   1. Calibrate: Run representative data, compute per-channel min/max/scale
///   2. At runtime: Quantize each channel independently using calibrated scales
///   3. Store quantized KV cache at 4-bit or 2-bit precision
///   4. Dequantize on-the-fly during attention computation
///
/// Key insight from KVQuant: For Keys, quantize BEFORE RoPE (rotary position
/// embedding) is applied. Pre-RoPE keys have much smoother distributions and
/// quantize better.
///
/// Calibration can be done offline (ahead of time) for per-channel.
/// Store calibration data as .ronn-cache-cal files.
///
/// Reference: ../KVQuant-reference/ for calibration and kernel implementations
```

### 2c. Attention Sink + Eviction (`eviction/`)

```rust
/// Not all cached tokens are equally important. Eviction strategies drop
/// low-value tokens to bound cache size for arbitrarily long sequences.
///
/// Attention Sink (from StreamingLLM):
///   - LLMs concentrate attention on the first few tokens regardless of content
///   - Always keep tokens 0..K (the "sink", typically K=4-8) at full precision
///   - Keep a sliding window of recent tokens
///   - Evict everything in between
///
/// Heavy Hitter Oracle (H2O):
///   - Track cumulative attention scores per cached token
///   - Tokens that receive high attention across many queries = "heavy hitters"
///   - Keep heavy hitters + recent window, evict the rest
///
/// Adaptive eviction (for RONN):
///   - Use ronn-memory's importance scoring from the working memory tier
///   - Combine attention-based scoring with semantic importance
///   - Under memory pressure: more aggressive eviction
///   - When idle: consolidate (compress rather than evict)
```

### 2d. Tiered Cache (`tiered.rs`)

```rust
/// This is the RONN-specific innovation: map KV cache to the existing
/// three-tier memory architecture.
///
/// Tier 1 — Working Memory (Hot):
///   - Last ~256-512 tokens
///   - Full fp16/bf16 precision
///   - Stored on GPU / fastest memory
///   - No compression overhead
///
/// Tier 2 — Episodic Memory (Warm):
///   - Tokens 512 to ~4096
///   - Quantized to 4-bit per-channel
///   - May include attention-sink tokens at full precision
///   - Stored on GPU or unified memory
///
/// Tier 3 — Semantic Memory (Cold):
///   - Tokens beyond 4096
///   - Aggressive compression: 2-bit or 1-bit coupled quantization
///   - OR low-rank approximation (PALU-style)
///   - OR evicted entirely with summary tokens
///   - Stored in system RAM
///
/// Promotion/demotion:
///   - Tokens naturally move from Tier 1 → 2 → 3 as newer tokens arrive
///   - If attention suddenly spikes on a Tier 3 token, promote to Tier 2
///   - Sleep consolidation: background task that optimizes tier boundaries
///
/// Integration with ronn-memory:
///   - Reuse the existing tier management infrastructure
///   - Cache tiers mirror the memory tiers exactly
///   - Consolidation uses the same sleep cycle mechanism
```

## Integration Points

### ronn-memory
- Tiered cache reuses the multi-tier infrastructure
- Sleep consolidation triggers cache compression
- Importance scoring feeds into eviction decisions

### ronn-graph
- Add `KVCacheOptimizationPass` that configures cache strategy per layer
- Some layers may benefit from more aggressive compression than others

### ronn-providers
- Paged attention requires provider-specific block allocation
- GPU provider: allocate blocks in VRAM
- CPU provider: allocate blocks in system RAM
- Unified memory (S1 Max): treat as single pool with affinity hints

### ronn-speculative (Feature 1)
- When speculation fails, rejected KV entries must be efficiently reclaimed
- Paged attention makes this cheap — just free the blocks

## CLI

```bash
# Run with KV cache compression
ronn infer --model model.onnx --kv-cache-bits 4 --prompt "Hello"

# Calibrate KV quantization parameters
ronn calibrate-cache --model model.onnx --calibration-data data.jsonl --output model.ronn-cache-cal

# Run with tiered caching
ronn infer --model model.onnx --kv-cache tiered --tier1-size 512 --tier2-bits 4 --tier3-bits 2

# Run with eviction (infinite context)
ronn infer --model model.onnx --kv-cache streaming --window-size 4096 --sink-size 8

# Benchmark cache strategies
ronn benchmark-cache --model model.onnx --strategies "fp16,int4,int2,tiered" --prompts long_bench.jsonl
```

---

# FEATURE 3: Continuous Batching
**Branch:** `feature/continuous-batching`
**New crate:** `ronn-batching`
**Estimated complexity:** Medium
**Expected impact:** Throughput for concurrent requests

## Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/continuous-batching
```

**Key references:**
- vLLM PagedAttention paper: https://arxiv.org/abs/2309.06180
- Orca continuous batching: https://www.usenix.org/conference/osdi22/presentation/yu
- SGLang RadixAttention: https://arxiv.org/abs/2312.07104

## Core Problem

Standard batching: Pad all sequences to the longest, waste compute on padding tokens. When one sequence finishes, wait for entire batch before starting new sequences.

Continuous (iteration-level) batching: Each iteration can add/remove sequences. No waiting, no padding waste. Critical when Fox Den has multiple agents, MCP tools, or concurrent users hitting RONN.

## Module Structure

```
crates/ronn-batching/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── scheduler/
│   │   ├── mod.rs              # Request scheduling and prioritization
│   │   ├── fcfs.rs             # First-come-first-served (baseline)
│   │   ├── priority.rs         # Priority-based (MCP tool calls > background tasks)
│   │   └── preemption.rs       # Preempt low-priority requests under memory pressure
│   ├── batcher.rs              # Iteration-level batching engine
│   ├── sequence.rs             # Sequence state management (prefill, decode, finished)
│   ├── prefix_cache.rs         # RadixAttention — cache and reuse common prefixes
│   └── metrics.rs              # Throughput, latency percentiles, queue depth
```

## Implementation Details

### 3a. Iteration-Level Batching (`batcher.rs`)

```rust
/// Each inference iteration processes a DYNAMIC set of sequences.
///
/// Standard batching (bad):
///   Batch = [seq1(len=100), seq2(len=500), seq3(len=50)]
///   → Pad all to 500 tokens, waste 850 tokens of compute
///   → seq1 finishes first, but must wait for seq3 to finish batch
///
/// Continuous batching (good):
///   Iteration 1: Process [seq1, seq2, seq3] — each at their own length
///   Iteration 2: seq1 finishes, seq4 joins → [seq2, seq3, seq4]
///   Iteration 3: seq3 finishes, seq5 joins → [seq2, seq4, seq5]
///
/// Key states per sequence:
///   - WAITING: In queue, not yet started
///   - PREFILL: Processing initial prompt (compute-bound)
///   - DECODE: Generating tokens one at a time (memory-bound)
///   - FINISHED: Complete, results ready
///
/// Prefill and decode have different compute profiles:
///   - Prefill: High parallelism, can process all prompt tokens at once
///   - Decode: Sequential, one token per step, memory-bandwidth-bound
///   - They can be interleaved: run prefill for new sequences alongside
///     decode steps for existing sequences (chunked prefill)
```

### 3b. Prefix Caching (`prefix_cache.rs`)

```rust
/// Many requests share common prefixes:
///   - System prompts (every request starts the same)
///   - CRISP architecture docs (Fox Den always includes these)
///   - Few-shot examples
///
/// RadixAttention (from SGLang) uses a radix tree to cache KV entries
/// for common prefixes. When a new request arrives:
///   1. Match its token sequence against the radix tree
///   2. Reuse cached KV entries for the matching prefix
///   3. Only compute KV for the new, unmatched suffix
///
/// This integrates with ronn-cache (Feature 2):
///   - Cached prefixes use paged attention blocks
///   - Shared blocks use copy-on-write semantics
///   - Popular prefixes stay in Tier 1 (full precision)
///
/// This is a MASSIVE win for Fox Den:
///   - Every MCP tool call includes the same context preamble
///   - System prompts are computed once, shared across all requests
///   - Repeated agent loops (reflect → plan → act) reuse prior context
```

### 3c. Priority Scheduling (`scheduler/priority.rs`)

```rust
/// Not all requests are equal in Fox Den:
///
/// Priority levels:
///   P0 (Critical): User-facing interactive responses
///   P1 (High):     MCP tool calls that block user flow
///   P2 (Normal):   Background agent tasks
///   P3 (Low):      Batch processing, indexing, fine-tuning data generation
///
/// The scheduler:
///   1. Maintains priority queues
///   2. Admits requests based on available memory (KV cache budget)
///   3. Can preempt P3 requests to make room for P0
///   4. Preemption options:
///      a. Swap: Move KV cache to CPU RAM, resume later
///      b. Recompute: Drop KV cache, recompute when resumed
///      c. On S1 Max: Swap is nearly free due to unified memory
```

## Integration Points

### ronn-cache (Feature 2)
- Paged attention provides the memory management for batched sequences
- Prefix cache stores shared KV blocks
- Memory budget enforcement for scheduling decisions

### ronn-speculative (Feature 1)
- Speculative decoding within batched inference requires careful scheduling
- Draft tokens from one sequence shouldn't starve compute for others
- Consider: speculate only for P0 requests, standard decode for P2/P3

### ronn-api
- Extend the API to support concurrent request submission
- Add streaming response support (tokens as they're generated)
- Request priority as an API parameter

### ronn-hrm
- HRM can influence scheduling: if HRM detects a request needs System 2
  deep reasoning, allocate more compute budget and deprioritize others

## CLI

```bash
# Start RONN as a server with continuous batching
ronn serve --model model.onnx --max-batch-size 32 --port 8080

# Configure scheduling
ronn serve --model model.onnx --scheduler priority --preemption swap

# Enable prefix caching
ronn serve --model model.onnx --prefix-cache --prefix-cache-size 4096
```

---

# FEATURE 4: Early Exit Inference
**Branch:** `feature/early-exit`
**New crate:** Integrated into `ronn-hrm` (extends existing crate)
**Estimated complexity:** Low-Medium
**Expected speedup:** 1.5-2.5x

## Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/early-exit
```

**Key references:**
- SpecEE paper: https://dl.acm.org/doi/10.1145/3695053.3730996
- LayerSkip: https://arxiv.org/abs/2404.16710
- CALM (Confident Adaptive Language Modeling): https://arxiv.org/abs/2207.07061

## Core Concept

Not every token needs the full depth of the model. "The" probably doesn't need 80 layers to predict. Early exit lets "easy" tokens bail out at layer 16 while "hard" tokens use all 80 layers.

This is the SIMPLEST optimization to implement because it extends ronn-hrm's existing System 1/2 routing rather than requiring a new crate.

## Implementation

### Extend `ronn-hrm` with early exit capability

```rust
/// Add to ronn-hrm/src/:
///
/// early_exit.rs — Early exit decision module
///
/// At each layer boundary, a lightweight confidence estimator decides:
///   "Is the current hidden state confident enough to produce the output token?"
///
/// Confidence estimation options (implement all three, benchmark):
///
/// 1. Softmax entropy:
///    - Apply LM head to intermediate hidden state
///    - Compute entropy of the resulting distribution
///    - Low entropy = high confidence = can exit
///    - Threshold: entropy < 0.5 nats (tunable)
///
/// 2. Cosine similarity:
///    - Compare hidden state at layer L with hidden state at layer L-1
///    - If they're very similar, additional layers won't change the output
///    - Threshold: cosine_sim > 0.99 (tunable)
///
/// 3. Trained classifier:
///    - Small MLP per layer: hidden_state → binary (exit/continue)
///    - Train on calibration data with ground truth "when would output change"
///    - Most accurate but requires training
///
/// HRM integration:
///   System 1 = exit at early layer (fast, ~60% of tokens)
///   System 2 = run to completion (accurate, ~40% of tokens)
///   The HRM router uses confidence estimation to make this decision
///   Track per-layer exit statistics to optimize thresholds
```

### Combining with Speculative Decoding (Feature 1)

```rust
/// Early exit + speculative decoding = self-speculative decoding.
/// This is implemented in Feature 1's self_draft.rs, but the early exit
/// confidence estimation lives here in ronn-hrm.
///
/// The flow:
///   1. Run layers 1-16, check confidence
///   2. If confident: propose this as a draft token (System 1)
///   3. Continue running layers 17-80 on the PREVIOUS token to verify (System 2)
///   4. Pipeline: layer 1-16 of token T+1 overlaps with layer 17-80 of token T
///
/// This achieves speculative decoding speedup WITHOUT any draft model.
```

### Module additions to ronn-hrm

```
crates/ronn-hrm/src/
├── ... (existing files)
├── early_exit/
│   ├── mod.rs              # Early exit controller
│   ├── confidence.rs       # Confidence estimation (entropy, cosine, trained)
│   ├── calibration.rs      # Threshold calibration from sample data
│   └── statistics.rs       # Per-layer exit rate tracking
```

## Integration Points

### ronn-graph
- Add an optimization pass that inserts exit checks at configurable layer boundaries
- Not every layer needs a check — typically every 4th or 8th layer

### ronn-speculative (Feature 1)
- Self-speculative mode uses early exit as its draft mechanism
- Shared confidence estimation code

### ronn-cache (Feature 2)
- Tokens that exit early still need KV cache entries for subsequent attention
- But: early-exit tokens' KV entries can be compressed more aggressively
  (they were "easy" tokens, their cached representations matter less)

## CLI

```bash
# Run with early exit
ronn infer --model model.onnx --early-exit --min-layers 16 --confidence-threshold 0.5

# Calibrate exit thresholds
ronn calibrate-exit --model model.onnx --calibration-data data.jsonl --output exit-config.json

# Benchmark early exit vs full model
ronn benchmark-exit --model model.onnx --prompts bench.jsonl --report exit-stats.json
```

---

# FEATURE 5: Retrieval-Augmented Speculative Decoding (RASD)
**Branch:** `feature/rasd`
**Extends:** `ronn-speculative` (Feature 1) + `ronn-memory`
**Estimated complexity:** Low
**Expected speedup:** Variable (huge for RAG/code gen, minimal for open-ended chat)

## Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/rasd
```

**Key references:**
- RASD paper: https://arxiv.org/abs/2503.03434
- REST (Retrieval-Based Speculative Decoding): https://arxiv.org/abs/2311.08252
- Suffix Decoding in vLLM: uses pattern matching against previous generations

## Core Concept

When the model is likely to output text that exists in a known corpus (documentation, code, templates), skip neural computation entirely and just look it up.

This is the bridge between RONN's memory systems and its inference engine.

## Implementation

### Extend `ronn-speculative/src/draft/ngram.rs` (from Feature 1)

```rust
/// RASD extends n-gram drafting with semantic retrieval.
///
/// Two-level drafting:
///
/// Level 1 — Exact N-gram Match:
///   - Maintain a suffix array/trie over the retrieval corpus
///   - Match last N generated tokens (N=3-8) against corpus
///   - If exact match: propose continuation as draft tokens
///   - O(log n) lookup, essentially free
///
/// Level 2 — Semantic Match (fallback when no exact match):
///   - Embed the current generation context
///   - Find nearest neighbor in corpus embedding index
///   - If close enough: propose that neighbor's continuation
///   - More expensive but catches paraphrased matches
///
/// Corpus sources (from Fox Den ecosystem):
///   - FoxStash: Persistent knowledge/context storage
///   - CRISP documentation: Architecture patterns
///   - Project files: Source code under development
///   - Previous conversation history: Common response patterns
///   - User-provided RAG documents
```

### New module: `crates/ronn-speculative/src/draft/retrieval.rs`

```rust
/// The retrieval index that backs RASD.
///
/// Design:
///   1. At startup or document ingestion:
///      - Tokenize all corpus documents
///      - Build suffix array for exact matching
///      - Build embedding index for semantic matching (using RONN's own inference!)
///      - Store in ronn-memory's semantic tier
///
///   2. During inference:
///      - After each generated token, query the index
///      - If match found with confidence > threshold:
///        → Use corpus continuation as draft tokens
///      - Otherwise: fall back to neural draft model (EAGLE, etc.)
///
///   3. Index updates:
///      - New documents added to FoxStash trigger index rebuild
///      - Sleep consolidation can optimize index in background
///      - Frequency-weighted: common patterns get priority in index
///
/// Performance characteristics:
///   - Code generation with open file context: 70-90% hit rate possible
///   - RAG with injected documents: 40-60% hit rate
///   - Open-ended creative writing: <10% hit rate (fall back to neural)
///   - The scheduler (Feature 1) tracks hit rate and disables RASD when unhelpful
```

### Integration with ronn-memory

```rust
/// ronn-memory already has the semantic tier with embeddings.
/// RASD plugs into this:
///
/// 1. Corpus ingestion → ronn-memory's semantic store
/// 2. Suffix array → new data structure in semantic tier
/// 3. Embedding index → reuse ronn-memory's existing vector search
/// 4. Cache hot patterns → working memory tier
///
/// The sleep consolidation cycle:
///   - Analyze which corpus entries had high hit rates
///   - Promote them to faster lookup structures
///   - Demote low-hit entries to save memory
///   - Rebuild index with updated frequency weights
```

## CLI

```bash
# Build retrieval index from corpus
ronn index-corpus --sources ./docs ./src --output corpus.ronn-index

# Run with RASD
ronn infer --model model.onnx --speculative rasd --corpus corpus.ronn-index --prompt "Hello"

# Update index incrementally
ronn index-update --index corpus.ronn-index --add ./new_docs/

# Benchmark RASD hit rates on your actual workload
ronn benchmark-rasd --model model.onnx --corpus corpus.ronn-index --prompts real_prompts.jsonl
```

---

# Cross-Feature Integration Summary

These features compound multiplicatively when combined:

```
Standard Inference: 1x baseline
+ Activation Sparsity (Feature 0):     2-5x   (skip inactive neurons)
+ KV Cache Compression (Feature 2):    1.5-3x (longer context, less memory)
+ Speculative Decoding (Feature 1):    2-4x   (multiple tokens per step)
+ Early Exit (Feature 4):              1.5-2x (skip layers for easy tokens)
+ RASD (Feature 5):                    Variable (free tokens from lookup)
+ Continuous Batching (Feature 3):     Throughput (serve multiple requests)

Compound: 10-60x theoretical improvement for favorable workloads
Realistic: 5-15x on typical coding/RAG tasks
```

## Implementation Order Recommendation

```
Phase 1 (Foundation):
  Feature 2: KV Cache (paged attention is infrastructure for everything else)
  Feature 4: Early Exit (lowest effort, extends existing HRM)

Phase 2 (Core Acceleration):
  Feature 1: Speculative Decoding (biggest single-feature speedup)
  Feature 3: Continuous Batching (required for multi-agent Fox Den)

Phase 3 (Compound Gains):
  Feature 5: RASD (plugs into Features 1 + memory system)
  Combine all features, benchmark compound speedups

Note: Feature 0 (Activation Sparsity from the previous prompt) can be
developed in parallel with all of these.
```

## Shared Infrastructure

These features share some common needs. Consider creating a shared module:

```
crates/ronn-inference-common/
├── src/
│   ├── token_context.rs    # Shared token/sequence representation
│   ├── kv_types.rs         # Key-value cache type definitions
│   ├── sampling.rs         # Token sampling utilities (top-k, top-p, temperature)
│   ├── benchmarking.rs     # Common benchmarking framework
│   └── calibration.rs      # Shared calibration data format
```

## Global Constraints

- **Pure Rust**: No C/C++ FFI unless absolutely necessary for GPU kernels
- **No code copying**: Study reference implementations, reimplement idiomatically
- **Feature-gated**: Each feature behind its own cargo feature flag
- **Backward compatible**: All features are additive, existing API unchanged
- **Async-first**: All new code async-compatible (tokio runtime)
- **Observable**: All features expose metrics for monitoring and tuning
- **Test-driven**: Correctness tests before optimization tests
- **Hardware-agnostic**: Degrade gracefully without GPU (just slower)
