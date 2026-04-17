# RONN: KV Cache Compression Implementation

## Context

You are working on **RONN** (Rust-Optimized Neural Networks), a Rust ML inference runtime at `https://github.com/Dieshen/RONN`. RONN is a brain-inspired neural network runtime with existing crates for core inference, graph optimization, execution providers, memory systems, learning, and hierarchical reasoning.

Your task is to create a new branch `feature/kv-cache-compression` and implement KV cache management as a new crate `ronn-cache`.

## Step 0: Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/kv-cache-compression

# Clone reference implementations (study only, reimplement in Rust)
git clone https://github.com/SqueezeAILab/KVQuant.git ../KVQuant-reference
git clone https://github.com/NVIDIA/kvpress.git ../kvpress-reference
```

**Key reference files to study:**
- `../KVQuant-reference/quant/` — Per-channel/per-token quantization with outlier handling
- `../KVQuant-reference/deployment/` — Custom CUDA kernels for quantized attention
- `../kvpress-reference/kvpress/presses/` — Multiple compression strategies (KnormPress, SnapKV, etc.)
- `../kvpress-reference/kvpress/pipeline.py` — How presses are applied as forward hooks

**Key papers:**
- KVQuant (NeurIPS 2024): https://arxiv.org/abs/2401.18079 — 3-4 bit KV cache, 10M context
- Coupled Quantization (NeurIPS 2024): https://arxiv.org/abs/2405.03917 — 1-bit per channel
- PALU (ICLR 2025): https://arxiv.org/abs/2407.21118 — Low-rank SVD compression
- PagedAttention / vLLM: https://arxiv.org/abs/2309.06180 — Virtual memory for KV cache
- StreamingLLM: https://arxiv.org/abs/2309.17453 — Attention sink + sliding window

## Step 1: Understand the Existing RONN Architecture

Read every `lib.rs` and `Cargo.toml` in the existing crates. Pay special attention to:
- **ronn-memory**: Has 3-tier architecture (working/episodic/semantic) with sleep consolidation. KV cache tiers will mirror this EXACTLY.
- **ronn-providers**: How GPU/CPU execution is dispatched. Paged attention needs provider-level block allocation.
- **ronn-core**: Tensor representations. You'll be adding quantized tensor types.

## Step 2: Create the `ronn-cache` Crate

```
crates/ronn-cache/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public API, feature flags
│   ├── paged/
│   │   ├── mod.rs              # PagedAttention virtual memory system
│   │   ├── page_table.rs       # Logical-to-physical block mapping
│   │   ├── block_manager.rs    # Physical block allocation on GPU/CPU/unified
│   │   └── copy_on_write.rs    # Shared prefix blocks (beam search, system prompts)
│   ├── quantization/
│   │   ├── mod.rs              # CacheQuantizer trait and implementations
│   │   ├── per_channel.rs      # Offline-calibrated per-channel quantization
│   │   ├── per_token.rs        # Online per-token quantization (handles outliers)
│   │   ├── coupled.rs          # Coupled quantization across interdependent channels
│   │   ├── calibration.rs      # Offline calibration data generation
│   │   └── kernels.rs          # Quantized matmul for attention computation
│   ├── eviction/
│   │   ├── mod.rs              # EvictionPolicy trait
│   │   ├── attention_sink.rs   # StreamingLLM: keep first K tokens + recent window
│   │   ├── heavy_hitter.rs     # H2O: keep high cumulative attention tokens
│   │   └── adaptive.rs         # Memory-pressure-driven eviction
│   ├── compression/
│   │   ├── mod.rs              # Beyond quantization
│   │   ├── low_rank.rs         # PALU-style SVD decomposition of KV matrices
│   │   └── merging.rs          # Merge semantically similar KV entries
│   ├── tiered.rs               # Three-tier cache matching ronn-memory architecture
│   └── metrics.rs              # Compression ratios, quality impact, memory savings
```

### Cargo.toml

```toml
[package]
name = "ronn-cache"
version = "0.1.0"
edition = "2021"
description = "KV cache management and compression for RONN inference runtime"

[features]
default = ["paged", "quantization"]
paged = []                              # PagedAttention virtual memory
quantization = []                       # KV cache quantization (4-bit, 2-bit)
eviction = []                           # Token eviction strategies
low-rank = ["dep:nalgebra"]             # SVD-based compression
tiered = ["paged", "quantization", "eviction"]  # Full tiered cache system
full = ["tiered", "low-rank"]

[dependencies]
ronn-core = { path = "../ronn-core" }
ronn-providers = { path = "../ronn-providers" }
ronn-memory = { path = "../ronn-memory" }
nalgebra = { version = "0.33", optional = true }
tokio = { version = "1", features = ["full"] }
tracing = "0.1"

[dev-dependencies]
criterion = "0.5"
approx = "0.5"
```

## Step 3: Core Problem Understanding

### Why KV cache is the bottleneck

For a 70B model at fp16 with 80 layers, each with 64 attention heads × 128 dim:
- Per token: 80 layers × 2 (K+V) × 64 heads × 128 dim × 2 bytes = ~2.6MB
- 8K context: ~20GB of KV cache
- 32K context: ~83GB of KV cache
- On 128GB S1 Max running 70B q4 (~35GB): KV cache at fp16 limits you to ~35K context

Compressing KV cache 4x (to 4-bit) means 4x longer contexts or 4x more concurrent requests.

### Key insight: KV activations have exploitable structure

1. **Per-channel patterns**: Some channels consistently have larger magnitudes. Calibrate once, quantize efficiently.
2. **Pre-RoPE keys quantize better**: Key vectors BEFORE rotary position embedding have smoother distributions.
3. **Attention sinks**: First few tokens receive disproportionate attention regardless of content.
4. **Outlier tokens**: A few tokens have extreme values that dominate quantization error. Handle them separately.

## Step 4: Implement Paged Attention

### `paged/page_table.rs`

```rust
/// PagedAttention: virtual memory for KV cache.
///
/// Problem: Standard attention pre-allocates max_seq_len for every request.
///   - Request 1: actual length 100, allocated 2048 → 95% waste
///   - Request 2: actual length 2000, allocated 2048 → 2% waste
///   - Average: ~50-60% memory wasted
///
/// Solution: Divide KV cache into fixed-size blocks, allocate on demand.
///
/// Design (study vLLM paper Section 4):
///   - Block size: 16 tokens (tunable, powers of 2 for alignment)
///   - Each block stores K and V for 16 tokens across all heads
///   - Page table maps: (sequence_id, logical_block_idx) → physical_block_id
///   - Blocks are non-contiguous in memory (like OS virtual memory)
///
/// Copy-on-write for shared prefixes:
///   - System prompt KV cache computed once
///   - Multiple requests share the same physical blocks
///   - Only allocate new blocks when sequences diverge
///   - Reference counting on physical blocks
///
/// On S1 Max unified memory:
///   - Physical blocks can live in GPU VRAM or system RAM
///   - Unlike discrete GPU, "spilling" to system RAM has NO PCIe penalty
///   - This is a massive advantage: paged attention is nearly free

pub struct PageTable {
    /// Maps (sequence_id, logical_block_index) → physical_block_id
    mappings: HashMap<(u64, usize), PhysicalBlockId>,

    /// Block metadata: reference count, memory location, precision
    block_metadata: Vec<BlockMetadata>,
}

pub struct BlockMetadata {
    /// Reference count (for copy-on-write shared blocks)
    ref_count: u32,

    /// Where this block physically resides
    location: BlockLocation,

    /// Precision of stored KV values
    precision: CachePrecision,

    /// Number of tokens actually stored (may be < block_size for last block)
    num_tokens: u16,
}

pub enum BlockLocation {
    /// In GPU VRAM (fastest)
    Gpu { device_id: u32, offset: usize },

    /// In system RAM
    Cpu { offset: usize },

    /// Unified memory (S1 Max) — single address space, GPU/CPU accessible
    Unified { offset: usize, affinity: MemoryAffinity },
}

pub enum CachePrecision {
    Fp16,
    Bf16,
    Int8,
    Int4,
    Int2,
    Int1,  // Coupled quantization
}
```

### `paged/block_manager.rs`

```rust
/// Manages physical block allocation across memory tiers.
///
/// The block manager is the allocator for the paged attention system.
/// It tracks free blocks, handles allocation/deallocation, and manages
/// memory pressure by triggering eviction or tier demotion.
///
/// On startup:
///   1. Query available GPU memory and system RAM
///   2. Pre-allocate block pools in each tier
///   3. Reserve some GPU memory for model weights and activations
///
/// Allocation strategy:
///   1. Try GPU first (fastest)
///   2. If GPU full: try unified memory (S1 Max only)
///   3. If unified full: try CPU RAM
///   4. If all full: trigger eviction or reject request
///
/// Integration with ronn-memory:
///   The block manager's tier logic mirrors ronn-memory's tier system.
///   Working memory blocks → GPU/fast tier
///   Episodic blocks → unified/medium tier
///   Semantic blocks → CPU/slow tier (heavily compressed)

pub struct BlockManager {
    /// Pool of GPU blocks
    gpu_pool: BlockPool,

    /// Pool of CPU blocks
    cpu_pool: BlockPool,

    /// VRAM budget (bytes) — respect this limit
    vram_budget: usize,

    /// Block size in tokens
    block_size: usize,
}
```

## Step 5: Implement Quantization

### `quantization/per_channel.rs` — Implement first (most practical)

```rust
/// Per-channel quantization: offline calibratable, no runtime overhead.
///
/// Algorithm:
///   1. CALIBRATION (offline, one-time):
///      a. Run representative data through the model
///      b. For each layer, each attention head, each channel:
///         - Record min/max activation values
///         - Compute optimal scale and zero-point for target bit-width
///      c. Save calibration parameters to .ronn-cache-cal file
///
///   2. RUNTIME (every token):
///      a. Compute K, V at full precision (normal forward pass)
///      b. Quantize K, V using pre-computed per-channel scales:
///         q = round((x - zero_point) / scale)
///         q = clamp(q, min_val, max_val)
///      c. Store quantized values in paged blocks
///      d. During attention: dequantize on-the-fly before matmul
///         x_approx = q * scale + zero_point
///
/// KEY INSIGHT from KVQuant:
///   Quantize Keys BEFORE RoPE (rotary position embedding).
///   Pre-RoPE keys have per-channel patterns that are consistent
///   across tokens. Post-RoPE keys have position-dependent patterns
///   that are harder to quantize.
///
///   This means: intercept key computation AFTER the K projection
///   but BEFORE RoPE application. Quantize there. Apply RoPE to
///   dequantized values during attention.
///
/// Reference: ../KVQuant-reference/quant/qKV.py for the quantization logic
/// Reference: ../KVQuant-reference/quant/calibrate.py for calibration

pub struct PerChannelQuantizer {
    /// Per-layer, per-head, per-channel quantization parameters
    /// Shape: [num_layers][num_heads][head_dim]
    key_scales: Vec<Vec<Vec<f32>>>,
    key_zero_points: Vec<Vec<Vec<f32>>>,
    value_scales: Vec<Vec<Vec<f32>>>,
    value_zero_points: Vec<Vec<Vec<f32>>>,

    /// Target bit-width (4, 3, or 2)
    bits: u8,
}

/// Calibration data format
pub struct CacheCalibration {
    /// Model identifier (to verify calibration matches model)
    model_hash: String,

    /// Bit-width this calibration was computed for
    bits: u8,

    /// Per-layer calibration parameters
    layers: Vec<LayerCalibration>,
}

pub struct LayerCalibration {
    pub key_scales: Vec<Vec<f32>>,       // [num_heads][head_dim]
    pub key_zero_points: Vec<Vec<f32>>,
    pub value_scales: Vec<Vec<f32>>,
    pub value_zero_points: Vec<Vec<f32>>,
    pub key_outlier_thresholds: Vec<Vec<f32>>,  // For dense-and-sparse
}
```

### `quantization/kernels.rs` — Quantized attention

```rust
/// Efficient attention computation with quantized KV cache.
///
/// The attention operation with quantized cache:
///   1. Dequantize K block: K_fp16 = K_int4 * scale + zero_point
///   2. Compute attention scores: scores = Q @ K_fp16^T / sqrt(d)
///   3. Apply softmax
///   4. Dequantize V block: V_fp16 = V_int4 * scale + zero_point
///   5. Compute output: out = softmax(scores) @ V_fp16
///
/// Optimization: fuse dequantization with matmul.
///   Instead of separate dequant + matmul:
///   out = Q @ (K_int4 * scale + zero_point)^T / sqrt(d)
///       = Q @ K_int4^T * scale^T / sqrt(d) + Q @ zero_point^T / sqrt(d)
///   The second term can be precomputed per query.
///
/// For GPU: implement as custom kernel (or use ronn-providers abstraction)
/// For CPU: use SIMD-friendly int4 unpacking + fused multiply-add
///
/// Reference: ../KVQuant-reference/deployment/ for CUDA kernel patterns
```

## Step 6: Implement Eviction Strategies

### `eviction/attention_sink.rs`

```rust
/// StreamingLLM attention sink eviction.
///
/// Observation: LLMs concentrate attention on the first few tokens
/// regardless of their content (they act as "attention sinks").
/// Removing these tokens causes catastrophic attention collapse.
///
/// Strategy:
///   1. ALWAYS keep tokens 0..sink_size (typically 4-8) at full precision
///   2. Keep a sliding window of the most recent `window_size` tokens
///   3. Everything between sinks and window: evict (free the blocks)
///
/// Result: Bounded memory usage regardless of sequence length.
///   Memory = (sink_size + window_size) * per_token_cost
///
/// Quality tradeoff:
///   - window_size=512: Good for most conversational tasks
///   - window_size=2048: Better for long-document tasks
///   - window_size=4096+: Approaches full-context quality
///
/// Integration with tiered cache:
///   Instead of hard eviction, demote to Tier 3 (aggressive compression)
///   This way, the information isn't lost, just harder to access.

pub struct AttentionSinkEviction {
    /// Number of initial tokens to always keep (attention sinks)
    sink_size: usize,

    /// Size of the recent token window
    window_size: usize,
}
```

### `eviction/heavy_hitter.rs`

```rust
/// H2O (Heavy-Hitter Oracle): keep tokens with highest cumulative attention.
///
/// Observation: A small fraction of tokens receive most of the attention
/// across all queries. These "heavy hitters" carry the most information.
///
/// Algorithm:
///   1. During attention computation, accumulate attention scores per KV token
///   2. Maintain a running sum: importance[token_pos] += sum(attention_weights)
///   3. When cache exceeds budget: evict tokens with lowest importance
///   4. Always keep attention sinks (first K tokens) regardless of score
///
/// This is more adaptive than sliding window because it keeps
/// IMPORTANT tokens even if they're old, and drops UNIMPORTANT
/// recent tokens.
///
/// Cost: O(1) per attention operation (just accumulate scores).
/// Eviction: O(n log k) to find top-k when triggered.

pub struct HeavyHitterEviction {
    /// Cumulative attention scores per position
    importance_scores: Vec<f32>,

    /// Maximum cache budget (in tokens)
    budget: usize,

    /// How many attention sinks to always protect
    sink_size: usize,

    /// Eviction trigger: evict when cache exceeds budget * trigger_ratio
    trigger_ratio: f32,
}
```

## Step 7: Implement Tiered Cache

### `tiered.rs` — The RONN-specific innovation

```rust
/// Three-tier KV cache matching ronn-memory's architecture.
///
/// This is where RONN differentiates from every other inference runtime.
/// Instead of a flat cache with one compression level, RONN manages
/// KV cache like its brain-inspired memory system.
///
/// ┌──────────────────────────────────────────────────────────┐
/// │  Tier 1: WORKING MEMORY (Hot)                            │
/// │  - Last 256-512 tokens                                   │
/// │  - Full fp16/bf16 precision                              │
/// │  - GPU VRAM / fastest unified memory                     │
/// │  - Zero compression overhead                             │
/// │  - Includes attention sink tokens (always full precision) │
/// ├──────────────────────────────────────────────────────────┤
/// │  Tier 2: EPISODIC MEMORY (Warm)                          │
/// │  - Tokens 512 to ~4096                                   │
/// │  - 4-bit per-channel quantization                        │
/// │  - GPU VRAM or unified memory                            │
/// │  - 4x compression vs fp16                                │
/// │  - Heavy hitter tokens promoted back to Tier 1 if needed │
/// ├──────────────────────────────────────────────────────────┤
/// │  Tier 3: SEMANTIC MEMORY (Cold)                          │
/// │  - Tokens beyond 4096                                    │
/// │  - 2-bit coupled quantization OR low-rank SVD            │
/// │  - System RAM (on discrete GPU) or cold unified memory   │
/// │  - 8-16x compression vs fp16                             │
/// │  - OR evicted entirely (replaced with summary tokens)    │
/// └──────────────────────────────────────────────────────────┘
///
/// Token lifecycle:
///   1. New token → Tier 1 (full precision, fast)
///   2. As newer tokens arrive, older tokens demote: Tier 1 → 2 → 3
///   3. Attention spike on cold token → promote back to Tier 2
///   4. Under memory pressure → aggressive eviction from Tier 3
///
/// Sleep consolidation (from ronn-memory):
///   During idle periods, a background task:
///   1. Analyzes which Tier 2 tokens are never attended → demote to Tier 3
///   2. Optimizes Tier 3 compression (recalibrate quantization, rebuild SVD)
///   3. Pre-computes summaries for very old context
///   4. Defragments page table (consolidate partially-full blocks)
///
/// The tiered cache uses paged attention (Step 4) for memory management,
/// quantization (Step 5) for compression, and eviction (Step 6) for
/// budget enforcement. It orchestrates all three.

pub struct TieredCache {
    /// Page table for virtual-to-physical block mapping
    page_table: PageTable,

    /// Block manager for allocation across memory tiers
    block_manager: BlockManager,

    /// Tier 1 configuration
    tier1: TierConfig,

    /// Tier 2 configuration
    tier2: TierConfig,

    /// Tier 3 configuration
    tier3: TierConfig,

    /// Eviction policy for each tier
    eviction_policy: Box<dyn EvictionPolicy>,

    /// Quantizer for tier transitions
    quantizer: Box<dyn CacheQuantizer>,

    /// Metrics
    metrics: CacheMetrics,
}

pub struct TierConfig {
    /// Maximum tokens in this tier
    max_tokens: usize,

    /// Precision for this tier
    precision: CachePrecision,

    /// Memory location preference
    preferred_location: BlockLocation,
}

impl TieredCache {
    /// Store a new KV entry. Automatically placed in Tier 1.
    pub fn store(&mut self, layer: usize, position: usize, key: &Tensor, value: &Tensor) -> Result<(), CacheError> {
        // 1. Allocate a slot in Tier 1
        // 2. Store at full precision
        // 3. If Tier 1 is full: demote oldest to Tier 2 (quantize)
        // 4. If Tier 2 is full: demote oldest to Tier 3 (compress further)
        // 5. If Tier 3 is full: evict using eviction policy
        todo!()
    }

    /// Retrieve KV entries for attention computation.
    /// Transparently dequantizes from whatever tier the data is in.
    pub fn retrieve(&self, layer: usize, positions: &[usize]) -> Result<(Tensor, Tensor), CacheError> {
        // 1. Look up positions in page table
        // 2. For each position, dequantize from stored precision to compute precision
        // 3. Track access patterns for heavy hitter scoring
        // 4. If a cold token is accessed frequently, flag for promotion
        todo!()
    }

    /// Background consolidation (called by ronn-memory's sleep cycle).
    pub async fn consolidate(&mut self) {
        // 1. Analyze access patterns since last consolidation
        // 2. Promote frequently-accessed cold tokens
        // 3. Demote rarely-accessed warm tokens
        // 4. Recalibrate quantization parameters if distributions shifted
        // 5. Defragment page table
        todo!()
    }
}
```

## Step 8: Integration with Existing RONN Crates

### ronn-memory Integration
- TieredCache mirrors and reuses ronn-memory's 3-tier infrastructure
- Sleep consolidation triggers cache compression via `consolidate()`
- Importance scoring in ronn-memory feeds into eviction decisions

### ronn-graph Integration
- Add `KVCacheOptimizationPass` that selects cache strategy per layer
- Insert quantize/dequantize nodes around KV storage/retrieval
- Some layers may benefit from more aggressive compression

### ronn-providers Integration
- `BlockAllocator` trait for provider-specific memory allocation
- GPU provider: VRAM blocks with CUDA memory alignment
- CPU provider: System RAM with cache-line alignment
- Unified provider (S1 Max): Single pool with GPU/CPU affinity hints

### ronn-speculative (companion feature)
- When speculation fails, rejected KV entries reclaimed via page deallocation
- Paged attention makes this O(1) — just decrement reference count
- Shared system prompt blocks amortize cost across speculative branches

## Step 9: CLI Commands

```bash
# Run with 4-bit KV cache
ronn infer --model model.onnx --kv-bits 4 --prompt "Hello"

# Calibrate quantization parameters (offline, one-time)
ronn calibrate-cache --model model.onnx --data calibration.jsonl --bits 4 --output model.ronn-cache-cal

# Run with tiered caching
ronn infer --model model.onnx --kv-cache tiered --tier1-tokens 512 --tier2-bits 4 --tier3-bits 2

# Run with streaming (infinite context via eviction)
ronn infer --model model.onnx --kv-cache streaming --window 4096 --sinks 8

# Benchmark cache strategies
ronn benchmark-cache --model model.onnx --strategies "fp16,int4,int2,tiered" --prompts long_context.jsonl

# Show cache memory usage during inference
ronn infer --model model.onnx --kv-bits 4 --cache-stats --prompt "Long document..."
```

## Step 10: Testing Strategy

1. **Quality preservation**: Perplexity with quantized cache must be within 1% of fp16 baseline at 4-bit, within 5% at 2-bit.
2. **Paged attention correctness**: Output must be identical to standard (non-paged) attention.
3. **Copy-on-write**: Shared prefix blocks must not corrupt when one sequence diverges.
4. **Tier transitions**: Verify tokens demote/promote correctly and data is preserved.
5. **Eviction correctness**: Attention sinks are never evicted. Heavy hitters survive longer.
6. **Memory budget**: Cache never exceeds configured VRAM budget.
7. **Consolidation**: Background consolidation doesn't corrupt active inference.

## Implementation Phases for Claude Code

**Phase 1 (Steps 0-3):** Setup, read existing code, create crate skeleton, understand the problem.

**Phase 2 (Step 4):** Paged attention — page table, block manager, allocation/deallocation. This is infrastructure for everything else. Test with fp16 first (no compression).

**Phase 3 (Step 5):** Per-channel quantization with calibration. Get 4-bit KV cache working with paged attention. Benchmark quality and memory savings.

**Phase 4 (Step 6):** Eviction strategies — attention sink first, then heavy hitter.

**Phase 5 (Steps 7-8):** Tiered cache system, integration with ronn-memory, CLI commands.

## Success Criteria

- [ ] Paged attention produces identical output to standard attention
- [ ] 4-bit KV cache within 1% perplexity of fp16 baseline
- [ ] 4x memory reduction demonstrated on real model
- [ ] Copy-on-write correctly shares system prompt KV blocks
- [ ] Streaming mode handles sequences longer than cache budget
- [ ] Tiered cache correctly promotes/demotes based on access patterns
- [ ] Sleep consolidation optimizes cache without corrupting inference
- [ ] No breaking changes to existing RONN API
- [ ] All features behind cargo feature flags
