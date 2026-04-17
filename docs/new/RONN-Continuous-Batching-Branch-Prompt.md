# RONN: Continuous Batching Implementation

## Context

You are working on **RONN** (Rust-Optimized Neural Networks), a Rust ML inference runtime at `https://github.com/Dieshen/RONN`. Your task is to create a new branch `feature/continuous-batching` and implement iteration-level continuous batching as a new crate `ronn-batching`.

This feature is **infrastructure** — it enables RONN to serve multiple concurrent requests efficiently, which is required for the Fox Den multi-agent orchestration platform.

## Step 0: Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/continuous-batching
```

**Key references (study concepts, not code):**
- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180 — Core paged attention + batching system
- Orca: https://www.usenix.org/conference/osdi22/presentation/yu — Pioneered iteration-level scheduling
- SGLang RadixAttention: https://arxiv.org/abs/2312.07104 — Prefix caching via radix tree

## Step 1: Understand Existing RONN Architecture

Read every `lib.rs` and `Cargo.toml`. Key crates:
- **ronn-api**: Current API layer and MCP integration. This is where incoming requests arrive.
- **ronn-hrm**: System 1/2 routing. Will influence request prioritization.
- **ronn-memory**: Tier management. Relevant for prefix caching.
- **ronn-cache** (companion feature): If available, paged attention provides memory management for batched sequences.

## Step 2: Create the `ronn-batching` Crate

```
crates/ronn-batching/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── request.rs          # Request/sequence lifecycle management
│   ├── batcher.rs          # Iteration-level batching engine
│   ├── scheduler/
│   │   ├── mod.rs          # Scheduler trait
│   │   ├── fcfs.rs         # First-come-first-served (baseline)
│   │   ├── priority.rs     # Priority-based (interactive > background)
│   │   └── preemption.rs   # Preempt low-priority under memory pressure
│   ├── prefix_cache.rs     # Radix tree prefix caching
│   ├── server.rs           # HTTP/gRPC server wrapping the batcher
│   └── metrics.rs          # Throughput, latency percentiles, queue depth
```

### Cargo.toml

```toml
[package]
name = "ronn-batching"
version = "0.1.0"
edition = "2021"
description = "Continuous batching and request scheduling for RONN"

[features]
default = ["http"]
http = ["dep:axum", "dep:tower"]
grpc = ["dep:tonic"]

[dependencies]
ronn-core = { path = "../ronn-core" }
ronn-providers = { path = "../ronn-providers" }
ronn-hrm = { path = "../ronn-hrm" }
ronn-cache = { path = "../ronn-cache", optional = true }
axum = { version = "0.7", optional = true }
tower = { version = "0.4", optional = true }
tonic = { version = "0.12", optional = true }
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
dashmap = "6"

[dev-dependencies]
criterion = "0.5"
reqwest = { version = "0.12", features = ["json"] }
```

## Step 3: Core Data Structures

### `request.rs` — Request Lifecycle

```rust
/// A request goes through these states:
///
///   WAITING → PREFILL → DECODING → FINISHED
///                ↓           ↓
///           PREEMPTED   PREEMPTED
///                ↓           ↓
///              (back to WAITING with saved state)
///
/// WAITING: Queued, not yet started. No GPU memory allocated.
/// PREFILL: Processing initial prompt tokens (compute-bound, parallelizable).
/// DECODING: Generating tokens one at a time (memory-bandwidth-bound).
/// FINISHED: Complete. Results available, memory being freed.
/// PREEMPTED: Paused to make room for higher priority request.

pub struct InferenceRequest {
    /// Unique request identifier
    pub id: RequestId,

    /// Input token IDs
    pub prompt_tokens: Vec<u32>,

    /// Generation parameters
    pub config: GenerationConfig,

    /// Priority level
    pub priority: Priority,

    /// Current state
    pub state: RequestState,

    /// Tokens generated so far
    pub output_tokens: Vec<u32>,

    /// Channel to send streaming tokens back to caller
    pub response_tx: tokio::sync::mpsc::Sender<TokenEvent>,

    /// Timestamp for latency tracking
    pub created_at: std::time::Instant,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// User-facing interactive response
    Critical = 0,
    /// MCP tool call blocking user flow
    High = 1,
    /// Background agent task
    Normal = 2,
    /// Batch processing, indexing
    Low = 3,
}

pub enum RequestState {
    Waiting,
    Prefill { tokens_processed: usize },
    Decoding { kv_cache_handle: CacheHandle },
    Preempted { saved_state: PreemptedState },
    Finished { reason: FinishReason },
}

pub enum FinishReason {
    EosToken,
    MaxTokens,
    StopSequence,
    Cancelled,
    Error(String),
}
```

### `batcher.rs` — The Core Engine

```rust
/// Iteration-level continuous batching engine.
///
/// Standard batching (bad):
///   Form a fixed batch, process until ALL sequences finish, repeat.
///   If batch = [seq_100_tokens, seq_5000_tokens], the short sequence
///   wastes GPU time waiting for the long one.
///
/// Continuous batching (good):
///   Every ITERATION (single forward pass), the batcher can:
///   - Add new sequences that were waiting
///   - Remove sequences that just finished
///   - Preempt sequences to free memory
///
/// The batcher maintains two groups processed each iteration:
///   1. PREFILL group: New sequences having their prompts processed.
///      These are compute-bound and benefit from parallelism.
///   2. DECODE group: Existing sequences generating one token each.
///      These are memory-bandwidth-bound.
///
/// Chunked prefill: Instead of processing entire prompts at once,
///   break them into chunks and interleave with decode steps.
///   This prevents a long prompt from blocking decode for existing sequences.
///
/// Each iteration:
///   1. Scheduler selects which requests to run
///   2. Batcher assembles input tensors (batched across sequences)
///   3. Forward pass processes all sequences simultaneously
///   4. Batcher distributes outputs to individual sequences
///   5. Finished sequences removed, new sequences admitted

pub struct ContinuousBatcher {
    /// All active and waiting requests
    requests: DashMap<RequestId, InferenceRequest>,

    /// Scheduler decides what runs each iteration
    scheduler: Box<dyn Scheduler>,

    /// The model being served
    model: Box<dyn ronn_core::Model>,

    /// Maximum sequences in a single batch
    max_batch_size: usize,

    /// Maximum tokens across all sequences per iteration
    max_batch_tokens: usize,

    /// Prefill chunk size (0 = no chunking)
    prefill_chunk_size: usize,
}

impl ContinuousBatcher {
    /// Submit a new inference request. Returns immediately.
    /// Tokens stream back via the request's response channel.
    pub async fn submit(&self, request: InferenceRequest) -> Result<RequestId, BatchError> {
        // 1. Validate request
        // 2. Add to waiting queue
        // 3. Return ID (actual processing happens in run_loop)
        todo!()
    }

    /// The main batching loop. Runs continuously while server is active.
    pub async fn run_loop(&mut self) -> Result<(), BatchError> {
        loop {
            // 1. Ask scheduler which requests to run this iteration
            let batch = self.scheduler.select_batch(&self.requests, self.max_batch_size, self.max_batch_tokens);

            if batch.is_empty() {
                // No work to do, brief sleep to avoid busy-waiting
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                continue;
            }

            // 2. Separate into prefill and decode groups
            let (prefill_group, decode_group) = batch.partition_by_state();

            // 3. Assemble batched input tensors
            // - Prefill: concatenated prompt chunks with position offsets
            // - Decode: single token per sequence + KV cache references
            let batch_input = self.assemble_batch(&prefill_group, &decode_group);

            // 4. Single forward pass through the model
            let batch_output = self.model.forward(&batch_input).await?;

            // 5. Distribute outputs
            // - Prefill sequences: advance position, transition to DECODING if done
            // - Decode sequences: sample token, check for EOS, stream to client
            self.distribute_outputs(&batch_output, &prefill_group, &decode_group).await;

            // 6. Cleanup finished sequences, admit new ones
            self.cleanup_and_admit().await;
        }
    }

    /// Cancel a running request.
    pub async fn cancel(&self, id: RequestId) -> Result<(), BatchError> {
        todo!()
    }
}
```

## Step 4: Implement Scheduling

### `scheduler/priority.rs`

```rust
/// Priority scheduler for Fox Den multi-agent workloads.
///
/// Fox Den sends different types of requests:
///   P0 (Critical): User typing in chat, waiting for response
///   P1 (High):     MCP tool call that blocks the agent loop
///   P2 (Normal):   Background agent reasoning/planning
///   P3 (Low):      Batch indexing, embedding generation
///
/// The scheduler:
///   1. Always processes P0 requests first
///   2. P1 requests admitted if memory allows
///   3. P2/P3 only run when there's spare capacity
///   4. Under memory pressure: preempt P3 → P2 → P1 (never P0)
///
/// Preemption strategies:
///   - Swap: Save KV cache to CPU RAM, resume later
///     (On S1 Max unified memory, this is nearly free)
///   - Recompute: Discard KV cache, recompute from prompt when resumed
///     (Cheaper if preempted for a long time)
///   - The scheduler picks based on expected preemption duration

pub struct PriorityScheduler {
    /// Priority queues
    queues: [VecDeque<RequestId>; 4],

    /// Memory budget tracker
    memory_budget: MemoryBudget,

    /// How to handle preemption
    preemption_policy: PreemptionPolicy,
}

pub enum PreemptionPolicy {
    /// Save KV cache to CPU RAM (fast resume, uses more memory)
    Swap,
    /// Discard KV cache (slow resume, saves memory)
    Recompute,
    /// Auto-select based on available memory and expected wait time
    Auto,
}
```

## Step 5: Implement Prefix Caching

### `prefix_cache.rs`

```rust
/// RadixAttention-style prefix caching.
///
/// Many requests share common prefixes:
///   - System prompt: "You are a helpful assistant..."
///   - CRISP architecture docs injected by Fox Den
///   - Few-shot examples
///   - Common tool descriptions
///
/// The prefix cache uses a radix tree (trie) indexed by token sequences.
/// Each node stores a reference to KV cache blocks for that prefix.
///
/// Example:
///   Request A: [sys_prompt] + "Write a function..."
///   Request B: [sys_prompt] + "Explain this code..."
///   Request C: [sys_prompt] + [crisp_docs] + "Build a service..."
///
///   Radix tree:
///     [sys_prompt] → KV blocks (shared by A, B, C)
///       ├── "Write..." → KV blocks (A only)
///       ├── "Explain..." → KV blocks (B only)
///       └── [crisp_docs] → KV blocks (shared by C and future similar requests)
///            └── "Build..." → KV blocks (C only)
///
/// On first request with a prefix: compute and cache KV blocks.
/// On subsequent requests with same prefix: skip prefill for cached portion.
///
/// This is a MASSIVE win for Fox Den where every request includes
/// the same system prompt and architecture context.
///
/// Integration with ronn-cache:
///   Cached prefix blocks use copy-on-write paged attention.
///   Reference-counted physical blocks shared across sequences.
///
/// Eviction: LRU on prefix nodes when cache memory exceeds budget.

pub struct PrefixCache {
    /// Radix tree root
    root: RadixNode,

    /// Maximum memory for cached prefixes
    max_memory: usize,

    /// Current memory usage
    current_memory: usize,
}

struct RadixNode {
    /// Token sequence for this node's edge
    tokens: Vec<u32>,

    /// KV cache block references for this prefix
    kv_blocks: Option<Vec<BlockRef>>,

    /// Children, keyed by first token of child edge
    children: HashMap<u32, RadixNode>,

    /// Last access time for LRU eviction
    last_accessed: std::time::Instant,

    /// Number of active sequences using this prefix
    ref_count: u32,
}
```

## Step 6: Server

### `server.rs`

```rust
/// HTTP server exposing the batching engine.
///
/// Endpoints:
///   POST /v1/completions      — Submit completion request (streaming SSE)
///   POST /v1/chat/completions — Chat completion (OpenAI-compatible)
///   DELETE /v1/requests/{id}  — Cancel a request
///   GET /v1/health            — Health check
///   GET /v1/metrics           — Prometheus-style metrics
///
/// Streaming: Use Server-Sent Events (SSE) for token-by-token streaming.
/// Each token is sent as it's generated, not waiting for the full response.
///
/// OpenAI compatibility: Support the same request/response format as
/// OpenAI's API so Fox Den can switch between cloud and local seamlessly.
///
/// The server is a thin layer over ContinuousBatcher — all the logic
/// lives in the batcher and scheduler.
```

## Step 7: Integration

### ronn-api
- Replace or wrap existing API with the batching server
- MCP tool calls route through the batcher with appropriate priority

### ronn-hrm
- HRM can influence scheduling: if a request triggers System 2 reasoning,
  allocate more compute budget and time slice
- HRM confidence scores could inform preemption decisions

### ronn-cache (companion feature)
- Paged attention provides the memory management layer
- Prefix cache stores shared KV blocks
- Memory budget from block manager informs scheduling decisions

### ronn-speculative (companion feature)
- Speculative decoding integrates per-sequence within the batch
- Scheduler can choose to speculate only for P0 requests
- Draft tokens don't count against the batch token budget

## Step 8: CLI Commands

```bash
# Start RONN as a batching server
ronn serve --model model.onnx --port 8080 --max-batch-size 32

# With priority scheduling
ronn serve --model model.onnx --scheduler priority --preemption auto

# With prefix caching
ronn serve --model model.onnx --prefix-cache --prefix-cache-memory 4G

# With all optimizations
ronn serve --model model.onnx \
  --scheduler priority \
  --prefix-cache \
  --kv-bits 4 \
  --speculative auto \
  --max-batch-size 64 \
  --port 8080
```

## Step 9: Testing

1. **Correctness**: Batched output must be identical to single-sequence output for the same prompts.
2. **Throughput**: Measure tokens/sec with increasing concurrent requests (1, 4, 16, 32).
3. **Latency**: P50, P95, P99 time-to-first-token and inter-token latency under load.
4. **Priority**: P0 requests maintain low latency even when P3 requests saturate the system.
5. **Preemption**: Preempted requests resume correctly with identical output.
6. **Prefix cache hit rate**: Measure with realistic Fox Den workloads.
7. **Memory**: System never exceeds configured VRAM budget under any load.

## Implementation Phases for Claude Code

**Phase 1:** Request lifecycle, FCFS scheduler, basic batching loop (no prefix cache, no priority).

**Phase 2:** Priority scheduler, preemption (swap-based first, recompute later).

**Phase 3:** Prefix caching, integration with ronn-cache paged attention.

**Phase 4:** HTTP server, streaming, OpenAI-compatible API.

**Phase 5:** Metrics, benchmarking, load testing.

## Success Criteria

- [ ] Batched inference produces identical output to single-sequence
- [ ] Throughput scales with concurrent requests (>3x at batch_size=16 vs sequential)
- [ ] P0 latency stays under 100ms TTFT even with 32 concurrent P3 requests
- [ ] Preempted requests resume with correct output
- [ ] Prefix cache eliminates redundant system prompt computation
- [ ] Memory usage stays within budget under all conditions
- [ ] OpenAI-compatible API works with standard clients
- [ ] No breaking changes to existing RONN API
