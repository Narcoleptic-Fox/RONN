# RONN: Speculative Decoding Implementation

## Context

You are working on **RONN** (Rust-Optimized Neural Networks), a Rust ML inference runtime at `https://github.com/Dieshen/RONN`. RONN is a brain-inspired neural network runtime with existing crates for core inference, graph optimization, execution providers, memory systems, learning, and hierarchical reasoning.

Your task is to create a new branch `feature/speculative-decoding` and implement lossless speculative decoding as a new crate `ronn-speculative`.

## Step 0: Setup

```bash
# Clone RONN and create feature branch
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/speculative-decoding

# Clone reference implementations (DO NOT copy code — study patterns, reimplement in Rust)
git clone https://github.com/SafeAILab/EAGLE.git ../EAGLE-reference
git clone https://github.com/sgl-project/SpecForge.git ../SpecForge-reference
```

**Key reference files to study:**
- `../EAGLE-reference/eagle/model/ea_model.py` — EAGLE draft model architecture and tree-structured generation
- `../EAGLE-reference/eagle/model/cnets.py` — The lightweight single-layer draft head
- `../EAGLE-reference/eagle/model/choices_table.py` — Static tree topologies for draft structures
- `../EAGLE-reference/eagle/ge_data/` — Training data generation pipeline
- `../SpecForge-reference/specforge/` — Production-grade EAGLE-3 training with TTT

**Key papers:**
- EAGLE-3 (NeurIPS 2025): https://arxiv.org/abs/2503.01840 — State-of-the-art, multi-level feature fusion, Training-Time Testing
- Original Speculative Sampling: https://arxiv.org/abs/2302.01318 — Core algorithm (Leviathan et al.)
- EAGLE-2 (EMNLP 2024): https://arxiv.org/abs/2406.16858 — Dynamic draft tree construction
- LayerSkip: https://arxiv.org/abs/2404.16710 — Self-speculative via early exit
- CALM: https://arxiv.org/abs/2207.07061 — Confident Adaptive Language Modeling

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
- How ronn-hrm routes between fast/slow paths — **this is critical**: speculative decoding IS System 1/2 routing. System 1 = draft model (fast, approximate), System 2 = target model (slow, verifies).

## Step 2: Create the `ronn-speculative` Crate

Create a new crate at `crates/ronn-speculative/` with the following module structure:

```
crates/ronn-speculative/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API, feature flags, re-exports
│   ├── draft/
│   │   ├── mod.rs          # DraftModel trait definition
│   │   ├── independent.rs  # Separate small model as drafter
│   │   ├── eagle.rs        # EAGLE-3 style single-layer draft head
│   │   ├── self_draft.rs   # Self-speculative via early exit (no separate model)
│   │   └── ngram.rs        # N-gram lookup from retrieval corpus (zero neural cost)
│   ├── verify/
│   │   ├── mod.rs          # Verification engine
│   │   ├── rejection.rs    # Rejection sampling — maintains EXACT output distribution
│   │   └── tree_verify.rs  # Tree-structured parallel verification
│   ├── tree/
│   │   ├── mod.rs          # Draft tree data structures
│   │   ├── static_tree.rs  # Fixed tree topologies (from EAGLE choices tables)
│   │   └── dynamic_tree.rs # EAGLE-2 style confidence-based adaptive trees
│   ├── scheduler.rs        # Adaptive speculation length and strategy selection
│   ├── pipeline.rs         # End-to-end speculative inference loop
│   ├── training.rs         # EAGLE draft head training pipeline
│   └── metrics.rs          # Acceptance rates, speedup, per-position stats
```

### Cargo.toml

```toml
[package]
name = "ronn-speculative"
version = "0.1.0"
edition = "2021"
description = "Lossless speculative decoding for RONN inference runtime"

[features]
default = ["self-draft", "ngram"]
eagle = ["dep:ronn-learning"]       # EAGLE requires training infrastructure
self-draft = []                      # Self-speculative via early exit
ngram = []                           # N-gram retrieval drafting
full = ["eagle", "self-draft", "ngram"]

[dependencies]
ronn-core = { path = "../ronn-core" }
ronn-graph = { path = "../ronn-graph" }
ronn-providers = { path = "../ronn-providers" }
ronn-hrm = { path = "../ronn-hrm" }
ronn-memory = { path = "../ronn-memory", optional = true }
ronn-learning = { path = "../ronn-learning", optional = true }
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
rand = "0.8"

[dev-dependencies]
criterion = "0.5"
approx = "0.5"        # For floating point comparison in losslessness tests
```

## Step 3: Core Data Structures and Traits

### `lib.rs` — Public API

```rust
//! # ronn-speculative
//!
//! Lossless speculative decoding for the RONN inference runtime.
//!
//! Speculative decoding accelerates autoregressive LLM inference by 2-4x
//! without changing the output distribution. A fast "draft" model proposes
//! multiple tokens; the full "target" model verifies them in one forward pass.
//!
//! ## Draft Strategies
//!
//! - **EAGLE**: Lightweight draft head reusing target model features (best quality)
//! - **Self-Draft**: Early exit from target model layers (no extra model needed)
//! - **N-gram**: Lookup-based drafting from a text corpus (zero neural cost)
//! - **Independent**: Separate smaller model as drafter (simplest to set up)
//!
//! ## Guarantees
//!
//! All strategies produce output distributions identical to standard
//! autoregressive decoding (within floating-point tolerance). This is
//! mathematically guaranteed by the rejection sampling verification step.

pub mod draft;
pub mod verify;
pub mod tree;
pub mod scheduler;
pub mod pipeline;
pub mod metrics;

#[cfg(feature = "eagle")]
pub mod training;
```

### `draft/mod.rs` — DraftModel Trait

```rust
/// Core trait that all draft strategies implement.
///
/// A draft model takes the current generation context and proposes
/// candidate next tokens, organized as a tree structure.
///
/// Implementations range from zero-cost (n-gram lookup) to trained
/// neural networks (EAGLE). The trait is designed so the verification
/// engine doesn't care which strategy produced the drafts.
pub trait DraftModel: Send + Sync {
    /// Generate a tree of draft token candidates.
    ///
    /// # Arguments
    /// * `context` - Current token sequence and KV cache state
    /// * `target_features` - Optional intermediate layer features from target model
    ///   (used by EAGLE; None for independent/ngram drafters)
    /// * `config` - Draft generation parameters (max tokens, tree width, etc.)
    ///
    /// # Returns
    /// A `DraftTree` containing candidate tokens with log-probabilities.
    fn draft(
        &mut self,
        context: &GenerationContext,
        target_features: Option<&LayerFeatures>,
        config: &DraftConfig,
    ) -> Result<DraftTree, SpeculativeError>;

    /// Memory footprint of this draft model (for scheduling decisions).
    fn memory_requirements(&self) -> MemoryFootprint;

    /// Human-readable name for metrics/logging.
    fn name(&self) -> &str;
}

/// Configuration for draft generation.
pub struct DraftConfig {
    /// Maximum number of draft tokens to generate.
    /// Longer drafts = more potential speedup but more wasted work on rejections.
    /// Start with 5, tune based on acceptance rate.
    pub max_draft_tokens: usize,

    /// Maximum tree width at each level.
    /// Width 1 = linear chain (simplest). Width 3+ = tree speculation.
    pub max_tree_width: usize,

    /// Minimum confidence to continue drafting.
    /// If draft model confidence drops below this, stop early.
    pub min_confidence: f32,

    /// Temperature for draft sampling (0.0 = greedy).
    pub temperature: f32,
}

/// Represents the generation context available to draft models.
pub struct GenerationContext {
    /// Token IDs generated so far.
    pub token_ids: Vec<u32>,

    /// KV cache state (opaque handle — draft models that need it will downcast).
    pub kv_cache: Option<Box<dyn std::any::Any + Send>>,

    /// Current sequence length.
    pub seq_len: usize,
}

/// Features extracted from intermediate layers of the target model.
/// Used by EAGLE-style drafters that reuse target model internals.
pub struct LayerFeatures {
    /// Features from selected layers, keyed by layer index.
    /// EAGLE-3 uses low/mid/high layer features.
    pub features: Vec<(usize, ronn_core::Tensor)>,
}
```

## Step 4: Implement Draft Models

### 4a. `draft/self_draft.rs` — Self-Speculative (implement first, simplest)

```rust
/// Self-speculative decoding: use the target model itself as the drafter
/// by exiting early from intermediate layers.
///
/// This is the simplest draft strategy and requires NO separate model.
///
/// Algorithm:
///   1. Run input through first `exit_layer` layers of the target model
///   2. Apply the language model head (final projection) to intermediate hidden state
///   3. Sample from the resulting distribution → these are draft tokens
///   4. Repeat autoregressively for `max_draft_tokens` steps
///   5. Then verify all drafts using the full model
///
/// The key parameter is `exit_layer`:
///   - Too early (e.g., layer 4 of 32): Fast but low acceptance rate
///   - Too late (e.g., layer 28 of 32): High acceptance but barely faster
///   - Sweet spot: ~50% of total layers (layer 16 of 32)
///
/// Integration with ronn-hrm:
///   This IS the System 1 path. HRM already routes between fast/slow.
///   Self-speculative decoding formalizes this into draft/verify.
///
/// Implementation notes:
///   - You need the ability to run a partial forward pass (layers 1..N)
///   - Check how ronn-providers handles this — you may need to add
///     a `execute_partial(start_layer, end_layer)` method
///   - The LM head (token prediction layer) must be applicable to
///     intermediate hidden states, not just final layer output
///   - Some models have layer norms before the LM head — apply those too
pub struct SelfDraftModel {
    /// Which layer to exit at for draft generation.
    exit_layer: usize,

    /// Total layers in the target model.
    total_layers: usize,

    /// Confidence estimator for adaptive exit (optional).
    /// If set, may exit even earlier for "easy" tokens.
    confidence_estimator: Option<ConfidenceEstimator>,
}

/// Confidence estimation for adaptive early exit.
/// Implement at least two methods, benchmark both:
pub enum ConfidenceEstimator {
    /// Entropy of softmax distribution at intermediate layer.
    /// Low entropy = model is confident = safe to exit.
    SoftmaxEntropy {
        threshold: f32,  // Exit if entropy < threshold (default: 0.5 nats)
    },

    /// Cosine similarity between consecutive layer outputs.
    /// High similarity = additional layers aren't changing the representation.
    LayerConvergence {
        threshold: f32,  // Exit if cosine_sim > threshold (default: 0.99)
    },
}
```

### 4b. `draft/ngram.rs` — N-gram Lookup (implement second, zero neural cost)

```rust
/// N-gram drafting: propose tokens by matching recent output against a text corpus.
///
/// Zero neural computation cost. Works when the model is likely to output
/// text that exists verbatim in the corpus:
///   - Code generation with open files as context
///   - RAG responses quoting from retrieved documents
///   - Boilerplate templates, common patterns
///   - Agent loops repeating similar structures
///
/// Algorithm:
///   1. Build a suffix array or trie over the tokenized corpus at startup
///   2. During generation, take the last N tokens (N=3-8) as a query
///   3. Search for this N-gram in the corpus
///   4. If found: the tokens following it in the corpus become draft tokens
///   5. If multiple matches: use the most frequent continuation, or
///      build a draft tree with multiple branches
///   6. Verify with target model as usual
///
/// Example:
///   Corpus contains: "fn main() { println!(\"Hello, world!\"); }"
///   Model just generated: ["fn", "main", "(", ")"]
///   N-gram match! Draft: ["{", "println", "!", "(", "\"Hello"]
///   → Target model verifies. If coding, likely accepts most of these.
///
/// Integration with ronn-memory:
///   The corpus can be sourced from ronn-memory's semantic tier.
///   This means your long-term knowledge becomes a free acceleration source.
///
/// Data structures:
///   - For small corpora (<10MB tokens): Trie with frequency counts
///   - For large corpora (>10MB tokens): Suffix array with LCP array
///   - Both support O(log n) or O(n_query) lookup
pub struct NgramDraftModel {
    /// The suffix data structure over the tokenized corpus.
    index: NgramIndex,

    /// How many recent tokens to use as the lookup key.
    ngram_size: usize,

    /// Minimum number of matches required to use a continuation.
    min_match_count: usize,
}

pub enum NgramIndex {
    /// Trie-based index. Good for corpora under ~10M tokens.
    Trie(TokenTrie),

    /// Suffix array index. Scales to very large corpora.
    SuffixArray(TokenSuffixArray),
}

// Implement both TokenTrie and TokenSuffixArray.
// The trie is simpler to implement first and sufficient for initial testing.
// Suffix array is more memory-efficient for production use.
```

### 4c. `draft/eagle.rs` — EAGLE-3 (implement third, highest quality)

```rust
/// EAGLE-3 draft model: a single transformer layer that reuses features
/// from the target model's intermediate layers.
///
/// This is state-of-the-art speculative decoding (NeurIPS 2025).
/// It achieves the highest acceptance rates but requires:
///   1. Training a draft head specific to each target model
///   2. Extracting intermediate features during target model forward pass
///
/// Architecture (study ../EAGLE-reference/eagle/model/cnets.py):
///
///   1. During target model inference, extract hidden states from:
///      - Low layer (e.g., layer 8 of 80)
///      - Mid layer (e.g., layer 40 of 80)
///      - High layer (e.g., layer 72 of 80)
///   2. Fuse these multi-level features via learned weighted combination
///   3. Concatenate fused features with token embedding
///   4. Feed into a single transformer decoder layer (~0.5-1B params)
///   5. This layer autoregressively generates draft tokens
///   6. Draft tokens form a TREE structure (not just linear chain)
///
/// EAGLE-3 key innovation over EAGLE-1/2:
///   Training-Time Testing (TTT) — during training, simulate multi-step
///   autoregressive generation so the draft head learns to handle
///   accumulated errors from its own previous predictions.
///   This makes it much more robust during actual inference.
///
/// Training cost: Surprisingly cheap.
///   - For a 70B target: ~1B parameter draft head
///   - Training data: ~70K dialogues from ShareGPT
///   - Time: 1-2 days on 4x A100 (40GB)
///   - Your S1 Max can train draft heads for models up to ~30B
///
/// Implementation steps:
///   1. Define feature extraction hooks (tap into target model layers)
///   2. Implement feature fusion module (learned weighted combination)
///   3. Build the single-layer autoregressive draft head
///   4. Implement tree-structured generation (multiple branches per step)
///   5. Build training pipeline (generate training data from target runs)
///
/// Study ../EAGLE-reference/eagle/model/ea_model.py for the full flow.
/// Study ../SpecForge-reference/ for production training pipeline.
pub struct EagleDraftModel {
    /// The lightweight draft head (single transformer layer + projection)
    draft_head: Box<dyn ronn_core::Model>,

    /// Which target model layers to extract features from.
    /// EAGLE-3 uses low/mid/high. Indices are 0-based.
    feature_layers: Vec<usize>,

    /// Learned weights for fusing multi-level features.
    fusion_weights: ronn_core::Tensor,

    /// Tree topology for draft generation.
    tree_config: TreeConfig,
}
```

### 4d. `draft/independent.rs` — Independent Draft Model (simplest to set up)

```rust
/// Independent draft model: a completely separate, smaller model.
///
/// Example: TinyLlama-1.1B drafting for Llama-70B.
///
/// Pros: Simple, no target model modification needed.
/// Cons: Lower acceptance rate than EAGLE, extra memory for draft model.
///
/// This is the baseline implementation. If you can get this working,
/// the other draft strategies are refinements on this pattern.
///
/// The draft model must share the SAME tokenizer as the target model.
pub struct IndependentDraftModel {
    /// The small draft model.
    model: Box<dyn ronn_core::Model>,

    /// Draft model name for metrics.
    name: String,
}
```

## Step 5: Implement Verification

### `verify/rejection.rs` — This is the MOST CRITICAL component

```rust
/// Rejection sampling verification.
///
/// THIS MUST BE MATHEMATICALLY CORRECT. The entire guarantee of speculative
/// decoding — that output distribution is identical to standard decoding —
/// depends on this implementation being right.
///
/// Algorithm (Leviathan et al. 2023, Section 2):
///
/// Given:
///   - Draft tokens: t_1, t_2, ..., t_K
///   - Draft model probabilities: q(t_i | context + t_1..t_{i-1})
///   - Target model probabilities: p(t_i | context + t_1..t_{i-1})
///     (computed via ONE forward pass of the target model on all K tokens)
///
/// For each draft token t_i in order:
///   1. Sample r ~ Uniform(0, 1)
///   2. If r < min(1, p(t_i) / q(t_i)):
///      → ACCEPT t_i, continue to t_{i+1}
///   3. Else:
///      → REJECT t_i
///      → Sample correction token from: norm(max(0, p - q))
///         (the residual distribution)
///      → STOP — discard all remaining draft tokens
///
/// Special case — greedy decoding (temperature = 0):
///   Just check: argmax(p) == t_i ? Accept : Reject.
///   No sampling needed. Deterministic.
///
/// Special case — tree verification:
///   Multiple draft branches are verified simultaneously.
///   Accept the longest branch that passes rejection sampling.
///   This is more complex — see tree_verify.rs.
///
/// CRITICAL TESTS:
///   1. Generate N=10000 tokens with speculative decoding
///   2. Generate N=10000 tokens with standard decoding
///   3. Token frequency distributions must match (chi-squared test, p > 0.05)
///   4. For greedy decoding: outputs must be BIT-FOR-BIT identical

pub struct RejectionSampler {
    /// Random number generator for acceptance decisions.
    rng: rand::rngs::StdRng,
}

impl RejectionSampler {
    /// Verify a sequence of draft tokens against target model probabilities.
    ///
    /// Returns the number of accepted tokens and optionally a correction token.
    pub fn verify(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[Vec<f32>],     // q(vocab) at each draft position
        target_probs: &[Vec<f32>],    // p(vocab) at each draft position
        temperature: f32,
    ) -> VerificationResult {
        // Implement the rejection sampling algorithm described above.
        // Be extremely careful with numerical precision.
        // Use f64 internally even if inputs are f32.
        todo!()
    }
}

pub struct VerificationResult {
    /// Number of draft tokens accepted (0 to K).
    pub accepted_count: usize,

    /// The correction token (sampled from residual distribution).
    /// This is the "bonus" token — even when all drafts are rejected,
    /// we always get at least one token from the verification step.
    pub correction_token: Option<u32>,

    /// Per-position acceptance (for metrics).
    pub per_position_accepted: Vec<bool>,
}
```

### `verify/tree_verify.rs` — Tree-structured verification

```rust
/// Tree verification extends rejection sampling to branching draft structures.
///
/// Instead of a linear chain [t1, t2, t3, t4], the draft is a tree:
///       t1
///      / | \
///    t2a t2b t2c
///    /     \
///  t3a    t3b
///
/// This hedges bets: even if the first branch is rejected,
/// alternative branches may succeed.
///
/// The target model verifies ALL branches in ONE forward pass using
/// tree attention (a specialized attention mask).
///
/// Implementation:
///   1. Flatten the tree into a sequence with a special attention mask
///   2. Run target model forward pass with this mask
///   3. Apply rejection sampling along each path from root to leaf
///   4. Accept the longest valid path
///
/// The attention mask ensures each node only attends to its ancestors,
/// not to sibling branches. This preserves causal semantics.
///
/// Reference: Study EAGLE-reference/eagle/model/ea_model.py for
/// how tree attention masks are constructed.
```

## Step 6: Implement the Pipeline

### `pipeline.rs` — End-to-end speculative inference loop

```rust
/// The main speculative decoding loop.
///
/// ```text
/// ┌─────────────────────────────────────────────────────┐
/// │                  SPECULATIVE LOOP                    │
/// │                                                     │
/// │  1. Draft model proposes K tokens (tree)            │
/// │  2. Target model verifies ALL K in ONE pass         │
/// │  3. Accept longest valid prefix via rejection       │
/// │  4. Append accepted tokens + correction to output   │
/// │  5. Update KV cache (keep accepted, discard rest)   │
/// │  6. Scheduler adjusts K for next iteration          │
/// │  7. Repeat until EOS or max_tokens                  │
/// └─────────────────────────────────────────────────────┘
/// ```
///
/// The pipeline orchestrates draft model, target model, verifier,
/// and scheduler into a coherent loop.

pub struct SpeculativePipeline {
    /// The target (large) model.
    target_model: Box<dyn ronn_core::Model>,

    /// The draft strategy.
    draft_model: Box<dyn DraftModel>,

    /// Rejection sampling verifier.
    verifier: RejectionSampler,

    /// Adaptive scheduling (draft length, strategy selection).
    scheduler: SpeculativeScheduler,

    /// Metrics collector.
    metrics: SpeculativeMetrics,
}

impl SpeculativePipeline {
    /// Generate tokens using speculative decoding.
    ///
    /// This is the primary public API. It returns a stream of tokens
    /// for compatibility with streaming inference.
    pub async fn generate(
        &mut self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<TokenStream, SpeculativeError> {
        // Main loop:
        // 1. If first iteration, run prefill on target model
        // 2. Extract target features if using EAGLE
        // 3. Draft K tokens using draft model
        // 4. Run target model on all K tokens (single forward pass)
        // 5. Verify via rejection sampling
        // 6. Accept tokens, update KV cache
        // 7. Update scheduler with acceptance stats
        // 8. Yield accepted tokens to stream
        // 9. If EOS or max_tokens: break
        todo!()
    }
}
```

### `scheduler.rs` — Adaptive Speculation

```rust
/// The scheduler dynamically adjusts speculation strategy based on
/// observed acceptance rates.
///
/// Parameters it controls:
///   1. Draft length K: More speculation vs less wasted work
///   2. Tree width: Wider trees hedge more but cost more to verify
///   3. Draft strategy: May switch between EAGLE/ngram/self-draft
///   4. Whether to speculate at all (fall back to standard decoding)
///
/// Decision logic:
///   - Track rolling acceptance rate over last N iterations
///   - If acceptance_rate > 0.8: increase K (model is predictable)
///   - If acceptance_rate < 0.3: decrease K or disable speculation
///   - If ngram hit rate is high: prefer ngram over neural draft
///   - If memory pressure: switch from EAGLE to self-draft
///
/// Integration with ronn-hrm:
///   The scheduler reports to HRM as a System 1/2 routing decision.
///   HRM can override the scheduler based on task-level context.

pub struct SpeculativeScheduler {
    /// Rolling window of acceptance rates.
    acceptance_history: VecDeque<f32>,

    /// Current draft length.
    current_k: usize,

    /// Minimum/maximum bounds for K.
    k_bounds: (usize, usize),

    /// Whether speculation is currently active.
    active: bool,

    /// Threshold below which to disable speculation.
    min_acceptance_rate: f32,
}
```

## Step 7: Integration with Existing RONN Crates

### ronn-hrm Integration

Add to `ronn-hrm`:
- `SpeculativePolicy`: A policy that HRM uses to decide draft/verify routing
- Map System 1 → draft model execution, System 2 → target model verification
- HRM tracks acceptance rates as part of its confidence modeling

### ronn-graph Integration

Add `SpeculativeDecodingPass` to `ronn-graph`:
- This optimization pass wraps autoregressive generation with the speculative pipeline
- Detects autoregressive patterns in the graph and replaces with draft-verify patterns
- Configurable: which draft strategy, initial K, etc.

### ronn-providers Integration

Extend the provider trait:
- `execute_partial(start_layer, end_layer)` for self-speculative decoding
- Feature extraction hooks for EAGLE (tap intermediate layer outputs)
- Draft model can run on a different provider than target model

### ronn-memory Integration

- N-gram corpus sourced from semantic memory tier
- KV cache for rejected tokens must be properly invalidated
- Working memory caches draft model states between speculation rounds

## Step 8: CLI Commands

```bash
# Run with auto-selected speculative decoding
ronn infer --model model.onnx --speculative auto --prompt "Hello"

# Run with specific draft model (independent)
ronn infer --model model.onnx --speculative independent --draft-model tiny.onnx --prompt "Hello"

# Run with EAGLE draft head
ronn infer --model model.onnx --speculative eagle --eagle-head eagle.onnx --prompt "Hello"

# Run with self-speculative (early exit at layer 16)
ronn infer --model model.onnx --speculative self --exit-layer 16 --prompt "Hello"

# Run with n-gram drafting from a corpus
ronn infer --model model.onnx --speculative ngram --corpus ./docs --prompt "Hello"

# Train an EAGLE draft head for a target model
ronn train-eagle --target model.onnx --data training.jsonl --output eagle.onnx --epochs 3

# Benchmark speculative vs standard decoding
ronn benchmark-spec --model model.onnx --draft-model tiny.onnx --prompts bench.jsonl

# Show detailed speculation metrics
ronn benchmark-spec --model model.onnx --speculative self --prompts bench.jsonl --report detailed
```

## Step 9: Testing Strategy

### Critical Tests (must pass before any optimization work)

1. **Losslessness (greedy)**: Speculative output MUST be bit-for-bit identical to standard greedy decoding. Run 100 prompts, compare every token.

2. **Losslessness (sampling)**: Generate 10K tokens with and without speculation at temperature=0.7. Chi-squared test on token distributions must pass (p > 0.05).

3. **Rejection sampler unit tests**: Feed known draft/target probability pairs, verify acceptance decisions match the algorithm exactly.

4. **Tree attention correctness**: Tree-structured drafts must produce same results as sequentially verifying each branch.

### Performance Tests

5. **Speedup measurement**: Wall-clock tokens/sec with and without speculation across code, prose, math prompts.

6. **Acceptance rate tracking**: Per-position acceptance rates for each draft strategy.

7. **Scheduler adaptation**: Verify scheduler increases K when acceptance is high, decreases when low.

8. **Memory overhead**: Measure additional memory from draft model, draft KV cache, tree structures.

## Implementation Phases for Claude Code

**Phase 1 (Steps 0-3):** Setup, read existing code, create crate skeleton, define all traits and data structures. No actual inference logic yet.

**Phase 2 (Steps 4a-4b):** Implement self-draft and n-gram drafters. These are the simplest and don't require training. Get the draft → verify → accept loop working end-to-end with these.

**Phase 3 (Steps 5-6):** Implement rejection sampling verifier and the pipeline. This is where correctness matters most. Write exhaustive tests. Tree verification can wait — start with linear chain verification.

**Phase 4 (Steps 4c-4d, 7-8):** Implement EAGLE draft model, independent drafter, full integration with HRM/graph/providers, CLI commands. Add tree verification.

**Phase 5 (Step 9):** Comprehensive testing, benchmarking, scheduler tuning.

## Success Criteria

- [ ] Greedy speculative output is BIT-FOR-BIT identical to standard greedy
- [ ] Sampling speculative output passes distribution equality test
- [ ] Self-draft achieves measurable speedup (target: 1.3-2x)
- [ ] N-gram draft shows high acceptance on code/RAG workloads
- [ ] Pipeline handles EOS, max_tokens, and error cases gracefully
- [ ] Metrics track acceptance rates, speedup, and per-position stats
- [ ] All features behind cargo feature flags
- [ ] Clean integration with ronn-hrm's System 1/2 framework
- [ ] No breaking changes to existing RONN API