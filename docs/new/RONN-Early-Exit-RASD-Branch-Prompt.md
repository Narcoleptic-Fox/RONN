# RONN: Early Exit + Retrieval-Augmented Speculative Decoding

## Context

You are working on **RONN** (Rust-Optimized Neural Networks), a Rust ML inference runtime at `https://github.com/Dieshen/RONN`. This prompt covers two lighter-weight features that extend existing crates rather than creating entirely new ones.

**Feature A: Early Exit Inference** — Extend `ronn-hrm` so tokens can exit at intermediate layers when the model is already confident. Directly leverages the existing System 1/2 routing.

**Feature B: Retrieval-Augmented Speculative Decoding (RASD)** — Extend `ronn-speculative` (from the speculative decoding branch) so known text corpora can provide zero-cost draft tokens.

Both features are lower effort individually, but compound powerfully with the other optimization features.

---

# FEATURE A: Early Exit Inference
**Branch:** `feature/early-exit`
**Extends:** `ronn-hrm` (existing crate, no new crate needed)
**Expected speedup:** 1.5-2.5x

## Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/early-exit
```

**Key references:**
- SpecEE: https://dl.acm.org/doi/10.1145/3695053.3730996 — 2.25-2.43x speedup via speculative early exit
- LayerSkip: https://arxiv.org/abs/2404.16710 — Self-speculative via early exit
- CALM: https://arxiv.org/abs/2207.07061 — Confident Adaptive Language Modeling (the original)

## Step 1: Understand Existing ronn-hrm

Read `crates/ronn-hrm/src/` thoroughly. The HRM already implements:
- System 1 (fast, intuitive) path
- System 2 (slow, deliberate) path
- A routing mechanism that decides which path to take

Early exit IS System 1/2 applied at the layer level:
- System 1 = first N layers sufficient → exit early
- System 2 = full depth needed → run all layers

## Step 2: Add Early Exit Modules to ronn-hrm

```
crates/ronn-hrm/src/
├── ... (existing files, do not modify)
├── early_exit/
│   ├── mod.rs              # Early exit controller
│   ├── confidence.rs       # Confidence estimation methods
│   ├── calibration.rs      # Threshold calibration
│   └── statistics.rs       # Per-layer exit rate tracking
```

### `early_exit/confidence.rs` — Implement all three, benchmark

```rust
/// Confidence estimation decides whether an intermediate layer's
/// hidden state is "good enough" to produce the output token.
///
/// If confidence is high → exit early (skip remaining layers).
/// If confidence is low → continue to the next layer.
///
/// Three estimation methods (implement all, let users choose):

/// Method 1: Softmax Entropy
/// Apply the LM head to the intermediate hidden state, compute
/// entropy of the resulting probability distribution.
/// Low entropy = peaked distribution = model is confident.
///
/// Advantages: Direct measure of prediction uncertainty.
/// Cost: One extra LM head application per check (~cheap, LM head is just a matmul).
///
/// Threshold: Exit if entropy < threshold (default 0.5 nats).
/// Calibrate on representative data.
pub struct EntropyConfidence {
    /// Exit if entropy falls below this value
    pub threshold: f32,
}

/// Method 2: Layer Convergence (Cosine Similarity)
/// Compare hidden state at layer L with hidden state at layer L-1.
/// If they're very similar, additional layers aren't changing the representation,
/// so further computation is wasted.
///
/// Advantages: No need for LM head — just compare two vectors.
/// Cost: One cosine similarity per check (very cheap).
///
/// Threshold: Exit if cosine_similarity > threshold (default 0.99).
pub struct ConvergenceConfidence {
    /// Exit if similarity exceeds this value
    pub threshold: f32,
}

/// Method 3: Trained Classifier
/// Small MLP at each candidate exit layer:
///   hidden_state → [linear → relu → linear → sigmoid] → exit probability
///
/// Train on calibration data where ground truth is "would the output token
/// change if we ran the remaining layers?"
///
/// Advantages: Most accurate, learns model-specific patterns.
/// Cost: One small MLP per check (~trivial).
/// Downside: Requires training per model.
pub struct TrainedConfidence {
    /// Trained MLP weights for each candidate exit layer
    pub classifiers: Vec<ExitClassifier>,
}

/// A small binary classifier for one candidate exit layer.
pub struct ExitClassifier {
    pub layer_index: usize,
    pub weights_1: ronn_core::Tensor,  // [hidden_dim, classifier_dim]
    pub bias_1: ronn_core::Tensor,
    pub weights_2: ronn_core::Tensor,  // [classifier_dim, 1]
    pub bias_2: ronn_core::Tensor,
}

/// Unified confidence estimator that wraps all methods.
pub enum ConfidenceEstimator {
    Entropy(EntropyConfidence),
    Convergence(ConvergenceConfidence),
    Trained(TrainedConfidence),
}

impl ConfidenceEstimator {
    /// Should we exit at this layer?
    pub fn should_exit(&self, hidden_state: &ronn_core::Tensor, prev_hidden_state: Option<&ronn_core::Tensor>) -> bool {
        match self {
            Self::Entropy(e) => {
                // 1. Apply LM head to hidden_state → logits
                // 2. Softmax → probabilities
                // 3. Compute entropy: H = -sum(p * log(p))
                // 4. Return H < e.threshold
                todo!()
            }
            Self::Convergence(c) => {
                // 1. Compute cosine similarity between hidden_state and prev_hidden_state
                // 2. Return similarity > c.threshold
                // Requires prev_hidden_state to be Some
                todo!()
            }
            Self::Trained(t) => {
                // 1. Find classifier for current layer
                // 2. Forward pass through MLP
                // 3. Return sigmoid(output) > 0.5
                todo!()
            }
        }
    }
}
```

### `early_exit/mod.rs` — The controller

```rust
/// Early exit controller that wraps model inference.
///
/// During a forward pass:
///   1. Process layers 1, 2, 3, ... checking confidence at intervals
///   2. When confidence exceeds threshold → apply LM head → return token
///   3. If no layer is confident enough → run all layers (standard path)
///
/// Check frequency: Not every layer (too expensive). Check every K-th layer.
///   - K=4 for 32-layer models (check at layers 8, 12, 16, 20, 24, 28)
///   - K=8 for 80-layer models (check at layers 16, 24, 32, 40, 48, 56, 64, 72)
///   - First check should be at ~25% depth (too early is always wrong)
///
/// Integration with ronn-hrm:
///   The early exit controller registers as a System 1 execution path.
///   HRM can override: "this request is complex, don't early-exit"
///   or "this request is latency-critical, be aggressive about exiting."

pub struct EarlyExitController {
    /// Confidence estimation method
    pub estimator: ConfidenceEstimator,

    /// Which layers to check (indices into model's layer stack)
    pub candidate_layers: Vec<usize>,

    /// Total layers in the model
    pub total_layers: usize,

    /// Statistics tracker
    pub stats: EarlyExitStatistics,
}

impl EarlyExitController {
    /// Run inference with early exit.
    ///
    /// This replaces the standard full forward pass.
    /// Returns the layer index where exit occurred (total_layers if no early exit).
    pub async fn forward_with_early_exit(
        &mut self,
        input: &ronn_core::Tensor,
        model: &dyn ronn_core::Model,
    ) -> Result<(ronn_core::Tensor, usize), EarlyExitError> {
        // 1. Run layers up to first candidate
        // 2. At each candidate: check confidence
        // 3. If confident: apply LM head, record exit layer, return
        // 4. If not: continue to next candidate
        // 5. If no exit: run remaining layers, return (standard path)
        // 6. Update statistics
        todo!()
    }
}
```

### `early_exit/calibration.rs`

```rust
/// Calibrate early exit thresholds on representative data.
///
/// Process:
///   1. Run calibration prompts through the FULL model (all layers)
///   2. At each candidate exit layer, record:
///      a. What token the early exit would predict
///      b. What token the full model predicts
///      c. The confidence score at that layer
///   3. Find the threshold that maximizes early exits while keeping
///      accuracy above target (default: 99% match with full model)
///
/// Output: A calibration file with per-layer thresholds.
///
/// The thresholds may differ per layer:
///   - Shallow layers need stricter thresholds (less reliable)
///   - Deep layers can use looser thresholds (already close to final answer)

pub struct ExitCalibration {
    /// Per-layer optimal thresholds
    pub layer_thresholds: Vec<f32>,

    /// Accuracy at each layer with the calibrated threshold
    pub layer_accuracy: Vec<f32>,

    /// Expected exit distribution (what fraction of tokens exit at each layer)
    pub exit_distribution: Vec<f32>,
}

pub async fn calibrate(
    model: &dyn ronn_core::Model,
    data: &[Vec<u32>],           // Calibration prompts (tokenized)
    candidate_layers: &[usize],
    target_accuracy: f32,         // Default: 0.99
) -> Result<ExitCalibration, CalibrationError> {
    // 1. For each prompt, run full forward pass, record all layer states
    // 2. At each candidate layer, compute what token would be predicted
    // 3. Compare to actual final token
    // 4. Binary search for threshold that achieves target_accuracy
    todo!()
}
```

### `early_exit/statistics.rs`

```rust
/// Track early exit behavior for monitoring and tuning.
///
/// Key metrics:
///   - Exit rate per layer: What fraction of tokens exit at each candidate layer?
///   - Average exit depth: Mean layer index across all tokens
///   - Speedup: Theoretical (layers_saved/total_layers) and actual wall-clock
///   - Accuracy: Rolling comparison with full-model output (sampling a subset)

pub struct EarlyExitStatistics {
    /// Count of exits at each candidate layer
    pub exit_counts: Vec<u64>,

    /// Total tokens processed
    pub total_tokens: u64,

    /// Running sum of exit layer indices (for mean calculation)
    pub layer_sum: u64,

    /// Wall-clock time saved vs full model (estimated)
    pub estimated_time_saved: std::time::Duration,
}
```

## Step 3: Integration with ronn-providers

The provider trait needs a new method:

```rust
/// Extend the execution provider trait to support partial forward passes.
///
/// Standard: execute(input) → runs ALL layers → output
/// New: execute_layers(input, start_layer, end_layer) → runs subset → output
///
/// This is required for:
///   - Early exit: run layers 0..N, check, maybe run N..M
///   - Self-speculative decoding: run layers 0..K for draft, K..total for verify
///
/// Implementation note: if the model is loaded as a single ONNX graph,
/// you'll need to identify layer boundaries and insert breakpoints.
/// Alternatively, decompose the model into per-layer subgraphs at load time.

pub trait ExecutionProvider {
    // ... existing methods ...

    /// Execute a subset of model layers.
    fn execute_layers(
        &self,
        input: &Tensor,
        start_layer: usize,
        end_layer: usize,  // exclusive
    ) -> Result<Tensor, ProviderError>;
}
```

## Step 4: CLI Commands

```bash
# Infer with early exit (auto-calibrated thresholds)
ronn infer --model model.onnx --early-exit --prompt "Hello"

# Specify minimum layer depth and confidence method
ronn infer --model model.onnx --early-exit --min-exit-layer 16 --confidence entropy --threshold 0.5

# Calibrate thresholds on representative data
ronn calibrate-exit --model model.onnx --data calibration.jsonl --target-accuracy 0.99 --output exit-config.json

# Load calibrated thresholds
ronn infer --model model.onnx --early-exit --exit-config exit-config.json --prompt "Hello"

# Benchmark early exit vs full model
ronn benchmark-exit --model model.onnx --prompts bench.jsonl --report exit-report.json
```

## Step 5: Testing

1. **Quality**: Early-exit output matches full-model output ≥99% of tokens at calibrated thresholds.
2. **Speedup**: Measurable wall-clock improvement (target: 1.3-2x).
3. **Adaptive behavior**: Easy tokens (articles, prepositions) exit early; hard tokens (rare words, reasoning) use full depth.
4. **Statistics accuracy**: Reported exit rates match actual behavior.
5. **Threshold stability**: Calibrated thresholds work across different prompts (not overfit to calibration data).

## Implementation Phases

**Phase 1:** Confidence estimators (entropy and convergence). No actual early exit yet — just measure what WOULD happen.

**Phase 2:** EarlyExitController with `execute_layers` in providers. Get actual early exit working.

**Phase 3:** Calibration pipeline. Generate and load calibration configs.

**Phase 4:** Statistics, CLI, benchmarking.

---

# FEATURE B: Retrieval-Augmented Speculative Decoding (RASD)
**Branch:** `feature/rasd`
**Extends:** `ronn-speculative` (from speculative decoding branch) + `ronn-memory`
**Expected speedup:** Variable (0% for creative writing, 50-90% for code/RAG)

## Setup

```bash
git clone https://github.com/Dieshen/RONN.git
cd RONN
git checkout -b feature/rasd

# Ensure the speculative decoding branch has been merged or cherry-picked
# RASD extends ronn-speculative's DraftModel trait and ngram.rs
```

**Key references:**
- RASD: https://arxiv.org/abs/2503.03434 — Retrieval-Augmented Speculative Decoding
- REST: https://arxiv.org/abs/2311.08252 — Retrieval-Based Speculative Decoding
- Suffix Decoding in vLLM: pattern-matching against previous generations

## Core Idea

When the model is about to output text that exists in a known corpus, skip neural computation and look it up. This is zero-cost drafting for predictable text.

**Where this shines in Fox Den:**
- Generating code that mirrors patterns in the project's existing codebase
- RAG responses that quote from retrieved documents
- Agent loops that repeat similar reasoning structures
- Boilerplate (imports, class definitions, CRISP patterns)
- Any response that echoes the system prompt or injected context

## Step 1: Implement the Retrieval Index

### New file: `crates/ronn-speculative/src/draft/retrieval.rs`

```rust
/// The retrieval index that powers RASD.
///
/// Two-level lookup:
///
/// Level 1 — Exact N-gram Match (O(log n), zero neural cost):
///   Maintain a suffix array over the tokenized corpus.
///   Match the last N generated tokens against the corpus.
///   If found: propose the continuation as draft tokens.
///
/// Level 2 — Semantic Match (fallback, more expensive):
///   Embed the current context using the model's own embeddings.
///   Find nearest neighbor in a vector index over corpus chunks.
///   If close enough: propose that chunk's continuation.
///   This catches paraphrased matches that exact n-gram would miss.
///
/// Level 1 should handle 70-90% of hits for code/RAG workloads.
/// Level 2 is optional and only needed for semantic deduplication.

/// The main index structure.
pub struct RetrievalIndex {
    /// Tokenized corpus for exact matching
    exact_index: NgramIndex,

    /// Optional: embedding-based semantic index
    semantic_index: Option<SemanticIndex>,

    /// Statistics for monitoring
    stats: RetrievalStats,
}

/// Suffix array for exact n-gram matching over token sequences.
///
/// Construction:
///   1. Concatenate all corpus documents into one token sequence
///      with separator tokens between documents
///   2. Build suffix array using SA-IS algorithm (O(n) construction)
///   3. Build LCP (Longest Common Prefix) array for fast lookup
///
/// Query:
///   1. Binary search in suffix array for the query n-gram
///   2. If found: extract continuation tokens from the corpus
///   3. If multiple matches: return the most frequent continuation,
///      or build a draft tree with multiple branches
///
/// Memory: ~8 bytes per corpus token (suffix array index + LCP value)
///   10M token corpus ≈ 80MB index

pub struct NgramIndex {
    /// The full tokenized corpus (concatenated documents)
    corpus_tokens: Vec<u32>,

    /// Suffix array (indices into corpus_tokens, sorted by suffix)
    suffix_array: Vec<u32>,

    /// LCP array for efficient range queries
    lcp_array: Vec<u32>,

    /// Document boundaries (to avoid matching across documents)
    doc_boundaries: Vec<usize>,

    /// N-gram size for queries
    ngram_size: usize,
}

impl NgramIndex {
    /// Build index from a collection of tokenized documents.
    pub fn build(documents: &[Vec<u32>], ngram_size: usize) -> Self {
        // 1. Concatenate documents with separator tokens
        // 2. Build suffix array (use SA-IS for O(n))
        //    - SA-IS is complex. For v1, use the simple O(n log n) approach:
        //      sort all suffixes using their first 2^k characters iteratively
        // 3. Build LCP array
        // 4. Record document boundaries
        todo!()
    }

    /// Search for n-gram and return continuation tokens.
    pub fn search(&self, query: &[u32], max_continuation: usize) -> Vec<NgramMatch> {
        // 1. Binary search suffix array for query prefix
        // 2. Find the range of matching suffixes
        // 3. For each match: extract the next `max_continuation` tokens
        // 4. Skip matches that cross document boundaries
        // 5. Rank by frequency (how many times does this continuation appear?)
        // 6. Return top matches
        todo!()
    }
}

pub struct NgramMatch {
    /// Continuation tokens (what comes after the n-gram in the corpus)
    pub tokens: Vec<u32>,

    /// How many times this exact continuation appeared
    pub frequency: u32,

    /// Source document index (for attribution/debugging)
    pub doc_index: usize,

    /// Position in source document
    pub doc_offset: usize,
}
```

### Implement the DraftModel trait for RASD

```rust
/// RASD as a DraftModel — plugs into the speculative decoding pipeline.
///
/// The speculative pipeline doesn't care how draft tokens are produced.
/// RASD just provides them via corpus lookup instead of neural generation.

impl DraftModel for RetrievalDraftModel {
    fn draft(
        &mut self,
        context: &GenerationContext,
        _target_features: Option<&LayerFeatures>,  // Not needed for retrieval
        config: &DraftConfig,
    ) -> Result<DraftTree, SpeculativeError> {
        // 1. Extract last N tokens from context as query
        let query = &context.token_ids[context.token_ids.len().saturating_sub(self.index.ngram_size)..];

        // 2. Search index
        let matches = self.index.search(query, config.max_draft_tokens);

        if matches.is_empty() {
            // No match found — return empty tree (pipeline falls back to neural draft)
            return Ok(DraftTree::empty());
        }

        // 3. Build draft tree from matches
        // If single match: linear chain
        // If multiple matches: tree with branches
        let tree = if matches.len() == 1 {
            DraftTree::linear(&matches[0].tokens, &self.estimate_probs(&matches[0]))
        } else {
            DraftTree::branching(
                matches.iter()
                    .take(config.max_tree_width)
                    .map(|m| (&m.tokens[..], self.estimate_probs(m)))
                    .collect()
            )
        };

        self.stats.record_hit(matches[0].tokens.len());
        Ok(tree)
    }

    fn memory_requirements(&self) -> MemoryFootprint {
        MemoryFootprint {
            gpu_bytes: 0,  // Retrieval is CPU-only
            cpu_bytes: self.index.memory_usage(),
        }
    }

    fn name(&self) -> &str { "rasd" }
}

pub struct RetrievalDraftModel {
    index: RetrievalIndex,
    stats: RetrievalStats,
}

impl RetrievalDraftModel {
    /// Estimate draft probabilities for retrieved tokens.
    /// Since we don't have a neural model, estimate based on frequency.
    /// These don't need to be accurate — the target model verifies anyway.
    /// Conservative estimates (lower probs) just mean more rejections.
    fn estimate_probs(&self, m: &NgramMatch) -> Vec<f32> {
        // Simple heuristic: probability proportional to frequency
        // More sophisticated: use corpus-level token frequency
        let base_prob = (m.frequency as f32).ln().max(0.1) / 10.0;
        vec![base_prob.min(0.9); m.tokens.len()]
    }
}
```

## Step 2: Integration with ronn-memory

```rust
/// The retrieval corpus can be sourced from ronn-memory's semantic tier.
///
/// ronn-memory already stores long-term knowledge with embeddings.
/// RASD plugs into this by building its n-gram index over the
/// tokenized contents of the semantic store.
///
/// Workflow:
///   1. FoxStash document added → stored in ronn-memory semantic tier
///   2. RASD index builder tokenizes the document
///   3. Tokens appended to corpus, suffix array rebuilt (or incrementally updated)
///   4. During inference, RASD searches this index for draft tokens
///
/// Sleep consolidation integration:
///   During idle periods, ronn-memory's sleep cycle can:
///   1. Analyze which corpus entries had high hit rates → promote in index
///   2. Rebuild suffix array for better locality
///   3. Evict low-hit entries to save memory

pub struct MemoryIntegratedRasd {
    /// Core retrieval index
    index: RetrievalIndex,

    /// Reference to ronn-memory for dynamic corpus updates
    memory_ref: Arc<ronn_memory::MemoryManager>,

    /// Track which memory entries are indexed
    indexed_entries: HashSet<String>,
}

impl MemoryIntegratedRasd {
    /// Rebuild index from current memory contents.
    pub async fn sync_with_memory(&mut self) -> Result<(), RasdError> {
        // 1. Query ronn-memory semantic tier for all stored documents
        // 2. Tokenize any new/changed documents
        // 3. Rebuild or incrementally update the n-gram index
        // 4. Update indexed_entries set
        todo!()
    }
}
```

## Step 3: Hybrid Draft Strategy

```rust
/// The most powerful setup: RASD + neural draft as a fallback.
///
/// 1. Try RASD first (O(log n), zero neural cost)
/// 2. If no match: fall back to EAGLE or self-draft (neural)
/// 3. The scheduler tracks hit rates and adjusts strategy:
///    - High RASD hit rate (>60%): RASD-primary mode
///    - Low RASD hit rate (<10%): Neural-primary mode (skip RASD lookup)
///    - Mixed: Try RASD, fall back to neural per-iteration

pub struct HybridDraftModel {
    /// Primary: retrieval-based drafting
    retrieval: RetrievalDraftModel,

    /// Fallback: neural draft model (EAGLE, self-draft, etc.)
    neural: Box<dyn DraftModel>,

    /// Rolling hit rate for RASD
    hit_rate: f32,

    /// Below this hit rate, skip RASD and go straight to neural
    skip_threshold: f32,
}

impl DraftModel for HybridDraftModel {
    fn draft(&mut self, context: &GenerationContext, features: Option<&LayerFeatures>, config: &DraftConfig) -> Result<DraftTree, SpeculativeError> {
        // Skip RASD if hit rate is too low
        if self.hit_rate >= self.skip_threshold {
            let rasd_result = self.retrieval.draft(context, None, config)?;
            if !rasd_result.is_empty() {
                return Ok(rasd_result);
            }
        }

        // Fall back to neural
        self.neural.draft(context, features, config)
    }
}
```

## Step 4: CLI Commands

```bash
# Build retrieval index from files
ronn index-corpus --sources ./docs ./src --tokenizer model-tokenizer --output corpus.ronn-index

# Incrementally update index
ronn index-update --index corpus.ronn-index --add ./new_files/

# Run with RASD only
ronn infer --model model.onnx --speculative rasd --index corpus.ronn-index --prompt "Hello"

# Run with hybrid (RASD + EAGLE fallback)
ronn infer --model model.onnx --speculative hybrid --index corpus.ronn-index --eagle-head eagle.onnx --prompt "Hello"

# Benchmark RASD hit rates
ronn benchmark-rasd --model model.onnx --index corpus.ronn-index --prompts real_workload.jsonl
```

## Step 5: Testing

1. **Hit rate**: Measure n-gram hit rate on code generation, RAG, and open-ended tasks.
2. **Speedup**: Wall-clock improvement over neural-only speculation on high-hit workloads.
3. **Correctness**: RASD output identical to standard decoding (same rejection sampling guarantees).
4. **Index build time**: Suffix array construction time for corpora of 1K, 100K, 10M tokens.
5. **Memory usage**: Index memory footprint at different corpus sizes.
6. **Hybrid fallback**: Verify neural fallback activates when RASD has no matches.
7. **Incremental updates**: Adding documents to index doesn't corrupt existing entries.

## Implementation Phases

**Phase 1:** NgramIndex with suffix array (or simple trie for v1). Build, search, return continuations.

**Phase 2:** RetrievalDraftModel implementing DraftModel trait. Test with the speculative pipeline.

**Phase 3:** HybridDraftModel (RASD + neural fallback). Hit rate tracking.

**Phase 4:** ronn-memory integration, CLI, benchmarking.

## Success Criteria

### Early Exit
- [ ] ≥99% token accuracy at calibrated thresholds
- [ ] Measurable speedup (target: 1.3-2x)
- [ ] Easy tokens exit early, hard tokens use full depth
- [ ] All three confidence methods implemented and benchmarkable
- [ ] Integrates cleanly with HRM System 1/2 routing

### RASD
- [ ] >60% hit rate on code generation with project files as corpus
- [ ] >40% hit rate on RAG with injected documents as corpus
- [ ] Zero neural cost for matched tokens
- [ ] Hybrid fallback works seamlessly
- [ ] Index builds in <1 minute for 1M token corpus
- [ ] Output identical to standard decoding (rejection sampling guarantee)
