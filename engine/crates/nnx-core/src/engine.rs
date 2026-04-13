//! The `InferenceEngine` trait — the contract between orchestration (RONN)
//! and execution (NNX).
//!
//! Any backend that implements this trait can be driven by RONN's HRM,
//! speculative decoding, caching, and batching systems. The trait is
//! intentionally minimal: load a model, run forward passes, access the
//! KV cache, and read model metadata.

use crate::device::Device;
use crate::error::Result;
use crate::shape::Shape;
use crate::tensor::Tensor;
use std::path::Path;

/// Opaque handle to a loaded model. The engine decides what this contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelHandle(pub u64);

/// Configuration for model loading.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Preferred device for model weights.
    pub device: Device,
    /// Maximum memory budget in bytes (0 = unlimited).
    pub memory_budget: usize,
    /// Number of threads for CPU compute.
    pub num_threads: usize,
    /// Context length override (0 = use model default).
    pub context_length: usize,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            memory_budget: 0,
            num_threads: 0, // auto-detect
            context_length: 0, // model default
        }
    }
}

/// A batch of token IDs for forward pass input.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    /// Token IDs, shape: [batch_size, seq_len].
    pub token_ids: Vec<Vec<u32>>,
    /// Starting position for each sequence (for KV cache indexing).
    pub positions: Vec<usize>,
}

impl TokenBatch {
    /// Create a batch with a single sequence.
    pub fn single(tokens: Vec<u32>, position: usize) -> Self {
        Self {
            token_ids: vec![tokens],
            positions: vec![position],
        }
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        self.token_ids.len()
    }
}

/// Output from a forward pass: logits over the vocabulary.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Logits tensor, shape: [batch_size, vocab_size].
    pub logits: Tensor,
    /// Which layer features were extracted (if requested).
    pub layer_features: Option<Vec<(usize, Tensor)>>,
}

/// Static metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model architecture name (e.g., "llama", "mistral", "phi").
    pub architecture: String,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA/MQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length.
    pub max_context_length: usize,
    /// Total parameter count.
    pub num_parameters: u64,
    /// Model file size in bytes.
    pub file_size_bytes: u64,
    /// Quantization type description (e.g., "Q4_K_M", "F16").
    pub quantization: String,
}

/// Access to a model's KV cache for external management.
///
/// This is the interface that RONN's cache tiering, eviction policies,
/// and speculative decoding use to inspect and manipulate cached state.
pub trait KVCacheAccess: Send + Sync {
    /// Current number of cached tokens across all layers.
    fn cached_tokens(&self) -> usize;

    /// Maximum capacity in tokens.
    fn capacity(&self) -> usize;

    /// Memory usage in bytes.
    fn memory_usage_bytes(&self) -> usize;

    /// Clear all cached state.
    fn clear(&mut self);

    /// Truncate cache to keep only the first `n` tokens.
    fn truncate(&mut self, n: usize);
}

/// The core trait connecting orchestration to execution.
///
/// Orchestration layers (RONN) program against this trait.
/// Execution engines (NNX, or any future backend) implement it.
pub trait InferenceEngine: Send + Sync {
    /// Load a model from a file path.
    fn load_model(&self, path: &Path, config: &LoadConfig) -> Result<ModelHandle>;

    /// Unload a model and free its resources.
    fn unload_model(&self, handle: ModelHandle) -> Result<()>;

    /// Run a full forward pass, returning logits.
    fn forward(&self, handle: ModelHandle, input: &TokenBatch) -> Result<GenerationOutput>;

    /// Run a partial forward pass over a subset of layers.
    ///
    /// Required for early-exit inference and self-speculative decoding.
    /// `start_layer` is inclusive, `end_layer` is exclusive.
    fn forward_layers(
        &self,
        handle: ModelHandle,
        input: &Tensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Tensor>;

    /// Get model metadata.
    fn model_info(&self, handle: ModelHandle) -> Result<&ModelInfo>;

    /// Get a mutable reference to the KV cache for external management.
    fn kv_cache(&self, handle: ModelHandle) -> Result<&mut dyn KVCacheAccess>;

    /// Extract intermediate layer features during the next forward pass.
    ///
    /// Used by EAGLE-style speculative decoding to tap into target model layers.
    fn request_layer_features(&self, handle: ModelHandle, layer_indices: &[usize]) -> Result<()>;
}
