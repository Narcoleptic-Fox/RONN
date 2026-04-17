//! Minimal GPU inference configuration.
//!
//! This type carries only the numerical dimensions and runtime-evaluated
//! parameters that `GpuInference` in `nnx-cubecl` needs to run a decode
//! step.  It is intentionally free of the rich architecture-profile
//! machinery in `nnx-transformer::config` so that `nnx-cubecl` can depend
//! on `nnx-core` alone without creating a circular dependency.
//!
//! `nnx-transformer` converts its `ModelConfig` to a `GpuConfig` via
//! `ModelConfig::to_gpu_config()` before handing the model off to the GPU.

/// Which normalization strategy the model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuNormType {
    RMSNorm,
    LayerNorm,
}

/// Which feed-forward network variant the model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFFNType {
    /// SiLU-gated linear unit: silu(gate) * up, then down.
    SwiGLU,
    /// GELU-gated linear unit: gelu(gate) * up, then down.
    GeGLU,
    /// Plain GELU: gelu(fc1), then fc2. Only 2 matrices, no up projection.
    GELU,
}

/// Whether the transformer uses sequential or parallel attention+FFN blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBlockStyle {
    /// Attention residual first, then FFN residual (Llama, GPT-2, Gemma, etc.).
    Sequential,
    /// Attention and FFN computed from the same normed input, both residuals
    /// added together (Phi, StableLM).
    Parallel,
}

/// Which position encoding strategy is active at runtime.
///
/// A subset of `nnx-transformer::config::PosEncoding` containing only the
/// variants and parameters that the GPU forward pass needs to dispatch on.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuPosEncoding {
    /// Full-head Rotary Position Embedding with the given frequency base.
    RoPE { freq_base: f32 },
    /// Learned absolute position embeddings (GPT-2 style).
    Learned,
    /// Partial RoPE applied to the first `rotary_dim` dimensions per head (Phi).
    PartialRoPE { freq_base: f32, rotary_dim: usize },
    /// No position encoding at the attention level.
    None,
}

/// Opaque identifier for a physical KV cache page in the block allocator.
///
/// Defined in `nnx-core` (not `nnx-serving`) so that GPU kernel crates
/// like `nnx-cubecl` can accept page IDs without depending on the
/// higher-level `nnx-serving` crate, which would create a circular
/// dependency through `nnx-transformer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

impl PageId {
    /// Sentinel value for "no page" / "invalid".
    pub const INVALID: PageId = PageId(u32::MAX);
}

/// Activation quantization strategy for intermediate GPU buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuActivationQuant {
    /// No activation quantization; all intermediates remain f32.
    None,
    /// Q8_0 round-trip quantization for attention intermediates (Q, scores).
    Q8_0,
}

impl Default for GpuActivationQuant {
    fn default() -> Self {
        Self::None
    }
}

/// Minimal model configuration needed to run a GPU decode step.
///
/// Produced by `ModelConfig::to_gpu_config()` in `nnx-transformer`.
/// Stored inside `GpuInference<R>` so the forward pass can dispatch on it
/// without referring to any `nnx-transformer` types.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Embedding / residual stream dimension.
    pub hidden_dim: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of KV attention heads (GQA/MQA; equals `num_heads` for MHA).
    pub num_kv_heads: usize,
    /// Dimension of each attention head (`hidden_dim / num_heads` for most models).
    pub head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length supported by the KV cache.
    pub max_context_length: usize,
    /// Position encoding in use.
    pub pos_encoding: GpuPosEncoding,
    /// Per-element epsilon used in RMS-Norm / Layer-Norm.
    pub rms_norm_eps: f32,
    /// Optional embedding scale factor (e.g. `sqrt(hidden_dim)` for Gemma).
    pub embedding_scale: Option<f32>,
    /// Normalization type (RMSNorm or LayerNorm).
    pub norm_type: GpuNormType,
    /// FFN type (SwiGLU, GeGLU, or plain GELU).
    pub ffn_type: GpuFFNType,
    /// Block style (Sequential or Parallel).
    pub block_style: GpuBlockStyle,
    /// Whether Q/K/V projections have bias terms.
    pub has_qkv_bias: bool,
    /// Whether the output projection has a bias term.
    pub has_output_bias: bool,
    /// Activation quantization strategy (default: None).
    pub activation_quant: GpuActivationQuant,
}
