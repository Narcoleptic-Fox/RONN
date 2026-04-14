//! Model configuration extracted from GGUF metadata.
//!
//! Supports multiple architectures: Llama, GPT-2, Phi, Gemma, Qwen.

use nnx_gguf::GGUFMetadata;

/// Supported model architectures.
#[derive(Debug, Clone, PartialEq)]
pub enum Architecture {
    /// Llama, Mistral, CodeLlama, Llama-3 (RMSNorm + GQA + SwiGLU + RoPE)
    Llama,
    /// GPT-2, GPT-J (LayerNorm + MHA + GELU + learned position)
    GPT2,
    /// Phi-2, Phi-3 (LayerNorm + partial RoPE + parallel attn+FFN)
    Phi,
    /// Gemma, Gemma-2 (RMSNorm + GQA + GeGLU + RoPE + scaled embeddings)
    Gemma,
    /// Qwen, Qwen-2 (RMSNorm + GQA + SwiGLU + RoPE + QK bias)
    Qwen,
}

/// Normalization type used as the pre-norm in each block.
#[derive(Debug, Clone, PartialEq)]
pub enum NormType {
    /// Root Mean Square normalization (Llama, Gemma, Qwen)
    RMSNorm,
    /// Full Layer Normalization with mean subtraction (GPT-2, Phi)
    LayerNorm,
}

/// Feed-forward network variant.
#[derive(Debug, Clone, PartialEq)]
pub enum FFNType {
    /// SwiGLU: 3-matrix gated FFN with SiLU activation (Llama, Qwen)
    SwiGLU,
    /// GELU: 2-matrix standard FFN (GPT-2)
    GELU,
    /// GeGLU: 3-matrix gated FFN with GELU activation (Gemma)
    GeGLU,
}

/// Position encoding strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum PosEncoding {
    /// Rotary Position Embedding applied to full head dimension.
    RoPE { freq_base: f32 },
    /// Learned absolute position embeddings (GPT-2).
    Learned,
    /// RoPE applied to only the first `rotary_dim` dimensions of each head (Phi).
    PartialRoPE { freq_base: f32, rotary_dim: usize },
    /// No position encoding applied at the attention level.
    None,
}

/// Whether attention and FFN run sequentially or in parallel within a block.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockStyle {
    /// Standard: attn -> residual -> norm -> ffn -> residual
    Sequential,
    /// Phi-style: attn and ffn both operate on the same normed input, then both residuals are added
    Parallel,
}

/// All hyperparameters needed to construct a transformer model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Raw architecture string from GGUF metadata (e.g. "llama", "gpt2").
    pub architecture: String,
    /// Parsed architecture enum controlling dispatch behavior.
    pub arch: Architecture,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub max_context_length: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
    /// Which normalization to use as pre-norm in each block.
    pub norm_type: NormType,
    /// Which FFN variant to use.
    pub ffn_type: FFNType,
    /// Position encoding strategy.
    pub pos_encoding: PosEncoding,
    /// Whether attention and FFN are sequential or parallel.
    pub block_style: BlockStyle,
    /// Whether Q/K/V projections have bias terms.
    pub has_qkv_bias: bool,
    /// Whether output projection has a bias term.
    pub has_output_bias: bool,
    /// Gemma scales embeddings by sqrt(hidden_dim). None for other architectures.
    pub embedding_scale: Option<f32>,
}

impl ModelConfig {
    /// Extract configuration from GGUF metadata.
    pub fn from_gguf(metadata: &GGUFMetadata) -> Result<Self, String> {
        let architecture = metadata
            .architecture()
            .ok_or("missing general.architecture")?
            .to_string();

        let num_layers = metadata.num_layers().ok_or("missing block_count")? as usize;

        let hidden_dim = metadata.hidden_dim().ok_or("missing embedding_length")? as usize;

        let num_heads = metadata.num_heads().ok_or("missing attention.head_count")? as usize;
        if num_heads == 0 {
            return Err("attention.head_count must be non-zero".into());
        }

        let num_kv_heads = metadata.num_kv_heads().unwrap_or(num_heads as u32) as usize;
        if num_kv_heads == 0 {
            return Err("attention.head_count_kv must be non-zero".into());
        }
        if hidden_dim % num_heads != 0 {
            return Err(format!(
                "embedding_length {} is not divisible by attention.head_count {}",
                hidden_dim, num_heads
            ));
        }

        let head_dim = hidden_dim / num_heads;

        let intermediate_dim = metadata
            .feed_forward_dim()
            .unwrap_or((hidden_dim * 4 * 2 / 3) as u32) as usize;

        let vocab_size = metadata.vocab_size().unwrap_or(32000) as usize;

        let max_context_length = metadata.context_length().unwrap_or(4096) as usize;

        let rope_freq_base = metadata.rope_freq_base().unwrap_or(10000.0);

        let rms_norm_eps = metadata
            .get(&format!("{architecture}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        // Detect architecture and derive configuration
        let arch = match architecture.as_str() {
            "llama" | "mistral" => Architecture::Llama,
            "gpt2" | "gptj" => Architecture::GPT2,
            "phi2" | "phi3" | "phi" => Architecture::Phi,
            "gemma" | "gemma2" => Architecture::Gemma,
            "qwen" | "qwen2" => Architecture::Qwen,
            _ => return Err(format!("unsupported architecture: {}", architecture)),
        };

        let (norm_type, ffn_type, pos_encoding, block_style, has_qkv_bias, has_output_bias) =
            match arch {
                Architecture::Llama => (
                    NormType::RMSNorm,
                    FFNType::SwiGLU,
                    PosEncoding::RoPE {
                        freq_base: rope_freq_base,
                    },
                    BlockStyle::Sequential,
                    false,
                    false,
                ),
                Architecture::GPT2 => (
                    NormType::LayerNorm,
                    FFNType::GELU,
                    PosEncoding::Learned,
                    BlockStyle::Sequential,
                    true,
                    true,
                ),
                Architecture::Phi => (
                    NormType::LayerNorm,
                    FFNType::SwiGLU,
                    PosEncoding::PartialRoPE {
                        freq_base: rope_freq_base,
                        rotary_dim: head_dim / 2,
                    },
                    BlockStyle::Parallel,
                    true,
                    true,
                ),
                Architecture::Gemma => (
                    NormType::RMSNorm,
                    FFNType::GeGLU,
                    PosEncoding::RoPE {
                        freq_base: rope_freq_base,
                    },
                    BlockStyle::Sequential,
                    false,
                    false,
                ),
                Architecture::Qwen => (
                    NormType::RMSNorm,
                    FFNType::SwiGLU,
                    PosEncoding::RoPE {
                        freq_base: rope_freq_base,
                    },
                    BlockStyle::Sequential,
                    true,
                    false,
                ),
            };

        let embedding_scale = match arch {
            Architecture::Gemma => Some((hidden_dim as f32).sqrt()),
            _ => None,
        };

        let config = Self {
            architecture,
            arch,
            num_layers,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
            max_context_length,
            rope_freq_base,
            rms_norm_eps,
            norm_type,
            ffn_type,
            pos_encoding,
            block_style,
            has_qkv_bias,
            has_output_bias,
            embedding_scale,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration invariants that would otherwise fail deep in kernels.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads == 0 {
            return Err("num_heads must be non-zero".into());
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be non-zero".into());
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be non-zero".into());
        }
        if self.hidden_dim % self.num_heads != 0 {
            return Err(format!(
                "hidden_dim {} must be divisible by num_heads {}",
                self.hidden_dim, self.num_heads
            ));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "num_heads {} must be divisible by num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.head_dim == 0 {
            return Err("head_dim must be non-zero".into());
        }
        if self.max_context_length == 0 {
            return Err("max_context_length must be non-zero".into());
        }
        match self.pos_encoding {
            PosEncoding::RoPE { .. } => {
                if self.head_dim % 2 != 0 {
                    return Err(format!(
                        "RoPE requires even head_dim, got {}",
                        self.head_dim
                    ));
                }
            }
            PosEncoding::PartialRoPE { rotary_dim, .. } => {
                if rotary_dim == 0 {
                    return Err("PartialRoPE rotary_dim must be non-zero".into());
                }
                if rotary_dim > self.head_dim {
                    return Err(format!(
                        "PartialRoPE rotary_dim {} exceeds head_dim {}",
                        rotary_dim, self.head_dim
                    ));
                }
                if rotary_dim % 2 != 0 {
                    return Err(format!(
                        "PartialRoPE rotary_dim must be even, got {}",
                        rotary_dim
                    ));
                }
            }
            PosEncoding::Learned | PosEncoding::None => {}
        }
        Ok(())
    }

    /// Total parameter count estimate.
    pub fn estimated_params(&self) -> u64 {
        let embed = self.vocab_size * self.hidden_dim;
        let attn_per_layer = 4 * self.hidden_dim * self.hidden_dim;
        let ffn_per_layer = match self.ffn_type {
            FFNType::GELU => 2 * self.hidden_dim * self.intermediate_dim,
            FFNType::SwiGLU | FFNType::GeGLU => 3 * self.hidden_dim * self.intermediate_dim,
        };
        let norm_per_layer = 2 * self.hidden_dim;
        let per_layer = attn_per_layer + ffn_per_layer + norm_per_layer;
        let head = self.hidden_dim * self.vocab_size;
        (embed + per_layer * self.num_layers + head) as u64
    }

    /// Create a default Llama-style config for testing purposes.
    #[cfg(test)]
    pub(crate) fn test_llama(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            architecture: "test".into(),
            arch: Architecture::Llama,
            num_layers: 2,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: NormType::RMSNorm,
            ffn_type: FFNType::SwiGLU,
            pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
            block_style: BlockStyle::Sequential,
            has_qkv_bias: false,
            has_output_bias: false,
            embedding_scale: None,
        }
    }
}
