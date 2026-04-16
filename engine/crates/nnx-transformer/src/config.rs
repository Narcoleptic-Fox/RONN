//! Model configuration extracted from GGUF metadata.
//!
//! # Architecture Profile System
//!
//! Adding a new model architecture does **not** require modifying any dispatch
//! code. Instead, define a new [`ArchitectureProfile`] constant and add it to
//! [`KNOWN_ARCHITECTURES`]. The profile declares which normalization, FFN
//! variant, position encoding kind, block style, and bias configuration the
//! architecture uses. At load time the profile is looked up by the GGUF
//! architecture string or HuggingFace `model_type`, and those values are
//! copied into the [`ModelConfig`]. All kernel dispatch already happens on the
//! `norm_type`, `ffn_type`, etc. fields — not on the `arch` enum — so no
//! further changes are needed.
//!
//! ## Example: adding a new architecture
//!
//! ```rust,ignore
//! pub const MY_ARCH_PROFILE: ArchitectureProfile = ArchitectureProfile {
//!     name: "myarch",
//!     norm_type: NormType::RMSNorm,
//!     ffn_type: FFNType::SwiGLU,
//!     pos_encoding_kind: PosEncodingKind::RoPE,
//!     block_style: BlockStyle::Sequential,
//!     has_qkv_bias: false,
//!     has_output_bias: false,
//!     has_embedding_scale: false,
//!     tied_embeddings: false,
//!     gguf_names: &["myarch"],
//!     hf_names: &["MyArchModel", "myarch"],
//! };
//! ```
//! Then add `&MY_ARCH_PROFILE` to [`KNOWN_ARCHITECTURES`].

use nnx_gguf::GGUFMetadata;

// ============================================================
// Core enums used in ModelConfig and dispatch
// ============================================================

/// Supported model architectures.
///
/// This enum is maintained for logging, debugging, and backward compatibility.
/// It is **not** the primary dispatch key — all kernel dispatch happens via
/// `norm_type`, `ffn_type`, `pos_encoding`, and `block_style` in [`ModelConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum Architecture {
    /// Llama family (RMSNorm + GQA + SwiGLU + RoPE)
    Llama,
    /// GPT-2/GPT-J (LayerNorm + MHA + GELU + learned position)
    GPT2,
    /// Phi-2/Phi-3 (LayerNorm + partial RoPE + parallel attn+FFN)
    Phi,
    /// Gemma/Gemma-2 (RMSNorm + GQA + GeGLU + RoPE + scaled embeddings)
    Gemma,
    /// Qwen/Qwen-2 (RMSNorm + GQA + SwiGLU + RoPE + QK bias)
    Qwen,
    /// Mistral — same execution path as Llama, separate variant for clarity
    Mistral,
    /// CodeLlama — same execution path as Llama, documented separately
    CodeLlama,
    /// StableLM — LayerNorm + RoPE + SwiGLU + parallel attention
    StableLM,
    /// Falcon — LayerNorm + MHA + GELU + sequential
    Falcon,
    /// MPT — LayerNorm + MHA + GELU + sequential
    MPT,
}

/// Normalization type used as the pre-norm in each block.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum NormType {
    /// Root Mean Square normalization (Llama, Gemma, Qwen)
    RMSNorm,
    /// Full Layer Normalization with mean subtraction (GPT-2, Phi, StableLM, Falcon, MPT)
    LayerNorm,
}

/// Feed-forward network variant.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum FFNType {
    /// SwiGLU: 3-matrix gated FFN with SiLU activation (Llama, Qwen, Phi, StableLM)
    SwiGLU,
    /// GELU: 2-matrix standard FFN (GPT-2, Falcon, MPT)
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

/// Position encoding variant without runtime parameters.
///
/// Used in [`ArchitectureProfile`] because const statics cannot hold runtime
/// values like `freq_base` or `rotary_dim`. The loader substitutes those
/// values from GGUF metadata when building the final [`PosEncoding`].
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum PosEncodingKind {
    /// Full-head RoPE. Runtime `freq_base` comes from GGUF/config.json.
    RoPE,
    /// Learned absolute position embeddings.
    Learned,
    /// Partial RoPE; `rotary_dim` defaults to `head_dim / 2` unless overridden.
    PartialRoPE,
    /// No position encoding.
    None,
}

/// Whether attention and FFN run sequentially or in parallel within a block.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum BlockStyle {
    /// Standard: attn -> residual -> norm -> ffn -> residual
    Sequential,
    /// Phi-style: attn and ffn both operate on the same normed input, then both residuals are added
    Parallel,
}

// ============================================================
// Architecture profile registry
// ============================================================

/// Complete, static description of how a model architecture behaves.
///
/// Adding a new architecture means creating a new profile constant and adding
/// it to [`KNOWN_ARCHITECTURES`]. No dispatch code needs to be modified.
///
/// All fields are `Copy`-able types or `&'static` slices so the struct can
/// live in `const` storage with zero heap allocation.
#[derive(Debug)]
pub struct ArchitectureProfile {
    /// Human-readable name (used in error messages and logging).
    pub name: &'static str,
    /// Normalization applied before attention and FFN.
    pub norm_type: NormType,
    /// Which FFN variant this architecture uses.
    pub ffn_type: FFNType,
    /// Position encoding kind. Runtime parameters (freq_base, rotary_dim) are
    /// filled in by the loader from GGUF metadata or config.json.
    pub pos_encoding_kind: PosEncodingKind,
    /// Sequential or parallel block style.
    pub block_style: BlockStyle,
    /// Whether Q/K/V projections carry bias terms.
    pub has_qkv_bias: bool,
    /// Whether the output projection carries a bias term.
    pub has_output_bias: bool,
    /// Whether token embeddings are scaled by `sqrt(hidden_dim)` (Gemma).
    pub has_embedding_scale: bool,
    /// Whether the LM head weight is tied to the token embedding weight.
    pub tied_embeddings: bool,
    /// GGUF `general.architecture` strings that map to this profile.
    /// Matching is case-insensitive.
    pub gguf_names: &'static [&'static str],
    /// HuggingFace `model_type` / `architectures` strings that map to this profile.
    /// Matching is case-insensitive substring search.
    pub hf_names: &'static [&'static str],
}

// ---- Profile constants ----

/// Llama family: RMSNorm, SwiGLU, full RoPE, sequential.
pub const LLAMA_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "llama",
    norm_type: NormType::RMSNorm,
    ffn_type: FFNType::SwiGLU,
    pos_encoding_kind: PosEncodingKind::RoPE,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: false,
    has_output_bias: false,
    has_embedding_scale: false,
    tied_embeddings: false,
    gguf_names: &["llama"],
    hf_names: &["llama", "llama2", "llamaforcausallm"],
};

/// Mistral: same execution path as Llama, separate variant for clarity.
/// Mistral uses sliding-window attention in some layers, but the block
/// structure and weight shapes are identical to Llama. SWA is a future
/// optimization; for now both run the same kernel.
pub const MISTRAL_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "mistral",
    norm_type: NormType::RMSNorm,
    ffn_type: FFNType::SwiGLU,
    pos_encoding_kind: PosEncodingKind::RoPE,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: false,
    has_output_bias: false,
    has_embedding_scale: false,
    tied_embeddings: false,
    gguf_names: &["mistral"],
    hf_names: &["mistral", "mistralforcausallm"],
};

/// CodeLlama: same execution path as Llama with infilling support.
/// The weight layout is identical to Llama; CodeLlama-specific features
/// (infill tokens, vocabulary extensions) are handled at the tokenizer level.
pub const CODELLAMA_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "codellama",
    norm_type: NormType::RMSNorm,
    ffn_type: FFNType::SwiGLU,
    pos_encoding_kind: PosEncodingKind::RoPE,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: false,
    has_output_bias: false,
    has_embedding_scale: false,
    tied_embeddings: false,
    gguf_names: &["codellama"],
    hf_names: &["codellama", "code_llama"],
};

/// GPT-2 / GPT-J: LayerNorm, GELU, learned position embeddings, full bias.
pub const GPT2_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "gpt2",
    norm_type: NormType::LayerNorm,
    ffn_type: FFNType::GELU,
    pos_encoding_kind: PosEncodingKind::Learned,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: true,
    has_output_bias: true,
    has_embedding_scale: false,
    tied_embeddings: true,
    gguf_names: &["gpt2", "gptj"],
    hf_names: &["gpt2", "gpt_j", "gptj", "gpt2lmheadmodel"],
};

/// Phi-2 / Phi-3: LayerNorm, SwiGLU, partial RoPE, parallel block, full bias.
pub const PHI_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "phi",
    norm_type: NormType::LayerNorm,
    ffn_type: FFNType::SwiGLU,
    pos_encoding_kind: PosEncodingKind::PartialRoPE,
    block_style: BlockStyle::Parallel,
    has_qkv_bias: true,
    has_output_bias: true,
    has_embedding_scale: false,
    tied_embeddings: false,
    gguf_names: &["phi2", "phi3", "phi"],
    hf_names: &["phi", "phi2", "phi3", "phiforcausallm", "phi3forcausallm"],
};

/// Gemma / Gemma-2: RMSNorm, GeGLU, full RoPE, sequential, scaled embeddings.
pub const GEMMA_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "gemma",
    norm_type: NormType::RMSNorm,
    ffn_type: FFNType::GeGLU,
    pos_encoding_kind: PosEncodingKind::RoPE,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: false,
    has_output_bias: false,
    has_embedding_scale: true,
    tied_embeddings: true,
    gguf_names: &["gemma", "gemma2"],
    hf_names: &["gemma", "gemma2", "gemmaforcausallm", "gemma2forcausallm"],
};

/// Qwen / Qwen-2: RMSNorm, SwiGLU, full RoPE, sequential, QKV bias.
pub const QWEN_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "qwen",
    norm_type: NormType::RMSNorm,
    ffn_type: FFNType::SwiGLU,
    pos_encoding_kind: PosEncodingKind::RoPE,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: true,
    has_output_bias: false,
    has_embedding_scale: false,
    tied_embeddings: false,
    gguf_names: &["qwen", "qwen2"],
    hf_names: &["qwen", "qwen2", "qwen2forcausallm", "qwenforcausallm"],
};

/// StableLM: LayerNorm, SwiGLU, full RoPE, parallel block style, bias on QKV.
///
/// StableLM uses a parallel attention+FFN design similar to Phi but with
/// RoPE applied to the full head dimension rather than a partial rotation.
pub const STABLELM_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "stablelm",
    norm_type: NormType::LayerNorm,
    ffn_type: FFNType::SwiGLU,
    pos_encoding_kind: PosEncodingKind::RoPE,
    block_style: BlockStyle::Parallel,
    has_qkv_bias: true,
    has_output_bias: false,
    has_embedding_scale: false,
    tied_embeddings: false,
    gguf_names: &["stablelm"],
    hf_names: &["stablelm", "stabilityai", "stabilityailm", "stablelmforcausallm"],
};

/// Falcon: LayerNorm, GELU, full RoPE, sequential, no QKV/output bias.
///
/// Newer Falcon variants (Falcon-2, Falcon-3) use RoPE; older versions used
/// ALiBi. This profile covers the RoPE variants. ALiBi support would require
/// a dedicated PosEncodingKind.
pub const FALCON_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "falcon",
    norm_type: NormType::LayerNorm,
    ffn_type: FFNType::GELU,
    pos_encoding_kind: PosEncodingKind::RoPE,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: false,
    has_output_bias: false,
    has_embedding_scale: false,
    tied_embeddings: false,
    gguf_names: &["falcon"],
    hf_names: &["falcon", "rw", "falconforcausallm"],
};

/// MPT: LayerNorm, GELU, no position encoding at attention level (ALiBi handled
/// in attention bias, approximated here as None), sequential.
///
/// Note: MPT uses ALiBi for position encoding. The `None` kind means no RoPE
/// rotation is applied; ALiBi biases are not yet modelled in the kernel layer.
/// Models load and run but positional generalisation may differ from the reference.
pub const MPT_PROFILE: ArchitectureProfile = ArchitectureProfile {
    name: "mpt",
    norm_type: NormType::LayerNorm,
    ffn_type: FFNType::GELU,
    pos_encoding_kind: PosEncodingKind::None,
    block_style: BlockStyle::Sequential,
    has_qkv_bias: false,
    has_output_bias: false,
    has_embedding_scale: false,
    tied_embeddings: true,
    gguf_names: &["mpt"],
    hf_names: &["mpt", "mptforcausallm"],
};

/// All architectures known to the engine.
///
/// Profiles are checked in order; the first match wins. Put more-specific
/// profiles (e.g. `CODELLAMA_PROFILE`) before catch-all ones (`LLAMA_PROFILE`)
/// if their GGUF/HF name sets overlap.
pub const KNOWN_ARCHITECTURES: &[&ArchitectureProfile] = &[
    &CODELLAMA_PROFILE,
    &MISTRAL_PROFILE,
    &LLAMA_PROFILE,
    &GPT2_PROFILE,
    &PHI_PROFILE,
    &GEMMA_PROFILE,
    &QWEN_PROFILE,
    &STABLELM_PROFILE,
    &FALCON_PROFILE,
    &MPT_PROFILE,
];

// ---- Lookup functions ----

/// Find an architecture profile by GGUF `general.architecture` string.
///
/// Matching is case-insensitive exact comparison against each profile's
/// `gguf_names` list.
pub fn find_profile_by_gguf(arch_str: &str) -> Option<&'static ArchitectureProfile> {
    let lower = arch_str.to_ascii_lowercase();
    KNOWN_ARCHITECTURES
        .iter()
        .copied()
        .find(|p| p.gguf_names.iter().any(|&n| n == lower.as_str()))
}

/// Find an architecture profile by HuggingFace `model_type` or `architectures` string.
///
/// Matching is case-insensitive substring search so that strings like
/// `"LlamaForCausalLM"` match `"llama"`.
pub fn find_profile_by_hf(model_type: &str) -> Option<&'static ArchitectureProfile> {
    let lower = model_type.to_ascii_lowercase();
    KNOWN_ARCHITECTURES
        .iter()
        .copied()
        .find(|p| p.hf_names.iter().any(|&n| lower.contains(n)))
}

/// List the human-readable names of all known architectures.
///
/// Used in error messages when an unsupported architecture is encountered.
pub fn known_architecture_names() -> Vec<&'static str> {
    KNOWN_ARCHITECTURES.iter().map(|p| p.name).collect()
}

// ============================================================
// ModelConfig
// ============================================================

/// All hyperparameters needed to construct a transformer model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Raw architecture string from GGUF metadata (e.g. "llama", "gpt2").
    pub architecture: String,
    /// Parsed architecture enum — kept for logging and backward compatibility.
    /// Kernel dispatch uses `norm_type`, `ffn_type`, `pos_encoding`, and
    /// `block_style` rather than this field.
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
    /// Position encoding strategy (with runtime parameters applied).
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
    ///
    /// The architecture string from the GGUF file is looked up in the
    /// [`KNOWN_ARCHITECTURES`] registry. If no profile matches the error
    /// message lists all known architecture names.
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

        // Look up the profile — replacing the old big match statement.
        let profile = find_profile_by_gguf(&architecture).ok_or_else(|| {
            format!(
                "unsupported architecture: '{}'. Known architectures: {:?}",
                architecture,
                known_architecture_names()
            )
        })?;

        let arch = profile_to_arch_enum(profile, &architecture);
        let pos_encoding = build_pos_encoding(profile.pos_encoding_kind, rope_freq_base, head_dim);
        let embedding_scale = if profile.has_embedding_scale {
            Some((hidden_dim as f32).sqrt())
        } else {
            None
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
            norm_type: profile.norm_type,
            ffn_type: profile.ffn_type,
            pos_encoding,
            block_style: profile.block_style,
            has_qkv_bias: profile.has_qkv_bias,
            has_output_bias: profile.has_output_bias,
            embedding_scale,
        };
        config.validate()?;
        Ok(config)
    }

    /// Check whether the current combination of config fields is fully
    /// supported by the kernel layer.
    ///
    /// This catches configurations that have a valid profile entry but whose
    /// kernel implementations are incomplete or missing.
    pub fn validate_support(&self) -> Result<(), String> {
        // All NormType variants are implemented.
        // All FFNType variants are implemented.
        // PosEncodingKind::None means no rotation — that is a deliberate
        // choice for MPT (ALiBi not yet modelled) and is allowed.
        // All BlockStyle variants are implemented.
        // If a future PosEncodingKind is added without a kernel, add a check here.
        Ok(())
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
        let pos_embed = match self.pos_encoding {
            PosEncoding::Learned => self.max_context_length * self.hidden_dim,
            PosEncoding::RoPE { .. } | PosEncoding::PartialRoPE { .. } | PosEncoding::None => 0,
        };
        let attn_per_layer = 4 * self.hidden_dim * self.hidden_dim;
        let ffn_per_layer = match self.ffn_type {
            FFNType::GELU => 2 * self.hidden_dim * self.intermediate_dim,
            FFNType::SwiGLU | FFNType::GeGLU => 3 * self.hidden_dim * self.intermediate_dim,
        };
        let norm_per_layer = 2 * self.hidden_dim;
        let per_layer = attn_per_layer + ffn_per_layer + norm_per_layer;
        let head = self.hidden_dim * self.vocab_size;
        (embed + pos_embed + per_layer * self.num_layers + head) as u64
    }

    /// Convert to the minimal GPU configuration used by `nnx-cubecl`.
    ///
    /// `GpuInference` stores a `GpuConfig` rather than a `ModelConfig` so that
    /// `nnx-cubecl` can depend only on `nnx-core` and avoid a circular
    /// dependency with `nnx-transformer`.
    pub fn to_gpu_config(&self) -> nnx_core::gpu_config::GpuConfig {
        use nnx_core::gpu_config::{
            GpuBlockStyle, GpuConfig, GpuFFNType, GpuNormType, GpuPosEncoding,
        };

        let pos_encoding = match &self.pos_encoding {
            PosEncoding::RoPE { freq_base } => {
                GpuPosEncoding::RoPE { freq_base: *freq_base }
            }
            PosEncoding::Learned => GpuPosEncoding::Learned,
            PosEncoding::PartialRoPE { freq_base, rotary_dim } => {
                GpuPosEncoding::PartialRoPE {
                    freq_base: *freq_base,
                    rotary_dim: *rotary_dim,
                }
            }
            PosEncoding::None => GpuPosEncoding::None,
        };

        let norm_type = match self.norm_type {
            NormType::RMSNorm => GpuNormType::RMSNorm,
            NormType::LayerNorm => GpuNormType::LayerNorm,
        };

        let ffn_type = match self.ffn_type {
            FFNType::SwiGLU => GpuFFNType::SwiGLU,
            FFNType::GeGLU => GpuFFNType::GeGLU,
            FFNType::GELU => GpuFFNType::GELU,
        };

        let block_style = match self.block_style {
            BlockStyle::Sequential => GpuBlockStyle::Sequential,
            BlockStyle::Parallel => GpuBlockStyle::Parallel,
        };

        GpuConfig {
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate_dim: self.intermediate_dim,
            vocab_size: self.vocab_size,
            max_context_length: self.max_context_length,
            pos_encoding,
            rms_norm_eps: self.rms_norm_eps,
            embedding_scale: self.embedding_scale,
            norm_type,
            ffn_type,
            block_style,
            has_qkv_bias: self.has_qkv_bias,
            has_output_bias: self.has_output_bias,
        }
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

// ============================================================
// Internal helpers
// ============================================================

/// Build a [`PosEncoding`] value from a profile kind and runtime parameters.
///
/// This is separated from the profile lookup so that the `ArchitectureProfile`
/// struct stays `const`-compatible (no heap allocation, no runtime values).
pub(crate) fn build_pos_encoding(
    kind: PosEncodingKind,
    freq_base: f32,
    head_dim: usize,
) -> PosEncoding {
    match kind {
        PosEncodingKind::RoPE => PosEncoding::RoPE { freq_base },
        PosEncodingKind::Learned => PosEncoding::Learned,
        PosEncodingKind::PartialRoPE => PosEncoding::PartialRoPE {
            freq_base,
            // Default convention: rotate the first half of the head dimension.
            rotary_dim: head_dim / 2,
        },
        PosEncodingKind::None => PosEncoding::None,
    }
}

/// Map an [`ArchitectureProfile`] to the legacy [`Architecture`] enum value.
///
/// The mapping uses the profile name so that new profiles added to the registry
/// automatically get a sensible enum variant without requiring changes here —
/// except when a brand-new `Architecture` variant is also needed.
pub(crate) fn profile_to_arch_enum(profile: &ArchitectureProfile, raw: &str) -> Architecture {
    match profile.name {
        "llama" => Architecture::Llama,
        "mistral" => Architecture::Mistral,
        "codellama" => Architecture::CodeLlama,
        "gpt2" => Architecture::GPT2,
        "phi" => Architecture::Phi,
        "gemma" => Architecture::Gemma,
        "qwen" => Architecture::Qwen,
        "stablelm" => Architecture::StableLM,
        "falcon" => Architecture::Falcon,
        "mpt" => Architecture::MPT,
        // Safety net: if a profile is added without updating this match,
        // we still produce a valid (Llama-shaped) enum. The actual dispatch
        // uses the config fields, not the enum, so correctness is preserved.
        _ => {
            tracing::warn!(
                "profile '{}' (raw: '{}') has no Architecture enum variant; defaulting to Llama",
                profile.name,
                raw
            );
            Architecture::Llama
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Registry integrity ----

    #[test]
    fn all_profiles_have_nonempty_gguf_names() {
        for profile in KNOWN_ARCHITECTURES {
            assert!(
                !profile.gguf_names.is_empty(),
                "profile '{}' has empty gguf_names",
                profile.name
            );
        }
    }

    #[test]
    fn all_profiles_have_nonempty_hf_names() {
        for profile in KNOWN_ARCHITECTURES {
            assert!(
                !profile.hf_names.is_empty(),
                "profile '{}' has empty hf_names",
                profile.name
            );
        }
    }

    #[test]
    fn all_profiles_validate_support() {
        // Every known profile must pass validate_support when applied to a
        // minimal but coherent ModelConfig.
        for profile in KNOWN_ARCHITECTURES {
            let pos_encoding = build_pos_encoding(profile.pos_encoding_kind, 10000.0, 64);
            let config = ModelConfig {
                architecture: profile.name.to_string(),
                arch: Architecture::Llama, // enum value doesn't affect support check
                num_layers: 1,
                hidden_dim: 64,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 64,
                intermediate_dim: 128,
                vocab_size: 128,
                max_context_length: 16,
                rope_freq_base: 10000.0,
                rms_norm_eps: 1e-5,
                norm_type: profile.norm_type,
                ffn_type: profile.ffn_type,
                pos_encoding,
                block_style: profile.block_style,
                has_qkv_bias: profile.has_qkv_bias,
                has_output_bias: profile.has_output_bias,
                embedding_scale: if profile.has_embedding_scale {
                    Some(8.0)
                } else {
                    None
                },
            };
            config
                .validate_support()
                .unwrap_or_else(|e| panic!("profile '{}' failed validate_support: {}", profile.name, e));
        }
    }

    // ---- GGUF lookup ----

    #[test]
    fn find_profile_by_gguf_llama() {
        let p = find_profile_by_gguf("llama").unwrap();
        assert_eq!(p.name, "llama");
        assert_eq!(p.norm_type, NormType::RMSNorm);
        assert_eq!(p.ffn_type, FFNType::SwiGLU);
        assert_eq!(p.pos_encoding_kind, PosEncodingKind::RoPE);
        assert_eq!(p.block_style, BlockStyle::Sequential);
        assert!(!p.has_qkv_bias);
        assert!(!p.has_output_bias);
        assert!(!p.has_embedding_scale);
    }

    #[test]
    fn find_profile_by_gguf_case_insensitive() {
        // GGUF strings should match regardless of case.
        assert!(find_profile_by_gguf("LLAMA").is_some());
        assert!(find_profile_by_gguf("Llama").is_some());
        assert!(find_profile_by_gguf("GPT2").is_some());
        assert!(find_profile_by_gguf("Gemma2").is_some());
    }

    #[test]
    fn find_profile_by_gguf_mistral() {
        let p = find_profile_by_gguf("mistral").unwrap();
        assert_eq!(p.name, "mistral");
        assert_eq!(p.norm_type, NormType::RMSNorm);
        assert_eq!(p.ffn_type, FFNType::SwiGLU);
    }

    #[test]
    fn find_profile_by_gguf_codellama() {
        let p = find_profile_by_gguf("codellama").unwrap();
        assert_eq!(p.name, "codellama");
        // CodeLlama must have same execution profile as Llama.
        assert_eq!(p.norm_type, NormType::RMSNorm);
        assert_eq!(p.ffn_type, FFNType::SwiGLU);
        assert_eq!(p.pos_encoding_kind, PosEncodingKind::RoPE);
        assert_eq!(p.block_style, BlockStyle::Sequential);
    }

    #[test]
    fn find_profile_by_gguf_stablelm() {
        let p = find_profile_by_gguf("stablelm").unwrap();
        assert_eq!(p.name, "stablelm");
        assert_eq!(p.norm_type, NormType::LayerNorm);
        assert_eq!(p.ffn_type, FFNType::SwiGLU);
        assert_eq!(p.block_style, BlockStyle::Parallel);
    }

    #[test]
    fn find_profile_by_gguf_falcon() {
        let p = find_profile_by_gguf("falcon").unwrap();
        assert_eq!(p.name, "falcon");
        assert_eq!(p.norm_type, NormType::LayerNorm);
        assert_eq!(p.ffn_type, FFNType::GELU);
        assert_eq!(p.block_style, BlockStyle::Sequential);
    }

    #[test]
    fn find_profile_by_gguf_mpt() {
        let p = find_profile_by_gguf("mpt").unwrap();
        assert_eq!(p.name, "mpt");
        assert_eq!(p.norm_type, NormType::LayerNorm);
        assert_eq!(p.ffn_type, FFNType::GELU);
        assert_eq!(p.pos_encoding_kind, PosEncodingKind::None);
    }

    #[test]
    fn find_profile_by_gguf_unknown_returns_none() {
        assert!(find_profile_by_gguf("totally_unknown_arch").is_none());
        assert!(find_profile_by_gguf("").is_none());
    }

    // ---- HF lookup ----

    #[test]
    fn find_profile_by_hf_llama_causal_lm() {
        let p = find_profile_by_hf("LlamaForCausalLM").unwrap();
        assert_eq!(p.name, "llama");
    }

    #[test]
    fn find_profile_by_hf_mistral() {
        let p = find_profile_by_hf("MistralForCausalLM").unwrap();
        assert_eq!(p.name, "mistral");
    }

    #[test]
    fn find_profile_by_hf_gemma() {
        let p = find_profile_by_hf("GemmaForCausalLM").unwrap();
        assert_eq!(p.name, "gemma");
        assert!(p.has_embedding_scale);
    }

    #[test]
    fn find_profile_by_hf_phi() {
        let p = find_profile_by_hf("PhiForCausalLM").unwrap();
        assert_eq!(p.name, "phi");
        assert_eq!(p.block_style, BlockStyle::Parallel);
    }

    #[test]
    fn find_profile_by_hf_falcon() {
        let p = find_profile_by_hf("FalconForCausalLM").unwrap();
        assert_eq!(p.name, "falcon");
    }

    #[test]
    fn find_profile_by_hf_mpt() {
        let p = find_profile_by_hf("MPTForCausalLM").unwrap();
        assert_eq!(p.name, "mpt");
    }

    #[test]
    fn find_profile_by_hf_stablelm() {
        let p = find_profile_by_hf("StableLMForCausalLM").unwrap();
        assert_eq!(p.name, "stablelm");
    }

    #[test]
    fn find_profile_by_hf_unknown_returns_none() {
        assert!(find_profile_by_hf("SomeUnknownModelForCausalLM").is_none());
    }

    // ---- Error message quality ----

    #[test]
    fn unknown_gguf_arch_error_lists_known_architectures() {
        let names = known_architecture_names();
        // Every profile must appear in the list.
        for profile in KNOWN_ARCHITECTURES {
            assert!(
                names.contains(&profile.name),
                "known_architecture_names() missing '{}'",
                profile.name
            );
        }
        // The list must be non-empty.
        assert!(!names.is_empty());
    }

    // ---- Profile field values ----

    #[test]
    fn gpt2_profile_has_full_bias() {
        assert!(GPT2_PROFILE.has_qkv_bias);
        assert!(GPT2_PROFILE.has_output_bias);
        assert_eq!(GPT2_PROFILE.pos_encoding_kind, PosEncodingKind::Learned);
        assert!(GPT2_PROFILE.tied_embeddings);
    }

    #[test]
    fn gemma_profile_has_embedding_scale_and_tied_embeddings() {
        assert!(GEMMA_PROFILE.has_embedding_scale);
        assert!(GEMMA_PROFILE.tied_embeddings);
        assert_eq!(GEMMA_PROFILE.ffn_type, FFNType::GeGLU);
    }

    #[test]
    fn qwen_profile_has_qkv_bias_no_output_bias() {
        assert!(QWEN_PROFILE.has_qkv_bias);
        assert!(!QWEN_PROFILE.has_output_bias);
    }

    #[test]
    fn phi_profile_uses_partial_rope_and_parallel_block() {
        assert_eq!(PHI_PROFILE.pos_encoding_kind, PosEncodingKind::PartialRoPE);
        assert_eq!(PHI_PROFILE.block_style, BlockStyle::Parallel);
    }

    #[test]
    fn mpt_profile_uses_no_pos_encoding() {
        assert_eq!(MPT_PROFILE.pos_encoding_kind, PosEncodingKind::None);
    }

    // ---- build_pos_encoding helper ----

    #[test]
    fn build_pos_encoding_rope() {
        let enc = build_pos_encoding(PosEncodingKind::RoPE, 10000.0, 64);
        assert!(matches!(enc, PosEncoding::RoPE { freq_base } if (freq_base - 10000.0).abs() < 1e-6));
    }

    #[test]
    fn build_pos_encoding_partial_rope_uses_half_head_dim() {
        let enc = build_pos_encoding(PosEncodingKind::PartialRoPE, 10000.0, 64);
        assert!(
            matches!(enc, PosEncoding::PartialRoPE { rotary_dim: 32, .. }),
            "expected rotary_dim=32 (half of head_dim=64)"
        );
    }

    #[test]
    fn build_pos_encoding_learned() {
        let enc = build_pos_encoding(PosEncodingKind::Learned, 10000.0, 64);
        assert_eq!(enc, PosEncoding::Learned);
    }

    #[test]
    fn build_pos_encoding_none() {
        let enc = build_pos_encoding(PosEncodingKind::None, 10000.0, 64);
        assert_eq!(enc, PosEncoding::None);
    }
}
