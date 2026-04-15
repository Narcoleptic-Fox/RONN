//! Transformer model architecture for NNX.
//!
//! Supports multiple model architectures via a data-driven profile registry.
//! Each architecture can use different normalization (RMSNorm/LayerNorm),
//! FFN variants (SwiGLU/GeGLU/GELU), position encoding (RoPE/partial RoPE/learned),
//! and block styles (sequential/parallel).
//!
//! Adding a new architecture means adding a new [`config::ArchitectureProfile`]
//! constant — no dispatch code changes required.
//!
//! ## Architecture
//!
//! ```text
//! Token ID -> Embedding Lookup -> [hidden_dim]
//!     |
//!     +--- Block 0: Norm -> Attention (GQA/MHA + RoPE/learned + KV Cache) -> Residual
//!     |                -> Norm -> FFN (SwiGLU/GeGLU/GELU) -> Residual
//!     +--- Block 1: ...
//!     +--- ...
//!     +--- Block N: ...
//!     |
//!     -> Final Norm -> LM Head (linear -> vocab) -> Logits
//!     -> Sampler (temperature, top-k, top-p) -> Token ID
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use nnx_transformer::backend::NnxBackend;
//! use nnx_core::engine::{InferenceEngine, LoadConfig};
//! use std::path::Path;
//!
//! let backend = NnxBackend::new();
//! let handle = backend.load_model(Path::new("model.gguf"), &LoadConfig::default()).unwrap();
//! let info = backend.model_info_cloned(handle).unwrap();
//! println!("Loaded: {} — {} layers", info.architecture, info.num_layers);
//! ```

pub mod attention;
pub mod backend;
pub mod block;
pub mod cache;
pub mod config;
pub mod ffn;
pub mod generate;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod loader;
pub mod model;
pub mod sampler;
pub mod tokenizer;
pub mod weight_names;
pub mod weights;

pub use backend::NnxBackend;
pub use cache::KVCache;
pub use config::{
    Architecture, ArchitectureProfile, BlockStyle, FFNType, KNOWN_ARCHITECTURES, ModelConfig,
    NormType, PosEncoding, PosEncodingKind, find_profile_by_gguf, find_profile_by_hf,
    known_architecture_names,
};
pub use generate::{GenerateConfig, GenerateOutput, StopReason, generate};
pub use model::{Model, ModelWeights};
pub use sampler::SamplerConfig;
pub use tokenizer::Tokenizer;
pub use weights::Matrix;
