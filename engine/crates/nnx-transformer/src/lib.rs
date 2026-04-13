//! Transformer model architecture for NNX.
//!
//! Implements Llama-family transformer inference: GQA attention with RoPE,
//! SwiGLU FFN, RMSNorm, KV caching, and autoregressive generation.
//!
//! ## Architecture
//!
//! ```text
//! Token ID → Embedding Lookup → [hidden_dim]
//!     │
//!     ├─── Block 0: RMSNorm → Attention (GQA + RoPE + KV Cache) → Residual
//!     │                → RMSNorm → SwiGLU FFN → Residual
//!     ├─── Block 1: ...
//!     ├─── ...
//!     ├─── Block N: ...
//!     │
//!     → Final RMSNorm → LM Head (linear → vocab) → Logits
//!     → Sampler (temperature, top-k, top-p) → Token ID
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use nnx_transformer::{loader, generate, GenerateConfig, SamplerConfig};
//!
//! let mut model = loader::load_gguf(std::path::Path::new("model.gguf")).unwrap();
//! let config = GenerateConfig {
//!     max_tokens: 100,
//!     sampler: SamplerConfig::greedy(),
//!     ..Default::default()
//! };
//! let result = generate(&mut model, &[1], &config); // [1] = BOS token
//! println!("Generated {} tokens", result.tokens.len());
//! ```

pub mod attention;
pub mod block;
pub mod cache;
pub mod config;
pub mod ffn;
pub mod generate;
pub mod loader;
pub mod model;
pub mod sampler;
pub mod tokenizer;

pub use cache::KVCache;
pub use config::ModelConfig;
pub use generate::{GenerateConfig, GenerateOutput, StopReason, generate};
pub use model::{Model, ModelWeights};
pub use sampler::SamplerConfig;
pub use tokenizer::Tokenizer;
