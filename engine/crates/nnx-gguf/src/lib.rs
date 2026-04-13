//! GGUF file format parser with memory-mapped loading.
//!
//! GGUF is the standard format for local LLM inference (llama.cpp ecosystem).
//! A single file contains model weights, tokenizer, architecture metadata,
//! and quantization parameters.
//!
//! ## File Layout
//!
//! ```text
//! ┌─────────────────────────┐
//! │ Header (magic, version) │
//! ├─────────────────────────┤
//! │ Metadata key-value pairs│
//! ├─────────────────────────┤
//! │ Tensor descriptors      │
//! ├─────────────────────────┤
//! │ Tensor data (aligned)   │
//! └─────────────────────────┘
//! ```

pub mod metadata;
pub mod parser;
pub mod types;

pub use metadata::GGUFMetadata;
pub use parser::GGUFFile;
pub use types::GGUFValueType;
