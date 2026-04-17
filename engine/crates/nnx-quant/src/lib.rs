//! Quantization block types and dequantization kernels.
//!
//! Supports the GGML quantization formats used by GGUF and GGML files.
//! Each block type defines a packed representation and a dequantization
//! function that produces f32 output.

pub mod blocks;
pub mod dequant;
pub mod encode;
pub mod types;

pub use types::GGMLType;
