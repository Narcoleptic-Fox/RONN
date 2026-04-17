//! ONNX model loading for NNX.
//!
//! This crate provides a narrow engine-side ONNX capability:
//! - decode ONNX protobuf models with `prost`
//! - inspect dense initializers and tensor metadata
//! - decode supported dense tensors into `f32`
//! - construct dense `nnx-transformer` Llama-family models from ONNX initializers

pub mod error;
pub mod loader;
pub mod proto;

pub use error::{OnnxError, Result};
pub use loader::{
    Initializer, OnnxParser, ParsedModel, TensorDataEncoding, TensorElementType, TensorMetadata,
};
