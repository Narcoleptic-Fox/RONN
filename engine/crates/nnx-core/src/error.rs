//! Error types for the NNX engine.

use thiserror::Error;

/// Convenience result type for NNX operations.
pub type Result<T> = std::result::Result<T, EngineError>;

/// Top-level error type for the NNX inference engine.
#[derive(Error, Debug)]
pub enum EngineError {
    /// Model file could not be loaded or parsed.
    #[error("model load error: {0}")]
    ModelLoad(String),

    /// Unsupported model format or architecture.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Tensor shape mismatch during computation.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Data type mismatch or unsupported conversion.
    #[error("dtype error: {0}")]
    DType(String),

    /// Quantization or dequantization error.
    #[error("quantization error: {0}")]
    Quantization(String),

    /// Compute kernel error.
    #[error("kernel error: {0}")]
    Kernel(String),

    /// KV cache error (capacity, corruption, etc.).
    #[error("cache error: {0}")]
    Cache(String),

    /// Device or memory allocation error.
    #[error("device error: {0}")]
    Device(String),

    /// Generation error (sampling, decoding).
    #[error("generation error: {0}")]
    Generation(String),

    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
