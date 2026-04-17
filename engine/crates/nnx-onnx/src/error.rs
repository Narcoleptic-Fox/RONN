//! Error types for engine-side ONNX parsing and model loading.

use thiserror::Error;

/// Convenience result type for `nnx-onnx`.
pub type Result<T> = std::result::Result<T, OnnxError>;

/// Errors returned while parsing ONNX protobuf payloads or building NNX models.
#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("protobuf decode error: {0}")]
    Decode(#[from] prost::DecodeError),

    #[error("model has no graph")]
    MissingGraph,

    #[error("initializer is missing a name")]
    UnnamedInitializer,

    #[error("missing initializer '{0}'")]
    MissingInitializer(String),

    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDType(String),

    #[error("external tensor data is not supported for '{0}'")]
    ExternalDataUnsupported(String),

    #[error("invalid shape for '{name}': {shape:?}")]
    InvalidShape { name: String, shape: Vec<i64> },

    #[error("shape mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("invalid tensor data for '{name}': {reason}")]
    InvalidTensorData { name: String, reason: String },

    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
}
