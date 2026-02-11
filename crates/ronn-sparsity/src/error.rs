//! Error types for the ronn-sparsity crate.

use thiserror::Error;

/// Result type for sparsity operations.
pub type Result<T> = std::result::Result<T, SparsityError>;

/// Errors that can occur in the sparsity system.
#[derive(Error, Debug)]
pub enum SparsityError {
    /// Error during activation profiling.
    #[error("Profiling error: {0}")]
    Profiling(String),

    /// Error during neuron classification.
    #[error("Classification error: {0}")]
    Classification(String),

    /// Error during activation prediction.
    #[error("Prediction error: {0}")]
    Prediction(String),

    /// Error during predictor training.
    #[error("Training error: {0}")]
    Training(String),

    /// Error during sparse computation.
    #[error("Sparse operation error: {0}")]
    SparseOp(String),

    /// Error during scheduling/routing.
    #[error("Scheduling error: {0}")]
    Scheduling(String),

    /// Error during format reading/writing.
    #[error("Format error: {0}")]
    Format(String),

    /// Error from underlying core operations.
    #[error("Core error: {0}")]
    Core(#[from] ronn_core::error::CoreError),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
