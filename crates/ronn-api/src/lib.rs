// High-level API for RONN runtime
//
// Provides a simple, ergonomic interface for loading models and running inference.

pub mod async_session;
pub mod batch;
mod error;
mod model;
mod session;

pub use async_session::{AsyncBatchProcessor, AsyncSession};
pub use batch::{BatchConfig, BatchProcessor, BatchRequest, BatchStats, BatchStrategy};
pub use error::{Error, Result};
pub use model::Model;
pub use session::{InferenceSession, SessionBuilder, SessionOptions};

// Re-export commonly used types
pub use ronn_core::tensor::Tensor;
pub use ronn_core::types::DataType;
pub use ronn_graph::OptimizationLevel;
pub use ronn_providers::{ExecutionProvider, ProviderType};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{Model, SessionBuilder, SessionOptions, Tensor};
    pub use ronn_graph::OptimizationLevel;
    pub use ronn_providers::ProviderType;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_options() {
        let options = SessionOptions::default();
        assert_eq!(options.optimization_level(), OptimizationLevel::O2);
    }
}
