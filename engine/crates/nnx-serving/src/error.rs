//! Error types for the serving layer.

use thiserror::Error;

/// Errors specific to the serving infrastructure.
#[derive(Error, Debug)]
pub enum ServingError {
    /// No free pages available in the block allocator.
    #[error("out of pages: {0}")]
    OutOfPages(String),

    /// Page reference count error (double-free, overflow).
    #[error("ref count error: {0}")]
    RefCount(String),

    /// Copy-on-write failed.
    #[error("copy-on-write error: {0}")]
    CopyOnWrite(String),

    /// Invalid page ID.
    #[error("invalid page: {0}")]
    InvalidPage(String),

    /// Scheduler error (admission, preemption, state machine violation).
    #[error("scheduler error: {0}")]
    Scheduler(String),

    /// Underlying engine error.
    #[error("engine error: {0}")]
    Engine(#[from] nnx_core::EngineError),
}

/// Convenience result type for serving operations.
pub type Result<T> = std::result::Result<T, ServingError>;
