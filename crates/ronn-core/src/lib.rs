//! RONN Core Runtime Engine
//!
//! This crate provides the foundational components of the RONN (Rust ONNX Neural Network)
//! runtime, including tensor operations, model graph representation, and core execution
//! interfaces.
//!
//! ## Architecture
//!
//! The core engine follows a layered architecture:
//! - **Types**: Fundamental data structures for tensors, graphs, and metadata
//! - **Session**: Management of inference sessions and resource isolation
//! - **Tensor**: Multi-dimensional array operations with Candle integration
//! - **Graph**: Model representation and manipulation utilities
//!
//! ## Example
//!
//! ```rust
//! use ronn_core::{Tensor, DataType, TensorLayout};
//!
//! // Create a 2x3 tensor with zeros
//! let tensor = Tensor::zeros(vec![2, 3], DataType::F32, TensorLayout::RowMajor)?;
//! assert_eq!(tensor.shape(), vec![2, 3]);
//! assert_eq!(tensor.numel(), 6);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

/// Error types for core operations
pub mod error;
pub mod graph;
pub mod logging;
pub mod memory_pool;
pub mod ops;
pub mod profiling;
pub mod session;
pub mod simd;
pub mod tensor;
pub mod types;

// Re-export commonly used types
pub use error::{CoreError, Result};
pub use graph::{GraphBuilder, GraphStatistics};
pub use memory_pool::{global_pool, MemoryPool, PoolConfig, PoolStats, PooledBuffer};
pub use ops::{ArithmeticOps, MatrixOps, ReductionOps, ShapeOps};
pub use profiling::{
    global_profiler, init_profiler, CategoryStats, OperationStats, ProfileConfig, ProfileEvent,
    ProfileReport, Profiler,
};
pub use session::{
    GlobalStatistics, InferenceSession, SessionConfig, SessionManager, SessionStatistics,
};
pub use simd::{simd_features, SimdFeatures, SimdLevel};
pub use tensor::Tensor;
pub use types::{
    AttributeValue, CompiledKernel, DataType, ExecutionProvider, GraphEdge, GraphNode, KernelStats,
    MemoryInfo, MemoryType, MemoryUsage, ModelGraph, NodeAttribute, NodeId, OperatorSpec,
    OptimizationLevel, PerformanceProfile, ProviderCapability, ProviderConfig, ProviderId,
    ProviderType, ResourceRequirements, SessionId, SubGraph, TensorAllocator, TensorBuffer,
    TensorLayout,
};
