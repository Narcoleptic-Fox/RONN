//! NNX Core — foundational types and traits for the NNX inference engine.
//!
//! This crate defines the vocabulary shared across all NNX crates:
//! tensor representation, data types, device abstraction, and the
//! [`InferenceEngine`] trait that orchestration layers build upon.

pub mod backend;
pub mod device;
pub mod dtype;
pub mod engine;
pub mod error;
pub mod gpu_config;
pub mod shape;
pub mod tensor;

pub use backend::KernelBackend;
pub use device::{Device, DeviceId};
pub use dtype::DType;
pub use engine::{
    GenerationOutput, InferenceEngine, KVCacheAccess, KVStore, LoadConfig, ModelHandle, ModelInfo,
    RequestHandle, TokenBatch,
};
pub use error::{EngineError, Result};
pub use gpu_config::{GpuBlockStyle, GpuConfig, GpuFFNType, GpuNormType, GpuPosEncoding, PageId};
pub use shape::Shape;
pub use tensor::{Tensor, TensorView};
