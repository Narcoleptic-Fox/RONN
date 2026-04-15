//! NNX CubeCL — GPU compute kernels for the NNX inference engine.
//!
//! This crate provides GPU implementations of the hot-path operations
//! used during transformer inference:
//!
//! - Matrix-vector multiply (matvec) — the dominant cost in each layer
//! - RMS normalization — used between every transformer block
//! - Softmax — applied to attention scores
//! - RoPE — rotary position encoding applied to Q/K
//! - Activations — SiLU, GELU, elementwise mul/add for FFN
//!
//! All kernels are written using CubeCL's `#[cube]` macro, making them
//! portable across CUDA, ROCm, Metal, Vulkan, and WebGPU backends.
//!
//! # Backend Selection
//!
//! Kernels are generic over `R: cubecl::prelude::Runtime`. The caller
//! chooses the backend at the call site:
//!
//! ```ignore
//! // CUDA
//! launch_rms_norm::<cubecl::cuda::CudaRuntime>(&client, ...);
//! // WebGPU (Metal/Vulkan/DX12)
//! launch_rms_norm::<cubecl::wgpu::WgpuRuntime>(&client, ...);
//! ```

pub mod activations;
pub mod attention;
pub mod backend;
pub mod inference;
pub mod matmul;
pub mod normalization;
pub mod paged_kv;
pub mod quantized;
pub mod rope;
pub mod softmax;

pub use backend::{CubeclBackend, GpuBuffer};
pub use inference::{
    GpuInference, GpuLayerCache, GpuLayerWeights, GpuModelWeights, RawLayerWeights,
};
pub use paged_kv::{GpuPagePool, GpuPhysicalPage};

// Re-export the wgpu runtime so callers that use the `gpu` feature of
// nnx-transformer can refer to `nnx_cubecl::WgpuRuntime` without needing
// a direct dependency on the `cubecl` crate.
#[cfg(feature = "wgpu")]
pub use cubecl::wgpu::WgpuRuntime;
