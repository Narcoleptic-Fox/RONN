//! CPU execution provider module.
//!
//! This module provides CPU-based execution with SIMD optimizations,
//! multi-threading support, and NUMA awareness.

pub mod allocator;
pub mod kernels;
pub mod provider;
pub mod simd;

pub use allocator::CpuMemoryAllocator;
pub use kernels::CpuKernel;
pub use provider::{
    CpuExecutionProvider, CpuProviderConfig, create_cpu_provider, create_cpu_provider_with_config,
    create_numa_cpu_provider,
};
pub use simd::{SimdCapabilities, detect_simd_capabilities};
