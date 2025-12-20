//! GPU execution provider module using Candle backend.
//!
//! This module provides GPU-accelerated execution using the Candle library
//! for CUDA and Metal backends.

pub mod allocator;
pub mod cuda_kernels;
pub mod memory_manager;
pub mod provider;
pub mod topology;

pub use allocator::GpuMemoryAllocator;
pub use cuda_kernels::{
    CompiledCudaKernel, CudaCompileOptions, CudaKernelManager, KernelLaunchConfig,
};
pub use memory_manager::{MultiGpuMemoryConfig, MultiGpuMemoryManager, SyncStrategy};
pub use provider::{GpuExecutionProvider, create_gpu_provider, create_gpu_provider_with_config};
pub use topology::{
    BandwidthOptimizedPlacement, GpuTopology, GpuTopologyManager, LocalityAwarePlacement,
    PlacementPlan, PlacementStrategy, PowerEfficientPlacement, TopologyConfig, Workload,
    WorkloadType,
};
