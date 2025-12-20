//! RONN Execution Provider Framework
//!
//! This crate provides the execution provider framework for RONN, including:
//! - Provider registry and management system
//! - Memory allocator implementations with pooling and SIMD alignment
//! - CPU execution provider with SIMD optimizations and multi-threading
//! - GPU execution provider using Candle backend
//! - Kernel compilation framework with operator fusion
//!
//! ## Architecture
//!
//! The provider framework follows a layered architecture:
//! - **Registry**: Central provider management and selection
//! - **Allocators**: Memory management with different strategies
//! - **Providers**: Hardware-specific execution implementations
//! - **Compiler**: Subgraph optimization and kernel compilation
//!
//! ## Example
//!
//! ```rust
//! use ronn_providers::{
//!     ProviderRegistry, create_cpu_provider, create_gpu_provider,
//!     KernelCompiler, FusionConfig, MemoryConfig
//! };
//! use ronn_core::{SubGraph, GraphNode};
//! use std::sync::Arc;
//!
//! // Create provider registry
//! let registry = ProviderRegistry::new();
//!
//! // Register CPU provider
//! let cpu_provider = create_cpu_provider()?;
//! registry.register_provider(cpu_provider)?;
//!
//! // Try to register GPU provider (may fail if no GPU)
//! if let Ok(gpu_provider) = create_gpu_provider() {
//!     registry.register_provider(gpu_provider)?;
//! }
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![warn(unsafe_code)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

pub mod allocator;
pub mod compiler;
pub mod cpu;
pub mod gpu;
pub mod registry;

// Specialized execution providers
#[cfg(feature = "bitnet")]
pub mod bitnet;
#[cfg(feature = "custom-hardware")]
pub mod custom;
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export commonly used types and functions
pub use allocator::{
    AlignedMemoryAllocator, PoolConfig, PooledMemoryAllocator, SystemMemoryAllocator,
    calculate_tensor_size, get_alignment_requirement, get_simd_alignment,
};
pub use compiler::{
    CompilationResult, CompilationStats, FusionConfig, FusionType, KernelCompiler, MemoryConfig,
    MemoryPlan, TensorInfo,
};
pub use cpu::{
    CpuExecutionProvider, CpuKernel, CpuMemoryAllocator, SimdCapabilities, create_cpu_provider,
    create_cpu_provider_with_config, create_numa_cpu_provider, detect_simd_capabilities,
};
pub use gpu::{
    BandwidthOptimizedPlacement, CudaCompileOptions, CudaKernelManager, GpuExecutionProvider,
    GpuMemoryAllocator, GpuTopology, GpuTopologyManager, LocalityAwarePlacement,
    MultiGpuMemoryConfig, MultiGpuMemoryManager, PlacementPlan, PlacementStrategy,
    PowerEfficientPlacement, SyncStrategy, TopologyConfig, Workload, WorkloadType,
    create_gpu_provider, create_gpu_provider_with_config,
};
pub use registry::{ProviderRegistry, RegistryStatistics};

// Re-export core types that providers use
pub use ronn_core::{ExecutionProvider, ProviderType};

// Specialized provider re-exports
#[cfg(feature = "bitnet")]
pub use bitnet::{
    BinaryTensor, BitNetExecutionProvider, BitNetKernel, BitNetOperation, BitNetProviderConfig,
    BitNetQuantizer, QuantizationMethod, TernaryTensor, create_bitnet_provider,
};
#[cfg(feature = "custom-hardware")]
pub use custom::{
    CustomHardwareProvider, CustomProviderRegistry, HardwareCapability, NpuConfig, NpuProvider,
    TpuConfig, TpuProvider, create_npu_provider, create_tpu_provider,
};
#[cfg(feature = "wasm")]
pub use wasm::{WasmBridge, WasmExecutionProvider, WasmProviderConfig, create_wasm_provider};

/// Result type alias for provider operations.
pub type Result<T> = anyhow::Result<T>;

/// Create and configure a complete provider system with CPU and optional GPU.
pub fn create_provider_system() -> Result<ProviderRegistry> {
    let registry = ProviderRegistry::new();

    // Always register CPU provider
    let cpu_provider = create_cpu_provider()?;
    registry.register_provider(cpu_provider)?;

    // Try to register GPU provider if available
    match create_gpu_provider() {
        Ok(gpu_provider) => {
            registry.register_provider(gpu_provider)?;
            tracing::info!("Registered both CPU and GPU providers");
        }
        Err(e) => {
            tracing::info!("GPU provider not available: {}, using CPU only", e);
        }
    }

    Ok(registry)
}

/// Create a CPU-only provider system.
pub fn create_cpu_only_system() -> Result<ProviderRegistry> {
    let registry = ProviderRegistry::new();
    let cpu_provider = create_cpu_provider()?;
    registry.register_provider(cpu_provider)?;
    tracing::info!("Registered CPU-only provider system");
    Ok(registry)
}

/// Create a comprehensive provider system with all available providers.
///
/// This includes CPU, GPU (if available), and all specialized providers
/// (BitNet, WebAssembly, Custom Hardware) based on enabled features.
pub fn create_comprehensive_provider_system() -> Result<ProviderRegistry> {
    let registry = create_provider_system()?;

    // Register BitNet provider if feature is enabled
    #[cfg(feature = "bitnet")]
    {
        match create_bitnet_provider() {
            Ok(bitnet_provider) => {
                registry.register_provider(bitnet_provider)?;
                tracing::info!("Registered BitNet provider for 1-bit quantized models");
            }
            Err(e) => {
                tracing::warn!("BitNet provider registration failed: {}", e);
            }
        }
    }

    // Register WebAssembly provider if feature is enabled
    #[cfg(feature = "wasm")]
    {
        match create_wasm_provider() {
            Ok(wasm_provider) => {
                registry.register_provider(wasm_provider)?;
                tracing::info!("Registered WebAssembly provider for browser deployment");
            }
            Err(e) => {
                tracing::warn!("WebAssembly provider registration failed: {}", e);
            }
        }
    }

    // Register custom hardware providers if feature is enabled
    #[cfg(feature = "custom-hardware")]
    {
        // Register NPU provider
        match create_npu_provider() {
            Ok(npu_provider) => {
                registry.register_provider(npu_provider)?;
                tracing::info!("Registered NPU provider");
            }
            Err(e) => {
                tracing::debug!("NPU provider registration failed: {}", e);
            }
        }

        // Register TPU provider
        match create_tpu_provider() {
            Ok(tpu_provider) => {
                registry.register_provider(tpu_provider)?;
                tracing::info!("Registered TPU provider");
            }
            Err(e) => {
                tracing::debug!("TPU provider registration failed: {}", e);
            }
        }
    }

    Ok(registry)
}

/// Create a kernel compiler with performance-optimized settings.
pub fn create_performance_compiler() -> KernelCompiler {
    let fusion_config = FusionConfig {
        enable_fusion: true,
        max_fusion_depth: 6,
        enable_elementwise_fusion: true,
        enable_conv_fusion: true,
        enable_matmul_fusion: true,
    };

    let memory_config = MemoryConfig {
        enable_optimization: true,
        prefer_row_major: true,
        enable_tensor_reuse: true,
        max_memory_overhead: 0.3, // Allow 30% overhead for better performance
    };

    KernelCompiler::with_config(fusion_config, memory_config)
}

/// Create a kernel compiler with memory-optimized settings.
pub fn create_memory_optimized_compiler() -> KernelCompiler {
    let fusion_config = FusionConfig {
        enable_fusion: true,
        max_fusion_depth: 3,
        enable_elementwise_fusion: true,
        enable_conv_fusion: false, // Reduce memory usage
        enable_matmul_fusion: true,
    };

    let memory_config = MemoryConfig {
        enable_optimization: true,
        prefer_row_major: true,
        enable_tensor_reuse: true,
        max_memory_overhead: 0.1, // Minimize memory overhead
    };

    KernelCompiler::with_config(fusion_config, memory_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::{DataType, GraphNode, SubGraph, Tensor, TensorAllocator, TensorLayout};
    use std::collections::HashMap;

    #[test]
    fn test_provider_system_creation() -> Result<()> {
        let registry = create_provider_system()?;
        let stats = registry.get_statistics();

        // Should have at least CPU provider
        assert!(stats.provider_count >= 1);
        assert!(stats.total_supported_ops > 0);
        assert!(!stats.preference_order.is_empty());

        Ok(())
    }

    #[test]
    fn test_cpu_only_system() -> Result<()> {
        let registry = create_cpu_only_system()?;
        let stats = registry.get_statistics();

        assert_eq!(stats.provider_count, 1);
        assert_eq!(stats.preference_order.len(), 1);
        assert_eq!(stats.preference_order[0], ronn_core::ProviderId::CPU);

        Ok(())
    }

    #[test]
    fn test_kernel_compiler_variants() -> Result<()> {
        let perf_compiler = create_performance_compiler();
        let memory_compiler = create_memory_optimized_compiler();

        // Create a simple test subgraph
        let subgraph = SubGraph {
            nodes: vec![GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["input1".to_string(), "input2".to_string()],
                outputs: vec!["temp1".to_string()],
                name: Some("test_add".to_string()),
            }],
            edges: vec![],
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["temp1".to_string()],
        };

        // Both compilers should be able to compile the subgraph
        let perf_result = perf_compiler.compile(&subgraph)?;
        let memory_result = memory_compiler.compile(&subgraph)?;

        assert!(perf_result.fused_ops.len() > 0);
        assert!(memory_result.fused_ops.len() > 0);

        Ok(())
    }

    #[test]
    fn test_end_to_end_execution() -> Result<()> {
        // Create provider system
        let registry = create_cpu_only_system()?;

        // Create test subgraph
        let subgraph = SubGraph {
            nodes: vec![GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["input1".to_string(), "input2".to_string()],
                outputs: vec!["output1".to_string()],
                name: Some("test_add".to_string()),
            }],
            edges: vec![],
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
        };

        // Compile subgraph
        let (provider_id, kernel) = registry.compile_subgraph(subgraph)?;
        assert_eq!(provider_id, ronn_core::ProviderId::CPU);

        // Create test inputs
        let input1 = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;
        let input2 = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;
        let inputs = vec![input1, input2];

        // Execute kernel
        let outputs = kernel.execute(&inputs)?;
        assert!(!outputs.is_empty());

        // Check performance stats
        let stats = kernel.get_performance_stats();
        assert_eq!(stats.execution_count, 1);

        Ok(())
    }

    #[test]
    fn test_allocator_integration() -> Result<()> {
        let registry = create_cpu_only_system()?;
        let cpu_provider = registry
            .get_provider(ronn_core::ProviderId::CPU)
            .expect("CPU provider should exist");

        let allocator = cpu_provider.get_allocator();

        // Test allocation
        let buffer = allocator.allocate(&[100], DataType::F32)?;
        assert_eq!(buffer.size, 400); // 100 * 4 bytes
        assert_eq!(buffer.memory_type, ronn_core::MemoryType::SystemRAM);

        // Check memory info
        let memory_info = allocator.get_memory_info();
        assert!(memory_info.allocated_bytes > 0);

        // Deallocate
        allocator.deallocate(buffer)?;

        Ok(())
    }

    #[test]
    fn test_simd_detection() {
        let capabilities = detect_simd_capabilities();

        // Should detect at least basic capabilities on most systems
        #[cfg(target_arch = "x86_64")]
        {
            assert!(capabilities.sse2);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // On ARM64, FMA should be available
            assert!(capabilities.fma);
        }

        println!("Detected SIMD capabilities: {:?}", capabilities);
    }

    #[test]
    fn test_memory_pooling() -> Result<()> {
        let config = PoolConfig {
            max_buffers_per_bucket: 4,
            max_pool_size: 1024 * 1024, // 1MB
            bucket_granularity: 64,
        };

        let allocator = PooledMemoryAllocator::new(config);

        // Allocate and deallocate to test pooling
        let buffer1 = allocator.allocate(&[64], DataType::F32)?; // 256 bytes
        allocator.deallocate(buffer1)?;

        let buffer2 = allocator.allocate(&[64], DataType::F32)?; // Should reuse
        allocator.deallocate(buffer2)?;

        let hit_rate = allocator.get_hit_rate();
        assert!(hit_rate >= 0.0 && hit_rate <= 1.0);

        Ok(())
    }

    #[test]
    fn test_provider_preference_order() -> Result<()> {
        let registry = create_provider_system()?;

        // Get initial preference order
        let initial_order = registry.get_preference_order();
        assert!(!initial_order.is_empty());

        // Try to set custom order (CPU first)
        let custom_order = vec![ronn_core::ProviderId::CPU];
        registry.set_preference_order(custom_order.clone())?;

        let updated_order = registry.get_preference_order();
        assert_eq!(updated_order, custom_order);

        Ok(())
    }
}
