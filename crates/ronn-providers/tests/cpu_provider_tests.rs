//! Comprehensive tests for the CPU execution provider.
//!
//! This module tests the CPU provider's functionality including:
//! - SIMD capability detection and optimization
//! - NUMA-aware memory allocation
//! - Multi-threading with Rayon
//! - Operator support and execution
//! - Configuration and customization

use anyhow::Result;
use ronn_core::{
    DataType, ExecutionProvider, GraphNode, MemoryType, OperatorSpec, PerformanceProfile,
    ProviderConfig, ProviderId, SubGraph, Tensor, TensorLayout,
};
use ronn_providers::{
    cpu::{CpuExecutionProvider, CpuProviderConfig, detect_simd_capabilities},
    create_cpu_provider, create_cpu_provider_with_config, create_numa_cpu_provider,
};
use std::collections::HashMap;

// ============================================================================
// Provider Creation Tests
// ============================================================================

#[test]
fn test_cpu_provider_creation() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    assert_eq!(provider.provider_id(), ProviderId::CPU);

    let capability = provider.get_capability();
    assert_eq!(capability.performance_profile, PerformanceProfile::CPU);
    assert!(!capability.supported_ops.is_empty());

    Ok(())
}

#[test]
fn test_cpu_provider_with_config() -> Result<()> {
    let config = CpuProviderConfig {
        thread_count: Some(2),
        memory_limit: Some(128 * 1024 * 1024), // 128MB
        numa_node: 0,
        enable_simd: false,
        enable_fusion: true,
        thread_pool_name: "test-pool".to_string(),
    };

    let provider = CpuExecutionProvider::with_config(config)?;

    assert_eq!(provider.get_thread_pool().current_num_threads(), 2);
    assert_eq!(provider.get_config().numa_node, 0);
    assert!(!provider.get_config().enable_simd);
    assert!(provider.get_config().enable_fusion);

    Ok(())
}

#[test]
fn test_cpu_provider_default_config() -> Result<()> {
    let config = CpuProviderConfig::default();

    assert!(config.thread_count.is_none()); // Auto-detect
    assert!(config.memory_limit.is_none()); // No limit
    assert_eq!(config.numa_node, -1); // No preference
    assert!(config.enable_simd); // Enabled by default
    assert!(config.enable_fusion); // Enabled by default

    Ok(())
}

#[test]
fn test_cpu_provider_factory_functions() -> Result<()> {
    // Test default provider
    let provider1 = create_cpu_provider()?;
    assert_eq!(provider1.provider_id(), ProviderId::CPU);

    // Test with custom config
    let config = CpuProviderConfig {
        thread_count: Some(1),
        ..Default::default()
    };
    let provider2 = create_cpu_provider_with_config(config)?;
    assert_eq!(provider2.provider_id(), ProviderId::CPU);

    // Test NUMA-aware provider
    let provider3 = create_numa_cpu_provider(0)?;
    assert_eq!(provider3.provider_id(), ProviderId::CPU);

    Ok(())
}

// ============================================================================
// SIMD Capability Tests
// ============================================================================

#[test]
fn test_simd_capability_detection() {
    let capabilities = detect_simd_capabilities();

    // On x86_64, SSE2 should always be available
    #[cfg(target_arch = "x86_64")]
    {
        assert!(capabilities.sse2);
    }

    // On ARM64, FMA should typically be available
    #[cfg(target_arch = "aarch64")]
    {
        // ARM64 typically has FMA support
        assert!(capabilities.fma);
    }

    // Print detected capabilities for visibility
    println!("SIMD Capabilities:");
    println!("  SSE2: {}", capabilities.sse2);
    println!("  SSE4.1: {}", capabilities.sse41);
    println!("  AVX: {}", capabilities.avx);
    println!("  AVX2: {}", capabilities.avx2);
    println!("  AVX-512F: {}", capabilities.avx512f);
    println!("  FMA: {}", capabilities.fma);
}

#[test]
fn test_simd_in_capability_report() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;
    let capability = provider.get_capability();

    // Check that SIMD features are reported
    let resource_req = capability.resource_requirements;
    assert!(!resource_req.cpu_features.is_empty());

    println!("Reported CPU features: {:?}", resource_req.cpu_features);

    Ok(())
}

#[test]
fn test_simd_disabled() -> Result<()> {
    let config = CpuProviderConfig {
        enable_simd: false,
        ..Default::default()
    };

    let provider = CpuExecutionProvider::with_config(config)?;
    let simd_caps = provider.get_simd_capabilities();

    // When SIMD is disabled, all capabilities should be false
    assert!(!simd_caps.sse2);
    assert!(!simd_caps.sse41);
    assert!(!simd_caps.avx);
    assert!(!simd_caps.avx2);
    assert!(!simd_caps.avx512f);
    assert!(!simd_caps.fma);

    Ok(())
}

// ============================================================================
// Thread Pool Tests
// ============================================================================

#[test]
fn test_thread_pool_creation() -> Result<()> {
    let config = CpuProviderConfig {
        thread_count: Some(4),
        ..Default::default()
    };

    let provider = CpuExecutionProvider::with_config(config)?;
    let thread_pool = provider.get_thread_pool();

    assert_eq!(thread_pool.current_num_threads(), 4);

    Ok(())
}

#[test]
fn test_thread_pool_auto_detection() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;
    let thread_pool = provider.get_thread_pool();

    // Should have at least 1 thread
    assert!(thread_pool.current_num_threads() >= 1);

    // Should leave at least one core for system
    let system_cores = num_cpus::get();
    assert!(thread_pool.current_num_threads() < system_cores || system_cores == 1);

    println!(
        "Auto-detected {} threads from {} CPU cores",
        thread_pool.current_num_threads(),
        system_cores
    );

    Ok(())
}

#[test]
fn test_thread_pool_custom_name() -> Result<()> {
    let config = CpuProviderConfig {
        thread_pool_name: "custom-test-pool".to_string(),
        ..Default::default()
    };

    let provider = CpuExecutionProvider::with_config(config)?;

    // Provider should be created successfully with custom name
    assert_eq!(provider.get_config().thread_pool_name, "custom-test-pool");

    Ok(())
}

// ============================================================================
// Operation Support Tests
// ============================================================================

#[test]
fn test_basic_operation_support() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    // Test arithmetic operations
    assert!(provider.supports_operation("Add"));
    assert!(provider.supports_operation("Sub"));
    assert!(provider.supports_operation("Mul"));
    assert!(provider.supports_operation("Div"));

    // Test matrix operations
    assert!(provider.supports_operation("MatMul"));
    assert!(provider.supports_operation("Gemm"));

    // Test shape operations
    assert!(provider.supports_operation("Reshape"));
    assert!(provider.supports_operation("Transpose"));
    assert!(provider.supports_operation("Flatten"));

    // Test activation functions
    assert!(provider.supports_operation("ReLU"));
    assert!(provider.supports_operation("Sigmoid"));
    assert!(provider.supports_operation("Tanh"));
    assert!(provider.supports_operation("Softmax"));

    Ok(())
}

#[test]
fn test_unsupported_operation() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    assert!(!provider.supports_operation("NonexistentOp"));
    assert!(!provider.supports_operation("FutureFeature"));

    Ok(())
}

#[test]
fn test_can_handle_batch() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let ops = vec![
        OperatorSpec {
            op_type: "Add".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
        OperatorSpec {
            op_type: "MatMul".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
        OperatorSpec {
            op_type: "ReLU".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
    ];

    let results = provider.can_handle(&ops);
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|&supported| supported));

    Ok(())
}

#[test]
fn test_operation_count() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;
    let capability = provider.get_capability();

    // CPU provider should support a reasonable number of operations
    assert!(capability.supported_ops.len() >= 20);

    println!(
        "CPU provider supports {} operations",
        capability.supported_ops.len()
    );

    Ok(())
}

// ============================================================================
// Data Type Support Tests
// ============================================================================

#[test]
fn test_supported_data_types() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;
    let capability = provider.get_capability();

    // Verify essential data types are supported
    assert!(capability.data_types.contains(&DataType::F32));
    assert!(capability.data_types.contains(&DataType::F16));
    assert!(capability.data_types.contains(&DataType::F64));
    assert!(capability.data_types.contains(&DataType::I8));
    assert!(capability.data_types.contains(&DataType::I32));
    assert!(capability.data_types.contains(&DataType::U8));
    assert!(capability.data_types.contains(&DataType::Bool));

    Ok(())
}

// ============================================================================
// Memory Allocator Tests
// ============================================================================

#[test]
fn test_cpu_allocator() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;
    let allocator = provider.get_allocator();

    // Test basic allocation
    let buffer = allocator.allocate(&[100], DataType::F32)?;
    assert_eq!(buffer.size, 400);
    assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

    // Verify alignment (should be SIMD-aligned)
    assert!(buffer.alignment >= 4);
    assert!(buffer.ptr as usize % buffer.alignment == 0);

    allocator.deallocate(buffer)?;

    Ok(())
}

#[test]
fn test_numa_allocator() -> Result<()> {
    let provider = create_numa_cpu_provider(0)?;
    let allocator = provider.get_allocator();

    // Test allocation with NUMA preference
    let buffer = allocator.allocate(&[100], DataType::F32)?;
    assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

    allocator.deallocate(buffer)?;

    Ok(())
}

// ============================================================================
// Subgraph Compilation Tests
// ============================================================================

#[test]
fn test_simple_subgraph_compilation() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            name: Some("add_node".to_string()),
        }],
        edges: vec![],
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
    };

    let kernel = provider.compile_subgraph(subgraph)?;

    // Verify kernel stats
    let stats = kernel.get_performance_stats();
    assert_eq!(stats.execution_count, 0); // Not executed yet

    Ok(())
}

#[test]
fn test_multi_node_subgraph_compilation() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![
            GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["input1".to_string(), "input2".to_string()],
                outputs: vec!["temp1".to_string()],
                name: Some("add_node".to_string()),
            },
            GraphNode {
                id: 1,
                op_type: "ReLU".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["temp1".to_string()],
                outputs: vec!["output1".to_string()],
                name: Some("relu_node".to_string()),
            },
        ],
        edges: vec![],
        inputs: vec!["input1".to_string(), "input2".to_string()],
        outputs: vec!["output1".to_string()],
    };

    let kernel = provider.compile_subgraph(subgraph)?;
    let stats = kernel.get_performance_stats();
    assert_eq!(stats.execution_count, 0);

    Ok(())
}

#[test]
fn test_unsupported_subgraph_compilation() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "UnsupportedOperation".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
            name: Some("unsupported_node".to_string()),
        }],
        edges: vec![],
        inputs: vec!["input1".to_string()],
        outputs: vec!["output1".to_string()],
    };

    let result = provider.compile_subgraph(subgraph);
    assert!(result.is_err());

    Ok(())
}

// ============================================================================
// Kernel Execution Tests
// ============================================================================

#[test]
fn test_kernel_execution() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            name: Some("add".to_string()),
        }],
        edges: vec![],
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
    };

    let kernel = provider.compile_subgraph(subgraph)?;

    // Create input tensors
    let tensor_a = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;
    let tensor_b = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;

    // Execute kernel
    let outputs = kernel.execute(&[tensor_a, tensor_b])?;

    assert!(!outputs.is_empty());

    // Check performance stats updated
    let stats = kernel.get_performance_stats();
    assert_eq!(stats.execution_count, 1);
    assert!(stats.average_time_us > 0.0);

    Ok(())
}

#[test]
fn test_kernel_multiple_executions() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            name: Some("add".to_string()),
        }],
        edges: vec![],
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
    };

    let kernel = provider.compile_subgraph(subgraph)?;

    // Execute multiple times
    for _ in 0..5 {
        let tensor_a = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;
        let tensor_b = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;
        kernel.execute(&[tensor_a, tensor_b])?;
    }

    // Check stats
    let stats = kernel.get_performance_stats();
    assert_eq!(stats.execution_count, 5);

    Ok(())
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_provider_reconfiguration() -> Result<()> {
    let mut provider = CpuExecutionProvider::new()?;

    let config = ProviderConfig {
        thread_count: Some(2),
        memory_limit: Some(256 * 1024 * 1024),
        optimization_level: ronn_core::OptimizationLevel::Basic,
        custom_options: {
            let mut opts = HashMap::new();
            opts.insert("enable_simd".to_string(), "true".to_string());
            opts.insert("numa_node".to_string(), "0".to_string());
            opts
        },
    };

    provider.configure(config)?;

    assert_eq!(provider.get_config().memory_limit, Some(256 * 1024 * 1024));
    assert!(provider.get_config().enable_simd);
    assert_eq!(provider.get_config().numa_node, 0);

    Ok(())
}

#[test]
fn test_custom_options() -> Result<()> {
    let mut provider = CpuExecutionProvider::new()?;

    let config = ProviderConfig {
        thread_count: None,
        memory_limit: None,
        optimization_level: ronn_core::OptimizationLevel::Aggressive,
        custom_options: {
            let mut opts = HashMap::new();
            opts.insert("enable_fusion".to_string(), "false".to_string());
            opts
        },
    };

    provider.configure(config)?;

    assert!(!provider.get_config().enable_fusion);

    Ok(())
}

// ============================================================================
// Cost Estimation Tests
// ============================================================================

#[test]
fn test_operation_cost_estimation() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let add_op = OperatorSpec {
        op_type: "Add".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    };

    let matmul_op = OperatorSpec {
        op_type: "MatMul".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    };

    let conv_op = OperatorSpec {
        op_type: "Conv".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    };

    let add_cost = provider.estimate_cost(&add_op);
    let matmul_cost = provider.estimate_cost(&matmul_op);
    let conv_cost = provider.estimate_cost(&conv_op);

    // Costs should be ordered: Add < MatMul < Conv
    assert!(add_cost < matmul_cost);
    assert!(matmul_cost < conv_cost);

    println!(
        "Operation costs - Add: {}, MatMul: {}, Conv: {}",
        add_cost, matmul_cost, conv_cost
    );

    Ok(())
}

// ============================================================================
// Shutdown Tests
// ============================================================================

#[test]
fn test_provider_shutdown() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;
    provider.shutdown()?;

    // After shutdown, provider should have cleaned up resources
    // (thread pool will be dropped automatically)

    Ok(())
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_end_to_end_workflow() -> Result<()> {
    // Create provider
    let provider = CpuExecutionProvider::new()?;

    // Compile subgraph
    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            name: Some("add".to_string()),
        }],
        edges: vec![],
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
    };

    let kernel = provider.compile_subgraph(subgraph)?;

    // Create tensors
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![0.5, 1.0, 1.5, 2.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Execute
    let outputs = kernel.execute(&[a, b])?;
    assert!(!outputs.is_empty());

    // Clean up
    provider.shutdown()?;

    Ok(())
}

#[test]
fn test_provider_with_all_optimizations() -> Result<()> {
    let config = CpuProviderConfig {
        thread_count: Some(4),
        memory_limit: None,
        numa_node: -1,
        enable_simd: true,
        enable_fusion: true,
        thread_pool_name: "optimized-pool".to_string(),
    };

    let provider = CpuExecutionProvider::with_config(config)?;

    // Verify all optimizations are enabled
    assert!(provider.get_config().enable_simd);
    assert!(provider.get_config().enable_fusion);
    assert_eq!(provider.get_thread_pool().current_num_threads(), 4);

    // Test execution
    let subgraph = SubGraph {
        nodes: vec![
            GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["temp".to_string()],
                name: Some("add".to_string()),
            },
            GraphNode {
                id: 1,
                op_type: "ReLU".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["temp".to_string()],
                outputs: vec!["out".to_string()],
                name: Some("relu".to_string()),
            },
        ],
        edges: vec![],
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["out".to_string()],
    };

    let kernel = provider.compile_subgraph(subgraph)?;

    let a = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;
    let b = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;

    let outputs = kernel.execute(&[a, b])?;
    assert!(!outputs.is_empty());

    Ok(())
}
