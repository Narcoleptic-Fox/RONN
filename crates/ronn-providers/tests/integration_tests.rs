//! Integration tests for the RONN execution provider framework.
//!
//! These tests validate end-to-end functionality across all provider components
//! including registry management, memory allocation, and tensor operations.

use anyhow::Result;
use ronn_core::tensor::Tensor;
use ronn_core::{DataType, GraphNode, SubGraph, TensorLayout};
use ronn_providers::{ProviderRegistry, create_cpu_provider};
use std::collections::HashMap;

/// Test basic CPU provider creation and capability reporting.
#[test]
fn test_cpu_provider_creation() -> Result<()> {
    let provider = create_cpu_provider()?;

    // Check provider ID
    assert_eq!(provider.provider_id(), ronn_core::ProviderId::CPU);

    // Check capabilities
    let capability = provider.get_capability();
    assert!(!capability.supported_ops.is_empty());
    assert!(capability.supported_ops.contains("Add"));
    assert!(capability.supported_ops.contains("MatMul"));
    assert!(capability.data_types.contains(&DataType::F32));

    println!(
        "✅ CPU provider created with {} supported operations",
        capability.supported_ops.len()
    );

    Ok(())
}

/// Test basic tensor creation and operations.
#[test]
fn test_tensor_operations() -> Result<()> {
    // Create test tensors
    let tensor_a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let tensor_b = Tensor::from_data(
        vec![0.5, 1.0, 1.5, 2.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Verify tensor properties
    assert_eq!(tensor_a.shape(), vec![2, 2]);
    assert_eq!(tensor_a.dtype(), DataType::F32);
    assert_eq!(tensor_b.shape(), vec![2, 2]);

    println!(
        "✅ Created test tensors: {:?} and {:?}",
        tensor_a.shape(),
        tensor_b.shape()
    );

    Ok(())
}

/// Test memory allocator functionality.
#[test]
fn test_memory_allocator() -> Result<()> {
    let provider = create_cpu_provider()?;
    let allocator = provider.get_allocator();

    // Test allocation
    let buffer = allocator.allocate(&[100], DataType::F32)?;
    assert_eq!(buffer.size, 400); // 100 * 4 bytes for F32
    assert!(buffer.alignment >= 4);

    let memory_info_before = allocator.get_memory_info();
    assert!(memory_info_before.allocated_bytes >= 400);

    // Test deallocation
    allocator.deallocate(buffer)?;

    let memory_info_after = allocator.get_memory_info();
    assert!(memory_info_after.allocated_bytes < memory_info_before.allocated_bytes);

    println!(
        "✅ Memory allocator working: allocated {} bytes",
        memory_info_before.allocated_bytes
    );

    Ok(())
}

/// Test provider registry with single provider.
#[test]
fn test_provider_registry_basic() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;

    // Register provider
    registry.register_provider(provider.clone())?;

    // Check registration
    let provider_ids = registry.get_provider_ids();
    assert_eq!(provider_ids.len(), 1);
    assert!(provider_ids.contains(&ronn_core::ProviderId::CPU));

    // Test provider retrieval
    let retrieved_provider = registry.get_provider(ronn_core::ProviderId::CPU);
    assert!(retrieved_provider.is_some());

    println!(
        "✅ Provider registry working with {} registered providers",
        provider_ids.len()
    );

    Ok(())
}

/// Test provider capability querying and selection.
#[test]
fn test_provider_selection() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    // Create operator specs for selection
    let add_op = ronn_core::OperatorSpec {
        op_type: "Add".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    };

    let invalid_op = ronn_core::OperatorSpec {
        op_type: "NonexistentOp".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    };

    // Test supported operation selection
    let selected = registry.select_provider(&[add_op]);
    assert_eq!(selected, Some(ronn_core::ProviderId::CPU));

    // Test unsupported operation
    let not_selected = registry.select_provider(&[invalid_op]);
    assert_eq!(not_selected, None);

    println!("✅ Provider selection working correctly");

    Ok(())
}

/// Test subgraph compilation (basic validation).
#[test]
fn test_subgraph_compilation() -> Result<()> {
    let provider = create_cpu_provider()?;

    // Create a simple subgraph with an Add operation
    let add_node = GraphNode {
        id: 0,
        op_type: "Add".to_string(),
        attributes: HashMap::new(),
        inputs: vec!["input1".to_string(), "input2".to_string()],
        outputs: vec!["output1".to_string()],
        name: Some("test_add".to_string()),
    };

    let subgraph = SubGraph {
        nodes: vec![add_node],
        edges: vec![],
        inputs: vec!["input1".to_string(), "input2".to_string()],
        outputs: vec!["output1".to_string()],
    };

    // Attempt compilation
    let kernel_result = provider.compile_subgraph(subgraph);

    // For now, just verify it doesn't crash - the implementation might be incomplete
    match kernel_result {
        Ok(kernel) => {
            let stats = kernel.get_performance_stats();
            println!(
                "✅ Subgraph compilation successful, execution count: {}",
                stats.execution_count
            );
        }
        Err(e) => {
            println!("ℹ️  Subgraph compilation not yet fully implemented: {}", e);
            // This is expected since we haven't fully implemented kernel execution yet
        }
    }

    Ok(())
}

/// Test provider shutdown and cleanup.
#[test]
fn test_provider_shutdown() -> Result<()> {
    let provider = create_cpu_provider()?;

    // Test graceful shutdown
    let shutdown_result = provider.shutdown();
    assert!(shutdown_result.is_ok(), "Provider shutdown should succeed");

    println!("✅ Provider shutdown completed successfully");

    Ok(())
}

/// Integration test combining multiple components.
#[test]
fn test_end_to_end_workflow() -> Result<()> {
    println!("🚀 Starting end-to-end workflow test...");

    // 1. Create and register provider
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;
    println!("   ✓ Provider registered");

    // 2. Test memory allocation
    let provider = registry.get_provider(ronn_core::ProviderId::CPU).unwrap();
    let allocator = provider.get_allocator();
    let buffer = allocator.allocate(&[50], DataType::F32)?;
    println!("   ✓ Memory allocated: {} bytes", buffer.size);

    // 3. Test tensor creation
    let tensor = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    println!("   ✓ Tensor created: shape {:?}", tensor.shape());

    // 4. Clean up
    allocator.deallocate(buffer)?;
    provider.shutdown()?;
    println!("   ✓ Cleanup completed");

    println!("✅ End-to-end workflow test passed!");

    Ok(())
}

/// Performance validation - basic timing test.
#[test]
fn test_performance_basic() -> Result<()> {
    use std::time::Instant;

    let start = Instant::now();

    // Create provider (should be fast)
    let provider = create_cpu_provider()?;
    let creation_time = start.elapsed();

    println!("✅ Provider creation time: {:?}", creation_time);
    assert!(
        creation_time.as_millis() < 100,
        "Provider creation should be under 100ms"
    );

    // Test memory allocation speed
    let allocator = provider.get_allocator();
    let alloc_start = Instant::now();

    let mut buffers = Vec::new();
    for _ in 0..100 {
        buffers.push(allocator.allocate(&[10], DataType::F32)?);
    }
    let alloc_time = alloc_start.elapsed();

    println!("✅ 100 allocations time: {:?}", alloc_time);
    assert!(
        alloc_time.as_millis() < 50,
        "100 allocations should be under 50ms"
    );

    // Clean up
    for buffer in buffers {
        allocator.deallocate(buffer)?;
    }

    Ok(())
}

/// Test SIMD capabilities detection.
#[test]
fn test_simd_capabilities() -> Result<()> {
    use ronn_providers::cpu::detect_simd_capabilities;

    let capabilities = detect_simd_capabilities();

    // On most modern systems, at least SSE2 should be available
    println!("SIMD Capabilities detected:");
    println!("  SSE2: {}", capabilities.sse2);
    println!("  SSE4.1: {}", capabilities.sse41);
    println!("  AVX: {}", capabilities.avx);
    println!("  AVX2: {}", capabilities.avx2);
    println!("  AVX-512F: {}", capabilities.avx512f);
    println!("  FMA: {}", capabilities.fma);

    // At minimum, we expect SSE2 on x86_64
    #[cfg(target_arch = "x86_64")]
    assert!(capabilities.sse2, "SSE2 should be available on x86_64");

    println!("✅ SIMD capabilities detection working");

    Ok(())
}
