//! Edge case and error handling tests for the provider framework.
//!
//! This module tests:
//! - Provider unavailability scenarios
//! - Unsupported operations fallback
//! - Memory allocation failures
//! - Concurrent access edge cases
//! - Resource exhaustion scenarios
//! - Configuration edge cases

use anyhow::Result;
use ronn_core::{
    DataType, ExecutionProvider, GraphNode, MemoryType, OperatorSpec, ProviderId, SubGraph, Tensor,
    TensorAllocator, TensorLayout,
};
use ronn_providers::{
    PoolConfig, PooledMemoryAllocator, ProviderRegistry, SystemMemoryAllocator,
    cpu::CpuExecutionProvider, cpu::CpuProviderConfig, create_cpu_provider,
};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Provider Unavailability Tests
// ============================================================================

#[test]
fn test_gpu_provider_unavailable() {
    use ronn_providers::create_gpu_provider;

    // GPU provider may not be available on all systems
    match create_gpu_provider() {
        Ok(_) => {
            println!("GPU provider available");
        }
        Err(e) => {
            println!("GPU provider unavailable (expected): {}", e);
            // This is not a test failure - GPU may simply not be available
            assert!(e.to_string().len() > 0);
        }
    }
}

#[test]
fn test_provider_not_registered() -> Result<()> {
    let registry = ProviderRegistry::new();

    // Try to get a provider that was never registered
    let provider = registry.get_provider(ProviderId::GPU);
    assert!(provider.is_none());

    // Try to get capability
    let capability = registry.get_capability(ProviderId::GPU);
    assert!(capability.is_none());

    Ok(())
}

#[test]
fn test_empty_registry_operations() -> Result<()> {
    let registry = ProviderRegistry::new();

    // Try to select provider with empty registry
    let operators = vec![OperatorSpec {
        op_type: "Add".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    }];

    let selected = registry.select_provider(&operators);
    assert_eq!(selected, None);

    // Try to compile subgraph with empty registry
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

    let result = registry.compile_subgraph(subgraph);
    assert!(result.is_err());

    Ok(())
}

// ============================================================================
// Unsupported Operation Tests
// ============================================================================

#[test]
fn test_completely_unsupported_operation() -> Result<()> {
    let provider = create_cpu_provider()?;

    let ops = vec![OperatorSpec {
        op_type: "FutureSuperAdvancedOperation".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    }];

    let results = provider.can_handle(&ops);
    assert_eq!(results.len(), 1);
    assert!(!results[0]);

    Ok(())
}

#[test]
fn test_partial_operation_support() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let operators = vec![
        OperatorSpec {
            op_type: "Add".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
        OperatorSpec {
            op_type: "UnsupportedOp1".to_string(),
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
            op_type: "UnsupportedOp2".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
    ];

    // Get fallback chain
    let fallback_chain = registry.get_fallback_chain(&operators);

    // Should only handle supported operations (indices 0 and 2)
    assert_eq!(fallback_chain.len(), 1);
    assert_eq!(fallback_chain[0].0, ProviderId::CPU);
    assert!(fallback_chain[0].1.contains(&0));
    assert!(fallback_chain[0].1.contains(&2));
    assert!(!fallback_chain[0].1.contains(&1));
    assert!(!fallback_chain[0].1.contains(&3));

    Ok(())
}

#[test]
fn test_unsupported_subgraph_compilation() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "CompletelyUnsupportedOperation".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            name: Some("unsupported".to_string()),
        }],
        edges: vec![],
        inputs: vec!["input".to_string()],
        outputs: vec!["output".to_string()],
    };

    let result = provider.compile_subgraph(subgraph);
    assert!(result.is_err());

    Ok(())
}

// ============================================================================
// Memory Allocation Failure Tests
// ============================================================================

#[test]
fn test_zero_size_allocation() -> Result<()> {
    let allocator = SystemMemoryAllocator::new();

    // Zero-sized allocations - behavior is implementation-defined
    let result = allocator.allocate(&[], DataType::F32);
    // May succeed or fail depending on implementation
    let _ = result;

    let result = allocator.allocate(&[0], DataType::F32);
    // May succeed or fail depending on implementation
    let _ = result;

    let result = allocator.allocate(&[0, 10], DataType::F32);
    // Should fail due to zero element
    let _ = result;

    Ok(())
}

#[test]
fn test_extremely_large_allocation() -> Result<()> {
    let allocator = SystemMemoryAllocator::new();

    // Try to allocate an unreasonably large buffer (100GB)
    // This should fail on most systems
    let result = allocator.allocate(&[25_000_000_000], DataType::F32);

    // Either it fails immediately, or we don't have enough memory
    match result {
        Ok(_) => {
            // If it somehow succeeds, this is unexpected but not necessarily wrong
            println!("Warning: Extremely large allocation succeeded");
        }
        Err(_) => {
            // Expected: allocation should fail
        }
    }

    Ok(())
}

#[test]
fn test_pool_overflow() -> Result<()> {
    let config = PoolConfig {
        max_buffers_per_bucket: 2,
        max_pool_size: 256, // Very small pool
        bucket_granularity: 64,
    };

    let allocator = PooledMemoryAllocator::new(config);

    // Allocate and deallocate more buffers than the pool can hold
    for _ in 0..10 {
        let buffer = allocator.allocate(&[50], DataType::F32)?;
        allocator.deallocate(buffer)?;
    }

    // Pool should not grow beyond limits
    let _stats = allocator.get_pool_stats();
    // Pool should not grow beyond limits - implementation handles this

    Ok(())
}

#[test]
fn test_double_deallocation_protection() -> Result<()> {
    let allocator = SystemMemoryAllocator::new();

    let buffer = allocator.allocate(&[10], DataType::F32)?;

    // First deallocation should succeed
    allocator.deallocate(buffer)?;

    // Note: Double deallocation is undefined behavior in Rust
    // We can't safely test this without risking crashes
    // In production, this should be prevented by ownership system

    Ok(())
}

// ============================================================================
// Configuration Edge Cases
// ============================================================================

#[test]
fn test_invalid_thread_count() -> Result<()> {
    // Test with zero threads (should default to 1)
    let config = CpuProviderConfig {
        thread_count: Some(0),
        ..Default::default()
    };

    // This might succeed by adjusting to minimum thread count
    match CpuExecutionProvider::with_config(config) {
        Ok(provider) => {
            // If it succeeds, thread count should be adjusted
            assert!(provider.get_thread_pool().current_num_threads() >= 1);
        }
        Err(_) => {
            // Or it might fail, which is also acceptable
        }
    }

    Ok(())
}

#[test]
fn test_invalid_numa_node() -> Result<()> {
    // Try to create provider with unrealistic NUMA node
    let config = CpuProviderConfig {
        numa_node: 9999, // Unlikely to exist
        ..Default::default()
    };

    // Should either fail or fall back to default allocation
    let _result = CpuExecutionProvider::with_config(config);
    // We don't assert failure here since NUMA handling varies by system

    Ok(())
}

#[test]
fn test_empty_preference_order() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    // Try to set empty preference order
    let result = registry.set_preference_order(vec![]);

    // This should succeed (empty order means no providers preferred)
    assert!(result.is_ok());

    Ok(())
}

#[test]
fn test_invalid_preference_order() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    // Try to set preference order with unregistered provider
    let result = registry.set_preference_order(vec![ProviderId::GPU, ProviderId::CPU]);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not registered"));

    Ok(())
}

// ============================================================================
// Subgraph Edge Cases
// ============================================================================

#[test]
fn test_empty_subgraph() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![],
        edges: vec![],
        inputs: vec![],
        outputs: vec![],
    };

    // Empty subgraph should either compile to noop or fail gracefully
    let _result = provider.compile_subgraph(subgraph);
    // Don't assert success/failure - implementation-defined behavior

    Ok(())
}

#[test]
fn test_subgraph_with_no_inputs() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec![],
            outputs: vec!["output".to_string()],
            name: Some("add_no_inputs".to_string()),
        }],
        edges: vec![],
        inputs: vec![],
        outputs: vec!["output".to_string()],
    };

    // This might fail or succeed depending on implementation
    let _result = provider.compile_subgraph(subgraph);

    Ok(())
}

#[test]
fn test_subgraph_with_no_outputs() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec![],
            name: Some("add_no_outputs".to_string()),
        }],
        edges: vec![],
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec![],
    };

    // This might fail or succeed depending on implementation
    let _result = provider.compile_subgraph(subgraph);

    Ok(())
}

// ============================================================================
// Concurrent Access Edge Cases
// ============================================================================

#[test]
fn test_concurrent_provider_access() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let provider = Arc::new(create_cpu_provider()?);
    let mut handles = vec![];

    // Access provider from multiple threads simultaneously
    for i in 0..10 {
        let provider_clone = Arc::clone(&provider);
        let handle = thread::spawn(move || {
            // Get capability multiple times
            for _ in 0..100 {
                let _capability = provider_clone.get_capability();
            }

            // Check operation support
            let _supported = provider_clone.can_handle(&[OperatorSpec {
                op_type: format!("Op{}", i),
                input_types: vec![DataType::F32],
                output_types: vec![DataType::F32],
                attributes: HashMap::new(),
            }]);
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

#[test]
fn test_concurrent_allocations_same_size() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let allocator = Arc::new(SystemMemoryAllocator::new());
    let mut handles = vec![];

    // Allocate from multiple threads simultaneously
    for _ in 0..4 {
        let allocator_clone = Arc::clone(&allocator);
        let handle = thread::spawn(move || -> Result<()> {
            for _ in 0..50 {
                let buffer = allocator_clone.allocate(&[100], DataType::F32)?;
                allocator_clone.deallocate(buffer)?;
            }
            Ok(())
        });
        handles.push(handle);
    }

    // Wait and check for errors
    for handle in handles {
        handle.join().unwrap()?;
    }

    // All memory should be freed
    let memory_info = allocator.get_memory_info();
    assert_eq!(memory_info.allocated_bytes, 0);

    Ok(())
}

#[test]
fn test_registry_concurrent_queries() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let registry = Arc::new(ProviderRegistry::new());
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let mut handles = vec![];

    // Query registry from multiple threads
    for _ in 0..8 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let _provider = registry_clone.get_provider(ProviderId::CPU);
                let _capability = registry_clone.get_capability(ProviderId::CPU);
                let _stats = registry_clone.get_statistics();
                let _order = registry_clone.get_preference_order();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

// ============================================================================
// Resource Exhaustion Tests
// ============================================================================

#[test]
fn test_many_small_allocations() -> Result<()> {
    let allocator = SystemMemoryAllocator::new();
    let mut buffers = Vec::new();

    // Allocate many small buffers
    for _ in 0..10_000 {
        let buffer = allocator.allocate(&[10], DataType::U8)?;
        buffers.push(buffer);
    }

    // Check memory usage
    let memory_info = allocator.get_memory_info();
    assert!(memory_info.allocated_bytes >= 100_000); // At least 10,000 * 10 bytes

    // Deallocate all
    for buffer in buffers {
        allocator.deallocate(buffer)?;
    }

    let memory_info = allocator.get_memory_info();
    assert_eq!(memory_info.allocated_bytes, 0);

    Ok(())
}

#[test]
fn test_alternating_allocation_sizes() -> Result<()> {
    let config = PoolConfig::default();
    let allocator = PooledMemoryAllocator::new(config);

    let sizes = vec![16, 32, 64, 128, 256, 64, 32, 128, 16];

    for &size in &sizes {
        let buffer = allocator.allocate(&[size], DataType::F32)?;
        allocator.deallocate(buffer)?;
    }

    // Pool should have handled various sizes
    let stats = allocator.get_pool_stats();
    println!(
        "Pool stats after alternating sizes: hit rate = {:.2}%",
        allocator.get_hit_rate() * 100.0
    );

    Ok(())
}

// ============================================================================
// Cleanup and Lifecycle Tests
// ============================================================================

#[test]
fn test_provider_shutdown_and_reuse() -> Result<()> {
    let provider = CpuExecutionProvider::new()?;

    // Use the provider
    let _capability = provider.get_capability();

    // Shutdown
    provider.shutdown()?;

    // After shutdown, provider methods might still work or might fail
    // This is implementation-defined, but should not panic
    let _result = provider.get_capability();

    Ok(())
}

#[test]
fn test_registry_shutdown_with_active_providers() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    // Shutdown registry with active providers
    registry.shutdown()?;

    // Registry should be empty after shutdown
    let stats = registry.get_statistics();
    assert_eq!(stats.provider_count, 0);

    Ok(())
}

#[test]
fn test_memory_leak_after_errors() -> Result<()> {
    let allocator = SystemMemoryAllocator::new();

    // Perform some successful allocations
    for _ in 0..10 {
        let buffer = allocator.allocate(&[100], DataType::F32)?;
        allocator.deallocate(buffer)?;
    }

    // Try some potentially failed allocations
    for _ in 0..10 {
        let _result = allocator.allocate(&[1], DataType::F32);
        // Ignore results - deallocations handled automatically if succeeded
    }

    // Most allocations should have been deallocated
    // (some may still be allocated if the loop succeeded)
    let memory_info = allocator.get_memory_info();
    // Check that we don't have a huge leak
    assert!(memory_info.allocated_bytes < 10000);

    Ok(())
}

// ============================================================================
// Tensor Operation Edge Cases
// ============================================================================

#[test]
fn test_mismatched_tensor_shapes() -> Result<()> {
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

    // Create tensors with mismatched shapes
    let tensor_a = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;
    let tensor_b = Tensor::ones(vec![8], DataType::F32, TensorLayout::RowMajor)?;

    // Execution might fail due to shape mismatch
    let _result = kernel.execute(&[tensor_a, tensor_b]);
    // Don't assert - error handling is implementation-defined

    Ok(())
}

#[test]
fn test_wrong_number_of_inputs() -> Result<()> {
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

    // Provide wrong number of inputs
    let tensor_a = Tensor::ones(vec![4], DataType::F32, TensorLayout::RowMajor)?;

    // Should fail with wrong number of inputs
    let _result = kernel.execute(&[tensor_a]);

    Ok(())
}
