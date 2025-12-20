//! Comprehensive tests for the provider framework.
//!
//! This module tests the core provider framework functionality including:
//! - Provider registration and management
//! - Capability reporting and querying
//! - Provider selection and fallback chains
//! - Configuration and lifecycle management

use anyhow::Result;
use ronn_core::{
    AttributeValue, DataType, ExecutionProvider, KernelStats, MemoryType, MemoryUsage,
    OperatorSpec, PerformanceProfile, ProviderCapability, ProviderConfig, ProviderId,
    ResourceRequirements, SubGraph, Tensor, TensorAllocator,
};
use ronn_providers::{ProviderRegistry, create_cpu_provider};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ============================================================================
// Provider Registration Tests
// ============================================================================

#[test]
fn test_provider_registration() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;

    // Initial state
    assert_eq!(registry.get_provider_ids().len(), 0);

    // Register provider
    registry.register_provider(provider.clone())?;
    assert_eq!(registry.get_provider_ids().len(), 1);
    assert!(registry.get_provider_ids().contains(&ProviderId::CPU));

    // Verify we can retrieve the provider
    let retrieved = registry.get_provider(ProviderId::CPU);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().provider_id(), ProviderId::CPU);

    Ok(())
}

#[test]
fn test_duplicate_provider_registration() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider1 = create_cpu_provider()?;
    let provider2 = create_cpu_provider()?;

    // First registration should succeed
    registry.register_provider(provider1)?;

    // Second registration of same provider type should fail
    let result = registry.register_provider(provider2);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("already registered")
    );

    Ok(())
}

#[test]
fn test_provider_unregistration() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;

    registry.register_provider(provider)?;
    assert_eq!(registry.get_provider_ids().len(), 1);

    // Unregister provider
    registry.unregister_provider(ProviderId::CPU)?;
    assert_eq!(registry.get_provider_ids().len(), 0);

    // Verify provider is no longer available
    assert!(registry.get_provider(ProviderId::CPU).is_none());

    Ok(())
}

#[test]
fn test_unregister_nonexistent_provider() -> Result<()> {
    let registry = ProviderRegistry::new();

    // Attempting to unregister a provider that was never registered should fail
    let result = registry.unregister_provider(ProviderId::CPU);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));

    Ok(())
}

// ============================================================================
// Capability Tests
// ============================================================================

#[test]
fn test_provider_capability_reporting() -> Result<()> {
    let provider = create_cpu_provider()?;
    let capability = provider.get_capability();

    // Check basic capability fields
    assert!(!capability.supported_ops.is_empty());
    assert!(!capability.data_types.is_empty());
    assert!(!capability.memory_types.is_empty());
    assert_eq!(capability.performance_profile, PerformanceProfile::CPU);

    // Verify essential operations are supported
    assert!(capability.supported_ops.contains("Add"));
    assert!(capability.supported_ops.contains("MatMul"));
    assert!(capability.supported_ops.contains("ReLU"));

    // Verify essential data types
    assert!(capability.data_types.contains(&DataType::F32));

    // Verify memory types
    assert!(capability.memory_types.contains(&MemoryType::SystemRAM));

    Ok(())
}

#[test]
fn test_capability_querying_from_registry() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    // Query capability through registry
    let capability = registry.get_capability(ProviderId::CPU);
    assert!(capability.is_some());

    let cap = capability.unwrap();
    assert_eq!(cap.performance_profile, PerformanceProfile::CPU);
    assert!(!cap.supported_ops.is_empty());

    Ok(())
}

#[test]
fn test_can_handle_operations() -> Result<()> {
    let provider = create_cpu_provider()?;

    let operators = vec![
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
            op_type: "UnsupportedOp".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
    ];

    let results = provider.can_handle(&operators);

    // First two should be supported, last one should not
    assert_eq!(results.len(), 3);
    assert!(results[0]); // Add
    assert!(results[1]); // MatMul
    assert!(!results[2]); // UnsupportedOp

    Ok(())
}

// ============================================================================
// Provider Selection Tests
// ============================================================================

#[test]
fn test_provider_selection_single_provider() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let operators = vec![OperatorSpec {
        op_type: "Add".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    }];

    let selected = registry.select_provider(&operators);
    assert_eq!(selected, Some(ProviderId::CPU));

    Ok(())
}

#[test]
fn test_provider_selection_unsupported_operation() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let operators = vec![OperatorSpec {
        op_type: "CompletelyUnsupportedOperation".to_string(),
        input_types: vec![DataType::F32],
        output_types: vec![DataType::F32],
        attributes: HashMap::new(),
    }];

    let selected = registry.select_provider(&operators);
    assert_eq!(selected, None);

    Ok(())
}

#[test]
fn test_provider_selection_mixed_operations() -> Result<()> {
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
            op_type: "UnsupportedOp".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
    ];

    // With mixed supported/unsupported, no provider can handle all operations
    let selected = registry.select_provider(&operators);
    assert_eq!(selected, None);

    Ok(())
}

// ============================================================================
// Fallback Chain Tests
// ============================================================================

#[test]
fn test_fallback_chain_single_provider() -> Result<()> {
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
            op_type: "MatMul".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        },
    ];

    let fallback_chain = registry.get_fallback_chain(&operators);

    // Should have one entry with CPU provider handling all operations
    assert_eq!(fallback_chain.len(), 1);
    assert_eq!(fallback_chain[0].0, ProviderId::CPU);
    assert_eq!(fallback_chain[0].1, vec![0, 1]);

    Ok(())
}

#[test]
fn test_fallback_chain_partial_support() -> Result<()> {
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
    ];

    let fallback_chain = registry.get_fallback_chain(&operators);

    // CPU provider should handle operations 0 and 2 (Add and MatMul)
    assert_eq!(fallback_chain.len(), 1);
    assert_eq!(fallback_chain[0].0, ProviderId::CPU);
    assert_eq!(fallback_chain[0].1, vec![0, 2]);

    Ok(())
}

// ============================================================================
// Preference Order Tests
// ============================================================================

#[test]
fn test_default_preference_order() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let order = registry.get_preference_order();
    assert_eq!(order.len(), 1);
    assert_eq!(order[0], ProviderId::CPU);

    Ok(())
}

#[test]
fn test_custom_preference_order() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    // Set custom preference order
    let custom_order = vec![ProviderId::CPU];
    registry.set_preference_order(custom_order.clone())?;

    let order = registry.get_preference_order();
    assert_eq!(order, custom_order);

    Ok(())
}

#[test]
fn test_invalid_preference_order() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    // Try to set preference order with unregistered provider
    let invalid_order = vec![ProviderId::CPU, ProviderId::GPU];
    let result = registry.set_preference_order(invalid_order);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not registered"));

    Ok(())
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_registry_statistics() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let stats = registry.get_statistics();

    assert_eq!(stats.provider_count, 1);
    assert!(stats.total_supported_ops > 0);
    assert_eq!(stats.preference_order.len(), 1);
    assert_eq!(stats.preference_order[0], ProviderId::CPU);

    Ok(())
}

#[test]
fn test_statistics_no_providers() -> Result<()> {
    let registry = ProviderRegistry::new();
    let stats = registry.get_statistics();

    assert_eq!(stats.provider_count, 0);
    assert_eq!(stats.total_supported_ops, 0);
    assert_eq!(stats.preference_order.len(), 0);

    Ok(())
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_provider_configuration() -> Result<()> {
    // Create a mutable provider directly
    let mut provider = ronn_providers::CpuExecutionProvider::new()?;

    // Create a configuration
    let config = ProviderConfig {
        thread_count: Some(4),
        memory_limit: Some(256 * 1024 * 1024), // 256MB
        optimization_level: ronn_core::OptimizationLevel::Aggressive,
        custom_options: HashMap::new(),
    };

    // Configure the provider
    let result = provider.configure(config);
    assert!(result.is_ok());

    Ok(())
}

// ============================================================================
// Shutdown and Cleanup Tests
// ============================================================================

#[test]
fn test_provider_shutdown() -> Result<()> {
    let provider = create_cpu_provider()?;
    let result = provider.shutdown();
    assert!(result.is_ok());

    Ok(())
}

#[test]
fn test_registry_shutdown() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let result = registry.shutdown();
    assert!(result.is_ok());

    // After shutdown, registry should be empty
    let stats = registry.get_statistics();
    assert_eq!(stats.provider_count, 0);

    Ok(())
}

// ============================================================================
// Subgraph Compilation Tests
// ============================================================================

#[test]
fn test_compile_subgraph_via_registry() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let subgraph = SubGraph {
        nodes: vec![ronn_core::GraphNode {
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

    let result = registry.compile_subgraph(subgraph);
    assert!(result.is_ok());

    let (provider_id, kernel) = result.unwrap();
    assert_eq!(provider_id, ProviderId::CPU);

    // Verify kernel stats
    let stats = kernel.get_performance_stats();
    assert_eq!(stats.execution_count, 0); // Not executed yet

    Ok(())
}

#[test]
fn test_compile_unsupported_subgraph() -> Result<()> {
    let registry = ProviderRegistry::new();
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let subgraph = SubGraph {
        nodes: vec![ronn_core::GraphNode {
            id: 0,
            op_type: "CompletelyUnsupportedOp".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
            name: Some("unsupported_op".to_string()),
        }],
        edges: vec![],
        inputs: vec!["input1".to_string()],
        outputs: vec!["output1".to_string()],
    };

    let result = registry.compile_subgraph(subgraph);
    assert!(result.is_err());

    Ok(())
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[test]
fn test_concurrent_provider_registration() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let registry = Arc::new(ProviderRegistry::new());
    let mut handles = vec![];

    // Try to register providers from multiple threads (should only succeed once)
    for _ in 0..4 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let provider = create_cpu_provider().unwrap();
            registry_clone.register_provider(provider)
        });
        handles.push(handle);
    }

    // Collect results
    let mut success_count = 0;
    let mut error_count = 0;

    for handle in handles {
        match handle.join().unwrap() {
            Ok(_) => success_count += 1,
            Err(_) => error_count += 1,
        }
    }

    // Only one thread should succeed
    assert_eq!(success_count, 1);
    assert_eq!(error_count, 3);

    Ok(())
}

#[test]
fn test_concurrent_provider_queries() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let registry = Arc::new(ProviderRegistry::new());
    let provider = create_cpu_provider()?;
    registry.register_provider(provider)?;

    let mut handles = vec![];

    // Query providers from multiple threads concurrently
    for _ in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let provider = registry_clone.get_provider(ProviderId::CPU);
            assert!(provider.is_some());

            let capability = registry_clone.get_capability(ProviderId::CPU);
            assert!(capability.is_some());

            let stats = registry_clone.get_statistics();
            assert_eq!(stats.provider_count, 1);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}
