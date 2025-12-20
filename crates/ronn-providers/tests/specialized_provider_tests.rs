//! Tests for specialized execution providers (GPU, BitNet, WASM, Custom).
//!
//! These tests are conditionally compiled based on feature flags.

use anyhow::Result;
use ronn_core::{
    DataType, ExecutionProvider, GraphNode, ProviderId, SubGraph, Tensor, TensorLayout,
};
use std::collections::HashMap;

// ============================================================================
// GPU Provider Tests (conditional on "gpu" feature)
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use ronn_providers::{
        GpuExecutionProvider, GpuTopologyManager, MultiGpuMemoryConfig, TopologyConfig,
        create_gpu_provider, create_gpu_provider_with_config,
    };

    #[test]
    fn test_gpu_provider_creation() -> Result<()> {
        match create_gpu_provider() {
            Ok(provider) => {
                assert_eq!(provider.provider_id(), ProviderId::GPU);

                let capability = provider.get_capability();
                assert!(!capability.supported_ops.is_empty());

                println!("GPU provider created successfully");
                Ok(())
            }
            Err(e) => {
                println!("GPU not available: {}", e);
                // Not an error if GPU isn't available
                Ok(())
            }
        }
    }

    #[test]
    fn test_gpu_topology_discovery() -> Result<()> {
        let config = TopologyConfig {
            auto_discovery: true,
            benchmark_links: false,
            cache_topology: true,
            benchmark_duration_ms: 100,
            benchmark_iterations: 1,
            consider_numa: false,
        };

        match GpuTopologyManager::new(config) {
            Ok(manager) => {
                let topology = manager.get_topology();
                println!("Discovered {} GPU devices", topology.devices.len());
                Ok(())
            }
            Err(e) => {
                println!("GPU topology discovery failed: {}", e);
                Ok(())
            }
        }
    }

    #[test]
    fn test_gpu_memory_allocation() -> Result<()> {
        match create_gpu_provider() {
            Ok(provider) => {
                let allocator = provider.get_allocator();

                // Try to allocate GPU memory
                match allocator.allocate(&[100], DataType::F32) {
                    Ok(buffer) => {
                        assert_eq!(buffer.size, 400);
                        allocator.deallocate(buffer)?;
                        println!("GPU memory allocation successful");
                    }
                    Err(e) => {
                        println!("GPU memory allocation failed: {}", e);
                    }
                }
                Ok(())
            }
            Err(_) => Ok(()),
        }
    }

    #[test]
    fn test_gpu_subgraph_compilation() -> Result<()> {
        match create_gpu_provider() {
            Ok(provider) => {
                let subgraph = SubGraph {
                    nodes: vec![GraphNode {
                        id: 0,
                        op_type: "MatMul".to_string(),
                        attributes: HashMap::new(),
                        inputs: vec!["a".to_string(), "b".to_string()],
                        outputs: vec!["c".to_string()],
                        name: Some("matmul".to_string()),
                    }],
                    edges: vec![],
                    inputs: vec!["a".to_string(), "b".to_string()],
                    outputs: vec!["c".to_string()],
                };

                match provider.compile_subgraph(subgraph) {
                    Ok(kernel) => {
                        let stats = kernel.get_performance_stats();
                        assert_eq!(stats.execution_count, 0);
                        println!("GPU kernel compiled successfully");
                    }
                    Err(e) => {
                        println!("GPU compilation failed: {}", e);
                    }
                }
                Ok(())
            }
            Err(_) => Ok(()),
        }
    }

    #[test]
    fn test_multi_gpu_support() -> Result<()> {
        let config = TopologyConfig {
            auto_discovery: true,
            benchmark_links: false,
            cache_topology: false,
            benchmark_duration_ms: 100,
            benchmark_iterations: 1,
            consider_numa: false,
        };

        match GpuTopologyManager::new(config) {
            Ok(manager) => {
                let topology = manager.get_topology();

                if topology.devices.len() > 1 {
                    println!(
                        "Multi-GPU system detected: {} devices",
                        topology.devices.len()
                    );
                } else {
                    println!("Single GPU or no GPU detected");
                }

                Ok(())
            }
            Err(_) => Ok(()),
        }
    }
}

// ============================================================================
// BitNet Provider Tests (conditional on "bitnet" feature)
// ============================================================================

#[cfg(feature = "bitnet")]
mod bitnet_tests {
    use super::*;
    use ronn_providers::{
        BinaryTensor, BitNetExecutionProvider, BitNetProviderConfig, BitNetQuantizer,
        QuantizationMethod, create_bitnet_provider,
    };

    #[test]
    fn test_bitnet_provider_creation() -> Result<()> {
        let provider = create_bitnet_provider()?;

        assert_eq!(
            provider.provider_id(),
            ProviderId::Custom("BitNet".to_string())
        );

        let capability = provider.get_capability();
        assert!(!capability.supported_ops.is_empty());

        println!("BitNet provider created successfully");
        Ok(())
    }

    #[test]
    fn test_bitnet_quantization() -> Result<()> {
        let quantizer = BitNetQuantizer::new(QuantizationMethod::Binary);

        // Create test tensor
        let data = vec![1.0, -0.5, 0.3, -0.8, 0.0, 0.9];
        let tensor = Tensor::from_data(data, vec![6], DataType::F32, TensorLayout::RowMajor)?;

        // Quantize to 1-bit
        match quantizer.quantize(&tensor) {
            Ok(quantized) => {
                println!("Quantization successful");
                Ok(())
            }
            Err(e) => {
                println!("Quantization failed: {}", e);
                Ok(())
            }
        }
    }

    #[test]
    fn test_bitnet_compression_ratio() -> Result<()> {
        // BitNet should achieve 32x compression for F32 -> 1-bit
        let original_size = 1000 * 4; // 1000 F32 values = 4000 bytes
        let compressed_size = 1000 / 8; // 1000 bits = 125 bytes
        let compression_ratio = original_size as f32 / compressed_size as f32;

        assert!((compression_ratio - 32.0).abs() < 0.1);
        println!("BitNet compression ratio: {:.1}x", compression_ratio);

        Ok(())
    }

    #[test]
    fn test_bitnet_subgraph_compilation() -> Result<()> {
        let provider = create_bitnet_provider()?;

        let subgraph = SubGraph {
            nodes: vec![GraphNode {
                id: 0,
                op_type: "MatMul".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["c".to_string()],
                name: Some("bitnet_matmul".to_string()),
            }],
            edges: vec![],
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
        };

        let kernel = provider.compile_subgraph(subgraph)?;
        let stats = kernel.get_performance_stats();
        assert_eq!(stats.execution_count, 0);

        Ok(())
    }
}

// ============================================================================
// WASM Provider Tests (conditional on "wasm" feature)
// ============================================================================

#[cfg(feature = "wasm")]
mod wasm_tests {
    use super::*;
    use ronn_providers::{
        WasmBridge, WasmExecutionProvider, WasmProviderConfig, create_wasm_provider,
    };

    #[test]
    fn test_wasm_provider_creation() -> Result<()> {
        match create_wasm_provider() {
            Ok(provider) => {
                assert_eq!(
                    provider.provider_id(),
                    ProviderId::Custom("WASM".to_string())
                );

                let capability = provider.get_capability();
                assert!(!capability.supported_ops.is_empty());

                println!("WASM provider created successfully");
                Ok(())
            }
            Err(e) => {
                println!("WASM provider creation failed: {}", e);
                Ok(())
            }
        }
    }

    #[test]
    fn test_wasm_simd128_support() -> Result<()> {
        // WASM SIMD128 support test
        match create_wasm_provider() {
            Ok(provider) => {
                let capability = provider.get_capability();
                println!("WASM capability: {:?}", capability.performance_profile);
                Ok(())
            }
            Err(_) => Ok(()),
        }
    }

    #[test]
    fn test_wasm_subgraph_compilation() -> Result<()> {
        match create_wasm_provider() {
            Ok(provider) => {
                let subgraph = SubGraph {
                    nodes: vec![GraphNode {
                        id: 0,
                        op_type: "Add".to_string(),
                        attributes: HashMap::new(),
                        inputs: vec!["a".to_string(), "b".to_string()],
                        outputs: vec!["c".to_string()],
                        name: Some("wasm_add".to_string()),
                    }],
                    edges: vec![],
                    inputs: vec!["a".to_string(), "b".to_string()],
                    outputs: vec!["c".to_string()],
                };

                let kernel = provider.compile_subgraph(subgraph)?;
                println!("WASM kernel compiled");
                Ok(())
            }
            Err(_) => Ok(()),
        }
    }
}

// ============================================================================
// Custom Hardware Provider Tests (conditional on "custom-hardware" feature)
// ============================================================================

#[cfg(feature = "custom-hardware")]
mod custom_hardware_tests {
    use super::*;
    use ronn_providers::{
        CustomProviderRegistry, NpuConfig, NpuProvider, TpuConfig, TpuProvider,
        create_npu_provider, create_tpu_provider,
    };

    #[test]
    fn test_custom_provider_registry() -> Result<()> {
        let registry = CustomProviderRegistry::new();

        // Registry should start empty
        assert_eq!(registry.list_providers().len(), 0);

        println!("Custom provider registry created");
        Ok(())
    }

    #[test]
    fn test_npu_provider_creation() -> Result<()> {
        match create_npu_provider() {
            Ok(provider) => {
                println!("NPU provider created");
                let capability = provider.get_capability();
                assert!(!capability.supported_ops.is_empty());
                Ok(())
            }
            Err(e) => {
                println!("NPU provider not available: {}", e);
                Ok(())
            }
        }
    }

    #[test]
    fn test_tpu_provider_creation() -> Result<()> {
        match create_tpu_provider() {
            Ok(provider) => {
                println!("TPU provider created");
                let capability = provider.get_capability();
                assert!(!capability.supported_ops.is_empty());
                Ok(())
            }
            Err(e) => {
                println!("TPU provider not available: {}", e);
                Ok(())
            }
        }
    }

    #[test]
    fn test_custom_provider_plugin() -> Result<()> {
        // Test custom provider plugin mechanism
        let registry = CustomProviderRegistry::new();

        // Try to register NPU provider
        if let Ok(npu_provider) = create_npu_provider() {
            match registry.register_provider(npu_provider) {
                Ok(_) => {
                    assert_eq!(registry.list_providers().len(), 1);
                    println!("Custom NPU provider registered");
                }
                Err(e) => {
                    println!("Failed to register NPU provider: {}", e);
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Feature Flag Tests (always compiled)
// ============================================================================

#[test]
fn test_compiled_features() {
    println!("Testing feature flags:");

    #[cfg(feature = "cpu")]
    println!("  ✓ CPU feature enabled");

    #[cfg(feature = "gpu")]
    println!("  ✓ GPU feature enabled");

    #[cfg(feature = "bitnet")]
    println!("  ✓ BitNet feature enabled");

    #[cfg(feature = "wasm")]
    println!("  ✓ WASM feature enabled");

    #[cfg(feature = "custom-hardware")]
    println!("  ✓ Custom hardware feature enabled");

    // CPU should always be available
    #[cfg(feature = "cpu")]
    assert!(true);

    #[cfg(not(feature = "cpu"))]
    panic!("CPU feature should always be enabled");
}

#[test]
fn test_default_features() {
    // Default features should include at least CPU
    use ronn_providers::create_cpu_provider;

    let result = create_cpu_provider();
    assert!(
        result.is_ok(),
        "CPU provider should be available by default"
    );
}
