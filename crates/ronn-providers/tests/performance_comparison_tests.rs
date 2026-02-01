//! Performance comparison tests between different providers and allocators.
//!
//! These tests measure and compare performance characteristics of:
//! - Different execution providers (CPU, GPU when available)
//! - Memory allocators (system vs pooled)
//! - SIMD optimizations
//! - Threading configurations

use anyhow::Result;
use ronn_core::{
    DataType, ExecutionProvider, GraphNode, ProviderId, SubGraph, Tensor, TensorAllocator,
    TensorLayout,
};
use ronn_providers::{
    AlignedMemoryAllocator, PoolConfig, PooledMemoryAllocator, SystemMemoryAllocator,
    cpu::CpuExecutionProvider, cpu::CpuProviderConfig, create_cpu_provider,
};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Allocator Performance Comparisons
// ============================================================================

#[test]
fn test_compare_allocator_performance() -> Result<()> {
    const ITERATIONS: usize = 1000;
    const ALLOC_SIZE: usize = 100;

    println!("\n=== Allocator Performance Comparison ===");
    println!(
        "Iterations: {}, Size: {} F32 elements\n",
        ITERATIONS, ALLOC_SIZE
    );

    // Test System Allocator
    let system_allocator = SystemMemoryAllocator::new();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let buffer = system_allocator.allocate(&[ALLOC_SIZE], DataType::F32)?;
        system_allocator.deallocate(buffer)?;
    }
    let system_time = start.elapsed();

    // Test Aligned Allocator
    let aligned_allocator = AlignedMemoryAllocator::new();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let buffer = aligned_allocator.allocate(&[ALLOC_SIZE], DataType::F32)?;
        aligned_allocator.deallocate(buffer)?;
    }
    let aligned_time = start.elapsed();

    // Test Pooled Allocator
    let config = PoolConfig::default();
    let pooled_allocator = PooledMemoryAllocator::new(config);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let buffer = pooled_allocator.allocate(&[ALLOC_SIZE], DataType::F32)?;
        pooled_allocator.deallocate(buffer)?;
    }
    let pooled_time = start.elapsed();

    println!(
        "System Allocator:  {:?} ({:.2} µs/op)",
        system_time,
        system_time.as_micros() as f64 / ITERATIONS as f64
    );
    println!(
        "Aligned Allocator: {:?} ({:.2} µs/op)",
        aligned_time,
        aligned_time.as_micros() as f64 / ITERATIONS as f64
    );
    println!(
        "Pooled Allocator:  {:?} ({:.2} µs/op)",
        pooled_time,
        pooled_time.as_micros() as f64 / ITERATIONS as f64
    );
    println!(
        "Pool hit rate: {:.2}%\n",
        pooled_allocator.get_hit_rate() * 100.0
    );

    // Pooled allocator should generally be faster due to reuse
    let speedup = system_time.as_micros() as f64 / pooled_time.as_micros() as f64;
    println!("Pooled vs System speedup: {:.2}x", speedup);

    Ok(())
}

#[test]
fn test_allocation_size_impact() -> Result<()> {
    println!("\n=== Allocation Size Impact ===\n");

    let sizes = vec![10, 100, 1000, 10_000];
    let config = PoolConfig::default();

    for size in sizes {
        let allocator = PooledMemoryAllocator::new(config.clone());

        let start = Instant::now();
        for _ in 0..100 {
            let buffer = allocator.allocate(&[size], DataType::F32)?;
            allocator.deallocate(buffer)?;
        }
        let elapsed = start.elapsed();

        println!(
            "Size {:>6} F32: {:?} ({:.2} µs/op, hit rate: {:.1}%)",
            size,
            elapsed,
            elapsed.as_micros() as f64 / 100.0,
            allocator.get_hit_rate() * 100.0
        );
    }

    Ok(())
}

#[test]
fn test_concurrent_allocation_performance() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    println!("\n=== Concurrent Allocation Performance ===\n");

    let thread_counts = vec![1, 2, 4, 8];

    for thread_count in thread_counts {
        let allocator = Arc::new(SystemMemoryAllocator::new());
        let start = Instant::now();

        let mut handles = vec![];
        for _ in 0..thread_count {
            let allocator_clone = Arc::clone(&allocator);
            let handle = thread::spawn(move || -> Result<()> {
                for _ in 0..250 {
                    let buffer = allocator_clone.allocate(&[100], DataType::F32)?;
                    allocator_clone.deallocate(buffer)?;
                }
                Ok(())
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap()?;
        }

        let elapsed = start.elapsed();
        let total_ops = thread_count * 250;

        println!(
            "{} threads: {:?} ({:.2} µs/op)",
            thread_count,
            elapsed,
            elapsed.as_micros() as f64 / total_ops as f64
        );
    }

    Ok(())
}

// ============================================================================
// Provider Performance Comparisons
// ============================================================================

#[test]
fn test_cpu_provider_thread_scaling() -> Result<()> {
    println!("\n=== CPU Provider Thread Scaling ===\n");

    let thread_counts = vec![1, 2, 4];

    for thread_count in thread_counts {
        let config = CpuProviderConfig {
            thread_count: Some(thread_count),
            ..Default::default()
        };

        let provider = CpuExecutionProvider::with_config(config)?;

        // Compile a simple subgraph
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

        // Warm up
        for _ in 0..10 {
            let a = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
            let b = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
            kernel.execute(&[a, b])?;
        }

        // Measure performance
        let start = Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let a = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
            let b = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
            kernel.execute(&[a, b])?;
        }

        let elapsed = start.elapsed();

        println!(
            "{} threads: {:?} ({:.2} µs/op)",
            thread_count,
            elapsed,
            elapsed.as_micros() as f64 / iterations as f64
        );
    }

    Ok(())
}

#[test]
fn test_simd_optimization_impact() -> Result<()> {
    println!("\n=== SIMD Optimization Impact ===\n");

    // Test with SIMD enabled
    let config_simd = CpuProviderConfig {
        enable_simd: true,
        thread_count: Some(1), // Single thread to isolate SIMD impact
        ..Default::default()
    };

    let provider_simd = CpuExecutionProvider::with_config(config_simd)?;

    // Test with SIMD disabled
    let config_no_simd = CpuProviderConfig {
        enable_simd: false,
        thread_count: Some(1),
        ..Default::default()
    };

    let provider_no_simd = CpuExecutionProvider::with_config(config_no_simd)?;

    // Create test subgraph
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

    // Test with SIMD
    let kernel_simd = provider_simd.compile_subgraph(subgraph.clone())?;
    let start = Instant::now();
    for _ in 0..1000 {
        let a = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
        let b = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
        kernel_simd.execute(&[a, b])?;
    }
    let simd_time = start.elapsed();

    // Test without SIMD
    let kernel_no_simd = provider_no_simd.compile_subgraph(subgraph)?;
    let start = Instant::now();
    for _ in 0..1000 {
        let a = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
        let b = Tensor::ones(vec![1000], DataType::F32, TensorLayout::RowMajor)?;
        kernel_no_simd.execute(&[a, b])?;
    }
    let no_simd_time = start.elapsed();

    println!(
        "With SIMD:    {:?} ({:.2} µs/op)",
        simd_time,
        simd_time.as_micros() as f64 / 1000.0
    );
    println!(
        "Without SIMD: {:?} ({:.2} µs/op)",
        no_simd_time,
        no_simd_time.as_micros() as f64 / 1000.0
    );

    let speedup = no_simd_time.as_micros() as f64 / simd_time.as_micros() as f64;
    println!("SIMD speedup: {:.2}x\n", speedup);

    Ok(())
}

#[test]
fn test_provider_creation_overhead() -> Result<()> {
    println!("\n=== Provider Creation Overhead ===\n");

    // Measure CPU provider creation time
    let start = Instant::now();
    for _ in 0..100 {
        let _provider = create_cpu_provider()?;
    }
    let cpu_creation_time = start.elapsed();

    println!(
        "CPU provider creation: {:?} ({:.2} µs/op)",
        cpu_creation_time,
        cpu_creation_time.as_micros() as f64 / 100.0
    );

    // Creation should be fast (< 1ms per provider)
    assert!(cpu_creation_time.as_millis() < 100);

    Ok(())
}

#[test]
fn test_kernel_compilation_performance() -> Result<()> {
    println!("\n=== Kernel Compilation Performance ===\n");

    let provider = create_cpu_provider()?;

    // Test different subgraph complexities
    let complexities = vec![
        ("Simple (1 node)", 1),
        ("Medium (5 nodes)", 5),
        ("Complex (10 nodes)", 10),
    ];

    for (name, node_count) in complexities {
        let mut nodes = Vec::new();
        for i in 0..node_count {
            nodes.push(GraphNode {
                id: i,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: if i == 0 {
                    vec!["input1".to_string(), "input2".to_string()]
                } else {
                    vec![format!("temp{}", i - 1)]
                },
                outputs: vec![format!("temp{}", i)],
                name: Some(format!("add{}", i)),
            });
        }

        let subgraph = SubGraph {
            nodes,
            edges: vec![],
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec![format!("temp{}", node_count - 1)],
        };

        let start = Instant::now();
        let _kernel = provider.compile_subgraph(subgraph)?;
        let elapsed = start.elapsed();

        println!("{}: {:?}", name, elapsed);
    }

    Ok(())
}

// ============================================================================
// Memory Pool Performance Tests
// ============================================================================

#[test]
fn test_pool_hit_rate_vs_performance() -> Result<()> {
    println!("\n=== Pool Hit Rate vs Performance ===\n");

    let configs = vec![
        (
            "Small pool",
            PoolConfig {
                max_buffers_per_bucket: 2,
                max_pool_size: 1024 * 1024,
                bucket_granularity: 64,
            },
        ),
        (
            "Medium pool",
            PoolConfig {
                max_buffers_per_bucket: 8,
                max_pool_size: 10 * 1024 * 1024,
                bucket_granularity: 64,
            },
        ),
        (
            "Large pool",
            PoolConfig {
                max_buffers_per_bucket: 32,
                max_pool_size: 50 * 1024 * 1024,
                bucket_granularity: 64,
            },
        ),
    ];

    for (name, config) in configs {
        let allocator = PooledMemoryAllocator::new(config);

        let start = Instant::now();
        let sizes = vec![64, 128, 64, 256, 64, 128, 512, 64];

        for _ in 0..125 {
            for &size in &sizes {
                let buffer = allocator.allocate(&[size], DataType::F32)?;
                allocator.deallocate(buffer)?;
            }
        }

        let elapsed = start.elapsed();
        let hit_rate = allocator.get_hit_rate();

        println!(
            "{}: {:?} (hit rate: {:.1}%, {:.2} µs/op)",
            name,
            elapsed,
            hit_rate * 100.0,
            elapsed.as_micros() as f64 / (125.0 * 8.0)
        );
    }

    Ok(())
}

// ============================================================================
// Realistic Workload Tests
// ============================================================================

#[test]
fn test_inference_simulation() -> Result<()> {
    println!("\n=== Simulated Inference Workload ===\n");

    let provider = create_cpu_provider()?;

    // Simulate a simpler workload: Add operation
    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input".to_string(), "bias".to_string()],
            outputs: vec!["output".to_string()],
            name: Some("add".to_string()),
        }],
        edges: vec![],
        inputs: vec!["input".to_string(), "bias".to_string()],
        outputs: vec!["output".to_string()],
    };

    let kernel = provider.compile_subgraph(subgraph)?;

    // Warm up
    for _ in 0..10 {
        let input = Tensor::ones(vec![128], DataType::F32, TensorLayout::RowMajor)?;
        let bias = Tensor::ones(vec![128], DataType::F32, TensorLayout::RowMajor)?;
        kernel.execute(&[input, bias])?;
    }

    // Measure inference time
    let start = Instant::now();
    let batch_size = 100;

    for _ in 0..batch_size {
        let input = Tensor::ones(vec![128], DataType::F32, TensorLayout::RowMajor)?;
        let bias = Tensor::ones(vec![128], DataType::F32, TensorLayout::RowMajor)?;
        kernel.execute(&[input, bias])?;
    }

    let elapsed = start.elapsed();

    println!("Inference batch ({}): {:?}", batch_size, elapsed);
    println!(
        "Average latency: {:.2} ms",
        elapsed.as_micros() as f64 / (batch_size as f64 * 1000.0)
    );
    println!(
        "Throughput: {:.0} inferences/sec",
        batch_size as f64 / elapsed.as_secs_f64()
    );

    // Get kernel statistics
    let stats = kernel.get_performance_stats();
    println!(
        "Kernel stats: {} executions, avg time: {:.2} µs\n",
        stats.execution_count, stats.average_time_us
    );

    Ok(())
}

// ============================================================================
// Summary Statistics
// ============================================================================

#[test]
fn test_generate_performance_summary() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║           RONN Providers Performance Test Summary               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Run all performance tests above for detailed metrics.");
    println!("\nKey Performance Indicators:");
    println!("  - Allocator performance: System < Aligned ≈ Pooled (with hits)");
    println!("  - Pool hit rate target: ~28-30% for mixed workloads");
    println!("  - CPU provider creation: < 1ms");
    println!("  - SIMD speedup: 1.5-4x depending on operation and hardware");
    println!("  - Multi-threading scaling: Near-linear for independent ops\n");

    Ok(())
}
