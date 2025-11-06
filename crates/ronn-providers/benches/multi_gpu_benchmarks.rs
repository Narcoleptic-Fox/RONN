//! Multi-GPU Performance Benchmarks
//!
//! Comprehensive benchmarks for multi-GPU functionality including:
//! - GPU capability queries
//! - Multi-GPU statistics
//! - Topology information retrieval
//! - Load balancing and device selection
//! - Configuration access patterns
//!
//! Note: Many benchmarks require actual GPU hardware and may be skipped
//! if GPU support is not available or compiled without the `gpu` feature.
//!
//! These benchmarks focus on GPU provider management and metadata operations
//! rather than actual tensor operations, which require complex Candle tensor
//! conversions.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ronn_providers::GpuExecutionProvider;
use std::sync::Arc;

/// GPU capabilities and configuration benchmarks
fn bench_gpu_capabilities(c: &mut Criterion) {
    let provider = match GpuExecutionProvider::new() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping capability benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_capabilities");

    // Tensor cores availability
    group.bench_function("has_tensor_cores", |b| {
        b.iter(|| {
            let has_tc = provider.has_tensor_cores();
            black_box(has_tc)
        });
    });

    // GPU memory info
    group.bench_function("get_gpu_memory_info", |b| {
        b.iter(|| {
            let info = provider.get_gpu_memory_info();
            black_box(info)
        });
    });

    // Configuration access
    group.bench_function("get_config", |b| {
        b.iter(|| {
            let config = provider.get_config();
            black_box(config)
        });
    });

    // Operation support checks
    let op_types = vec!["MatMul", "Add", "ReLU", "Conv", "BatchNorm"];
    for op_type in op_types {
        group.bench_function(&format!("supports_{}", op_type), |b| {
            b.iter(|| {
                let supports = provider.supports_operation(black_box(op_type));
                black_box(supports)
            });
        });
    }

    // Device access
    group.bench_function("get_device", |b| {
        b.iter(|| {
            let device = provider.device();
            black_box(device)
        });
    });

    group.finish();
}

/// Load balancing and device selection benchmarks
fn bench_load_balancing(c: &mut Criterion) {
    let provider = match GpuExecutionProvider::new() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping load balancing benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("load_balancing");

    // Device count
    group.bench_function("device_count", |b| {
        b.iter(|| {
            let count = provider.device_count();
            black_box(count)
        });
    });

    // Device availability checks
    for device_id in 0..4 {
        group.bench_function(&format!("has_device_{}", device_id), |b| {
            b.iter(|| {
                let has_device = provider.has_device(black_box(device_id));
                black_box(has_device)
            });
        });
    }

    // Multi-GPU statistics (if multi-GPU available)
    if provider.device_count() > 1 {
        group.bench_function("get_multi_gpu_stats", |b| {
            b.iter(|| {
                let stats = provider.get_multi_gpu_stats();
                black_box(stats)
            });
        });

        // Global memory statistics
        group.bench_function("get_global_memory_stats", |b| {
            b.iter(|| {
                let stats = provider.get_global_memory_stats();
                black_box(stats)
            });
        });

        // P2P connectivity
        group.bench_function("get_p2p_connectivity", |b| {
            b.iter(|| {
                let connectivity = provider.get_p2p_connectivity();
                black_box(connectivity)
            });
        });

        // P2P availability checks
        group.bench_function("is_p2p_available", |b| {
            b.iter(|| {
                let available = provider.is_p2p_available(black_box(0), black_box(1));
                black_box(available)
            });
        });
    }

    group.finish();
}

/// Memory management query benchmarks
fn bench_memory_operations(c: &mut Criterion) {
    let provider = match GpuExecutionProvider::new() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping memory operation benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("memory_operations");

    // Memory statistics retrieval
    group.bench_function("get_memory_statistics", |b| {
        b.iter(|| {
            let stats = provider.get_memory_statistics();
            black_box(stats)
        });
    });

    // Memory synchronization
    group.bench_function("synchronize_memory", |b| {
        b.iter(|| {
            let result = provider.synchronize_memory();
            black_box(result)
        });
    });

    group.finish();
}

/// Custom CUDA kernel capability benchmarks
fn bench_custom_kernels(c: &mut Criterion) {
    let provider = match GpuExecutionProvider::new() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping custom kernel benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("custom_kernels");

    // Check if custom kernels are available
    for device_id in 0..provider.device_count().min(4) {
        group.bench_function(&format!("has_custom_kernels_{}", device_id), |b| {
            b.iter(|| {
                let has_kernels = provider.has_custom_kernels(black_box(device_id));
                black_box(has_kernels)
            });
        });

        // Get available ops if kernels available
        if provider.has_custom_kernels(device_id) {
            group.bench_function(&format!("get_custom_kernel_ops_{}", device_id), |b| {
                b.iter(|| {
                    let ops = provider.get_custom_kernel_ops(black_box(device_id));
                    black_box(ops)
                });
            });
        }
    }

    // Kernel cache statistics
    group.bench_function("kernel_cache_stats", |b| {
        b.iter(|| {
            let stats = provider.get_kernel_cache_stats();
            black_box(stats)
        });
    });

    // Clear kernel caches (only bench if safe)
    group.bench_function("clear_kernel_caches", |b| {
        b.iter(|| {
            provider.clear_kernel_caches();
        });
    });

    group.finish();
}

/// Topology-aware workload optimization benchmarks
fn bench_topology_optimization(c: &mut Criterion) {
    let provider = match GpuExecutionProvider::new() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping topology benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("topology_optimization");

    // Get topology information
    group.bench_function("get_topology", |b| {
        b.iter(|| {
            let topology = provider.get_topology();
            black_box(topology)
        });
    });

    // Only run workload optimization benchmarks if topology is available
    let topology = provider.get_topology();
    if topology.is_none() {
        println!("GPU topology information not available, skipping workload benchmarks");
        group.finish();
        return;
    }

    // Create test workloads with new API structure
    use ronn_providers::gpu::topology::{
        ComputeRequirements, CommunicationPattern, CommunicationType, MemoryAccessType,
        MemoryPattern, PrecisionPreference, Workload, WorkloadConstraints, WorkloadType,
    };

    let workloads = vec![
        (
            "compute_intensive",
            Workload {
                id: "compute_test".to_string(),
                workload_type: WorkloadType::Inference,
                compute_requirements: ComputeRequirements {
                    min_compute_capability: "7.0".to_string(),
                    memory_per_device: 256 * 1024 * 1024, // 256MB
                    preferred_device_count: 1,
                    benefits_from_tensor_cores: true,
                    precision_preference: PrecisionPreference::FP32,
                },
                memory_patterns: vec![MemoryPattern {
                    pattern_type: MemoryAccessType::Sequential,
                    data_size: 256 * 1024 * 1024,
                    frequency: 1.0,
                    shared_across_devices: false,
                }],
                communication_patterns: vec![],
                constraints: WorkloadConstraints {
                    max_latency_ms: Some(10.0),
                    min_throughput: None,
                    power_budget: None,
                    fault_tolerance: false,
                },
            },
        ),
        (
            "memory_bound",
            Workload {
                id: "memory_test".to_string(),
                workload_type: WorkloadType::Training,
                compute_requirements: ComputeRequirements {
                    min_compute_capability: "7.0".to_string(),
                    memory_per_device: 1024 * 1024 * 1024, // 1GB
                    preferred_device_count: 2,
                    benefits_from_tensor_cores: false,
                    precision_preference: PrecisionPreference::FP16,
                },
                memory_patterns: vec![MemoryPattern {
                    pattern_type: MemoryAccessType::Random,
                    data_size: 1024 * 1024 * 1024,
                    frequency: 0.8,
                    shared_across_devices: true,
                }],
                communication_patterns: vec![CommunicationPattern {
                    src_device: Some(0),
                    dst_device: Some(1),
                    comm_type: CommunicationType::P2P,
                    data_volume: 128 * 1024 * 1024,
                    frequency: 0.5,
                }],
                constraints: WorkloadConstraints {
                    max_latency_ms: Some(50.0),
                    min_throughput: Some(100.0),
                    power_budget: Some(300.0),
                    fault_tolerance: false,
                },
            },
        ),
    ];

    for (workload_name, workload) in &workloads {
        // Optimize workload placement with default strategy
        group.bench_function(&format!("optimize_placement_{}", workload_name), |b| {
            b.iter(|| {
                let result = provider.optimize_workload_placement(black_box(workload), black_box("locality_aware"));
                black_box(result)
            });
        });

        // Compare placement strategies
        group.bench_function(&format!("compare_strategies_{}", workload_name), |b| {
            let strategies = vec![
                "locality_aware".to_string(),
                "bandwidth_optimized".to_string(),
                "power_efficient".to_string(),
            ];
            b.iter(|| {
                let result = provider.compare_placement_strategies(black_box(workload), black_box(&strategies));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    multi_gpu_benches,
    bench_gpu_capabilities,
    bench_load_balancing,
    bench_memory_operations,
    bench_custom_kernels,
    bench_topology_optimization
);

criterion_main!(multi_gpu_benches);
