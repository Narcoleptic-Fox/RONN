//! Multi-GPU Performance Benchmarks
//!
//! Comprehensive benchmarks for multi-GPU functionality including:
//! - Memory transfer patterns across multiple GPUs
//! - Custom CUDA kernel execution performance
//! - Topology-aware placement effectiveness
//! - Load balancing strategies
//! - Synchronization overhead measurements

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ronn_core::{DataType, GraphNode, SubGraph, Tensor, TensorLayout};
use ronn_providers::{
    BandwidthOptimizedPlacement, CudaKernelManager, GpuExecutionProvider, GpuTopologyManager,
    LocalityAwarePlacement, MultiGpuMemoryConfig, MultiGpuMemoryManager, PowerEfficientPlacement,
    SyncStrategy, TopologyConfig, Workload, WorkloadType, create_gpu_provider,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Benchmark data sizes for testing
const BENCHMARK_SIZES: &[usize] = &[
    1024,             // 1KB
    1024 * 16,        // 16KB
    1024 * 64,        // 64KB
    1024 * 256,       // 256KB
    1024 * 1024,      // 1MB
    1024 * 1024 * 4,  // 4MB
    1024 * 1024 * 16, // 16MB
    1024 * 1024 * 64, // 64MB
];

/// Multi-GPU memory transfer benchmarks
fn bench_memory_transfers(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Try to create GPU provider (skip if no GPU available)
    let provider = match create_gpu_provider() {
        Ok(p) => p,
        Err(_) => {
            println!("GPU not available, skipping memory transfer benchmarks");
            return;
        }
    };

    let config = MultiGpuMemoryConfig {
        enable_peer_to_peer: true,
        enable_unified_memory: false,
        memory_pool_size: 512 * 1024 * 1024, // 512MB per GPU
        sync_strategy: SyncStrategy::Async,
        enable_memory_prefetching: true,
    };

    let memory_manager = rt.block_on(async { MultiGpuMemoryManager::new(config).await });

    if memory_manager.is_err() {
        println!("Failed to create multi-GPU memory manager, skipping benchmarks");
        return;
    }

    let memory_manager = Arc::new(memory_manager.unwrap());

    let mut group = c.benchmark_group("memory_transfers");

    for &size in BENCHMARK_SIZES {
        let elements = size / std::mem::size_of::<f32>();

        // Single device allocation benchmark
        group.bench_with_input(
            BenchmarkId::new("single_device_alloc", size),
            &elements,
            |b, &elements| {
                b.iter(|| {
                    let tensor = rt.block_on(async {
                        let data = vec![1.0f32; elements];
                        let tensor = Tensor::from_data(
                            data,
                            vec![elements],
                            DataType::F32,
                            TensorLayout::RowMajor,
                        )
                        .unwrap();
                        memory_manager
                            .allocate_on_device(black_box(tensor), 0)
                            .await
                    });
                    black_box(tensor)
                });
            },
        );

        // Multi-device distribution benchmark
        group.bench_with_input(
            BenchmarkId::new("multi_device_distribute", size),
            &elements,
            |b, &elements| {
                b.iter(|| {
                    let result = rt.block_on(async {
                        let data = vec![1.0f32; elements];
                        let tensor = Tensor::from_data(
                            data,
                            vec![elements],
                            DataType::F32,
                            TensorLayout::RowMajor,
                        )
                        .unwrap();
                        memory_manager
                            .distribute_tensor(black_box(tensor), &[0, 1])
                            .await
                    });
                    black_box(result)
                });
            },
        );

        // Peer-to-peer transfer benchmark
        if rt
            .block_on(memory_manager.can_access_peer(0, 1))
            .unwrap_or(false)
        {
            group.bench_with_input(
                BenchmarkId::new("p2p_transfer", size),
                &elements,
                |b, &elements| {
                    let data = vec![1.0f32; elements];
                    let tensor = Tensor::from_data(
                        data,
                        vec![elements],
                        DataType::F32,
                        TensorLayout::RowMajor,
                    )
                    .unwrap();
                    let device_tensor = rt
                        .block_on(async { memory_manager.allocate_on_device(tensor, 0).await })
                        .unwrap();

                    b.iter(|| {
                        let result = rt.block_on(async {
                            memory_manager
                                .transfer_between_devices(&device_tensor, 0, black_box(1))
                                .await
                        });
                        black_box(result)
                    });
                },
            );
        }

        // Synchronization overhead benchmark
        group.bench_with_input(
            BenchmarkId::new("sync_overhead", size),
            &elements,
            |b, &elements| {
                b.iter(|| {
                    let result = rt.block_on(async {
                        let devices = vec![0, 1];
                        memory_manager
                            .synchronize_devices(black_box(&devices))
                            .await
                    });
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// CUDA kernel execution benchmarks
fn bench_cuda_kernels(c: &mut Criterion) {
    let kernel_manager = match CudaKernelManager::new() {
        Ok(km) => km,
        Err(_) => {
            println!("CUDA not available, skipping kernel benchmarks");
            return;
        }
    };

    // Compile test kernels
    let simple_kernel = r#"
        extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    "#;

    let fused_kernel = r#"
        extern "C" __global__ void vector_add_mul(float* a, float* b, float* c, float* d, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float temp = a[idx] + b[idx];
                d[idx] = temp * c[idx];
            }
        }
    "#;

    let simple_compiled = kernel_manager
        .compile_kernel(simple_kernel, "vector_add", &Default::default())
        .unwrap();
    let fused_compiled = kernel_manager
        .compile_kernel(fused_kernel, "vector_add_mul", &Default::default())
        .unwrap();

    let mut group = c.benchmark_group("cuda_kernels");

    for &size in BENCHMARK_SIZES {
        let elements = size / std::mem::size_of::<f32>();

        // Simple kernel benchmark
        group.bench_with_input(
            BenchmarkId::new("simple_kernel", size),
            &elements,
            |b, &elements| {
                let a = vec![1.0f32; elements];
                let b_vec = vec![2.0f32; elements];
                let c = vec![0.0f32; elements];

                b.iter(|| {
                    let result = simple_compiled.launch(
                        black_box(&[
                            a.as_ptr() as *const u8,
                            b_vec.as_ptr() as *const u8,
                            c.as_ptr() as *const u8,
                            &elements as *const usize as *const u8,
                        ]),
                        black_box((elements + 255) / 256),
                        black_box(256),
                    );
                    black_box(result)
                });
            },
        );

        // Fused kernel benchmark
        group.bench_with_input(
            BenchmarkId::new("fused_kernel", size),
            &elements,
            |b, &elements| {
                let a = vec![1.0f32; elements];
                let b_vec = vec![2.0f32; elements];
                let c = vec![3.0f32; elements];
                let d = vec![0.0f32; elements];

                b.iter(|| {
                    let result = fused_compiled.launch(
                        black_box(&[
                            a.as_ptr() as *const u8,
                            b_vec.as_ptr() as *const u8,
                            c.as_ptr() as *const u8,
                            d.as_ptr() as *const u8,
                            &elements as *const usize as *const u8,
                        ]),
                        black_box((elements + 255) / 256),
                        black_box(256),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Topology-aware placement benchmarks
fn bench_topology_placement(c: &mut Criterion) {
    let config = TopologyConfig {
        enable_numa_awareness: true,
        enable_bandwidth_profiling: true,
        enable_power_monitoring: false,
        profiling_duration_ms: 100,
        cache_topology_info: true,
    };

    let topology_manager = match GpuTopologyManager::new(config) {
        Ok(tm) => tm,
        Err(_) => {
            println!("Failed to create topology manager, skipping placement benchmarks");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Initialize topology
    if rt.block_on(topology_manager.discover_topology()).is_err() {
        println!("Failed to discover topology, skipping placement benchmarks");
        return;
    }

    let placement_strategies = vec![
        (
            "locality_aware",
            Box::new(LocalityAwarePlacement::new())
                as Box<dyn ronn_providers::PlacementStrategy + Send + Sync>,
        ),
        (
            "bandwidth_optimized",
            Box::new(BandwidthOptimizedPlacement::new()),
        ),
        ("power_efficient", Box::new(PowerEfficientPlacement::new())),
    ];

    let mut group = c.benchmark_group("topology_placement");

    // Different workload types for testing
    let workloads = vec![
        (
            "compute_intensive",
            Workload {
                id: "compute_test".to_string(),
                workload_type: WorkloadType::ComputeIntensive,
                estimated_compute_ops: 1_000_000,
                estimated_memory_usage: 256 * 1024 * 1024, // 256MB
                communication_pattern: ronn_providers::CommunicationPattern::AllToAll,
                priority: 1.0,
            },
        ),
        (
            "memory_bound",
            Workload {
                id: "memory_test".to_string(),
                workload_type: WorkloadType::MemoryBound,
                estimated_compute_ops: 10_000,
                estimated_memory_usage: 1024 * 1024 * 1024, // 1GB
                communication_pattern: ronn_providers::CommunicationPattern::Broadcast,
                priority: 1.0,
            },
        ),
        (
            "communication_heavy",
            Workload {
                id: "comm_test".to_string(),
                workload_type: WorkloadType::CommunicationHeavy,
                estimated_compute_ops: 100_000,
                estimated_memory_usage: 128 * 1024 * 1024, // 128MB
                communication_pattern: ronn_providers::CommunicationPattern::Ring,
                priority: 1.0,
            },
        ),
    ];

    for (strategy_name, strategy) in placement_strategies {
        for (workload_name, workload) in &workloads {
            group.bench_function(&format!("{}_{}", strategy_name, workload_name), |b| {
                b.iter(|| {
                    let topology = rt.block_on(topology_manager.get_topology()).unwrap();
                    let plan =
                        strategy.create_placement_plan(black_box(workload), black_box(&topology));
                    black_box(plan)
                });
            });
        }
    }

    group.finish();
}

/// End-to-end multi-GPU execution benchmarks
fn bench_end_to_end_execution(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let provider = match create_gpu_provider() {
        Ok(p) => p,
        Err(_) => {
            println!("GPU not available, skipping end-to-end benchmarks");
            return;
        }
    };

    let provider = Arc::new(provider);

    // Create test subgraph with multiple operations
    let subgraph = SubGraph {
        nodes: vec![
            GraphNode {
                id: 0,
                op_type: "MatMul".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["input1".to_string(), "input2".to_string()],
                outputs: vec!["temp1".to_string()],
                name: Some("matmul1".to_string()),
            },
            GraphNode {
                id: 1,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["temp1".to_string(), "input3".to_string()],
                outputs: vec!["temp2".to_string()],
                name: Some("add1".to_string()),
            },
            GraphNode {
                id: 2,
                op_type: "ReLU".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["temp2".to_string()],
                outputs: vec!["output1".to_string()],
                name: Some("relu1".to_string()),
            },
        ],
        edges: vec![],
        inputs: vec![
            "input1".to_string(),
            "input2".to_string(),
            "input3".to_string(),
        ],
        outputs: vec!["output1".to_string()],
    };

    let mut group = c.benchmark_group("end_to_end");

    // Different matrix sizes for benchmarking
    let matrix_sizes = &[64, 128, 256, 512, 1024];

    for &size in matrix_sizes {
        group.bench_with_input(
            BenchmarkId::new("single_gpu_execution", size),
            &size,
            |b, &size| {
                let input1 =
                    Tensor::ones(vec![size, size], DataType::F32, TensorLayout::RowMajor).unwrap();
                let input2 =
                    Tensor::ones(vec![size, size], DataType::F32, TensorLayout::RowMajor).unwrap();
                let input3 =
                    Tensor::ones(vec![size, size], DataType::F32, TensorLayout::RowMajor).unwrap();
                let inputs = vec![input1, input2, input3];

                b.iter(|| {
                    let result = rt.block_on(async {
                        provider
                            .execute_subgraph(black_box(&subgraph), black_box(&inputs))
                            .await
                    });
                    black_box(result)
                });
            },
        );

        // Multi-GPU execution with topology optimization
        if rt.block_on(async { provider.get_topology().is_some() }) {
            group.bench_with_input(
                BenchmarkId::new("multi_gpu_optimized", size),
                &size,
                |b, &size| {
                    let workload = Workload {
                        id: format!("benchmark_{}", size),
                        workload_type: WorkloadType::ComputeIntensive,
                        estimated_compute_ops: (size * size * size) as u64,
                        estimated_memory_usage: (size * size * std::mem::size_of::<f32>() * 3)
                            as u64,
                        communication_pattern:
                            ronn_providers::CommunicationPattern::PipelineParallel,
                        priority: 1.0,
                    };

                    let input1 =
                        Tensor::ones(vec![size, size], DataType::F32, TensorLayout::RowMajor)
                            .unwrap();
                    let input2 =
                        Tensor::ones(vec![size, size], DataType::F32, TensorLayout::RowMajor)
                            .unwrap();
                    let input3 =
                        Tensor::ones(vec![size, size], DataType::F32, TensorLayout::RowMajor)
                            .unwrap();
                    let inputs = vec![input1, input2, input3];

                    b.iter(|| {
                        let result = rt.block_on(async {
                            // First optimize placement
                            let devices = provider
                                .auto_select_devices(black_box(&workload))
                                .await
                                .unwrap_or(vec![0]);

                            // Then execute with optimized placement
                            provider
                                .execute_subgraph_on_devices(
                                    black_box(&subgraph),
                                    black_box(&inputs),
                                    black_box(&devices),
                                )
                                .await
                        });
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Load balancing strategy benchmarks
fn bench_load_balancing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let provider = match create_gpu_provider() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping load balancing benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("load_balancing");

    // Create multiple concurrent workloads
    let workloads = (0..8)
        .map(|i| {
            Workload {
                id: format!("workload_{}", i),
                workload_type: match i % 3 {
                    0 => WorkloadType::ComputeIntensive,
                    1 => WorkloadType::MemoryBound,
                    _ => WorkloadType::CommunicationHeavy,
                },
                estimated_compute_ops: 100_000 * (i + 1) as u64,
                estimated_memory_usage: 64 * 1024 * 1024 * (i + 1) as u64, // 64MB * (i+1)
                communication_pattern: ronn_providers::CommunicationPattern::AllToAll,
                priority: (i + 1) as f32 / 8.0,
            }
        })
        .collect::<Vec<_>>();

    // Sequential execution benchmark
    group.bench_function("sequential_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                for workload in black_box(&workloads) {
                    let devices = provider
                        .auto_select_devices(workload)
                        .await
                        .unwrap_or(vec![0]);
                    black_box(devices);
                }
            });
        });
    });

    // Concurrent load balancing benchmark
    group.bench_function("concurrent_load_balancing", |b| {
        b.iter(|| {
            rt.block_on(async {
                let tasks = workloads
                    .iter()
                    .map(|workload| {
                        let provider = provider.clone();
                        tokio::spawn(async move {
                            provider
                                .auto_select_devices(workload)
                                .await
                                .unwrap_or(vec![0])
                        })
                    })
                    .collect::<Vec<_>>();

                let results = futures::future::join_all(tasks).await;
                black_box(results);
            });
        });
    });

    group.finish();
}

criterion_group!(
    multi_gpu_benches,
    bench_memory_transfers,
    bench_cuda_kernels,
    bench_topology_placement,
    bench_end_to_end_execution,
    bench_load_balancing
);

criterion_main!(multi_gpu_benches);
