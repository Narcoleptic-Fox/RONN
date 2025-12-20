//! Core Operations Benchmark
//!
//! Measures performance of fundamental RONN operations:
//! - Tensor creation and manipulation
//! - HRM complexity assessment and routing
//! - Provider-specific operations
//!
//! Run with: cargo bench --bench core_operations

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ronn_core::{DataType, Tensor, TensorLayout};
use ronn_hrm::{HierarchicalReasoningModule, RoutingStrategy};
use std::time::Duration;

/// Benchmark tensor creation across different sizes
fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    let sizes = vec![
        (vec![1, 10], "tiny"),
        (vec![1, 100], "small"),
        (vec![1, 1000], "medium"),
        (vec![1, 10000], "large"),
    ];

    for (shape, name) in sizes {
        let numel: usize = shape.iter().product();
        group.throughput(Throughput::Elements(numel as u64));

        group.bench_with_input(BenchmarkId::new("from_vec", name), &shape, |b, shape| {
            let data: Vec<f32> = (0..shape.iter().product::<usize>())
                .map(|x| x as f32)
                .collect();

            b.iter(|| {
                Tensor::from_data(
                    black_box(data.clone()),
                    black_box(shape.clone()),
                    DataType::F32,
                    TensorLayout::RowMajor,
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark tensor operations (add, multiply, etc.)
fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");

    let shape = vec![1, 1000];
    let data1: Vec<f32> = (0..1000).map(|x| x as f32).collect();
    let data2: Vec<f32> = (0..1000).map(|x| (x as f32) * 2.0).collect();

    let tensor1 =
        Tensor::from_data(data1, shape.clone(), DataType::F32, TensorLayout::RowMajor).unwrap();

    let _tensor2 =
        Tensor::from_data(data2, shape.clone(), DataType::F32, TensorLayout::RowMajor).unwrap();

    group.throughput(Throughput::Elements(1000));

    group.bench_function("to_vec", |b| {
        b.iter(|| {
            let _ = black_box(&tensor1).to_vec().unwrap();
        });
    });

    group.bench_function("clone", |b| {
        b.iter(|| {
            let _ = black_box(&tensor1).clone();
        });
    });

    group.finish();
}

/// Benchmark HRM complexity assessment
fn bench_hrm_complexity_assessment(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrm_complexity_assessment");

    let mut hrm = HierarchicalReasoningModule::new();

    let test_cases = vec![
        (vec![1, 4], "tiny"),
        (vec![1, 16], "small"),
        (vec![1, 64], "medium"),
        (vec![1, 256], "large"),
    ];

    for (shape, name) in test_cases {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|x| (x as f32).sin()).collect();

        let tensor =
            Tensor::from_data(data, shape.clone(), DataType::F32, TensorLayout::RowMajor).unwrap();

        group.throughput(Throughput::Elements(numel as u64));

        group.bench_with_input(
            BenchmarkId::new("assess_and_route", name),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let _ = black_box(&mut hrm).process(black_box(tensor)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark HRM routing strategies
fn bench_hrm_routing_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrm_routing_strategies");

    let strategies = vec![
        (RoutingStrategy::AlwaysSystem1, "always_system1"),
        (RoutingStrategy::AlwaysSystem2, "always_system2"),
        (RoutingStrategy::AdaptiveComplexity, "adaptive_complexity"),
        (RoutingStrategy::AdaptiveHybrid, "adaptive_hybrid"),
    ];

    let shape = vec![1, 64];
    let data: Vec<f32> = (0..64)
        .map(|x| (x as f32).sin() * (x as f32).cos())
        .collect();

    let tensor = Tensor::from_data(data, shape, DataType::F32, TensorLayout::RowMajor).unwrap();

    group.throughput(Throughput::Elements(64));

    for (strategy, name) in strategies {
        group.bench_with_input(
            BenchmarkId::new("strategy", name),
            &strategy,
            |b, &strategy| {
                let mut hrm = HierarchicalReasoningModule::with_strategy(strategy);
                b.iter(|| {
                    let _ = black_box(&mut hrm).process(black_box(&tensor)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark end-to-end HRM processing at scale
fn bench_hrm_e2e_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("hrm_e2e_processing");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    let mut hrm = HierarchicalReasoningModule::with_strategy(RoutingStrategy::AdaptiveComplexity);

    // Batch processing scenario
    let batch_sizes = vec![1, 10, 100];

    for batch_size in batch_sizes {
        let shape = vec![batch_size, 256];
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|x| (x as f32 * 0.01).sin()).collect();

        let tensor = Tensor::from_data(data, shape, DataType::F32, TensorLayout::RowMajor).unwrap();

        group.throughput(Throughput::Elements(numel as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_process", batch_size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let _ = black_box(&mut hrm).process(black_box(tensor)).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_tensor_operations,
    bench_hrm_complexity_assessment,
    bench_hrm_routing_strategies,
    bench_hrm_e2e_processing
);

criterion_main!(benches);
