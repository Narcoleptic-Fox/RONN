//! Baseline Performance Regression Tracking
//!
//! This module tracks performance against the targets defined in TASKS.md:
//! - Latency: <10ms P50, <30ms P95 for inference
//! - Memory: <4GB total system usage
//! - Binary Size: <50MB inference, <200MB full system
//! - Throughput: >1000 inferences/second on 16-core CPU
//!
//! Run with: cargo bench --bench regression

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group};
use ronn_api::{Model, SessionOptions};
use ronn_core::{DataType, Tensor, TensorLayout};
use ronn_graph::OptimizationLevel;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Performance targets from TASKS.md
pub struct PerformanceTargets {
    /// P50 latency target in milliseconds
    pub latency_p50_ms: f64,
    /// P95 latency target in milliseconds
    pub latency_p95_ms: f64,
    /// Throughput target in inferences per second
    pub throughput_target: f64,
    /// Memory usage target in GB
    pub memory_usage_gb: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            latency_p50_ms: 10.0,
            latency_p95_ms: 30.0,
            throughput_target: 1000.0,
            memory_usage_gb: 4.0,
        }
    }
}

/// Helper to create test input
fn create_test_input(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

/// Benchmark latency against P50 and P95 targets
pub fn bench_latency_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_targets");

    // Configure to capture percentiles
    group.significance_level(0.1).sample_size(1000);

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping latency target benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();
    let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O2);

    let session = model.create_session(options).unwrap();
    let input = create_test_input(vec![1, 3, 224, 224]);

    group.bench_function("inference_latency", |b| {
        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    // Print target information
    let targets = PerformanceTargets::default();
    eprintln!("\n=== Latency Targets ===");
    eprintln!("P50 target: <{} ms", targets.latency_p50_ms);
    eprintln!("P95 target: <{} ms", targets.latency_p95_ms);
    eprintln!("Check criterion HTML report for actual percentiles");

    group.finish();
}

/// Benchmark throughput against target (>1000 inferences/sec)
pub fn bench_throughput_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_targets");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping throughput target benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();
    let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O3);

    let session = model.create_session(options).unwrap();
    let input = create_test_input(vec![1, 3, 224, 224]);

    // Measure throughput over 10 seconds
    group.measurement_time(Duration::from_secs(10));

    // Set throughput to measure inferences per second
    group.throughput(Throughput::Elements(1));

    group.bench_function("continuous_inference", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut inputs = HashMap::new();
                inputs.insert("input", input.clone());
                let _result = session.run(black_box(inputs)).unwrap();
            }
            start.elapsed()
        });
    });

    let targets = PerformanceTargets::default();
    eprintln!("\n=== Throughput Target ===");
    eprintln!("Target: >{} inferences/second", targets.throughput_target);
    eprintln!("Check criterion report for actual throughput");

    group.finish();
}

/// Track memory usage patterns
pub fn bench_memory_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_targets");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping memory target benchmark: test model not found");
        return;
    }

    // Test different model sizes
    let test_sizes = vec![
        ("small", vec![1, 3, 64, 64]),
        ("medium", vec![1, 3, 224, 224]),
        ("large", vec![1, 3, 512, 512]),
    ];

    for (size_name, shape) in test_sizes {
        group.bench_with_input(
            BenchmarkId::new("memory_usage", size_name),
            &shape,
            |b, shape| {
                let model = Model::load(&model_path).unwrap();
                let session = model.create_session_default().unwrap();

                b.iter(|| {
                    let input = create_test_input(shape.clone());
                    let mut inputs = HashMap::new();
                    inputs.insert("input", input);
                    let _result = session.run(black_box(inputs)).unwrap();
                });
            },
        );
    }

    let targets = PerformanceTargets::default();
    eprintln!("\n=== Memory Target ===");
    eprintln!("Target: <{} GB total system usage", targets.memory_usage_gb);
    eprintln!(
        "Note: Use external profilers (valgrind, heaptrack) for accurate memory measurements"
    );

    group.finish();
}

/// Benchmark session creation overhead
pub fn bench_initialization_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("initialization_overhead");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping initialization overhead benchmark: test model not found");
        return;
    }

    group.bench_function("session_creation", |b| {
        b.iter(|| {
            let model = Model::load(black_box(&model_path)).unwrap();
            let _session = model.create_session_default().unwrap();
        });
    });

    eprintln!("\n=== Initialization Overhead ===");
    eprintln!("Lower is better - minimize cold start time");

    group.finish();
}

/// Benchmark batch processing efficiency
pub fn bench_batch_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_efficiency");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping batch efficiency benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();
    let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O2);

    let session = model.create_session(options).unwrap();

    // Test different batch sizes
    let batch_sizes = [1, 2, 4, 8, 16, 32];

    for batch_size in batch_sizes.iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                let input = create_test_input(vec![batch_size, 3, 224, 224]);

                b.iter(|| {
                    let mut inputs = HashMap::new();
                    inputs.insert("input", input.clone());
                    let _result = session.run(black_box(inputs)).unwrap();
                });
            },
        );
    }

    eprintln!("\n=== Batch Efficiency ===");
    eprintln!("Larger batches should have better throughput per item");

    group.finish();
}

/// Benchmark optimization level impact on performance
pub fn bench_optimization_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_regression");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping optimization regression benchmark: test model not found");
        return;
    }

    let optimization_levels = [
        ("O0", OptimizationLevel::O0),
        ("O1", OptimizationLevel::O1),
        ("O2", OptimizationLevel::O2),
        ("O3", OptimizationLevel::O3),
    ];

    for (name, level) in optimization_levels.iter() {
        group.bench_with_input(BenchmarkId::new("opt_level", name), level, |b, &level| {
            let model = Model::load(&model_path).unwrap();
            let options = SessionOptions::new().with_optimization_level(level);

            let session = model.create_session(options).unwrap();
            let input = create_test_input(vec![1, 3, 224, 224]);

            b.iter(|| {
                let mut inputs = HashMap::new();
                inputs.insert("input", input.clone());
                let _result = session.run(black_box(inputs)).unwrap();
            });
        });
    }

    eprintln!("\n=== Optimization Regression ===");
    eprintln!("Higher optimization levels should show improved performance");

    group.finish();
}

/// Track allocation patterns to prevent regressions
pub fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping allocation patterns benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();
    let session = model.create_session_default().unwrap();

    // Single inference - measure allocation overhead
    group.bench_function("single_inference_allocation", |b| {
        b.iter(|| {
            let input = create_test_input(vec![1, 3, 224, 224]);
            let mut inputs = HashMap::new();
            inputs.insert("input", input);
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    // Reused tensor - should have less allocation overhead
    group.bench_function("reused_tensor_allocation", |b| {
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    eprintln!("\n=== Allocation Patterns ===");
    eprintln!("Reused tensors should show reduced allocation overhead");

    group.finish();
}

/// End-to-end performance baseline
pub fn bench_e2e_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_baseline");

    // Increase measurement time for stable baselines
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(100);

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping e2e baseline benchmark: test model not found");
        return;
    }

    group.bench_function("full_pipeline_baseline", |b| {
        b.iter(|| {
            // Full pipeline: load model, create session, run inference
            let model = Model::load(black_box(&model_path)).unwrap();
            let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O2);

            let session = model.create_session(options).unwrap();
            let input = create_test_input(vec![1, 3, 224, 224]);
            let mut inputs = HashMap::new();
            inputs.insert("input", input);
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    eprintln!("\n=== End-to-End Baseline ===");
    eprintln!("This is the baseline for detecting any performance regressions");
    eprintln!("Track this metric over time to ensure consistent performance");

    group.finish();
}

criterion_group!(
    regression_benches,
    bench_latency_target,
    bench_throughput_target,
    bench_memory_target,
    bench_initialization_overhead,
    bench_batch_efficiency,
    bench_optimization_regression,
    bench_allocation_patterns,
    bench_e2e_baseline
);
