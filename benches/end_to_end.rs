//! End-to-End Benchmarks for RONN Runtime
//!
//! Comprehensive benchmarks covering the full inference pipeline:
//! - Model loading and initialization
//! - Graph optimization at different levels
//! - Inference execution with different providers
//! - Memory usage and throughput measurements
//!
//! Run with: cargo bench --bench end_to_end

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ronn_api::{Model, SessionOptions};
use ronn_core::{DataType, Tensor, TensorLayout};
use ronn_graph::OptimizationLevel;
// ProviderType not needed with new Model API
// use ronn_providers::ProviderType;
use std::path::PathBuf;
use std::time::Duration;

/// Helper to create test input tensors
fn create_test_input(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

/// Helper to get the first input name from a model
fn get_first_input_name(model: &Model) -> &str {
    model
        .input_names()
        .first()
        .expect("Model should have at least one input")
}

/// Benchmark full pipeline: Load → Optimize → Execute
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    // Configure measurement parameters
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Get path to test model
    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!(
            "Skipping full_pipeline benchmark: test model not found at {:?}",
            model_path
        );
        return;
    }

    // Test different optimization levels
    let optimization_levels = [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ];

    for opt_level in optimization_levels.iter() {
        group.bench_with_input(
            BenchmarkId::new("optimization_level", format!("{:?}", opt_level)),
            opt_level,
            |b, &opt_level| {
                b.iter(|| {
                    // Load model
                    let model = Model::load(black_box(&model_path)).unwrap();

                    // Create session with optimization level
                    let options = SessionOptions::new().with_optimization_level(opt_level);
                    let session = model.create_session(black_box(options)).unwrap();

                    // Create input (use first input name from model)
                    let input = create_test_input(vec![1, 3, 224, 224]);
                    let input_name = get_first_input_name(&model);

                    // Run inference
                    let mut inputs = std::collections::HashMap::new();
                    inputs.insert(input_name, input);
                    let _result = session.run(black_box(inputs)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark model loading time
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping model_loading benchmark: test model not found");
        return;
    }

    group.bench_function("load_simple_model", |b| {
        b.iter(|| {
            let model = Model::load(black_box(&model_path)).unwrap();
            let _session = model.create_session_default().unwrap();
        });
    });

    group.finish();
}

/// Benchmark inference latency with different batch sizes
fn bench_inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_latency");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping inference_latency benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();
    let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O2);
    let session = model.create_session(options).unwrap();
    let input_name = get_first_input_name(&model);

    // Test different batch sizes
    let batch_sizes = [1, 4, 8, 16, 32];

    for batch_size in batch_sizes.iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                let input = create_test_input(vec![batch_size, 3, 224, 224]);

                b.iter(|| {
                    let mut inputs = std::collections::HashMap::new();
                    inputs.insert(input_name, input.clone());
                    let _result = session.run(black_box(inputs)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark inference throughput (inferences per second)
fn bench_inference_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_throughput");

    // Increase measurement time for throughput tests
    group.measurement_time(Duration::from_secs(15));

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping inference_throughput benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();
    let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O3);
    let session = model.create_session(options).unwrap();
    let input = create_test_input(vec![1, 3, 224, 224]);
    let input_name = get_first_input_name(&model);

    group.bench_function("continuous_inference", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let mut inputs = std::collections::HashMap::new();
                inputs.insert(input_name, input.clone());
                let _result = session.run(black_box(inputs)).unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping memory_usage benchmark: test model not found");
        return;
    }

    // Test different tensor sizes to measure memory allocation patterns
    let tensor_sizes = [
        (1, 3, 64, 64),   // Small
        (1, 3, 224, 224), // Medium
        (1, 3, 512, 512), // Large
    ];

    for (i, &(batch, channels, height, width)) in tensor_sizes.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("tensor_size", i),
            &(batch, channels, height, width),
            |b, &(batch, channels, height, width)| {
                let model = Model::load(&model_path).unwrap();
                let session = model.create_session_default().unwrap();
                let input_name = get_first_input_name(&model);

                b.iter(|| {
                    let input = create_test_input(vec![batch, channels, height, width]);
                    let mut inputs = std::collections::HashMap::new();
                    inputs.insert(input_name, input);
                    let _result = session.run(black_box(inputs)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cold start vs warm start performance
fn bench_cold_vs_warm_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_vs_warm");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping cold_vs_warm benchmark: test model not found");
        return;
    }

    // Cold start: create new session each time
    group.bench_function("cold_start", |b| {
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let model = Model::load(black_box(&model_path)).unwrap();
            let session = model.create_session_default().unwrap();
            let input_name = get_first_input_name(&model);
            let mut inputs = std::collections::HashMap::new();
            inputs.insert(input_name, input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    // Warm start: reuse session
    group.bench_function("warm_start", |b| {
        let model = Model::load(&model_path).unwrap();
        let session = model.create_session_default().unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);
        let input_name = get_first_input_name(&model);

        b.iter(|| {
            let mut inputs = std::collections::HashMap::new();
            inputs.insert(input_name, input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    group.finish();
}

/// Benchmark optimization pass overhead
fn bench_optimization_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_overhead");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping optimization_overhead benchmark: test model not found");
        return;
    }

    let optimization_levels = [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ];

    for opt_level in optimization_levels.iter() {
        group.bench_with_input(
            BenchmarkId::new("opt_level", format!("{:?}", opt_level)),
            opt_level,
            |b, &opt_level| {
                b.iter(|| {
                    let model = Model::load(black_box(&model_path)).unwrap();
                    let options = SessionOptions::new().with_optimization_level(opt_level);
                    let _session = model.create_session(black_box(options)).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    end_to_end_benches,
    bench_full_pipeline,
    bench_model_loading,
    bench_inference_latency,
    bench_inference_throughput,
    bench_memory_usage,
    bench_cold_vs_warm_start,
    bench_optimization_overhead
);

criterion_main!(end_to_end_benches);
