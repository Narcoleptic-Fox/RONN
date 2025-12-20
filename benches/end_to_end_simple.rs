//! Simplified End-to-End Benchmarks for RONN Runtime
//!
//! This is a simplified version that works with the current API.
//! Run with: cargo bench --bench end_to_end_simple

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ronn_api::{Model, SessionOptions};
use ronn_core::{DataType, Tensor, TensorLayout};
use ronn_graph::OptimizationLevel;
use std::collections::HashMap;
use std::path::PathBuf;

/// Helper to create test input tensors
fn create_test_input(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

/// Helper to prepare inputs
fn prepare_inputs(tensor: Tensor) -> HashMap<&'static str, Tensor> {
    let mut inputs = HashMap::new();
    inputs.insert("input", tensor);
    inputs
}

/// Benchmark model loading
fn bench_model_loading(c: &mut Criterion) {
    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!(
            "Skipping model_loading benchmark: test model not found at {:?}",
            model_path
        );
        return;
    }

    c.bench_function("model_loading", |b| {
        b.iter(|| {
            let _model = Model::load(black_box(&model_path)).unwrap();
        });
    });
}

/// Benchmark session creation with different optimization levels
fn bench_session_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("session_creation");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping session_creation benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();

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
                    let options = SessionOptions::new().with_optimization_level(opt_level);
                    let _session = model.create_session(black_box(options)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark inference latency
fn bench_inference_latency(c: &mut Criterion) {
    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping inference_latency benchmark: test model not found");
        return;
    }

    let model = Model::load(&model_path).unwrap();
    let session = model.create_session_default().unwrap();

    let input = create_test_input(vec![1, 10]); // Simple input shape

    c.bench_function("inference_latency", |b| {
        let inputs = prepare_inputs(input.clone());
        b.iter(|| {
            let _result = session.run(black_box(inputs.clone())).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_model_loading,
    bench_session_creation,
    bench_inference_latency
);

criterion_main!(benches);
