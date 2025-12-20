//! Comparative benchmarks against ONNX Runtime
//!
//! This module compares RONN's performance against the official ONNX Runtime.
//! These benchmarks are only available when the `comparative` feature is enabled.
//!
//! Run with: cargo bench --bench comparative --features comparative

#[cfg(feature = "comparative")]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[cfg(feature = "comparative")]
use ronn_api::{Environment, InferenceSession, SessionConfig};

#[cfg(feature = "comparative")]
use ronn_core::{DataType, Tensor, TensorLayout};

#[cfg(feature = "comparative")]
use std::path::PathBuf;

#[cfg(feature = "comparative")]
/// Helper to create test tensors for both runtimes
fn create_test_tensor(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

#[cfg(feature = "comparative")]
/// Benchmark inference latency: RONN vs ONNX Runtime
fn bench_inference_latency_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_comparison");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping comparison benchmark: test model not found");
        return;
    }

    // RONN inference
    group.bench_function("ronn_inference", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let config = SessionConfig::default();
        let session = InferenceSession::new(&env, &model_path, config).unwrap();
        let input = create_test_tensor(vec![1, 3, 224, 224]);

        b.iter(|| {
            let _result = session.run(black_box(vec![input.clone()])).unwrap();
        });
    });

    // ONNX Runtime inference
    group.bench_function("onnx_runtime_inference", |b| {
        use onnxruntime::{
            GraphOptimizationLevel, environment::Environment as OrtEnv,
            session::Session as OrtSession, tensor::OrtOwnedTensor,
        };

        let ort_env = OrtEnv::builder()
            .with_name("ort_benchmark")
            .build()
            .unwrap();

        let session = ort_env
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(&model_path)
            .unwrap();

        // Create input array matching RONN's test input
        let input_shape = vec![1, 3, 224, 224];
        let total_elements: usize = input_shape.iter().product();
        let input_data: Vec<f32> = vec![1.0; total_elements];

        b.iter(|| {
            let input_tensor =
                ndarray::Array::from_shape_vec(input_shape.clone(), input_data.clone()).unwrap();

            let _outputs: Vec<OrtOwnedTensor<f32, _>> =
                session.run(vec![black_box(input_tensor)]).unwrap();
        });
    });

    group.finish();
}

#[cfg(feature = "comparative")]
/// Benchmark model loading time: RONN vs ONNX Runtime
fn bench_model_loading_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("loading_comparison");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping loading comparison: test model not found");
        return;
    }

    // RONN model loading
    group.bench_function("ronn_loading", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let config = SessionConfig::default();

        b.iter(|| {
            let _session =
                InferenceSession::new(&env, black_box(&model_path), black_box(config.clone()))
                    .unwrap();
        });
    });

    // ONNX Runtime model loading
    group.bench_function("onnx_runtime_loading", |b| {
        use onnxruntime::environment::Environment as OrtEnv;

        let ort_env = OrtEnv::builder()
            .with_name("ort_benchmark")
            .build()
            .unwrap();

        b.iter(|| {
            let _session = ort_env
                .new_session_builder()
                .unwrap()
                .with_model_from_file(black_box(&model_path))
                .unwrap();
        });
    });

    group.finish();
}

#[cfg(feature = "comparative")]
/// Benchmark throughput: RONN vs ONNX Runtime
fn bench_throughput_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_comparison");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping throughput comparison: test model not found");
        return;
    }

    let num_iterations = 100;

    // RONN throughput
    group.bench_function("ronn_throughput", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let config = SessionConfig::default();
        let session = InferenceSession::new(&env, &model_path, config).unwrap();
        let input = create_test_tensor(vec![1, 3, 224, 224]);

        b.iter(|| {
            for _ in 0..num_iterations {
                let _result = session.run(black_box(vec![input.clone()])).unwrap();
            }
        });
    });

    // ONNX Runtime throughput
    group.bench_function("onnx_runtime_throughput", |b| {
        use onnxruntime::{environment::Environment as OrtEnv, tensor::OrtOwnedTensor};

        let ort_env = OrtEnv::builder()
            .with_name("ort_benchmark")
            .build()
            .unwrap();

        let session = ort_env
            .new_session_builder()
            .unwrap()
            .with_model_from_file(&model_path)
            .unwrap();

        let input_shape = vec![1, 3, 224, 224];
        let total_elements: usize = input_shape.iter().product();
        let input_data: Vec<f32> = vec![1.0; total_elements];

        b.iter(|| {
            for _ in 0..num_iterations {
                let input_tensor =
                    ndarray::Array::from_shape_vec(input_shape.clone(), input_data.clone())
                        .unwrap();

                let _outputs: Vec<OrtOwnedTensor<f32, _>> =
                    session.run(vec![black_box(input_tensor)]).unwrap();
            }
        });
    });

    group.finish();
}

#[cfg(feature = "comparative")]
/// Benchmark different batch sizes: RONN vs ONNX Runtime
fn bench_batch_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_comparison");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping batch size comparison: test model not found");
        return;
    }

    let batch_sizes = [1, 4, 8, 16];

    for batch_size in batch_sizes.iter() {
        // RONN with different batch sizes
        group.bench_with_input(
            BenchmarkId::new("ronn", batch_size),
            batch_size,
            |b, &batch_size| {
                let env = Environment::new("benchmark_env").unwrap();
                let config = SessionConfig::default();
                let session = InferenceSession::new(&env, &model_path, config).unwrap();
                let input = create_test_tensor(vec![batch_size, 3, 224, 224]);

                b.iter(|| {
                    let _result = session.run(black_box(vec![input.clone()])).unwrap();
                });
            },
        );

        // ONNX Runtime with different batch sizes
        group.bench_with_input(
            BenchmarkId::new("onnx_runtime", batch_size),
            batch_size,
            |b, &batch_size| {
                use onnxruntime::{environment::Environment as OrtEnv, tensor::OrtOwnedTensor};

                let ort_env = OrtEnv::builder()
                    .with_name("ort_benchmark")
                    .build()
                    .unwrap();

                let session = ort_env
                    .new_session_builder()
                    .unwrap()
                    .with_model_from_file(&model_path)
                    .unwrap();

                let input_shape = vec![batch_size, 3, 224, 224];
                let total_elements: usize = input_shape.iter().product();
                let input_data: Vec<f32> = vec![1.0; total_elements];

                b.iter(|| {
                    let input_tensor =
                        ndarray::Array::from_shape_vec(input_shape.clone(), input_data.clone())
                            .unwrap();

                    let _outputs: Vec<OrtOwnedTensor<f32, _>> =
                        session.run(vec![black_box(input_tensor)]).unwrap();
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "comparative")]
criterion_group!(
    comparative_benches,
    bench_inference_latency_comparison,
    bench_model_loading_comparison,
    bench_throughput_comparison,
    bench_batch_size_comparison
);

#[cfg(feature = "comparative")]
criterion_main!(comparative_benches);

#[cfg(not(feature = "comparative"))]
fn main() {
    eprintln!("Comparative benchmarks require the 'comparative' feature flag.");
    eprintln!("Run with: cargo bench --bench comparative --features comparative");
}
