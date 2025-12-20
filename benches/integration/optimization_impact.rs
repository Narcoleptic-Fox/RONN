//! Optimization Impact Benchmarks
//!
//! Measures the effectiveness of different graph optimization passes:
//! - Constant folding
//! - Node fusion
//! - Layout optimization
//! - Dead code elimination
//! - Provider-specific optimizations
//!
//! Run with: cargo bench --bench integration

use criterion::{BenchmarkId, Criterion, black_box, criterion_group};
use ronn_api::{Model, SessionOptions};
use ronn_core::{DataType, GraphBuilder, ModelGraph, Tensor, TensorLayout};
use ronn_graph::{
    ConstantFoldingPass, DeadCodeEliminationPass, OptimizationLevel, OptimizationPass, Optimizer,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Helper to create test input
fn create_test_input(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

/// Create a graph suitable for constant folding
fn create_constant_heavy_graph() -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    // Add constant operations
    let const1_id = builder.add_op("Constant", Some("const1".to_string()));
    builder.add_output(const1_id, "const1_value");

    let const2_id = builder.add_op("Constant", Some("const2".to_string()));
    builder.add_output(const2_id, "const2_value");

    // Add computation that can be folded
    let add_id = builder.add_op("Add", Some("add_consts".to_string()));
    builder
        .add_input(add_id, "const1_value")
        .add_input(add_id, "const2_value")
        .add_output(add_id, "const_sum");

    builder.connect(const1_id, add_id, "const1_value").unwrap();
    builder.connect(const2_id, add_id, "const2_value").unwrap();

    // Final operation with input
    let final_id = builder.add_op("Mul", Some("final_mul".to_string()));
    builder
        .add_input(final_id, "input_tensor")
        .add_input(final_id, "const_sum")
        .add_output(final_id, "output");

    builder.connect(input_id, final_id, "input_tensor").unwrap();
    builder.connect(add_id, final_id, "const_sum").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output".to_string()]);

    builder.build().unwrap()
}

/// Benchmark constant folding impact
pub fn bench_constant_folding_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("constant_folding_impact");

    let graph = create_constant_heavy_graph();

    // Without constant folding
    group.bench_function("without_constant_folding", |b| {
        b.iter(|| {
            let mut g = graph.clone();
            // Run optimizer with O0 (no optimizations)
            let optimizer = Optimizer::new(OptimizationLevel::O0);
            black_box(optimizer.optimize(&mut g).unwrap())
        });
    });

    // With constant folding
    group.bench_function("with_constant_folding", |b| {
        b.iter(|| {
            let mut g = graph.clone();
            let pass = ConstantFoldingPass;
            black_box(pass.run(&mut g).unwrap())
        });
    });

    group.finish();
}

/// Benchmark node fusion impact
pub fn bench_fusion_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_impact");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping fusion impact benchmark: test model not found");
        return;
    }

    // Without fusion (O0)
    group.bench_function("without_fusion", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O0);

        let session = model.create_session(options).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    // With fusion (O2)
    group.bench_function("with_fusion", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O2);

        let session = model.create_session(options).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    group.finish();
}

/// Benchmark layout optimization impact
pub fn bench_layout_optimization_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_optimization_impact");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping layout optimization benchmark: test model not found");
        return;
    }

    // Without layout optimization
    group.bench_function("without_layout_opt", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O0);

        let session = model.create_session(options).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    // With layout optimization
    group.bench_function("with_layout_opt", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O3);

        let session = model.create_session(options).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    group.finish();
}

/// Benchmark dead code elimination impact
pub fn bench_dead_code_elimination_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("dead_code_elimination_impact");

    // Create graph with dead code
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    // Dead branch (not connected to output)
    let dead1_id = builder.add_op("Conv", Some("dead_conv".to_string()));
    builder
        .add_input(dead1_id, "input_tensor")
        .add_output(dead1_id, "dead_output");
    builder.connect(input_id, dead1_id, "input_tensor").unwrap();

    // Live branch
    let live_id = builder.add_op("Relu", Some("live_relu".to_string()));
    builder
        .add_input(live_id, "input_tensor")
        .add_output(live_id, "output");
    builder.connect(input_id, live_id, "input_tensor").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output".to_string()]);

    let graph = builder.build().unwrap();

    // Without DCE
    group.bench_function("without_dce", |b| {
        b.iter(|| {
            let mut g = graph.clone();
            let optimizer = Optimizer::new(OptimizationLevel::O0);
            black_box(optimizer.optimize(&mut g).unwrap())
        });
    });

    // With DCE
    group.bench_function("with_dce", |b| {
        b.iter(|| {
            let mut g = graph.clone();
            let pass = DeadCodeEliminationPass;
            black_box(pass.run(&mut g).unwrap())
        });
    });

    group.finish();
}

/// Benchmark CPU-specific optimization impact
pub fn bench_cpu_optimization_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_optimization_impact");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping CPU optimization benchmark: test model not found");
        return;
    }

    use ronn_providers::ProviderType;

    // Without CPU-specific optimizations
    group.bench_function("without_cpu_opt", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new()
            .with_provider(ProviderType::CPU)
            .with_optimization_level(OptimizationLevel::O0);

        let session = model.create_session(options).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    // With CPU-specific optimizations
    group.bench_function("with_cpu_opt", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new()
            .with_provider(ProviderType::CPU)
            .with_optimization_level(OptimizationLevel::O3);

        let session = model.create_session(options).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    group.finish();
}

/// Benchmark GPU-specific optimization impact
pub fn bench_gpu_optimization_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_optimization_impact");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping GPU optimization benchmark: test model not found");
        return;
    }

    use ronn_providers::ProviderType;

    // Without GPU-specific optimizations
    group.bench_function("without_gpu_opt", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new()
            .with_provider(ProviderType::GPU)
            .with_optimization_level(OptimizationLevel::O0);

        let session = match model.create_session(options) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("GPU not available, skipping GPU optimization benchmark");
                return;
            }
        };

        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    // With GPU-specific optimizations
    group.bench_function("with_gpu_opt", |b| {
        let model = Model::load(&model_path).unwrap();
        let options = SessionOptions::new()
            .with_provider(ProviderType::GPU)
            .with_optimization_level(OptimizationLevel::O3);

        let session = match model.create_session(options) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("GPU not available, skipping GPU optimization benchmark");
                return;
            }
        };

        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let mut inputs = HashMap::new();
            inputs.insert("input", input.clone());
            let _result = session.run(black_box(inputs)).unwrap();
        });
    });

    group.finish();
}

/// Benchmark all optimization levels end-to-end
pub fn bench_optimization_levels_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_levels_e2e");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping optimization levels benchmark: test model not found");
        return;
    }

    let levels = [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ];

    for level in levels.iter() {
        group.bench_with_input(
            BenchmarkId::new("opt_level", format!("{:?}", level)),
            level,
            |b, &level| {
                let model = Model::load(&model_path).unwrap();
                let options = SessionOptions::new().with_optimization_level(level);

                let session = model.create_session(options).unwrap();
                let input = create_test_input(vec![1, 3, 224, 224]);

                b.iter(|| {
                    let mut inputs = HashMap::new();
                    inputs.insert("input", input.clone());
                    let _result = session.run(black_box(inputs)).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    optimization_impact_benches,
    bench_constant_folding_impact,
    bench_fusion_impact,
    bench_layout_optimization_impact,
    bench_dead_code_elimination_impact,
    bench_cpu_optimization_impact,
    bench_gpu_optimization_impact,
    bench_optimization_levels_e2e
);
