// Performance benchmarks for graph optimization passes
//
// Run with: cargo bench --package ronn-graph

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ronn_core::{GraphBuilder, ModelGraph};
use ronn_graph::{
    ConstantFoldingPass, CpuOptimizationPass, DeadCodeEliminationPass, GpuOptimizationPass,
    LayoutOptimizationPass, NodeFusionPass, OptimizationLevel, OptimizationPass, Optimizer,
};

// Graph creation helpers

fn create_simple_graph(size: usize) -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let mut prev_id = input_id;
    let mut prev_output = "input_tensor".to_string();

    for i in 0..size {
        let conv_id = builder.add_op("Conv", Some(format!("conv_{}", i)));
        let output_name = format!("conv_{}_out", i);

        builder
            .add_input(conv_id, &prev_output)
            .add_output(conv_id, &output_name);

        builder.connect(prev_id, conv_id, &prev_output).unwrap();

        prev_id = conv_id;
        prev_output = output_name;
    }

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec![prev_output]);

    builder.build().unwrap()
}

fn create_fusible_graph(layers: usize) -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let mut prev_id = input_id;
    let mut prev_output = "input_tensor".to_string();

    for i in 0..layers {
        // Conv
        let conv_id = builder.add_op("Conv", Some(format!("conv_{}", i)));
        let conv_output = format!("conv_{}_out", i);
        builder
            .add_input(conv_id, &prev_output)
            .add_output(conv_id, &conv_output);
        builder.connect(prev_id, conv_id, &prev_output).unwrap();

        // BatchNorm
        let bn_id = builder.add_op("BatchNormalization", Some(format!("bn_{}", i)));
        let bn_output = format!("bn_{}_out", i);
        builder
            .add_input(bn_id, &conv_output)
            .add_output(bn_id, &bn_output);
        builder.connect(conv_id, bn_id, &conv_output).unwrap();

        // ReLU
        let relu_id = builder.add_op("Relu", Some(format!("relu_{}", i)));
        let relu_output = format!("relu_{}_out", i);
        builder
            .add_input(relu_id, &bn_output)
            .add_output(relu_id, &relu_output);
        builder.connect(bn_id, relu_id, &bn_output).unwrap();

        prev_id = relu_id;
        prev_output = relu_output;
    }

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec![prev_output]);

    builder.build().unwrap()
}

fn create_wide_graph(width: usize) -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let mut output_tensors = vec![];

    for i in 0..width {
        let conv_id = builder.add_op("Conv", Some(format!("conv_{}", i)));
        let output_name = format!("conv_{}_out", i);

        builder
            .add_input(conv_id, "input_tensor")
            .add_output(conv_id, &output_name);

        builder.connect(input_id, conv_id, "input_tensor").unwrap();

        output_tensors.push(output_name);
    }

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(output_tensors);

    builder.build().unwrap()
}

// Individual pass benchmarks

fn bench_constant_folding_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("constant_folding_pass");

    for size in [10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_simple_graph(size);
            let pass = ConstantFoldingPass;

            b.iter(|| {
                let mut g = graph.clone();
                black_box(pass.run(&mut g).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_dead_code_elimination_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("dead_code_elimination_pass");

    for size in [10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_simple_graph(size);
            let pass = DeadCodeEliminationPass;

            b.iter(|| {
                let mut g = graph.clone();
                black_box(pass.run(&mut g).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_node_fusion_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_fusion_pass");

    for layers in [5, 10, 20, 40] {
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &layers| {
                let graph = create_fusible_graph(layers);
                let pass = NodeFusionPass;

                b.iter(|| {
                    let mut g = graph.clone();
                    black_box(pass.run(&mut g).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_layout_optimization_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_optimization_pass");

    for size in [10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_simple_graph(size);
            let pass = LayoutOptimizationPass;

            b.iter(|| {
                let mut g = graph.clone();
                black_box(pass.run(&mut g).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_cpu_optimization_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_optimization_pass");

    for size in [10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_simple_graph(size);
            let pass = CpuOptimizationPass;

            b.iter(|| {
                let mut g = graph.clone();
                black_box(pass.run(&mut g).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_gpu_optimization_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_optimization_pass");

    for size in [10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_simple_graph(size);
            let pass = GpuOptimizationPass;

            b.iter(|| {
                let mut g = graph.clone();
                black_box(pass.run(&mut g).unwrap())
            });
        });
    }

    group.finish();
}

// Optimizer level benchmarks

fn bench_optimizer_o1(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_o1");

    for size in [10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let graph = create_simple_graph(size);
            let optimizer = Optimizer::new(OptimizationLevel::O1);

            b.iter(|| {
                let mut g = graph.clone();
                black_box(optimizer.optimize(&mut g).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_optimizer_o2(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_o2");

    for layers in [5, 10, 20, 40] {
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &layers| {
                let graph = create_fusible_graph(layers);
                let optimizer = Optimizer::new(OptimizationLevel::O2);

                b.iter(|| {
                    let mut g = graph.clone();
                    black_box(optimizer.optimize(&mut g).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_optimizer_o3(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_o3");

    for layers in [5, 10, 20, 40] {
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &layers| {
                let graph = create_fusible_graph(layers);
                let optimizer = Optimizer::new(OptimizationLevel::O3);

                b.iter(|| {
                    let mut g = graph.clone();
                    black_box(optimizer.optimize(&mut g).unwrap())
                });
            },
        );
    }

    group.finish();
}

// Graph structure impact benchmarks

fn bench_deep_vs_wide_graphs(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_structure");

    let deep_graph = create_simple_graph(100);
    let wide_graph = create_wide_graph(100);

    group.bench_function("deep_graph_100", |b| {
        let optimizer = Optimizer::new(OptimizationLevel::O3);
        b.iter(|| {
            let mut g = deep_graph.clone();
            black_box(optimizer.optimize(&mut g).unwrap())
        });
    });

    group.bench_function("wide_graph_100", |b| {
        let optimizer = Optimizer::new(OptimizationLevel::O3);
        b.iter(|| {
            let mut g = wide_graph.clone();
            black_box(optimizer.optimize(&mut g).unwrap())
        });
    });

    group.finish();
}

// Optimization level comparison

fn bench_optimization_levels_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_levels");

    let graph = create_fusible_graph(20);

    for level in [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ] {
        group.bench_function(format!("{:?}", level), |b| {
            let optimizer = Optimizer::new(level);
            b.iter(|| {
                let mut g = graph.clone();
                black_box(optimizer.optimize(&mut g).unwrap())
            });
        });
    }

    group.finish();
}

// Iteration convergence benchmark

fn bench_convergence_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence");

    let graph = create_fusible_graph(10);
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    group.bench_function("first_optimization", |b| {
        b.iter(|| {
            let mut g = graph.clone();
            black_box(optimizer.optimize(&mut g).unwrap())
        });
    });

    // Create a pre-optimized graph
    let mut pre_optimized = graph.clone();
    optimizer.optimize(&mut pre_optimized).unwrap();

    group.bench_function("second_optimization_idempotent", |b| {
        b.iter(|| {
            let mut g = pre_optimized.clone();
            black_box(optimizer.optimize(&mut g).unwrap())
        });
    });

    group.finish();
}

criterion_group!(
    passes,
    bench_constant_folding_pass,
    bench_dead_code_elimination_pass,
    bench_node_fusion_pass,
    bench_layout_optimization_pass,
    bench_cpu_optimization_pass,
    bench_gpu_optimization_pass
);

criterion_group!(
    optimizers,
    bench_optimizer_o1,
    bench_optimizer_o2,
    bench_optimizer_o3,
    bench_optimization_levels_comparison
);

criterion_group!(
    graph_structure,
    bench_deep_vs_wide_graphs,
    bench_convergence_iterations
);

criterion_main!(passes, optimizers, graph_structure);
