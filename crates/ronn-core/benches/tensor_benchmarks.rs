//! Benchmark tests for core tensor operations.
//!
//! Run with: cargo bench --package ronn-core

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ronn_core::{ArithmeticOps, DataType, MatrixOps, ReductionOps, ShapeOps, Tensor, TensorLayout};

fn create_tensor(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

fn bench_arithmetic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic");

    for size in [100, 1000, 10000].iter() {
        let a = create_tensor(vec![*size]);
        let b = create_tensor(vec![*size]);

        group.bench_with_input(BenchmarkId::new("add", size), size, |bencher, _| {
            bencher.iter(|| black_box(a.add(&b).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("mul", size), size, |bencher, _| {
            bencher.iter(|| black_box(a.mul(&b).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("add_scalar", size), size, |bencher, _| {
            bencher.iter(|| black_box(a.add_scalar(5.0).unwrap()));
        });
    }

    group.finish();
}

fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix");

    for size in [10, 50, 100].iter() {
        let a = create_tensor(vec![*size, *size]);
        let b = create_tensor(vec![*size, *size]);

        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bencher, _| {
            bencher.iter(|| black_box(a.matmul(&b).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("transpose", size), size, |bencher, _| {
            bencher.iter(|| black_box(a.transpose().unwrap()));
        });
    }

    group.finish();
}

fn bench_shape_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape");

    for size in [1000, 10000, 100000].iter() {
        let tensor = create_tensor(vec![*size]);

        group.bench_with_input(BenchmarkId::new("reshape", size), size, |bencher, _| {
            let new_shape = if size % 100 == 0 {
                vec![size / 100, 100]
            } else {
                vec![*size]
            };
            bencher.iter(|| black_box(tensor.reshape(&new_shape).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("flatten", size), size, |bencher, _| {
            let t = create_tensor(vec![size / 10, 10]);
            bencher.iter(|| black_box(t.flatten().unwrap()));
        });
    }

    group.finish();
}

fn bench_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction");

    for size in [1000, 10000, 100000].iter() {
        let tensor = create_tensor(vec![*size]);

        group.bench_with_input(BenchmarkId::new("sum_all", size), size, |bencher, _| {
            bencher.iter(|| black_box(tensor.sum_all().unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("mean_all", size), size, |bencher, _| {
            bencher.iter(|| black_box(tensor.mean_all().unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("max_all", size), size, |bencher, _| {
            bencher.iter(|| black_box(tensor.max_all().unwrap()));
        });
    }

    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation");

    for size in [1000, 10000, 100000].iter() {
        let tensor = create_tensor(vec![*size]);

        group.bench_with_input(BenchmarkId::new("relu", size), size, |bencher, _| {
            bencher.iter(|| black_box(tensor.relu().unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |bencher, _| {
            bencher.iter(|| black_box(tensor.sigmoid().unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("tanh", size), size, |bencher, _| {
            bencher.iter(|| black_box(tensor.tanh().unwrap()));
        });
    }

    group.finish();
}

fn bench_broadcasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");

    let sizes = [(10, 100), (100, 1000), (1000, 10000)];

    for (rows, cols) in sizes.iter() {
        let matrix = create_tensor(vec![*rows, *cols]);
        let vector = create_tensor(vec![*cols]);

        group.bench_with_input(
            BenchmarkId::new("broadcast_add", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, _| {
                bencher.iter(|| black_box(matrix.add(&vector).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_arithmetic_operations,
    bench_matrix_operations,
    bench_shape_operations,
    bench_reduction_operations,
    bench_activation_functions,
    bench_broadcasting
);

criterion_main!(benches);
