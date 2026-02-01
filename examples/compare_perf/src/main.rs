//! Quick RONN vs baseline performance comparison
//! Run with: cargo run -p compare_perf --release

use ronn_core::{DataType, MatrixOps, Tensor, TensorLayout};
use std::time::Instant;

fn main() {
    println!("🦊 RONN Performance Comparison\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Benchmark parameters
    let sizes = vec![
        ("Small (1K)", vec![32, 32]),
        ("Medium (1M)", vec![1000, 1000]),
        ("Large (10M)", vec![3162, 3162]),
    ];
    
    let iterations = 1000;
    
    println!("\n📊 Tensor Creation Benchmarks ({} iterations)\n", iterations);
    
    for (name, shape) in &sizes {
        let total_elements: usize = shape.iter().product();
        
        // Benchmark RONN tensor creation
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Tensor::ones(shape.clone(), DataType::F32, TensorLayout::RowMajor).unwrap();
        }
        let ronn_time = start.elapsed();
        let ronn_per_op = ronn_time.as_nanos() as f64 / iterations as f64;
        let ronn_throughput = (total_elements as f64 * iterations as f64) / ronn_time.as_secs_f64() / 1e9;
        
        // Benchmark Vec creation (baseline)
        let start = Instant::now();
        for _ in 0..iterations {
            let _: Vec<f32> = vec![1.0f32; total_elements];
        }
        let vec_time = start.elapsed();
        let vec_per_op = vec_time.as_nanos() as f64 / iterations as f64;
        
        let speedup = vec_per_op / ronn_per_op;
        
        println!("{:15} | RONN: {:>8.1} ns | Vec: {:>8.1} ns | {:.1}x | {:.2} Gelem/s",
            name, ronn_per_op, vec_per_op, speedup, ronn_throughput);
    }
    
    println!("\n📊 MatMul Benchmarks\n");
    
    let matmul_sizes = vec![
        ("64x64", 64),
        ("256x256", 256),
        ("512x512", 512),
    ];
    
    for (name, n) in &matmul_sizes {
        let a = Tensor::ones(vec![*n, *n], DataType::F32, TensorLayout::RowMajor).unwrap();
        let b = Tensor::ones(vec![*n, *n], DataType::F32, TensorLayout::RowMajor).unwrap();
        
        // Warmup
        let _ = a.matmul(&b);
        
        let iters = if *n <= 128 { 100 } else { 10 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.matmul(&b);
        }
        let elapsed = start.elapsed();
        let per_op_us = elapsed.as_micros() as f64 / iters as f64;
        let gflops = (2.0 * (*n as f64).powi(3)) / (per_op_us * 1000.0);
        
        println!("{:15} | {:>8.1} µs | {:>6.2} GFLOPS", name, per_op_us, gflops);
    }
    
    println!("\n✅ Benchmark complete");
    println!("\n💡 Note: For end-to-end model comparison, use specific ONNX models");
}
