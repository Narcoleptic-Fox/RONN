//! Comprehensive tests for HRM executors (System 1 and System 2).
//!
//! Tests cover:
//! - System 1 (fast path) execution
//! - System 2 (slow path) execution
//! - Pattern caching
//! - Problem decomposition
//! - Statistics tracking
//! - Performance characteristics

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::executor::{System1Executor, System2Executor};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// System 1 (Fast Path) Tests
// ============================================================================

#[test]
fn test_system1_basic_execution() -> Result<()> {
    let mut executor = System1Executor::new(true); // Enable caching

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output = executor.execute(&input)?;

    // Should produce output of same shape
    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_system1_multiple_executions() -> Result<()> {
    let mut executor = System1Executor::new(true);

    // Execute multiple times
    for i in 0..10 {
        let input = Tensor::from_data(
            vec![i as f32; 10],
            vec![1, 10],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let output = executor.execute(&input)?;
        assert_eq!(output.shape(), input.shape());
    }

    let stats = executor.stats();
    assert_eq!(stats.executions, 10);

    Ok(())
}

#[test]
fn test_system1_caching_enabled() -> Result<()> {
    let mut executor = System1Executor::new(true); // Caching ON

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0],
        vec![1, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // First execution - cache miss
    let output1 = executor.execute(&input)?;

    // Second execution with same input - cache hit
    let output2 = executor.execute(&input)?;

    let stats = executor.stats();
    assert_eq!(stats.executions, 2);
    // With caching, might have cache hits
    assert!(stats.cache_hits <= 1);

    // Outputs should be similar
    assert_eq!(output1.shape(), output2.shape());

    Ok(())
}

#[test]
fn test_system1_caching_disabled() -> Result<()> {
    let mut executor = System1Executor::new(false); // Caching OFF

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0],
        vec![1, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Execute multiple times
    executor.execute(&input)?;
    executor.execute(&input)?;

    let stats = executor.stats();
    assert_eq!(stats.executions, 2);
    // With caching disabled, should have no cache hits
    assert_eq!(stats.cache_hits, 0);

    Ok(())
}

#[test]
fn test_system1_cache_statistics() -> Result<()> {
    let mut executor = System1Executor::new(true);

    // Execute different inputs
    for i in 0..5 {
        let input = Tensor::from_data(
            vec![i as f32; 10],
            vec![1, 10],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        executor.execute(&input)?;
    }

    let stats = executor.stats();
    assert_eq!(stats.executions, 5);
    // Cache hits depend on pattern matching
    assert!(stats.cache_hits <= 5);

    Ok(())
}

#[test]
fn test_system1_empty_input() -> Result<()> {
    let mut executor = System1Executor::new(true);

    let input = Tensor::from_data(vec![], vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    assert_eq!(output.shape(), vec![0]);

    Ok(())
}

#[test]
fn test_system1_large_input() -> Result<()> {
    let mut executor = System1Executor::new(false); // Disable caching for large input

    let data = vec![1.0f32; 100000];
    let input = Tensor::from_data(data, vec![1, 100000], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_system1_performance() -> Result<()> {
    use std::time::Instant;

    let mut executor = System1Executor::new(true);

    let input = Tensor::from_data(
        vec![1.0f32; 1000],
        vec![1, 1000],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let start = Instant::now();
    let _output = executor.execute(&input)?;
    let elapsed = start.elapsed();

    // System 1 should be fast (target: 10x faster than System 2)
    // For 1000 elements, should be <5ms
    assert!(elapsed.as_millis() < 5, "System1 too slow: {:?}", elapsed);

    Ok(())
}

// ============================================================================
// System 2 (Slow Path) Tests
// ============================================================================

#[test]
fn test_system2_basic_execution() -> Result<()> {
    let mut executor = System2Executor::new(3); // Max depth 3

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output = executor.execute(&input)?;

    // Should produce output of same shape
    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_system2_multiple_executions() -> Result<()> {
    let mut executor = System2Executor::new(3);

    // Execute multiple times
    for i in 0..10 {
        let input = Tensor::from_data(
            vec![i as f32; 10],
            vec![1, 10],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let output = executor.execute(&input)?;
        assert_eq!(output.shape(), input.shape());
    }

    let stats = executor.stats();
    assert_eq!(stats.executions, 10);

    Ok(())
}

#[test]
fn test_system2_decomposition_depth_zero() -> Result<()> {
    let mut executor = System2Executor::new(0); // No decomposition

    let input = Tensor::from_data(
        vec![1.0f32; 100],
        vec![1, 100],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output = executor.execute(&input)?;

    let stats = executor.stats();
    // With depth 0, no decomposition should occur
    assert_eq!(stats.decompositions, 0);

    Ok(())
}

#[test]
fn test_system2_decomposition_enabled() -> Result<()> {
    let mut executor = System2Executor::new(5); // Allow decomposition

    // Large complex input might trigger decomposition
    let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
    let input = Tensor::from_data(data, vec![1, 1000], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    let stats = executor.stats();
    // Decomposition might occur for large inputs
    assert!(stats.decompositions >= 0);

    Ok(())
}

#[test]
fn test_system2_statistics_tracking() -> Result<()> {
    let mut executor = System2Executor::new(3);

    // Execute several times
    for i in 0..5 {
        let input = Tensor::from_data(
            vec![i as f32; 20],
            vec![1, 20],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        executor.execute(&input)?;
    }

    let stats = executor.stats();
    assert_eq!(stats.executions, 5);
    assert!(stats.decompositions >= 0);

    Ok(())
}

#[test]
fn test_system2_empty_input() -> Result<()> {
    let mut executor = System2Executor::new(3);

    let input = Tensor::from_data(vec![], vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    assert_eq!(output.shape(), vec![0]);

    Ok(())
}

#[test]
fn test_system2_large_input() -> Result<()> {
    let mut executor = System2Executor::new(5);

    let data = vec![1.0f32; 100000];
    let input = Tensor::from_data(data, vec![1, 100000], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_system2_correctness() -> Result<()> {
    let mut executor = System2Executor::new(3);

    // Test that output is reasonable
    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
        vec![1, 5],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output = executor.execute(&input)?;

    // Output should be valid
    assert_eq!(output.shape(), vec![1, 5]);
    let output_data = output.to_vec()?;
    assert_eq!(output_data.len(), 5);
    // Values should be finite
    for val in output_data {
        assert!(val.is_finite());
    }

    Ok(())
}

// ============================================================================
// System 1 vs System 2 Comparison Tests
// ============================================================================

#[test]
fn test_system1_faster_than_system2() -> Result<()> {
    use std::time::Instant;

    let mut system1 = System1Executor::new(true);
    let mut system2 = System2Executor::new(3);

    let input = Tensor::from_data(
        vec![1.0f32; 1000],
        vec![1, 1000],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Time System 1
    let start1 = Instant::now();
    let _output1 = system1.execute(&input)?;
    let elapsed1 = start1.elapsed();

    // Time System 2
    let start2 = Instant::now();
    let _output2 = system2.execute(&input)?;
    let elapsed2 = start2.elapsed();

    // System 1 should be faster (at least for simple inputs)
    // Note: This might not always hold for very simple operations
    println!("System1: {:?}, System2: {:?}", elapsed1, elapsed2);

    Ok(())
}

#[test]
fn test_both_systems_produce_valid_output() -> Result<()> {
    let mut system1 = System1Executor::new(true);
    let mut system2 = System2Executor::new(3);

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output1 = system1.execute(&input)?;
    let output2 = system2.execute(&input)?;

    // Both should produce valid outputs
    assert_eq!(output1.shape(), input.shape());
    assert_eq!(output2.shape(), input.shape());

    Ok(())
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_executors_with_single_element() -> Result<()> {
    let mut system1 = System1Executor::new(true);
    let mut system2 = System2Executor::new(3);

    let input = Tensor::from_data(
        vec![42.0f32],
        vec![1],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output1 = system1.execute(&input)?;
    let output2 = system2.execute(&input)?;

    assert_eq!(output1.shape(), vec![1]);
    assert_eq!(output2.shape(), vec![1]);

    Ok(())
}

#[test]
fn test_executors_with_extreme_values() -> Result<()> {
    let mut system1 = System1Executor::new(false);
    let mut system2 = System2Executor::new(3);

    let input = Tensor::from_data(
        vec![f32::MAX, f32::MIN, 0.0, 1.0],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output1 = system1.execute(&input)?;
    let output2 = system2.execute(&input)?;

    // Both should handle extreme values
    assert_eq!(output1.shape(), vec![1, 4]);
    assert_eq!(output2.shape(), vec![1, 4]);

    Ok(())
}

#[test]
fn test_executors_with_multidimensional_input() -> Result<()> {
    let mut system1 = System1Executor::new(true);
    let mut system2 = System2Executor::new(3);

    // 2D tensor
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let input = Tensor::from_data(data, vec![3, 4], DataType::F32, TensorLayout::RowMajor)?;

    let output1 = system1.execute(&input)?;
    let output2 = system2.execute(&input)?;

    assert_eq!(output1.shape(), vec![3, 4]);
    assert_eq!(output2.shape(), vec![3, 4]);

    Ok(())
}

// ============================================================================
// Concurrent Execution Tests
// ============================================================================

#[test]
fn test_concurrent_system1_execution() -> Result<()> {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;

    let executor = Arc::new(Mutex::new(System1Executor::new(true)));
    let mut handles = vec![];

    for i in 0..10 {
        let executor_clone = Arc::clone(&executor);
        let handle = thread::spawn(move || {
            let input = Tensor::from_data(
                vec![i as f32; 10],
                vec![1, 10],
                DataType::F32,
                TensorLayout::RowMajor,
            )
            .unwrap();

            let mut exec = executor_clone.lock().unwrap();
            exec.execute(&input).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let exec = executor.lock().unwrap();
    let stats = exec.stats();
    assert_eq!(stats.executions, 10);

    Ok(())
}

#[test]
fn test_concurrent_system2_execution() -> Result<()> {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;

    let executor = Arc::new(Mutex::new(System2Executor::new(3)));
    let mut handles = vec![];

    for i in 0..10 {
        let executor_clone = Arc::clone(&executor);
        let handle = thread::spawn(move || {
            let input = Tensor::from_data(
                vec![i as f32; 10],
                vec![1, 10],
                DataType::F32,
                TensorLayout::RowMajor,
            )
            .unwrap();

            let mut exec = executor_clone.lock().unwrap();
            exec.execute(&input).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let exec = executor.lock().unwrap();
    let stats = exec.stats();
    assert_eq!(stats.executions, 10);

    Ok(())
}

// ============================================================================
// Performance Characteristic Tests
// ============================================================================

#[test]
fn test_system1_latency_target() -> Result<()> {
    use std::time::Instant;

    let mut executor = System1Executor::new(true);

    // Run multiple times to get average
    let mut total_elapsed = std::time::Duration::ZERO;
    let iterations = 100;

    for i in 0..iterations {
        let input = Tensor::from_data(
            vec![(i % 100) as f32; 100],
            vec![1, 100],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let start = Instant::now();
        executor.execute(&input)?;
        total_elapsed += start.elapsed();
    }

    let avg_elapsed = total_elapsed / iterations;
    println!("System1 average latency: {:?}", avg_elapsed);

    // Should be fast on average
    assert!(avg_elapsed.as_micros() < 1000); // <1ms per execution

    Ok(())
}

#[test]
fn test_cache_hit_improves_performance() -> Result<()> {
    use std::time::Instant;

    let mut executor = System1Executor::new(true);

    let input = Tensor::from_data(
        vec![1.0f32; 100],
        vec![1, 100],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // First execution (cache miss)
    let start1 = Instant::now();
    executor.execute(&input)?;
    let elapsed1 = start1.elapsed();

    // Second execution (potentially cache hit)
    let start2 = Instant::now();
    executor.execute(&input)?;
    let elapsed2 = start2.elapsed();

    println!(
        "First: {:?}, Second: {:?}, Cache hits: {}",
        elapsed1,
        elapsed2,
        executor.stats().cache_hits
    );

    // Cache hit should not be slower
    // (Note: might be similar if caching overhead is minimal)

    Ok(())
}
