//! Tests for HRM executors (LowLevel and HighLevel).

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::executor::{LowLevelExecutor, HighLevelPlanner};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn test_low_level_executor_basic() -> Result<()> {
    let executor = LowLevelExecutor::new();

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
fn test_low_level_executor_empty() -> Result<()> {
    let executor = LowLevelExecutor::new();

    let input = Tensor::from_data(vec![], vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    assert_eq!(output.shape(), vec![0]);

    Ok(())
}

#[test]
fn test_low_level_executor_large() -> Result<()> {
    let executor = LowLevelExecutor::new();

    let data = vec![1.0f32; 10000];
    let input = Tensor::from_data(data, vec![1, 10000], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_high_level_planner_basic() -> Result<()> {
    let planner = HighLevelPlanner::new();

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output = planner.execute(&input)?;

    // Should produce output of same shape
    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_high_level_planner_empty() -> Result<()> {
    let planner = HighLevelPlanner::new();

    let input = Tensor::from_data(vec![], vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let output = planner.execute(&input)?;

    assert_eq!(output.shape(), vec![0]);

    Ok(())
}

#[test]
fn test_high_level_planner_large() -> Result<()> {
    let planner = HighLevelPlanner::new();

    let data = vec![1.0f32; 10000];
    let input = Tensor::from_data(data, vec![1, 10000], DataType::F32, TensorLayout::RowMajor)?;

    let output = planner.execute(&input)?;

    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_both_executors_produce_valid_output() -> Result<()> {
    let low_level = LowLevelExecutor::new();
    let high_level = HighLevelPlanner::new();

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let output1 = low_level.execute(&input)?;
    let output2 = high_level.execute(&input)?;

    // Both should produce valid outputs
    assert_eq!(output1.shape(), input.shape());
    assert_eq!(output2.shape(), input.shape());

    Ok(())
}

#[test]
fn test_executor_performance() -> Result<()> {
    use std::time::Instant;

    let low_level = LowLevelExecutor::new();

    let input = Tensor::from_data(
        vec![1.0f32; 1000],
        vec![1, 1000],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let start = Instant::now();
    let _output = low_level.execute(&input)?;
    let elapsed = start.elapsed();

    // Should be fast (< 50ms)
    assert!(elapsed.as_millis() < 50, "Executor too slow: {:?}", elapsed);

    Ok(())
}
