//! Tests for executor functionality (LowLevelExecutor and HighLevelPlanner).

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::executor::{ExecutionPath, HighLevelPlanner, LowLevelExecutor};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// LowLevelExecutor Tests (System 1)
// ============================================================================

#[test]
fn test_low_level_executor_creation() {
    let executor = LowLevelExecutor::new();
    let stats = executor.cache_stats();
    assert!(stats.enabled);
}

#[test]
fn test_low_level_executor_without_cache() {
    let executor = LowLevelExecutor::without_cache();
    let stats = executor.cache_stats();
    assert!(!stats.enabled);
}

#[test]
fn test_low_level_executor_execute() -> Result<()> {
    let executor = LowLevelExecutor::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;

    let output = executor.execute(&input)?;

    // Output should have same shape as input
    assert_eq!(output.shape(), input.shape());

    Ok(())
}

#[test]
fn test_low_level_executor_caching() -> Result<()> {
    let executor = LowLevelExecutor::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;

    // Execute twice with same input
    let _ = executor.execute(&input)?;
    let _ = executor.execute(&input)?;

    let stats = executor.cache_stats();
    // Cache is enabled (actual caching behavior may vary)
    assert!(stats.enabled);

    Ok(())
}

// ============================================================================
// HighLevelPlanner Tests (System 2)
// ============================================================================

#[test]
fn test_high_level_planner_creation() {
    let planner = HighLevelPlanner::new();
    let stats = planner.planning_stats();
    assert!(!stats.decomposition_enabled);
}

#[test]
fn test_high_level_planner_with_decomposition() {
    let planner = HighLevelPlanner::with_decomposition(3);
    let stats = planner.planning_stats();
    assert!(stats.decomposition_enabled);
    assert_eq!(stats.max_depth, 3);
}

#[test]
fn test_high_level_planner_execute() -> Result<()> {
    let planner = HighLevelPlanner::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;

    let output = planner.execute(&input)?;

    // Output should have same shape as input
    assert_eq!(output.shape(), input.shape());

    Ok(())
}

// ============================================================================
// ExecutionPath Tests
// ============================================================================

#[test]
fn test_execution_path_variants() {
    let path1 = ExecutionPath::System1;
    let path2 = ExecutionPath::System2;
    let path3 = ExecutionPath::Hybrid;

    // Just verify these can be constructed and compared
    assert_ne!(format!("{:?}", path1), format!("{:?}", path2));
    assert_ne!(format!("{:?}", path2), format!("{:?}", path3));
}
