//! Integration tests for the Hierarchical Reasoning Module (HRM).

mod complexity_tests;
mod executor_tests;
mod router_tests;

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::{HierarchicalReasoningModule, RoutingStrategy};
use ronn_hrm::executor::ExecutionPath;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn test_hrm_creation() {
    let hrm = HierarchicalReasoningModule::new();
    assert_eq!(hrm.metrics().total_inferences(), 0);
}

#[test]
fn test_hrm_with_strategy() {
    let hrm = HierarchicalReasoningModule::with_strategy(RoutingStrategy::AlwaysSystem1);
    assert_eq!(hrm.metrics().total_inferences(), 0);
}

#[test]
fn test_hrm_process_simple_input() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::new();

    // Simple input should use System 1
    let input = Tensor::from_data(
        vec![1.0f32; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = hrm.process(&input)?;

    assert_eq!(result.output.shape(), input.shape());
    // Simple input should route to System1
    assert!(matches!(result.path_taken, ExecutionPath::System1));

    Ok(())
}

#[test]
fn test_hrm_always_system1() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::with_strategy(RoutingStrategy::AlwaysSystem1);

    let input = Tensor::from_data(
        vec![1.0f32; 1000],
        vec![1, 1000],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = hrm.process(&input)?;

    assert_eq!(result.path_taken, ExecutionPath::System1);
    assert_eq!(hrm.metrics().system1_count, 1);

    Ok(())
}

#[test]
fn test_hrm_always_system2() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::with_strategy(RoutingStrategy::AlwaysSystem2);

    let input = Tensor::from_data(
        vec![1.0f32; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = hrm.process(&input)?;

    assert_eq!(result.path_taken, ExecutionPath::System2);
    assert_eq!(hrm.metrics().system2_count, 1);

    Ok(())
}

#[test]
fn test_hrm_metrics_tracking() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::new();

    // Process multiple inputs
    for size in [10, 100, 1000] {
        let data = vec![1.0f32; size];
        let input = Tensor::from_data(data, vec![1, size], DataType::F32, TensorLayout::RowMajor)?;
        hrm.process(&input)?;
    }

    let metrics = hrm.metrics();
    assert_eq!(metrics.total_inferences(), 3);

    Ok(())
}

#[test]
fn test_hrm_reset_metrics() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::new();

    let input = Tensor::from_data(
        vec![1.0f32; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    hrm.process(&input)?;
    assert_eq!(hrm.metrics().total_inferences(), 1);

    hrm.reset_metrics();
    assert_eq!(hrm.metrics().total_inferences(), 0);

    Ok(())
}

#[test]
fn test_hrm_performance() -> Result<()> {
    use std::time::Instant;

    let mut hrm = HierarchicalReasoningModule::new();

    let input = Tensor::from_data(
        vec![1.0f32; 100],
        vec![1, 100],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let start = Instant::now();
    let _result = hrm.process(&input)?;
    let elapsed = start.elapsed();

    // HRM should be fast (< 50ms)
    assert!(elapsed.as_millis() < 50, "HRM too slow: {:?}", elapsed);

    Ok(())
}
