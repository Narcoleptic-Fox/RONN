//! Integration tests for the Hierarchical Reasoning Module (HRM).

mod complexity_tests;
mod executor_tests;
mod router_tests;

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::executor::ExecutionPath;
use ronn_hrm::{HierarchicalReasoningModule, RoutingStrategy};

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
    let data = vec![1.0f32; 100];
    let input = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    assert_eq!(hrm.metrics().total_inferences(), 1);
    assert!(result.confidence > 0.0);

    Ok(())
}

#[test]
fn test_hrm_process_varied_input() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::new();

    // Varied input with higher complexity
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 10.0).collect();
    let input = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    assert_eq!(hrm.metrics().total_inferences(), 1);
    assert!(result.output.shape() == input.shape());

    Ok(())
}

#[test]
fn test_hrm_metrics_tracking() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::new();

    // Process multiple inputs
    for i in 0..5 {
        let data = vec![i as f32; 100];
        let input = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;
        let _ = hrm.process(&input)?;
    }

    assert_eq!(hrm.metrics().total_inferences(), 5);

    Ok(())
}

#[test]
fn test_hrm_metrics_reset() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::new();

    // Process some inputs
    let data = vec![1.0f32; 100];
    let input = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;
    let _ = hrm.process(&input)?;

    assert_eq!(hrm.metrics().total_inferences(), 1);

    // Reset metrics
    hrm.reset_metrics();

    assert_eq!(hrm.metrics().total_inferences(), 0);

    Ok(())
}

#[test]
fn test_hrm_with_always_system1() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::with_strategy(RoutingStrategy::AlwaysSystem1);

    let data = vec![1.0f32; 100];
    let input = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    assert!(matches!(result.path_taken, ExecutionPath::System1));

    Ok(())
}

#[test]
fn test_hrm_with_always_system2() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::with_strategy(RoutingStrategy::AlwaysSystem2);

    let data = vec![1.0f32; 100];
    let input = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    assert!(matches!(result.path_taken, ExecutionPath::System2));

    Ok(())
}

#[test]
fn test_hrm_with_adaptive_hybrid() -> Result<()> {
    let mut hrm = HierarchicalReasoningModule::with_strategy(RoutingStrategy::AdaptiveHybrid);

    let data = vec![1.0f32; 100];
    let input = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    // Adaptive can route to any path
    assert!(result.confidence > 0.0);

    Ok(())
}
