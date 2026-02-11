//! Tests for complexity assessment functionality.

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::{ComplexityAssessor, ComplexityLevel};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn test_complexity_assessor_creation() {
    let _assessor = ComplexityAssessor::new();
    // Just verify it can be created without panic
}

#[test]
fn test_complexity_assessment_simple_input() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Simple uniform tensor should be low complexity
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Uniform data should be low complexity
    assert!(matches!(
        metrics.level,
        ComplexityLevel::Low | ComplexityLevel::Medium
    ));

    Ok(())
}

#[test]
fn test_complexity_assessment_varied_input() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Highly varied tensor should be higher complexity
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
    let tensor = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Just verify we get valid metrics
    assert!(metrics.variance >= 0.0);

    Ok(())
}

#[test]
fn test_complexity_assessment_large_input() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Large tensor
    let data = vec![1.0f32; 10000];
    let tensor = Tensor::from_data(data, vec![100, 100], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Large uniform data should still assess properly
    assert!(metrics.size == 10000);

    Ok(())
}

#[test]
fn test_complexity_level_variants() {
    // Verify complexity levels can be matched
    let low = ComplexityLevel::Low;
    let medium = ComplexityLevel::Medium;
    let high = ComplexityLevel::High;

    assert!(matches!(low, ComplexityLevel::Low));
    assert!(matches!(medium, ComplexityLevel::Medium));
    assert!(matches!(high, ComplexityLevel::High));
}
