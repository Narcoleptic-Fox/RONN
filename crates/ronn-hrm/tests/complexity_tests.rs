//! Tests for HRM complexity assessment.

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::complexity::{ComplexityAssessor, ComplexityLevel};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn test_assess_low_complexity() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Small tensor with low variance
    let data = vec![1.0f32, 1.0, 1.0, 1.0];
    let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    assert_eq!(metrics.level, ComplexityLevel::Low);
    assert!(metrics.size < 100);
    assert!(metrics.complexity_score < 0.33);
    assert!(metrics.uncertainty >= 0.0 && metrics.uncertainty <= 1.0);

    Ok(())
}

#[test]
fn test_assess_medium_complexity() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Medium-sized tensor with moderate variance
    let data: Vec<f32> = (0..500).map(|x| (x as f32).sin()).collect();
    let tensor = Tensor::from_data(data, vec![1, 500], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    assert!(metrics.size == 500);
    assert!(metrics.complexity_score >= 0.0 && metrics.complexity_score <= 1.0);
    // Medium could be Low or Medium depending on variance
    assert!(matches!(
        metrics.level,
        ComplexityLevel::Low | ComplexityLevel::Medium
    ));

    Ok(())
}

#[test]
fn test_assess_high_complexity() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Very large tensor with high variance
    let data: Vec<f32> = (0..5000)
        .map(|x| (x as f32).sin() * (x as f32).cos() * (x as f32 % 100.0))
        .collect();
    let tensor = Tensor::from_data(data, vec![1, 5000], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Should be High or at least Medium
    assert!(matches!(
        metrics.level,
        ComplexityLevel::High | ComplexityLevel::Medium
    ));
    assert!(metrics.size > 1000);

    Ok(())
}

#[test]
fn test_empty_tensor() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Edge case: Empty tensor
    let data = vec![];
    let tensor = Tensor::from_data(data, vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    assert_eq!(metrics.level, ComplexityLevel::Low);
    assert_eq!(metrics.size, 0);
    assert_eq!(metrics.complexity_score, 0.0);

    Ok(())
}

#[test]
fn test_single_element_tensor() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    let data = vec![42.0f32];
    let tensor = Tensor::from_data(data, vec![1], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    assert_eq!(metrics.level, ComplexityLevel::Low);
    assert_eq!(metrics.size, 1);

    Ok(())
}

#[test]
fn test_custom_thresholds() -> Result<()> {
    // Lower thresholds mean more things are "complex"
    let assessor = ComplexityAssessor::with_thresholds(10, 100, 0.1);

    let data = vec![1.0f32; 50];
    let tensor = Tensor::from_data(data, vec![1, 50], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // 50 elements is between low (10) and high (100) thresholds
    assert_eq!(metrics.size, 50);
    assert!(metrics.complexity_score > 0.0);

    Ok(())
}

#[test]
fn test_assessment_performance() -> Result<()> {
    use std::time::Instant;

    let assessor = ComplexityAssessor::new();

    // Large tensor for performance testing
    let data: Vec<f32> = (0..100_000).map(|x| x as f32).collect();
    let tensor = Tensor::from_data(
        data,
        vec![1, 100_000],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let start = Instant::now();
    let _metrics = assessor.assess(&tensor)?;
    let elapsed = start.elapsed();

    // Assessment should be fast (< 100ms for 100K elements)
    assert!(
        elapsed.as_millis() < 100,
        "Assessment too slow: {:?}",
        elapsed
    );

    Ok(())
}

#[test]
fn test_repeated_assessments() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    // Multiple assessments should give consistent results
    let metrics1 = assessor.assess(&tensor)?;
    let metrics2 = assessor.assess(&tensor)?;
    let metrics3 = assessor.assess(&tensor)?;

    assert_eq!(metrics1.level, metrics2.level);
    assert_eq!(metrics2.level, metrics3.level);
    assert_eq!(metrics1.complexity_score, metrics2.complexity_score);
    assert_eq!(metrics1.variance, metrics2.variance);

    Ok(())
}
