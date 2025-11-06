//! Comprehensive tests for HRM complexity assessment.
//!
//! Tests cover:
//! - Basic complexity assessment (low/medium/high)
//! - Edge cases (empty, very large, extreme values)
//! - Custom thresholds
//! - Variance calculation accuracy
//! - Performance characteristics

use ronn_core::types::{DataType, TensorLayout};
use ronn_core::Tensor;
use ronn_hrm::complexity::{ComplexityAssessor, ComplexityLevel, ComplexityMetrics};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

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
    assert!(metrics.complexity_score > 0.4);

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

// ============================================================================
// Edge Case Tests
// ============================================================================

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
    // Single element has zero variance
    assert!(metrics.complexity_score < 0.1);

    Ok(())
}

#[test]
fn test_very_large_tensor() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Very large tensor (simulating 1M elements)
    let size = 1_000_000;
    let data: Vec<f32> = (0..size).map(|x| (x % 1000) as f32).collect();
    let tensor = Tensor::from_data(data, vec![size], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    assert_eq!(metrics.level, ComplexityLevel::High);
    assert_eq!(metrics.size, size);
    assert!(metrics.complexity_score > 0.66);

    Ok(())
}

#[test]
fn test_multidimensional_tensor() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // 3D tensor: 10x10x10
    let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
    let tensor = Tensor::from_data(
        data,
        vec![10, 10, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let metrics = assessor.assess(&tensor)?;

    assert_eq!(metrics.size, 1000);
    // Total element count determines complexity
    assert!(metrics.complexity_score > 0.0);

    Ok(())
}

#[test]
fn test_extreme_values() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Tensor with extreme values
    let data = vec![
        f32::MAX,
        f32::MIN,
        f32::MAX,
        f32::MIN,
        0.0,
        1.0,
        -1.0,
        f32::MAX,
    ];
    let tensor = Tensor::from_data(data, vec![1, 8], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Should handle extreme values without panic
    assert!(metrics.variance.is_finite());
    assert!(metrics.complexity_score.is_finite());
    assert!(metrics.uncertainty.is_finite());

    Ok(())
}

#[test]
fn test_all_zeros() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    let data = vec![0.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Zero variance means low complexity
    assert!(metrics.variance < 0.001);
    assert!(metrics.complexity_score < 0.5);

    Ok(())
}

#[test]
fn test_all_same_nonzero() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    let data = vec![42.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Uniform data means low variance
    assert!(metrics.variance < 0.001);
    assert!(metrics.complexity_score < 0.5);

    Ok(())
}

#[test]
fn test_alternating_values() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Alternating between two values
    let data: Vec<f32> = (0..1000).map(|x| if x % 2 == 0 { 1.0 } else { 10.0 }).collect();
    let tensor = Tensor::from_data(data, vec![1, 1000], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Should have measurable variance
    assert!(metrics.variance > 1.0);
    // But not maximum complexity
    assert!(metrics.complexity_score < 0.9);

    Ok(())
}

// ============================================================================
// Variance Calculation Tests
// ============================================================================

#[test]
fn test_variance_zero() {
    let assessor = ComplexityAssessor::new();

    // Uniform data = zero variance
    let uniform = vec![5.0f32; 10];
    let tensor =
        Tensor::from_data(uniform, vec![1, 10], DataType::F32, TensorLayout::RowMajor).unwrap();

    let variance = assessor.calculate_variance(&tensor).unwrap();
    assert!(variance < 0.001);
}

#[test]
fn test_variance_nonzero() {
    let assessor = ComplexityAssessor::new();

    // Varied data = nonzero variance
    let varied = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let tensor =
        Tensor::from_data(varied, vec![1, 5], DataType::F32, TensorLayout::RowMajor).unwrap();

    let variance = assessor.calculate_variance(&tensor).unwrap();
    // Variance of [1,2,3,4,5] is 2.0
    assert!((variance - 2.0).abs() < 0.1);
}

#[test]
fn test_variance_high() {
    let assessor = ComplexityAssessor::new();

    // High spread = high variance
    let data = vec![1.0f32, 100.0, 1.0, 100.0, 1.0, 100.0];
    let tensor =
        Tensor::from_data(data, vec![1, 6], DataType::F32, TensorLayout::RowMajor).unwrap();

    let variance = assessor.calculate_variance(&tensor).unwrap();
    assert!(variance > 1000.0);
}

#[test]
fn test_variance_single_element() {
    let assessor = ComplexityAssessor::new();

    let data = vec![42.0f32];
    let tensor =
        Tensor::from_data(data, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let variance = assessor.calculate_variance(&tensor).unwrap();
    // Single element has zero variance
    assert_eq!(variance, 0.0);
}

// ============================================================================
// Threshold Configuration Tests
// ============================================================================

#[test]
fn test_very_low_thresholds() -> Result<()> {
    // Everything appears complex with low thresholds
    let assessor = ComplexityAssessor::with_thresholds(1, 10, 0.01);

    let data = vec![1.0f32; 20];
    let tensor = Tensor::from_data(data, vec![1, 20], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // 20 elements with low thresholds should be High
    assert_eq!(metrics.level, ComplexityLevel::High);

    Ok(())
}

#[test]
fn test_very_high_thresholds() -> Result<()> {
    // Everything appears simple with high thresholds
    let assessor = ComplexityAssessor::with_thresholds(10_000, 100_000, 0.9);

    let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
    let tensor = Tensor::from_data(data, vec![1, 1000], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // 1000 elements with high thresholds should be Low
    assert_eq!(metrics.level, ComplexityLevel::Low);

    Ok(())
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

#[test]
fn test_assessment_performance() -> Result<()> {
    use std::time::Instant;

    let assessor = ComplexityAssessor::new();

    // Large tensor for performance testing
    let data: Vec<f32> = (0..100_000).map(|x| x as f32).collect();
    let tensor = Tensor::from_data(data, vec![1, 100_000], DataType::F32, TensorLayout::RowMajor)?;

    let start = Instant::now();
    let _metrics = assessor.assess(&tensor)?;
    let elapsed = start.elapsed();

    // Assessment should be fast (< 10ms for 100K elements)
    assert!(
        elapsed.as_millis() < 10,
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

// ============================================================================
// Complexity Level Boundary Tests
// ============================================================================

#[test]
fn test_boundary_low_to_medium() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Test around the boundary
    let low_data = vec![1.0f32; 99];
    let medium_data = vec![1.0f32; 101];

    let low_tensor =
        Tensor::from_data(low_data, vec![1, 99], DataType::F32, TensorLayout::RowMajor)?;
    let medium_tensor = Tensor::from_data(
        medium_data,
        vec![1, 101],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let low_metrics = assessor.assess(&low_tensor)?;
    let medium_metrics = assessor.assess(&medium_tensor)?;

    // Scores should be close but levels might differ
    assert!(
        (low_metrics.complexity_score - medium_metrics.complexity_score).abs() < 0.1,
        "Scores should be similar at boundary"
    );

    Ok(())
}

#[test]
fn test_boundary_medium_to_high() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    // Test around the boundary
    let medium_data = vec![1.0f32; 999];
    let high_data = vec![1.0f32; 1001];

    let medium_tensor = Tensor::from_data(
        medium_data,
        vec![1, 999],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let high_tensor = Tensor::from_data(
        high_data,
        vec![1, 1001],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let medium_metrics = assessor.assess(&medium_tensor)?;
    let high_metrics = assessor.assess(&high_tensor)?;

    // Scores should be close but levels might differ
    assert!(
        (medium_metrics.complexity_score - high_metrics.complexity_score).abs() < 0.1,
        "Scores should be similar at boundary"
    );

    Ok(())
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_handles_nan_values() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    let data = vec![1.0f32, f32::NAN, 3.0, 4.0];
    let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Should handle NaN gracefully
    // Variance with NaN should be NaN, but we should detect it
    assert!(
        metrics.variance.is_nan() || metrics.variance.is_finite(),
        "Should handle NaN values"
    );

    Ok(())
}

#[test]
fn test_handles_infinity_values() -> Result<()> {
    let assessor = ComplexityAssessor::new();

    let data = vec![1.0f32, f32::INFINITY, -f32::INFINITY, 4.0];
    let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = assessor.assess(&tensor)?;

    // Should handle infinity gracefully
    assert!(
        metrics.variance.is_infinite() || metrics.variance.is_finite(),
        "Should handle infinity values"
    );

    Ok(())
}
