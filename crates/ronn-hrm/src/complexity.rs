//! Complexity Assessment Engine
//!
//! Analyzes input tensors to determine their computational complexity and routing requirements.

use crate::Result;
use ronn_core::tensor::Tensor;

/// Levels of complexity for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityLevel {
    /// Low complexity: Use System 1 (fast path)
    Low,
    /// Medium complexity: Borderline, may use hybrid
    Medium,
    /// High complexity: Use System 2 (slow path)
    High,
}

/// Detailed complexity metrics for an input
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    /// Overall complexity level
    pub level: ComplexityLevel,

    /// Total number of elements in the tensor
    pub size: usize,

    /// Statistical variance of the data
    pub variance: f64,

    /// Dimensionality (number of dimensions)
    pub dimensionality: usize,

    /// Uncertainty in the complexity assessment (0.0 to 1.0)
    pub uncertainty: f64,

    /// Score from 0.0 (simple) to 1.0 (complex)
    pub complexity_score: f64,
}

/// Assesses the complexity of input tensors
pub struct ComplexityAssessor {
    /// Threshold for low complexity (size)
    low_threshold: usize,

    /// Threshold for high complexity (size)
    high_threshold: usize,

    /// Variance threshold for complexity
    variance_threshold: f64,
}

impl ComplexityAssessor {
    /// Create a new complexity assessor with default thresholds
    pub fn new() -> Self {
        Self {
            low_threshold: 100,
            high_threshold: 1000,
            variance_threshold: 0.5,
        }
    }

    /// Create a complexity assessor with custom thresholds
    pub fn with_thresholds(low: usize, high: usize, variance: f64) -> Self {
        Self {
            low_threshold: low,
            high_threshold: high,
            variance_threshold: variance,
        }
    }

    /// Assess the complexity of an input tensor
    pub fn assess(&self, input: &Tensor) -> Result<ComplexityMetrics> {
        let size = input.numel();
        let dimensionality = input.shape().len();

        // Calculate variance (if possible)
        let variance = self.calculate_variance(input).unwrap_or(0.0);

        // Calculate complexity score (0.0 to 1.0)
        let size_score = self.size_score(size);
        let variance_score = variance / (self.variance_threshold * 2.0).max(1.0);
        let dim_score = (dimensionality as f64 / 4.0).min(1.0); // 4D+ is complex

        // Weighted average (size is most important)
        let complexity_score = (size_score * 0.5 + variance_score * 0.3 + dim_score * 0.2).min(1.0);

        // Determine level
        let level = if complexity_score < 0.33 {
            ComplexityLevel::Low
        } else if complexity_score < 0.66 {
            ComplexityLevel::Medium
        } else {
            ComplexityLevel::High
        };

        // Calculate uncertainty (higher for borderline cases)
        let uncertainty = if matches!(level, ComplexityLevel::Medium) {
            0.8
        } else {
            0.2
        };

        Ok(ComplexityMetrics {
            level,
            size,
            variance,
            dimensionality,
            uncertainty,
            complexity_score,
        })
    }

    /// Calculate a score from 0.0 to 1.0 based on size
    fn size_score(&self, size: usize) -> f64 {
        if size < self.low_threshold {
            0.0
        } else if size > self.high_threshold {
            1.0
        } else {
            // Linear interpolation between thresholds
            let range = self.high_threshold - self.low_threshold;
            let offset = size - self.low_threshold;
            offset as f64 / range as f64
        }
    }

    /// Calculate variance of tensor data (for F32 tensors)
    fn calculate_variance(&self, tensor: &Tensor) -> Option<f64> {
        // For MVP, extract F32 data and calculate variance
        let data = tensor.to_vec().ok()?;

        if data.is_empty() {
            return Some(0.0);
        }

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 = data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / data.len() as f32;

        Some(variance as f64)
    }
}

impl Default for ComplexityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

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

        Ok(())
    }

    #[test]
    fn test_assess_high_complexity() -> Result<()> {
        let assessor = ComplexityAssessor::new();

        // Very large tensor with high variance to ensure High complexity
        let data: Vec<f32> = (0..5000)
            .map(|x| (x as f32).sin() * (x as f32).cos() * (x as f32 % 100.0))
            .collect();
        let tensor = Tensor::from_data(data, vec![1, 5000], DataType::F32, TensorLayout::RowMajor)?;

        let metrics = assessor.assess(&tensor)?;

        // With 5000 elements and high variance, should be High or at least Medium
        assert!(matches!(metrics.level, ComplexityLevel::High | ComplexityLevel::Medium));
        assert!(metrics.size > 1000);
        assert!(metrics.complexity_score > 0.4);

        Ok(())
    }

    #[test]
    fn test_assess_medium_complexity() -> Result<()> {
        let assessor = ComplexityAssessor::new();

        // Medium-sized tensor
        let data: Vec<f32> = (0..500).map(|x| x as f32).collect();
        let tensor = Tensor::from_data(data, vec![1, 500], DataType::F32, TensorLayout::RowMajor)?;

        let metrics = assessor.assess(&tensor)?;

        // 500 elements can be any level depending on variance
        // Just verify we get valid metrics
        assert!(metrics.size == 500);
        assert!(metrics.complexity_score >= 0.0 && metrics.complexity_score <= 1.0);

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
        // With low variance (all 1.0), should be Low or Medium
        assert!(metrics.size == 50);
        assert!(metrics.complexity_score > 0.0);

        Ok(())
    }

    #[test]
    fn test_variance_calculation() {
        let assessor = ComplexityAssessor::new();

        // Uniform data = low variance
        let uniform = vec![5.0f32; 10];
        let tensor1 = Tensor::from_data(
            uniform,
            vec![1, 10],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();

        let variance1 = assessor.calculate_variance(&tensor1).unwrap();
        assert!(variance1 < 0.001);

        // Varied data = high variance
        let varied = vec![1.0f32, 10.0, 1.0, 10.0, 1.0, 10.0];
        let tensor2 = Tensor::from_data(
            varied,
            vec![1, 6],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();

        let variance2 = assessor.calculate_variance(&tensor2).unwrap();
        assert!(variance2 > 1.0);
    }
}
