//! Hierarchical Reasoning Module (HRM) - Brain-Inspired Dual-Process Inference
//!
//! Implements a dual-process cognitive architecture inspired by Kahneman's System 1/System 2 theory:
//! - **System 1 (Fast)**: Automatic, intuitive processing using quantized models (BitNet)
//! - **System 2 (Slow)**: Deliberative, analytical processing using full precision models
//!
//! ## Architecture
//!
//! ```text
//! Input → Complexity Assessment → Routing Decision
//!                                        ↓
//!                            ┌───────────┴───────────┐
//!                            ↓                       ↓
//!                    System 1 (Fast)        System 2 (Slow)
//!                    BitNet Provider        Full Precision
//!                    Low Complexity         High Complexity
//!                            ↓                       ↓
//!                            └───────────┬───────────┘
//!                                        ↓
//!                                     Output
//! ```

pub mod complexity;
pub mod executor;
pub mod router;
pub mod sparsity;

pub use complexity::{ComplexityAssessor, ComplexityLevel, ComplexityMetrics};
pub use executor::{ExecutionPath, HighLevelPlanner, LowLevelExecutor};
pub use router::{HRMRouter, RoutingDecision, RoutingStrategy};
pub use sparsity::{SparsityRouter, SparsityRoutingConfig, SparsityRoutingDecision};

use ronn_core::tensor::Tensor;
use thiserror::Error;

/// Errors that can occur in the HRM system
#[derive(Error, Debug)]
pub enum HRMError {
    #[error("Complexity assessment failed: {0}")]
    ComplexityAssessment(String),

    #[error("Execution failed: {0}")]
    Execution(String),

    #[error("Routing failed: {0}")]
    Routing(String),

    #[error("Core error: {0}")]
    Core(#[from] ronn_core::error::CoreError),
}

pub type Result<T> = std::result::Result<T, HRMError>;

/// Main HRM coordinator that manages dual-process inference
pub struct HierarchicalReasoningModule {
    router: HRMRouter,
    low_level: LowLevelExecutor,
    high_level: HighLevelPlanner,
    metrics: HRMMetrics,
}

impl HierarchicalReasoningModule {
    /// Create a new HRM with default configuration
    pub fn new() -> Self {
        Self {
            router: HRMRouter::new(RoutingStrategy::AdaptiveComplexity),
            low_level: LowLevelExecutor::new(),
            high_level: HighLevelPlanner::new(),
            metrics: HRMMetrics::default(),
        }
    }

    /// Create a new HRM with custom routing strategy
    pub fn with_strategy(strategy: RoutingStrategy) -> Self {
        Self {
            router: HRMRouter::new(strategy),
            low_level: LowLevelExecutor::new(),
            high_level: HighLevelPlanner::new(),
            metrics: HRMMetrics::default(),
        }
    }

    /// Process input through the HRM system
    pub fn process(&mut self, input: &Tensor) -> Result<ProcessingResult> {
        // 1. Assess complexity
        let complexity = self.router.assess_complexity(input)?;

        // 2. Make routing decision
        let decision = self.router.route(&complexity)?;

        // 3. Execute on appropriate path
        let result = match decision.path {
            ExecutionPath::System1 => {
                self.metrics.system1_count += 1;
                self.low_level.execute(input)?
            }
            ExecutionPath::System2 => {
                self.metrics.system2_count += 1;
                self.high_level.execute(input)?
            }
            ExecutionPath::Hybrid => {
                self.metrics.hybrid_count += 1;
                self.hybrid_execute(input, &complexity)?
            }
        };

        Ok(ProcessingResult {
            output: result,
            path_taken: decision.path,
            complexity_metrics: complexity,
            confidence: decision.confidence,
        })
    }

    /// Hybrid execution: use both paths and combine results
    fn hybrid_execute(&self, input: &Tensor, complexity: &ComplexityMetrics) -> Result<Tensor> {
        // Use System 1 for initial fast pass
        let system1_result = self.low_level.execute(input)?;

        // If complexity is borderline, validate with System 2
        if complexity.uncertainty > 0.5 {
            let system2_result = self.high_level.execute(input)?;

            // Blend results based on confidence
            // For MVP, just return System 2 result (more accurate)
            Ok(system2_result)
        } else {
            Ok(system1_result)
        }
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &HRMMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = HRMMetrics::default();
    }
}

impl Default for HierarchicalReasoningModule {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing through the HRM
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// The output tensor
    pub output: Tensor,

    /// Which execution path was taken
    pub path_taken: ExecutionPath,

    /// Complexity metrics for this input
    pub complexity_metrics: ComplexityMetrics,

    /// Confidence in the routing decision (0.0 to 1.0)
    pub confidence: f64,
}

/// Performance metrics for the HRM system
#[derive(Debug, Clone, Default)]
pub struct HRMMetrics {
    /// Number of times System 1 (fast path) was used
    pub system1_count: u64,

    /// Number of times System 2 (slow path) was used
    pub system2_count: u64,

    /// Number of times hybrid execution was used
    pub hybrid_count: u64,
}

impl HRMMetrics {
    /// Get total number of inferences
    pub fn total_inferences(&self) -> u64 {
        self.system1_count + self.system2_count + self.hybrid_count
    }

    /// Get percentage of inferences using System 1
    pub fn system1_percentage(&self) -> f64 {
        let total = self.total_inferences();
        if total == 0 {
            0.0
        } else {
            (self.system1_count as f64 / total as f64) * 100.0
        }
    }

    /// Get percentage of inferences using System 2
    pub fn system2_percentage(&self) -> f64 {
        let total = self.total_inferences();
        if total == 0 {
            0.0
        } else {
            (self.system2_count as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

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
    fn test_process_simple_input() -> Result<()> {
        let mut hrm = HierarchicalReasoningModule::new();

        // Create a simple input tensor (small size = low complexity)
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

        let result = hrm.process(&tensor)?;

        // Small tensors should route to System 1
        assert!(matches!(result.path_taken, ExecutionPath::System1));
        assert_eq!(hrm.metrics().system1_count, 1);

        Ok(())
    }

    #[test]
    fn test_process_complex_input() -> Result<()> {
        let mut hrm = HierarchicalReasoningModule::new();

        // Create a complex input tensor (large size + high variance)
        let data: Vec<f32> = (0..5000)
            .map(|x| (x as f32).sin() * (x as f32).cos() * (x as f32 % 100.0))
            .collect();
        let tensor = Tensor::from_data(data, vec![1, 5000], DataType::F32, TensorLayout::RowMajor)?;

        let result = hrm.process(&tensor)?;

        // Large/complex tensors should not route to Hybrid
        assert!(matches!(
            result.path_taken,
            ExecutionPath::System1 | ExecutionPath::System2
        ));
        assert!(hrm.metrics().total_inferences() == 1);

        Ok(())
    }

    #[test]
    fn test_metrics_tracking() -> Result<()> {
        let mut hrm = HierarchicalReasoningModule::new();

        // Process multiple inputs
        let simple = Tensor::from_data(
            vec![1.0f32; 4],
            vec![1, 4],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let complex = Tensor::from_data(
            (0..1000).map(|x| x as f32).collect(),
            vec![1, 1000],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        hrm.process(&simple)?;
        hrm.process(&complex)?;
        hrm.process(&simple)?;

        assert_eq!(hrm.metrics().total_inferences(), 3);
        assert!(hrm.metrics().system1_count >= 1);

        Ok(())
    }
}
