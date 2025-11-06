//! Execution Paths for System 1 and System 2
//!
//! Implements the dual execution paths inspired by cognitive psychology:
//! - System 1: Fast, automatic, intuitive processing (uses quantized models)
//! - System 2: Slow, deliberate, analytical processing (uses full precision)

use crate::Result;
use ronn_core::tensor::Tensor;

/// Execution path selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPath {
    /// System 1: Fast, automatic processing (BitNet/quantized)
    System1,

    /// System 2: Slow, deliberate processing (full precision)
    System2,

    /// Hybrid: Use both systems and combine results
    Hybrid,
}

/// Low-level executor for System 1 (fast path)
///
/// This executor is designed for:
/// - Simple, repeated patterns
/// - Low-complexity inputs
/// - Time-critical inference
/// - Edge deployment scenarios
///
/// Uses quantized models (BitNet) for efficiency.
pub struct LowLevelExecutor {
    /// Pattern cache for frequent inputs
    cache_enabled: bool,

    /// Number of cached patterns
    cache_size: usize,
}

impl LowLevelExecutor {
    /// Create a new low-level executor
    pub fn new() -> Self {
        Self {
            cache_enabled: true,
            cache_size: 0,
        }
    }

    /// Create executor with caching disabled
    pub fn without_cache() -> Self {
        Self {
            cache_enabled: false,
            cache_size: 0,
        }
    }

    /// Execute on the fast path (System 1)
    ///
    /// For MVP, this simulates BitNet execution by:
    /// 1. Quantizing the input (simulated)
    /// 2. Performing fast approximate computation
    /// 3. Dequantizing the output (simulated)
    pub fn execute(&self, input: &Tensor) -> Result<Tensor> {
        // For MVP: Simulate fast execution by returning a simplified result
        // In production, this would route to BitNet provider

        // Simple pass-through with simulated quantization noise
        // Real implementation would use actual BitNet provider
        let output = self.simulate_fast_inference(input)?;

        Ok(output)
    }

    /// Simulate fast inference (MVP implementation)
    fn simulate_fast_inference(&self, input: &Tensor) -> Result<Tensor> {
        // For demonstration: just return a copy
        // Real implementation would:
        // 1. Quantize to 1-bit
        // 2. Run through BitNet model
        // 3. Dequantize output

        Ok(input.clone())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            enabled: self.cache_enabled,
            size: self.cache_size,
            hit_rate: 0.0, // Would track in real implementation
        }
    }
}

impl Default for LowLevelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level planner for System 2 (slow path)
///
/// This executor is designed for:
/// - Complex, novel inputs
/// - High-accuracy requirements
/// - Reasoning tasks
/// - Critical decisions
///
/// Uses full precision models for maximum accuracy.
pub struct HighLevelPlanner {
    /// Whether to use problem decomposition
    decomposition_enabled: bool,

    /// Maximum planning depth
    max_depth: usize,
}

impl HighLevelPlanner {
    /// Create a new high-level planner
    pub fn new() -> Self {
        Self {
            decomposition_enabled: false,
            max_depth: 3,
        }
    }

    /// Create planner with problem decomposition
    pub fn with_decomposition(max_depth: usize) -> Self {
        Self {
            decomposition_enabled: true,
            max_depth,
        }
    }

    /// Execute on the slow path (System 2)
    ///
    /// For MVP, this simulates full precision execution.
    /// In production, this would:
    /// 1. Analyze the problem structure
    /// 2. Potentially decompose into subproblems
    /// 3. Route to full precision provider
    /// 4. Perform careful, accurate computation
    pub fn execute(&self, input: &Tensor) -> Result<Tensor> {
        // For MVP: Simulate deliberate processing
        let output = self.simulate_deliberate_inference(input)?;

        Ok(output)
    }

    /// Simulate deliberate inference (MVP implementation)
    fn simulate_deliberate_inference(&self, input: &Tensor) -> Result<Tensor> {
        // For demonstration: just return a copy
        // Real implementation would:
        // 1. Analyze input structure
        // 2. Potentially decompose problem
        // 3. Run through full precision model
        // 4. Verify output quality

        Ok(input.clone())
    }

    /// Get planning statistics
    pub fn planning_stats(&self) -> PlanningStats {
        PlanningStats {
            decomposition_enabled: self.decomposition_enabled,
            max_depth: self.max_depth,
            avg_depth: 0.0, // Would track in real implementation
        }
    }
}

impl Default for HighLevelPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics for System 1
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub enabled: bool,
    pub size: usize,
    pub hit_rate: f64,
}

/// Planning statistics for System 2
#[derive(Debug, Clone)]
pub struct PlanningStats {
    pub decomposition_enabled: bool,
    pub max_depth: usize,
    pub avg_depth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

    #[test]
    fn test_system1_executor() -> Result<()> {
        let executor = LowLevelExecutor::new();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data.clone(), vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

        let output = executor.execute(&input)?;

        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_system1_without_cache() -> Result<()> {
        let executor = LowLevelExecutor::without_cache();
        let stats = executor.cache_stats();

        assert!(!stats.enabled);

        Ok(())
    }

    #[test]
    fn test_system2_executor() -> Result<()> {
        let executor = HighLevelPlanner::new();

        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let input = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

        let output = executor.execute(&input)?;

        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_system2_with_decomposition() -> Result<()> {
        let executor = HighLevelPlanner::with_decomposition(5);
        let stats = executor.planning_stats();

        assert!(stats.decomposition_enabled);
        assert_eq!(stats.max_depth, 5);

        Ok(())
    }

    #[test]
    fn test_execution_paths() {
        assert_eq!(ExecutionPath::System1, ExecutionPath::System1);
        assert_ne!(ExecutionPath::System1, ExecutionPath::System2);
    }
}
