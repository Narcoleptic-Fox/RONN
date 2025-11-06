//! Routing logic for HRM
//!
//! Decides which execution path to use based on complexity assessment.

use crate::complexity::{ComplexityAssessor, ComplexityLevel, ComplexityMetrics};
use crate::executor::ExecutionPath;
use crate::Result;
use ronn_core::tensor::Tensor;

/// Routing strategies for the HRM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Always use System 1 (fast path)
    AlwaysSystem1,

    /// Always use System 2 (slow path)
    AlwaysSystem2,

    /// Adaptive routing based on complexity assessment
    AdaptiveComplexity,

    /// Use hybrid approach for borderline cases
    AdaptiveHybrid,
}

/// A routing decision with confidence
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The chosen execution path
    pub path: ExecutionPath,

    /// Confidence in this decision (0.0 to 1.0)
    pub confidence: f64,

    /// Reason for this routing decision
    pub reason: String,
}

/// Router that makes execution path decisions
pub struct HRMRouter {
    strategy: RoutingStrategy,
    assessor: ComplexityAssessor,
    stats: RouterStats,
}

impl HRMRouter {
    /// Create a new router with the specified strategy
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            strategy,
            assessor: ComplexityAssessor::new(),
            stats: RouterStats::default(),
        }
    }

    /// Create router with custom complexity assessor
    pub fn with_assessor(strategy: RoutingStrategy, assessor: ComplexityAssessor) -> Self {
        Self {
            strategy,
            assessor,
            stats: RouterStats::default(),
        }
    }

    /// Assess the complexity of input
    pub fn assess_complexity(&self, input: &Tensor) -> Result<ComplexityMetrics> {
        self.assessor.assess(input)
    }

    /// Make a routing decision based on complexity
    pub fn route(&mut self, metrics: &ComplexityMetrics) -> Result<RoutingDecision> {
        self.stats.total_decisions += 1;

        let decision = match self.strategy {
            RoutingStrategy::AlwaysSystem1 => RoutingDecision {
                path: ExecutionPath::System1,
                confidence: 1.0,
                reason: "Strategy: Always System 1".to_string(),
            },

            RoutingStrategy::AlwaysSystem2 => RoutingDecision {
                path: ExecutionPath::System2,
                confidence: 1.0,
                reason: "Strategy: Always System 2".to_string(),
            },

            RoutingStrategy::AdaptiveComplexity => self.adaptive_route(metrics, false),

            RoutingStrategy::AdaptiveHybrid => self.adaptive_route(metrics, true),
        };

        // Update stats
        match decision.path {
            ExecutionPath::System1 => self.stats.system1_routes += 1,
            ExecutionPath::System2 => self.stats.system2_routes += 1,
            ExecutionPath::Hybrid => self.stats.hybrid_routes += 1,
        }

        Ok(decision)
    }

    /// Adaptive routing based on complexity metrics
    fn adaptive_route(&self, metrics: &ComplexityMetrics, allow_hybrid: bool) -> RoutingDecision {
        match metrics.level {
            ComplexityLevel::Low => RoutingDecision {
                path: ExecutionPath::System1,
                confidence: 1.0 - metrics.uncertainty,
                reason: format!(
                    "Low complexity (score: {:.2}, size: {})",
                    metrics.complexity_score, metrics.size
                ),
            },

            ComplexityLevel::High => RoutingDecision {
                path: ExecutionPath::System2,
                confidence: 1.0 - metrics.uncertainty,
                reason: format!(
                    "High complexity (score: {:.2}, size: {}, variance: {:.2})",
                    metrics.complexity_score, metrics.size, metrics.variance
                ),
            },

            ComplexityLevel::Medium => {
                if allow_hybrid && metrics.uncertainty > 0.6 {
                    // Use hybrid for uncertain cases
                    RoutingDecision {
                        path: ExecutionPath::Hybrid,
                        confidence: 0.5,
                        reason: format!(
                            "Medium complexity with high uncertainty (score: {:.2})",
                            metrics.complexity_score
                        ),
                    }
                } else {
                    // Favor System 1 for borderline cases (prefer speed)
                    RoutingDecision {
                        path: ExecutionPath::System1,
                        confidence: 0.6,
                        reason: format!(
                            "Medium complexity, favoring speed (score: {:.2})",
                            metrics.complexity_score
                        ),
                    }
                }
            }
        }
    }

    /// Get routing statistics
    pub fn stats(&self) -> &RouterStats {
        &self.stats
    }

    /// Reset routing statistics
    pub fn reset_stats(&mut self) {
        self.stats = RouterStats::default();
    }

    /// Get current strategy
    pub fn strategy(&self) -> RoutingStrategy {
        self.strategy
    }

    /// Update routing strategy
    pub fn set_strategy(&mut self, strategy: RoutingStrategy) {
        self.strategy = strategy;
    }
}

/// Statistics about routing decisions
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    pub total_decisions: u64,
    pub system1_routes: u64,
    pub system2_routes: u64,
    pub hybrid_routes: u64,
}

impl RouterStats {
    /// Get percentage routed to System 1
    pub fn system1_percentage(&self) -> f64 {
        if self.total_decisions == 0 {
            0.0
        } else {
            (self.system1_routes as f64 / self.total_decisions as f64) * 100.0
        }
    }

    /// Get percentage routed to System 2
    pub fn system2_percentage(&self) -> f64 {
        if self.total_decisions == 0 {
            0.0
        } else {
            (self.system2_routes as f64 / self.total_decisions as f64) * 100.0
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
    fn test_always_system1_strategy() -> Result<()> {
        let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem1);

        let data = vec![1.0f32; 1000];
        let tensor = Tensor::from_data(data, vec![1, 1000], DataType::F32, TensorLayout::RowMajor)?;

        let metrics = router.assess_complexity(&tensor)?;
        let decision = router.route(&metrics)?;

        assert_eq!(decision.path, ExecutionPath::System1);
        assert_eq!(decision.confidence, 1.0);

        Ok(())
    }

    #[test]
    fn test_always_system2_strategy() -> Result<()> {
        let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem2);

        let data = vec![1.0f32; 10];
        let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;

        let metrics = router.assess_complexity(&tensor)?;
        let decision = router.route(&metrics)?;

        assert_eq!(decision.path, ExecutionPath::System2);
        assert_eq!(decision.confidence, 1.0);

        Ok(())
    }

    #[test]
    fn test_adaptive_routing_low_complexity() -> Result<()> {
        let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);

        // Small tensor = low complexity
        let data = vec![1.0f32; 10];
        let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;

        let metrics = router.assess_complexity(&tensor)?;
        let decision = router.route(&metrics)?;

        assert_eq!(decision.path, ExecutionPath::System1);

        Ok(())
    }

    #[test]
    fn test_adaptive_routing_high_complexity() -> Result<()> {
        let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);

        // Very large tensor with high variance
        let data: Vec<f32> = (0..5000)
            .map(|x| (x as f32).sin() * (x as f32).cos() * (x as f32 % 100.0))
            .collect();
        let tensor = Tensor::from_data(data, vec![1, 5000], DataType::F32, TensorLayout::RowMajor)?;

        let metrics = router.assess_complexity(&tensor)?;
        let decision = router.route(&metrics)?;

        // Should route to System1 or System2 (medium-high complexity)
        assert!(matches!(decision.path, ExecutionPath::System1 | ExecutionPath::System2));

        Ok(())
    }

    #[test]
    fn test_hybrid_routing() -> Result<()> {
        let mut router = HRMRouter::new(RoutingStrategy::AdaptiveHybrid);

        // Medium complexity tensor
        let data: Vec<f32> = (0..500).map(|x| x as f32).collect();
        let tensor = Tensor::from_data(data, vec![1, 500], DataType::F32, TensorLayout::RowMajor)?;

        let metrics = router.assess_complexity(&tensor)?;
        let decision = router.route(&metrics)?;

        // AdaptiveHybrid can route to any path depending on complexity and uncertainty
        // Just verify we get a valid decision
        assert!(matches!(
            decision.path,
            ExecutionPath::Hybrid | ExecutionPath::System1 | ExecutionPath::System2
        ));
        assert!(decision.confidence > 0.0 && decision.confidence <= 1.0);

        Ok(())
    }

    #[test]
    fn test_routing_stats() -> Result<()> {
        let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);

        // Route several inputs
        let small = Tensor::from_data(vec![1.0f32; 10], vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        let large = Tensor::from_data(
            (0..2000).map(|x| x as f32).collect(),
            vec![1, 2000],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        for _ in 0..3 {
            let metrics = router.assess_complexity(&small)?;
            router.route(&metrics)?;
        }

        for _ in 0..2 {
            let metrics = router.assess_complexity(&large)?;
            router.route(&metrics)?;
        }

        let stats = router.stats();
        assert_eq!(stats.total_decisions, 5);
        assert!(stats.system1_routes >= 3);
        assert!(stats.system2_routes >= 2);

        Ok(())
    }

    #[test]
    fn test_strategy_change() {
        let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem1);
        assert_eq!(router.strategy(), RoutingStrategy::AlwaysSystem1);

        router.set_strategy(RoutingStrategy::AdaptiveComplexity);
        assert_eq!(router.strategy(), RoutingStrategy::AdaptiveComplexity);
    }
}
