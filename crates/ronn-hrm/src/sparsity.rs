//! Sparsity-aware routing for the HRM system.
//!
//! Maps the hot/cold neuron paradigm to the existing System 1/System 2 framework:
//! - Hot neurons = System 1 (fast, always ready, GPU-resident)
//! - Cold neurons = System 2 (on-demand, computed only when predicted active)
//!
//! The HRM can dynamically adjust sparsity thresholds based on task type:
//! - Creative/exploratory tasks -> lower sparsity (activate more neurons)
//! - Routine/pattern-matched tasks -> higher sparsity (maximum speed)

use crate::complexity::{ComplexityLevel, ComplexityMetrics};
use crate::executor::ExecutionPath;
use serde::{Deserialize, Serialize};

/// Sparsity-aware routing decision for FFN layers.
#[derive(Debug, Clone)]
pub struct SparsityRoutingDecision {
    /// Whether to use sparse execution for this input.
    pub use_sparse: bool,
    /// Prediction threshold adjustment (higher = more aggressive sparsity).
    pub prediction_threshold: f32,
    /// Which execution path maps to hot neurons.
    pub hot_path: ExecutionPath,
    /// Which execution path maps to cold neurons.
    pub cold_path: ExecutionPath,
    /// Reason for the decision.
    pub reason: String,
}

/// Configuration for sparsity-aware HRM routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityRoutingConfig {
    /// Base prediction threshold (default 0.5).
    pub base_threshold: f32,
    /// Threshold adjustment for low-complexity (routine) inputs.
    /// Higher means more aggressive sparsity.
    pub low_complexity_threshold: f32,
    /// Threshold adjustment for high-complexity (creative) inputs.
    /// Lower means more neurons activated.
    pub high_complexity_threshold: f32,
    /// Minimum sparsity ratio to bother using sparse execution.
    pub min_sparsity_for_sparse: f64,
}

impl Default for SparsityRoutingConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.5,
            low_complexity_threshold: 0.7,   // Routine tasks: skip more
            high_complexity_threshold: 0.3,    // Complex tasks: keep more active
            min_sparsity_for_sparse: 0.3,
        }
    }
}

/// Sparsity-aware router that integrates with the HRM complexity assessment.
pub struct SparsityRouter {
    config: SparsityRoutingConfig,
}

impl SparsityRouter {
    /// Create a new sparsity router with the given configuration.
    pub fn new(config: SparsityRoutingConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SparsityRoutingConfig::default())
    }

    /// Make a sparsity routing decision based on complexity metrics.
    ///
    /// This maps the HRM's complexity assessment to sparsity parameters:
    /// - Low complexity (System 1) -> aggressive sparsity, maximum speed
    /// - High complexity (System 2) -> conservative sparsity, maximum accuracy
    /// - Medium complexity -> balanced approach
    pub fn route(&self, metrics: &ComplexityMetrics) -> SparsityRoutingDecision {
        match metrics.level {
            ComplexityLevel::Low => SparsityRoutingDecision {
                use_sparse: true,
                prediction_threshold: self.config.low_complexity_threshold,
                hot_path: ExecutionPath::System1,
                cold_path: ExecutionPath::System1,
                reason: format!(
                    "Low complexity (score: {:.2}): aggressive sparsity, System 1 path",
                    metrics.complexity_score
                ),
            },

            ComplexityLevel::High => SparsityRoutingDecision {
                use_sparse: true,
                prediction_threshold: self.config.high_complexity_threshold,
                hot_path: ExecutionPath::System2,
                cold_path: ExecutionPath::System2,
                reason: format!(
                    "High complexity (score: {:.2}): conservative sparsity, System 2 path",
                    metrics.complexity_score
                ),
            },

            ComplexityLevel::Medium => {
                let threshold = self.config.base_threshold;
                SparsityRoutingDecision {
                    use_sparse: true,
                    prediction_threshold: threshold,
                    hot_path: ExecutionPath::System1,
                    cold_path: ExecutionPath::System2,
                    reason: format!(
                        "Medium complexity (score: {:.2}): balanced sparsity, hybrid path",
                        metrics.complexity_score
                    ),
                }
            }
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SparsityRoutingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(level: ComplexityLevel, score: f64) -> ComplexityMetrics {
        ComplexityMetrics {
            level,
            size: 100,
            variance: 0.5,
            dimensionality: 2,
            uncertainty: 0.3,
            complexity_score: score,
        }
    }

    #[test]
    fn test_low_complexity_routing() {
        let router = SparsityRouter::with_defaults();
        let metrics = make_metrics(ComplexityLevel::Low, 0.2);
        let decision = router.route(&metrics);

        assert!(decision.use_sparse);
        assert!(decision.prediction_threshold > 0.5); // Aggressive sparsity
        assert_eq!(decision.hot_path, ExecutionPath::System1);
    }

    #[test]
    fn test_high_complexity_routing() {
        let router = SparsityRouter::with_defaults();
        let metrics = make_metrics(ComplexityLevel::High, 0.8);
        let decision = router.route(&metrics);

        assert!(decision.use_sparse);
        assert!(decision.prediction_threshold < 0.5); // Conservative
        assert_eq!(decision.hot_path, ExecutionPath::System2);
    }

    #[test]
    fn test_medium_complexity_routing() {
        let router = SparsityRouter::with_defaults();
        let metrics = make_metrics(ComplexityLevel::Medium, 0.5);
        let decision = router.route(&metrics);

        assert!(decision.use_sparse);
        assert!((decision.prediction_threshold - 0.5).abs() < f32::EPSILON);
        // Medium maps hot to System1, cold to System2
        assert_eq!(decision.hot_path, ExecutionPath::System1);
        assert_eq!(decision.cold_path, ExecutionPath::System2);
    }
}
