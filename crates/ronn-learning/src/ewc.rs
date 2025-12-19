//! Elastic Weight Consolidation (EWC) - Protecting important weights

use crate::timescales::WeightUpdate;
use crate::Result;
use ronn_core::tensor::Tensor;

/// Importance weights for EWC
#[derive(Debug, Clone)]
pub struct ImportanceWeights {
    pub weights: Vec<f32>,
}

impl ImportanceWeights {
    pub fn new(size: usize) -> Self {
        Self {
            weights: vec![0.0; size],
        }
    }
}

/// Elastic Weight Consolidation for continual learning
pub struct ElasticWeightConsolidation {
    lambda: f64, // EWC regularization strength
    importance: ImportanceWeights,
    optimal_weights: Vec<f32>, // Weights from previous task
}

impl ElasticWeightConsolidation {
    /// Create new EWC with regularization strength
    pub fn new(lambda: f64) -> Self {
        let size = 100; // Match learner size

        Self {
            lambda,
            importance: ImportanceWeights::new(size),
            optimal_weights: vec![0.0; size],
        }
    }

    /// Compute importance of weights for a task
    pub fn compute_importance(&mut self, task_data: &[(Tensor, Tensor)]) -> Result<()> {
        // For MVP: Simulate importance computation
        // In production: Compute Fisher Information Matrix

        let importance_increment = (task_data.len() as f32 * 0.1).min(1.0);
        for weight in self.importance.weights.iter_mut() {
            // Simulate: weights that change more are more important
            *weight += importance_increment;
        }

        Ok(())
    }

    /// Constrain weight update to protect important weights
    pub fn constrain_update(&self, update: &WeightUpdate) -> Result<WeightUpdate> {
        let mut constrained_fast = update.fast_delta.clone();
        let mut constrained_slow = update.slow_delta.clone();
        let mut total_penalty = 0.0;

        // Apply EWC penalty to protect important weights
        for i in 0..constrained_fast.len().min(self.importance.weights.len()) {
            let importance = self.importance.weights[i] as f64;

            // Penalize changes to important weights
            let penalty = self.lambda * importance;

            constrained_fast[i] *= (1.0 - penalty as f32).max(0.0);
            constrained_slow[i] *= (1.0 - penalty as f32).max(0.0);

            total_penalty += penalty;
        }

        let avg_penalty = total_penalty / constrained_fast.len() as f64;

        Ok(WeightUpdate {
            fast_delta: constrained_fast,
            slow_delta: constrained_slow,
            ewc_penalty: avg_penalty,
            ..update.clone()
        })
    }

    /// Update optimal weights after task completion
    pub fn update_optimal(&mut self, current_weights: &[f32]) {
        self.optimal_weights = current_weights.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

    #[test]
    fn test_ewc_creation() {
        let ewc = ElasticWeightConsolidation::new(0.4);
        assert_eq!(ewc.lambda, 0.4);
    }

    #[test]
    fn test_compute_importance() -> Result<()> {
        let mut ewc = ElasticWeightConsolidation::new(0.4);

        let task_data = vec![(
            Tensor::from_data(
                vec![1.0f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
            Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
        )];

        ewc.compute_importance(&task_data)?;

        // Importance should be non-zero after computation
        assert!(ewc.importance.weights[0] > 0.0);

        Ok(())
    }

    #[test]
    fn test_constrain_update() -> Result<()> {
        let mut ewc = ElasticWeightConsolidation::new(0.5);

        // Set some importance
        ewc.importance.weights[0] = 1.0;
        ewc.importance.weights[1] = 0.1;

        let update = WeightUpdate {
            fast_delta: vec![1.0; 100],
            slow_delta: vec![0.1; 100],
            fast_magnitude: 1.0,
            slow_magnitude: 0.1,
            ewc_penalty: 0.0,
        };

        let constrained = ewc.constrain_update(&update)?;

        // Updates to important weights should be reduced
        assert!(constrained.fast_delta[0] < update.fast_delta[0]);
        assert!(constrained.ewc_penalty > 0.0);

        Ok(())
    }
}
