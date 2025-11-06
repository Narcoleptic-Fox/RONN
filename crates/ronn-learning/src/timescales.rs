//! Multi-Timescale Learning - Fast and slow weight adaptation

use crate::Result;
use ronn_core::tensor::Tensor;

/// Configuration for multi-timescale learning
#[derive(Debug, Clone)]
pub struct TimescaleConfig {
    pub fast_lr: f64,  // Fast weight learning rate
    pub slow_lr: f64,  // Slow weight learning rate
    pub consolidation_rate: f64, // Rate of consolidation from fast to slow
}

impl Default for TimescaleConfig {
    fn default() -> Self {
        Self {
            fast_lr: 0.01,
            slow_lr: 0.0001,
            consolidation_rate: 0.1,
        }
    }
}

/// A weight update with fast and slow components
#[derive(Debug, Clone)]
pub struct WeightUpdate {
    pub fast_delta: Vec<f32>,
    pub slow_delta: Vec<f32>,
    pub fast_magnitude: f64,
    pub slow_magnitude: f64,
    pub ewc_penalty: f64,
}

impl WeightUpdate {
    /// Scale the update by a factor
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            fast_delta: self.fast_delta.iter().map(|x| x * factor as f32).collect(),
            slow_delta: self.slow_delta.iter().map(|x| x * factor as f32).collect(),
            fast_magnitude: self.fast_magnitude * factor,
            slow_magnitude: self.slow_magnitude * factor,
            ewc_penalty: self.ewc_penalty,
        }
    }
}

/// Applied weight update result
#[derive(Debug, Clone)]
pub struct AppliedUpdate {
    pub fast_magnitude: f64,
    pub slow_magnitude: f64,
}

/// Multi-timescale learner with fast and slow weights
pub struct MultiTimescaleLearner {
    config: TimescaleConfig,
    fast_weights: Vec<f32>,
    slow_weights: Vec<f32>,
    weight_size: usize,
}

impl MultiTimescaleLearner {
    /// Create new multi-timescale learner
    pub fn new(config: TimescaleConfig) -> Self {
        // For MVP: Fixed size weights (in production, would be model-dependent)
        let weight_size = 100;

        Self {
            config,
            fast_weights: vec![0.0; weight_size],
            slow_weights: vec![0.0; weight_size],
            weight_size,
        }
    }

    /// Compute weight update from input and target
    pub fn compute_update(&self, _input: &Tensor, _target: &Tensor) -> Result<WeightUpdate> {
        // For MVP: Simulate gradient computation
        // In production: Compute actual gradients via backprop

        let fast_delta: Vec<f32> = (0..self.weight_size)
            .map(|i| (i as f32 * 0.01).sin() * self.config.fast_lr as f32)
            .collect();

        let slow_delta: Vec<f32> = fast_delta
            .iter()
            .map(|x| x * self.config.slow_lr as f32 / self.config.fast_lr as f32)
            .collect();

        let fast_magnitude = fast_delta.iter().map(|x| x.abs() as f64).sum::<f64>() / fast_delta.len() as f64;
        let slow_magnitude = slow_delta.iter().map(|x| x.abs() as f64).sum::<f64>() / slow_delta.len() as f64;

        Ok(WeightUpdate {
            fast_delta,
            slow_delta,
            fast_magnitude,
            slow_magnitude,
            ewc_penalty: 0.0,
        })
    }

    /// Apply weight update
    pub fn apply_update(&mut self, update: &WeightUpdate) -> Result<AppliedUpdate> {
        // Update fast weights
        for (i, delta) in update.fast_delta.iter().enumerate() {
            if i < self.fast_weights.len() {
                self.fast_weights[i] += delta;
            }
        }

        // Update slow weights
        for (i, delta) in update.slow_delta.iter().enumerate() {
            if i < self.slow_weights.len() {
                self.slow_weights[i] += delta;
            }
        }

        Ok(AppliedUpdate {
            fast_magnitude: update.fast_magnitude,
            slow_magnitude: update.slow_magnitude,
        })
    }

    /// Consolidate fast weights into slow weights
    pub fn consolidate(&mut self) -> Result<()> {
        for i in 0..self.weight_size {
            let transfer = self.fast_weights[i] * self.config.consolidation_rate as f32;
            self.slow_weights[i] += transfer;
            self.fast_weights[i] -= transfer;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

    #[test]
    fn test_learner_creation() {
        let learner = MultiTimescaleLearner::new(TimescaleConfig::default());
        assert_eq!(learner.weight_size, 100);
    }

    #[test]
    fn test_compute_update() -> Result<()> {
        let learner = MultiTimescaleLearner::new(TimescaleConfig::default());

        let input = Tensor::from_data(vec![1.0f32, 2.0], vec![1, 2], DataType::F32, TensorLayout::RowMajor)?;
        let target = Tensor::from_data(vec![0.5f32], vec![1, 1], DataType::F32, TensorLayout::RowMajor)?;

        let update = learner.compute_update(&input, &target)?;

        assert!(update.fast_magnitude > 0.0);
        assert!(update.slow_magnitude > 0.0);
        assert!(update.fast_magnitude > update.slow_magnitude); // Fast should be larger

        Ok(())
    }

    #[test]
    fn test_apply_update() -> Result<()> {
        let mut learner = MultiTimescaleLearner::new(TimescaleConfig::default());

        let input = Tensor::from_data(vec![1.0f32], vec![1, 1], DataType::F32, TensorLayout::RowMajor)?;
        let target = Tensor::from_data(vec![0.5f32], vec![1, 1], DataType::F32, TensorLayout::RowMajor)?;

        let update = learner.compute_update(&input, &target)?;
        let applied = learner.apply_update(&update)?;

        assert!(applied.fast_magnitude > 0.0);

        Ok(())
    }

    #[test]
    fn test_consolidation() -> Result<()> {
        let mut learner = MultiTimescaleLearner::new(TimescaleConfig::default());

        // Set some fast weights
        learner.fast_weights[0] = 1.0;
        learner.fast_weights[1] = 2.0;

        let initial_fast_sum: f32 = learner.fast_weights.iter().sum();
        let initial_slow_sum: f32 = learner.slow_weights.iter().sum();

        learner.consolidate()?;

        let final_fast_sum: f32 = learner.fast_weights.iter().sum();
        let final_slow_sum: f32 = learner.slow_weights.iter().sum();

        // Fast weights should decrease, slow weights should increase
        assert!(final_fast_sum < initial_fast_sum);
        assert!(final_slow_sum > initial_slow_sum);

        Ok(())
    }
}
