//! Lightweight neural network for predicting active neurons.
//!
//! Each FFN layer gets a small MLP predictor that takes the hidden state
//! entering the FFN as input and outputs a binary mask predicting which
//! neurons will fire. This allows skipping inactive neurons before computing
//! the full FFN, providing the core speedup of the PowerInfer approach.
//!
//! Architecture per layer:
//!   Input: hidden state (hidden_dim)
//!   -> Linear(hidden_dim, predictor_dim) -> ReLU
//!   -> Linear(predictor_dim, ffn_dim) -> Sigmoid
//!   Output: activation probability per neuron (ffn_dim)
//!
//! Training uses binary cross-entropy on recorded activation patterns.

use crate::error::{Result, SparsityError};
use crate::profiler::{ActivationProfile, ActivationType};
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Configuration for the activation predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    /// Dimension of the predictor hidden layer.
    /// Typically 128-256, much smaller than ffn_dim.
    pub predictor_dim: usize,
    /// Learning rate for training.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Mini-batch size for training.
    pub batch_size: usize,
    /// Threshold for converting sigmoid output to binary mask.
    pub prediction_threshold: f32,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            predictor_dim: 128,
            learning_rate: 0.001,
            num_epochs: 10,
            batch_size: 32,
            prediction_threshold: 0.5,
        }
    }
}

/// Weights for a single-layer predictor MLP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorWeights {
    /// Layer index this predictor serves.
    pub layer_id: usize,
    /// Input dimension (hidden_dim of the model).
    pub input_dim: usize,
    /// Hidden dimension of the predictor.
    pub hidden_dim: usize,
    /// Output dimension (ffn_dim of the layer).
    pub output_dim: usize,
    /// First linear layer weights: [hidden_dim, input_dim] (row-major).
    pub w1: Vec<f32>,
    /// First linear layer bias: [hidden_dim].
    pub b1: Vec<f32>,
    /// Second linear layer weights: [output_dim, hidden_dim] (row-major).
    pub w2: Vec<f32>,
    /// Second linear layer bias: [output_dim].
    pub b2: Vec<f32>,
}

impl PredictorWeights {
    /// Create randomly initialized weights (Xavier/Glorot uniform).
    pub fn random(layer_id: usize, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let scale1 = (6.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
        let scale2 = (6.0 / (hidden_dim + output_dim) as f64).sqrt() as f32;

        let w1: Vec<f32> = (0..hidden_dim * input_dim)
            .map(|_| (fastrand::f32() * 2.0 - 1.0) * scale1)
            .collect();
        let b1 = vec![0.0; hidden_dim];

        let w2: Vec<f32> = (0..output_dim * hidden_dim)
            .map(|_| (fastrand::f32() * 2.0 - 1.0) * scale2)
            .collect();
        let b2 = vec![0.0; output_dim];

        Self {
            layer_id,
            input_dim,
            hidden_dim,
            output_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }
}

/// A trained activation predictor for a single FFN layer.
pub struct LayerPredictor {
    weights: PredictorWeights,
    config: PredictorConfig,
}

impl LayerPredictor {
    /// Create a predictor from pre-trained weights.
    pub fn from_weights(weights: PredictorWeights, config: PredictorConfig) -> Self {
        Self { weights, config }
    }

    /// Predict active neurons given a hidden state input.
    ///
    /// Returns a boolean mask where `true` means the neuron is predicted active.
    pub fn predict(&self, hidden_state: &Tensor) -> Result<Vec<bool>> {
        let input_data = hidden_state.to_vec().map_err(|e| {
            SparsityError::Prediction(format!("Failed to extract hidden state: {}", e))
        })?;

        let input_dim = self.weights.input_dim;
        let hidden_dim = self.weights.hidden_dim;
        let output_dim = self.weights.output_dim;

        if input_data.len() != input_dim {
            return Err(SparsityError::Prediction(format!(
                "Expected input dim {}, got {}",
                input_dim,
                input_data.len()
            )));
        }

        // Layer 1: Linear + ReLU
        let mut hidden = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            let mut sum = self.weights.b1[i];
            let row_offset = i * input_dim;
            for j in 0..input_dim {
                sum += self.weights.w1[row_offset + j] * input_data[j];
            }
            // ReLU
            hidden[i] = sum.max(0.0);
        }

        // Layer 2: Linear + Sigmoid
        let mut output = vec![0.0f32; output_dim];
        for i in 0..output_dim {
            let mut sum = self.weights.b2[i];
            let row_offset = i * hidden_dim;
            for j in 0..hidden_dim {
                sum += self.weights.w2[row_offset + j] * hidden[j];
            }
            // Sigmoid
            output[i] = 1.0 / (1.0 + (-sum).exp());
        }

        // Threshold to binary mask
        let mask: Vec<bool> = output
            .iter()
            .map(|&p| p > self.config.prediction_threshold)
            .collect();

        Ok(mask)
    }

    /// Predict and return activation probabilities (pre-threshold).
    pub fn predict_probabilities(&self, hidden_state: &Tensor) -> Result<Vec<f32>> {
        let input_data = hidden_state.to_vec().map_err(|e| {
            SparsityError::Prediction(format!("Failed to extract hidden state: {}", e))
        })?;

        let input_dim = self.weights.input_dim;
        let hidden_dim = self.weights.hidden_dim;
        let output_dim = self.weights.output_dim;

        if input_data.len() != input_dim {
            return Err(SparsityError::Prediction(format!(
                "Expected input dim {}, got {}",
                input_dim,
                input_data.len()
            )));
        }

        // Layer 1: Linear + ReLU
        let mut hidden = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            let mut sum = self.weights.b1[i];
            let row_offset = i * input_dim;
            for j in 0..input_dim {
                sum += self.weights.w1[row_offset + j] * input_data[j];
            }
            hidden[i] = sum.max(0.0);
        }

        // Layer 2: Linear + Sigmoid
        let mut output = vec![0.0f32; output_dim];
        for i in 0..output_dim {
            let mut sum = self.weights.b2[i];
            let row_offset = i * hidden_dim;
            for j in 0..hidden_dim {
                sum += self.weights.w2[row_offset + j] * hidden[j];
            }
            output[i] = 1.0 / (1.0 + (-sum).exp());
        }

        Ok(output)
    }

    /// Get a reference to the weights.
    pub fn weights(&self) -> &PredictorWeights {
        &self.weights
    }
}

/// Training data for a predictor: pairs of (hidden_state, activation_labels).
pub struct PredictorTrainingData {
    /// Hidden state inputs, shape: [num_samples, input_dim].
    pub inputs: Vec<Vec<f32>>,
    /// Binary activation labels, shape: [num_samples, output_dim].
    pub labels: Vec<Vec<f32>>,
}

/// Trains activation predictors from profiling data.
pub struct PredictorTrainer {
    config: PredictorConfig,
}

impl PredictorTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: PredictorConfig) -> Self {
        Self { config }
    }

    /// Train a predictor for a single layer given training data.
    ///
    /// Uses simple SGD with binary cross-entropy loss.
    pub fn train_layer(
        &self,
        layer_id: usize,
        input_dim: usize,
        output_dim: usize,
        training_data: &PredictorTrainingData,
    ) -> Result<PredictorWeights> {
        let hidden_dim = self.config.predictor_dim;
        let mut weights = PredictorWeights::random(layer_id, input_dim, hidden_dim, output_dim);
        let lr = self.config.learning_rate as f32;
        let num_samples = training_data.inputs.len();

        if num_samples == 0 {
            return Err(SparsityError::Training("No training data provided".into()));
        }

        info!(
            "Training predictor for layer {}: input_dim={}, hidden_dim={}, output_dim={}, samples={}",
            layer_id, input_dim, hidden_dim, output_dim, num_samples
        );

        for epoch in 0..self.config.num_epochs {
            let mut total_loss = 0.0f64;

            // Simple SGD over all samples
            for sample_idx in 0..num_samples {
                let input = &training_data.inputs[sample_idx];
                let label = &training_data.labels[sample_idx];

                // Forward pass - Layer 1: Linear + ReLU
                let mut hidden = vec![0.0f32; hidden_dim];
                for i in 0..hidden_dim {
                    let mut sum = weights.b1[i];
                    let row_offset = i * input_dim;
                    for j in 0..input_dim {
                        sum += weights.w1[row_offset + j] * input[j];
                    }
                    hidden[i] = sum.max(0.0); // ReLU
                }

                // Forward pass - Layer 2: Linear + Sigmoid
                let mut output = vec![0.0f32; output_dim];
                let mut pre_sigmoid = vec![0.0f32; output_dim];
                for i in 0..output_dim {
                    let mut sum = weights.b2[i];
                    let row_offset = i * hidden_dim;
                    for j in 0..hidden_dim {
                        sum += weights.w2[row_offset + j] * hidden[j];
                    }
                    pre_sigmoid[i] = sum;
                    output[i] = 1.0 / (1.0 + (-sum).exp());
                }

                // Binary cross-entropy loss
                for i in 0..output_dim {
                    let p = output[i].clamp(1e-7, 1.0 - 1e-7);
                    let y = label[i];
                    total_loss += -(y as f64 * (p as f64).ln()
                        + (1.0 - y as f64) * (1.0 - p as f64).ln());
                }

                // Backward pass - dL/d_output (sigmoid + BCE gradient)
                let mut d_output = vec![0.0f32; output_dim];
                for i in 0..output_dim {
                    d_output[i] = output[i] - label[i]; // sigmoid + BCE simplifies to (pred - target)
                }

                // Gradients for Layer 2
                for i in 0..output_dim {
                    let row_offset = i * hidden_dim;
                    for j in 0..hidden_dim {
                        weights.w2[row_offset + j] -= lr * d_output[i] * hidden[j];
                    }
                    weights.b2[i] -= lr * d_output[i];
                }

                // Backward through Layer 2 to hidden
                let mut d_hidden = vec![0.0f32; hidden_dim];
                for j in 0..hidden_dim {
                    let mut sum = 0.0f32;
                    for i in 0..output_dim {
                        sum += d_output[i] * weights.w2[i * hidden_dim + j];
                    }
                    // ReLU gradient
                    d_hidden[j] = if hidden[j] > 0.0 { sum } else { 0.0 };
                }

                // Gradients for Layer 1
                for i in 0..hidden_dim {
                    let row_offset = i * input_dim;
                    for j in 0..input_dim {
                        weights.w1[row_offset + j] -= lr * d_hidden[i] * input[j];
                    }
                    weights.b1[i] -= lr * d_hidden[i];
                }
            }

            let avg_loss = total_loss / (num_samples * output_dim) as f64;
            if epoch % 2 == 0 || epoch == self.config.num_epochs - 1 {
                debug!(
                    "Layer {} epoch {}/{}: avg BCE loss = {:.6}",
                    layer_id,
                    epoch + 1,
                    self.config.num_epochs,
                    avg_loss
                );
            }
        }

        info!("Predictor training complete for layer {}", layer_id);
        Ok(weights)
    }
}

/// Collection of trained predictors for all FFN layers in a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPredictors {
    /// Per-layer predictor weights.
    pub layer_weights: HashMap<usize, PredictorWeights>,
    /// Configuration used for training.
    pub config: PredictorConfig,
    /// Model hash for validation.
    pub model_hash: String,
}

impl ModelPredictors {
    /// Serialize to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| {
            SparsityError::Prediction(format!("Failed to serialize predictors: {}", e))
        })
    }

    /// Deserialize from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| {
            SparsityError::Prediction(format!("Failed to deserialize predictors: {}", e))
        })
    }

    /// Get a predictor for a specific layer.
    pub fn get_layer_predictor(&self, layer_id: usize) -> Option<LayerPredictor> {
        self.layer_weights.get(&layer_id).map(|w| {
            LayerPredictor::from_weights(w.clone(), self.config.clone())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_forward_pass() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let weights = PredictorWeights::random(0, 16, 8, 32);
        let config = PredictorConfig::default();
        let predictor = LayerPredictor::from_weights(weights, config);

        let input_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let input = Tensor::from_data(input_data, vec![16], DataType::F32, TensorLayout::RowMajor)?;
        let mask = predictor.predict(&input)?;
        assert_eq!(mask.len(), 32);

        let probs = predictor.predict_probabilities(&input)?;
        assert_eq!(probs.len(), 32);
        for &p in &probs {
            assert!(p >= 0.0 && p <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_predictor_training() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = PredictorConfig {
            predictor_dim: 4,
            learning_rate: 0.01,
            num_epochs: 5,
            batch_size: 2,
            prediction_threshold: 0.5,
        };

        let trainer = PredictorTrainer::new(config.clone());

        // Simple training data: all-ones input -> first half active
        let num_samples = 10;
        let input_dim = 8;
        let output_dim = 4;

        let training_data = PredictorTrainingData {
            inputs: (0..num_samples)
                .map(|_| vec![1.0f32; input_dim])
                .collect(),
            labels: (0..num_samples)
                .map(|_| vec![1.0, 1.0, 0.0, 0.0])
                .collect(),
        };

        let weights = trainer.train_layer(0, input_dim, output_dim, &training_data)?;
        assert_eq!(weights.input_dim, input_dim);
        assert_eq!(weights.output_dim, output_dim);
        assert_eq!(weights.hidden_dim, 4);

        Ok(())
    }

    #[test]
    fn test_model_predictors_serialization() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let mut layer_weights = HashMap::new();
        layer_weights.insert(0, PredictorWeights::random(0, 16, 8, 32));
        layer_weights.insert(1, PredictorWeights::random(1, 16, 8, 64));

        let predictors = ModelPredictors {
            layer_weights,
            config: PredictorConfig::default(),
            model_hash: "test_model".into(),
        };

        let bytes = predictors.to_bytes()?;
        let restored = ModelPredictors::from_bytes(&bytes)?;
        assert_eq!(restored.layer_weights.len(), 2);
        assert!(restored.get_layer_predictor(0).is_some());
        assert!(restored.get_layer_predictor(1).is_some());
        assert!(restored.get_layer_predictor(2).is_none());

        Ok(())
    }
}
