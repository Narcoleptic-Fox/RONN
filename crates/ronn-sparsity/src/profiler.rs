//! Activation profiling for neuron sparsity analysis.
//!
//! Records which neurons fire (produce non-zero or above-threshold output)
//! at each FFN layer during calibration runs. The resulting activation profile
//! is used by the classifier to determine hot/cold neuron splits and by the
//! predictor trainer to generate training labels.
//!
//! PowerInfer reference: powerinfer-py/ profiling approach — profile on
//! ~1000 representative samples using ReLU activation as the sparsity signal.

use crate::error::{Result, SparsityError};
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Activation function type, determines how we detect "active" neurons.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    /// ReLU: natural sparsity — neuron is active if output > 0.
    ReLU,
    /// SiLU/Swish: threshold-based — neuron is active if |output| > threshold.
    SiLU,
    /// GELU: threshold-based — neuron is active if |output| > threshold.
    GELU,
}

/// Configuration for the activation profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Activation function used by the model's FFN layers.
    pub activation_type: ActivationType,
    /// Threshold for non-ReLU activations: neuron is "active" if |output| > threshold.
    pub activation_threshold: f32,
    /// Number of calibration samples to process.
    pub num_samples: usize,
    /// Model identifier hash (for profile validation).
    pub model_hash: String,
    /// Calibration dataset identifier hash.
    pub dataset_hash: String,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            activation_type: ActivationType::ReLU,
            activation_threshold: 0.1,
            num_samples: 1000,
            model_hash: String::new(),
            dataset_hash: String::new(),
        }
    }
}

/// Per-neuron activation statistics within a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronStats {
    /// Number of times this neuron was activated.
    pub activation_count: u64,
    /// Total number of samples observed.
    pub total_samples: u64,
    /// Sum of activation magnitudes (for average activation strength).
    pub magnitude_sum: f64,
}

impl NeuronStats {
    fn new() -> Self {
        Self {
            activation_count: 0,
            total_samples: 0,
            magnitude_sum: 0.0,
        }
    }

    /// Activation frequency: activation_count / total_samples.
    pub fn frequency(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.activation_count as f64 / self.total_samples as f64
    }

    /// Average activation magnitude when active.
    pub fn avg_magnitude(&self) -> f64 {
        if self.activation_count == 0 {
            return 0.0;
        }
        self.magnitude_sum / self.activation_count as f64
    }
}

/// Activation profile for a single FFN layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    /// Layer index.
    pub layer_id: usize,
    /// Per-neuron statistics, indexed by neuron id.
    pub neuron_stats: Vec<NeuronStats>,
    /// Total number of neurons in this layer.
    pub num_neurons: usize,
    /// Number of calibration samples profiled.
    pub samples_profiled: u64,
}

impl LayerProfile {
    /// Create a new layer profile for a layer with `num_neurons` neurons.
    pub fn new(layer_id: usize, num_neurons: usize) -> Self {
        Self {
            layer_id,
            neuron_stats: (0..num_neurons).map(|_| NeuronStats::new()).collect(),
            num_neurons,
            samples_profiled: 0,
        }
    }

    /// Record activation data from a single forward pass through this layer.
    ///
    /// `activations` should be a 1D or 2D tensor where the last dimension
    /// corresponds to neurons. For batch inputs, activations has shape
    /// [batch_size, num_neurons].
    pub fn record_activations(
        &mut self,
        activations: &Tensor,
        activation_type: ActivationType,
        threshold: f32,
    ) -> Result<()> {
        let data = activations
            .to_vec()
            .map_err(|e| SparsityError::Profiling(format!("Failed to extract activations: {}", e)))?;
        let shape = activations.shape();

        // Determine batch size and neuron count from shape
        let (batch_size, neuron_count) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                // Flatten all but last dimension as batch
                let neuron_count = *shape.last().unwrap();
                let batch_size = data.len() / neuron_count;
                (batch_size, neuron_count)
            }
        };

        if neuron_count != self.num_neurons {
            return Err(SparsityError::Profiling(format!(
                "Layer {} expected {} neurons, got {}",
                self.layer_id, self.num_neurons, neuron_count
            )));
        }

        for batch_idx in 0..batch_size {
            let offset = batch_idx * neuron_count;
            for neuron_idx in 0..neuron_count {
                let value = data[offset + neuron_idx];
                let is_active = match activation_type {
                    ActivationType::ReLU => value > 0.0,
                    ActivationType::SiLU | ActivationType::GELU => value.abs() > threshold,
                };

                let stats = &mut self.neuron_stats[neuron_idx];
                stats.total_samples += 1;
                if is_active {
                    stats.activation_count += 1;
                    stats.magnitude_sum += value.abs() as f64;
                }
            }
            self.samples_profiled += 1;
        }

        Ok(())
    }

    /// Get activation frequencies for all neurons as a vector.
    pub fn activation_frequencies(&self) -> Vec<f64> {
        self.neuron_stats.iter().map(|s| s.frequency()).collect()
    }

    /// Average sparsity across the layer (fraction of neurons inactive).
    pub fn avg_sparsity(&self) -> f64 {
        let freqs = self.activation_frequencies();
        if freqs.is_empty() {
            return 0.0;
        }
        let avg_active_frac = freqs.iter().sum::<f64>() / freqs.len() as f64;
        1.0 - avg_active_frac
    }
}

/// Complete activation profile for a model (all FFN layers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationProfile {
    /// Model identifier hash.
    pub model_hash: String,
    /// Calibration dataset identifier hash.
    pub dataset_hash: String,
    /// Profiler configuration used.
    pub config: ProfilerConfig,
    /// Per-layer profiles, keyed by layer index.
    pub layer_profiles: HashMap<usize, LayerProfile>,
    /// Total calibration samples processed.
    pub total_samples: u64,
}

impl ActivationProfile {
    /// Serialize the profile to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| SparsityError::Profiling(format!("Failed to serialize profile: {}", e)))
    }

    /// Deserialize a profile from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| SparsityError::Profiling(format!("Failed to deserialize profile: {}", e)))
    }
}

/// Activation profiler: runs calibration data through a model and records
/// neuron activation patterns at each FFN layer.
pub struct ActivationProfiler {
    config: ProfilerConfig,
    layer_profiles: HashMap<usize, LayerProfile>,
    total_samples: u64,
}

impl ActivationProfiler {
    /// Create a new profiler with the given configuration.
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            layer_profiles: HashMap::new(),
            total_samples: 0,
        }
    }

    /// Register an FFN layer for profiling.
    pub fn register_layer(&mut self, layer_id: usize, num_neurons: usize) {
        self.layer_profiles
            .insert(layer_id, LayerProfile::new(layer_id, num_neurons));
        debug!(
            "Registered layer {} with {} neurons for profiling",
            layer_id, num_neurons
        );
    }

    /// Record activations for a specific layer from one forward pass.
    pub fn record(&mut self, layer_id: usize, activations: &Tensor) -> Result<()> {
        let profile = self.layer_profiles.get_mut(&layer_id).ok_or_else(|| {
            SparsityError::Profiling(format!("Layer {} not registered for profiling", layer_id))
        })?;

        profile.record_activations(
            activations,
            self.config.activation_type,
            self.config.activation_threshold,
        )?;

        Ok(())
    }

    /// Mark the completion of one full calibration sample (all layers recorded).
    pub fn complete_sample(&mut self) {
        self.total_samples += 1;
        if self.total_samples % 100 == 0 {
            debug!("Profiled {} calibration samples", self.total_samples);
        }
    }

    /// Finalize the profiler and produce an ActivationProfile.
    pub fn finalize(self) -> ActivationProfile {
        info!(
            "Profiling complete: {} samples across {} layers",
            self.total_samples,
            self.layer_profiles.len()
        );
        for (layer_id, profile) in &self.layer_profiles {
            info!(
                "  Layer {}: {:.1}% average sparsity",
                layer_id,
                profile.avg_sparsity() * 100.0
            );
        }

        ActivationProfile {
            model_hash: self.config.model_hash.clone(),
            dataset_hash: self.config.dataset_hash.clone(),
            config: self.config,
            layer_profiles: self.layer_profiles,
            total_samples: self.total_samples,
        }
    }

    /// Get a reference to the current layer profiles (for inspection).
    pub fn layer_profiles(&self) -> &HashMap<usize, LayerProfile> {
        &self.layer_profiles
    }

    /// Get the number of samples profiled so far.
    pub fn samples_profiled(&self) -> u64 {
        self.total_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_stats() {
        let mut stats = NeuronStats::new();
        assert_eq!(stats.frequency(), 0.0);

        stats.total_samples = 100;
        stats.activation_count = 80;
        stats.magnitude_sum = 160.0;

        assert!((stats.frequency() - 0.8).abs() < f64::EPSILON);
        assert!((stats.avg_magnitude() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_layer_profile_relu() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut profile = LayerProfile::new(0, 4);

        // Simulate ReLU activations: neurons 0,1 always active, 2,3 sometimes
        let data = vec![1.0, 0.5, 0.0, -0.1]; // neuron 0,1 active; 2,3 inactive
        let tensor =
            Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        profile.record_activations(&tensor, ActivationType::ReLU, 0.0)?;

        let data2 = vec![0.8, 0.3, 0.2, 0.0];
        let tensor2 =
            Tensor::from_data(data2, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        profile.record_activations(&tensor2, ActivationType::ReLU, 0.0)?;

        let freqs = profile.activation_frequencies();
        assert!((freqs[0] - 1.0).abs() < f64::EPSILON); // always active
        assert!((freqs[1] - 1.0).abs() < f64::EPSILON); // always active
        assert!((freqs[2] - 0.5).abs() < f64::EPSILON); // active half the time
        assert!((freqs[3] - 0.0).abs() < f64::EPSILON); // never active

        Ok(())
    }

    #[test]
    fn test_layer_profile_silu() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut profile = LayerProfile::new(0, 3);

        // SiLU values: active if |value| > threshold (0.1)
        let data = vec![0.5, 0.05, -0.3];
        let tensor =
            Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;
        profile.record_activations(&tensor, ActivationType::SiLU, 0.1)?;

        let freqs = profile.activation_frequencies();
        assert!((freqs[0] - 1.0).abs() < f64::EPSILON); // |0.5| > 0.1
        assert!((freqs[1] - 0.0).abs() < f64::EPSILON); // |0.05| < 0.1
        assert!((freqs[2] - 1.0).abs() < f64::EPSILON); // |-0.3| > 0.1

        Ok(())
    }

    #[test]
    fn test_profiler_workflow() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = ProfilerConfig {
            activation_type: ActivationType::ReLU,
            activation_threshold: 0.0,
            num_samples: 10,
            model_hash: "test_model".into(),
            dataset_hash: "test_data".into(),
        };
        let mut profiler = ActivationProfiler::new(config);
        profiler.register_layer(0, 8);
        profiler.register_layer(1, 16);

        // Simulate calibration samples with explicit f32 data
        for i in 0..5 {
            let data0: Vec<f32> = (0..8).map(|j| ((i * 8 + j) as f32) * 0.1 - 0.2).collect();
            let act0 = Tensor::from_data(data0, vec![1, 8], DataType::F32, TensorLayout::RowMajor)?;
            profiler.record(0, &act0)?;

            let data1: Vec<f32> = (0..16).map(|j| ((i * 16 + j) as f32) * 0.05 - 0.1).collect();
            let act1 = Tensor::from_data(data1, vec![1, 16], DataType::F32, TensorLayout::RowMajor)?;
            profiler.record(1, &act1)?;

            profiler.complete_sample();
        }

        let profile = profiler.finalize();
        assert_eq!(profile.total_samples, 5);
        assert_eq!(profile.layer_profiles.len(), 2);
        assert!(profile.layer_profiles.contains_key(&0));
        assert!(profile.layer_profiles.contains_key(&1));

        // Test serialization round-trip
        let bytes = profile.to_bytes()?;
        let restored = ActivationProfile::from_bytes(&bytes)?;
        assert_eq!(restored.total_samples, 5);

        Ok(())
    }

    #[test]
    fn test_batch_activations() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut profile = LayerProfile::new(0, 4);

        // Batch of 3 samples, 4 neurons each
        let data = vec![
            1.0, 0.0, 1.0, 0.0, // sample 0: neurons 0,2 active
            0.0, 1.0, 0.0, 1.0, // sample 1: neurons 1,3 active
            1.0, 1.0, 0.0, 0.0, // sample 2: neurons 0,1 active
        ];
        let tensor =
            Tensor::from_data(data, vec![3, 4], DataType::F32, TensorLayout::RowMajor)?;
        profile.record_activations(&tensor, ActivationType::ReLU, 0.0)?;

        assert_eq!(profile.samples_profiled, 3);
        let freqs = profile.activation_frequencies();
        // neuron 0: 2/3, neuron 1: 2/3, neuron 2: 1/3, neuron 3: 1/3
        assert!((freqs[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((freqs[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((freqs[2] - 1.0 / 3.0).abs() < 1e-10);
        assert!((freqs[3] - 1.0 / 3.0).abs() < 1e-10);

        Ok(())
    }
}
