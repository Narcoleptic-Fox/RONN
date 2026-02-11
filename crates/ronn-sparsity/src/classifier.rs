//! Hot/cold neuron classification from activation profiles.
//!
//! Given an activation profile, classifies each neuron in each FFN layer as
//! either "hot" (high activation frequency, always computed on GPU) or "cold"
//! (low activation frequency, computed on-demand or skipped).
//!
//! This maps directly to RONN's HRM System 1/System 2 paradigm:
//! - Hot neurons = System 1 (fast, always ready, GPU-resident)
//! - Cold neurons = System 2 (on-demand, CPU or skipped)

use crate::error::{Result, SparsityError};
use crate::profiler::ActivationProfile;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Classification of a single neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronClass {
    /// Hot neuron: always computed, GPU-resident.
    Hot,
    /// Cold neuron: computed on-demand based on predictor output.
    Cold,
}

/// Per-layer classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerClassification {
    /// Layer index.
    pub layer_id: usize,
    /// Classification for each neuron (indexed by neuron id).
    pub neuron_classes: Vec<NeuronClass>,
    /// Indices of hot neurons.
    pub hot_indices: Vec<usize>,
    /// Indices of cold neurons.
    pub cold_indices: Vec<usize>,
    /// Total neurons.
    pub total_neurons: usize,
    /// Activation threshold used for classification.
    pub hot_threshold: f64,
}

impl LayerClassification {
    /// Fraction of neurons classified as hot.
    pub fn hot_ratio(&self) -> f64 {
        if self.total_neurons == 0 {
            return 0.0;
        }
        self.hot_indices.len() as f64 / self.total_neurons as f64
    }

    /// Fraction of neurons classified as cold.
    pub fn cold_ratio(&self) -> f64 {
        1.0 - self.hot_ratio()
    }
}

/// Configuration for the neuron classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Neurons with activation frequency above this threshold are classified as hot.
    /// Default: 0.8 (fires 80%+ of the time).
    pub hot_threshold: f64,
    /// Optional VRAM budget in bytes. If set, the classifier will adjust the
    /// hot/cold split to fit within the budget.
    pub vram_budget_bytes: Option<usize>,
    /// Estimated bytes per neuron weight (for memory budget calculations).
    /// Default: 4 (float32) or 2 (float16).
    pub bytes_per_weight: usize,
    /// Minimum hot ratio per layer. Even with tight memory budgets, we keep
    /// at least this fraction of neurons hot for baseline performance.
    pub min_hot_ratio: f64,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 0.8,
            vram_budget_bytes: None,
            bytes_per_weight: 4,
            min_hot_ratio: 0.05,
        }
    }
}

/// Complete model classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelClassification {
    /// Per-layer classifications.
    pub layer_classifications: HashMap<usize, LayerClassification>,
    /// Configuration used.
    pub config: ClassifierConfig,
    /// Model hash from the profile.
    pub model_hash: String,
    /// Total hot neurons across all layers.
    pub total_hot: usize,
    /// Total cold neurons across all layers.
    pub total_cold: usize,
    /// Estimated GPU memory needed for hot neurons (bytes).
    pub estimated_gpu_memory: usize,
}

impl ModelClassification {
    /// Serialize to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| {
            SparsityError::Classification(format!("Failed to serialize classification: {}", e))
        })
    }

    /// Deserialize from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| {
            SparsityError::Classification(format!("Failed to deserialize classification: {}", e))
        })
    }
}

/// Neuron classifier: takes an activation profile and produces hot/cold classification.
pub struct NeuronClassifier {
    config: ClassifierConfig,
}

impl NeuronClassifier {
    /// Create a classifier with the given configuration.
    pub fn new(config: ClassifierConfig) -> Self {
        Self { config }
    }

    /// Create a classifier with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ClassifierConfig::default())
    }

    /// Classify neurons from an activation profile.
    pub fn classify(&self, profile: &ActivationProfile) -> Result<ModelClassification> {
        let mut layer_classifications = HashMap::new();
        let mut total_hot = 0usize;
        let mut total_cold = 0usize;

        for (&layer_id, layer_profile) in &profile.layer_profiles {
            let classification = self.classify_layer(layer_profile)?;
            total_hot += classification.hot_indices.len();
            total_cold += classification.cold_indices.len();
            layer_classifications.insert(layer_id, classification);
        }

        let estimated_gpu_memory = total_hot * self.config.bytes_per_weight;

        let mut result = ModelClassification {
            layer_classifications,
            config: self.config.clone(),
            model_hash: profile.model_hash.clone(),
            total_hot,
            total_cold,
            estimated_gpu_memory,
        };

        // If we have a VRAM budget, adjust the classification
        if let Some(budget) = self.config.vram_budget_bytes {
            if result.estimated_gpu_memory > budget {
                info!(
                    "GPU memory estimate ({} bytes) exceeds budget ({} bytes), adjusting split",
                    result.estimated_gpu_memory, budget
                );
                self.adjust_for_budget(&mut result, profile, budget)?;
            }
        }

        info!(
            "Classification complete: {} hot, {} cold, estimated GPU memory: {} bytes",
            result.total_hot, result.total_cold, result.estimated_gpu_memory
        );

        Ok(result)
    }

    /// Classify a single layer's neurons.
    fn classify_layer(
        &self,
        layer_profile: &crate::profiler::LayerProfile,
    ) -> Result<LayerClassification> {
        let frequencies = layer_profile.activation_frequencies();
        let total_neurons = layer_profile.num_neurons;

        let mut neuron_classes = Vec::with_capacity(total_neurons);
        let mut hot_indices = Vec::new();
        let mut cold_indices = Vec::new();

        for (idx, &freq) in frequencies.iter().enumerate() {
            if freq >= self.config.hot_threshold {
                neuron_classes.push(NeuronClass::Hot);
                hot_indices.push(idx);
            } else {
                neuron_classes.push(NeuronClass::Cold);
                cold_indices.push(idx);
            }
        }

        // Enforce minimum hot ratio
        let min_hot_count = (total_neurons as f64 * self.config.min_hot_ratio).ceil() as usize;
        if hot_indices.len() < min_hot_count && !cold_indices.is_empty() {
            // Promote the most frequently active cold neurons to hot
            let mut cold_with_freq: Vec<_> = cold_indices
                .iter()
                .map(|&idx| (idx, frequencies[idx]))
                .collect();
            cold_with_freq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let needed = min_hot_count.saturating_sub(hot_indices.len());
            for i in 0..needed.min(cold_with_freq.len()) {
                let idx = cold_with_freq[i].0;
                neuron_classes[idx] = NeuronClass::Hot;
            }

            // Rebuild index lists
            hot_indices.clear();
            cold_indices.clear();
            for (idx, &class) in neuron_classes.iter().enumerate() {
                match class {
                    NeuronClass::Hot => hot_indices.push(idx),
                    NeuronClass::Cold => cold_indices.push(idx),
                }
            }
        }

        debug!(
            "Layer {}: {} hot, {} cold ({:.1}% hot)",
            layer_profile.layer_id,
            hot_indices.len(),
            cold_indices.len(),
            hot_indices.len() as f64 / total_neurons.max(1) as f64 * 100.0
        );

        Ok(LayerClassification {
            layer_id: layer_profile.layer_id,
            neuron_classes,
            hot_indices,
            cold_indices,
            total_neurons,
            hot_threshold: self.config.hot_threshold,
        })
    }

    /// Adjust classification to fit within a VRAM budget.
    fn adjust_for_budget(
        &self,
        classification: &mut ModelClassification,
        profile: &ActivationProfile,
        budget_bytes: usize,
    ) -> Result<()> {
        // Collect all neurons with their frequencies across all layers
        let mut all_neurons: Vec<(usize, usize, f64)> = Vec::new(); // (layer_id, neuron_idx, freq)

        for (&layer_id, layer_profile) in &profile.layer_profiles {
            let freqs = layer_profile.activation_frequencies();
            for (neuron_idx, &freq) in freqs.iter().enumerate() {
                all_neurons.push((layer_id, neuron_idx, freq));
            }
        }

        // Sort by frequency descending — most active neurons get GPU priority
        all_neurons.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Calculate how many neurons we can fit
        let max_hot_neurons = budget_bytes / self.config.bytes_per_weight.max(1);

        // Reset all to cold first
        for (_layer_id, layer_class) in classification.layer_classifications.iter_mut() {
            for class in layer_class.neuron_classes.iter_mut() {
                *class = NeuronClass::Cold;
            }
        }

        // Assign the top N most active neurons as hot
        let mut hot_count = 0;
        for &(layer_id, neuron_idx, _freq) in all_neurons.iter() {
            if hot_count >= max_hot_neurons {
                break;
            }
            if let Some(layer_class) = classification.layer_classifications.get_mut(&layer_id) {
                layer_class.neuron_classes[neuron_idx] = NeuronClass::Hot;
                hot_count += 1;
            }
        }

        // Rebuild index lists and counts
        classification.total_hot = 0;
        classification.total_cold = 0;
        for (_layer_id, layer_class) in classification.layer_classifications.iter_mut() {
            layer_class.hot_indices.clear();
            layer_class.cold_indices.clear();
            for (idx, &class) in layer_class.neuron_classes.iter().enumerate() {
                match class {
                    NeuronClass::Hot => layer_class.hot_indices.push(idx),
                    NeuronClass::Cold => layer_class.cold_indices.push(idx),
                }
            }
            classification.total_hot += layer_class.hot_indices.len();
            classification.total_cold += layer_class.cold_indices.len();
        }
        classification.estimated_gpu_memory = classification.total_hot * self.config.bytes_per_weight;

        info!(
            "Budget-adjusted: {} hot neurons, {} cold, GPU memory: {} / {} bytes",
            classification.total_hot,
            classification.total_cold,
            classification.estimated_gpu_memory,
            budget_bytes
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::{ActivationProfiler, ActivationType, ProfilerConfig};
    use ronn_core::tensor::Tensor;
    use ronn_core::types::{DataType, TensorLayout};

    fn create_test_profile() -> std::result::Result<ActivationProfile, Box<dyn std::error::Error>> {
        let config = ProfilerConfig {
            activation_type: ActivationType::ReLU,
            activation_threshold: 0.0,
            num_samples: 10,
            model_hash: "test".into(),
            dataset_hash: "test".into(),
        };
        let mut profiler = ActivationProfiler::new(config);
        profiler.register_layer(0, 8);

        // Create data where neurons 0-3 are always active, 4-7 never active
        for _ in 0..10 {
            let data = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
            let tensor =
                Tensor::from_data(data, vec![1, 8], DataType::F32, TensorLayout::RowMajor)?;
            profiler.record(0, &tensor)?;
            profiler.complete_sample();
        }

        Ok(profiler.finalize())
    }

    #[test]
    fn test_basic_classification() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let profile = create_test_profile()?;
        let classifier = NeuronClassifier::new(ClassifierConfig {
            hot_threshold: 0.5,
            min_hot_ratio: 0.0,
            ..Default::default()
        });

        let result = classifier.classify(&profile)?;
        let layer_class = &result.layer_classifications[&0];

        assert_eq!(layer_class.hot_indices.len(), 4);
        assert_eq!(layer_class.cold_indices.len(), 4);
        assert!((layer_class.hot_ratio() - 0.5).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_min_hot_ratio() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let profile = create_test_profile()?;

        // Set very high threshold so nothing would be hot by default
        let classifier = NeuronClassifier::new(ClassifierConfig {
            hot_threshold: 1.1, // nothing reaches this
            min_hot_ratio: 0.25, // require at least 25% hot
            ..Default::default()
        });

        let result = classifier.classify(&profile)?;
        let layer_class = &result.layer_classifications[&0];

        // 25% of 8 = 2 neurons minimum
        assert!(layer_class.hot_indices.len() >= 2);

        Ok(())
    }

    #[test]
    fn test_vram_budget_adjustment() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let profile = create_test_profile()?;

        // Budget only allows 2 neurons (2 * 4 bytes = 8 bytes)
        let classifier = NeuronClassifier::new(ClassifierConfig {
            hot_threshold: 0.5,
            vram_budget_bytes: Some(8),
            bytes_per_weight: 4,
            min_hot_ratio: 0.0,
        });

        let result = classifier.classify(&profile)?;
        assert!(result.total_hot <= 2);
        assert!(result.estimated_gpu_memory <= 8);

        Ok(())
    }

    #[test]
    fn test_classification_serialization() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let profile = create_test_profile()?;
        let classifier = NeuronClassifier::with_defaults();
        let result = classifier.classify(&profile)?;

        let bytes = result.to_bytes()?;
        let restored = ModelClassification::from_bytes(&bytes)?;
        assert_eq!(restored.total_hot, result.total_hot);
        assert_eq!(restored.total_cold, result.total_cold);

        Ok(())
    }
}
