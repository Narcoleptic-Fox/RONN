//! Sparse model format support for loading and saving sparsity-aware models.
//!
//! Supports two workflows:
//! 1. Profile + Convert: Standard ONNX model -> profile -> train predictors -> .ronn-sparse
//! 2. Import: Load sparsity metadata from external formats
//!
//! The `.ronn-sparse` format bundles:
//! - Original model weights (quantized)
//! - Per-layer activation profiles
//! - Per-layer predictor weights
//! - Hot/cold classification metadata
//! - Target hardware profile

use crate::classifier::ModelClassification;
use crate::error::{Result, SparsityError};
use crate::predictor::ModelPredictors;
use crate::profiler::ActivationProfile;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Magic bytes for the .ronn-sparse format header.
const RONN_SPARSE_MAGIC: &[u8; 8] = b"RONNSPRS";

/// Current format version.
const RONN_SPARSE_VERSION: u32 = 1;

/// Hardware target profile for sparse model optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// GPU memory available in bytes.
    pub gpu_memory_bytes: usize,
    /// Number of CPU cores available.
    pub cpu_cores: usize,
    /// Target device description.
    pub device_description: String,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            gpu_memory_bytes: 16 * 1024 * 1024 * 1024,
            cpu_cores: num_cpus(),
            device_description: "default".to_string(),
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Complete sparse model bundle containing all sparsity metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseModelBundle {
    /// Format version.
    pub version: u32,
    /// Model identifier hash.
    pub model_hash: String,
    /// Activation profile from calibration.
    pub profile: ActivationProfile,
    /// Trained predictor weights.
    pub predictors: ModelPredictors,
    /// Hot/cold neuron classification.
    pub classification: ModelClassification,
    /// Target hardware profile.
    pub hardware_profile: HardwareProfile,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl SparseModelBundle {
    /// Create a new bundle from components.
    pub fn new(
        profile: ActivationProfile,
        predictors: ModelPredictors,
        classification: ModelClassification,
        hardware_profile: HardwareProfile,
    ) -> Self {
        Self {
            version: RONN_SPARSE_VERSION,
            model_hash: profile.model_hash.clone(),
            profile,
            predictors,
            classification,
            hardware_profile,
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Serialize the bundle to bytes (.ronn-sparse format).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        // Write magic header
        buffer.extend_from_slice(RONN_SPARSE_MAGIC);

        // Write version (4 bytes, little-endian)
        buffer.extend_from_slice(&self.version.to_le_bytes());

        // Serialize the rest as JSON
        let json_data = serde_json::to_vec(self).map_err(|e| {
            SparsityError::Format(format!("Failed to serialize sparse bundle: {}", e))
        })?;

        // Write JSON length (8 bytes, little-endian)
        buffer.extend_from_slice(&(json_data.len() as u64).to_le_bytes());

        // Write JSON data
        buffer.extend_from_slice(&json_data);

        info!(
            "Serialized sparse model bundle: {} bytes ({} layers)",
            buffer.len(),
            self.classification.layer_classifications.len()
        );

        Ok(buffer)
    }

    /// Deserialize a bundle from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 20 {
            return Err(SparsityError::Format(
                "Data too short for .ronn-sparse format".into(),
            ));
        }

        // Verify magic header
        if &data[0..8] != RONN_SPARSE_MAGIC {
            return Err(SparsityError::Format(
                "Invalid magic header: not a .ronn-sparse file".into(),
            ));
        }

        // Read version
        let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        if version != RONN_SPARSE_VERSION {
            return Err(SparsityError::Format(format!(
                "Unsupported format version: {} (expected {})",
                version, RONN_SPARSE_VERSION
            )));
        }

        // Read JSON length
        let json_len = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]) as usize;

        if data.len() < 20 + json_len {
            return Err(SparsityError::Format(format!(
                "Data truncated: expected {} bytes of JSON, got {}",
                json_len,
                data.len() - 20
            )));
        }

        let json_data = &data[20..20 + json_len];
        let bundle: Self = serde_json::from_slice(json_data).map_err(|e| {
            SparsityError::Format(format!("Failed to deserialize sparse bundle: {}", e))
        })?;

        info!(
            "Loaded sparse model bundle: model={}, {} layers",
            bundle.model_hash,
            bundle.classification.layer_classifications.len()
        );

        Ok(bundle)
    }
}

/// Summary information about a sparse model (quick inspection without full load).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseModelInfo {
    /// Model hash.
    pub model_hash: String,
    /// Number of FFN layers with sparsity.
    pub num_layers: usize,
    /// Total hot neurons across all layers.
    pub total_hot: usize,
    /// Total cold neurons across all layers.
    pub total_cold: usize,
    /// Overall hot ratio.
    pub hot_ratio: f64,
    /// Estimated GPU memory for hot neurons.
    pub estimated_gpu_memory: usize,
    /// Hardware profile description.
    pub target_hardware: String,
    /// Number of calibration samples used.
    pub calibration_samples: u64,
}

impl SparseModelBundle {
    /// Extract summary information without the full weight data.
    pub fn info(&self) -> SparseModelInfo {
        SparseModelInfo {
            model_hash: self.model_hash.clone(),
            num_layers: self.classification.layer_classifications.len(),
            total_hot: self.classification.total_hot,
            total_cold: self.classification.total_cold,
            hot_ratio: if self.classification.total_hot + self.classification.total_cold > 0 {
                self.classification.total_hot as f64
                    / (self.classification.total_hot + self.classification.total_cold) as f64
            } else {
                0.0
            },
            estimated_gpu_memory: self.classification.estimated_gpu_memory,
            target_hardware: self.hardware_profile.device_description.clone(),
            calibration_samples: self.profile.total_samples,
        }
    }
}

/// Builder for creating sparse model bundles from individual components.
pub struct SparseModelBuilder {
    profile: Option<ActivationProfile>,
    predictors: Option<ModelPredictors>,
    classification: Option<ModelClassification>,
    hardware_profile: HardwareProfile,
    metadata: HashMap<String, String>,
}

impl SparseModelBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            profile: None,
            predictors: None,
            classification: None,
            hardware_profile: HardwareProfile::default(),
            metadata: HashMap::new(),
        }
    }

    /// Set the activation profile.
    pub fn profile(mut self, profile: ActivationProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Set the trained predictors.
    pub fn predictors(mut self, predictors: ModelPredictors) -> Self {
        self.predictors = Some(predictors);
        self
    }

    /// Set the neuron classification.
    pub fn classification(mut self, classification: ModelClassification) -> Self {
        self.classification = Some(classification);
        self
    }

    /// Set the target hardware profile.
    pub fn hardware(mut self, hardware: HardwareProfile) -> Self {
        self.hardware_profile = hardware;
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the sparse model bundle.
    pub fn build(self) -> Result<SparseModelBundle> {
        let profile = self
            .profile
            .ok_or_else(|| SparsityError::Format("Missing activation profile".into()))?;
        let predictors = self
            .predictors
            .ok_or_else(|| SparsityError::Format("Missing predictors".into()))?;
        let classification = self
            .classification
            .ok_or_else(|| SparsityError::Format("Missing classification".into()))?;

        let mut bundle = SparseModelBundle::new(
            profile,
            predictors,
            classification,
            self.hardware_profile,
        );
        bundle.metadata = self.metadata;

        Ok(bundle)
    }
}

impl Default for SparseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::{ClassifierConfig, NeuronClassifier};
    use crate::predictor::{PredictorConfig, PredictorWeights};
    use crate::profiler::{ActivationProfiler, ActivationType, ProfilerConfig};
    use ronn_core::tensor::Tensor;
    use ronn_core::types::{DataType, TensorLayout};

    fn create_test_bundle() -> std::result::Result<SparseModelBundle, Box<dyn std::error::Error>> {
        // Create profile
        let config = ProfilerConfig {
            activation_type: ActivationType::ReLU,
            activation_threshold: 0.0,
            num_samples: 5,
            model_hash: "test_model".into(),
            dataset_hash: "test_data".into(),
        };
        let mut profiler = ActivationProfiler::new(config);
        profiler.register_layer(0, 8);
        for _ in 0..5 {
            let data = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
            let tensor =
                Tensor::from_data(data, vec![1, 8], DataType::F32, TensorLayout::RowMajor)?;
            profiler.record(0, &tensor)?;
            profiler.complete_sample();
        }
        let profile = profiler.finalize();

        // Create classification
        let classifier = NeuronClassifier::new(ClassifierConfig {
            hot_threshold: 0.5,
            min_hot_ratio: 0.0,
            ..Default::default()
        });
        let classification = classifier.classify(&profile)?;

        // Create predictors
        let mut layer_weights = HashMap::new();
        layer_weights.insert(0, PredictorWeights::random(0, 4, 4, 8));

        let predictors = ModelPredictors {
            layer_weights,
            config: PredictorConfig::default(),
            model_hash: "test_model".into(),
        };

        let bundle = SparseModelBuilder::new()
            .profile(profile)
            .predictors(predictors)
            .classification(classification)
            .metadata("source", "unit_test")
            .build()?;

        Ok(bundle)
    }

    #[test]
    fn test_bundle_serialization() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let bundle = create_test_bundle()?;

        let bytes = bundle.to_bytes()?;
        assert!(bytes.len() > 20);
        assert_eq!(&bytes[0..8], RONN_SPARSE_MAGIC);

        let restored = SparseModelBundle::from_bytes(&bytes)?;
        assert_eq!(restored.model_hash, "test_model");
        assert_eq!(restored.version, RONN_SPARSE_VERSION);
        assert_eq!(
            restored.classification.layer_classifications.len(),
            bundle.classification.layer_classifications.len()
        );

        Ok(())
    }

    #[test]
    fn test_bundle_info() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let bundle = create_test_bundle()?;
        let info = bundle.info();

        assert_eq!(info.model_hash, "test_model");
        assert_eq!(info.num_layers, 1);
        assert_eq!(info.total_hot + info.total_cold, 8);
        assert!(info.hot_ratio >= 0.0 && info.hot_ratio <= 1.0);
        assert_eq!(info.calibration_samples, 5);

        Ok(())
    }

    #[test]
    fn test_invalid_format() {
        let result = SparseModelBundle::from_bytes(b"not valid data");
        assert!(result.is_err());

        let result = SparseModelBundle::from_bytes(b"");
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_fields() {
        let result = SparseModelBuilder::new().build();
        assert!(result.is_err());
    }
}
