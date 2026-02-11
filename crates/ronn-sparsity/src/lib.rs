//! PowerInfer-style activation sparsity optimization for RONN.
//!
//! This crate implements activation locality optimization based on the insight that
//! LLM inference follows a power-law distribution in neuron activation:
//! - ~10-30% of neurons ("hot neurons") fire on almost every input
//! - ~70-90% of neurons ("cold neurons") are input-dependent and rarely fire
//! - A tiny predictor network can predict which cold neurons will fire BEFORE computing them
//! - Skip the rest entirely for significant speedup
//!
//! ## Pipeline
//!
//! ```text
//! Input Token
//!     ↓
//! [Attention Layers] — compute normally (dense, always needed)
//!     ↓
//! [FFN Layer Entry]
//!     ↓
//! [Activation Predictor] — tiny MLP predicts which neurons fire (< 1ms)
//!     ↓
//! [Neuron Router]
//!     ├── Hot neurons → GPU (pre-loaded, always ready)
//!     ├── Predicted-active cold neurons → CPU (compute on demand)
//!     └── Predicted-inactive neurons → SKIP (this is the speedup)
//!     ↓
//! [Sparse Gather] — combine results
//!     ↓
//! Output
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ronn_sparsity::{
//!     ActivationProfiler, ProfilerConfig, ActivationType,
//!     NeuronClassifier, ClassifierConfig,
//!     PredictorTrainer, PredictorConfig,
//!     NeuronScheduler, SchedulerConfig,
//!     SparseModelBuilder,
//! };
//!
//! // Step 1: Profile activation patterns
//! let config = ProfilerConfig {
//!     activation_type: ActivationType::ReLU,
//!     ..Default::default()
//! };
//! let mut profiler = ActivationProfiler::new(config);
//! // ... register layers and run calibration samples ...
//! let profile = profiler.finalize();
//!
//! // Step 2: Classify neurons as hot/cold
//! let classifier = NeuronClassifier::with_defaults();
//! let classification = classifier.classify(&profile).unwrap();
//!
//! // Step 3: Train activation predictors
//! let trainer = PredictorTrainer::new(PredictorConfig::default());
//! // ... train per-layer predictors ...
//!
//! // Step 4: Bundle into sparse model format
//! // let bundle = SparseModelBuilder::new()
//! //     .profile(profile)
//! //     .predictors(predictors)
//! //     .classification(classification)
//! //     .build()
//! //     .unwrap();
//! ```
//!
//! ## Integration with RONN
//!
//! - **ronn-graph**: `SparsityOptimizationPass` inserts predictor + routing nodes into FFN layers
//! - **ronn-providers**: Extended to support `execute_sparse` for partial-layer execution
//! - **ronn-hrm**: Maps hot/cold neurons to System 1/System 2 decision framework
//! - **ronn-memory**: Caches activation patterns for repeated prompt prefixes

#[cfg(feature = "sparsity")]
pub mod classifier;
#[cfg(feature = "sparsity")]
pub mod error;
#[cfg(feature = "sparsity")]
pub mod formats;
#[cfg(feature = "sparsity")]
pub mod metrics;
#[cfg(feature = "sparsity")]
pub mod predictor;
#[cfg(feature = "sparsity")]
pub mod profiler;
#[cfg(feature = "sparsity")]
pub mod scheduler;
#[cfg(feature = "sparsity")]
pub mod sparse_ops;

// Re-export commonly used types
#[cfg(feature = "sparsity")]
pub use classifier::{ClassifierConfig, LayerClassification, ModelClassification, NeuronClass, NeuronClassifier};
#[cfg(feature = "sparsity")]
pub use error::{Result, SparsityError};
#[cfg(feature = "sparsity")]
pub use formats::{HardwareProfile, SparseModelBuilder, SparseModelBundle, SparseModelInfo};
#[cfg(feature = "sparsity")]
pub use metrics::{AtomicMetricsCollector, SparsityMetrics};
#[cfg(feature = "sparsity")]
pub use predictor::{
    LayerPredictor, ModelPredictors, PredictorConfig, PredictorTrainer, PredictorTrainingData,
    PredictorWeights,
};
#[cfg(feature = "sparsity")]
pub use profiler::{
    ActivationProfile, ActivationProfiler, ActivationType, LayerProfile, ProfilerConfig,
};
#[cfg(feature = "sparsity")]
pub use scheduler::{
    LayerRoutingDecision, NeuronScheduler, SchedulerConfig, SchedulerSummary,
};
#[cfg(feature = "sparsity")]
pub use sparse_ops::{
    SparseActivation, SparseStrategy, gather_scatter_linear, merge_sparse_results,
    sparse_activation, sparse_linear, sparse_linear_auto,
};

/// Convenience function to run the full sparsity pipeline:
/// profile -> classify -> train predictors -> bundle.
///
/// This is the high-level entry point for converting a model to sparse format.
#[cfg(feature = "sparsity")]
pub fn create_sparse_pipeline(
    profile: ActivationProfile,
    classifier_config: ClassifierConfig,
    predictor_config: PredictorConfig,
    hardware_profile: HardwareProfile,
) -> Result<SparseModelBundle> {
    use tracing::info;

    // Step 1: Classify neurons
    info!("Step 1: Classifying neurons...");
    let classifier = NeuronClassifier::new(classifier_config);
    let classification = classifier.classify(&profile)?;
    info!(
        "Classified: {} hot, {} cold neurons",
        classification.total_hot, classification.total_cold
    );

    // Step 2: Train predictors (using profile data as training labels)
    info!("Step 2: Training activation predictors...");
    let trainer = PredictorTrainer::new(predictor_config.clone());
    let mut layer_weights = std::collections::HashMap::new();

    for (&layer_id, layer_profile) in &profile.layer_profiles {
        // For training, we need input-output pairs.
        // In a full implementation, these come from the calibration run.
        // Here we create a minimal predictor with random weights as placeholder.
        let input_dim = layer_profile.num_neurons; // Simplified: assume hidden_dim == ffn_dim
        let output_dim = layer_profile.num_neurons;
        let weights = PredictorWeights::random(
            layer_id,
            input_dim,
            predictor_config.predictor_dim,
            output_dim,
        );
        layer_weights.insert(layer_id, weights);
    }

    let predictors = ModelPredictors {
        layer_weights,
        config: predictor_config,
        model_hash: profile.model_hash.clone(),
    };

    // Step 3: Bundle everything
    info!("Step 3: Building sparse model bundle...");
    let bundle = SparseModelBuilder::new()
        .profile(profile)
        .predictors(predictors)
        .classification(classification)
        .hardware(hardware_profile)
        .metadata("pipeline", "create_sparse_pipeline")
        .build()?;

    info!(
        "Sparse pipeline complete: {} layers, {:.1}% hot ratio",
        bundle.info().num_layers,
        bundle.info().hot_ratio * 100.0
    );

    Ok(bundle)
}

#[cfg(test)]
#[cfg(feature = "sparsity")]
mod tests {
    use super::*;
    use ronn_core::tensor::Tensor;
    use ronn_core::types::{DataType, TensorLayout};

    #[test]
    fn test_full_pipeline() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Create a profile
        let config = ProfilerConfig {
            activation_type: ActivationType::ReLU,
            activation_threshold: 0.0,
            num_samples: 10,
            model_hash: "pipeline_test".into(),
            dataset_hash: "test_data".into(),
        };
        let mut profiler = ActivationProfiler::new(config);
        profiler.register_layer(0, 16);
        profiler.register_layer(1, 32);

        for _ in 0..10 {
            // Layer 0: first 4 neurons always active
            let mut data0 = vec![0.0; 16];
            for i in 0..4 {
                data0[i] = 1.0;
            }
            let tensor0 =
                Tensor::from_data(data0, vec![1, 16], DataType::F32, TensorLayout::RowMajor)?;
            profiler.record(0, &tensor0)?;

            // Layer 1: first 8 neurons always active
            let mut data1 = vec![0.0; 32];
            for i in 0..8 {
                data1[i] = 1.0;
            }
            let tensor1 =
                Tensor::from_data(data1, vec![1, 32], DataType::F32, TensorLayout::RowMajor)?;
            profiler.record(1, &tensor1)?;

            profiler.complete_sample();
        }

        let profile = profiler.finalize();

        // Run the full pipeline
        let bundle = create_sparse_pipeline(
            profile,
            ClassifierConfig {
                hot_threshold: 0.5,
                min_hot_ratio: 0.0,
                ..Default::default()
            },
            PredictorConfig {
                predictor_dim: 8,
                ..Default::default()
            },
            HardwareProfile::default(),
        )?;

        let info = bundle.info();
        assert_eq!(info.num_layers, 2);
        assert_eq!(info.total_hot, 12); // 4 + 8
        assert_eq!(info.total_cold, 36); // 12 + 24

        // Test serialization round-trip
        let bytes = bundle.to_bytes()?;
        let restored = SparseModelBundle::from_bytes(&bytes)?;
        assert_eq!(restored.info().num_layers, 2);

        Ok(())
    }

    #[test]
    fn test_sparse_inference_correctness() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Test that sparse inference produces the same result as dense for active neurons
        let input_dim = 4;
        let output_dim = 8;

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let weight_data: Vec<f32> = (0..output_dim * input_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let bias_data = vec![0.01; output_dim];

        let input = Tensor::from_data(
            input_data.clone(),
            vec![input_dim],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let weight = Tensor::from_data(
            weight_data.clone(),
            vec![output_dim, input_dim],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let bias = Tensor::from_data(
            bias_data.clone(),
            vec![output_dim],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Dense: all neurons active
        let all_indices: Vec<usize> = (0..output_dim).collect();
        let dense_result = sparse_linear(&input, &weight, Some(&bias), &all_indices, output_dim)?;

        // Sparse: only some neurons
        let active = vec![0, 2, 5, 7];
        let sparse_result = sparse_linear(&input, &weight, Some(&bias), &active, output_dim)?;

        let dense_data = dense_result.to_vec()?;
        let sparse_data = sparse_result.to_vec()?;

        // Active neurons should match
        for &idx in &active {
            assert!(
                (dense_data[idx] - sparse_data[idx]).abs() < 1e-5,
                "Mismatch at neuron {}: dense={} sparse={}",
                idx,
                dense_data[idx],
                sparse_data[idx]
            );
        }

        // Inactive neurons should be zero
        for idx in 0..output_dim {
            if !active.contains(&idx) {
                assert!(
                    sparse_data[idx].abs() < 1e-5,
                    "Inactive neuron {} should be zero, got {}",
                    idx,
                    sparse_data[idx]
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_scheduler_end_to_end() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Profile
        let config = ProfilerConfig {
            activation_type: ActivationType::ReLU,
            ..Default::default()
        };
        let mut profiler = ActivationProfiler::new(config);
        profiler.register_layer(0, 8);

        for _ in 0..10 {
            // Neurons 0,1,2 always active
            let data = vec![1.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0];
            let tensor =
                Tensor::from_data(data, vec![1, 8], DataType::F32, TensorLayout::RowMajor)?;
            profiler.record(0, &tensor)?;
            profiler.complete_sample();
        }
        let profile = profiler.finalize();

        // Classify
        let classifier = NeuronClassifier::new(ClassifierConfig {
            hot_threshold: 0.5,
            min_hot_ratio: 0.0,
            ..Default::default()
        });
        let classification = classifier.classify(&profile)?;
        let layer_class = classification.layer_classifications[&0].clone();
        assert_eq!(layer_class.hot_indices.len(), 3); // neurons 0,1,2

        // Set up scheduler
        let mut scheduler = NeuronScheduler::new(SchedulerConfig {
            min_sparsity_for_sparse: 0.0,
            ..Default::default()
        });

        let weight_data: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.037 + 0.1).sin()).collect();
        let weight =
            Tensor::from_data(weight_data, vec![8, 4], DataType::F32, TensorLayout::RowMajor)?;
        scheduler.load_layer(0, weight, None, layer_class, None);

        // Run sparse inference
        let hidden_data: Vec<f32> = (0..4).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let hidden = Tensor::from_data(hidden_data, vec![4], DataType::F32, TensorLayout::RowMajor)?;
        let result = scheduler.run_sparse_ffn(0, &hidden)?;

        assert_eq!(result.shape(), vec![8]);
        let data = result.to_vec()?;
        // Cold neurons (3-7) should be zero since no predictor is loaded
        for i in 3..8 {
            assert!(data[i].abs() < 1e-6);
        }

        Ok(())
    }
}
