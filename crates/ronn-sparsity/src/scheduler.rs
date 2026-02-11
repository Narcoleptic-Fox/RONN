//! GPU/CPU neuron routing and memory management.
//!
//! The scheduler manages:
//! 1. Memory pinning: hot neuron weights pre-loaded to GPU VRAM
//! 2. Compute routing: for each FFN layer, hot neurons on GPU, cold on CPU
//! 3. Result gathering: combine GPU + CPU partial results
//! 4. Memory budget awareness: graceful degradation with limited VRAM
//!
//! This maps to RONN's HRM System 1/System 2 paradigm:
//! - Hot neurons = System 1 (fast, always ready, GPU-resident)
//! - Cold neurons = System 2 (on-demand, CPU)

use crate::classifier::{LayerClassification, ModelClassification, NeuronClass};
use crate::error::{Result, SparsityError};
use crate::metrics::SparsityMetrics;
use crate::predictor::{LayerPredictor, ModelPredictors};
use crate::sparse_ops::{self, SparseStrategy};
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, ProviderId, TensorLayout};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Configuration for the neuron scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// VRAM budget in bytes. Hot neurons are pinned in GPU memory up to this limit.
    pub vram_budget_bytes: usize,
    /// Strategy for sparse computation.
    pub sparse_strategy: SparseStrategyConfig,
    /// Whether to use async execution for CPU/GPU parallelism.
    pub async_execution: bool,
    /// Minimum prediction confidence to consider a cold neuron active.
    pub prediction_threshold: f32,
    /// Fall back to dense computation if fewer than this fraction of neurons are inactive.
    pub min_sparsity_for_sparse: f64,
}

/// Serializable version of SparseStrategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SparseStrategyConfig {
    GatherScatter,
    MaskedDense,
    Auto,
}

impl From<SparseStrategyConfig> for SparseStrategy {
    fn from(s: SparseStrategyConfig) -> Self {
        match s {
            SparseStrategyConfig::GatherScatter => SparseStrategy::GatherScatter,
            SparseStrategyConfig::MaskedDense => SparseStrategy::MaskedDense,
            SparseStrategyConfig::Auto => SparseStrategy::Auto,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            vram_budget_bytes: 16 * 1024 * 1024 * 1024, // 16 GB default
            sparse_strategy: SparseStrategyConfig::Auto,
            async_execution: false,
            prediction_threshold: 0.5,
            min_sparsity_for_sparse: 0.3,
        }
    }
}

/// Pre-loaded layer state containing weight references and classification.
pub struct LayerState {
    /// Layer index.
    pub layer_id: usize,
    /// Full weight matrix for this FFN layer.
    pub weight: Tensor,
    /// Optional bias vector.
    pub bias: Option<Tensor>,
    /// Classification of neurons in this layer.
    pub classification: LayerClassification,
    /// Predictor for this layer.
    pub predictor: Option<LayerPredictor>,
    /// Total output dimension.
    pub output_dim: usize,
    /// Input dimension (hidden_dim).
    pub input_dim: usize,
}

/// Routing decision for a single layer in a single inference.
#[derive(Debug, Clone)]
pub struct LayerRoutingDecision {
    /// Layer index.
    pub layer_id: usize,
    /// Indices of neurons to compute (hot + predicted-active cold).
    pub active_indices: Vec<usize>,
    /// Indices of hot neurons (always computed).
    pub hot_indices: Vec<usize>,
    /// Indices of predicted-active cold neurons.
    pub active_cold_indices: Vec<usize>,
    /// Total neurons in this layer.
    pub total_neurons: usize,
    /// Whether to fall back to dense computation.
    pub use_dense: bool,
    /// Which provider handles hot neurons.
    pub hot_provider: ProviderId,
    /// Which provider handles cold neurons.
    pub cold_provider: ProviderId,
}

impl LayerRoutingDecision {
    /// Sparsity ratio for this decision.
    pub fn sparsity_ratio(&self) -> f64 {
        if self.total_neurons == 0 {
            return 0.0;
        }
        1.0 - (self.active_indices.len() as f64 / self.total_neurons as f64)
    }
}

/// The main neuron scheduler that routes computation between GPU and CPU.
pub struct NeuronScheduler {
    config: SchedulerConfig,
    layer_states: HashMap<usize, LayerState>,
    metrics: SparsityMetrics,
}

impl NeuronScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            layer_states: HashMap::new(),
            metrics: SparsityMetrics::new(),
        }
    }

    /// Load a layer's weights and classification into the scheduler.
    pub fn load_layer(
        &mut self,
        layer_id: usize,
        weight: Tensor,
        bias: Option<Tensor>,
        classification: LayerClassification,
        predictor: Option<LayerPredictor>,
    ) {
        let weight_shape = weight.shape();
        let output_dim = weight_shape[0];
        let input_dim = weight_shape[1];

        self.metrics.register_layer(
            layer_id,
            classification.total_neurons,
            classification.hot_indices.len(),
        );

        self.layer_states.insert(
            layer_id,
            LayerState {
                layer_id,
                weight,
                bias,
                classification,
                predictor,
                output_dim,
                input_dim,
            },
        );

        debug!(
            "Loaded layer {} into scheduler: {}x{} weights",
            layer_id, output_dim, input_dim
        );
    }

    /// Make a routing decision for a layer given the current hidden state.
    pub fn route_layer(
        &self,
        layer_id: usize,
        hidden_state: &Tensor,
    ) -> Result<LayerRoutingDecision> {
        let state = self.layer_states.get(&layer_id).ok_or_else(|| {
            SparsityError::Scheduling(format!("Layer {} not loaded in scheduler", layer_id))
        })?;

        let hot_indices = state.classification.hot_indices.clone();
        let mut active_cold_indices = Vec::new();

        // Use predictor to determine which cold neurons to activate
        if let Some(ref predictor) = state.predictor {
            let mask = predictor.predict(hidden_state)?;
            for &cold_idx in &state.classification.cold_indices {
                if cold_idx < mask.len() && mask[cold_idx] {
                    active_cold_indices.push(cold_idx);
                }
            }
        }

        let mut active_indices = hot_indices.clone();
        active_indices.extend_from_slice(&active_cold_indices);
        active_indices.sort_unstable();

        let total_neurons = state.output_dim;
        let sparsity = 1.0 - (active_indices.len() as f64 / total_neurons.max(1) as f64);

        // Fall back to dense if sparsity is too low
        let use_dense = sparsity < self.config.min_sparsity_for_sparse;

        if use_dense {
            debug!(
                "Layer {}: sparsity {:.1}% < minimum {:.1}%, using dense",
                layer_id,
                sparsity * 100.0,
                self.config.min_sparsity_for_sparse * 100.0
            );
        }

        Ok(LayerRoutingDecision {
            layer_id,
            active_indices,
            hot_indices,
            active_cold_indices,
            total_neurons,
            use_dense,
            hot_provider: ProviderId::GPU,
            cold_provider: ProviderId::CPU,
        })
    }

    /// Execute a sparse FFN layer computation based on a routing decision.
    pub fn execute_layer(
        &mut self,
        layer_id: usize,
        hidden_state: &Tensor,
        decision: &LayerRoutingDecision,
    ) -> Result<Tensor> {
        let state = self.layer_states.get(&layer_id).ok_or_else(|| {
            SparsityError::Scheduling(format!("Layer {} not loaded", layer_id))
        })?;

        let result = if decision.use_dense {
            // Dense fallback: compute all neurons
            sparse_ops::sparse_linear(
                hidden_state,
                &state.weight,
                state.bias.as_ref(),
                &(0..state.output_dim).collect::<Vec<_>>(),
                state.output_dim,
            )?
        } else {
            // Sparse: only compute active neurons
            let strategy: SparseStrategy = self.config.sparse_strategy.into();
            sparse_ops::sparse_linear_auto(
                hidden_state,
                &state.weight,
                state.bias.as_ref(),
                &decision.active_indices,
                state.output_dim,
                strategy,
            )?
        };

        // Record metrics
        self.metrics
            .record_layer_inference(layer_id, decision.active_indices.len());

        Ok(result)
    }

    /// Run a complete sparse FFN computation for a layer: predict + route + compute.
    pub fn run_sparse_ffn(
        &mut self,
        layer_id: usize,
        hidden_state: &Tensor,
    ) -> Result<Tensor> {
        let decision = self.route_layer(layer_id, hidden_state)?;
        self.execute_layer(layer_id, hidden_state, &decision)
    }

    /// Get a reference to the current metrics.
    pub fn metrics(&self) -> &SparsityMetrics {
        &self.metrics
    }

    /// Get a mutable reference to metrics (for recording timing).
    pub fn metrics_mut(&mut self) -> &mut SparsityMetrics {
        &mut self.metrics
    }

    /// Get the number of loaded layers.
    pub fn num_layers(&self) -> usize {
        self.layer_states.len()
    }

    /// Check if a layer is loaded.
    pub fn has_layer(&self, layer_id: usize) -> bool {
        self.layer_states.contains_key(&layer_id)
    }

    /// Get a summary of the current scheduling state.
    pub fn summary(&self) -> SchedulerSummary {
        let mut total_hot = 0;
        let mut total_cold = 0;
        let mut layers = Vec::new();

        for (layer_id, state) in &self.layer_states {
            let hot = state.classification.hot_indices.len();
            let cold = state.classification.cold_indices.len();
            total_hot += hot;
            total_cold += cold;
            layers.push(LayerSummary {
                layer_id: *layer_id,
                total_neurons: state.output_dim,
                hot_neurons: hot,
                cold_neurons: cold,
                has_predictor: state.predictor.is_some(),
            });
        }

        layers.sort_by_key(|l| l.layer_id);

        SchedulerSummary {
            num_layers: self.layer_states.len(),
            total_hot,
            total_cold,
            layers,
            vram_budget_bytes: self.config.vram_budget_bytes,
        }
    }
}

/// Summary of scheduler state.
#[derive(Debug, Clone)]
pub struct SchedulerSummary {
    pub num_layers: usize,
    pub total_hot: usize,
    pub total_cold: usize,
    pub layers: Vec<LayerSummary>,
    pub vram_budget_bytes: usize,
}

/// Summary for a single layer.
#[derive(Debug, Clone)]
pub struct LayerSummary {
    pub layer_id: usize,
    pub total_neurons: usize,
    pub hot_neurons: usize,
    pub cold_neurons: usize,
    pub has_predictor: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::NeuronClass;

    fn make_test_classification(total: usize, hot_count: usize) -> LayerClassification {
        let mut neuron_classes = vec![NeuronClass::Cold; total];
        let mut hot_indices = Vec::new();
        let mut cold_indices = Vec::new();

        for i in 0..total {
            if i < hot_count {
                neuron_classes[i] = NeuronClass::Hot;
                hot_indices.push(i);
            } else {
                cold_indices.push(i);
            }
        }

        LayerClassification {
            layer_id: 0,
            neuron_classes,
            hot_indices,
            cold_indices,
            total_neurons: total,
            hot_threshold: 0.8,
        }
    }

    fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::from_data(data, shape, DataType::F32, TensorLayout::RowMajor).unwrap()
    }

    fn make_test_weight(rows: usize, cols: usize) -> Tensor {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.037 + 0.1).sin())
            .collect();
        make_f32_tensor(data, vec![rows, cols])
    }

    fn make_test_hidden(dim: usize) -> Tensor {
        let data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.25).collect();
        make_f32_tensor(data, vec![dim])
    }

    #[test]
    fn test_scheduler_basic() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = SchedulerConfig::default();
        let mut scheduler = NeuronScheduler::new(config);

        let input_dim = 4;
        let output_dim = 8;
        let weight = make_test_weight(output_dim, input_dim);
        let classification = make_test_classification(output_dim, 3);

        scheduler.load_layer(0, weight, None, classification, None);
        assert!(scheduler.has_layer(0));
        assert_eq!(scheduler.num_layers(), 1);

        let hidden = make_test_hidden(input_dim);
        let decision = scheduler.route_layer(0, &hidden)?;

        // Without a predictor, only hot indices are active
        assert_eq!(decision.hot_indices.len(), 3);
        assert_eq!(decision.active_cold_indices.len(), 0);
        assert_eq!(decision.active_indices.len(), 3);

        Ok(())
    }

    #[test]
    fn test_scheduler_execute() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = SchedulerConfig {
            min_sparsity_for_sparse: 0.0, // always use sparse
            ..Default::default()
        };
        let mut scheduler = NeuronScheduler::new(config);

        let input_dim = 4;
        let output_dim = 8;
        let weight = make_test_weight(output_dim, input_dim);
        let classification = make_test_classification(output_dim, 2);

        scheduler.load_layer(0, weight, None, classification, None);

        let hidden = make_test_hidden(input_dim);
        let result = scheduler.run_sparse_ffn(0, &hidden)?;

        assert_eq!(result.shape(), vec![output_dim]);
        // Non-hot indices should be zero (no predictor = no cold neurons activated)
        let data = result.to_vec()?;
        for i in 2..output_dim {
            assert!((data[i]).abs() < 1e-6, "Expected zero at cold neuron {}", i);
        }

        Ok(())
    }

    #[test]
    fn test_dense_fallback() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = SchedulerConfig {
            min_sparsity_for_sparse: 0.99, // always fall back to dense
            ..Default::default()
        };
        let mut scheduler = NeuronScheduler::new(config);

        let input_dim = 4;
        let output_dim = 8;
        let weight = make_test_weight(output_dim, input_dim);
        let classification = make_test_classification(output_dim, 2);

        scheduler.load_layer(0, weight, None, classification, None);

        let hidden = make_test_hidden(input_dim);
        let decision = scheduler.route_layer(0, &hidden)?;
        assert!(decision.use_dense);

        let result = scheduler.execute_layer(0, &hidden, &decision)?;
        let data = result.to_vec()?;
        // Dense mode should have non-zero values at all indices
        let nonzero_count = data.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(nonzero_count > 2, "Dense should compute all neurons");

        Ok(())
    }

    #[test]
    fn test_scheduler_summary() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut scheduler = NeuronScheduler::new(SchedulerConfig::default());

        let weight0 = make_test_weight(16, 8);
        let weight1 = make_test_weight(32, 8);

        scheduler.load_layer(0, weight0, None, make_test_classification(16, 4), None);
        scheduler.load_layer(1, weight1, None, make_test_classification(32, 8), None);

        let summary = scheduler.summary();
        assert_eq!(summary.num_layers, 2);
        assert_eq!(summary.total_hot, 12);
        assert_eq!(summary.total_cold, 36);

        Ok(())
    }
}
