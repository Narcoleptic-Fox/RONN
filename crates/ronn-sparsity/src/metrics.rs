//! Sparsity ratio tracking and speedup measurement.
//!
//! Tracks activation sparsity metrics across layers and inference runs,
//! providing visibility into how much computation is being skipped and
//! the resulting speedup.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Per-layer sparsity statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSparsityStats {
    /// Layer identifier.
    pub layer_id: usize,
    /// Total neurons in this layer.
    pub total_neurons: usize,
    /// Number of hot neurons (always computed).
    pub hot_neurons: usize,
    /// Average number of neurons predicted active per inference.
    pub avg_active_neurons: f64,
    /// Average sparsity ratio (fraction of neurons skipped).
    pub avg_sparsity_ratio: f64,
    /// Number of inferences measured.
    pub inference_count: u64,
    /// Cumulative active neuron count (for running average).
    cumulative_active: f64,
}

impl LayerSparsityStats {
    /// Create new stats for a layer.
    pub fn new(layer_id: usize, total_neurons: usize, hot_neurons: usize) -> Self {
        Self {
            layer_id,
            total_neurons,
            hot_neurons,
            avg_active_neurons: 0.0,
            avg_sparsity_ratio: 0.0,
            inference_count: 0,
            cumulative_active: 0.0,
        }
    }

    /// Record an inference observation.
    pub fn record(&mut self, active_neurons: usize) {
        self.inference_count += 1;
        self.cumulative_active += active_neurons as f64;
        self.avg_active_neurons = self.cumulative_active / self.inference_count as f64;
        self.avg_sparsity_ratio =
            1.0 - (self.avg_active_neurons / self.total_neurons.max(1) as f64);
    }
}

/// Timing information for sparse vs dense comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Total sparse inference time accumulated.
    pub total_sparse_time_us: u64,
    /// Total dense inference time accumulated (estimated or measured).
    pub total_dense_time_us: u64,
    /// Number of timing samples.
    pub sample_count: u64,
}

impl Default for TimingStats {
    fn default() -> Self {
        Self {
            total_sparse_time_us: 0,
            total_dense_time_us: 0,
            sample_count: 0,
        }
    }
}

impl TimingStats {
    /// Record a timing observation.
    pub fn record(&mut self, sparse_us: u64, dense_estimate_us: u64) {
        self.total_sparse_time_us += sparse_us;
        self.total_dense_time_us += dense_estimate_us;
        self.sample_count += 1;
    }

    /// Compute the speedup ratio (dense_time / sparse_time).
    pub fn speedup_ratio(&self) -> f64 {
        if self.total_sparse_time_us == 0 {
            return 1.0;
        }
        self.total_dense_time_us as f64 / self.total_sparse_time_us as f64
    }

    /// Average sparse inference time in microseconds.
    pub fn avg_sparse_time_us(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        self.total_sparse_time_us as f64 / self.sample_count as f64
    }
}

/// Aggregated sparsity metrics across all layers and inferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityMetrics {
    /// Per-layer statistics.
    pub layer_stats: HashMap<usize, LayerSparsityStats>,
    /// Timing information.
    pub timing: TimingStats,
    /// Total inferences processed.
    pub total_inferences: u64,
    /// Number of predictor invocations.
    pub predictor_invocations: u64,
    /// Total neurons skipped across all layers and inferences.
    pub total_neurons_skipped: u64,
    /// Total neurons computed across all layers and inferences.
    pub total_neurons_computed: u64,
}

impl Default for SparsityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SparsityMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self {
            layer_stats: HashMap::new(),
            timing: TimingStats::default(),
            total_inferences: 0,
            predictor_invocations: 0,
            total_neurons_skipped: 0,
            total_neurons_computed: 0,
        }
    }

    /// Register a layer for tracking.
    pub fn register_layer(&mut self, layer_id: usize, total_neurons: usize, hot_neurons: usize) {
        self.layer_stats
            .insert(layer_id, LayerSparsityStats::new(layer_id, total_neurons, hot_neurons));
    }

    /// Record an inference event for a specific layer.
    pub fn record_layer_inference(&mut self, layer_id: usize, active_neurons: usize) {
        if let Some(stats) = self.layer_stats.get_mut(&layer_id) {
            let skipped = stats.total_neurons.saturating_sub(active_neurons);
            self.total_neurons_skipped += skipped as u64;
            self.total_neurons_computed += active_neurons as u64;
            stats.record(active_neurons);
        }
        self.predictor_invocations += 1;
    }

    /// Record a complete inference pass (across all layers).
    pub fn record_inference(&mut self, sparse_time_us: u64, dense_estimate_us: u64) {
        self.total_inferences += 1;
        self.timing.record(sparse_time_us, dense_estimate_us);
    }

    /// Overall sparsity ratio across all layers.
    pub fn overall_sparsity_ratio(&self) -> f64 {
        let total = self.total_neurons_skipped + self.total_neurons_computed;
        if total == 0 {
            return 0.0;
        }
        self.total_neurons_skipped as f64 / total as f64
    }

    /// Overall speedup ratio.
    pub fn overall_speedup(&self) -> f64 {
        self.timing.speedup_ratio()
    }

    /// Generate a human-readable summary report.
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Sparsity Metrics Summary ===\n");
        report.push_str(&format!("Total inferences: {}\n", self.total_inferences));
        report.push_str(&format!(
            "Overall sparsity ratio: {:.1}%\n",
            self.overall_sparsity_ratio() * 100.0
        ));
        report.push_str(&format!("Overall speedup: {:.2}x\n", self.overall_speedup()));
        report.push_str(&format!(
            "Avg sparse inference: {:.1} us\n",
            self.timing.avg_sparse_time_us()
        ));
        report.push_str(&format!(
            "Neurons skipped: {} / {} ({:.1}%)\n",
            self.total_neurons_skipped,
            self.total_neurons_skipped + self.total_neurons_computed,
            self.overall_sparsity_ratio() * 100.0
        ));

        report.push_str("\nPer-layer breakdown:\n");
        let mut layers: Vec<_> = self.layer_stats.values().collect();
        layers.sort_by_key(|s| s.layer_id);
        for stats in layers {
            report.push_str(&format!(
                "  Layer {}: {:.0}/{} active ({:.1}% sparse)\n",
                stats.layer_id,
                stats.avg_active_neurons,
                stats.total_neurons,
                stats.avg_sparsity_ratio * 100.0,
            ));
        }

        report
    }
}

/// Thread-safe metrics collector for concurrent inference.
#[derive(Debug)]
pub struct AtomicMetricsCollector {
    total_inferences: AtomicU64,
    total_neurons_skipped: AtomicU64,
    total_neurons_computed: AtomicU64,
    predictor_invocations: AtomicU64,
}

impl Default for AtomicMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicMetricsCollector {
    /// Create a new atomic metrics collector.
    pub fn new() -> Self {
        Self {
            total_inferences: AtomicU64::new(0),
            total_neurons_skipped: AtomicU64::new(0),
            total_neurons_computed: AtomicU64::new(0),
            predictor_invocations: AtomicU64::new(0),
        }
    }

    /// Record neurons skipped/computed for one layer in one inference.
    pub fn record_layer(&self, active: u64, total: u64) {
        self.total_neurons_computed.fetch_add(active, Ordering::Relaxed);
        self.total_neurons_skipped
            .fetch_add(total.saturating_sub(active), Ordering::Relaxed);
        self.predictor_invocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a completed inference.
    pub fn record_inference(&self) {
        self.total_inferences.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the current counters.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_inferences: self.total_inferences.load(Ordering::Relaxed),
            total_neurons_skipped: self.total_neurons_skipped.load(Ordering::Relaxed),
            total_neurons_computed: self.total_neurons_computed.load(Ordering::Relaxed),
            predictor_invocations: self.predictor_invocations.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of atomic metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub total_inferences: u64,
    pub total_neurons_skipped: u64,
    pub total_neurons_computed: u64,
    pub predictor_invocations: u64,
}

impl MetricsSnapshot {
    /// Sparsity ratio from snapshot.
    pub fn sparsity_ratio(&self) -> f64 {
        let total = self.total_neurons_skipped + self.total_neurons_computed;
        if total == 0 {
            return 0.0;
        }
        self.total_neurons_skipped as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_sparsity_stats() {
        let mut stats = LayerSparsityStats::new(0, 1000, 200);
        stats.record(300);
        assert_eq!(stats.inference_count, 1);
        assert!((stats.avg_active_neurons - 300.0).abs() < f64::EPSILON);
        assert!((stats.avg_sparsity_ratio - 0.7).abs() < f64::EPSILON);

        stats.record(400);
        assert_eq!(stats.inference_count, 2);
        assert!((stats.avg_active_neurons - 350.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timing_stats() {
        let mut timing = TimingStats::default();
        timing.record(100, 500);
        timing.record(150, 500);
        assert_eq!(timing.sample_count, 2);
        assert!((timing.speedup_ratio() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sparsity_metrics() {
        let mut metrics = SparsityMetrics::new();
        metrics.register_layer(0, 1000, 200);
        metrics.register_layer(1, 2000, 400);

        metrics.record_layer_inference(0, 300);
        metrics.record_layer_inference(1, 500);
        metrics.record_inference(100, 500);

        assert_eq!(metrics.total_inferences, 1);
        assert_eq!(metrics.total_neurons_computed, 800);
        assert_eq!(metrics.total_neurons_skipped, 2200);
        assert!(metrics.overall_sparsity_ratio() > 0.7);
    }

    #[test]
    fn test_atomic_collector() {
        let collector = AtomicMetricsCollector::new();
        collector.record_layer(300, 1000);
        collector.record_inference();

        let snap = collector.snapshot();
        assert_eq!(snap.total_inferences, 1);
        assert_eq!(snap.total_neurons_computed, 300);
        assert_eq!(snap.total_neurons_skipped, 700);
    }
}
