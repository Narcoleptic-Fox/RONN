//! Session lifecycle management for inference contexts.
//!
//! This module provides thread-safe session management with resource isolation,
//! configuration, and graceful cleanup.

use crate::tensor::Tensor;
use crate::types::{ModelGraph, OptimizationLevel, ProviderId, SessionId};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for inference sessions.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Number of worker threads for this session.
    pub thread_count: Option<usize>,
    /// Memory limit in bytes for this session.
    pub memory_limit: Option<usize>,
    /// Optimization level for model execution.
    pub optimization_level: OptimizationLevel,
    /// Preferred execution providers in priority order.
    pub preferred_providers: Vec<ProviderId>,
    /// Session timeout in seconds.
    pub timeout_seconds: Option<u64>,
    /// Maximum number of concurrent inferences.
    pub max_concurrent_inferences: Option<usize>,
    /// Enable performance metrics collection.
    pub enable_metrics: bool,
    /// Custom configuration options.
    pub custom_options: HashMap<String, String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            thread_count: None,
            memory_limit: None,
            optimization_level: OptimizationLevel::Basic,
            preferred_providers: vec![ProviderId::CPU],
            timeout_seconds: Some(30),
            max_concurrent_inferences: Some(10),
            enable_metrics: true,
            custom_options: HashMap::new(),
        }
    }
}

/// Runtime statistics for a session.
#[derive(Debug, Clone)]
pub struct SessionStatistics {
    /// Total number of inferences performed.
    pub total_inferences: u64,
    /// Total inference time in milliseconds.
    pub total_inference_time_ms: u64,
    /// Average inference time in milliseconds.
    pub average_inference_time_ms: f64,
    /// Minimum inference time in milliseconds.
    pub min_inference_time_ms: Option<u64>,
    /// Maximum inference time in milliseconds.
    pub max_inference_time_ms: Option<u64>,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: usize,
    /// Current memory usage in bytes.
    pub current_memory_bytes: usize,
    /// Number of errors encountered.
    pub error_count: u64,
    /// Session creation time.
    pub created_at: Instant,
    /// Last inference time.
    pub last_inference_at: Option<Instant>,
}

impl Default for SessionStatistics {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            total_inference_time_ms: 0,
            average_inference_time_ms: 0.0,
            min_inference_time_ms: None,
            max_inference_time_ms: None,
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            error_count: 0,
            created_at: Instant::now(),
            last_inference_at: None,
        }
    }
}

/// Resource usage tracking for sessions.
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Current memory usage in bytes.
    current_memory: usize,
    /// Peak memory usage in bytes.
    peak_memory: usize,
    /// Number of active inferences.
    active_inferences: usize,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            current_memory: 0,
            peak_memory: 0,
            active_inferences: 0,
        }
    }
}

/// An active inference session.
#[derive(Debug)]
pub struct InferenceSession {
    /// Unique session identifier.
    pub id: SessionId,
    /// The model graph for this session.
    pub model: Arc<ModelGraph>,
    /// Session configuration.
    pub config: SessionConfig,
    /// Runtime statistics.
    pub statistics: Arc<RwLock<SessionStatistics>>,
    /// Resource usage tracking.
    resource_usage: Arc<RwLock<ResourceUsage>>,
    /// Session creation time.
    created_at: Instant,
    /// Whether the session is marked for deletion.
    marked_for_deletion: bool,
}

impl InferenceSession {
    /// Create a new inference session.
    pub fn new(model: ModelGraph, config: SessionConfig) -> Self {
        let id = SessionId::new_v4();
        let created_at = Instant::now();

        let mut statistics = SessionStatistics::default();
        statistics.created_at = created_at;

        Self {
            id,
            model: Arc::new(model),
            config,
            statistics: Arc::new(RwLock::new(statistics)),
            resource_usage: Arc::new(RwLock::new(ResourceUsage::default())),
            created_at,
            marked_for_deletion: false,
        }
    }

    /// Run inference on the session.
    pub async fn run_inference(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let start_time = Instant::now();

        // Check resource limits
        self.check_resource_limits().await?;

        // Increment active inference count
        {
            let mut usage = self.resource_usage.write().await;
            usage.active_inferences += 1;

            if let Some(max_concurrent) = self.config.max_concurrent_inferences {
                if usage.active_inferences > max_concurrent {
                    usage.active_inferences -= 1;
                    return Err(anyhow!("Max concurrent inferences exceeded"));
                }
            }
        }

        // Simulate inference (in real implementation, this would use execution providers)
        let result = self.execute_inference(inputs).await;

        // Update statistics
        let inference_time = start_time.elapsed();
        self.update_statistics(inference_time, result.is_ok()).await;

        // Decrement active inference count
        {
            let mut usage = self.resource_usage.write().await;
            usage.active_inferences = usage.active_inferences.saturating_sub(1);
        }

        result
    }

    /// Check if resource limits are exceeded.
    async fn check_resource_limits(&self) -> Result<()> {
        let usage = self.resource_usage.read().await;

        if let Some(memory_limit) = self.config.memory_limit {
            if usage.current_memory > memory_limit {
                return Err(anyhow!(
                    "Memory limit exceeded: {} > {}",
                    usage.current_memory,
                    memory_limit
                ));
            }
        }

        // Check timeout
        if let Some(timeout_seconds) = self.config.timeout_seconds {
            let timeout = Duration::from_secs(timeout_seconds);
            if self.created_at.elapsed() > timeout {
                return Err(anyhow!("Session timeout exceeded"));
            }
        }

        Ok(())
    }

    /// Execute the actual inference (placeholder implementation).
    async fn execute_inference(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // This is a placeholder implementation
        // In the real implementation, this would:
        // 1. Select appropriate execution provider
        // 2. Compile the model graph into executable kernels
        // 3. Execute the kernels with the given inputs
        // 4. Return the results

        // For now, just validate inputs match expected graph inputs
        if inputs.len() != self.model.inputs.len() {
            return Err(anyhow!(
                "Input tensor count mismatch: expected {}, got {}",
                self.model.inputs.len(),
                inputs.len()
            ));
        }

        // Simulate some work
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Create dummy outputs based on graph outputs
        let outputs: Result<Vec<Tensor>> = self
            .model
            .outputs
            .iter()
            .enumerate()
            .map(|(i, _output_name)| {
                // Create a small output tensor as placeholder
                Tensor::ones(
                    vec![1, 10],
                    crate::types::DataType::F32,
                    crate::types::TensorLayout::RowMajor,
                )
                .map_err(|e| anyhow!("Failed to create output tensor {}: {}", i, e))
            })
            .collect();

        outputs
    }

    /// Update session statistics.
    async fn update_statistics(&self, inference_time: Duration, success: bool) {
        let mut stats = self.statistics.write().await;

        let inference_time_ms = inference_time.as_millis() as u64;

        if success {
            stats.total_inferences += 1;
            stats.total_inference_time_ms += inference_time_ms;
            stats.average_inference_time_ms =
                stats.total_inference_time_ms as f64 / stats.total_inferences as f64;

            stats.min_inference_time_ms = Some(
                stats
                    .min_inference_time_ms
                    .map_or(inference_time_ms, |min| min.min(inference_time_ms)),
            );

            stats.max_inference_time_ms = Some(
                stats
                    .max_inference_time_ms
                    .map_or(inference_time_ms, |max| max.max(inference_time_ms)),
            );
        } else {
            stats.error_count += 1;
        }

        stats.last_inference_at = Some(Instant::now());
    }

    /// Get session statistics.
    pub async fn get_statistics(&self) -> SessionStatistics {
        self.statistics.read().await.clone()
    }

    /// Get current resource usage.
    pub async fn get_resource_usage(&self) -> ResourceUsage {
        self.resource_usage.read().await.clone()
    }

    /// Mark session for deletion.
    pub fn mark_for_deletion(&mut self) {
        self.marked_for_deletion = true;
    }

    /// Check if session is marked for deletion.
    pub fn is_marked_for_deletion(&self) -> bool {
        self.marked_for_deletion
    }

    /// Get session age.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Thread-safe session manager for managing inference sessions.
#[derive(Debug)]
pub struct SessionManager {
    /// Active sessions storage.
    sessions: DashMap<SessionId, Arc<InferenceSession>>,
    /// Global resource limits.
    global_memory_limit: Option<usize>,
    /// Maximum number of concurrent sessions.
    max_sessions: Option<usize>,
    /// Default session configuration.
    default_config: SessionConfig,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            global_memory_limit: None,
            max_sessions: Some(100),
            default_config: SessionConfig::default(),
        }
    }

    /// Create a new session manager with configuration.
    pub fn with_config(
        global_memory_limit: Option<usize>,
        max_sessions: Option<usize>,
        default_config: SessionConfig,
    ) -> Self {
        Self {
            sessions: DashMap::new(),
            global_memory_limit,
            max_sessions,
            default_config,
        }
    }

    /// Create a new inference session.
    pub async fn create_session(&self, model: ModelGraph) -> Result<SessionId> {
        self.create_session_with_config(model, None).await
    }

    /// Create a new inference session with custom configuration.
    pub async fn create_session_with_config(
        &self,
        model: ModelGraph,
        config: Option<SessionConfig>,
    ) -> Result<SessionId> {
        // Check session limits
        if let Some(max_sessions) = self.max_sessions {
            if self.sessions.len() >= max_sessions {
                // Try cleanup first
                self.cleanup_expired_sessions().await;

                if self.sessions.len() >= max_sessions {
                    return Err(anyhow!(
                        "Maximum number of sessions reached: {}",
                        max_sessions
                    ));
                }
            }
        }

        // Validate model
        model
            .validate()
            .map_err(|e| anyhow!("Invalid model graph: {}", e))?;

        let session_config = config.unwrap_or_else(|| self.default_config.clone());
        let session = Arc::new(InferenceSession::new(model, session_config));
        let session_id = session.id;

        self.sessions.insert(session_id, session);

        tracing::info!(
            "Created session {} with {} nodes",
            session_id,
            self.sessions.get(&session_id).unwrap().model.nodes.len()
        );

        Ok(session_id)
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: SessionId) -> Option<Arc<InferenceSession>> {
        self.sessions
            .get(&session_id)
            .map(|entry| entry.value().clone())
    }

    /// Run inference on a session.
    pub async fn run_inference(
        &self,
        session_id: SessionId,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>> {
        let session = self
            .get_session(session_id)
            .ok_or_else(|| anyhow!("Session not found: {}", session_id))?;

        if session.is_marked_for_deletion() {
            return Err(anyhow!("Session is marked for deletion: {}", session_id));
        }

        session.run_inference(&inputs).await
    }

    /// Destroy a session.
    pub async fn destroy_session(&self, session_id: SessionId) -> Result<()> {
        if let Some((_, session)) = self.sessions.remove(&session_id) {
            // Wait for any ongoing inferences to complete
            let timeout = Duration::from_secs(5);
            let start = Instant::now();

            while start.elapsed() < timeout {
                let usage = session.get_resource_usage().await;
                if usage.active_inferences == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            tracing::info!("Destroyed session {}", session_id);
            Ok(())
        } else {
            Err(anyhow!("Session not found: {}", session_id))
        }
    }

    /// Get session statistics.
    pub async fn get_session_statistics(&self, session_id: SessionId) -> Result<SessionStatistics> {
        let session = self
            .get_session(session_id)
            .ok_or_else(|| anyhow!("Session not found: {}", session_id))?;

        Ok(session.get_statistics().await)
    }

    /// List all active session IDs.
    pub fn list_sessions(&self) -> Vec<SessionId> {
        self.sessions.iter().map(|entry| *entry.key()).collect()
    }

    /// Get the number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Cleanup expired sessions.
    pub async fn cleanup_expired_sessions(&self) -> usize {
        let mut removed_count = 0;
        let max_age = Duration::from_secs(3600); // 1 hour

        let expired_sessions: Vec<SessionId> = self
            .sessions
            .iter()
            .filter_map(|entry| {
                let session = entry.value();
                if session.age() > max_age || session.is_marked_for_deletion() {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();

        for session_id in expired_sessions {
            if self.destroy_session(session_id).await.is_ok() {
                removed_count += 1;
            }
        }

        if removed_count > 0 {
            tracing::info!("Cleaned up {} expired sessions", removed_count);
        }

        removed_count
    }

    /// Get global statistics across all sessions.
    pub async fn get_global_statistics(&self) -> GlobalStatistics {
        let mut global_stats = GlobalStatistics::default();

        for entry in self.sessions.iter() {
            let session = entry.value();
            let stats = session.get_statistics().await;
            let usage = session.get_resource_usage().await;

            global_stats.total_sessions += 1;
            global_stats.total_inferences += stats.total_inferences;
            global_stats.total_errors += stats.error_count;
            global_stats.total_memory_bytes += usage.current_memory;
            global_stats.active_inferences += usage.active_inferences as u64;
        }

        global_stats
    }

    /// Shutdown the session manager and cleanup all sessions.
    pub async fn shutdown(&self) -> Result<()> {
        let session_ids: Vec<SessionId> = self.list_sessions();

        tracing::info!(
            "Shutting down session manager with {} active sessions",
            session_ids.len()
        );

        for session_id in session_ids {
            if let Err(e) = self.destroy_session(session_id).await {
                tracing::warn!("Failed to destroy session {}: {}", session_id, e);
            }
        }

        Ok(())
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global statistics across all sessions.
#[derive(Debug, Clone, Default)]
pub struct GlobalStatistics {
    /// Total number of active sessions.
    pub total_sessions: usize,
    /// Total inferences across all sessions.
    pub total_inferences: u64,
    /// Total errors across all sessions.
    pub total_errors: u64,
    /// Total memory usage across all sessions.
    pub total_memory_bytes: usize,
    /// Total active inferences across all sessions.
    pub active_inferences: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::types::{DataType, TensorLayout};

    fn create_test_graph() -> ModelGraph {
        let mut builder = GraphBuilder::new();

        let input_id = builder.add_op("Input", Some("input_layer".to_string()));
        builder.add_output(input_id, "input_tensor");

        let conv_id = builder.add_op("Conv", Some("conv_layer".to_string()));
        builder
            .add_input(conv_id, "input_tensor")
            .add_output(conv_id, "conv_output");

        builder.connect(input_id, conv_id, "input_tensor").unwrap();
        builder
            .set_inputs(vec!["input_tensor".to_string()])
            .set_outputs(vec!["conv_output".to_string()]);

        builder.build().unwrap()
    }

    #[tokio::test]
    async fn test_session_creation() -> Result<()> {
        let manager = SessionManager::new();
        let graph = create_test_graph();

        let session_id = manager.create_session(graph).await?;
        assert!(manager.get_session(session_id).is_some());
        assert_eq!(manager.session_count(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_inference() -> Result<()> {
        let manager = SessionManager::new();
        let graph = create_test_graph();

        let session_id = manager.create_session(graph).await?;

        // Create test input
        let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;
        let inputs = vec![input];

        let outputs = manager.run_inference(session_id, inputs).await?;
        assert_eq!(outputs.len(), 1);

        let stats = manager.get_session_statistics(session_id).await?;
        assert_eq!(stats.total_inferences, 1);
        assert!(stats.average_inference_time_ms > 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_destruction() -> Result<()> {
        let manager = SessionManager::new();
        let graph = create_test_graph();

        let session_id = manager.create_session(graph).await?;
        assert_eq!(manager.session_count(), 1);

        manager.destroy_session(session_id).await?;
        assert_eq!(manager.session_count(), 0);
        assert!(manager.get_session(session_id).is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_session_limits() -> Result<()> {
        let config = SessionConfig::default();
        let manager = SessionManager::with_config(None, Some(1), config);

        let graph1 = create_test_graph();
        let graph2 = create_test_graph();

        // First session should succeed
        let _session_id1 = manager.create_session(graph1).await?;

        // Second session should fail due to limit
        let result = manager.create_session(graph2).await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_inferences() -> Result<()> {
        let mut config = SessionConfig::default();
        config.max_concurrent_inferences = Some(2);

        let manager = Arc::new(SessionManager::with_config(None, None, config.clone()));
        let graph = create_test_graph();

        let session_id = manager
            .create_session_with_config(graph, Some(config))
            .await?;

        let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

        // Launch multiple concurrent inferences
        let handles: Vec<_> = (0..5)
            .map(|_| {
                let manager = Arc::clone(&manager);
                let input = input.clone();
                tokio::spawn(async move { manager.run_inference(session_id, vec![input]).await })
            })
            .collect();

        let results: Vec<_> = futures::future::join_all(handles).await;

        // Some should succeed, some should fail due to concurrency limit
        let successes = results
            .iter()
            .filter(|r| r.as_ref().unwrap().is_ok())
            .count();
        let failures = results
            .iter()
            .filter(|r| r.as_ref().unwrap().is_err())
            .count();

        assert!(successes > 0);
        assert!(failures > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_global_statistics() -> Result<()> {
        let manager = SessionManager::new();
        let graph = create_test_graph();

        let session_id1 = manager.create_session(graph.clone()).await?;
        let session_id2 = manager.create_session(graph).await?;

        let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

        // Run inference on both sessions
        manager
            .run_inference(session_id1, vec![input.clone()])
            .await?;
        manager.run_inference(session_id2, vec![input]).await?;

        let global_stats = manager.get_global_statistics().await;
        assert_eq!(global_stats.total_sessions, 2);
        assert_eq!(global_stats.total_inferences, 2);

        Ok(())
    }
}
