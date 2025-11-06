//! Async inference API for non-blocking, high-throughput inference.
//!
//! This module provides async/await support for RONN inference sessions,
//! enabling efficient concurrent request handling for production workloads.

use crate::{Session, SessionOptions};
use ronn_core::{Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Async wrapper for inference sessions.
///
/// Provides non-blocking inference operations using Tokio runtime.
/// Ideal for web services, concurrent request handling, and high-throughput scenarios.
///
/// # Example
///
/// ```no_run
/// use ronn_api::AsyncSession;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let session = AsyncSession::from_file("model.onnx").await?;
///
///     let mut inputs = HashMap::new();
///     inputs.insert("input".to_string(), tensor);
///
///     let outputs = session.run(inputs).await?;
///     Ok(())
/// }
/// ```
pub struct AsyncSession {
    inner: Arc<RwLock<Session>>,
}

impl AsyncSession {
    /// Create a new async session from an ONNX model file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ronn_api::AsyncSession;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let session = AsyncSession::from_file("model.onnx").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let session = tokio::task::spawn_blocking(move || {
            Session::from_file(path)
        })
        .await
        .map_err(|e| ronn_core::CoreError::from(e.to_string()))??;

        Ok(Self {
            inner: Arc::new(RwLock::new(session)),
        })
    }

    /// Create a new async session with custom options.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    /// * `options` - Session configuration options
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ronn_api::{AsyncSession, SessionOptions, OptimizationLevel};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let options = SessionOptions::new()
    ///     .with_optimization_level(OptimizationLevel::O3);
    ///
    /// let session = AsyncSession::with_options("model.onnx", options).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_options(
        path: impl AsRef<std::path::Path>,
        options: SessionOptions,
    ) -> Result<Self> {
        let session = tokio::task::spawn_blocking(move || {
            Session::with_options(path, options)
        })
        .await
        .map_err(|e| ronn_core::CoreError::from(e.to_string()))??;

        Ok(Self {
            inner: Arc::new(RwLock::new(session)),
        })
    }

    /// Run async inference on the provided inputs.
    ///
    /// This method is non-blocking and returns a future that resolves to the outputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input names to tensors
    ///
    /// # Returns
    ///
    /// A future that resolves to a map of output names to tensors.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ronn_api::AsyncSession;
    /// # use std::collections::HashMap;
    /// # async fn example(session: AsyncSession, inputs: HashMap<String, ronn_core::Tensor>) -> Result<(), Box<dyn std::error::Error>> {
    /// let outputs = session.run(inputs).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let session = self.inner.read().await;

        // Run inference in blocking thread pool to avoid blocking tokio runtime
        tokio::task::spawn_blocking(move || {
            session.run(inputs)
        })
        .await
        .map_err(|e| ronn_core::CoreError::from(e.to_string()))?
    }

    /// Run inference with read lock (allows concurrent reads if session supports it).
    ///
    /// # Note
    ///
    /// This is the same as `run()` for now, but could be optimized for
    /// concurrent inference in the future.
    pub async fn run_concurrent(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        self.run(inputs).await
    }

    /// Clone the async session for sharing across tasks.
    ///
    /// This is cheap (only clones the Arc), allowing multiple tasks
    /// to share the same session.
    pub fn clone_handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Clone for AsyncSession {
    fn clone(&self) -> Self {
        self.clone_handle()
    }
}

/// Batch processor for async inference.
///
/// Collects multiple requests and processes them in batches for improved throughput.
pub struct AsyncBatchProcessor {
    session: AsyncSession,
    max_batch_size: usize,
    timeout_ms: u64,
}

impl AsyncBatchProcessor {
    /// Create a new batch processor.
    ///
    /// # Arguments
    ///
    /// * `session` - The async session to use for inference
    /// * `max_batch_size` - Maximum number of requests to batch together
    /// * `timeout_ms` - Maximum time to wait for batch to fill (milliseconds)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ronn_api::{AsyncSession, AsyncBatchProcessor};
    /// # async fn example(session: AsyncSession) {
    /// let processor = AsyncBatchProcessor::new(
    ///     session,
    ///     32,    // batch up to 32 requests
    ///     10,    // wait max 10ms for batch to fill
    /// );
    /// # }
    /// ```
    pub fn new(session: AsyncSession, max_batch_size: usize, timeout_ms: u64) -> Self {
        Self {
            session,
            max_batch_size,
            timeout_ms,
        }
    }

    /// Submit a request for batched inference.
    ///
    /// The request will be batched with other concurrent requests up to
    /// `max_batch_size` or until `timeout_ms` elapses.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input tensors for this request
    ///
    /// # Returns
    ///
    /// Output tensors for this specific request
    pub async fn infer(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // For now, just pass through to session
        // In a full implementation, this would use a channel to collect
        // requests and batch them together
        self.session.run(inputs).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_session_creation() {
        // This would require a real model file
        // Placeholder for when we have test models
    }

    #[tokio::test]
    async fn test_concurrent_inference() {
        // Test multiple concurrent inference requests
        // Placeholder for when we have test models
    }

    #[tokio::test]
    async fn test_batch_processor() {
        // Test batch processing
        // Placeholder for when we have test models
    }
}
