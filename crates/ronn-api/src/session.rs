use crate::error::{Error, Result};
use dashmap::DashMap;
use ronn_core::ModelGraph;
use ronn_core::tensor::Tensor;
use ronn_graph::{OptimizationLevel, Optimizer};
use ronn_onnx::LoadedModel;
use ronn_providers::{ProviderRegistry, ProviderType};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Options for configuring an inference session
#[derive(Debug, Clone)]
pub struct SessionOptions {
    optimization_level: OptimizationLevel,
    provider_type: ProviderType,
    num_threads: Option<usize>,
    enable_profiling: bool,
}

impl SessionOptions {
    /// Create new session options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Set execution provider
    pub fn with_provider(mut self, provider: ProviderType) -> Self {
        self.provider_type = provider;
        self
    }

    /// Set number of threads for CPU execution
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Enable profiling
    pub fn with_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    /// Get optimization level
    pub fn optimization_level(&self) -> OptimizationLevel {
        self.optimization_level
    }

    /// Get provider type
    pub fn provider_type(&self) -> ProviderType {
        self.provider_type
    }
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::O2,
            provider_type: ProviderType::CPU,
            num_threads: None,
            enable_profiling: false,
        }
    }
}

/// Builder for creating inference sessions
pub struct SessionBuilder {
    model: Arc<LoadedModel>,
    options: SessionOptions,
}

impl SessionBuilder {
    /// Create a new session builder
    pub fn new(model: Arc<LoadedModel>, options: SessionOptions) -> Self {
        Self { model, options }
    }

    /// Build the inference session
    pub fn build(self) -> Result<InferenceSession> {
        info!(
            "Building inference session with options: {:?}",
            self.options
        );

        // Clone the model graph for optimization
        let mut graph = self.model.graph().clone();

        // Apply optimizations
        let optimizer = Optimizer::new(self.options.optimization_level);
        let stats = optimizer.optimize(&mut graph)?;
        info!(
            "Optimization completed: {} changes in {} iterations",
            stats.total_changes(),
            stats.iterations
        );

        // Initialize provider registry with available providers
        let provider_registry = ronn_providers::create_provider_system().map_err(|e| {
            Error::ProviderError(format!("Failed to create provider system: {}", e))
        })?;

        // Get the requested provider
        let provider = provider_registry
            .get_provider(self.options.provider_type)
            .ok_or_else(|| {
                Error::ProviderError(format!(
                    "Provider {:?} not available",
                    self.options.provider_type
                ))
            })?;

        info!("Using execution provider: {:?}", provider.provider_id());

        let provider_type = self.options.provider_type;

        Ok(InferenceSession {
            model: self.model,
            graph,
            options: self.options,
            provider_registry,
            provider_type,
            value_cache: Arc::new(DashMap::new()),
        })
    }
}

/// An inference session for running a model
pub struct InferenceSession {
    model: Arc<LoadedModel>,
    graph: ModelGraph,
    options: SessionOptions,
    provider_registry: ProviderRegistry,
    provider_type: ProviderType,
    value_cache: Arc<DashMap<String, Tensor>>,
}

impl InferenceSession {
    /// Run inference synchronously
    ///
    /// # Example
    /// ```no_run
    /// use ronn_api::{Model, Tensor};
    /// use ronn_core::{DataType, TensorLayout};
    /// use std::collections::HashMap;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = Model::load("model.onnx")?;
    /// let session = model.create_session_default()?;
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("input", Tensor::zeros(
    ///     vec![1, 3, 224, 224],
    ///     DataType::F32,
    ///     TensorLayout::RowMajor
    /// )?);
    ///
    /// let outputs = session.run(inputs)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run(&self, inputs: HashMap<&str, Tensor>) -> Result<HashMap<String, Tensor>> {
        debug!("Running inference with {} inputs", inputs.len());

        // Validate inputs
        self.validate_inputs(&inputs)?;

        // Load initializers into cache
        for (name, tensor) in self.model.initializers() {
            self.value_cache.insert(name.clone(), tensor.clone());
        }

        // Load input tensors into cache
        for (name, tensor) in inputs {
            self.value_cache.insert(name.to_string(), tensor);
        }

        // Execute the graph
        self.execute_graph()?;

        // Collect outputs
        let mut outputs = HashMap::new();
        for output_info in self.model.outputs() {
            if let Some(tensor) = self.value_cache.get(&output_info.name) {
                outputs.insert(output_info.name.clone(), tensor.clone());
            } else {
                return Err(Error::InferenceError(format!(
                    "Output tensor not found: {}",
                    output_info.name
                )));
            }
        }

        debug!("Inference completed with {} outputs", outputs.len());
        Ok(outputs)
    }

    /// Run inference asynchronously
    pub async fn run_async(
        &self,
        inputs: HashMap<&str, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // For now, just wrap synchronous execution
        // Full async implementation would use tokio::spawn_blocking
        tokio::task::spawn_blocking(move || {
            // This is a simplified version - full implementation would handle the move properly
            Err(Error::InferenceError(
                "Async inference not yet implemented".to_string(),
            ))
        })
        .await
        .map_err(|e| Error::InferenceError(format!("Async execution failed: {}", e)))?
    }

    /// Run inference on a batch of inputs
    pub fn run_batch(
        &self,
        batch: Vec<HashMap<&str, Tensor>>,
    ) -> Result<Vec<HashMap<String, Tensor>>> {
        batch.into_iter().map(|inputs| self.run(inputs)).collect()
    }

    fn validate_inputs(&self, inputs: &HashMap<&str, Tensor>) -> Result<()> {
        for input_info in self.model.inputs() {
            if !inputs.contains_key(input_info.name.as_str()) {
                return Err(Error::InvalidInput(format!(
                    "Missing required input: {}",
                    input_info.name
                )));
            }
        }
        Ok(())
    }

    fn execute_graph(&self) -> Result<()> {
        // Execute nodes in topological order
        for node in self.graph.nodes() {
            debug!("Executing node: {} ({})", node.id, node.op_type);

            // Collect input tensors
            let input_tensors: Vec<Tensor> = node
                .inputs
                .iter()
                .filter_map(|name| self.value_cache.get(name).map(|t| t.clone()))
                .collect();

            // Get the operator implementation
            let op_registry = ronn_onnx::OperatorRegistry::new();
            let op = op_registry.get(&node.op_type).map_err(|e| {
                Error::InferenceError(format!("Operator {} not supported: {}", node.op_type, e))
            })?;

            // Execute the operator
            let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
            let outputs = op
                .execute(&input_refs, &node.attributes)
                .map_err(|e| Error::InferenceError(format!("Operator execution failed: {}", e)))?;

            // Store output tensors
            for (i, tensor) in outputs.into_iter().enumerate() {
                if i < node.outputs.len() {
                    self.value_cache.insert(node.outputs[i].clone(), tensor);
                }
            }
        }

        Ok(())
    }

    /// Get session options
    pub fn options(&self) -> &SessionOptions {
        &self.options
    }

    /// Get the model graph
    pub fn graph(&self) -> &ModelGraph {
        &self.graph
    }
}
