//! WebAssembly execution provider implementation.
//!
//! This provider enables neural network inference in web browsers and edge
//! environments with optimal performance using WebAssembly SIMD and efficient
//! memory management within browser constraints.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ronn_core::{
    CompiledKernel, DataType, ExecutionProvider, MemoryType, OperatorSpec, PerformanceProfile,
    ProviderCapability, ProviderConfig, ProviderId, ResourceRequirements, SubGraph,
    TensorAllocator,
};
use tracing::{debug, info, warn};

use super::allocator::{WasmMemoryStats, create_wasm_allocator, create_wasm_allocator_with_limit};
use super::bridge::{WasmBridge, WasmBridgeConfig, WorkerPool};
use super::kernels::{WasmKernel, WasmSimd128Ops, create_wasm_kernel};

/// WebAssembly execution provider for browser deployment.
pub struct WasmExecutionProvider {
    /// Provider configuration.
    config: WasmProviderConfig,
    /// Memory allocator optimized for WASM linear memory.
    allocator: Arc<dyn TensorAllocator>,
    /// JavaScript bridge for interoperability.
    bridge: WasmBridge,
    /// Set of supported operations.
    supported_ops: HashSet<String>,
    /// Compiled kernel cache.
    kernel_cache: HashMap<String, Box<dyn CompiledKernel>>,
    /// Worker pool for parallel processing.
    worker_pool: Option<WorkerPool>,
    /// Memory usage statistics.
    memory_stats: WasmMemoryStats,
}

/// Configuration for WebAssembly execution provider.
#[derive(Debug, Clone)]
pub struct WasmProviderConfig {
    /// Memory limit for WASM linear memory.
    pub memory_limit_bytes: usize,
    /// Enable SIMD128 optimizations when available.
    pub enable_simd: bool,
    /// Enable kernel caching.
    pub enable_kernel_caching: bool,
    /// Enable Web Workers for parallelization.
    pub enable_web_workers: bool,
    /// Number of Web Workers (None = auto-detect).
    pub worker_count: Option<usize>,
    /// Bridge configuration.
    pub bridge_config: WasmBridgeConfig,
    /// Optimization level for code generation.
    pub optimization_level: u8,
    /// Enable IndexedDB caching for models.
    pub enable_model_caching: bool,
}

impl Default for WasmProviderConfig {
    fn default() -> Self {
        Self {
            memory_limit_bytes: 256 * 1024 * 1024, // 256MB default
            enable_simd: WasmSimd128Ops::is_simd_available(),
            enable_kernel_caching: true,
            enable_web_workers: true,
            worker_count: None, // Auto-detect
            bridge_config: WasmBridgeConfig::default(),
            optimization_level: 2, // Balanced optimization
            enable_model_caching: true,
        }
    }
}

impl WasmExecutionProvider {
    /// Create a new WASM execution provider with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(WasmProviderConfig::default())
    }

    /// Create a WASM execution provider with custom configuration.
    pub fn with_config(config: WasmProviderConfig) -> Result<Self> {
        info!(
            "Creating WASM execution provider with {}MB memory limit",
            config.memory_limit_bytes / (1024 * 1024)
        );

        // Create memory allocator with the specified limit
        let allocator = create_wasm_allocator_with_limit(config.memory_limit_bytes);

        // Create JavaScript bridge
        let bridge = WasmBridge::with_config(config.bridge_config.clone());

        // Define supported operations
        let mut supported_ops = HashSet::new();

        // Basic arithmetic operations (always supported)
        supported_ops.insert("Add".to_string());
        supported_ops.insert("Sub".to_string());
        supported_ops.insert("Mul".to_string());
        supported_ops.insert("Div".to_string());

        // Matrix operations
        supported_ops.insert("MatMul".to_string());
        supported_ops.insert("Gemm".to_string());

        // Activation functions
        supported_ops.insert("ReLU".to_string());
        supported_ops.insert("Sigmoid".to_string());
        supported_ops.insert("Tanh".to_string());
        supported_ops.insert("Softmax".to_string());

        // Convolution operations (basic support)
        supported_ops.insert("Conv".to_string());

        // Normalization
        supported_ops.insert("BatchNormalization".to_string());

        // Shape operations
        supported_ops.insert("Reshape".to_string());
        supported_ops.insert("Transpose".to_string());

        // Pooling operations
        supported_ops.insert("MaxPool".to_string());
        supported_ops.insert("AveragePool".to_string());

        if config.enable_simd {
            info!("WASM SIMD128 optimizations enabled");
        } else {
            info!("WASM SIMD128 not available, using scalar fallbacks");
        }

        info!("WASM provider supports {} operations", supported_ops.len());

        Ok(Self {
            config,
            allocator,
            bridge,
            supported_ops,
            kernel_cache: HashMap::new(),
            worker_pool: None,
            memory_stats: WasmMemoryStats::new(),
        })
    }

    /// Initialize the provider (async initialization for worker pool).
    pub async fn initialize(&mut self) -> Result<()> {
        if self.config.enable_web_workers {
            info!("Initializing Web Worker pool");
            self.worker_pool = Some(self.bridge.initialize_workers().await?);
        }

        info!("WASM execution provider initialized successfully");
        Ok(())
    }

    /// Get the current configuration.
    pub fn get_config(&self) -> &WasmProviderConfig {
        &self.config
    }

    /// Get memory usage statistics.
    pub fn get_memory_stats(&self) -> &WasmMemoryStats {
        &self.memory_stats
    }

    /// Check if an operation is supported.
    pub fn supports_operation(&self, op_type: &str) -> bool {
        self.supported_ops.contains(op_type)
    }

    /// Estimate execution performance for an operation.
    pub fn estimate_performance(&self, op_spec: &OperatorSpec) -> f32 {
        let base_score = match op_spec.op_type.as_str() {
            "Add" | "Sub" | "Mul" | "Div" => 0.9, // Very fast element-wise ops
            "ReLU" | "Sigmoid" | "Tanh" => 0.8,   // Fast activation functions
            "MatMul" | "Gemm" => 0.7,             // Matrix ops, SIMD helps
            "Conv" => 0.6,                        // Convolution is expensive
            "Softmax" => 0.7,                     // Reasonable performance
            "BatchNormalization" => 0.8,          // Fast normalization
            "Reshape" | "Transpose" => 0.9,       // Memory operations
            "MaxPool" | "AveragePool" => 0.8,     // Pooling operations
            _ => 0.5,                             // Unknown operations
        };

        // Apply SIMD boost
        if self.config.enable_simd {
            base_score * 1.2 // 20% boost for SIMD
        } else {
            base_score
        }
    }

    /// Create a kernel for the specified operation.
    fn create_kernel_for_operation(&self, op_type: &str) -> Result<WasmKernel> {
        if !self.supports_operation(op_type) {
            return Err(anyhow!(
                "Operation {} not supported by WASM provider",
                op_type
            ));
        }

        Ok(create_wasm_kernel(op_type))
    }

    /// Generate cache key for kernel caching.
    fn generate_cache_key(&self, subgraph: &SubGraph) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the subgraph structure
        for node in &subgraph.nodes {
            node.op_type.hash(&mut hasher);
            node.inputs.len().hash(&mut hasher);
            node.outputs.len().hash(&mut hasher);
        }

        // Include WASM-specific factors
        self.config.enable_simd.hash(&mut hasher);
        self.config.optimization_level.hash(&mut hasher);

        format!("wasm_{:x}", hasher.finish())
    }

    /// Get available memory for computation.
    pub fn get_available_memory(&self) -> usize {
        let memory_info = self.allocator.get_memory_info();
        memory_info
            .total_bytes
            .saturating_sub(memory_info.allocated_bytes)
    }

    /// Get cache statistics from the bridge.
    pub fn get_cache_stats(&self) -> super::bridge::CacheStats {
        self.bridge.get_cache_stats()
    }
}

impl Default for WasmExecutionProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create default WASM provider")
    }
}

impl ExecutionProvider for WasmExecutionProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::WebAssembly
    }

    fn get_capability(&self) -> ProviderCapability {
        let mut cpu_features = vec!["wasm32".to_string()];

        if self.config.enable_simd {
            cpu_features.push("simd128".to_string());
        }

        if self.config.enable_web_workers {
            cpu_features.push("web-workers".to_string());
        }

        ProviderCapability {
            supported_ops: self.supported_ops.clone(),
            data_types: vec![
                DataType::F32,  // Primary data type
                DataType::F16,  // Half precision when supported
                DataType::U8,   // Quantized data
                DataType::I8,   // Signed quantized data
                DataType::I32,  // Integer operations
                DataType::U32,  // Unsigned integers
                DataType::Bool, // Boolean operations
            ],
            memory_types: vec![MemoryType::SystemRAM], // WASM linear memory
            performance_profile: PerformanceProfile::MemoryOptimized,
            resource_requirements: ResourceRequirements {
                min_memory_bytes: Some(32 * 1024 * 1024), // 32MB minimum
                cpu_features,
                gpu_memory_bytes: None,
            },
        }
    }

    fn can_handle(&self, operators: &[OperatorSpec]) -> Vec<bool> {
        operators
            .iter()
            .map(|op| self.supports_operation(&op.op_type))
            .collect()
    }

    fn compile_subgraph(&self, subgraph: SubGraph) -> Result<Box<dyn CompiledKernel>> {
        debug!(
            "Compiling WASM subgraph with {} nodes",
            subgraph.nodes.len()
        );

        // Check cache first
        let cache_key = self.generate_cache_key(&subgraph);
        if self.config.enable_kernel_caching {
            if let Some(_cached_kernel) = self.kernel_cache.get(&cache_key) {
                debug!("Retrieved cached WASM kernel for key: {}", cache_key);
                // Note: This is a simplified approach - real implementation would need proper cloning
                return Err(anyhow!("Kernel caching not fully implemented"));
            }
        }

        // For now, compile single-node subgraphs
        if subgraph.nodes.len() != 1 {
            return Err(anyhow!(
                "WASM provider currently supports only single-node subgraphs"
            ));
        }

        let node = &subgraph.nodes[0];
        debug!("Compiling WASM kernel for operation: {}", node.op_type);

        let kernel = self.create_kernel_for_operation(&node.op_type)?;

        info!("Successfully compiled WASM kernel for {}", node.op_type);
        Ok(Box::new(kernel))
    }

    fn get_allocator(&self) -> Arc<dyn TensorAllocator> {
        self.allocator.clone()
    }

    fn configure(&mut self, config: ProviderConfig) -> Result<()> {
        // Update configuration based on generic provider config
        if let Some(memory_limit) = config.memory_limit {
            self.config.memory_limit_bytes = memory_limit;

            // Recreate allocator with new memory limit
            self.allocator = create_wasm_allocator_with_limit(memory_limit);
        }

        // Update bridge cache size based on memory limit
        if let Some(memory_limit) = config.memory_limit {
            self.config.bridge_config.max_cache_size = (memory_limit / 8).min(64 * 1024 * 1024);
            self.bridge = WasmBridge::with_config(self.config.bridge_config.clone());
        }

        // Update optimization level
        match config.optimization_level {
            ronn_core::OptimizationLevel::None => {
                self.config.optimization_level = 0;
                self.config.enable_simd = false;
                self.config.enable_kernel_caching = false;
            }
            ronn_core::OptimizationLevel::Basic => {
                self.config.optimization_level = 1;
                self.config.enable_simd = WasmSimd128Ops::is_simd_available();
                self.config.enable_kernel_caching = true;
            }
            ronn_core::OptimizationLevel::Aggressive => {
                self.config.optimization_level = 2;
                self.config.enable_simd = WasmSimd128Ops::is_simd_available();
                self.config.enable_kernel_caching = true;
                self.config.enable_web_workers = true;
            }
            ronn_core::OptimizationLevel::Custom => {
                // Parse custom options
                for (key, value) in &config.custom_options {
                    match key.as_str() {
                        "enable_simd" => {
                            self.config.enable_simd = value.parse().unwrap_or(true)
                                && WasmSimd128Ops::is_simd_available();
                        }
                        "enable_web_workers" => {
                            self.config.enable_web_workers = value.parse().unwrap_or(true);
                        }
                        "enable_model_caching" => {
                            self.config.enable_model_caching = value.parse().unwrap_or(true);
                        }
                        "worker_count" => {
                            if let Ok(count) = value.parse::<usize>() {
                                self.config.worker_count = Some(count);
                            }
                        }
                        _ => warn!("Unknown WASM config option: {}", key),
                    }
                }
            }
        }

        info!(
            "WASM provider reconfigured with optimization level: {:?}",
            config.optimization_level
        );
        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        info!("Shutting down WASM execution provider");

        // Log final statistics
        let memory_info = self.allocator.get_memory_info();
        let cache_stats = self.get_cache_stats();

        info!("WASM provider statistics:");
        info!("  Memory allocated: {} bytes", memory_info.allocated_bytes);
        info!("  Peak memory: {} bytes", memory_info.peak_bytes);
        info!("  Cache entries: {}", cache_stats.entry_count);
        info!("  Cache size: {} bytes", cache_stats.total_size);
        info!("  Cache hit rate: {:.2}%", cache_stats.hit_rate * 100.0);

        if let Some(ref pool) = self.worker_pool {
            info!("  Available workers: {}", pool.available_count());
        }

        Ok(())
    }
}

/// Create a WASM execution provider with default configuration.
pub fn create_wasm_provider() -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(WasmExecutionProvider::new()?))
}

/// Create a WASM execution provider with custom configuration.
pub fn create_wasm_provider_with_config(
    config: WasmProviderConfig,
) -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(WasmExecutionProvider::with_config(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::{AttributeValue, GraphNode};

    #[test]
    fn test_wasm_provider_creation() -> Result<()> {
        let provider = WasmExecutionProvider::new()?;

        assert_eq!(provider.provider_id(), ProviderId::WebAssembly);
        assert!(provider.supports_operation("Add"));
        assert!(provider.supports_operation("MatMul"));
        assert!(!provider.supports_operation("LSTM")); // Not implemented

        Ok(())
    }

    #[test]
    fn test_provider_capabilities() -> Result<()> {
        let provider = WasmExecutionProvider::new()?;
        let capability = provider.get_capability();

        assert!(capability.supported_ops.contains("Add"));
        assert!(capability.supported_ops.contains("ReLU"));
        assert!(capability.supported_ops.contains("MatMul"));
        assert!(capability.data_types.contains(&DataType::F32));
        assert!(capability.data_types.contains(&DataType::U8));
        assert_eq!(
            capability.performance_profile,
            PerformanceProfile::MemoryOptimized
        );

        Ok(())
    }

    #[test]
    fn test_performance_estimation() -> Result<()> {
        let provider = WasmExecutionProvider::with_config(WasmProviderConfig {
            enable_simd: true,
            ..Default::default()
        })?;

        let add_spec = OperatorSpec {
            op_type: "Add".to_string(),
            input_types: vec![DataType::F32, DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        };

        let performance = provider.estimate_performance(&add_spec);
        assert!(performance > 0.5); // Should be decent performance
        assert!(performance > 0.9); // SIMD boost should make it very fast

        Ok(())
    }

    #[test]
    fn test_subgraph_compilation() -> Result<()> {
        let provider = WasmExecutionProvider::new()?;

        let subgraph = SubGraph {
            nodes: vec![GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["A".to_string(), "B".to_string()],
                outputs: vec!["C".to_string()],
                name: Some("test_add".to_string()),
            }],
            edges: vec![],
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
        };

        let kernel = provider.compile_subgraph(subgraph)?;
        assert_eq!(kernel.get_performance_stats().execution_count, 0);

        Ok(())
    }

    #[test]
    fn test_memory_management() -> Result<()> {
        let provider = WasmExecutionProvider::with_config(WasmProviderConfig {
            memory_limit_bytes: 64 * 1024 * 1024, // 64MB limit
            ..Default::default()
        })?;

        let available_memory = provider.get_available_memory();
        assert!(available_memory > 0);
        assert!(available_memory <= 64 * 1024 * 1024);

        Ok(())
    }

    #[test]
    fn test_provider_configuration() -> Result<()> {
        let mut provider = WasmExecutionProvider::new()?;

        let config = ProviderConfig {
            thread_count: None,
            memory_limit: Some(128 * 1024 * 1024), // 128MB
            optimization_level: ronn_core::OptimizationLevel::Aggressive,
            custom_options: {
                let mut opts = HashMap::new();
                opts.insert("enable_simd".to_string(), "false".to_string());
                opts.insert("enable_web_workers".to_string(), "true".to_string());
                opts
            },
        };

        provider.configure(config)?;

        assert_eq!(provider.config.memory_limit_bytes, 128 * 1024 * 1024);
        // SIMD might still be enabled if available and explicitly enabled in custom options

        Ok(())
    }

    #[test]
    fn test_cache_key_generation() -> Result<()> {
        let provider = WasmExecutionProvider::new()?;

        let subgraph1 = SubGraph {
            nodes: vec![GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["A".to_string(), "B".to_string()],
                outputs: vec!["C".to_string()],
                name: Some("test".to_string()),
            }],
            edges: vec![],
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
        };

        let subgraph2 = SubGraph {
            nodes: vec![GraphNode {
                id: 0,
                op_type: "Mul".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["A".to_string(), "B".to_string()],
                outputs: vec!["C".to_string()],
                name: Some("test".to_string()),
            }],
            edges: vec![],
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
        };

        let key1 = provider.generate_cache_key(&subgraph1);
        let key2 = provider.generate_cache_key(&subgraph2);

        assert_ne!(key1, key2); // Different operations should have different keys
        assert!(key1.starts_with("wasm_"));
        assert!(key2.starts_with("wasm_"));

        Ok(())
    }
}
