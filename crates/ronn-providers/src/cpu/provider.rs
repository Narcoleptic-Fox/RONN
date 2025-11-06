//! CPU execution provider implementation.
//!
//! This module provides a complete CPU execution provider with SIMD optimizations,
//! multi-threading support, and NUMA awareness.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use rayon::{ThreadPool, ThreadPoolBuilder};
use ronn_core::{
    CompiledKernel, DataType, ExecutionProvider, MemoryType, OperatorSpec, PerformanceProfile,
    ProviderCapability, ProviderConfig, ProviderId, ResourceRequirements, SubGraph,
    TensorAllocator,
};
use tracing::{debug, info, warn};

use super::{
    allocator::{create_cpu_allocator, create_numa_cpu_allocator},
    kernels::CpuKernel,
    simd::{detect_simd_capabilities, SimdCapabilities},
};

/// CPU execution provider with SIMD optimizations and multi-threading.
pub struct CpuExecutionProvider {
    /// Provider configuration.
    config: CpuProviderConfig,
    /// SIMD capabilities detected at initialization.
    simd_capabilities: SimdCapabilities,
    /// Thread pool for parallel execution.
    thread_pool: ThreadPool,
    /// Memory allocator for this provider.
    allocator: Arc<dyn TensorAllocator>,
    /// Set of supported operations.
    supported_ops: HashSet<String>,
}

/// Configuration for CPU execution provider.
#[derive(Debug, Clone)]
pub struct CpuProviderConfig {
    /// Number of worker threads (None = auto-detect).
    pub thread_count: Option<usize>,
    /// Memory limit in bytes (None = no limit).
    pub memory_limit: Option<usize>,
    /// NUMA node preference (-1 = no preference).
    pub numa_node: i32,
    /// Enable SIMD optimizations.
    pub enable_simd: bool,
    /// Enable operator fusion.
    pub enable_fusion: bool,
    /// Thread pool name for debugging.
    pub thread_pool_name: String,
}

impl Default for CpuProviderConfig {
    fn default() -> Self {
        Self {
            thread_count: None,  // Auto-detect based on CPU cores
            memory_limit: None,  // No memory limit
            numa_node: -1,       // No NUMA preference
            enable_simd: true,   // Enable SIMD by default
            enable_fusion: true, // Enable operator fusion
            thread_pool_name: "cpu-provider".to_string(),
        }
    }
}

impl CpuExecutionProvider {
    /// Create a new CPU execution provider with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(CpuProviderConfig::default())
    }

    /// Create a CPU execution provider with custom configuration.
    pub fn with_config(config: CpuProviderConfig) -> Result<Self> {
        let simd_capabilities = if config.enable_simd {
            detect_simd_capabilities()
        } else {
            SimdCapabilities::default() // Disabled SIMD
        };

        info!("Detected SIMD capabilities: {:?}", simd_capabilities);

        // Determine thread count
        let thread_count = config.thread_count.unwrap_or_else(|| {
            let cores = num_cpus::get();
            // Leave one core for system tasks
            (cores - 1).max(1)
        });

        // Create thread pool
        let thread_pool_name = config.thread_pool_name.clone();
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(move |i| format!("{}-worker-{}", thread_pool_name, i))
            .build()
            .map_err(|e| anyhow!("Failed to create thread pool: {}", e))?;

        info!("Created CPU thread pool with {} threads", thread_count);

        // Create allocator (NUMA-aware if specified)
        let allocator: Arc<dyn TensorAllocator> = if config.numa_node >= 0 {
            create_numa_cpu_allocator(config.numa_node)
        } else {
            create_cpu_allocator()
        };

        // Define supported operations
        let mut supported_ops = HashSet::new();

        // Basic arithmetic operations
        supported_ops.insert("Add".to_string());
        supported_ops.insert("Sub".to_string());
        supported_ops.insert("Mul".to_string());
        supported_ops.insert("Div".to_string());

        // Matrix operations
        supported_ops.insert("MatMul".to_string());
        supported_ops.insert("Gemm".to_string());

        // Shape operations
        supported_ops.insert("Reshape".to_string());
        supported_ops.insert("Transpose".to_string());
        supported_ops.insert("Flatten".to_string());
        supported_ops.insert("Squeeze".to_string());
        supported_ops.insert("Unsqueeze".to_string());

        // Reduction operations
        supported_ops.insert("Sum".to_string());
        supported_ops.insert("Mean".to_string());
        supported_ops.insert("Max".to_string());
        supported_ops.insert("Min".to_string());
        supported_ops.insert("ArgMax".to_string());
        supported_ops.insert("ArgMin".to_string());

        // Activation functions
        supported_ops.insert("ReLU".to_string());
        supported_ops.insert("Sigmoid".to_string());
        supported_ops.insert("Tanh".to_string());
        supported_ops.insert("Softmax".to_string());

        // Convolution operations (basic support)
        supported_ops.insert("Conv".to_string());
        supported_ops.insert("MaxPool".to_string());
        supported_ops.insert("AveragePool".to_string());

        // Normalization
        supported_ops.insert("BatchNormalization".to_string());

        // Utility operations
        supported_ops.insert("Concat".to_string());
        supported_ops.insert("Split".to_string());
        supported_ops.insert("Slice".to_string());
        supported_ops.insert("Gather".to_string());

        info!(
            "CPU provider supports {} operation types",
            supported_ops.len()
        );

        Ok(Self {
            config,
            simd_capabilities,
            thread_pool,
            allocator,
            supported_ops,
        })
    }

    /// Get the current configuration.
    pub fn get_config(&self) -> &CpuProviderConfig {
        &self.config
    }

    /// Get SIMD capabilities.
    pub fn get_simd_capabilities(&self) -> &SimdCapabilities {
        &self.simd_capabilities
    }

    /// Get the thread pool.
    pub fn get_thread_pool(&self) -> &ThreadPool {
        &self.thread_pool
    }

    /// Check if an operation type is supported.
    pub fn supports_operation(&self, op_type: &str) -> bool {
        self.supported_ops.contains(op_type)
    }

    /// Estimate execution cost for an operation (for provider selection).
    pub fn estimate_cost(&self, op_spec: &OperatorSpec) -> f64 {
        // Simple cost estimation based on operation type
        // In practice, this would consider input sizes, CPU load, etc.
        match op_spec.op_type.as_str() {
            "Add" | "Sub" | "Mul" | "Div" => 1.0, // Very fast
            "ReLU" | "Sigmoid" | "Tanh" => 2.0,   // Fast
            "MatMul" | "Gemm" => 10.0,            // Medium cost
            "Conv" => 20.0,                       // Higher cost
            "BatchNormalization" => 5.0,          // Medium-low cost
            "Softmax" => 8.0,                     // Medium cost
            _ => 1.0,                             // Default cost
        }
    }
}

impl Default for CpuExecutionProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create default CPU provider")
    }
}

impl ExecutionProvider for CpuExecutionProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::CPU
    }

    fn get_capability(&self) -> ProviderCapability {
        // Build CPU features list
        let mut cpu_features = Vec::new();

        if self.simd_capabilities.sse2 {
            cpu_features.push("sse2".to_string());
        }
        if self.simd_capabilities.sse41 {
            cpu_features.push("sse4.1".to_string());
        }
        if self.simd_capabilities.avx {
            cpu_features.push("avx".to_string());
        }
        if self.simd_capabilities.avx2 {
            cpu_features.push("avx2".to_string());
        }
        if self.simd_capabilities.avx512f {
            cpu_features.push("avx512f".to_string());
        }
        if self.simd_capabilities.fma {
            cpu_features.push("fma".to_string());
        }

        ProviderCapability {
            supported_ops: self.supported_ops.clone(),
            data_types: vec![
                DataType::F32,
                DataType::F16,
                DataType::F64,
                DataType::I8,
                DataType::I32,
                DataType::U8,
                DataType::U32,
                DataType::Bool,
            ],
            memory_types: vec![MemoryType::SystemRAM],
            performance_profile: PerformanceProfile::CPU,
            resource_requirements: ResourceRequirements {
                min_memory_bytes: Some(64 * 1024 * 1024), // 64MB minimum
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
        debug!("Compiling subgraph with {} nodes", subgraph.nodes.len());

        // Validate that all operations are supported
        for node in &subgraph.nodes {
            if !self.supports_operation(&node.op_type) {
                return Err(anyhow!(
                    "Unsupported operation '{}' in subgraph",
                    node.op_type
                ));
            }
        }

        // Compile the kernel
        let kernel = CpuKernel::compile(subgraph, self.simd_capabilities.clone())?;

        debug!("Successfully compiled CPU kernel");

        Ok(Box::new(kernel))
    }

    fn get_allocator(&self) -> Arc<dyn TensorAllocator> {
        self.allocator.clone()
    }

    fn configure(&mut self, config: ProviderConfig) -> Result<()> {
        // Update thread count if specified
        if let Some(thread_count) = config.thread_count {
            if thread_count != self.thread_pool.current_num_threads() {
                warn!(
                    "Thread count change requested ({} -> {}), but requires provider recreation",
                    self.thread_pool.current_num_threads(),
                    thread_count
                );
                // Would need to recreate the thread pool in a real implementation
            }
        }

        // Update memory limit
        if let Some(memory_limit) = config.memory_limit {
            self.config.memory_limit = Some(memory_limit);
            info!("Updated memory limit to {} bytes", memory_limit);
        }

        // Handle custom options
        for (key, value) in &config.custom_options {
            match key.as_str() {
                "numa_node" => {
                    if let Ok(numa_node) = value.parse::<i32>() {
                        self.config.numa_node = numa_node;
                        info!("Updated NUMA node preference to {}", numa_node);
                        // Would need to recreate allocator in a real implementation
                    }
                }
                "enable_simd" => {
                    if let Ok(enable_simd) = value.parse::<bool>() {
                        self.config.enable_simd = enable_simd;
                        info!("Updated SIMD enablement to {}", enable_simd);
                    }
                }
                "enable_fusion" => {
                    if let Ok(enable_fusion) = value.parse::<bool>() {
                        self.config.enable_fusion = enable_fusion;
                        info!("Updated fusion enablement to {}", enable_fusion);
                    }
                }
                _ => {
                    warn!("Unknown configuration option: {}", key);
                }
            }
        }

        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        info!("Shutting down CPU execution provider");

        // The thread pool will be dropped automatically
        // Memory allocator cleanup is handled by Drop traits

        debug!("CPU provider shutdown complete");

        Ok(())
    }
}

/// Create a default CPU execution provider.
pub fn create_cpu_provider() -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(CpuExecutionProvider::new()?))
}

/// Create a CPU execution provider with custom configuration.
pub fn create_cpu_provider_with_config(
    config: CpuProviderConfig,
) -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(CpuExecutionProvider::with_config(config)?))
}

/// Create a NUMA-aware CPU execution provider.
pub fn create_numa_cpu_provider(numa_node: i32) -> Result<Arc<dyn ExecutionProvider>> {
    let config = CpuProviderConfig {
        numa_node,
        ..Default::default()
    };
    create_cpu_provider_with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::GraphNode;
    use std::collections::HashMap;

    #[test]
    fn test_provider_creation() -> Result<()> {
        let provider = CpuExecutionProvider::new()?;

        assert_eq!(provider.provider_id(), ProviderId::CPU);

        let capability = provider.get_capability();
        assert_eq!(capability.performance_profile, PerformanceProfile::CPU);
        assert!(!capability.supported_ops.is_empty());
        assert!(capability.data_types.contains(&DataType::F32));

        Ok(())
    }

    #[test]
    fn test_provider_with_config() -> Result<()> {
        let config = CpuProviderConfig {
            thread_count: Some(2),
            numa_node: 0,
            enable_simd: false,
            ..Default::default()
        };

        let provider = CpuExecutionProvider::with_config(config)?;

        assert_eq!(provider.get_thread_pool().current_num_threads(), 2);
        assert_eq!(provider.get_config().numa_node, 0);
        assert!(!provider.get_config().enable_simd);

        Ok(())
    }

    #[test]
    fn test_operation_support() -> Result<()> {
        let provider = CpuExecutionProvider::new()?;

        // Test basic operations
        assert!(provider.supports_operation("Add"));
        assert!(provider.supports_operation("MatMul"));
        assert!(provider.supports_operation("ReLU"));
        assert!(!provider.supports_operation("NonexistentOp"));

        // Test can_handle
        let ops = vec![
            OperatorSpec {
                op_type: "Add".to_string(),
                input_types: vec![DataType::F32],
                output_types: vec![DataType::F32],
                attributes: HashMap::new(),
            },
            OperatorSpec {
                op_type: "InvalidOp".to_string(),
                input_types: vec![DataType::F32],
                output_types: vec![DataType::F32],
                attributes: HashMap::new(),
            },
        ];

        let support_results = provider.can_handle(&ops);
        assert_eq!(support_results, vec![true, false]);

        Ok(())
    }

    #[test]
    fn test_subgraph_compilation() -> Result<()> {
        let provider = CpuExecutionProvider::new()?;

        let node = GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
            name: Some("test_add".to_string()),
        };

        let subgraph = SubGraph {
            nodes: vec![node],
            edges: vec![],
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
        };

        let kernel = provider.compile_subgraph(subgraph)?;

        // Should have compiled successfully
        let stats = kernel.get_performance_stats();
        assert_eq!(stats.execution_count, 0); // Not executed yet

        Ok(())
    }

    #[test]
    fn test_configuration_update() -> Result<()> {
        let mut provider = CpuExecutionProvider::new()?;

        let config = ProviderConfig {
            thread_count: Some(4),
            memory_limit: Some(128 * 1024 * 1024), // 128MB
            optimization_level: ronn_core::OptimizationLevel::Aggressive,
            custom_options: {
                let mut opts = HashMap::new();
                opts.insert("enable_simd".to_string(), "false".to_string());
                opts.insert("numa_node".to_string(), "1".to_string());
                opts
            },
        };

        provider.configure(config)?;

        // Configuration should have been updated
        assert_eq!(provider.get_config().memory_limit, Some(128 * 1024 * 1024));
        assert!(!provider.get_config().enable_simd);
        assert_eq!(provider.get_config().numa_node, 1);

        Ok(())
    }

    #[test]
    fn test_cost_estimation() -> Result<()> {
        let provider = CpuExecutionProvider::new()?;

        let add_op = OperatorSpec {
            op_type: "Add".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        };

        let conv_op = OperatorSpec {
            op_type: "Conv".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        };

        let add_cost = provider.estimate_cost(&add_op);
        let conv_cost = provider.estimate_cost(&conv_op);

        // Convolution should be more expensive than addition
        assert!(conv_cost > add_cost);

        Ok(())
    }

    #[test]
    fn test_provider_shutdown() -> Result<()> {
        let provider = CpuExecutionProvider::new()?;

        // Should shutdown without errors
        provider.shutdown()?;

        Ok(())
    }

    #[test]
    fn test_allocator() -> Result<()> {
        let provider = CpuExecutionProvider::new()?;
        let allocator = provider.get_allocator();

        // Test basic allocation
        let buffer = allocator.allocate(&[100], DataType::F32)?;
        assert_eq!(buffer.size, 400); // 100 * 4 bytes
        assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

        allocator.deallocate(buffer)?;

        Ok(())
    }

    #[test]
    fn test_factory_functions() -> Result<()> {
        // Test default provider creation
        let provider1 = create_cpu_provider()?;
        assert_eq!(provider1.provider_id(), ProviderId::CPU);

        // Test provider with custom config
        let config = CpuProviderConfig {
            thread_count: Some(1),
            ..Default::default()
        };
        let provider2 = create_cpu_provider_with_config(config)?;
        assert_eq!(provider2.provider_id(), ProviderId::CPU);

        // Test NUMA-aware provider
        let provider3 = create_numa_cpu_provider(0)?;
        assert_eq!(provider3.provider_id(), ProviderId::CPU);

        Ok(())
    }
}
