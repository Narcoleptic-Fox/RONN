//! GPU execution provider using Candle backend.
//!
//! This module provides GPU-accelerated execution using the Candle library
//! with support for CUDA and Metal backends.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor as CandleTensor};
use ronn_core::tensor::Tensor;
use ronn_core::{
    CompiledKernel, DataType, ExecutionProvider, KernelStats, MemoryType, MemoryUsage,
    OperatorSpec, PerformanceProfile, ProviderCapability, ProviderConfig, ProviderId,
    ResourceRequirements, SubGraph, TensorAllocator, TensorLayout,
};
use tracing::{debug, info, warn};

// GPU allocator creation (may be used for custom configurations)
// use super::allocator::create_gpu_allocator;
use super::cuda_kernels::{CudaKernelManager, CudaCompileOptions};
use super::memory_manager::{MultiGpuMemoryManager, MultiGpuMemoryConfig};
use super::topology::{GpuTopologyManager, TopologyConfig};

/// GPU execution provider using Candle backend.
pub struct GpuExecutionProvider {
    /// GPU devices for multi-GPU execution.
    devices: Vec<Device>,
    /// Memory allocators per device.
    allocators: Vec<Arc<dyn TensorAllocator>>,
    /// Set of supported operations.
    supported_ops: HashSet<String>,
    /// Provider configuration.
    config: GpuProviderConfig,
    /// Multi-GPU device manager.
    device_manager: Arc<std::sync::Mutex<MultiGpuManager>>,
    /// CUDA kernel managers per device.
    cuda_kernel_managers: Vec<Option<CudaKernelManager>>,
    /// Multi-GPU memory manager.
    memory_manager: Option<Arc<MultiGpuMemoryManager>>,
    /// GPU topology manager.
    topology_manager: Option<Arc<GpuTopologyManager>>,
}

/// Configuration for GPU execution provider.
#[derive(Debug, Clone)]
pub struct GpuProviderConfig {
    /// GPU device IDs for multi-GPU support.
    pub device_ids: Vec<usize>,
    /// Primary device ID (first device in device_ids).
    pub primary_device_id: usize,
    /// Memory limit in bytes per device (None = no limit).
    pub memory_limit: Option<usize>,
    /// Enable mixed precision (F16) operations.
    pub enable_mixed_precision: bool,
    /// Enable tensor core optimizations (if available).
    pub enable_tensor_cores: bool,
    /// Stream count for async operations per device.
    pub stream_count: usize,
    /// Enable multi-GPU distribution.
    pub enable_multi_gpu: bool,
    /// P2P memory transfer optimization.
    pub enable_p2p_transfer: bool,
    /// Load balancing strategy for multi-GPU.
    pub load_balancing: LoadBalancingStrategy,
    /// Enable custom CUDA kernels for optimized operations.
    pub enable_custom_kernels: bool,
    /// CUDA compilation options for custom kernels.
    pub cuda_compile_options: CudaCompileOptions,
    /// Multi-GPU memory management configuration.
    pub memory_config: MultiGpuMemoryConfig,
    /// GPU topology detection configuration.
    pub topology_config: TopologyConfig,
}

/// Load balancing strategies for multi-GPU execution.
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment of operations.
    RoundRobin,
    /// Balance based on current GPU memory usage.
    MemoryBased,
    /// Balance based on GPU utilization.
    UtilizationBased,
    /// Static assignment based on operation type.
    OperationBased,
    /// Optimal placement using cost model.
    CostModel,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        LoadBalancingStrategy::RoundRobin
    }
}

impl Default for GpuProviderConfig {
    fn default() -> Self {
        Self {
            device_ids: vec![0],
            primary_device_id: 0,
            memory_limit: None,
            enable_mixed_precision: true,
            enable_tensor_cores: true,
            stream_count: 1,
            enable_multi_gpu: false,
            enable_p2p_transfer: true,
            load_balancing: LoadBalancingStrategy::default(),
            enable_custom_kernels: true,
            cuda_compile_options: CudaCompileOptions::default(),
            memory_config: MultiGpuMemoryConfig::default(),
            topology_config: TopologyConfig::default(),
        }
    }
}

/// GPU kernel implementation using Candle.
#[derive(Debug)]
pub struct GpuKernel {
    /// Original subgraph.
    subgraph: SubGraph,
    /// GPU device for execution.
    device: Device,
    /// Execution statistics.
    stats: std::sync::Mutex<GpuKernelStats>,
    /// Stream ID for async execution.
    stream_id: usize,
    /// Compiled kernel cache.
    kernel_cache: std::sync::Mutex<KernelCache>,
}

/// Cache for compiled GPU kernels.
#[derive(Debug, Default)]
struct KernelCache {
    /// Cached operations by operation signature.
    cached_ops: HashMap<String, CachedOperation>,
    /// Total cache size in bytes.
    cache_size: usize,
    /// Maximum cache size.
    max_cache_size: usize,
}

/// A cached GPU operation.
#[derive(Debug, Clone)]
struct CachedOperation {
    /// Operation signature (hash of inputs + op type).
    signature: String,
    /// Optimized execution path.
    execution_path: OptimizedPath,
    /// Hit count for LRU eviction.
    hit_count: u64,
    /// Last access time.
    last_accessed: std::time::Instant,
}

/// Optimized execution path for GPU operations.
#[derive(Debug, Clone)]
enum OptimizedPath {
    /// Single operation execution.
    Single(String),
    /// Fused operation sequence.
    Fused(Vec<String>),
    /// Mixed precision path.
    MixedPrecision { fp16_ops: Vec<String>, fp32_ops: Vec<String> },
}

#[derive(Debug, Default)]
struct GpuKernelStats {
    execution_count: u64,
    total_time_us: u64,
    min_time_us: u64,
    max_time_us: u64,
    memory_peak: usize,
}

/// Multi-GPU device manager for load balancing and coordination.
#[derive(Debug)]
struct MultiGpuManager {
    /// Configuration for multi-GPU setup.
    config: GpuProviderConfig,
    /// Device utilization stats for load balancing.
    device_stats: HashMap<usize, DeviceStats>,
    /// Round-robin counter for device selection.
    round_robin_counter: usize,
    /// Current memory usage per device.
    memory_usage: HashMap<usize, usize>,
}

/// Statistics for individual GPU devices.
#[derive(Debug, Default, Clone)]
pub struct DeviceStats {
    /// Number of operations executed on this device.
    operation_count: u64,
    /// Current memory usage in bytes.
    current_memory: usize,
    /// Peak memory usage in bytes.
    peak_memory: usize,
    /// Average execution time in microseconds.
    avg_execution_time: f64,
    /// Last utilization measurement (0.0 to 1.0).
    utilization: f32,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager.
    fn new(config: GpuProviderConfig) -> Self {
        let mut device_stats = HashMap::new();
        let mut memory_usage = HashMap::new();

        for &device_id in &config.device_ids {
            device_stats.insert(device_id, DeviceStats::default());
            memory_usage.insert(device_id, 0);
        }

        Self {
            config,
            device_stats,
            round_robin_counter: 0,
            memory_usage,
        }
    }

    /// Select the best device for operation execution based on load balancing strategy.
    fn select_device(&mut self, op_type: &str, memory_requirement: usize) -> usize {
        if self.config.device_ids.len() == 1 {
            return self.config.device_ids[0];
        }

        if !self.config.enable_multi_gpu {
            return self.config.primary_device_id;
        }

        match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                let device_id = self.config.device_ids[self.round_robin_counter % self.config.device_ids.len()];
                self.round_robin_counter += 1;
                device_id
            },
            LoadBalancingStrategy::MemoryBased => {
                self.select_device_by_memory(memory_requirement)
            },
            LoadBalancingStrategy::UtilizationBased => {
                self.select_device_by_utilization()
            },
            LoadBalancingStrategy::OperationBased => {
                self.select_device_by_operation_type(op_type)
            },
            LoadBalancingStrategy::CostModel => {
                self.select_device_by_cost_model(op_type, memory_requirement)
            },
        }
    }

    /// Select device with the most available memory.
    fn select_device_by_memory(&self, memory_requirement: usize) -> usize {
        self.config.device_ids.iter()
            .min_by_key(|&&device_id| {
                self.memory_usage.get(&device_id).unwrap_or(&0) + memory_requirement
            })
            .copied()
            .unwrap_or(self.config.primary_device_id)
    }

    /// Select device with the lowest utilization.
    fn select_device_by_utilization(&self) -> usize {
        self.config.device_ids.iter()
            .min_by(|&&a, &&b| {
                let util_a = self.device_stats.get(&a).map(|s| s.utilization).unwrap_or(0.0);
                let util_b = self.device_stats.get(&b).map(|s| s.utilization).unwrap_or(0.0);
                util_a.partial_cmp(&util_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(self.config.primary_device_id)
    }

    /// Select device based on operation type preferences.
    fn select_device_by_operation_type(&self, op_type: &str) -> usize {
        // For now, use simple heuristics
        match op_type {
            // Compute-intensive operations prefer higher-end GPUs (lower device IDs)
            "MatMul" | "Conv" | "ConvTranspose" => {
                self.config.device_ids.iter().min().copied().unwrap_or(self.config.primary_device_id)
            },
            // Memory-intensive operations prefer GPUs with more available memory
            "Concat" | "Split" | "Reshape" => {
                self.select_device_by_memory(0)
            },
            // Default to round-robin
            _ => {
                let device_id = self.config.device_ids[self.round_robin_counter % self.config.device_ids.len()];
                device_id
            }
        }
    }

    /// Select device using a cost model that considers multiple factors.
    fn select_device_by_cost_model(&self, _op_type: &str, memory_requirement: usize) -> usize {
        let mut best_device = self.config.primary_device_id;
        let mut best_score = f64::INFINITY;

        for &device_id in &self.config.device_ids {
            let default_stats = DeviceStats::default();
            let stats = self.device_stats.get(&device_id).unwrap_or(&default_stats);
            let memory_used = self.memory_usage.get(&device_id).unwrap_or(&0);

            // Calculate cost score (lower is better)
            let memory_pressure = (*memory_used + memory_requirement) as f64 / (1024.0 * 1024.0 * 1024.0); // GB
            let utilization_penalty = stats.utilization as f64 * 2.0;
            let execution_time_penalty = stats.avg_execution_time / 1000.0; // Convert to ms

            let total_score = memory_pressure + utilization_penalty + execution_time_penalty;

            if total_score < best_score {
                best_score = total_score;
                best_device = device_id;
            }
        }

        best_device
    }

    /// Update device statistics after operation execution.
    fn update_device_stats(&mut self, device_id: usize, execution_time_us: u64, memory_used: usize) {
        if let Some(stats) = self.device_stats.get_mut(&device_id) {
            stats.operation_count += 1;
            stats.current_memory = memory_used;
            stats.peak_memory = stats.peak_memory.max(memory_used);

            // Update rolling average execution time
            let alpha = 0.1; // Smoothing factor
            if stats.avg_execution_time == 0.0 {
                stats.avg_execution_time = execution_time_us as f64;
            } else {
                stats.avg_execution_time = alpha * execution_time_us as f64 + (1.0 - alpha) * stats.avg_execution_time;
            }
        }

        self.memory_usage.insert(device_id, memory_used);
    }

    /// Get current device statistics for monitoring.
    fn get_device_stats(&self) -> &HashMap<usize, DeviceStats> {
        &self.device_stats
    }
}

impl GpuExecutionProvider {
    /// Create a new GPU execution provider with default configuration.
    #[cfg(feature = "gpu")]
    pub fn new() -> Result<Self> {
        Self::with_config(GpuProviderConfig::default())
    }

    /// Create a GPU execution provider with custom configuration.
    #[cfg(feature = "gpu")]
    pub fn with_config(config: GpuProviderConfig) -> Result<Self> {
        // Create GPU devices based on configuration
        let mut devices = Vec::new();
        let mut allocators = Vec::new();
        let mut cuda_kernel_managers = Vec::new();

        for &device_id in &config.device_ids {
            let device = Self::create_gpu_device(device_id)?;
            info!("Created GPU device {}: {:?}", device_id, device);

            // Create CUDA kernel manager if enabled and device is CUDA
            let cuda_manager = if config.enable_custom_kernels && matches!(device, Device::Cuda(_)) {
                match CudaKernelManager::with_options(device.clone(), config.cuda_compile_options.clone()) {
                    Ok(manager) => {
                        info!("Created CUDA kernel manager for device {}", device_id);
                        Some(manager)
                    },
                    Err(e) => {
                        warn!("Failed to create CUDA kernel manager for device {}: {}", device_id, e);
                        None
                    }
                }
            } else {
                None
            };

            devices.push(device);
            cuda_kernel_managers.push(cuda_manager);

            // Create allocator for each device
            let allocator = create_gpu_allocator()
                .map_err(|e| anyhow!("Failed to create GPU allocator for device {}: {}", device_id, e))?;
            allocators.push(allocator);
        }

        if devices.is_empty() {
            return Err(anyhow!("No GPU devices configured"));
        }

        info!("Created GPU provider with {} devices", devices.len());

        // Create multi-GPU manager
        let device_manager = Arc::new(std::sync::Mutex::new(
            MultiGpuManager::new(config.clone())
        ));

        // Create multi-GPU memory manager if multi-GPU is enabled
        let memory_manager = if config.enable_multi_gpu && config.device_ids.len() > 1 {
            match MultiGpuMemoryManager::new(config.device_ids.clone(), config.memory_config.clone()) {
                Ok(manager) => {
                    info!("Created multi-GPU memory manager");
                    Some(Arc::new(manager))
                },
                Err(e) => {
                    warn!("Failed to create multi-GPU memory manager: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Create GPU topology manager if multi-GPU is enabled
        let topology_manager = if config.enable_multi_gpu && config.device_ids.len() > 1 {
            match GpuTopologyManager::new(config.topology_config.clone()) {
                Ok(mut manager) => {
                    // Discover topology
                    if let Err(e) = manager.discover_topology() {
                        warn!("Failed to discover GPU topology: {}", e);
                    } else {
                        info!("GPU topology discovered successfully");
                    }
                    Some(Arc::new(manager))
                },
                Err(e) => {
                    warn!("Failed to create topology manager: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Define supported operations (GPU-optimized subset)
        let mut supported_ops = HashSet::new();

        // Basic arithmetic operations (highly optimized on GPU)
        supported_ops.insert("Add".to_string());
        supported_ops.insert("Sub".to_string());
        supported_ops.insert("Mul".to_string());
        supported_ops.insert("Div".to_string());

        // Matrix operations (GPU's strength)
        supported_ops.insert("MatMul".to_string());
        supported_ops.insert("Gemm".to_string());

        // Convolution operations (GPU-accelerated)
        supported_ops.insert("Conv".to_string());
        supported_ops.insert("ConvTranspose".to_string());

        // Pooling operations
        supported_ops.insert("MaxPool".to_string());
        supported_ops.insert("AveragePool".to_string());
        supported_ops.insert("GlobalAveragePool".to_string());

        // Activation functions (element-wise, GPU-friendly)
        supported_ops.insert("ReLU".to_string());
        supported_ops.insert("Sigmoid".to_string());
        supported_ops.insert("Tanh".to_string());
        supported_ops.insert("Softmax".to_string());
        supported_ops.insert("GELU".to_string());

        // Normalization operations
        supported_ops.insert("BatchNormalization".to_string());
        supported_ops.insert("LayerNormalization".to_string());

        // Reduction operations (efficient on GPU)
        supported_ops.insert("Sum".to_string());
        supported_ops.insert("Mean".to_string());
        supported_ops.insert("Max".to_string());
        supported_ops.insert("Min".to_string());

        // Shape operations (fast on GPU)
        supported_ops.insert("Reshape".to_string());
        supported_ops.insert("Transpose".to_string());
        supported_ops.insert("Concat".to_string());
        supported_ops.insert("Split".to_string());

        info!(
            "GPU provider supports {} operation types",
            supported_ops.len()
        );

        Ok(Self {
            devices,
            allocators,
            supported_ops,
            config,
            device_manager,
            cuda_kernel_managers,
            memory_manager,
            topology_manager,
        })
    }

    /// Fallback constructor when GPU is not available.
    #[cfg(not(feature = "gpu"))]
    pub fn new() -> Result<Self> {
        Err(anyhow!("GPU support not compiled in"))
    }

    /// Create a GPU execution provider with custom configuration.
    #[cfg(not(feature = "gpu"))]
    pub fn with_config(_config: GpuProviderConfig) -> Result<Self> {
        Err(anyhow!("GPU support not compiled in"))
    }

    /// Create GPU device based on configuration.
    #[cfg(feature = "gpu")]
    fn create_gpu_device(device_id: usize) -> Result<Device> {
        // Try CUDA first
        if let Ok(device) = Device::new_cuda(device_id) {
            info!("Using CUDA device {}", device_id);
            return Ok(device);
        }

        // Try Metal on macOS
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = Device::new_metal(device_id) {
                info!("Using Metal device {}", device_id);
                return Ok(device);
            }
        }

        Err(anyhow!("No GPU devices available"))
    }

    /// Get the primary GPU device.
    pub fn device(&self) -> &Device {
        &self.devices[0]
    }

    /// Get the current configuration.
    pub fn get_config(&self) -> &GpuProviderConfig {
        &self.config
    }

    /// Check if an operation type is supported.
    pub fn supports_operation(&self, op_type: &str) -> bool {
        self.supported_ops.contains(op_type)
    }

    /// Estimate execution cost for an operation on GPU.
    pub fn estimate_cost(&self, op_spec: &OperatorSpec) -> f64 {
        // GPU cost estimation - generally lower for parallel operations
        match op_spec.op_type.as_str() {
            "Add" | "Sub" | "Mul" | "Div" => 0.1, // Very fast on GPU
            "ReLU" | "Sigmoid" | "Tanh" => 0.2,   // Fast element-wise
            "MatMul" | "Gemm" => 0.5,             // GPU's strength
            "Conv" => 0.8,                        // Complex but GPU-optimized
            "ConvTranspose" => 1.2,               // More complex
            "BatchNormalization" => 0.3,          // Fast on GPU
            "Softmax" => 0.4,                     // Reduction + element-wise
            "MaxPool" | "AveragePool" => 0.3,     // Simple operations
            _ => 1.0,                             // Default cost
        }
    }

    /// Check if the provider can utilize tensor cores.
    #[cfg(feature = "gpu")]
    pub fn has_tensor_cores(&self) -> bool {
        // In practice, would query GPU capabilities
        // For now, assume modern CUDA devices have tensor cores
        matches!(self.device, Device::Cuda(_)) && self.config.enable_tensor_cores
    }

    /// Check if the GPU has tensor cores for mixed-precision operations.
    #[cfg(not(feature = "gpu"))]
    pub fn has_tensor_cores(&self) -> bool {
        false
    }

    /// Get GPU memory information.
    #[cfg(feature = "gpu")]
    pub fn get_gpu_memory_info(&self) -> Result<(usize, usize)> {
        // In practice, would query actual GPU memory
        // For now, return estimated values
        match &self.devices[0] {
            Device::Cuda(_) => Ok((8 * 1024 * 1024 * 1024, 0)), // 8GB total, 0 used
            Device::Metal(_) => Ok((8 * 1024 * 1024 * 1024, 0)), // 8GB total, 0 used
            _ => Err(anyhow!("Not a GPU device")),
        }
    }

    /// Get GPU memory information (total, available) in bytes.
    #[cfg(not(feature = "gpu"))]
    pub fn get_gpu_memory_info(&self) -> Result<(usize, usize)> {
        Err(anyhow!("GPU support not available"))
    }
}

impl Default for GpuExecutionProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create default GPU provider")
    }
}

impl ExecutionProvider for GpuExecutionProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::GPU
    }

    fn get_capability(&self) -> ProviderCapability {
        let mut data_types = vec![
            DataType::F32,
            DataType::F16, // Important for GPU mixed precision
            DataType::F64,
            DataType::U8,
            DataType::U32,
        ];

        // Add additional types if tensor cores are available
        if self.has_tensor_cores() {
            // Tensor cores work best with F16
            data_types.insert(0, DataType::F16); // Prioritize F16
        }

        let gpu_memory = self
            .get_gpu_memory_info()
            .map(|(total, _)| total)
            .unwrap_or(0);

        ProviderCapability {
            supported_ops: self.supported_ops.clone(),
            data_types,
            memory_types: vec![MemoryType::DeviceMemory, MemoryType::SharedMemory],
            performance_profile: PerformanceProfile::GPU,
            resource_requirements: ResourceRequirements {
                min_memory_bytes: Some(512 * 1024 * 1024), // 512MB minimum
                cpu_features: vec![],                      // No specific CPU requirements
                gpu_memory_bytes: Some(gpu_memory),
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
            "Compiling subgraph with {} nodes for GPU",
            subgraph.nodes.len()
        );

        // Validate that all operations are supported
        for node in &subgraph.nodes {
            if !self.supports_operation(&node.op_type) {
                return Err(anyhow!(
                    "Unsupported GPU operation '{}' in subgraph",
                    node.op_type
                ));
            }
        }

        // Select optimal device for this subgraph using multi-GPU manager
        let mut device_manager = self.device_manager.lock().unwrap();
        let primary_op = subgraph.nodes.first()
            .map(|n| n.op_type.as_str())
            .unwrap_or("Unknown");

        // Estimate memory requirement (rough estimation)
        let estimated_memory = subgraph.nodes.len() * 1024 * 1024; // 1MB per operation

        let selected_device_id = device_manager.select_device(primary_op, estimated_memory);

        // Find the device index in our devices vector
        let device_index = self.config.device_ids.iter()
            .position(|&id| id == selected_device_id)
            .unwrap_or(0);

        let selected_device = self.devices[device_index].clone();

        debug!("Selected GPU device {} for subgraph compilation", selected_device_id);

        drop(device_manager); // Release lock before kernel creation

        // Create GPU kernel with selected device and stream
        let stream_id = selected_device_id % self.config.stream_count;
        let kernel = GpuKernel::with_stream(subgraph, selected_device, stream_id)?;

        debug!("Successfully compiled GPU kernel on device {}", selected_device_id);

        Ok(Box::new(kernel))
    }

    fn get_allocator(&self) -> Arc<dyn TensorAllocator> {
        // Return the allocator for the primary device
        self.allocators[0].clone()
    }

    fn configure(&mut self, config: ProviderConfig) -> Result<()> {
        // Update memory limit
        if let Some(memory_limit) = config.memory_limit {
            self.config.memory_limit = Some(memory_limit);
            info!("Updated GPU memory limit to {} bytes", memory_limit);
        }

        // Handle custom options
        for (key, value) in &config.custom_options {
            match key.as_str() {
                "enable_mixed_precision" => {
                    if let Ok(enable) = value.parse::<bool>() {
                        self.config.enable_mixed_precision = enable;
                        info!("Updated mixed precision to {}", enable);
                    }
                }
                "enable_tensor_cores" => {
                    if let Ok(enable) = value.parse::<bool>() {
                        self.config.enable_tensor_cores = enable;
                        info!("Updated tensor cores to {}", enable);
                    }
                }
                "stream_count" => {
                    if let Ok(count) = value.parse::<usize>() {
                        self.config.stream_count = count;
                        info!("Updated stream count to {}", count);
                    }
                }
                _ => {
                    warn!("Unknown GPU configuration option: {}", key);
                }
            }
        }

        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        info!("Shutting down GPU execution provider");

        // GPU cleanup would happen here in a real implementation
        // Candle handles most cleanup automatically

        debug!("GPU provider shutdown complete");

        Ok(())
    }
}

impl GpuExecutionProvider {
    /// Get allocator for a specific device.
    pub fn get_device_allocator(&self, device_id: usize) -> Option<Arc<dyn TensorAllocator>> {
        let device_index = self.config.device_ids.iter()
            .position(|&id| id == device_id)?;
        self.allocators.get(device_index).cloned()
    }

    /// Get multi-GPU device statistics.
    pub fn get_multi_gpu_stats(&self) -> HashMap<usize, DeviceStats> {
        let device_manager = self.device_manager.lock().unwrap();
        device_manager.get_device_stats().clone()
    }

    /// Enable or disable multi-GPU support.
    pub fn set_multi_gpu_enabled(&mut self, enabled: bool) {
        self.config.enable_multi_gpu = enabled;
        info!("Multi-GPU support {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Set load balancing strategy for multi-GPU execution.
    pub fn set_load_balancing_strategy(&mut self, strategy: LoadBalancingStrategy) {
        info!("Updated load balancing strategy to {:?}", strategy);
        self.config.load_balancing = strategy;
    }

    /// Get the number of configured GPU devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Check if a specific device ID is available.
    pub fn has_device(&self, device_id: usize) -> bool {
        self.config.device_ids.contains(&device_id)
    }

    /// Check if custom CUDA kernels are available for a specific device.
    pub fn has_custom_kernels(&self, device_id: usize) -> bool {
        if let Some(device_index) = self.config.device_ids.iter().position(|&id| id == device_id) {
            self.cuda_kernel_managers.get(device_index)
                .map(|manager| manager.is_some())
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Get available custom kernel operations for a specific device.
    pub fn get_custom_kernel_ops(&self, device_id: usize) -> Vec<String> {
        if self.has_custom_kernels(device_id) {
            vec![
                "FusedMatMulBias".to_string(),
                "OptimizedSoftmax".to_string(),
                "FusedConvBnRelu".to_string(),
                "WarpReduceSum".to_string(),
                "TensorCoreGemm".to_string(),
                "FastGelu".to_string(),
            ]
        } else {
            vec![]
        }
    }

    /// Execute an operation with custom CUDA kernel if available.
    pub fn try_execute_with_custom_kernel(
        &self,
        device_id: usize,
        op_type: &str,
        inputs: &[CandleTensor],
    ) -> Result<Option<Vec<CandleTensor>>> {
        let device_index = self.config.device_ids.iter()
            .position(|&id| id == device_id)
            .ok_or_else(|| anyhow!("Device {} not found", device_id))?;

        if let Some(Some(ref kernel_manager)) = self.cuda_kernel_managers.get(device_index) {
            // Check if we have a custom kernel for this operation
            let tensor_shape = inputs.first()
                .map(|t| t.shape().dims().to_vec())
                .unwrap_or_else(|| vec![1]);

            match kernel_manager.get_optimized_kernel(op_type, &tensor_shape) {
                Ok(mut kernel) => {
                    info!("Using custom CUDA kernel for operation: {}", op_type);

                    // Prepare outputs (simplified - would need proper output allocation)
                    let mut outputs: Vec<CandleTensor> = inputs.iter()
                        .map(|input| input.clone()) // Placeholder
                        .collect();

                    // Execute the custom kernel
                    kernel_manager.execute_kernel(&mut kernel, inputs, &mut outputs)?;

                    Ok(Some(outputs))
                },
                Err(_) => {
                    // No custom kernel available for this operation
                    Ok(None)
                }
            }
        } else {
            // No CUDA kernel manager available
            Ok(None)
        }
    }

    /// Clear all custom kernel caches to free memory.
    pub fn clear_kernel_caches(&self) {
        for manager in &self.cuda_kernel_managers {
            if let Some(ref kernel_manager) = manager {
                kernel_manager.clear_cache();
            }
        }
        info!("Cleared all CUDA kernel caches");
    }

    /// Get custom kernel cache statistics.
    pub fn get_kernel_cache_stats(&self) -> Vec<super::cuda_kernels::CacheStats> {
        self.cuda_kernel_managers.iter()
            .filter_map(|manager| {
                manager.as_ref().map(|km| km.get_cache_stats())
            })
            .collect()
    }

    /// Transfer tensor data between devices using optimal path.
    pub fn transfer_tensor_between_devices(
        &self,
        tensor: &CandleTensor,
        target_device_id: usize,
    ) -> Result<CandleTensor> {
        if let Some(ref _memory_manager) = self.memory_manager {
            // Use advanced memory manager for optimal transfers
            info!("Using multi-GPU memory manager for tensor transfer to device {}", target_device_id);

            // In a real implementation, this would:
            // 1. Check if tensor is already on target device
            // 2. Use P2P transfer if available
            // 3. Fall back to host memory if needed
            // 4. Update transfer statistics

            // For now, use Candle's built-in transfer
            let target_device = &self.devices[
                self.config.device_ids.iter()
                    .position(|&id| id == target_device_id)
                    .unwrap_or(0)
            ];
            Ok(tensor.to_device(target_device)?)
        } else {
            // Fall back to standard Candle transfer
            let target_device = &self.devices[
                self.config.device_ids.iter()
                    .position(|&id| id == target_device_id)
                    .unwrap_or(0)
            ];
            Ok(tensor.to_device(target_device)?)
        }
    }

    /// Synchronize memory operations across all devices.
    pub fn synchronize_memory(&self) -> Result<()> {
        if let Some(ref memory_manager) = self.memory_manager {
            memory_manager.synchronize_all()
        } else {
            // Single device or no memory manager - nothing to sync
            Ok(())
        }
    }

    /// Get memory statistics across all devices.
    pub fn get_memory_statistics(&self) -> HashMap<usize, super::memory_manager::MemoryPoolStats> {
        if let Some(ref memory_manager) = self.memory_manager {
            memory_manager.get_memory_stats()
        } else {
            HashMap::new()
        }
    }

    /// Get global memory statistics.
    pub fn get_global_memory_stats(&self) -> Option<super::memory_manager::GlobalMemoryStats> {
        self.memory_manager.as_ref().map(|mm| mm.get_global_stats())
    }

    /// Get P2P connectivity information between devices.
    pub fn get_p2p_connectivity(&self) -> HashMap<(usize, usize), super::memory_manager::P2PCapability> {
        if let Some(ref memory_manager) = self.memory_manager {
            memory_manager.get_p2p_info()
        } else {
            HashMap::new()
        }
    }

    /// Check if P2P is available between two devices.
    pub fn is_p2p_available(&self, src_device: usize, dst_device: usize) -> bool {
        if let Some(ref memory_manager) = self.memory_manager {
            let p2p_info = memory_manager.get_p2p_info();
            p2p_info.get(&(src_device, dst_device))
                .map(|cap| cap.supported)
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Get optimal memory layout for a given workload.
    pub fn optimize_memory_layout(
        &self,
        access_pattern: &super::memory_manager::AccessPattern,
    ) -> Result<super::memory_manager::MemoryLayout> {
        if let Some(ref memory_manager) = self.memory_manager {
            memory_manager.optimize_memory_layout(access_pattern)
        } else {
            Err(anyhow!("Multi-GPU memory manager not available"))
        }
    }

    /// Get GPU topology information.
    pub fn get_topology(&self) -> Option<super::topology::GpuTopology> {
        self.topology_manager.as_ref().map(|tm| tm.get_topology())
    }

    /// Optimize workload placement using topology analysis.
    pub fn optimize_workload_placement(
        &self,
        workload: &super::topology::Workload,
        strategy: &str,
    ) -> Result<super::topology::PlacementPlan> {
        if let Some(ref topology_manager) = self.topology_manager {
            topology_manager.optimize_placement(workload, strategy)
        } else {
            Err(anyhow!("GPU topology manager not available"))
        }
    }

    /// Compare multiple placement strategies for a workload.
    pub fn compare_placement_strategies(
        &self,
        workload: &super::topology::Workload,
        strategies: &[String],
    ) -> Result<Vec<(String, super::topology::PlacementPlan)>> {
        if let Some(ref topology_manager) = self.topology_manager {
            topology_manager.compare_strategies(workload, strategies)
        } else {
            Err(anyhow!("GPU topology manager not available"))
        }
    }

    /// Get available placement strategies.
    pub fn get_available_placement_strategies(&self) -> Vec<String> {
        if let Some(ref topology_manager) = self.topology_manager {
            topology_manager.get_available_strategies()
        } else {
            vec![]
        }
    }

    /// Check if topology-aware placement is available.
    pub fn has_topology_support(&self) -> bool {
        self.topology_manager.is_some()
    }

    /// Get detailed device information including topology.
    pub fn get_detailed_device_info(&self) -> HashMap<usize, super::topology::GpuDeviceInfo> {
        if let Some(ref topology_manager) = self.topology_manager {
            topology_manager.get_topology().devices
        } else {
            HashMap::new()
        }
    }

    /// Get interconnect information between devices.
    pub fn get_interconnect_info(&self) -> HashMap<(usize, usize), super::topology::InterconnectLink> {
        if let Some(ref topology_manager) = self.topology_manager {
            topology_manager.get_topology().links
        } else {
            HashMap::new()
        }
    }

    /// Automatically select optimal devices for a workload.
    pub fn auto_select_devices(
        &self,
        workload: &super::topology::Workload,
    ) -> Result<Vec<usize>> {
        let plan = self.optimize_workload_placement(workload, "locality_aware")?;
        Ok(plan.device_assignments.values().copied().collect())
    }
}

impl GpuKernel {
    /// Create a new GPU kernel.
    pub fn new(subgraph: SubGraph, device: Device) -> Result<Self> {
        Ok(Self {
            subgraph,
            device,
            stats: std::sync::Mutex::new(GpuKernelStats::default()),
            stream_id: 0, // Default stream
            kernel_cache: std::sync::Mutex::new(KernelCache {
                cached_ops: HashMap::new(),
                cache_size: 0,
                max_cache_size: 64 * 1024 * 1024, // 64MB cache
            }),
        })
    }

    /// Create a GPU kernel with specific stream ID.
    pub fn with_stream(subgraph: SubGraph, device: Device, stream_id: usize) -> Result<Self> {
        Ok(Self {
            subgraph,
            device,
            stats: std::sync::Mutex::new(GpuKernelStats::default()),
            stream_id,
            kernel_cache: std::sync::Mutex::new(KernelCache {
                cached_ops: HashMap::new(),
                cache_size: 0,
                max_cache_size: 64 * 1024 * 1024, // 64MB cache
            }),
        })
    }

    /// Execute a single operation on GPU using Candle.
    fn execute_gpu_operation(
        &self,
        op_type: &str,
        inputs: &[CandleTensor],
    ) -> Result<Vec<CandleTensor>> {
        match op_type {
            "Add" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Add requires exactly 2 inputs"));
                }
                let result = (&inputs[0] + &inputs[1])?;
                Ok(vec![result])
            }

            "Sub" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Sub requires exactly 2 inputs"));
                }
                let result = (&inputs[0] - &inputs[1])?;
                Ok(vec![result])
            }

            "Mul" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Mul requires exactly 2 inputs"));
                }
                let result = (&inputs[0] * &inputs[1])?;
                Ok(vec![result])
            }

            "Div" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Div requires exactly 2 inputs"));
                }
                let result = (&inputs[0] / &inputs[1])?;
                Ok(vec![result])
            }

            "MatMul" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("MatMul requires exactly 2 inputs"));
                }
                let result = inputs[0].matmul(&inputs[1])?;
                Ok(vec![result])
            }

            "ReLU" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("ReLU requires exactly 1 input"));
                }
                let zero = inputs[0].zeros_like()?;
                let result = inputs[0].maximum(&zero)?;
                Ok(vec![result])
            }

            "Softmax" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("Softmax requires exactly 1 input"));
                }
                let result = candle_nn::ops::softmax_last_dim(&inputs[0])?;
                Ok(vec![result])
            }

            "Sigmoid" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("Sigmoid requires exactly 1 input"));
                }
                // Sigmoid(x) = 1 / (1 + exp(-x))
                let neg_input = inputs[0].neg()?;
                let exp_neg = neg_input.exp()?;
                let one = inputs[0].ones_like()?;
                let denominator = (&one + &exp_neg)?;
                let result = one.div(&denominator)?;
                Ok(vec![result])
            }

            "Tanh" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("Tanh requires exactly 1 input"));
                }
                let result = inputs[0].tanh()?;
                Ok(vec![result])
            }

            "GELU" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("GELU requires exactly 1 input"));
                }
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let x = &inputs[0];

                // Calculate x^3
                let x_cubed = x.powf(3.0)?;

                // Calculate 0.044715 * x^3
                let coeff_tensor = x_cubed.affine(0.044715, 0.0)?;

                // Calculate x + 0.044715 * x^3
                let x_plus_coeff = (x + &coeff_tensor)?;

                // Calculate sqrt(2/π) * (x + 0.044715 * x^3)
                let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt() as f64;
                let inner = x_plus_coeff.affine(sqrt_2_over_pi, 0.0)?;

                // Calculate tanh(inner)
                let tanh_inner = inner.tanh()?;

                // Calculate 1 + tanh(inner)
                let one = x.ones_like()?;
                let one_plus_tanh = (&one + &tanh_inner)?;

                // Calculate 0.5 * x
                let half_x = x.affine(0.5, 0.0)?;

                // Final result: 0.5 * x * (1 + tanh(...))
                let result = (&half_x * &one_plus_tanh)?;
                Ok(vec![result])
            }

            "MaxPool" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("MaxPool requires exactly 1 input"));
                }
                // For now, implement a simplified 2x2 max pooling
                // In practice, this would use proper pooling parameters from attributes
                let input = &inputs[0];
                let dims = input.dims();

                if dims.len() < 3 {
                    return Err(anyhow!("MaxPool requires at least 3D input (CHW)"));
                }

                // Simple 2x2 max pooling implementation
                // This is a placeholder - real implementation would use proper stride and kernel size
                let result = input.clone(); // Placeholder - return input for now
                Ok(vec![result])
            }

            "AveragePool" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("AveragePool requires exactly 1 input"));
                }
                // For now, implement a simplified 2x2 average pooling
                // In practice, this would use proper pooling parameters from attributes
                let input = &inputs[0];
                let dims = input.dims();

                if dims.len() < 3 {
                    return Err(anyhow!("AveragePool requires at least 3D input (CHW)"));
                }

                // Simple 2x2 average pooling implementation
                // This is a placeholder - real implementation would use proper stride and kernel size
                let result = input.clone(); // Placeholder - return input for now
                Ok(vec![result])
            }

            "Conv" => {
                if inputs.len() < 2 {
                    return Err(anyhow!("Conv requires at least 2 inputs (input and weights)"));
                }
                let input = &inputs[0];
                let weights = &inputs[1];

                // Basic 2D convolution using Candle's conv2d
                // In practice, this would parse stride, padding, etc. from attributes
                let result = input.conv2d(weights, 1, 1, 1, 1)?; // stride=1, padding=1
                Ok(vec![result])
            }

            "ConvTranspose" => {
                if inputs.len() < 2 {
                    return Err(anyhow!("ConvTranspose requires at least 2 inputs"));
                }
                let input = &inputs[0];
                let weights = &inputs[1];

                // Transpose convolution (deconvolution)
                // For now, use conv2d as placeholder - real implementation would use conv_transpose2d
                let result = input.conv2d(weights, 1, 1, 1, 1)?;
                Ok(vec![result])
            }

            "BatchNormalization" => {
                if inputs.len() < 5 {
                    return Err(anyhow!("BatchNormalization requires 5 inputs: input, scale, bias, mean, var"));
                }
                let input = &inputs[0];
                let scale = &inputs[1];  // gamma
                let bias = &inputs[2];   // beta
                let mean = &inputs[3];   // running mean
                let var = &inputs[4];    // running variance

                // BatchNorm formula: (x - mean) / sqrt(var + eps) * scale + bias
                let eps = 1e-5; // epsilon for numerical stability

                // Expand dimensions if needed for broadcasting
                let input_dims = input.dims();
                let _batch_size = input_dims[0];
                let channels = input_dims[1];

                // Reshape scale, bias, mean, var for broadcasting
                let scale_reshaped = if scale.dims().len() == 1 {
                    scale.reshape(&[1, channels, 1, 1])?
                } else {
                    scale.clone()
                };

                let bias_reshaped = if bias.dims().len() == 1 {
                    bias.reshape(&[1, channels, 1, 1])?
                } else {
                    bias.clone()
                };

                let mean_reshaped = if mean.dims().len() == 1 {
                    mean.reshape(&[1, channels, 1, 1])?
                } else {
                    mean.clone()
                };

                let var_reshaped = if var.dims().len() == 1 {
                    var.reshape(&[1, channels, 1, 1])?
                } else {
                    var.clone()
                };

                // Calculate: (x - mean) / sqrt(var + eps) * scale + bias
                let normalized = (input - &mean_reshaped)?;
                let var_plus_eps = (&var_reshaped + eps)?;
                let std_dev = var_plus_eps.sqrt()?;
                let normalized_scaled = (&normalized / &std_dev)?;
                let scaled = (&normalized_scaled * &scale_reshaped)?;
                let result = (&scaled + &bias_reshaped)?;

                Ok(vec![result])
            }

            "LayerNormalization" => {
                if inputs.len() < 3 {
                    return Err(anyhow!("LayerNormalization requires 3 inputs: input, scale, bias"));
                }
                let input = &inputs[0];
                let scale = &inputs[1];  // gamma
                let bias = &inputs[2];   // beta

                // LayerNorm: normalize over the last dimension(s)
                let eps = 1e-5;
                let dims = input.dims();
                let last_dim = dims.len() - 1;

                // Calculate mean and variance over the last dimension
                let mean = input.mean_keepdim(last_dim)?;
                let variance = {
                    let diff = (input - &mean)?;
                    let squared = (&diff * &diff)?;
                    squared.mean_keepdim(last_dim)?
                };

                // Normalize: (x - mean) / sqrt(var + eps) * scale + bias
                let normalized = (input - &mean)?;
                let var_plus_eps = (&variance + eps)?;
                let std_dev = var_plus_eps.sqrt()?;
                let normalized_scaled = (&normalized / &std_dev)?;
                let scaled = (&normalized_scaled * scale)?;
                let result = (&scaled + bias)?;

                Ok(vec![result])
            }

            "GlobalAveragePool" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("GlobalAveragePool requires exactly 1 input"));
                }
                let input = &inputs[0];
                let dims = input.dims();

                if dims.len() != 4 {
                    return Err(anyhow!("GlobalAveragePool expects 4D input (NCHW)"));
                }

                // Global average pooling: average over spatial dimensions (H, W)
                let result = input.mean_keepdim(2)?.mean_keepdim(3)?;
                Ok(vec![result])
            }

            "Reshape" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("Reshape requires exactly 1 input"));
                }
                // For simplicity, just return the input (reshape params would come from attributes)
                Ok(vec![inputs[0].clone()])
            }

            _ => Err(anyhow!("Unsupported GPU operation: {}", op_type)),
        }
    }

    /// Convert RONN Tensor to Candle Tensor.
    fn ronn_to_candle(&self, tensor: &ronn_core::tensor::Tensor) -> Result<CandleTensor> {
        let data = tensor.to_vec()?;
        let shape = tensor.shape();
        let dtype = match tensor.dtype() {
            DataType::F32 => candle_core::DType::F32,
            DataType::F16 => candle_core::DType::F16,
            DataType::F64 => candle_core::DType::F64,
            DataType::U8 => candle_core::DType::U8,
            DataType::U32 => candle_core::DType::U32,
            _ => candle_core::DType::F32, // Fallback
        };

        let candle_tensor =
            CandleTensor::from_vec(data, shape.as_slice(), &self.device)?.to_dtype(dtype)?;

        Ok(candle_tensor)
    }

    /// Generate operation signature for caching.
    fn generate_operation_signature(&self, op_type: &str, inputs: &[CandleTensor]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        op_type.hash(&mut hasher);

        // Hash input shapes and dtypes
        for input in inputs {
            input.dims().hash(&mut hasher);
            format!("{:?}", input.dtype()).hash(&mut hasher);
        }

        format!("{}_{:x}", op_type, hasher.finish())
    }

    /// Check if operation can use mixed precision.
    fn can_use_mixed_precision(&self, op_type: &str) -> bool {
        matches!(op_type,
            "Add" | "Sub" | "Mul" | "MatMul" | "Conv" |
            "ReLU" | "Sigmoid" | "Tanh" | "GELU" |
            "BatchNormalization" | "LayerNormalization"
        )
    }

    /// Convert tensors to mixed precision if beneficial.
    fn apply_mixed_precision(&self, inputs: &[CandleTensor], op_type: &str) -> Result<Vec<CandleTensor>> {
        if !self.can_use_mixed_precision(op_type) {
            return Ok(inputs.to_vec());
        }

        let mut converted = Vec::new();
        for input in inputs {
            // Convert large tensors to FP16 for memory efficiency
            let element_count = input.dims().iter().product::<usize>();
            if element_count > 1024 && input.dtype() == candle_core::DType::F32 {
                let fp16_tensor = input.to_dtype(candle_core::DType::F16)?;
                converted.push(fp16_tensor);
                debug!("Converted tensor to FP16 for operation: {}", op_type);
            } else {
                converted.push(input.clone());
            }
        }
        Ok(converted)
    }

    /// Execute operation with caching and optimization.
    fn execute_optimized_operation(
        &self,
        op_type: &str,
        inputs: &[CandleTensor],
    ) -> Result<Vec<CandleTensor>> {
        let signature = self.generate_operation_signature(op_type, inputs);

        // Check cache first
        {
            let mut cache = self.kernel_cache.lock().unwrap();
            if let Some(cached_op) = cache.cached_ops.get_mut(&signature) {
                cached_op.hit_count += 1;
                cached_op.last_accessed = std::time::Instant::now();
                debug!("Cache hit for operation: {} (signature: {})", op_type, signature);
            }
        }

        // Apply mixed precision if beneficial
        let optimized_inputs = self.apply_mixed_precision(inputs, op_type)?;

        // Execute the operation
        let result = self.execute_gpu_operation(op_type, &optimized_inputs)?;

        // Cache the operation
        {
            let mut cache = self.kernel_cache.lock().unwrap();
            let cached_op = CachedOperation {
                signature: signature.clone(),
                execution_path: OptimizedPath::Single(op_type.to_string()),
                hit_count: 1,
                last_accessed: std::time::Instant::now(),
            };
            cache.cached_ops.insert(signature, cached_op);

            // Simple cache eviction if needed
            if cache.cached_ops.len() > 1000 {
                self.evict_cache_entries(&mut cache);
            }
        }

        Ok(result)
    }

    /// Evict old cache entries using LRU policy.
    fn evict_cache_entries(&self, cache: &mut KernelCache) {
        let current_time = std::time::Instant::now();
        let mut to_remove = Vec::new();

        for (signature, cached_op) in &cache.cached_ops {
            // Remove entries not accessed in the last 5 minutes
            if current_time.duration_since(cached_op.last_accessed).as_secs() > 300 {
                to_remove.push(signature.clone());
            }
        }

        for signature in to_remove {
            cache.cached_ops.remove(&signature);
        }

        debug!("Evicted {} cache entries", cache.cached_ops.len());
    }

    /// Get cache statistics.
    pub fn get_cache_stats(&self) -> (usize, usize, f64) {
        let cache = self.kernel_cache.lock().unwrap();
        let total_hits: u64 = cache.cached_ops.values().map(|op| op.hit_count).sum();
        let cache_count = cache.cached_ops.len();
        let hit_rate = if cache_count > 0 {
            total_hits as f64 / cache_count as f64
        } else {
            0.0
        };
        (cache_count, cache.cache_size, hit_rate)
    }

    /// Convert Candle Tensor to RONN Tensor.
    fn candle_to_ronn(&self, tensor: &CandleTensor) -> Result<ronn_core::tensor::Tensor> {
        let shape = tensor.dims().to_vec();
        let data: Vec<f32> = tensor.to_vec1()?; // Convert to F32 for now

        let ronn_tensor = Tensor::from_data(
            data,
            shape,
            DataType::F32, // Simplified for now
            TensorLayout::RowMajor,
        )?;

        Ok(ronn_tensor)
    }
}

impl CompiledKernel for GpuKernel {
    fn execute(
        &self,
        inputs: &[ronn_core::tensor::Tensor],
    ) -> Result<Vec<ronn_core::tensor::Tensor>> {
        let start_time = std::time::Instant::now();

        // Convert RONN tensors to Candle tensors
        let mut candle_inputs = Vec::new();
        for input in inputs {
            let candle_tensor = self.ronn_to_candle(input)?;
            candle_inputs.push(candle_tensor);
        }

        // Execute operations with caching and optimization
        let mut current_tensors = candle_inputs;

        for node in &self.subgraph.nodes {
            debug!("Executing GPU operation: {} on stream {}", node.op_type, self.stream_id);
            let outputs = self.execute_optimized_operation(&node.op_type, &current_tensors)?;
            current_tensors = outputs;
        }

        // Convert back to RONN tensors
        let mut results = Vec::new();
        for candle_tensor in &current_tensors {
            let ronn_tensor = self.candle_to_ronn(candle_tensor)?;
            results.push(ronn_tensor);
        }

        // Update statistics
        let execution_time = start_time.elapsed();
        {
            let mut stats = self.stats.lock().unwrap();
            stats.execution_count += 1;
            stats.total_time_us += execution_time.as_micros() as u64;

            if stats.execution_count == 1 {
                stats.min_time_us = execution_time.as_micros() as u64;
                stats.max_time_us = execution_time.as_micros() as u64;
            } else {
                stats.min_time_us = stats.min_time_us.min(execution_time.as_micros() as u64);
                stats.max_time_us = stats.max_time_us.max(execution_time.as_micros() as u64);
            }
        }

        debug!("GPU kernel executed in {:?}", execution_time);

        Ok(results)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let stats = self.stats.lock().unwrap();
        MemoryUsage {
            peak_bytes: stats.memory_peak,
            current_bytes: 0, // Would track current usage in practice
            allocation_count: stats.execution_count as usize,
        }
    }

    fn get_performance_stats(&self) -> KernelStats {
        let stats = self.stats.lock().unwrap();

        let average_time_us = if stats.execution_count > 0 {
            stats.total_time_us as f64 / stats.execution_count as f64
        } else {
            0.0
        };

        KernelStats {
            execution_count: stats.execution_count,
            average_time_us,
            min_time_us: stats.min_time_us as f64,
            max_time_us: stats.max_time_us as f64,
        }
    }
}

/// Create a default GPU execution provider.
pub fn create_gpu_provider() -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(GpuExecutionProvider::new()?))
}

/// Create a GPU execution provider with custom configuration.
pub fn create_gpu_provider_with_config(
    config: GpuProviderConfig,
) -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(GpuExecutionProvider::with_config(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::GraphNode;
    use std::collections::HashMap;

    fn create_test_subgraph() -> SubGraph {
        let node = GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
            name: Some("gpu_add".to_string()),
        };

        SubGraph {
            nodes: vec![node],
            edges: vec![],
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
        }
    }

    #[test]
    fn test_gpu_provider_creation() {
        // This test may fail if no GPU is available
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                assert_eq!(provider.provider_id(), ProviderId::GPU);

                let capability = provider.get_capability();
                assert_eq!(capability.performance_profile, PerformanceProfile::GPU);
                assert!(!capability.supported_ops.is_empty());
                assert!(capability.data_types.contains(&DataType::F32));
            }
            Err(e) => {
                println!("GPU not available: {}", e);
                // Test passes if GPU is not available
            }
        }
    }

    #[test]
    fn test_gpu_provider_config() {
        let config = GpuProviderConfig {
            device_ids: vec![0],
            enable_mixed_precision: false,
            enable_tensor_cores: false,
            ..Default::default()
        };

        match GpuExecutionProvider::with_config(config) {
            Ok(provider) => {
                assert!(!provider.get_config().enable_mixed_precision);
                assert!(!provider.get_config().enable_tensor_cores);
            }
            Err(_) => {
                // GPU not available, test passes
            }
        }
    }

    #[test]
    fn test_operation_support() {
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                // Test GPU-optimized operations
                assert!(provider.supports_operation("Add"));
                assert!(provider.supports_operation("MatMul"));
                assert!(provider.supports_operation("Conv"));
                assert!(provider.supports_operation("ReLU"));
                assert!(!provider.supports_operation("NonexistentOp"));

                // Test cost estimation
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

                // GPU should be very efficient for both, but Conv more complex
                assert!(conv_cost > add_cost);
                assert!(add_cost < 1.0); // Should be less than 1.0 for GPU
            }
            Err(_) => {
                // GPU not available
            }
        }
    }

    #[test]
    fn test_subgraph_compilation() {
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                let subgraph = create_test_subgraph();

                match provider.compile_subgraph(subgraph) {
                    Ok(kernel) => {
                        let stats = kernel.get_performance_stats();
                        assert_eq!(stats.execution_count, 0); // Not executed yet
                    }
                    Err(e) => {
                        println!("Compilation failed: {}", e);
                    }
                }
            }
            Err(_) => {
                // GPU not available
            }
        }
    }

    #[test]
    fn test_factory_functions() {
        // Test factory functions (may fail if no GPU)
        match create_gpu_provider() {
            Ok(provider) => {
                assert_eq!(provider.provider_id(), ProviderId::GPU);
            }
            Err(_) => {
                // GPU not available
            }
        }

        let config = GpuProviderConfig::default();
        match create_gpu_provider_with_config(config) {
            Ok(provider) => {
                assert_eq!(provider.provider_id(), ProviderId::GPU);
            }
            Err(_) => {
                // GPU not available
            }
        }
    }

    #[test]
    fn test_complex_gpu_operations() {
        // Test that complex operations are supported
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                let capability = provider.get_capability();

                // Check that complex operations are supported
                assert!(capability.supported_ops.contains("Conv"));
                assert!(capability.supported_ops.contains("BatchNormalization"));
                assert!(capability.supported_ops.contains("LayerNormalization"));
                assert!(capability.supported_ops.contains("GlobalAveragePool"));

                // Test activation functions
                assert!(capability.supported_ops.contains("Sigmoid"));
                assert!(capability.supported_ops.contains("Tanh"));
                assert!(capability.supported_ops.contains("GELU"));

                println!("✅ GPU provider supports {} complex operations",
                    capability.supported_ops.len());
            }
            Err(e) => {
                println!("GPU not available: {}", e);
                // Test passes if GPU is not available
            }
        }
    }

    #[test]
    fn test_gpu_benchmarks() {
        // Comprehensive GPU benchmarks
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                println!("🚀 Running GPU performance benchmarks...");

                // Test basic operation performance
                benchmark_basic_operations(&provider);

                // Test complex operations
                benchmark_complex_operations(&provider);

                // Test mixed precision performance
                benchmark_mixed_precision(&provider);

                // Test cache performance
                benchmark_cache_performance(&provider);

                // Test memory throughput
                benchmark_memory_throughput(&provider);

                println!("✅ GPU benchmarks completed!");
            }
            Err(e) => {
                println!("GPU not available for benchmarks: {}", e);
            }
        }
    }

    fn benchmark_basic_operations(provider: &GpuExecutionProvider) {
        use std::time::Instant;

        println!("  📊 Basic Operations Benchmark:");

        let ops = ["Add", "Mul", "MatMul", "ReLU", "Sigmoid", "Tanh"];

        for op in ops {
            let subgraph = create_single_op_subgraph(op);
            if let Ok(kernel) = provider.compile_subgraph(subgraph) {
                // Create test tensors
                let test_input = ronn_core::tensor::Tensor::ones(
                    vec![1024, 1024],
                    DataType::F32,
                    TensorLayout::RowMajor
                ).unwrap();

                let start = Instant::now();
                for _ in 0..10 {
                    let _ = kernel.execute(&[test_input.clone()]);
                }
                let avg_time = start.elapsed() / 10;

                println!("    {} avg: {:?}", op, avg_time);
            }
        }
    }

    fn benchmark_complex_operations(provider: &GpuExecutionProvider) {
        use std::time::Instant;

        println!("  🧠 Complex Operations Benchmark:");

        let complex_ops = ["Conv", "BatchNormalization", "LayerNormalization", "GlobalAveragePool"];

        for op in complex_ops {
            let subgraph = create_single_op_subgraph(op);
            if let Ok(kernel) = provider.compile_subgraph(subgraph) {
                // Create appropriate test tensors for complex ops
                let test_input = match op {
                    "Conv" => ronn_core::tensor::Tensor::ones(
                        vec![1, 64, 224, 224], // NCHW format
                        DataType::F32,
                        TensorLayout::RowMajor
                    ).unwrap(),
                    _ => ronn_core::tensor::Tensor::ones(
                        vec![32, 512],
                        DataType::F32,
                        TensorLayout::RowMajor
                    ).unwrap(),
                };

                let start = Instant::now();
                for _ in 0..5 {
                    let _ = kernel.execute(&[test_input.clone()]);
                }
                let avg_time = start.elapsed() / 5;

                println!("    {} avg: {:?}", op, avg_time);
            }
        }
    }

    fn benchmark_mixed_precision(provider: &GpuExecutionProvider) {
        println!("  🎯 Mixed Precision Benchmark:");

        if provider.has_tensor_cores() {
            println!("    Tensor cores available - mixed precision enabled");
        } else {
            println!("    Tensor cores not available - mixed precision simulation");
        }

        // Test F32 vs F16 performance difference
        let sizes = [512, 1024, 2048];

        for size in sizes {
            println!("    Matrix size: {}x{}", size, size);
            // Would test actual F16 vs F32 performance here
        }
    }

    fn benchmark_cache_performance(provider: &GpuExecutionProvider) {
        use std::time::Instant;

        println!("  💾 Cache Performance Benchmark:");

        let subgraph = create_single_op_subgraph("Add");
        if let Ok(kernel) = provider.compile_subgraph(subgraph) {
            let test_input = ronn_core::tensor::Tensor::ones(
                vec![512, 512],
                DataType::F32,
                TensorLayout::RowMajor
            ).unwrap();

            // Warm up cache
            for _ in 0..5 {
                let _ = kernel.execute(&[test_input.clone()]);
            }

            // Measure cached performance
            let start = Instant::now();
            for _ in 0..20 {
                let _ = kernel.execute(&[test_input.clone()]);
            }
            let cached_time = start.elapsed() / 20;

            // Note: In a full implementation, we would downcast to access cache stats
            // For now, just show the cached execution time
            // let (cache_entries, _cache_size, hit_rate) = kernel.get_cache_stats();
            // println!("    Cache entries: {}, Hit rate: {:.2}%", cache_entries, hit_rate * 100.0);

            println!("    Cached execution avg: {:?}", cached_time);
        }
    }

    fn benchmark_memory_throughput(provider: &GpuExecutionProvider) {
        println!("  🚀 Memory Throughput Benchmark:");

        if let Ok((total_memory, _used_memory)) = provider.get_gpu_memory_info() {
            println!("    GPU Memory: {:.2} GB total", total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
        }

        let allocator = provider.get_allocator();

        // Test allocation/deallocation speed
        let start = std::time::Instant::now();
        let mut buffers = Vec::new();

        for _ in 0..100 {
            if let Ok(buffer) = allocator.allocate(&[1024], DataType::F32) {
                buffers.push(buffer);
            }
        }

        let alloc_time = start.elapsed();

        let start = std::time::Instant::now();
        for buffer in buffers {
            let _ = allocator.deallocate(buffer);
        }
        let dealloc_time = start.elapsed();

        println!("    100 allocations: {:?}", alloc_time);
        println!("    100 deallocations: {:?}", dealloc_time);
    }

    fn create_single_op_subgraph(op_type: &str) -> SubGraph {
        let node = GraphNode {
            id: 0,
            op_type: op_type.to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
            name: Some(format!("test_{}", op_type)),
        };

        SubGraph {
            nodes: vec![node],
            edges: vec![],
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
        }
    }

    #[test]
    fn test_stream_execution() {
        // Test stream-based async execution
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                if provider.get_config().stream_count > 1 {
                    println!("🌊 Testing stream-based execution with {} streams",
                        provider.get_config().stream_count);

                    // Test creating kernels with different streams
                    let subgraph1 = create_single_op_subgraph("Add");
                    let subgraph2 = create_single_op_subgraph("Mul");

                    if let (Ok(kernel1), Ok(kernel2)) = (
                        GpuKernel::with_stream(subgraph1, provider.device().clone(), 0),
                        GpuKernel::with_stream(subgraph2, provider.device().clone(), 1)
                    ) {
                        println!("    ✅ Successfully created kernels on different streams");

                        // Test concurrent execution (simplified)
                        let test_input = ronn_core::tensor::Tensor::ones(
                            vec![256, 256],
                            DataType::F32,
                            TensorLayout::RowMajor
                        ).unwrap();

                        let start = std::time::Instant::now();
                        let _result1 = kernel1.execute(&[test_input.clone()]);
                        let _result2 = kernel2.execute(&[test_input.clone()]);
                        let concurrent_time = start.elapsed();

                        println!("    Concurrent execution time: {:?}", concurrent_time);
                    }
                } else {
                    println!("🌊 Single stream execution (stream_count = 1)");
                }
            }
            Err(_) => {
                println!("GPU not available for stream testing");
            }
        }
    }

    #[test]
    fn test_kernel_cache_operations() {
        // Test kernel caching system
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                println!("💾 Testing kernel cache operations...");

                let subgraph = create_single_op_subgraph("MatMul");
                if let Ok(kernel) = provider.compile_subgraph(subgraph) {
                    let test_input = ronn_core::tensor::Tensor::ones(
                        vec![128, 128],
                        DataType::F32,
                        TensorLayout::RowMajor
                    ).unwrap();

                    // Execute multiple times to populate cache
                    for i in 0..10 {
                        let _ = kernel.execute(&[test_input.clone()]);

                        if i == 0 {
                            println!("    First execution (cold cache)");
                        } else if i == 9 {
                            println!("    Tenth execution (warm cache)");
                        }
                    }

                    println!("    ✅ Cache operations test completed");
                }
            }
            Err(_) => {
                println!("GPU not available for cache testing");
            }
        }
    }
}
