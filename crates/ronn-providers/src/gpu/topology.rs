//! GPU topology detection and optimal workload placement.
//!
//! This module provides intelligent GPU topology discovery, interconnect analysis,
//! and optimal workload placement strategies for multi-GPU systems.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// GPU topology analyzer and workload placement optimizer.
#[derive(Debug)]
pub struct GpuTopologyManager {
    /// Discovered GPU topology information.
    topology: Arc<RwLock<GpuTopology>>,
    /// Workload placement strategies.
    placement_strategies: HashMap<String, Box<dyn PlacementStrategy + Send + Sync>>,
    /// Performance profiler for placement decisions.
    profiler: Arc<TopologyProfiler>,
    /// Configuration for topology detection.
    config: TopologyConfig,
}

/// Configuration for GPU topology detection and placement.
#[derive(Debug, Clone)]
pub struct TopologyConfig {
    /// Enable automatic topology discovery.
    pub auto_discovery: bool,
    /// Benchmark topology links for accurate measurements.
    pub benchmark_links: bool,
    /// Cache topology information to disk.
    pub cache_topology: bool,
    /// Minimum benchmark duration in milliseconds.
    pub benchmark_duration_ms: u64,
    /// Number of benchmark iterations for averaging.
    pub benchmark_iterations: usize,
    /// Enable NUMA topology consideration.
    pub consider_numa: bool,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            benchmark_links: true,
            cache_topology: true,
            benchmark_duration_ms: 100,
            benchmark_iterations: 5,
            consider_numa: true,
        }
    }
}

/// Complete GPU system topology information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTopology {
    /// Individual GPU devices in the system.
    pub devices: HashMap<usize, GpuDeviceInfo>,
    /// Interconnect links between GPUs.
    pub links: HashMap<(usize, usize), InterconnectLink>,
    /// NUMA topology information.
    pub numa_topology: Option<NumaTopology>,
    /// System-level characteristics.
    pub system_info: SystemInfo,
    /// Topology discovery timestamp.
    pub discovery_timestamp: std::time::SystemTime,
}

/// Information about an individual GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device ID.
    pub device_id: usize,
    /// GPU architecture (e.g., "Ampere", "Ada Lovelace").
    pub architecture: String,
    /// Compute capability (e.g., "8.6").
    pub compute_capability: String,
    /// Total memory in bytes.
    pub total_memory: usize,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth: f64,
    /// Number of streaming multiprocessors.
    pub sm_count: usize,
    /// Base clock frequency in MHz.
    pub base_clock_mhz: u32,
    /// Boost clock frequency in MHz.
    pub boost_clock_mhz: u32,
    /// Power consumption in watts.
    pub power_limit_watts: u32,
    /// PCI bus information.
    pub pci_info: PciInfo,
    /// NUMA node affinity.
    pub numa_node: Option<usize>,
    /// Device capabilities.
    pub capabilities: DeviceCapabilities,
}

/// PCI bus information for a GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PciInfo {
    /// PCI domain.
    pub domain: u16,
    /// PCI bus number.
    pub bus: u8,
    /// PCI device number.
    pub device: u8,
    /// PCI function number.
    pub function: u8,
    /// PCI device ID.
    pub device_id: u32,
    /// PCI vendor ID.
    pub vendor_id: u32,
}

/// GPU device capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Supports P2P memory access.
    pub p2p_supported: bool,
    /// Supports unified memory.
    pub unified_memory: bool,
    /// Supports cooperative kernels.
    pub cooperative_kernels: bool,
    /// Supports tensor cores.
    pub tensor_cores: bool,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Maximum grid dimensions.
    pub max_grid_dims: [u32; 3],
    /// Shared memory per block in bytes.
    pub shared_memory_per_block: usize,
}

/// Interconnect link between two GPUs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterconnectLink {
    /// Source device ID.
    pub src_device: usize,
    /// Destination device ID.
    pub dst_device: usize,
    /// Type of interconnect.
    pub link_type: InterconnectType,
    /// Bidirectional bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Latency in microseconds.
    pub latency_us: f64,
    /// Number of links/lanes.
    pub link_count: usize,
    /// Link utilization (0.0 to 1.0).
    pub utilization: f32,
    /// Quality score (higher is better).
    pub quality_score: f32,
}

/// Types of GPU interconnects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterconnectType {
    /// NVIDIA NVLink (high-speed direct connection).
    NVLink,
    /// PCIe connection.
    PCIe,
    /// System memory path (slowest).
    SystemMemory,
    /// NVIDIA NVSwitch (multi-GPU fabric).
    NVSwitch,
    /// AMD Infinity Fabric.
    InfinityFabric,
}

/// NUMA topology information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    /// Number of NUMA nodes.
    pub node_count: usize,
    /// GPU to NUMA node mapping.
    pub gpu_numa_map: HashMap<usize, usize>,
    /// CPU to NUMA node mapping.
    pub cpu_numa_map: HashMap<usize, usize>,
    /// Inter-NUMA node distances.
    pub numa_distances: HashMap<(usize, usize), f32>,
}

/// System-level information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Total number of GPUs.
    pub gpu_count: usize,
    /// CPU information.
    pub cpu_info: String,
    /// Total system memory in bytes.
    pub system_memory: usize,
    /// Operating system.
    pub os_info: String,
    /// CUDA driver version.
    pub cuda_version: Option<String>,
}

/// Workload placement strategy trait.
pub trait PlacementStrategy: std::fmt::Debug {
    /// Get strategy name.
    fn name(&self) -> &str;

    /// Optimize device placement for a given workload.
    fn optimize_placement(
        &self,
        workload: &Workload,
        topology: &GpuTopology,
    ) -> Result<PlacementPlan>;

    /// Estimate performance for a placement plan.
    fn estimate_performance(
        &self,
        plan: &PlacementPlan,
        workload: &Workload,
        topology: &GpuTopology,
    ) -> Result<PerformanceEstimate>;
}

/// Workload description for placement optimization.
#[derive(Debug, Clone)]
pub struct Workload {
    /// Workload identifier.
    pub id: String,
    /// Type of workload.
    pub workload_type: WorkloadType,
    /// Computational requirements.
    pub compute_requirements: ComputeRequirements,
    /// Memory access patterns.
    pub memory_patterns: Vec<MemoryPattern>,
    /// Communication patterns between components.
    pub communication_patterns: Vec<CommunicationPattern>,
    /// Performance constraints.
    pub constraints: WorkloadConstraints,
}

/// Types of GPU workloads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkloadType {
    /// Training neural networks.
    Training,
    /// Inference/prediction.
    Inference,
    /// High-performance computing.
    HPC,
    /// Data processing.
    DataProcessing,
    /// Graphics rendering.
    Graphics,
    /// General compute.
    Compute,
}

/// Computational requirements for a workload.
#[derive(Debug, Clone)]
pub struct ComputeRequirements {
    /// Required compute capability.
    pub min_compute_capability: String,
    /// Memory requirement per device in bytes.
    pub memory_per_device: usize,
    /// Preferred number of devices.
    pub preferred_device_count: usize,
    /// Whether tensor cores are beneficial.
    pub benefits_from_tensor_cores: bool,
    /// FP16 vs FP32 preference.
    pub precision_preference: PrecisionPreference,
}

/// Precision preferences for computations.
#[derive(Debug, Clone, Copy)]
pub enum PrecisionPreference {
    /// Prefers FP32 for accuracy.
    FP32,
    /// Prefers FP16 for speed.
    FP16,
    /// Mixed precision.
    Mixed,
    /// No preference.
    Any,
}

/// Memory access pattern description.
#[derive(Debug, Clone)]
pub struct MemoryPattern {
    /// Pattern type.
    pub pattern_type: MemoryAccessType,
    /// Data size in bytes.
    pub data_size: usize,
    /// Access frequency.
    pub frequency: f32,
    /// Whether data is shared across devices.
    pub shared_across_devices: bool,
}

/// Types of memory access patterns.
#[derive(Debug, Clone, Copy)]
pub enum MemoryAccessType {
    /// Sequential access.
    Sequential,
    /// Random access.
    Random,
    /// Stride access.
    Strided,
    /// Broadcast (one-to-many).
    Broadcast,
    /// Reduction (many-to-one).
    Reduction,
}

/// Communication pattern between workload components.
#[derive(Debug, Clone)]
pub struct CommunicationPattern {
    /// Source device preference.
    pub src_device: Option<usize>,
    /// Destination device preference.
    pub dst_device: Option<usize>,
    /// Communication type.
    pub comm_type: CommunicationType,
    /// Data volume in bytes.
    pub data_volume: usize,
    /// Communication frequency.
    pub frequency: f32,
}

/// Types of inter-device communication.
#[derive(Debug, Clone, Copy)]
pub enum CommunicationType {
    /// Point-to-point transfer.
    P2P,
    /// All-to-all communication.
    AllToAll,
    /// All-reduce operation.
    AllReduce,
    /// Broadcast operation.
    Broadcast,
    /// Scatter operation.
    Scatter,
    /// Gather operation.
    Gather,
}

/// Workload performance constraints.
#[derive(Debug, Clone)]
pub struct WorkloadConstraints {
    /// Maximum acceptable latency in milliseconds.
    pub max_latency_ms: Option<f64>,
    /// Minimum required throughput.
    pub min_throughput: Option<f64>,
    /// Power budget in watts.
    pub power_budget: Option<f32>,
    /// Required fault tolerance.
    pub fault_tolerance: bool,
}

/// Device placement plan.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    /// Device assignments for workload components.
    pub device_assignments: HashMap<String, usize>,
    /// Expected performance characteristics.
    pub performance_estimate: PerformanceEstimate,
    /// Resource utilization plan.
    pub resource_utilization: ResourceUtilization,
    /// Communication plan between devices.
    pub communication_plan: CommunicationPlan,
}

/// Performance estimate for a placement plan.
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    /// Expected execution time in milliseconds.
    pub execution_time_ms: f64,
    /// Expected throughput (operations/second).
    pub throughput: f64,
    /// Memory bandwidth utilization.
    pub memory_bandwidth_util: f32,
    /// Compute utilization across devices.
    pub compute_utilization: Vec<f32>,
    /// Communication overhead percentage.
    pub communication_overhead: f32,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

/// Resource utilization breakdown.
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Memory usage per device.
    pub memory_usage: HashMap<usize, usize>,
    /// Compute utilization per device.
    pub compute_utilization: HashMap<usize, f32>,
    /// Bandwidth utilization per link.
    pub bandwidth_utilization: HashMap<(usize, usize), f32>,
    /// Power consumption estimate.
    pub power_consumption: f32,
}

/// Communication plan between devices.
#[derive(Debug, Clone)]
pub struct CommunicationPlan {
    /// Communication routes between device pairs.
    pub routes: HashMap<(usize, usize), CommunicationRoute>,
    /// Expected communication volume per route.
    pub volume_per_route: HashMap<(usize, usize), usize>,
    /// Load balancing across links.
    pub load_balancing: HashMap<(usize, usize), f32>,
}

/// Communication route between two devices.
#[derive(Debug, Clone)]
pub struct CommunicationRoute {
    /// Source device.
    pub src: usize,
    /// Destination device.
    pub dst: usize,
    /// Intermediate hops (if any).
    pub hops: Vec<usize>,
    /// Route quality score.
    pub quality: f32,
    /// Expected latency.
    pub latency_us: f64,
    /// Available bandwidth.
    pub bandwidth_gbps: f64,
}

/// Topology profiler for benchmarking and performance modeling.
#[derive(Debug)]
pub struct TopologyProfiler {
    /// Bandwidth measurements between device pairs.
    bandwidth_cache: RwLock<HashMap<(usize, usize), f64>>,
    /// Latency measurements between device pairs.
    latency_cache: RwLock<HashMap<(usize, usize), f64>>,
    /// Compute performance profiles per device.
    compute_profiles: RwLock<HashMap<usize, ComputeProfile>>,
}

/// Compute performance profile for a device.
#[derive(Debug, Clone)]
pub struct ComputeProfile {
    /// Device ID.
    pub device_id: usize,
    /// FP32 peak performance in GFLOPS.
    pub fp32_gflops: f64,
    /// FP16 peak performance in GFLOPS.
    pub fp16_gflops: f64,
    /// Tensor core performance in TOPS.
    pub tensor_tops: f64,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth: f64,
}

impl GpuTopologyManager {
    /// Create a new GPU topology manager.
    pub fn new(config: TopologyConfig) -> Result<Self> {
        let topology = Arc::new(RwLock::new(GpuTopology {
            devices: HashMap::new(),
            links: HashMap::new(),
            numa_topology: None,
            system_info: SystemInfo {
                gpu_count: 0,
                cpu_info: "Unknown".to_string(),
                system_memory: 0,
                os_info: std::env::consts::OS.to_string(),
                cuda_version: None,
            },
            discovery_timestamp: std::time::SystemTime::now(),
        }));

        let profiler = Arc::new(TopologyProfiler::new());
        let mut placement_strategies = HashMap::new();

        // Register default placement strategies
        placement_strategies.insert(
            "locality_aware".to_string(),
            Box::new(LocalityAwarePlacement::new()) as Box<dyn PlacementStrategy + Send + Sync>,
        );
        placement_strategies.insert(
            "bandwidth_optimized".to_string(),
            Box::new(BandwidthOptimizedPlacement::new())
                as Box<dyn PlacementStrategy + Send + Sync>,
        );
        placement_strategies.insert(
            "power_efficient".to_string(),
            Box::new(PowerEfficientPlacement::new()) as Box<dyn PlacementStrategy + Send + Sync>,
        );

        Ok(Self {
            topology,
            placement_strategies,
            profiler,
            config,
        })
    }

    /// Discover and analyze GPU topology.
    pub fn discover_topology(&self) -> Result<()> {
        info!("Starting GPU topology discovery");

        let mut topology = self.topology.write().unwrap();

        // Simulate GPU device discovery
        let devices = self.discover_gpu_devices()?;
        topology.devices = devices;

        // Discover interconnect links
        let links = self.discover_interconnect_links(&topology.devices)?;
        topology.links = links;

        // Discover NUMA topology if enabled
        if self.config.consider_numa {
            topology.numa_topology = self.discover_numa_topology(&topology.devices)?;
        }

        // Update system info
        topology.system_info.gpu_count = topology.devices.len();
        topology.discovery_timestamp = std::time::SystemTime::now();

        info!(
            "Topology discovery completed: {} GPUs, {} links",
            topology.devices.len(),
            topology.links.len()
        );

        // Benchmark links if enabled
        if self.config.benchmark_links {
            drop(topology); // Release write lock
            self.benchmark_topology()?;
        }

        Ok(())
    }

    /// Get current topology information.
    pub fn get_topology(&self) -> GpuTopology {
        self.topology.read().unwrap().clone()
    }

    /// Optimize workload placement using specified strategy.
    pub fn optimize_placement(
        &self,
        workload: &Workload,
        strategy_name: &str,
    ) -> Result<PlacementPlan> {
        let topology = self.topology.read().unwrap();

        let strategy = self
            .placement_strategies
            .get(strategy_name)
            .ok_or_else(|| anyhow!("Unknown placement strategy: {}", strategy_name))?;

        debug!(
            "Optimizing placement for workload '{}' using strategy '{}'",
            workload.id, strategy_name
        );

        let plan = strategy.optimize_placement(workload, &topology)?;

        info!(
            "Generated placement plan for workload '{}': {} device assignments",
            workload.id,
            plan.device_assignments.len()
        );

        Ok(plan)
    }

    /// Compare multiple placement strategies for a workload.
    pub fn compare_strategies(
        &self,
        workload: &Workload,
        strategies: &[String],
    ) -> Result<Vec<(String, PlacementPlan)>> {
        let topology = self.topology.read().unwrap();
        let mut results = Vec::new();

        for strategy_name in strategies {
            if let Some(strategy) = self.placement_strategies.get(strategy_name) {
                match strategy.optimize_placement(workload, &topology) {
                    Ok(plan) => results.push((strategy_name.clone(), plan)),
                    Err(e) => warn!("Strategy '{}' failed: {}", strategy_name, e),
                }
            }
        }

        Ok(results)
    }

    /// Get available placement strategies.
    pub fn get_available_strategies(&self) -> Vec<String> {
        self.placement_strategies.keys().cloned().collect()
    }

    /// Register a custom placement strategy.
    pub fn register_strategy(
        &mut self,
        name: String,
        strategy: Box<dyn PlacementStrategy + Send + Sync>,
    ) {
        self.placement_strategies.insert(name, strategy);
    }

    // Private implementation methods

    fn discover_gpu_devices(&self) -> Result<HashMap<usize, GpuDeviceInfo>> {
        let mut devices = HashMap::new();

        // Simulate discovery of GPU devices
        for device_id in 0..4 {
            // Simulate 4 GPUs
            let device_info = GpuDeviceInfo {
                device_id,
                architecture: if device_id < 2 {
                    "Ampere".to_string()
                } else {
                    "Ada Lovelace".to_string()
                },
                compute_capability: if device_id < 2 {
                    "8.0".to_string()
                } else {
                    "8.9".to_string()
                },
                total_memory: 40 * 1024 * 1024 * 1024, // 40GB
                memory_bandwidth: if device_id < 2 { 1555.0 } else { 1008.0 },
                sm_count: if device_id < 2 { 108 } else { 128 },
                base_clock_mhz: if device_id < 2 { 1410 } else { 2230 },
                boost_clock_mhz: if device_id < 2 { 1695 } else { 2520 },
                power_limit_watts: if device_id < 2 { 400 } else { 450 },
                pci_info: PciInfo {
                    domain: 0,
                    bus: (device_id * 16) as u8,
                    device: 0,
                    function: 0,
                    device_id: if device_id < 2 { 0x20B0 } else { 0x2684 },
                    vendor_id: 0x10DE, // NVIDIA
                },
                numa_node: Some(device_id / 2),
                capabilities: DeviceCapabilities {
                    p2p_supported: true,
                    unified_memory: true,
                    cooperative_kernels: true,
                    tensor_cores: true,
                    max_threads_per_block: 1024,
                    max_grid_dims: [2147483647, 65535, 65535],
                    shared_memory_per_block: 49152,
                },
            };
            devices.insert(device_id, device_info);
        }

        Ok(devices)
    }

    fn discover_interconnect_links(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
    ) -> Result<HashMap<(usize, usize), InterconnectLink>> {
        let mut links = HashMap::new();

        for (&src_id, _) in devices {
            for (&dst_id, _) in devices {
                if src_id != dst_id {
                    let (link_type, bandwidth, latency) =
                        self.determine_link_characteristics(src_id, dst_id);

                    let link = InterconnectLink {
                        src_device: src_id,
                        dst_device: dst_id,
                        link_type,
                        bandwidth_gbps: bandwidth,
                        latency_us: latency,
                        link_count: if link_type == InterconnectType::NVLink {
                            4
                        } else {
                            1
                        },
                        utilization: 0.0,
                        quality_score: self.calculate_link_quality(link_type, bandwidth, latency),
                    };

                    links.insert((src_id, dst_id), link);
                }
            }
        }

        Ok(links)
    }

    fn determine_link_characteristics(
        &self,
        src: usize,
        dst: usize,
    ) -> (InterconnectType, f64, f64) {
        // Simulate different link types based on device proximity
        let distance = src.abs_diff(dst);

        match distance {
            1 => (InterconnectType::NVLink, 600.0, 1.0), // Adjacent devices with NVLink
            2 => (InterconnectType::NVSwitch, 600.0, 2.0), // Through NVSwitch
            3 => (InterconnectType::PCIe, 64.0, 5.0),    // PCIe Gen4 x16
            _ => (InterconnectType::SystemMemory, 12.8, 10.0), // Through system memory
        }
    }

    fn calculate_link_quality(
        &self,
        link_type: InterconnectType,
        bandwidth: f64,
        latency: f64,
    ) -> f32 {
        let base_score = match link_type {
            InterconnectType::NVLink => 1.0,
            InterconnectType::NVSwitch => 0.95,
            InterconnectType::InfinityFabric => 0.9,
            InterconnectType::PCIe => 0.5,
            InterconnectType::SystemMemory => 0.1,
        };

        let bandwidth_factor = (bandwidth / 600.0).min(1.0) as f32;
        let latency_factor = (10.0 / latency).min(1.0) as f32;

        base_score * bandwidth_factor * latency_factor
    }

    fn discover_numa_topology(
        &self,
        devices: &HashMap<usize, GpuDeviceInfo>,
    ) -> Result<Option<NumaTopology>> {
        let mut gpu_numa_map = HashMap::new();
        let mut numa_distances = HashMap::new();

        // Assign GPUs to NUMA nodes
        for (&device_id, device_info) in devices {
            if let Some(numa_node) = device_info.numa_node {
                gpu_numa_map.insert(device_id, numa_node);
            }
        }

        // Simulate NUMA distances
        for node_a in 0..2 {
            for node_b in 0..2 {
                let distance = if node_a == node_b { 1.0 } else { 2.1 };
                numa_distances.insert((node_a, node_b), distance);
            }
        }

        Ok(Some(NumaTopology {
            node_count: 2,
            gpu_numa_map,
            cpu_numa_map: HashMap::new(), // Would be populated in real implementation
            numa_distances,
        }))
    }

    fn benchmark_topology(&self) -> Result<()> {
        info!("Benchmarking GPU topology links");

        let topology = self.topology.read().unwrap();

        for (&(src, dst), _link) in &topology.links {
            if src < dst {
                // Only benchmark each pair once
                match self.benchmark_link(src, dst, &topology) {
                    Ok((bandwidth, latency)) => {
                        self.profiler
                            .update_link_performance(src, dst, bandwidth, latency);
                        debug!(
                            "Benchmarked link {}->{}: {:.1} GB/s, {:.1} μs",
                            src, dst, bandwidth, latency
                        );
                    }
                    Err(e) => warn!("Failed to benchmark link {}->{}: {}", src, dst, e),
                }
            }
        }

        info!("Topology benchmarking completed");
        Ok(())
    }

    fn benchmark_link(
        &self,
        src: usize,
        dst: usize,
        _topology: &GpuTopology,
    ) -> Result<(f64, f64)> {
        // Simulate benchmarking with some variability
        let base_latency = 1.0 + (src.abs_diff(dst) as f64) * 0.5;
        let latency = base_latency * (0.9 + 0.2 * fastrand::f64());

        let base_bandwidth = if src.abs_diff(dst) == 1 { 600.0 } else { 300.0 };
        let bandwidth = base_bandwidth * (0.85 + 0.15 * fastrand::f64());

        Ok((bandwidth, latency))
    }
}

impl TopologyProfiler {
    fn new() -> Self {
        Self {
            bandwidth_cache: RwLock::new(HashMap::new()),
            latency_cache: RwLock::new(HashMap::new()),
            compute_profiles: RwLock::new(HashMap::new()),
        }
    }

    fn update_link_performance(&self, src: usize, dst: usize, bandwidth: f64, latency: f64) {
        {
            let mut bandwidth_cache = self.bandwidth_cache.write().unwrap();
            bandwidth_cache.insert((src, dst), bandwidth);
            bandwidth_cache.insert((dst, src), bandwidth); // Bidirectional
        }

        {
            let mut latency_cache = self.latency_cache.write().unwrap();
            latency_cache.insert((src, dst), latency);
            latency_cache.insert((dst, src), latency); // Bidirectional
        }
    }
}

// Placement strategy implementations

/// Locality-aware placement strategy.
#[derive(Debug)]
pub struct LocalityAwarePlacement;

impl LocalityAwarePlacement {
    /// Create a new locality-aware placement strategy.
    pub fn new() -> Self {
        Self
    }
}

impl PlacementStrategy for LocalityAwarePlacement {
    fn name(&self) -> &str {
        "locality_aware"
    }

    fn optimize_placement(
        &self,
        workload: &Workload,
        topology: &GpuTopology,
    ) -> Result<PlacementPlan> {
        debug!("Optimizing placement using locality-aware strategy");

        let device_count = workload
            .compute_requirements
            .preferred_device_count
            .min(topology.devices.len());
        let mut device_assignments = HashMap::new();

        // Select devices with best interconnectivity
        let mut selected_devices = Vec::new();
        let mut available_devices: Vec<_> = topology.devices.keys().copied().collect();

        if !available_devices.is_empty() {
            // Start with first available device
            let first_device = available_devices[0];
            selected_devices.push(first_device);
            available_devices.retain(|&x| x != first_device);

            // Select remaining devices based on connectivity to already selected ones
            while selected_devices.len() < device_count && !available_devices.is_empty() {
                let next_device = self.find_best_connected_device(
                    &selected_devices,
                    &available_devices,
                    topology,
                );
                selected_devices.push(next_device);
                available_devices.retain(|&x| x != next_device);
            }
        }

        // Assign workload components to selected devices
        for (i, &device_id) in selected_devices.iter().enumerate() {
            device_assignments.insert(format!("component_{}", i), device_id);
        }

        let performance_estimate =
            self.estimate_performance_internal(workload, &selected_devices, topology)?;

        Ok(PlacementPlan {
            device_assignments,
            performance_estimate: performance_estimate.clone(),
            resource_utilization: ResourceUtilization {
                memory_usage: selected_devices
                    .iter()
                    .map(|&id| (id, workload.compute_requirements.memory_per_device))
                    .collect(),
                compute_utilization: selected_devices.iter().map(|&id| (id, 0.8)).collect(),
                bandwidth_utilization: HashMap::new(),
                power_consumption: selected_devices.len() as f32 * 300.0,
            },
            communication_plan: CommunicationPlan {
                routes: HashMap::new(),
                volume_per_route: HashMap::new(),
                load_balancing: HashMap::new(),
            },
        })
    }

    fn estimate_performance(
        &self,
        plan: &PlacementPlan,
        workload: &Workload,
        topology: &GpuTopology,
    ) -> Result<PerformanceEstimate> {
        let devices: Vec<usize> = plan.device_assignments.values().copied().collect();
        self.estimate_performance_internal(workload, &devices, topology)
    }
}

impl LocalityAwarePlacement {
    fn find_best_connected_device(
        &self,
        selected: &[usize],
        available: &[usize],
        topology: &GpuTopology,
    ) -> usize {
        let mut best_device = available[0];
        let mut best_score = 0.0;

        for &candidate in available {
            let mut total_score = 0.0;

            for &selected_device in selected {
                if let Some(link) = topology.links.get(&(selected_device, candidate)) {
                    total_score += link.quality_score as f64;
                }
            }

            if total_score > best_score {
                best_score = total_score;
                best_device = candidate;
            }
        }

        best_device
    }

    fn estimate_performance_internal(
        &self,
        workload: &Workload,
        devices: &[usize],
        topology: &GpuTopology,
    ) -> Result<PerformanceEstimate> {
        let compute_time = self.estimate_compute_time(workload, devices, topology);
        let comm_overhead = self.estimate_communication_overhead(workload, devices, topology);

        Ok(PerformanceEstimate {
            execution_time_ms: compute_time * (1.0 + comm_overhead),
            throughput: 1000.0 / (compute_time * (1.0 + comm_overhead)),
            memory_bandwidth_util: 0.7,
            compute_utilization: devices.iter().map(|_| 0.8).collect(),
            communication_overhead: comm_overhead as f32,
            confidence: 0.8,
        })
    }

    fn estimate_compute_time(
        &self,
        workload: &Workload,
        devices: &[usize],
        topology: &GpuTopology,
    ) -> f64 {
        let base_compute_time = 100.0; // Base 100ms

        // Adjust for device count (parallel efficiency)
        let parallel_efficiency = if devices.len() == 1 {
            1.0
        } else {
            0.9f64.powi(devices.len() as i32 - 1)
        };

        // Adjust for device capabilities
        let capability_factor = devices
            .iter()
            .filter_map(|&id| topology.devices.get(&id))
            .map(|dev| {
                if dev.capabilities.tensor_cores
                    && workload.compute_requirements.benefits_from_tensor_cores
                {
                    0.5
                } else {
                    1.0
                }
            })
            .fold(1.0, |acc, x| acc * x);

        base_compute_time * capability_factor / parallel_efficiency
    }

    fn estimate_communication_overhead(
        &self,
        workload: &Workload,
        devices: &[usize],
        topology: &GpuTopology,
    ) -> f64 {
        if devices.len() <= 1 {
            return 0.0;
        }

        let mut total_comm_time = 0.0;
        let comm_volume = workload
            .communication_patterns
            .iter()
            .map(|p| p.data_volume as f64)
            .sum::<f64>();

        if comm_volume > 0.0 {
            // Estimate communication time based on worst link
            let mut min_bandwidth = f64::INFINITY;
            for i in 0..devices.len() {
                for j in i + 1..devices.len() {
                    if let Some(link) = topology.links.get(&(devices[i], devices[j])) {
                        min_bandwidth = min_bandwidth.min(link.bandwidth_gbps);
                    }
                }
            }

            if min_bandwidth != f64::INFINITY {
                total_comm_time = comm_volume / (min_bandwidth * 1e9 / 8.0) * 1000.0;
                // Convert to ms
            }
        }

        total_comm_time / 100.0 // Normalize as overhead ratio
    }
}

/// Bandwidth-optimized placement strategy.
#[derive(Debug)]
pub struct BandwidthOptimizedPlacement;

impl BandwidthOptimizedPlacement {
    /// Create a new bandwidth-optimized placement strategy.
    pub fn new() -> Self {
        Self
    }
}

impl PlacementStrategy for BandwidthOptimizedPlacement {
    fn name(&self) -> &str {
        "bandwidth_optimized"
    }

    fn optimize_placement(
        &self,
        workload: &Workload,
        topology: &GpuTopology,
    ) -> Result<PlacementPlan> {
        debug!("Optimizing placement using bandwidth-optimized strategy");

        // Find devices with highest aggregate bandwidth
        let mut device_bandwidth_scores = HashMap::new();

        for (&device_id, _) in &topology.devices {
            let mut total_bandwidth = 0.0;
            for (&(src, dst), link) in &topology.links {
                if src == device_id || dst == device_id {
                    total_bandwidth += link.bandwidth_gbps;
                }
            }
            device_bandwidth_scores.insert(device_id, total_bandwidth);
        }

        // Select top devices by bandwidth score
        let mut sorted_devices: Vec<_> = device_bandwidth_scores.iter().collect();
        sorted_devices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let device_count = workload
            .compute_requirements
            .preferred_device_count
            .min(topology.devices.len());
        let selected_devices: Vec<usize> = sorted_devices
            .iter()
            .take(device_count)
            .map(|(id, _)| **id)
            .collect();

        let mut device_assignments = HashMap::new();
        for (i, &device_id) in selected_devices.iter().enumerate() {
            device_assignments.insert(format!("component_{}", i), device_id);
        }

        Ok(PlacementPlan {
            device_assignments,
            performance_estimate: PerformanceEstimate {
                execution_time_ms: 80.0,
                throughput: 12.5,
                memory_bandwidth_util: 0.9,
                compute_utilization: selected_devices.iter().map(|_| 0.85).collect(),
                communication_overhead: 0.1,
                confidence: 0.9,
            },
            resource_utilization: ResourceUtilization {
                memory_usage: selected_devices
                    .iter()
                    .map(|&id| (id, workload.compute_requirements.memory_per_device))
                    .collect(),
                compute_utilization: selected_devices.iter().map(|&id| (id, 0.85)).collect(),
                bandwidth_utilization: HashMap::new(),
                power_consumption: selected_devices.len() as f32 * 320.0,
            },
            communication_plan: CommunicationPlan {
                routes: HashMap::new(),
                volume_per_route: HashMap::new(),
                load_balancing: HashMap::new(),
            },
        })
    }

    fn estimate_performance(
        &self,
        plan: &PlacementPlan,
        _workload: &Workload,
        _topology: &GpuTopology,
    ) -> Result<PerformanceEstimate> {
        Ok(plan.performance_estimate.clone())
    }
}

/// Power-efficient placement strategy.
#[derive(Debug)]
pub struct PowerEfficientPlacement;

impl PowerEfficientPlacement {
    /// Create a new power-efficient placement strategy.
    pub fn new() -> Self {
        Self
    }
}

impl PlacementStrategy for PowerEfficientPlacement {
    fn name(&self) -> &str {
        "power_efficient"
    }

    fn optimize_placement(
        &self,
        workload: &Workload,
        topology: &GpuTopology,
    ) -> Result<PlacementPlan> {
        debug!("Optimizing placement using power-efficient strategy");

        // Select devices with best performance/power ratio
        let mut efficiency_scores = HashMap::new();

        for (&device_id, device_info) in &topology.devices {
            let perf_estimate = device_info.sm_count as f64 * device_info.boost_clock_mhz as f64;
            let efficiency = perf_estimate / device_info.power_limit_watts as f64;
            efficiency_scores.insert(device_id, efficiency);
        }

        let mut sorted_devices: Vec<_> = efficiency_scores.iter().collect();
        sorted_devices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let device_count = workload
            .compute_requirements
            .preferred_device_count
            .min(topology.devices.len());
        let selected_devices: Vec<usize> = sorted_devices
            .iter()
            .take(device_count)
            .map(|(id, _)| **id)
            .collect();

        let mut device_assignments = HashMap::new();
        for (i, &device_id) in selected_devices.iter().enumerate() {
            device_assignments.insert(format!("component_{}", i), device_id);
        }

        Ok(PlacementPlan {
            device_assignments,
            performance_estimate: PerformanceEstimate {
                execution_time_ms: 110.0,
                throughput: 9.1,
                memory_bandwidth_util: 0.6,
                compute_utilization: selected_devices.iter().map(|_| 0.7).collect(),
                communication_overhead: 0.15,
                confidence: 0.75,
            },
            resource_utilization: ResourceUtilization {
                memory_usage: selected_devices
                    .iter()
                    .map(|&id| (id, workload.compute_requirements.memory_per_device))
                    .collect(),
                compute_utilization: selected_devices.iter().map(|&id| (id, 0.7)).collect(),
                bandwidth_utilization: HashMap::new(),
                power_consumption: selected_devices.len() as f32 * 250.0,
            },
            communication_plan: CommunicationPlan {
                routes: HashMap::new(),
                volume_per_route: HashMap::new(),
                load_balancing: HashMap::new(),
            },
        })
    }

    fn estimate_performance(
        &self,
        plan: &PlacementPlan,
        _workload: &Workload,
        _topology: &GpuTopology,
    ) -> Result<PerformanceEstimate> {
        Ok(plan.performance_estimate.clone())
    }
}
