//! Kernel compilation framework for subgraph optimization and execution.
//!
//! This module provides utilities for analyzing subgraphs, detecting fusion
//! opportunities, and optimizing memory layouts for efficient execution.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{anyhow, Result};
use ronn_core::{DataType, GraphEdge, GraphNode, NodeId, SubGraph, TensorLayout};
use tracing::{debug, info};

/// Kernel compiler for subgraph analysis and optimization.
#[derive(Debug)]
pub struct KernelCompiler {
    /// Fusion optimization settings.
    fusion_config: FusionConfig,
    /// Memory optimization settings.
    memory_config: MemoryConfig,
}

/// Configuration for operation fusion optimization.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Enable operator fusion.
    pub enable_fusion: bool,
    /// Maximum number of operations to fuse together.
    pub max_fusion_depth: usize,
    /// Enable element-wise operation fusion.
    pub enable_elementwise_fusion: bool,
    /// Enable convolution fusion (Conv + BatchNorm + ReLU).
    pub enable_conv_fusion: bool,
    /// Enable matrix multiplication fusion (MatMul + Add).
    pub enable_matmul_fusion: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            max_fusion_depth: 4,
            enable_elementwise_fusion: true,
            enable_conv_fusion: true,
            enable_matmul_fusion: true,
        }
    }
}

/// Configuration for memory layout optimization.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Enable memory layout optimization.
    pub enable_optimization: bool,
    /// Prefer row-major layouts for CPU.
    pub prefer_row_major: bool,
    /// Enable tensor reuse when possible.
    pub enable_tensor_reuse: bool,
    /// Maximum memory overhead for optimization (percentage).
    pub max_memory_overhead: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            prefer_row_major: true,
            enable_tensor_reuse: true,
            max_memory_overhead: 0.2, // 20% overhead
        }
    }
}

/// Represents a fused operation combining multiple nodes.
#[derive(Debug, Clone)]
pub struct FusedOperation {
    /// Unique ID for this fused operation.
    pub id: usize,
    /// Original nodes that were fused together.
    pub nodes: Vec<GraphNode>,
    /// Type of fusion applied.
    pub fusion_type: FusionType,
    /// Input tensor indices.
    pub inputs: Vec<usize>,
    /// Output tensor indices.
    pub outputs: Vec<usize>,
    /// Estimated execution cost.
    pub cost: f64,
}

/// Types of operation fusion patterns.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionType {
    /// No fusion, single operation.
    None,
    /// Element-wise operations (Add, Mul, ReLU).
    ElementWise,
    /// Convolution + BatchNorm + ReLU.
    ConvBnRelu,
    /// Matrix multiplication + bias addition.
    MatMulBias,
    /// Custom fusion pattern.
    Custom(String),
}

/// Memory layout analysis and optimization results.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Total number of tensors needed.
    pub tensor_count: usize,
    /// Memory requirements for each tensor.
    pub tensor_info: Vec<TensorInfo>,
    /// Memory reuse mapping (tensor_id -> reused_from_id).
    pub reuse_map: HashMap<usize, usize>,
    /// Total estimated memory usage.
    pub total_memory: usize,
}

/// Information about a tensor in the memory plan.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor identifier.
    pub id: usize,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Data type of the tensor.
    pub dtype: DataType,
    /// Preferred memory layout.
    pub layout: TensorLayout,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Lifetime (first_use, last_use).
    pub lifetime: (usize, usize),
}

/// Compilation result containing optimized execution plan.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Optimized fused operations.
    pub fused_ops: Vec<FusedOperation>,
    /// Memory layout plan.
    pub memory_plan: MemoryPlan,
    /// Optimization statistics.
    pub stats: CompilationStats,
}

/// Statistics from the compilation process.
#[derive(Debug, Clone)]
pub struct CompilationStats {
    /// Original number of operations.
    pub original_ops: usize,
    /// Number of operations after fusion.
    pub fused_ops: usize,
    /// Memory reduction achieved (percentage).
    pub memory_reduction: f32,
    /// Estimated performance improvement (percentage).
    pub performance_improvement: f32,
}

impl KernelCompiler {
    /// Create a new kernel compiler with default configuration.
    pub fn new() -> Self {
        Self {
            fusion_config: FusionConfig::default(),
            memory_config: MemoryConfig::default(),
        }
    }

    /// Create a kernel compiler with custom configuration.
    pub fn with_config(fusion_config: FusionConfig, memory_config: MemoryConfig) -> Self {
        Self {
            fusion_config,
            memory_config,
        }
    }

    /// Compile a subgraph with optimizations.
    pub fn compile(&self, subgraph: &SubGraph) -> Result<CompilationResult> {
        debug!("Compiling subgraph with {} nodes", subgraph.nodes.len());

        // Step 1: Analyze subgraph topology
        let topology = self.analyze_topology(subgraph)?;

        // Step 2: Detect fusion opportunities
        let fusion_candidates = if self.fusion_config.enable_fusion {
            self.detect_fusion_opportunities(subgraph, &topology)?
        } else {
            // No fusion, create single-node operations
            subgraph
                .nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    FusedOperation {
                        id: i,
                        nodes: vec![node.clone()],
                        fusion_type: FusionType::None,
                        inputs: vec![i], // Simplified
                        outputs: vec![i],
                        cost: 1.0,
                    }
                })
                .collect()
        };

        // Step 3: Apply fusion optimization
        let fused_ops = self.apply_fusion(subgraph, fusion_candidates)?;

        // Step 4: Optimize memory layout
        let memory_plan = if self.memory_config.enable_optimization {
            self.optimize_memory_layout(subgraph, &fused_ops)?
        } else {
            self.create_basic_memory_plan(subgraph)?
        };

        // Step 5: Calculate statistics
        let stats = CompilationStats {
            original_ops: subgraph.nodes.len(),
            fused_ops: fused_ops.len(),
            memory_reduction: self.calculate_memory_reduction(subgraph, &memory_plan),
            performance_improvement: self.estimate_performance_improvement(&fused_ops),
        };

        info!(
            "Compilation complete: {} -> {} ops, {:.1}% memory reduction, {:.1}% performance improvement",
            stats.original_ops, stats.fused_ops, stats.memory_reduction * 100.0, stats.performance_improvement * 100.0
        );

        Ok(CompilationResult {
            fused_ops,
            memory_plan,
            stats,
        })
    }

    /// Analyze subgraph topology and dependencies.
    fn analyze_topology(&self, subgraph: &SubGraph) -> Result<TopologyInfo> {
        let mut topology = TopologyInfo {
            node_dependencies: HashMap::new(),
            node_dependents: HashMap::new(),
            execution_order: Vec::new(),
        };

        // Build dependency maps
        for edge in &subgraph.edges {
            topology
                .node_dependencies
                .entry(edge.to_node)
                .or_insert_with(Vec::new)
                .push(edge.from_node);

            topology
                .node_dependents
                .entry(edge.from_node)
                .or_insert_with(Vec::new)
                .push(edge.to_node);
        }

        // Topological sort to determine execution order
        topology.execution_order = self.topological_sort(subgraph)?;

        Ok(topology)
    }

    /// Perform topological sort on the subgraph.
    fn topological_sort(&self, subgraph: &SubGraph) -> Result<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Initialize in-degree and adjacency list
        for node in &subgraph.nodes {
            in_degree.insert(node.id, 0);
            adjacency.insert(node.id, Vec::new());
        }

        // Build adjacency list and calculate in-degrees
        for edge in &subgraph.edges {
            adjacency
                .get_mut(&edge.from_node)
                .unwrap()
                .push(edge.to_node);
            *in_degree.get_mut(&edge.to_node).unwrap() += 1;
        }

        // Kahn's algorithm
        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(node_id, _)| *node_id)
            .collect();

        let mut result = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            if let Some(neighbors) = adjacency.get(&node_id) {
                for &neighbor in neighbors {
                    let degree = in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if result.len() != subgraph.nodes.len() {
            return Err(anyhow!("Cycle detected in subgraph"));
        }

        Ok(result)
    }

    /// Detect fusion opportunities in the subgraph.
    fn detect_fusion_opportunities(
        &self,
        subgraph: &SubGraph,
        topology: &TopologyInfo,
    ) -> Result<Vec<FusedOperation>> {
        let mut fusion_candidates = Vec::new();
        let mut processed_nodes = HashSet::new();

        for &node_id in &topology.execution_order {
            if processed_nodes.contains(&node_id) {
                continue;
            }

            let node = subgraph.nodes.iter().find(|n| n.id == node_id).unwrap();
            let fusion_group =
                self.try_create_fusion_group(node, subgraph, topology, &processed_nodes)?;

            // Mark all nodes in the fusion group as processed
            for fused_node in &fusion_group.nodes {
                processed_nodes.insert(fused_node.id);
            }

            fusion_candidates.push(fusion_group);
        }

        Ok(fusion_candidates)
    }

    /// Try to create a fusion group starting from the given node.
    fn try_create_fusion_group(
        &self,
        start_node: &GraphNode,
        subgraph: &SubGraph,
        topology: &TopologyInfo,
        processed_nodes: &HashSet<NodeId>,
    ) -> Result<FusedOperation> {
        let mut fusion_group = vec![start_node.clone()];
        let mut current_node_id = start_node.id;

        // Try to extend the fusion group
        for _ in 0..self.fusion_config.max_fusion_depth - 1 {
            let current_node = subgraph
                .nodes
                .iter()
                .find(|n| n.id == current_node_id)
                .unwrap();
            if let Some(next_node) =
                self.find_fusable_successor(current_node, subgraph, topology, processed_nodes)?
            {
                current_node_id = next_node.id;
                fusion_group.push(next_node);
            } else {
                break;
            }
        }

        // Determine fusion type
        let fusion_type = self.classify_fusion_group(&fusion_group);

        // Estimate cost
        let cost = self.estimate_fusion_cost(&fusion_group, &fusion_type);

        Ok(FusedOperation {
            id: start_node.id, // Use start node ID as fusion ID
            nodes: fusion_group,
            fusion_type,
            inputs: vec![0],  // Simplified
            outputs: vec![0], // Simplified
            cost,
        })
    }

    /// Find a node that can be fused with the current node.
    fn find_fusable_successor(
        &self,
        current_node: &GraphNode,
        subgraph: &SubGraph,
        topology: &TopologyInfo,
        processed_nodes: &HashSet<NodeId>,
    ) -> Result<Option<GraphNode>> {
        if let Some(dependents) = topology.node_dependents.get(&current_node.id) {
            for &dependent_id in dependents {
                if processed_nodes.contains(&dependent_id) {
                    continue;
                }

                if let Some(dependent_node) = subgraph.nodes.iter().find(|n| n.id == dependent_id) {
                    if self.can_fuse(&current_node.op_type, &dependent_node.op_type) {
                        return Ok(Some(dependent_node.clone()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check if two operation types can be fused.
    fn can_fuse(&self, op1: &str, op2: &str) -> bool {
        match (op1, op2) {
            // Element-wise fusion
            ("Add", "ReLU") | ("Mul", "ReLU") | ("Sub", "ReLU") => {
                self.fusion_config.enable_elementwise_fusion
            }
            ("Add", "Add") | ("Mul", "Mul") => self.fusion_config.enable_elementwise_fusion,

            // Convolution fusion
            ("Conv", "BatchNormalization") | ("BatchNormalization", "ReLU") => {
                self.fusion_config.enable_conv_fusion
            }

            // Matrix multiplication fusion
            ("MatMul", "Add") | ("Gemm", "Add") => self.fusion_config.enable_matmul_fusion,

            _ => false,
        }
    }

    /// Classify the type of fusion for a group of operations.
    fn classify_fusion_group(&self, nodes: &[GraphNode]) -> FusionType {
        if nodes.len() == 1 {
            return FusionType::None;
        }

        let op_types: Vec<&str> = nodes.iter().map(|n| n.op_type.as_str()).collect();

        match op_types.as_slice() {
            ["Conv", "BatchNormalization", "ReLU"] => FusionType::ConvBnRelu,
            ["MatMul", "Add"] | ["Gemm", "Add"] => FusionType::MatMulBias,
            ops if ops
                .iter()
                .all(|op| matches!(*op, "Add" | "Mul" | "Sub" | "ReLU")) =>
            {
                FusionType::ElementWise
            }
            _ => FusionType::Custom(format!("{:?}", op_types)),
        }
    }

    /// Estimate the computational cost of a fusion group.
    fn estimate_fusion_cost(&self, nodes: &[GraphNode], fusion_type: &FusionType) -> f64 {
        let base_cost: f64 = nodes.len() as f64;

        match fusion_type {
            FusionType::None => base_cost,
            FusionType::ElementWise => base_cost * 0.7, // 30% savings
            FusionType::ConvBnRelu => base_cost * 0.6,  // 40% savings
            FusionType::MatMulBias => base_cost * 0.8,  // 20% savings
            FusionType::Custom(_) => base_cost * 0.9,   // 10% savings
        }
    }

    /// Apply fusion optimizations to create the final operation list.
    fn apply_fusion(
        &self,
        _subgraph: &SubGraph,
        fusion_candidates: Vec<FusedOperation>,
    ) -> Result<Vec<FusedOperation>> {
        // For now, just return the candidates as-is
        // In practice, would apply additional optimizations here
        Ok(fusion_candidates)
    }

    /// Optimize memory layout for the fused operations.
    fn optimize_memory_layout(
        &self,
        subgraph: &SubGraph,
        fused_ops: &[FusedOperation],
    ) -> Result<MemoryPlan> {
        let mut tensor_info = Vec::new();
        let mut reuse_map = HashMap::new();
        let mut total_memory = 0;

        // Analyze tensor lifetimes
        let lifetimes = self.analyze_tensor_lifetimes(fused_ops)?;

        // Create tensor info
        for (i, (shape, dtype)) in self.estimate_tensor_shapes(subgraph)?.iter().enumerate() {
            let size_bytes = self.calculate_tensor_size(shape, *dtype);
            let layout = if self.memory_config.prefer_row_major {
                TensorLayout::RowMajor
            } else {
                TensorLayout::ColumnMajor
            };

            let lifetime = lifetimes.get(&i).copied().unwrap_or((0, fused_ops.len()));

            let info = TensorInfo {
                id: i,
                shape: shape.clone(),
                dtype: *dtype,
                layout,
                size_bytes,
                lifetime,
            };

            tensor_info.push(info);
            total_memory += size_bytes;
        }

        // Detect reuse opportunities
        if self.memory_config.enable_tensor_reuse {
            reuse_map = self.detect_tensor_reuse(&tensor_info)?;
        }

        Ok(MemoryPlan {
            tensor_count: tensor_info.len(),
            tensor_info,
            reuse_map,
            total_memory,
        })
    }

    /// Create a basic memory plan without optimization.
    fn create_basic_memory_plan(&self, subgraph: &SubGraph) -> Result<MemoryPlan> {
        let tensor_shapes = self.estimate_tensor_shapes(subgraph)?;
        let mut tensor_info = Vec::new();
        let mut total_memory = 0;

        for (i, (shape, dtype)) in tensor_shapes.iter().enumerate() {
            let size_bytes = self.calculate_tensor_size(shape, *dtype);

            let info = TensorInfo {
                id: i,
                shape: shape.clone(),
                dtype: *dtype,
                layout: TensorLayout::RowMajor,
                size_bytes,
                lifetime: (0, subgraph.nodes.len()),
            };

            tensor_info.push(info);
            total_memory += size_bytes;
        }

        Ok(MemoryPlan {
            tensor_count: tensor_info.len(),
            tensor_info,
            reuse_map: HashMap::new(),
            total_memory,
        })
    }

    /// Estimate tensor shapes and types for the subgraph.
    fn estimate_tensor_shapes(&self, subgraph: &SubGraph) -> Result<Vec<(Vec<usize>, DataType)>> {
        // Simplified shape estimation
        let mut shapes = Vec::new();

        for _node in &subgraph.nodes {
            // Default to a medium-sized tensor
            shapes.push((vec![32, 32], DataType::F32));
        }

        Ok(shapes)
    }

    /// Calculate tensor size in bytes.
    fn calculate_tensor_size(&self, shape: &[usize], dtype: DataType) -> usize {
        let element_count: usize = shape.iter().product();
        let element_size = match dtype {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F64 | DataType::I64 => 8,
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
        };
        element_count * element_size
    }

    /// Analyze tensor lifetimes for memory reuse.
    fn analyze_tensor_lifetimes(
        &self,
        fused_ops: &[FusedOperation],
    ) -> Result<HashMap<usize, (usize, usize)>> {
        let mut lifetimes = HashMap::new();

        // Simplified lifetime analysis
        for (op_idx, fused_op) in fused_ops.iter().enumerate() {
            for &input_idx in &fused_op.inputs {
                let entry = lifetimes.entry(input_idx).or_insert((op_idx, op_idx));
                entry.0 = entry.0.min(op_idx);
                entry.1 = entry.1.max(op_idx);
            }
            for &output_idx in &fused_op.outputs {
                let entry = lifetimes.entry(output_idx).or_insert((op_idx, op_idx));
                entry.0 = entry.0.min(op_idx);
                entry.1 = entry.1.max(op_idx);
            }
        }

        Ok(lifetimes)
    }

    /// Detect tensor reuse opportunities.
    fn detect_tensor_reuse(&self, tensor_info: &[TensorInfo]) -> Result<HashMap<usize, usize>> {
        let mut reuse_map = HashMap::new();

        // Simple reuse detection: tensors with non-overlapping lifetimes and same size
        for i in 0..tensor_info.len() {
            for j in (i + 1)..tensor_info.len() {
                let tensor1 = &tensor_info[i];
                let tensor2 = &tensor_info[j];

                // Check if lifetimes don't overlap
                if tensor1.lifetime.1 < tensor2.lifetime.0 {
                    // Check if sizes are compatible
                    if tensor1.size_bytes == tensor2.size_bytes && tensor1.dtype == tensor2.dtype {
                        reuse_map.insert(j, i);
                        break; // tensor2 can reuse tensor1's memory
                    }
                }
            }
        }

        Ok(reuse_map)
    }

    /// Calculate memory reduction achieved.
    fn calculate_memory_reduction(&self, _subgraph: &SubGraph, memory_plan: &MemoryPlan) -> f32 {
        // Estimate original memory usage (no reuse)
        let original_memory: usize = memory_plan.tensor_info.iter().map(|t| t.size_bytes).sum();

        // Calculate memory with reuse
        let reused_memory: usize = memory_plan
            .reuse_map
            .values()
            .map(|&reused_from| memory_plan.tensor_info[reused_from].size_bytes)
            .sum();

        if original_memory > 0 {
            reused_memory as f32 / original_memory as f32
        } else {
            0.0
        }
    }

    /// Estimate performance improvement from fusion.
    fn estimate_performance_improvement(&self, fused_ops: &[FusedOperation]) -> f32 {
        let total_savings: f64 = fused_ops
            .iter()
            .map(|op| {
                let original_cost = op.nodes.len() as f64;
                (original_cost - op.cost).max(0.0)
            })
            .sum();

        let total_original_cost: f64 = fused_ops.iter().map(|op| op.nodes.len() as f64).sum();

        if total_original_cost > 0.0 {
            (total_savings / total_original_cost) as f32
        } else {
            0.0
        }
    }
}

/// Topology analysis results.
#[derive(Debug)]
struct TopologyInfo {
    node_dependencies: HashMap<NodeId, Vec<NodeId>>,
    node_dependents: HashMap<NodeId, Vec<NodeId>>,
    execution_order: Vec<NodeId>,
}

impl Default for KernelCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_subgraph() -> SubGraph {
        let nodes = vec![
            GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["input1".to_string(), "input2".to_string()],
                outputs: vec!["temp1".to_string()],
                name: Some("add1".to_string()),
            },
            GraphNode {
                id: 1,
                op_type: "ReLU".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["temp1".to_string()],
                outputs: vec!["output1".to_string()],
                name: Some("relu1".to_string()),
            },
        ];

        let edges = vec![GraphEdge {
            from_node: 0,
            to_node: 1,
            tensor_name: "temp1".to_string(),
            tensor_shape: Some(vec![32, 32]),
            tensor_dtype: DataType::F32,
        }];

        SubGraph {
            nodes,
            edges,
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
        }
    }

    #[test]
    fn test_compiler_creation() {
        let compiler = KernelCompiler::new();
        assert!(compiler.fusion_config.enable_fusion);
        assert!(compiler.memory_config.enable_optimization);
    }

    #[test]
    fn test_compilation() -> Result<()> {
        let compiler = KernelCompiler::new();
        let subgraph = create_test_subgraph();

        let result = compiler.compile(&subgraph)?;

        // Should have fewer operations due to fusion
        assert!(result.fused_ops.len() <= subgraph.nodes.len());
        assert!(result.memory_plan.tensor_count > 0);
        assert_eq!(result.stats.original_ops, 2);

        Ok(())
    }

    #[test]
    fn test_fusion_detection() -> Result<()> {
        let compiler = KernelCompiler::new();

        // Test fusable operations
        assert!(compiler.can_fuse("Add", "ReLU"));
        assert!(compiler.can_fuse("Conv", "BatchNormalization"));
        assert!(compiler.can_fuse("MatMul", "Add"));

        // Test non-fusable operations
        assert!(!compiler.can_fuse("Add", "Conv"));
        assert!(!compiler.can_fuse("ReLU", "MatMul"));

        Ok(())
    }

    #[test]
    fn test_topological_sort() -> Result<()> {
        let compiler = KernelCompiler::new();
        let subgraph = create_test_subgraph();

        let order = compiler.topological_sort(&subgraph)?;
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], 0); // Add comes first
        assert_eq!(order[1], 1); // ReLU comes second

        Ok(())
    }

    #[test]
    fn test_memory_planning() -> Result<()> {
        let compiler = KernelCompiler::new();
        let subgraph = create_test_subgraph();

        let memory_plan = compiler.create_basic_memory_plan(&subgraph)?;

        assert_eq!(memory_plan.tensor_count, 2);
        assert!(memory_plan.total_memory > 0);

        for tensor in &memory_plan.tensor_info {
            assert!(!tensor.shape.is_empty());
            assert!(tensor.size_bytes > 0);
        }

        Ok(())
    }

    #[test]
    fn test_fusion_classification() -> Result<()> {
        let compiler = KernelCompiler::new();

        // Test single node
        let single_node = vec![GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
            name: None,
        }];
        assert_eq!(
            compiler.classify_fusion_group(&single_node),
            FusionType::None
        );

        // Test Conv-BN-ReLU fusion
        let conv_bn_relu = vec![
            GraphNode {
                id: 0,
                op_type: "Conv".to_string(),
                attributes: HashMap::new(),
                inputs: vec![],
                outputs: vec![],
                name: None,
            },
            GraphNode {
                id: 1,
                op_type: "BatchNormalization".to_string(),
                attributes: HashMap::new(),
                inputs: vec![],
                outputs: vec![],
                name: None,
            },
            GraphNode {
                id: 2,
                op_type: "ReLU".to_string(),
                attributes: HashMap::new(),
                inputs: vec![],
                outputs: vec![],
                name: None,
            },
        ];
        assert_eq!(
            compiler.classify_fusion_group(&conv_bn_relu),
            FusionType::ConvBnRelu
        );

        Ok(())
    }

    #[test]
    fn test_custom_config() -> Result<()> {
        let fusion_config = FusionConfig {
            enable_fusion: false,
            max_fusion_depth: 2,
            ..Default::default()
        };

        let memory_config = MemoryConfig {
            enable_optimization: false,
            ..Default::default()
        };

        let compiler = KernelCompiler::with_config(fusion_config, memory_config);
        let subgraph = create_test_subgraph();

        let result = compiler.compile(&subgraph)?;

        // With fusion disabled, should have same number of operations
        assert_eq!(result.fused_ops.len(), subgraph.nodes.len());

        Ok(())
    }
}
