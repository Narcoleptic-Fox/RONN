//! Graph manipulation and validation utilities.
//!
//! This module provides utilities for working with model graphs, including
//! validation, topological ordering, traversal, and subgraph extraction.

use crate::types::{AttributeValue, GraphEdge, GraphNode, ModelGraph, NodeId, SubGraph};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};

impl ModelGraph {
    /// Create a new empty model graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    ///
    /// # Arguments
    /// * `node` - The graph node to add
    ///
    /// # Returns
    /// The node ID that was assigned
    pub fn add_node(&mut self, mut node: GraphNode) -> NodeId {
        let node_id = self.nodes.len();
        node.id = node_id;
        self.nodes.push(node);
        node_id
    }

    /// Add an edge to the graph.
    ///
    /// # Arguments
    /// * `edge` - The graph edge to add
    pub fn add_edge(&mut self, edge: GraphEdge) -> Result<()> {
        // Validate that the nodes exist
        if edge.from_node >= self.nodes.len() || edge.to_node >= self.nodes.len() {
            return Err(anyhow!("Edge references non-existent nodes"));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Get a node by ID.
    pub fn get_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(node_id)
    }

    /// Get a mutable node by ID.
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(node_id)
    }

    /// Get all nodes in the graph.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Get mutable access to all nodes.
    pub fn nodes_mut(&mut self) -> &mut Vec<GraphNode> {
        &mut self.nodes
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Create a ModelGraph from a list of nodes.
    pub fn from_nodes(nodes: Vec<GraphNode>) -> Self {
        Self {
            nodes,
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Find nodes by operation type.
    pub fn find_nodes_by_op(&self, op_type: &str) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|node| {
                if node.op_type == op_type {
                    Some(node.id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all edges connected to a node.
    pub fn get_node_edges(&self, node_id: NodeId) -> (Vec<&GraphEdge>, Vec<&GraphEdge>) {
        let incoming: Vec<&GraphEdge> = self
            .edges
            .iter()
            .filter(|edge| edge.to_node == node_id)
            .collect();

        let outgoing: Vec<&GraphEdge> = self
            .edges
            .iter()
            .filter(|edge| edge.from_node == node_id)
            .collect();

        (incoming, outgoing)
    }

    /// Validate the graph structure.
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate node IDs
        let mut seen_ids = HashSet::new();
        for node in &self.nodes {
            if !seen_ids.insert(node.id) {
                return Err(anyhow!("Duplicate node ID: {}", node.id));
            }
        }

        // Validate edges reference existing nodes
        for edge in &self.edges {
            if edge.from_node >= self.nodes.len() {
                return Err(anyhow!(
                    "Edge references non-existent from_node: {}",
                    edge.from_node
                ));
            }
            if edge.to_node >= self.nodes.len() {
                return Err(anyhow!(
                    "Edge references non-existent to_node: {}",
                    edge.to_node
                ));
            }
        }

        // Check for cycles using DFS
        if self.has_cycles()? {
            return Err(anyhow!("Graph contains cycles"));
        }

        // Validate input/output tensor names are used in nodes
        self.validate_input_output_tensors()?;

        Ok(())
    }

    /// Check if the graph has cycles.
    fn has_cycles(&self) -> Result<bool> {
        let mut state = vec![NodeState::Unvisited; self.nodes.len()];

        for node_id in 0..self.nodes.len() {
            if state[node_id] == NodeState::Unvisited {
                if self.has_cycles_dfs(node_id, &mut state)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn has_cycles_dfs(&self, node_id: NodeId, state: &mut Vec<NodeState>) -> Result<bool> {
        state[node_id] = NodeState::Visiting;

        let (_, outgoing) = self.get_node_edges(node_id);
        for edge in outgoing {
            match state[edge.to_node] {
                NodeState::Visiting => return Ok(true), // Back edge found - cycle detected
                NodeState::Unvisited => {
                    if self.has_cycles_dfs(edge.to_node, state)? {
                        return Ok(true);
                    }
                }
                NodeState::Visited => {} // Safe to ignore
            }
        }

        state[node_id] = NodeState::Visited;
        Ok(false)
    }

    /// Validate that input/output tensor names are used in the graph.
    fn validate_input_output_tensors(&self) -> Result<()> {
        let mut all_tensor_names: HashSet<String> = HashSet::new();

        // Collect all tensor names used in nodes
        for node in &self.nodes {
            for input in &node.inputs {
                all_tensor_names.insert(input.clone());
            }
            for output in &node.outputs {
                all_tensor_names.insert(output.clone());
            }
        }

        // Check that graph inputs exist in tensor names
        for input in &self.inputs {
            if !all_tensor_names.contains(input) {
                return Err(anyhow!("Graph input '{}' is not used by any node", input));
            }
        }

        // Check that graph outputs exist in tensor names
        for output in &self.outputs {
            if !all_tensor_names.contains(output) {
                return Err(anyhow!(
                    "Graph output '{}' is not produced by any node",
                    output
                ));
            }
        }

        Ok(())
    }

    /// Get topological ordering of nodes.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>> {
        let mut in_degree = vec![0; self.nodes.len()];

        // Calculate in-degrees
        for edge in &self.edges {
            in_degree[edge.to_node] += 1;
        }

        // Queue nodes with no dependencies
        let mut queue = VecDeque::new();
        for (node_id, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        let mut result = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            // Process outgoing edges
            let (_, outgoing) = self.get_node_edges(node_id);
            for edge in outgoing {
                in_degree[edge.to_node] -= 1;
                if in_degree[edge.to_node] == 0 {
                    queue.push_back(edge.to_node);
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(anyhow!(
                "Graph contains cycles - cannot perform topological sort"
            ));
        }

        Ok(result)
    }

    /// Extract a subgraph containing specified nodes.
    pub fn extract_subgraph(&self, node_ids: &[NodeId]) -> Result<SubGraph> {
        let node_set: HashSet<NodeId> = node_ids.iter().cloned().collect();

        // Validate all node IDs exist
        for &node_id in node_ids {
            if node_id >= self.nodes.len() {
                return Err(anyhow!("Node ID {} does not exist", node_id));
            }
        }

        // Create mapping from old node IDs to new node IDs
        let mut id_mapping = HashMap::new();
        let mut subgraph_nodes = Vec::new();

        for (new_id, &old_id) in node_ids.iter().enumerate() {
            id_mapping.insert(old_id, new_id);
            let mut node = self.nodes[old_id].clone();
            node.id = new_id;
            subgraph_nodes.push(node);
        }

        // Extract relevant edges
        let mut subgraph_edges = Vec::new();
        for edge in &self.edges {
            if node_set.contains(&edge.from_node) && node_set.contains(&edge.to_node) {
                let mut new_edge = edge.clone();
                new_edge.from_node = id_mapping[&edge.from_node];
                new_edge.to_node = id_mapping[&edge.to_node];
                subgraph_edges.push(new_edge);
            }
        }

        // Determine subgraph inputs and outputs
        let mut subgraph_inputs = HashSet::new();
        let mut subgraph_outputs = HashSet::new();

        for node in &subgraph_nodes {
            // Inputs that don't come from within the subgraph are external inputs
            for input in &node.inputs {
                let mut is_external = true;
                for other_node in &subgraph_nodes {
                    if other_node.outputs.contains(input) {
                        is_external = false;
                        break;
                    }
                }
                if is_external {
                    subgraph_inputs.insert(input.clone());
                }
            }

            // Outputs that don't go to nodes within the subgraph are external outputs
            for output in &node.outputs {
                let mut is_external = true;
                for other_node in &subgraph_nodes {
                    if other_node.inputs.contains(output) {
                        is_external = false;
                        break;
                    }
                }
                if is_external {
                    subgraph_outputs.insert(output.clone());
                }
            }
        }

        Ok(SubGraph {
            nodes: subgraph_nodes,
            edges: subgraph_edges,
            inputs: subgraph_inputs.into_iter().collect(),
            outputs: subgraph_outputs.into_iter().collect(),
        })
    }

    /// Count nodes by operation type.
    pub fn count_ops(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for node in &self.nodes {
            *counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Get graph statistics.
    pub fn statistics(&self) -> GraphStatistics {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();
        let op_counts = self.count_ops();
        let input_count = self.inputs.len();
        let output_count = self.outputs.len();

        let depth = self.calculate_depth();

        GraphStatistics {
            node_count,
            edge_count,
            op_counts,
            input_count,
            output_count,
            depth,
        }
    }

    /// Calculate the maximum depth of the graph.
    fn calculate_depth(&self) -> usize {
        if let Ok(topo_order) = self.topological_sort() {
            let mut depths = vec![0; self.nodes.len()];

            for &node_id in &topo_order {
                let (incoming, _) = self.get_node_edges(node_id);
                if incoming.is_empty() {
                    depths[node_id] = 0;
                } else {
                    let max_input_depth = incoming
                        .iter()
                        .map(|edge| depths[edge.from_node])
                        .max()
                        .unwrap_or(0);
                    depths[node_id] = max_input_depth + 1;
                }
            }

            depths.into_iter().max().unwrap_or(0)
        } else {
            0 // If there are cycles, depth is undefined
        }
    }
}

impl Default for ModelGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Node state for cycle detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    Unvisited,
    Visiting,
    Visited,
}

/// Graph statistics.
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    /// Total number of nodes.
    pub node_count: usize,
    /// Total number of edges.
    pub edge_count: usize,
    /// Count of each operation type.
    pub op_counts: HashMap<String, usize>,
    /// Number of graph inputs.
    pub input_count: usize,
    /// Number of graph outputs.
    pub output_count: usize,
    /// Maximum depth of the graph.
    pub depth: usize,
}

/// Graph builder for convenient graph construction.
pub struct GraphBuilder {
    graph: ModelGraph,
}

impl GraphBuilder {
    /// Create a new graph builder.
    pub fn new() -> Self {
        Self {
            graph: ModelGraph::new(),
        }
    }

    /// Add a node with the given operation type.
    pub fn add_op(&mut self, op_type: &str, name: Option<String>) -> NodeId {
        let node = GraphNode {
            id: 0, // Will be set by add_node
            op_type: op_type.to_string(),
            attributes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            name,
        };
        self.graph.add_node(node)
    }

    /// Add an input tensor to a node.
    pub fn add_input(&mut self, node_id: NodeId, tensor_name: &str) -> &mut Self {
        if let Some(node) = self.graph.get_node_mut(node_id) {
            node.inputs.push(tensor_name.to_string());
        }
        self
    }

    /// Add an output tensor to a node.
    pub fn add_output(&mut self, node_id: NodeId, tensor_name: &str) -> &mut Self {
        if let Some(node) = self.graph.get_node_mut(node_id) {
            node.outputs.push(tensor_name.to_string());
        }
        self
    }

    /// Add an attribute to a node.
    pub fn add_attribute(
        &mut self,
        node_id: NodeId,
        name: &str,
        value: AttributeValue,
    ) -> &mut Self {
        if let Some(node) = self.graph.get_node_mut(node_id) {
            node.attributes.insert(name.to_string(), value);
        }
        self
    }

    /// Connect two nodes with a tensor.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
        tensor_name: &str,
    ) -> Result<&mut Self> {
        let edge = GraphEdge {
            from_node,
            to_node,
            tensor_name: tensor_name.to_string(),
            tensor_shape: None,
            tensor_dtype: crate::types::DataType::F32, // Default
        };
        self.graph.add_edge(edge)?;
        Ok(self)
    }

    /// Set graph inputs.
    pub fn set_inputs(&mut self, inputs: Vec<String>) -> &mut Self {
        self.graph.inputs = inputs;
        self
    }

    /// Set graph outputs.
    pub fn set_outputs(&mut self, outputs: Vec<String>) -> &mut Self {
        self.graph.outputs = outputs;
        self
    }

    /// Build the final graph.
    pub fn build(self) -> Result<ModelGraph> {
        self.graph.validate()?;
        Ok(self.graph)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataType;

    #[test]
    fn test_graph_creation() {
        let mut graph = ModelGraph::new();
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.edges.len(), 0);

        let node = GraphNode {
            id: 0,
            op_type: "Conv".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
            name: Some("conv1".to_string()),
        };

        let node_id = graph.add_node(node);
        assert_eq!(node_id, 0);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_edge_addition() -> Result<()> {
        let mut graph = ModelGraph::new();

        // Add two nodes
        let node1 = GraphNode {
            id: 0,
            op_type: "Input".to_string(),
            attributes: HashMap::new(),
            inputs: vec![],
            outputs: vec!["tensor1".to_string()],
            name: Some("input".to_string()),
        };

        let node2 = GraphNode {
            id: 1,
            op_type: "Conv".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["tensor1".to_string()],
            outputs: vec!["tensor2".to_string()],
            name: Some("conv".to_string()),
        };

        let id1 = graph.add_node(node1);
        let id2 = graph.add_node(node2);

        let edge = GraphEdge {
            from_node: id1,
            to_node: id2,
            tensor_name: "tensor1".to_string(),
            tensor_shape: Some(vec![1, 3, 224, 224]),
            tensor_dtype: DataType::F32,
        };

        graph.add_edge(edge)?;
        assert_eq!(graph.edges.len(), 1);

        Ok(())
    }

    #[test]
    fn test_topological_sort() -> Result<()> {
        let mut graph = ModelGraph::new();

        // Create a simple linear graph: A -> B -> C
        let node_a = GraphNode {
            id: 0,
            op_type: "Input".to_string(),
            attributes: HashMap::new(),
            inputs: vec![],
            outputs: vec!["a_out".to_string()],
            name: Some("A".to_string()),
        };

        let node_b = GraphNode {
            id: 1,
            op_type: "Conv".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["a_out".to_string()],
            outputs: vec!["b_out".to_string()],
            name: Some("B".to_string()),
        };

        let node_c = GraphNode {
            id: 2,
            op_type: "ReLU".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["b_out".to_string()],
            outputs: vec!["c_out".to_string()],
            name: Some("C".to_string()),
        };

        let id_a = graph.add_node(node_a);
        let id_b = graph.add_node(node_b);
        let id_c = graph.add_node(node_c);

        graph.add_edge(GraphEdge {
            from_node: id_a,
            to_node: id_b,
            tensor_name: "a_out".to_string(),
            tensor_shape: None,
            tensor_dtype: DataType::F32,
        })?;

        graph.add_edge(GraphEdge {
            from_node: id_b,
            to_node: id_c,
            tensor_name: "b_out".to_string(),
            tensor_shape: None,
            tensor_dtype: DataType::F32,
        })?;

        let topo_order = graph.topological_sort()?;
        assert_eq!(topo_order, vec![0, 1, 2]);

        Ok(())
    }

    #[test]
    fn test_graph_builder() -> Result<()> {
        let mut builder = GraphBuilder::new();

        let input_id = builder.add_op("Input", Some("input_layer".to_string()));
        builder.add_output(input_id, "input_tensor");

        let conv_id = builder.add_op("Conv", Some("conv_layer".to_string()));
        builder
            .add_input(conv_id, "input_tensor")
            .add_output(conv_id, "conv_output")
            .add_attribute(conv_id, "kernel_size", AttributeValue::IntArray(vec![3, 3]));

        builder.connect(input_id, conv_id, "input_tensor")?;
        builder
            .set_inputs(vec!["input_tensor".to_string()])
            .set_outputs(vec!["conv_output".to_string()]);

        let graph = builder.build()?;
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.inputs, vec!["input_tensor"]);
        assert_eq!(graph.outputs, vec!["conv_output"]);

        Ok(())
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = ModelGraph::new();

        // Create a cycle: A -> B -> C -> A
        let node_a = GraphNode {
            id: 0,
            op_type: "A".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["c_out".to_string()],
            outputs: vec!["a_out".to_string()],
            name: Some("A".to_string()),
        };

        let node_b = GraphNode {
            id: 1,
            op_type: "B".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["a_out".to_string()],
            outputs: vec!["b_out".to_string()],
            name: Some("B".to_string()),
        };

        let node_c = GraphNode {
            id: 2,
            op_type: "C".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["b_out".to_string()],
            outputs: vec!["c_out".to_string()],
            name: Some("C".to_string()),
        };

        let id_a = graph.add_node(node_a);
        let id_b = graph.add_node(node_b);
        let id_c = graph.add_node(node_c);

        // Add edges to form a cycle
        graph
            .add_edge(GraphEdge {
                from_node: id_a,
                to_node: id_b,
                tensor_name: "a_out".to_string(),
                tensor_shape: None,
                tensor_dtype: DataType::F32,
            })
            .unwrap();

        graph
            .add_edge(GraphEdge {
                from_node: id_b,
                to_node: id_c,
                tensor_name: "b_out".to_string(),
                tensor_shape: None,
                tensor_dtype: DataType::F32,
            })
            .unwrap();

        graph
            .add_edge(GraphEdge {
                from_node: id_c,
                to_node: id_a,
                tensor_name: "c_out".to_string(),
                tensor_shape: None,
                tensor_dtype: DataType::F32,
            })
            .unwrap();

        // This should fail validation due to the cycle
        assert!(graph.validate().is_err());
        assert!(graph.has_cycles().unwrap());
    }

    #[test]
    fn test_subgraph_extraction() -> Result<()> {
        let mut graph = ModelGraph::new();

        // Create a graph: Input -> Conv1 -> Conv2 -> Output
        let input_id = graph.add_node(GraphNode {
            id: 0,
            op_type: "Input".to_string(),
            attributes: HashMap::new(),
            inputs: vec![],
            outputs: vec!["input_out".to_string()],
            name: Some("input".to_string()),
        });

        let conv1_id = graph.add_node(GraphNode {
            id: 1,
            op_type: "Conv".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input_out".to_string()],
            outputs: vec!["conv1_out".to_string()],
            name: Some("conv1".to_string()),
        });

        let conv2_id = graph.add_node(GraphNode {
            id: 2,
            op_type: "Conv".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["conv1_out".to_string()],
            outputs: vec!["conv2_out".to_string()],
            name: Some("conv2".to_string()),
        });

        // Add edges
        graph.add_edge(GraphEdge {
            from_node: input_id,
            to_node: conv1_id,
            tensor_name: "input_out".to_string(),
            tensor_shape: None,
            tensor_dtype: DataType::F32,
        })?;

        graph.add_edge(GraphEdge {
            from_node: conv1_id,
            to_node: conv2_id,
            tensor_name: "conv1_out".to_string(),
            tensor_shape: None,
            tensor_dtype: DataType::F32,
        })?;

        // Extract subgraph containing just the conv layers
        let subgraph = graph.extract_subgraph(&[conv1_id, conv2_id])?;

        assert_eq!(subgraph.nodes.len(), 2);
        assert_eq!(subgraph.edges.len(), 1);
        assert_eq!(subgraph.inputs, vec!["input_out"]);
        assert_eq!(subgraph.outputs, vec!["conv2_out"]);

        Ok(())
    }
}
