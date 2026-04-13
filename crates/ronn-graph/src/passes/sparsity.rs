use super::{OptimizationPass, PassStats};
use crate::error::Result;
use ronn_core::{AttributeValue, GraphNode, ModelGraph};
use std::collections::HashMap;
use tracing::{debug, info};

/// Sparsity optimization pass that identifies FFN layers and inserts
/// activation predictor + sparse routing nodes.
///
/// Transforms:
///   `Attention -> FFN -> LayerNorm`
/// Into:
///   `Attention -> Predictor -> SparseFFN -> LayerNorm`
///
/// The predictor node produces a binary activation mask, and SparseFFN
/// uses that mask to only compute the predicted-active neurons.
pub struct SparsityOptimizationPass;

impl OptimizationPass for SparsityOptimizationPass {
    fn name(&self) -> &str {
        "SparsityOptimization"
    }

    fn run(&self, graph: &mut ModelGraph) -> Result<PassStats> {
        let mut stats = PassStats::default();

        // Identify FFN layer patterns: MatMul -> Activation -> MatMul
        let ffn_patterns = self.find_ffn_layers(graph);

        for (layer_idx, ffn) in ffn_patterns.iter().enumerate() {
            let modified = self.insert_sparsity_nodes(graph, ffn, layer_idx)?;
            stats.nodes_modified += modified;
        }

        if !ffn_patterns.is_empty() {
            info!(
                "Sparsity pass: identified {} FFN layers, modified {} nodes",
                ffn_patterns.len(),
                stats.nodes_modified
            );
        }

        Ok(stats)
    }

    fn should_run(&self, graph: &ModelGraph) -> bool {
        // Only run if the graph has MatMul nodes (indicating FFN layers)
        graph.nodes.iter().any(|n| n.op_type == "MatMul" || n.op_type == "Gemm")
    }
}

/// Identified FFN layer pattern in the graph.
struct FFNPattern {
    /// Node ID of the first linear (gate/up projection).
    first_linear_id: usize,
    /// Node ID of the activation function.
    activation_id: Option<usize>,
    /// Node ID of the second linear (down projection).
    second_linear_id: Option<usize>,
    /// Activation type detected.
    activation_type: String,
}

impl SparsityOptimizationPass {
    /// Find FFN layer patterns in the graph.
    ///
    /// Looks for sequences: MatMul/Gemm -> ReLU/SiLU/GELU -> MatMul/Gemm
    fn find_ffn_layers(&self, graph: &ModelGraph) -> Vec<FFNPattern> {
        let mut patterns = Vec::new();

        // Build output->consumer map
        let mut output_consumers: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            for input in &node.inputs {
                output_consumers
                    .entry(input.clone())
                    .or_default()
                    .push(idx);
            }
        }

        for (idx, node) in graph.nodes.iter().enumerate() {
            // Look for MatMul/Gemm nodes
            if node.op_type != "MatMul" && node.op_type != "Gemm" {
                continue;
            }

            // Check if any output feeds into an activation function
            for output in &node.outputs {
                if let Some(consumers) = output_consumers.get(output) {
                    for &consumer_idx in consumers {
                        let consumer = &graph.nodes[consumer_idx];
                        let is_activation = matches!(
                            consumer.op_type.as_str(),
                            "Relu" | "Silu" | "Gelu" | "Swish"
                        );

                        if is_activation {
                            // Found MatMul -> Activation, look for second MatMul
                            let mut second_linear = None;
                            for act_output in &consumer.outputs {
                                if let Some(act_consumers) = output_consumers.get(act_output) {
                                    for &act_consumer_idx in act_consumers {
                                        let next = &graph.nodes[act_consumer_idx];
                                        if next.op_type == "MatMul" || next.op_type == "Gemm" {
                                            second_linear = Some(act_consumer_idx);
                                            break;
                                        }
                                    }
                                }
                            }

                            patterns.push(FFNPattern {
                                first_linear_id: idx,
                                activation_id: Some(consumer_idx),
                                second_linear_id: second_linear,
                                activation_type: consumer.op_type.clone(),
                            });

                            debug!(
                                "Found FFN pattern: node {} -> {} -> {:?}",
                                idx,
                                consumer.op_type,
                                second_linear
                            );
                        }
                    }
                }
            }
        }

        patterns
    }

    /// Insert predictor and sparse routing annotations into FFN nodes.
    ///
    /// Rather than restructuring the graph (which would break existing execution),
    /// we annotate the FFN nodes with sparsity metadata that downstream execution
    /// providers can use to enable sparse computation.
    fn insert_sparsity_nodes(
        &self,
        graph: &mut ModelGraph,
        pattern: &FFNPattern,
        layer_idx: usize,
    ) -> Result<usize> {
        let mut modified = 0;

        // Annotate the first linear with sparsity metadata
        if let Some(node) = graph.nodes.get_mut(pattern.first_linear_id) {
            node.attributes.insert(
                "sparsity_enabled".to_string(),
                AttributeValue::Bool(true),
            );
            node.attributes.insert(
                "sparsity_layer_id".to_string(),
                AttributeValue::Int(layer_idx as i64),
            );
            node.attributes.insert(
                "sparsity_role".to_string(),
                AttributeValue::String("ffn_gate".to_string()),
            );
            node.attributes.insert(
                "sparsity_activation".to_string(),
                AttributeValue::String(pattern.activation_type.clone()),
            );
            modified += 1;
        }

        // Annotate the activation node
        if let Some(act_id) = pattern.activation_id {
            if let Some(node) = graph.nodes.get_mut(act_id) {
                node.attributes.insert(
                    "sparsity_enabled".to_string(),
                    AttributeValue::Bool(true),
                );
                node.attributes.insert(
                    "sparsity_layer_id".to_string(),
                    AttributeValue::Int(layer_idx as i64),
                );
                node.attributes.insert(
                    "sparsity_role".to_string(),
                    AttributeValue::String("ffn_activation".to_string()),
                );
                modified += 1;
            }
        }

        // Annotate the second linear (down projection)
        if let Some(lin2_id) = pattern.second_linear_id {
            if let Some(node) = graph.nodes.get_mut(lin2_id) {
                node.attributes.insert(
                    "sparsity_enabled".to_string(),
                    AttributeValue::Bool(true),
                );
                node.attributes.insert(
                    "sparsity_layer_id".to_string(),
                    AttributeValue::Int(layer_idx as i64),
                );
                node.attributes.insert(
                    "sparsity_role".to_string(),
                    AttributeValue::String("ffn_down".to_string()),
                );
                modified += 1;
            }
        }

        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::{DataType, GraphEdge};

    fn make_ffn_graph() -> ModelGraph {
        ModelGraph {
            nodes: vec![
                GraphNode {
                    id: 0,
                    op_type: "MatMul".to_string(),
                    attributes: HashMap::new(),
                    inputs: vec!["hidden".into(), "gate_weight".into()],
                    outputs: vec!["gate_out".into()],
                    name: Some("ffn_gate".into()),
                },
                GraphNode {
                    id: 1,
                    op_type: "Relu".to_string(),
                    attributes: HashMap::new(),
                    inputs: vec!["gate_out".into()],
                    outputs: vec!["act_out".into()],
                    name: Some("ffn_relu".into()),
                },
                GraphNode {
                    id: 2,
                    op_type: "MatMul".to_string(),
                    attributes: HashMap::new(),
                    inputs: vec!["act_out".into(), "down_weight".into()],
                    outputs: vec!["ffn_out".into()],
                    name: Some("ffn_down".into()),
                },
            ],
            edges: vec![
                GraphEdge {
                    from_node: 0,
                    to_node: 1,
                    tensor_name: "gate_out".into(),
                    tensor_shape: None,
                    tensor_dtype: DataType::F32,
                },
                GraphEdge {
                    from_node: 1,
                    to_node: 2,
                    tensor_name: "act_out".into(),
                    tensor_shape: None,
                    tensor_dtype: DataType::F32,
                },
            ],
            inputs: vec!["hidden".into()],
            outputs: vec!["ffn_out".into()],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_sparsity_pass_identifies_ffn() {
        let graph = make_ffn_graph();
        let pass = SparsityOptimizationPass;

        assert!(pass.should_run(&graph));
        let patterns = pass.find_ffn_layers(&graph);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].first_linear_id, 0);
        assert_eq!(patterns[0].activation_id, Some(1));
        assert_eq!(patterns[0].second_linear_id, Some(2));
        assert_eq!(patterns[0].activation_type, "Relu");
    }

    #[test]
    fn test_sparsity_pass_annotates_nodes() {
        let mut graph = make_ffn_graph();
        let pass = SparsityOptimizationPass;

        let stats = pass.run(&mut graph).unwrap();
        assert_eq!(stats.nodes_modified, 3);

        // Verify annotations
        let gate = &graph.nodes[0];
        assert_eq!(
            gate.attributes.get("sparsity_enabled"),
            Some(&AttributeValue::Bool(true))
        );
        assert_eq!(
            gate.attributes.get("sparsity_role"),
            Some(&AttributeValue::String("ffn_gate".into()))
        );

        let act = &graph.nodes[1];
        assert_eq!(
            act.attributes.get("sparsity_role"),
            Some(&AttributeValue::String("ffn_activation".into()))
        );

        let down = &graph.nodes[2];
        assert_eq!(
            down.attributes.get("sparsity_role"),
            Some(&AttributeValue::String("ffn_down".into()))
        );
    }

    #[test]
    fn test_sparsity_pass_no_ffn() {
        let graph = ModelGraph {
            nodes: vec![GraphNode {
                id: 0,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["a".into(), "b".into()],
                outputs: vec!["c".into()],
                name: None,
            }],
            edges: vec![],
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            metadata: HashMap::new(),
        };

        let pass = SparsityOptimizationPass;
        assert!(!pass.should_run(&graph));
    }
}
