// Common test utilities for ronn-graph tests

use ronn_core::GraphBuilder;
use ronn_core::types::{AttributeValue, DataType, GraphEdge, GraphNode, ModelGraph};

/// Create a simple linear graph for testing
/// Input -> Conv -> ReLU -> Output
pub fn create_simple_conv_graph() -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv_id, "input_tensor")
        .add_output(conv_id, "conv_output")
        .add_attribute(conv_id, "kernel_size", AttributeValue::IntArray(vec![3, 3]))
        .add_attribute(conv_id, "stride", AttributeValue::IntArray(vec![1, 1]));

    let relu_id = builder.add_op("Relu", Some("relu1".to_string()));
    builder
        .add_input(relu_id, "conv_output")
        .add_output(relu_id, "output_tensor");

    builder.connect(input_id, conv_id, "input_tensor").unwrap();
    builder.connect(conv_id, relu_id, "conv_output").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    builder.build().unwrap()
}

/// Create a graph with Conv -> BatchNorm -> ReLU pattern (fusion candidate)
pub fn create_fusible_graph() -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv_id, "input_tensor")
        .add_output(conv_id, "conv_output");

    let bn_id = builder.add_op("BatchNormalization", Some("bn1".to_string()));
    builder
        .add_input(bn_id, "conv_output")
        .add_output(bn_id, "bn_output");

    let relu_id = builder.add_op("Relu", Some("relu1".to_string()));
    builder
        .add_input(relu_id, "bn_output")
        .add_output(relu_id, "output_tensor");

    builder.connect(input_id, conv_id, "input_tensor").unwrap();
    builder.connect(conv_id, bn_id, "conv_output").unwrap();
    builder.connect(bn_id, relu_id, "bn_output").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    builder.build().unwrap()
}

/// Create a graph with MatMul -> Add pattern (fusion candidate)
pub fn create_matmul_bias_graph() -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let matmul_id = builder.add_op("MatMul", Some("matmul1".to_string()));
    builder
        .add_input(matmul_id, "input_tensor")
        .add_output(matmul_id, "matmul_output");

    let add_id = builder.add_op("Add", Some("add_bias".to_string()));
    builder
        .add_input(add_id, "matmul_output")
        .add_output(add_id, "output_tensor");

    builder
        .connect(input_id, matmul_id, "input_tensor")
        .unwrap();
    builder.connect(matmul_id, add_id, "matmul_output").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    builder.build().unwrap()
}

/// Create a graph with dead code (unreachable nodes)
pub fn create_graph_with_dead_code() -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv1_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv1_id, "input_tensor")
        .add_output(conv1_id, "conv1_output");

    // Dead node - not connected to output
    let dead_id = builder.add_op("Conv", Some("dead_conv".to_string()));
    builder
        .add_input(dead_id, "input_tensor")
        .add_output(dead_id, "dead_output");

    let output_id = builder.add_op("Output", Some("output".to_string()));
    builder
        .add_input(output_id, "conv1_output")
        .add_output(output_id, "output_tensor");

    builder.connect(input_id, conv1_id, "input_tensor").unwrap();
    builder
        .connect(conv1_id, output_id, "conv1_output")
        .unwrap();

    // Note: dead_id intentionally not connected to output path

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    builder.build().unwrap()
}

/// Create an empty graph
pub fn create_empty_graph() -> ModelGraph {
    ModelGraph::new()
}

/// Create a graph with only constant operations
pub fn create_constant_graph() -> ModelGraph {
    let mut builder = GraphBuilder::new();

    // Constant initializer nodes
    let const1_id = builder.add_op("Constant", Some("const1".to_string()));
    builder
        .add_output(const1_id, "const1_output")
        .add_attribute(
            const1_id,
            "value",
            AttributeValue::FloatArray(vec![1.0, 2.0, 3.0]),
        );

    let const2_id = builder.add_op("Constant", Some("const2".to_string()));
    builder
        .add_output(const2_id, "const2_output")
        .add_attribute(
            const2_id,
            "value",
            AttributeValue::FloatArray(vec![4.0, 5.0, 6.0]),
        );

    // Add operation on constants (can be folded)
    let add_id = builder.add_op("Add", Some("add_constants".to_string()));
    builder
        .add_input(add_id, "const1_output")
        .add_input(add_id, "const2_output")
        .add_output(add_id, "output_tensor");

    builder.connect(const1_id, add_id, "const1_output").unwrap();
    builder.connect(const2_id, add_id, "const2_output").unwrap();

    builder
        .set_inputs(vec![])
        .set_outputs(vec!["output_tensor".to_string()]);

    builder.build().unwrap()
}

/// Create a graph with multiple Conv operations (layout optimization candidate)
pub fn create_conv_heavy_graph() -> ModelGraph {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let mut prev_output = "input_tensor".to_string();
    let mut prev_id = input_id;

    // Chain of 5 conv operations
    for i in 0..5 {
        let conv_id = builder.add_op("Conv", Some(format!("conv{}", i + 1)));
        let output_name = format!("conv{}_output", i + 1);

        builder
            .add_input(conv_id, &prev_output)
            .add_output(conv_id, &output_name);

        builder.connect(prev_id, conv_id, &prev_output).unwrap();

        prev_output = output_name;
        prev_id = conv_id;
    }

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec![prev_output]);

    builder.build().unwrap()
}

/// Count nodes of a specific operation type in the graph
pub fn count_nodes_by_type(graph: &ModelGraph, op_type: &str) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|n| n.op_type == op_type)
        .count()
}

/// Verify graph structure is valid
pub fn verify_graph_valid(graph: &ModelGraph) -> bool {
    graph.validate().is_ok()
}

/// Create a graph builder for custom test scenarios
pub fn builder() -> GraphBuilder {
    GraphBuilder::new()
}
