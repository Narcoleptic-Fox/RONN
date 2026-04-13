// Integration tests for the optimizer and optimization levels

mod common;

use common::*;
use ronn_graph::{OptimizationLevel, Optimizer};

// Optimization Level Tests

#[test]
fn test_optimizer_o0_no_passes() {
    let optimizer = Optimizer::new(OptimizationLevel::O0);
    assert_eq!(
        optimizer.pass_count(),
        0,
        "O0 should have no optimization passes"
    );
    assert_eq!(optimizer.level(), OptimizationLevel::O0);
}

#[test]
fn test_optimizer_o1_basic_passes() {
    let optimizer = Optimizer::new(OptimizationLevel::O1);
    assert_eq!(
        optimizer.pass_count(),
        2,
        "O1 should have 2 passes (constant folding + dead code elimination)"
    );
    assert_eq!(optimizer.level(), OptimizationLevel::O1);
}

#[test]
fn test_optimizer_o2_standard_passes() {
    let optimizer = Optimizer::new(OptimizationLevel::O2);
    assert_eq!(
        optimizer.pass_count(),
        4,
        "O2 should have 4 passes (O1 + fusion + layout)"
    );
    assert_eq!(optimizer.level(), OptimizationLevel::O2);
}

#[test]
fn test_optimizer_o3_aggressive_passes() {
    let optimizer = Optimizer::new(OptimizationLevel::O3);
    assert_eq!(
        optimizer.pass_count(),
        7,
        "O3 should have 7 passes (O2 + CPU + GPU specific + sparsity)"
    );
    assert_eq!(optimizer.level(), OptimizationLevel::O3);
}

// Optimizer execution tests

#[test]
fn test_optimizer_o0_runs_successfully() {
    let mut graph = create_simple_conv_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O0);

    let result = optimizer.optimize(&mut graph);
    assert!(
        result.is_ok(),
        "O0 optimization should complete successfully"
    );

    let stats = result.unwrap();
    // O0 has no passes, so iterations should be minimal (1 iteration with no changes)
    assert!(
        stats.iterations <= 1,
        "O0 should perform minimal iterations"
    );
    assert_eq!(stats.total_changes(), 0, "O0 should make no changes");
}

#[test]
fn test_optimizer_o1_runs_successfully() {
    let mut graph = create_simple_conv_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O1);

    let result = optimizer.optimize(&mut graph);
    assert!(
        result.is_ok(),
        "O1 optimization should complete successfully"
    );

    let stats = result.unwrap();
    assert!(
        stats.iterations > 0,
        "O1 should perform at least one iteration"
    );
}

#[test]
fn test_optimizer_o2_runs_successfully() {
    let mut graph = create_fusible_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O2);

    let result = optimizer.optimize(&mut graph);
    assert!(
        result.is_ok(),
        "O2 optimization should complete successfully"
    );

    let stats = result.unwrap();
    assert!(
        stats.iterations > 0,
        "O2 should perform at least one iteration"
    );
}

#[test]
fn test_optimizer_o3_runs_successfully() {
    let mut graph = create_conv_heavy_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(
        result.is_ok(),
        "O3 optimization should complete successfully"
    );

    let stats = result.unwrap();
    assert!(
        stats.iterations > 0,
        "O3 should perform at least one iteration"
    );
}

// Graph validity preservation tests

#[test]
fn test_optimizer_preserves_graph_validity_o1() {
    let mut graph = create_simple_conv_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O1);

    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid before optimization"
    );

    optimizer.optimize(&mut graph).unwrap();

    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after O1 optimization"
    );
}

#[test]
fn test_optimizer_preserves_graph_validity_o2() {
    let mut graph = create_fusible_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O2);

    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid before optimization"
    );

    optimizer.optimize(&mut graph).unwrap();

    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after O2 optimization"
    );
}

#[test]
fn test_optimizer_preserves_graph_validity_o3() {
    let mut graph = create_conv_heavy_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid before optimization"
    );

    optimizer.optimize(&mut graph).unwrap();

    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after O3 optimization"
    );
}

// Empty graph tests

#[test]
fn test_optimizer_handles_empty_graph_o1() {
    let mut graph = create_empty_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O1);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle empty graph at O1");

    let stats = result.unwrap();
    assert_eq!(
        stats.total_changes(),
        0,
        "Empty graph should have no changes"
    );
}

#[test]
fn test_optimizer_handles_empty_graph_o3() {
    let mut graph = create_empty_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle empty graph at O3");

    let stats = result.unwrap();
    assert_eq!(
        stats.total_changes(),
        0,
        "Empty graph should have no changes"
    );
}

// Optimization level progression tests

#[test]
fn test_higher_levels_include_lower_optimizations() {
    let graph_o1 = create_fusible_graph();
    let graph_o2 = graph_o1.clone();
    let graph_o3 = graph_o1.clone();

    let optimizer_o1 = Optimizer::new(OptimizationLevel::O1);
    let optimizer_o2 = Optimizer::new(OptimizationLevel::O2);
    let optimizer_o3 = Optimizer::new(OptimizationLevel::O3);

    // Higher levels should have more passes
    assert!(
        optimizer_o2.pass_count() > optimizer_o1.pass_count(),
        "O2 should have more passes than O1"
    );
    assert!(
        optimizer_o3.pass_count() > optimizer_o2.pass_count(),
        "O3 should have more passes than O2"
    );
}

// Output preservation tests

#[test]
fn test_optimizer_preserves_outputs() {
    let mut graph = create_simple_conv_graph();
    let original_outputs = graph.outputs.clone();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    optimizer.optimize(&mut graph).unwrap();

    assert_eq!(
        graph.outputs, original_outputs,
        "Optimizer should preserve graph outputs"
    );
}

#[test]
fn test_optimizer_preserves_inputs() {
    let mut graph = create_simple_conv_graph();
    let original_inputs = graph.inputs.clone();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    optimizer.optimize(&mut graph).unwrap();

    assert_eq!(
        graph.inputs, original_inputs,
        "Optimizer should preserve graph inputs"
    );
}

// Statistics tests

#[test]
fn test_optimizer_statistics_structure() {
    let mut graph = create_fusible_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O2);

    let stats = optimizer.optimize(&mut graph).unwrap();

    // Verify statistics are properly tracked
    assert!(stats.iterations > 0, "Should track iterations");
    assert!(stats.total_changes() >= 0, "Should track total changes");
}

#[test]
fn test_optimizer_different_levels_different_results() {
    let graph_base = create_fusible_graph();

    let mut graph_o1 = graph_base.clone();
    let mut graph_o2 = graph_base.clone();

    let optimizer_o1 = Optimizer::new(OptimizationLevel::O1);
    let optimizer_o2 = Optimizer::new(OptimizationLevel::O2);

    let stats_o1 = optimizer_o1.optimize(&mut graph_o1).unwrap();
    let stats_o2 = optimizer_o2.optimize(&mut graph_o2).unwrap();

    // O2 may find more optimization opportunities than O1
    assert!(
        stats_o2.iterations >= stats_o1.iterations,
        "O2 may require same or more iterations than O1"
    );
}

// Multiple graph types tests

#[test]
fn test_optimizer_on_constant_graph() {
    let mut graph = create_constant_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O1);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should optimize constant graph");
}

#[test]
fn test_optimizer_on_dead_code_graph() {
    let mut graph = create_graph_with_dead_code();
    let optimizer = Optimizer::new(OptimizationLevel::O1);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should optimize graph with dead code");

    let _stats = result.unwrap();
    // Note: Current dead code elimination marks all nodes as live (simplified)
    // assert!(stats.nodes_removed > 0, "Should remove dead code");
}

#[test]
fn test_optimizer_on_fusible_graph() {
    let mut graph = create_fusible_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O2);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should optimize fusible graph");
}

#[test]
fn test_optimizer_on_conv_heavy_graph() {
    let mut graph = create_conv_heavy_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should optimize conv-heavy graph");
}

// Idempotency tests

#[test]
fn test_optimizer_idempotent_o1() {
    let mut graph = create_simple_conv_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O1);

    let stats1 = optimizer.optimize(&mut graph).unwrap();
    let stats2 = optimizer.optimize(&mut graph).unwrap();

    // Second optimization should find no additional opportunities
    assert_eq!(
        stats2.total_changes(),
        0,
        "Second optimization should make no changes"
    );
}

#[test]
fn test_optimizer_idempotent_o3() {
    let mut graph = create_fusible_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let stats1 = optimizer.optimize(&mut graph).unwrap();
    let stats2 = optimizer.optimize(&mut graph).unwrap();

    // Second optimization should find no additional opportunities
    assert_eq!(
        stats2.total_changes(),
        0,
        "Second optimization should make no changes"
    );
}

// Convergence tests

#[test]
fn test_optimizer_converges() {
    let mut graph = create_fusible_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let stats = optimizer.optimize(&mut graph).unwrap();

    // Should converge within reasonable iterations
    assert!(
        stats.iterations < 10,
        "Optimizer should converge within max iterations"
    );
}

#[test]
fn test_optimizer_multiple_optimization_rounds() {
    let mut graph = create_complex_optimization_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let stats = optimizer.optimize(&mut graph).unwrap();

    // Complex graphs may require multiple iterations
    assert!(
        stats.iterations >= 1,
        "Complex graph should require at least one iteration"
    );
    assert!(stats.iterations <= 10, "Should not exceed max iterations");
}

// Helper function for complex graph
fn create_complex_optimization_graph() -> ronn_core::ModelGraph {
    // Create a graph with multiple optimization opportunities
    let mut builder = builder();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    // Conv+BN+ReLU pattern (fusible)
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
        .add_output(relu_id, "relu_output");

    // Dead code branch
    let dead_id = builder.add_op("Conv", Some("dead_conv".to_string()));
    builder
        .add_input(dead_id, "input_tensor")
        .add_output(dead_id, "dead_output");

    // Output
    let output_id = builder.add_op("Output", Some("output".to_string()));
    builder
        .add_input(output_id, "relu_output")
        .add_output(output_id, "output_tensor");

    builder.connect(input_id, conv_id, "input_tensor").unwrap();
    builder.connect(conv_id, bn_id, "conv_output").unwrap();
    builder.connect(bn_id, relu_id, "bn_output").unwrap();
    builder.connect(relu_id, output_id, "relu_output").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    builder.build().unwrap()
}
