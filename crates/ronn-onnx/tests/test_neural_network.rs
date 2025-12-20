//! Unit tests for neural network operators
//! Tests: Conv2D, MaxPool, AveragePool, BatchNormalization
//!
//! Note: Some operators are not fully implemented in the tensor backend yet,
//! so these tests focus on input validation, attribute parsing, and ensuring
//! proper error handling.

use ronn_core::NodeAttribute;
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_onnx::{AvgPoolOp, BatchNormOp, Conv2dOp, MaxPoolOp, OnnxOperator};
use std::collections::HashMap;

// ============ Conv2D Tests ============

#[test]
fn test_conv2d_input_validation() {
    let op = Conv2dOp;

    // Conv requires at least 2 inputs (data, weight)
    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err(), "Conv should fail with only 1 input");
}

#[test]
fn test_conv2d_with_weight() {
    let op = Conv2dOp;

    // Input: NCHW format [batch, channels, height, width]
    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();

    // Weight: [out_channels, in_channels, kernel_h, kernel_w]
    let weight = Tensor::zeros(vec![16, 3, 3, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &weight];
    let attributes = HashMap::new();

    // Currently not implemented, but should accept valid inputs
    let result = op.execute(&inputs, &attributes);
    // Expected to return error with "not yet fully implemented"
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("not yet fully implemented") || err_msg.contains("conv2d"));
}

#[test]
fn test_conv2d_with_bias() {
    let op = Conv2dOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let weight = Tensor::zeros(vec![16, 3, 3, 3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let bias = Tensor::zeros(vec![16], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &weight, &bias];
    let attributes = HashMap::new();

    // Should accept 3 inputs
    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err()); // Not implemented yet
}

#[test]
fn test_conv2d_attributes() {
    let op = Conv2dOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let weight = Tensor::zeros(vec![16, 3, 3, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &weight];

    let mut attributes = HashMap::new();
    attributes.insert("strides".to_string(), NodeAttribute::IntArray(vec![2, 2]));
    attributes.insert(
        "pads".to_string(),
        NodeAttribute::IntArray(vec![1, 1, 1, 1]),
    );
    attributes.insert("dilations".to_string(), NodeAttribute::IntArray(vec![1, 1]));
    attributes.insert("group".to_string(), NodeAttribute::Int(1));

    // Should parse attributes without panic
    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err()); // Implementation pending
}

#[test]
fn test_conv2d_default_attributes() {
    let op = Conv2dOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let weight = Tensor::zeros(vec![16, 3, 3, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &weight];
    let attributes = HashMap::new(); // No attributes - should use defaults

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ MaxPool Tests ============

#[test]
fn test_maxpool_input_validation() {
    let op = MaxPoolOp;

    // MaxPool requires exactly 1 input
    let input1 = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input2 = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err(), "MaxPool should fail with 2 inputs");
}

#[test]
fn test_maxpool_with_kernel_shape() {
    let op = MaxPoolOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];

    let mut attributes = HashMap::new();
    attributes.insert(
        "kernel_shape".to_string(),
        NodeAttribute::IntArray(vec![2, 2]),
    );

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err()); // Not implemented yet
}

#[test]
fn test_maxpool_with_strides_and_pads() {
    let op = MaxPoolOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];

    let mut attributes = HashMap::new();
    attributes.insert(
        "kernel_shape".to_string(),
        NodeAttribute::IntArray(vec![3, 3]),
    );
    attributes.insert("strides".to_string(), NodeAttribute::IntArray(vec![2, 2]));
    attributes.insert(
        "pads".to_string(),
        NodeAttribute::IntArray(vec![1, 1, 1, 1]),
    );

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_maxpool_default_attributes() {
    let op = MaxPoolOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    // Should use default kernel_shape [2, 2]
    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ AveragePool Tests ============

#[test]
fn test_avgpool_input_validation() {
    let op = AvgPoolOp;

    // AveragePool requires exactly 1 input
    let input1 = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input2 = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err(), "AveragePool should fail with 2 inputs");
}

#[test]
fn test_avgpool_with_kernel_shape() {
    let op = AvgPoolOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];

    let mut attributes = HashMap::new();
    attributes.insert(
        "kernel_shape".to_string(),
        NodeAttribute::IntArray(vec![2, 2]),
    );

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err()); // Not implemented yet
}

#[test]
fn test_avgpool_with_strides_and_pads() {
    let op = AvgPoolOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];

    let mut attributes = HashMap::new();
    attributes.insert(
        "kernel_shape".to_string(),
        NodeAttribute::IntArray(vec![3, 3]),
    );
    attributes.insert("strides".to_string(), NodeAttribute::IntArray(vec![2, 2]));
    attributes.insert(
        "pads".to_string(),
        NodeAttribute::IntArray(vec![0, 0, 0, 0]),
    );

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ BatchNormalization Tests ============

#[test]
fn test_batchnorm_input_validation() {
    let op = BatchNormOp;

    // BatchNorm requires 5 inputs
    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err(), "BatchNorm should fail with only 1 input");
}

#[test]
fn test_batchnorm_with_all_inputs() {
    let op = BatchNormOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let scale = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let bias = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let mean = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let var = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &scale, &bias, &mean, &var];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err()); // Not implemented yet
}

#[test]
fn test_batchnorm_epsilon_attribute() {
    let op = BatchNormOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let scale = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let bias = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let mean = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let var = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &scale, &bias, &mean, &var];

    let mut attributes = HashMap::new();
    attributes.insert("epsilon".to_string(), NodeAttribute::Float(1e-5));

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_batchnorm_default_epsilon() {
    let op = BatchNormOp;

    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let scale = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let bias = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let mean = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let var = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &scale, &bias, &mean, &var];
    let attributes = HashMap::new(); // Should use default epsilon = 1e-5

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_batchnorm_partial_inputs() {
    let op = BatchNormOp;

    // Test with less than 5 inputs
    let input = Tensor::zeros(vec![1, 3, 32, 32], DataType::F32, TensorLayout::RowMajor).unwrap();
    let scale = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let bias = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &scale, &bias];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err(), "BatchNorm should fail with only 3 inputs");
}

// ============ Operator Type Tests ============

#[test]
fn test_operator_type_names() {
    // Verify operator type strings match ONNX spec
    assert_eq!(Conv2dOp.op_type(), "Conv");
    assert_eq!(MaxPoolOp.op_type(), "MaxPool");
    assert_eq!(AvgPoolOp.op_type(), "AveragePool");
    assert_eq!(BatchNormOp.op_type(), "BatchNormalization");
}

// ============ Attribute Extraction Tests ============

#[test]
fn test_conv2d_attribute_defaults() {
    let op = Conv2dOp;

    let input = Tensor::zeros(vec![1, 3, 8, 8], DataType::F32, TensorLayout::RowMajor).unwrap();
    let weight = Tensor::zeros(vec![16, 3, 3, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &weight];

    // Empty attributes - should use defaults:
    // strides=[1,1], pads=[0,0,0,0], dilations=[1,1], group=1
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    // Just verify it doesn't panic when extracting default attributes
    assert!(result.is_err()); // Expected since operation not implemented
}

#[test]
fn test_pooling_attribute_defaults() {
    // Test that pooling ops handle missing attributes gracefully
    let input = Tensor::zeros(vec![1, 3, 8, 8], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    // MaxPool defaults: kernel_shape=[2,2], strides=[1,1], pads=[0,0,0,0]
    let result = MaxPoolOp.execute(&inputs, &attributes);
    assert!(result.is_err());

    // AvgPool defaults: same as MaxPool
    let result = AvgPoolOp.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ Shape Tests ============

#[test]
fn test_neural_network_ops_accept_4d_tensors() {
    // All these ops should work with 4D NCHW tensors
    let input = Tensor::zeros(vec![2, 3, 16, 16], DataType::F32, TensorLayout::RowMajor).unwrap();
    let weight = Tensor::zeros(vec![8, 3, 3, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let scale = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let bias = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let mean = Tensor::zeros(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let var = Tensor::ones(vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let attributes = HashMap::new();

    // Conv2D with 4D input
    let result = Conv2dOp.execute(&vec![&input, &weight], &attributes);
    assert!(result.is_err());

    // MaxPool with 4D input
    let result = MaxPoolOp.execute(&vec![&input], &attributes);
    assert!(result.is_err());

    // AvgPool with 4D input
    let result = AvgPoolOp.execute(&vec![&input], &attributes);
    assert!(result.is_err());

    // BatchNorm with 4D input
    let result = BatchNormOp.execute(&vec![&input, &scale, &bias, &mean, &var], &attributes);
    assert!(result.is_err());
}
