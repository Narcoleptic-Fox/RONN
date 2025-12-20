//! Unit tests for activation operators
//! Tests: Relu, Sigmoid, Tanh, Softmax, Gelu

use ronn_core::NodeAttribute;
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_onnx::{GeluOp, OnnxOperator, ReluOp, SigmoidOp, SoftmaxOp, TanhOp};
use std::collections::HashMap;

const EPSILON: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

fn approx_eq_vec(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| approx_eq(*x, *y, epsilon))
}

#[test]
fn test_relu_basic() {
    let op = ReluOp;

    // Test with mixed positive and negative values
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let input = Tensor::from_data(data, vec![5], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results.len(), 1);

    let output_data = results[0].to_vec().unwrap();
    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

    assert!(approx_eq_vec(&output_data, &expected, EPSILON));
}

#[test]
fn test_relu_all_positive() {
    let op = ReluOp;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(
        data.clone(),
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // All positive values should remain unchanged
    assert!(approx_eq_vec(&output_data, &data, EPSILON));
}

#[test]
fn test_relu_all_negative() {
    let op = ReluOp;

    let data = vec![-1.0, -2.0, -3.0, -4.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // All negative values should become zero
    assert!(output_data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_relu_wrong_input_count() {
    let op = ReluOp;

    let input1 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input2 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_sigmoid_basic() {
    let op = SigmoidOp;

    let data = vec![0.0, 1.0, -1.0, 2.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results.len(), 1);

    let output_data = results[0].to_vec().unwrap();

    // sigmoid(0) = 0.5
    assert!(approx_eq(output_data[0], 0.5, EPSILON));

    // sigmoid(x) should be in (0, 1) for all x
    assert!(output_data.iter().all(|&x| x > 0.0 && x < 1.0));

    // sigmoid(-x) = 1 - sigmoid(x)
    assert!(approx_eq(output_data[1] + output_data[2], 1.0, EPSILON));
}

#[test]
fn test_sigmoid_extreme_values() {
    let op = SigmoidOp;

    // Test with large positive and negative values
    let data = vec![-10.0, 10.0];
    let input = Tensor::from_data(data, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // sigmoid(-10) ≈ 0
    assert!(output_data[0] < 0.0001);
    // sigmoid(10) ≈ 1
    assert!(output_data[1] > 0.9999);
}

#[test]
fn test_tanh_basic() {
    let op = TanhOp;

    let data = vec![0.0, 1.0, -1.0, 2.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // tanh(0) = 0
    assert!(approx_eq(output_data[0], 0.0, EPSILON));

    // tanh(x) should be in (-1, 1) for all x
    assert!(output_data.iter().all(|&x| x > -1.0 && x < 1.0));

    // tanh is odd: tanh(-x) = -tanh(x)
    assert!(approx_eq(output_data[1], -output_data[2], EPSILON));
}

#[test]
fn test_tanh_extreme_values() {
    let op = TanhOp;

    let data = vec![-10.0, 10.0];
    let input = Tensor::from_data(data, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // tanh(-10) ≈ -1
    assert!(output_data[0] < -0.9999);
    // tanh(10) ≈ 1
    assert!(output_data[1] > 0.9999);
}

#[test]
fn test_softmax_1d() {
    let op = SoftmaxOp;

    let data = vec![1.0, 2.0, 3.0];
    let input = Tensor::from_data(data, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // Softmax output should sum to 1
    let sum: f32 = output_data.iter().sum();
    assert!(approx_eq(sum, 1.0, EPSILON));

    // All values should be positive
    assert!(output_data.iter().all(|&x| x > 0.0));

    // Output should be monotonic with input
    assert!(output_data[0] < output_data[1]);
    assert!(output_data[1] < output_data[2]);
}

#[test]
fn test_softmax_2d_default_axis() {
    let op = SoftmaxOp;

    // 2x3 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_data(data, vec![2, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // Check that each row sums to approximately 1
    let row1_sum: f32 = output_data[0..3].iter().sum();
    let row2_sum: f32 = output_data[3..6].iter().sum();

    assert!(approx_eq(row1_sum, 1.0, EPSILON));
    assert!(approx_eq(row2_sum, 1.0, EPSILON));
}

#[test]
fn test_softmax_with_axis() {
    let op = SoftmaxOp;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(0));

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // All values should be positive and in (0, 1)
    assert!(output_data.iter().all(|&x| x > 0.0 && x < 1.0));
}

#[test]
fn test_softmax_numerical_stability() {
    let op = SoftmaxOp;

    // Large values that could cause overflow without proper implementation
    let data = vec![1000.0, 1001.0, 1002.0];
    let input = Tensor::from_data(data, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes);

    // Should not panic or produce NaN/Inf
    assert!(results.is_ok());
    let output_data = results.unwrap()[0].to_vec().unwrap();
    assert!(output_data.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_gelu_basic() {
    let op = GeluOp;

    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let input = Tensor::from_data(data, vec![5], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // GELU(0) = 0
    assert!(approx_eq(output_data[2], 0.0, 0.01));

    // GELU should be approximately linear for positive values far from 0
    // For x > 3, GELU(x) ≈ x
    // For our test, GELU(2) should be close to 2
    assert!(output_data[4] > 1.8 && output_data[4] < 2.1);

    // GELU is odd-like (approximately)
    // GELU(-x) ≈ -GELU(x) for large |x|
    assert!(output_data[0].signum() == -1.0);
}

#[test]
fn test_gelu_properties() {
    let op = GeluOp;

    let data = vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    let input =
        Tensor::from_data(data.clone(), vec![6], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output_data = results[0].to_vec().unwrap();

    // GELU(x) ≤ x for x ≥ 0
    for (i, &val) in data.iter().enumerate() {
        if val >= 0.0 {
            assert!(output_data[i] <= val + 0.01); // Small tolerance for numerical errors
        }
    }

    // GELU should be monotonically increasing
    for i in 0..output_data.len() - 1 {
        assert!(output_data[i] < output_data[i + 1]);
    }
}

#[test]
fn test_gelu_wrong_input_count() {
    let op = GeluOp;

    let input1 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input2 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_activations_preserve_shape() {
    // Test that all activations preserve input shape
    let shapes = vec![vec![5], vec![2, 3], vec![2, 3, 4]];

    for shape in shapes {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();

        let input =
            Tensor::from_data(data, shape.clone(), DataType::F32, TensorLayout::RowMajor).unwrap();

        let inputs = vec![&input];
        let attributes = HashMap::new();

        // Test Relu
        let result = ReluOp.execute(&inputs, &attributes).unwrap();
        assert_eq!(result[0].shape(), shape);

        // Test Sigmoid
        let result = SigmoidOp.execute(&inputs, &attributes).unwrap();
        assert_eq!(result[0].shape(), shape);

        // Test Tanh
        let result = TanhOp.execute(&inputs, &attributes).unwrap();
        assert_eq!(result[0].shape(), shape);

        // Test Gelu
        let result = GeluOp.execute(&inputs, &attributes).unwrap();
        assert_eq!(result[0].shape(), shape);
    }
}
