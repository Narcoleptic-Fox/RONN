//! Unit tests for mathematical operators
//! Tests: Add, Sub, Mul, Div, MatMul

use ronn_core::NodeAttribute;
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_onnx::{AddOp, DivOp, MatMulOp, MulOp, OnnxOperator, SubOp};
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

// ============ Add Tests ============

#[test]
fn test_add_same_shape() {
    let op = AddOp;

    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![6.0, 8.0, 10.0, 12.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_add_broadcast_scalar() {
    let op = AddOp;

    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![10.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![11.0, 12.0, 13.0, 14.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_add_broadcast_vector() {
    let op = AddOp;

    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2 = vec![10.0, 20.0];

    let input1 =
        Tensor::from_data(data1, vec![3, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    // Each row of input1 gets [10, 20] added
    let expected = vec![11.0, 22.0, 13.0, 24.0, 15.0, 26.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_add_wrong_input_count() {
    let op = AddOp;

    let input = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ Sub Tests ============

#[test]
fn test_sub_same_shape() {
    let op = SubOp;

    let data1 = vec![10.0, 20.0, 30.0, 40.0];
    let data2 = vec![1.0, 2.0, 3.0, 4.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![9.0, 18.0, 27.0, 36.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_sub_broadcast() {
    let op = SubOp;

    let data1 = vec![10.0, 20.0, 30.0, 40.0];
    let data2 = vec![5.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![5.0, 15.0, 25.0, 35.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_sub_negative_results() {
    let op = SubOp;

    let data1 = vec![1.0, 2.0];
    let data2 = vec![5.0, 3.0];

    let input1 = Tensor::from_data(data1, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![-4.0, -1.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

// ============ Mul Tests ============

#[test]
fn test_mul_same_shape() {
    let op = MulOp;

    let data1 = vec![2.0, 3.0, 4.0, 5.0];
    let data2 = vec![1.0, 2.0, 3.0, 4.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![2.0, 6.0, 12.0, 20.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_mul_broadcast_scalar() {
    let op = MulOp;

    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![2.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![2.0, 4.0, 6.0, 8.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_mul_with_zero() {
    let op = MulOp;

    let data1 = vec![1.0, 2.0, 3.0];
    let data2 = vec![0.0, 1.0, 0.0];

    let input1 = Tensor::from_data(data1, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![0.0, 2.0, 0.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

// ============ Div Tests ============

#[test]
fn test_div_same_shape() {
    let op = DivOp;

    let data1 = vec![10.0, 20.0, 30.0, 40.0];
    let data2 = vec![2.0, 4.0, 5.0, 8.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![5.0, 5.0, 6.0, 5.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_div_broadcast_scalar() {
    let op = DivOp;

    let data1 = vec![10.0, 20.0, 30.0, 40.0];
    let data2 = vec![2.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![5.0, 10.0, 15.0, 20.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_div_fractional_results() {
    let op = DivOp;

    let data1 = vec![1.0, 2.0, 3.0];
    let data2 = vec![2.0, 3.0, 4.0];

    let input1 = Tensor::from_data(data1, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    let expected = vec![0.5, 2.0 / 3.0, 0.75];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

// ============ MatMul Tests ============

#[test]
fn test_matmul_2d_basic() {
    let op = MatMulOp;

    // 2x3 matrix
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // 3x2 matrix
    let data2 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![3, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    // Result should be 2x2
    assert_eq!(results[0].shape(), vec![2, 2]);

    // Manual calculation:
    // [1, 2, 3]   [7,  8 ]   [1*7+2*9+3*11,  1*8+2*10+3*12]   [58,  64]
    // [4, 5, 6] × [9,  10] = [4*7+5*9+6*11,  4*8+5*10+6*12] = [139, 154]
    //             [11, 12]
    let expected = vec![58.0, 64.0, 139.0, 154.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_matmul_square_matrices() {
    let op = MatMulOp;

    // 2x2 matrices
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
    // [3, 4] × [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_matmul_vector_matrix() {
    let op = MatMulOp;

    // 1x3 vector
    let data1 = vec![1.0, 2.0, 3.0];
    // 3x2 matrix
    let data2 = vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let input1 =
        Tensor::from_data(data1, vec![1, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![3, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    // Result should be 1x2
    assert_eq!(results[0].shape(), vec![1, 2]);

    // [1, 2, 3] × [4, 5]   = [1*4+2*6+3*8, 1*5+2*7+3*9] = [40, 46]
    //             [6, 7]
    //             [8, 9]
    let expected = vec![40.0, 46.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_matmul_identity() {
    let op = MatMulOp;

    // Identity matrix
    let data1 = vec![1.0, 0.0, 0.0, 1.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(
        data2.clone(),
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    let output = results[0].to_vec().unwrap();

    // Multiplying by identity should return the original matrix
    assert!(approx_eq_vec(&output, &data2, EPSILON));
}

#[test]
fn test_matmul_wrong_input_count() {
    let op = MatMulOp;

    let input = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_elementwise_commutativity() {
    // Test that Add and Mul are commutative
    let data1 = vec![1.0, 2.0, 3.0];
    let data2 = vec![4.0, 5.0, 6.0];

    let input1 = Tensor::from_data(data1, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let attributes = HashMap::new();

    // Test Add commutativity
    let add_op = AddOp;
    let result1 = add_op
        .execute(&vec![&input1, &input2], &attributes)
        .unwrap();
    let result2 = add_op
        .execute(&vec![&input2, &input1], &attributes)
        .unwrap();
    assert!(approx_eq_vec(
        &result1[0].to_vec().unwrap(),
        &result2[0].to_vec().unwrap(),
        EPSILON
    ));

    // Test Mul commutativity
    let mul_op = MulOp;
    let result1 = mul_op
        .execute(&vec![&input1, &input2], &attributes)
        .unwrap();
    let result2 = mul_op
        .execute(&vec![&input2, &input1], &attributes)
        .unwrap();
    assert!(approx_eq_vec(
        &result1[0].to_vec().unwrap(),
        &result2[0].to_vec().unwrap(),
        EPSILON
    ));
}
