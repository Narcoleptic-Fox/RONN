//! Unit tests for tensor manipulation operators
//! Tests: Reshape, Transpose, Concat, Split, Gather, Slice

use ronn_core::NodeAttribute;
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_onnx::{ConcatOp, GatherOp, OnnxOperator, ReshapeOp, SliceOp, SplitOp, TransposeOp};
use std::collections::HashMap;

const EPSILON: f32 = 1e-5;

fn approx_eq_vec(a: &[f32], b: &[f32], epsilon: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
}

// ============ Reshape Tests ============

#[test]
fn test_reshape_basic() {
    let op = ReshapeOp;

    // Original: 2x3
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_data(
        data.clone(),
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // Target shape: 3x2
    let shape_data = vec![3.0, 2.0];
    let shape_tensor =
        Tensor::from_data(shape_data, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &shape_tensor];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].shape(), vec![3, 2]);

    // Data should be preserved
    let output_data = results[0].to_vec().unwrap();
    assert!(approx_eq_vec(&output_data, &data, EPSILON));
}

#[test]
fn test_reshape_to_1d() {
    let op = ReshapeOp;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(
        data.clone(),
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    let shape_data = vec![4.0];
    let shape_tensor =
        Tensor::from_data(shape_data, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &shape_tensor];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![4]);

    let output_data = results[0].to_vec().unwrap();
    assert!(approx_eq_vec(&output_data, &data, EPSILON));
}

#[test]
fn test_reshape_to_3d() {
    let op = ReshapeOp;

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let input = Tensor::from_data(
        data.clone(),
        vec![12],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    let shape_data = vec![2.0, 3.0, 2.0];
    let shape_tensor =
        Tensor::from_data(shape_data, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &shape_tensor];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![2, 3, 2]);
}

#[test]
fn test_reshape_wrong_input_count() {
    let op = ReshapeOp;

    let input = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err(), "Reshape requires 2 inputs");
}

// ============ Transpose Tests ============

#[test]
fn test_transpose_2d_default() {
    let op = TransposeOp;

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_data(data, vec![2, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new(); // No perm attribute - should reverse dimensions

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![3, 2]);

    // [[1, 2, 3],    becomes    [[1, 4],
    //  [4, 5, 6]]                 [2, 5],
    //                             [3, 6]]
    let output_data = results[0].to_vec().unwrap();
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert!(approx_eq_vec(&output_data, &expected, EPSILON));
}

#[test]
fn test_transpose_with_perm() {
    let op = TransposeOp;

    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let input =
        Tensor::from_data(data, vec![2, 3, 4], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let mut attributes = HashMap::new();
    // Permute to [0, 2, 1] - swap last two dimensions
    attributes.insert("perm".to_string(), NodeAttribute::IntArray(vec![0, 2, 1]));

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![2, 4, 3]);
}

#[test]
fn test_transpose_1d_identity() {
    let op = TransposeOp;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input =
        Tensor::from_data(data.clone(), vec![4], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    // 1D transpose should be identity
    assert_eq!(results[0].shape(), vec![4]);

    let output_data = results[0].to_vec().unwrap();
    assert!(approx_eq_vec(&output_data, &data, EPSILON));
}

#[test]
fn test_transpose_wrong_input_count() {
    let op = TransposeOp;

    let input1 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input2 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ Concat Tests ============

#[test]
fn test_concat_1d_axis0() {
    let op = ConcatOp;

    let data1 = vec![1.0, 2.0, 3.0];
    let data2 = vec![4.0, 5.0];

    let input1 = Tensor::from_data(data1, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 = Tensor::from_data(data2, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(0));

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![5]);

    let output = results[0].to_vec().unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(approx_eq_vec(&output, &expected, EPSILON));
}

#[test]
fn test_concat_2d_axis0() {
    let op = ConcatOp;

    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![1, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(0));

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![3, 2]);
}

#[test]
fn test_concat_2d_axis1() {
    let op = ConcatOp;

    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0];

    let input1 =
        Tensor::from_data(data1, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let input2 =
        Tensor::from_data(data2, vec![2, 1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(1));

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![2, 3]);
}

#[test]
fn test_concat_multiple_tensors() {
    let op = ConcatOp;

    let data1 = vec![1.0];
    let data2 = vec![2.0];
    let data3 = vec![3.0];

    let input1 = Tensor::from_data(data1, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input2 = Tensor::from_data(data2, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input3 = Tensor::from_data(data3, vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2, &input3];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(0));

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].shape(), vec![3]);
}

#[test]
fn test_concat_no_inputs() {
    let op = ConcatOp;

    let inputs: Vec<&Tensor> = vec![];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(0));

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_concat_missing_axis() {
    let op = ConcatOp;

    let input1 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let input2 = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input1, &input2];
    let attributes = HashMap::new(); // Missing required axis attribute

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ Split Tests ============

#[test]
#[ignore] // Split operator not yet fully implemented
fn test_split_basic() {
    let op = SplitOp;

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_data(data, vec![6], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(0));
    attributes.insert("split".to_string(), NodeAttribute::IntArray(vec![3, 3]));

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].shape(), vec![3]);
    assert_eq!(results[1].shape(), vec![3]);
}

#[test]
#[ignore] // Split operator not yet fully implemented
fn test_split_default_axis() {
    let op = SplitOp;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, vec![4], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let mut attributes = HashMap::new();
    attributes.insert("split".to_string(), NodeAttribute::IntArray(vec![2, 2]));

    let results = op.execute(&inputs, &attributes);
    // Should use axis=0 by default
    assert!(results.is_ok());
}

#[test]
fn test_split_no_inputs() {
    let op = SplitOp;

    let inputs: Vec<&Tensor> = vec![];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ Gather Tests ============

#[test]
fn test_gather_input_validation() {
    let op = GatherOp;

    // Gather requires exactly 2 inputs (data, indices)
    let input = Tensor::zeros(vec![3, 4], DataType::F32, TensorLayout::RowMajor).unwrap();
    let inputs = vec![&input];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err(), "Gather requires 2 inputs");
}

#[test]
fn test_gather_with_indices() {
    let op = GatherOp;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, vec![4], DataType::F32, TensorLayout::RowMajor).unwrap();

    let indices_data = vec![0.0, 2.0, 3.0]; // Will be converted to indices
    let indices =
        Tensor::from_data(indices_data, vec![3], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &indices];
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), NodeAttribute::Int(0));

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err()); // Not fully implemented yet
}

#[test]
fn test_gather_default_axis() {
    let op = GatherOp;

    let input = Tensor::zeros(vec![3, 4], DataType::F32, TensorLayout::RowMajor).unwrap();
    let indices = Tensor::zeros(vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &indices];
    let attributes = HashMap::new(); // Should use axis=0 by default

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ Slice Tests ============

#[test]
fn test_slice_not_implemented() {
    let op = SliceOp;

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let input = Tensor::from_data(data, vec![3, 4], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let mut attributes = HashMap::new();
    attributes.insert("starts".to_string(), NodeAttribute::IntArray(vec![0]));
    attributes.insert("ends".to_string(), NodeAttribute::IntArray(vec![2]));

    let result = op.execute(&inputs, &attributes);
    // Currently returns "not yet fully implemented"
    assert!(result.is_err());
}

#[test]
fn test_slice_with_tensor_inputs() {
    let op = SliceOp;

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let input = Tensor::from_data(data, vec![3, 4], DataType::F32, TensorLayout::RowMajor).unwrap();

    let starts =
        Tensor::from_data(vec![0.0], vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();
    let ends =
        Tensor::from_data(vec![2.0], vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &starts, &ends];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

#[test]
fn test_slice_no_inputs() {
    let op = SliceOp;

    let inputs: Vec<&Tensor> = vec![];
    let attributes = HashMap::new();

    let result = op.execute(&inputs, &attributes);
    assert!(result.is_err());
}

// ============ Operator Type Tests ============

#[test]
fn test_tensor_op_type_names() {
    // Verify operator type strings match ONNX spec
    assert_eq!(ReshapeOp.op_type(), "Reshape");
    assert_eq!(TransposeOp.op_type(), "Transpose");
    assert_eq!(ConcatOp.op_type(), "Concat");
    assert_eq!(SplitOp.op_type(), "Split");
    assert_eq!(GatherOp.op_type(), "Gather");
    assert_eq!(SliceOp.op_type(), "Slice");
}

// ============ Edge Cases ============

#[test]
fn test_reshape_preserves_element_count() {
    let op = ReshapeOp;

    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let input = Tensor::from_data(
        data.clone(),
        vec![2, 3, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    let shape_data = vec![4.0, 6.0];
    let shape_tensor =
        Tensor::from_data(shape_data, vec![2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input, &shape_tensor];
    let attributes = HashMap::new();

    let results = op.execute(&inputs, &attributes).unwrap();
    assert_eq!(results[0].numel(), 24);

    let output_data = results[0].to_vec().unwrap();
    assert!(approx_eq_vec(&output_data, &data, EPSILON));
}

#[test]
fn test_transpose_twice_is_identity() {
    let op = TransposeOp;

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_data(
        data.clone(),
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    let inputs = vec![&input];
    let attributes = HashMap::new();

    // First transpose
    let results1 = op.execute(&inputs, &attributes).unwrap();
    let intermediate = &results1[0];

    // Second transpose
    let inputs2 = vec![intermediate];
    let results2 = op.execute(&inputs2, &attributes).unwrap();

    // Should be back to original shape
    assert_eq!(results2[0].shape(), vec![2, 3]);

    let output_data = results2[0].to_vec().unwrap();
    assert!(approx_eq_vec(&output_data, &data, EPSILON));
}
