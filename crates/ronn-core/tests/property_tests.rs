//! Property-based tests for tensor operations.
//!
//! This module uses proptest to validate invariants and properties
//! that should hold for all tensor operations.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use ronn_core::{ArithmeticOps, DataType, MatrixOps, ReductionOps, ShapeOps, Tensor, TensorLayout};

// Strategy for generating valid tensor shapes
fn tensor_shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..10, 1..4)
}

// Strategy for generating tensor data
fn tensor_data_strategy(size: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0_f32..100.0_f32, size..=size)
}

// Helper to create a tensor from generated data
fn create_test_tensor(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor, anyhow::Error> {
    Tensor::from_data(data, shape, DataType::F32, TensorLayout::RowMajor)
}

proptest! {
    #[test]
    fn test_addition_commutative(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data1 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();
        let data2 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let a = create_test_tensor(shape.clone(), data1).unwrap();
        let b = create_test_tensor(shape, data2).unwrap();

        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();

        let ab_data = ab.to_vec().unwrap();
        let ba_data = ba.to_vec().unwrap();

        for (x, y) in ab_data.iter().zip(ba_data.iter()) {
            prop_assert!((x - y).abs() < 1e-5, "Addition not commutative: {} vs {}", x, y);
        }
    }

    #[test]
    fn test_addition_associative(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data1 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();
        let data2 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();
        let data3 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let a = create_test_tensor(shape.clone(), data1).unwrap();
        let b = create_test_tensor(shape.clone(), data2).unwrap();
        let c = create_test_tensor(shape, data3).unwrap();

        let ab_c = a.add(&b).unwrap().add(&c).unwrap();
        let a_bc = a.add(&b.add(&c).unwrap()).unwrap();

        let ab_c_data = ab_c.to_vec().unwrap();
        let a_bc_data = a_bc.to_vec().unwrap();

        for (x, y) in ab_c_data.iter().zip(a_bc_data.iter()) {
            prop_assert!((x - y).abs() < 1e-4, "Addition not associative: {} vs {}", x, y);
        }
    }

    #[test]
    fn test_multiplication_commutative(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data1 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();
        let data2 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let a = create_test_tensor(shape.clone(), data1).unwrap();
        let b = create_test_tensor(shape, data2).unwrap();

        let ab = a.mul(&b).unwrap();
        let ba = b.mul(&a).unwrap();

        let ab_data = ab.to_vec().unwrap();
        let ba_data = ba.to_vec().unwrap();

        for (x, y) in ab_data.iter().zip(ba_data.iter()) {
            prop_assert!((x - y).abs() < 1e-5, "Multiplication not commutative: {} vs {}", x, y);
        }
    }

    #[test]
    fn test_addition_identity(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let a = create_test_tensor(shape.clone(), data).unwrap();
        let zero = Tensor::zeros(shape, DataType::F32, TensorLayout::RowMajor).unwrap();

        let result = a.add(&zero).unwrap();

        let a_data = a.to_vec().unwrap();
        let result_data = result.to_vec().unwrap();

        for (x, y) in a_data.iter().zip(result_data.iter()) {
            prop_assert!((x - y).abs() < 1e-6, "Addition identity failed: {} vs {}", x, y);
        }
    }

    #[test]
    fn test_multiplication_identity(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let a = create_test_tensor(shape.clone(), data).unwrap();
        let one = Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap();

        let result = a.mul(&one).unwrap();

        let a_data = a.to_vec().unwrap();
        let result_data = result.to_vec().unwrap();

        for (x, y) in a_data.iter().zip(result_data.iter()) {
            prop_assert!((x - y).abs() < 1e-6, "Multiplication identity failed: {} vs {}", x, y);
        }
    }

    #[test]
    fn test_reshape_preserves_elements(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let original = create_test_tensor(shape, data.clone()).unwrap();
        let flattened = original.flatten().unwrap();

        let original_data = original.to_vec().unwrap();
        let flattened_data = flattened.to_vec().unwrap();

        prop_assert_eq!(original_data, flattened_data, "Reshape changed element values");
        prop_assert_eq!(flattened.numel(), size, "Reshape changed element count");
    }

    #[test]
    fn test_transpose_transpose_identity(
        rows in 1usize..10,
        cols in 1usize..10
    ) {
        let size = rows * cols;
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let original = create_test_tensor(vec![rows, cols], data).unwrap();
        let transposed_once = MatrixOps::transpose(&original).unwrap();
        let transposed_twice = MatrixOps::transpose(&transposed_once).unwrap();

        let original_data = original.to_vec().unwrap();
        let transposed_data = transposed_twice.to_vec().unwrap();

        for (x, y) in original_data.iter().zip(transposed_data.iter()) {
            prop_assert!((x - y).abs() < 1e-6, "Transpose twice not identity: {} vs {}", x, y);
        }
    }

    #[test]
    fn test_sum_equals_element_sum(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor = create_test_tensor(shape, data.clone()).unwrap();
        let tensor_sum = tensor.sum_all().unwrap().to_vec().unwrap()[0];
        let manual_sum: f32 = data.iter().sum();

        prop_assert!((tensor_sum - manual_sum).abs() < 1e-4, "Sum mismatch: {} vs {}", tensor_sum, manual_sum);
    }

    #[test]
    fn test_mean_equals_sum_divided_by_count(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor = create_test_tensor(shape, data.clone()).unwrap();
        let tensor_mean = tensor.mean_all().unwrap().to_vec().unwrap()[0];
        let manual_mean: f32 = data.iter().sum::<f32>() / (size as f32);

        prop_assert!((tensor_mean - manual_mean).abs() < 1e-4, "Mean mismatch: {} vs {}", tensor_mean, manual_mean);
    }

    #[test]
    fn test_abs_is_non_negative(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor = create_test_tensor(shape, data).unwrap();
        let abs_tensor = tensor.abs().unwrap();
        let abs_data = abs_tensor.to_vec().unwrap();

        for value in abs_data {
            prop_assert!(value >= 0.0, "Absolute value is negative: {}", value);
        }
    }

    #[test]
    fn test_relu_is_non_negative(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor = create_test_tensor(shape, data).unwrap();
        let relu_tensor = tensor.relu().unwrap();
        let relu_data = relu_tensor.to_vec().unwrap();

        for value in relu_data {
            prop_assert!(value >= 0.0, "ReLU produced negative value: {}", value);
        }
    }

    #[test]
    fn test_softmax_sums_to_one(
        size in 2usize..20
    ) {
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor = create_test_tensor(vec![size], data).unwrap();
        let softmax = tensor.softmax(0).unwrap();
        let softmax_data = softmax.to_vec().unwrap();

        let sum: f32 = softmax_data.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "Softmax doesn't sum to 1: {}", sum);

        // All values should be in [0, 1]
        for value in softmax_data {
            prop_assert!(value >= 0.0 && value <= 1.0, "Softmax value out of range: {}", value);
        }
    }

    #[test]
    fn test_norm_is_non_negative(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor = create_test_tensor(shape, data).unwrap();
        let norm = tensor.norm().unwrap().to_vec().unwrap()[0];

        prop_assert!(norm >= 0.0, "Norm is negative: {}", norm);
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let original = create_test_tensor(shape, data).unwrap();
        let with_dim = original.unsqueeze(&[0]).unwrap();
        let back = with_dim.squeeze(None).unwrap();

        let original_data = original.to_vec().unwrap();
        let back_data = back.to_vec().unwrap();

        prop_assert_eq!(original_data, back_data, "Squeeze/unsqueeze changed data");
    }

    #[test]
    fn test_concat_preserves_total_elements(
        size in 2usize..20
    ) {
        let data1 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();
        let data2 = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor1 = create_test_tensor(vec![size], data1).unwrap();
        let tensor2 = create_test_tensor(vec![size], data2).unwrap();

        let concatenated = Tensor::concat(&[&tensor1, &tensor2], 0).unwrap();

        prop_assert_eq!(concatenated.numel(), 2 * size, "Concat changed total elements");
    }

    #[test]
    fn test_max_greater_equal_min(
        shape in tensor_shape_strategy()
    ) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut TestRunner::default()).unwrap().current();

        let tensor = create_test_tensor(shape, data).unwrap();
        let max_val = tensor.max_all().unwrap().to_vec().unwrap()[0];
        let min_val = tensor.min_all().unwrap().to_vec().unwrap()[0];

        prop_assert!(max_val >= min_val, "Max ({}) is less than min ({})", max_val, min_val);
    }
}
