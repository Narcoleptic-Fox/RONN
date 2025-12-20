//! Comprehensive tests for tensor shape operations.
//!
//! This module tests all shape manipulation operations including:
//! - Reshape, flatten, squeeze, unsqueeze
//! - Permute, expand, view
//! - Concat, stack, split, slice
//! - Edge cases and error handling

mod test_utils;

use anyhow::Result;
use ronn_core::{DataType, ShapeOps, Tensor, TensorLayout};
use test_utils::*;

#[test]
fn test_reshape_basic() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let reshaped = a.reshape(&[3, 2])?;
    assert_eq!(reshaped.shape(), vec![3, 2]);
    assert_eq!(reshaped.numel(), 6);
    Ok(())
}

#[test]
fn test_reshape_to_1d() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let reshaped = a.reshape(&[4])?;
    assert_eq!(reshaped.shape(), vec![4]);
    assert_tensor_eq(&reshaped, &[1.0, 2.0, 3.0, 4.0])?;
    Ok(())
}

#[test]
fn test_reshape_to_4d() -> Result<()> {
    let a = create_sequential_tensor(vec![24], DataType::F32)?;

    let reshaped = a.reshape(&[2, 3, 2, 2])?;
    assert_eq!(reshaped.shape(), vec![2, 3, 2, 2]);
    assert_eq!(reshaped.numel(), 24);
    Ok(())
}

#[test]
fn test_reshape_incompatible_size() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // Cannot reshape 4 elements to 6 elements
    assert!(a.reshape(&[2, 3]).is_err());
}

#[test]
fn test_flatten() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let flattened = a.flatten()?;
    assert_eq!(flattened.shape(), vec![8]);
    assert_tensor_eq(&flattened, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    Ok(())
}

#[test]
fn test_flatten_from_dimension() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let flattened = a.flatten_from(1)?;
    assert_eq!(flattened.shape(), vec![2, 4]);
    Ok(())
}

#[test]
fn test_squeeze_all_ones() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1, 2, 2, 1],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let squeezed = a.squeeze(None)?;
    assert_eq!(squeezed.shape(), vec![2, 2]);
    Ok(())
}

#[test]
fn test_squeeze_specific_dimension() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1, 2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let squeezed = a.squeeze_dim(0)?;
    assert_eq!(squeezed.shape(), vec![2, 2]);
    Ok(())
}

#[test]
fn test_squeeze_wrong_dimension() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // Cannot squeeze dimension that's not 1
    assert!(a.squeeze_dim(0).is_err());
}

#[test]
fn test_unsqueeze() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let unsqueezed = a.unsqueeze(&[0])?;
    assert_eq!(unsqueezed.shape(), vec![1, 2, 2]);

    let unsqueezed2 = a.unsqueeze(&[2])?;
    assert_eq!(unsqueezed2.shape(), vec![2, 2, 1]);
    Ok(())
}

#[test]
fn test_unsqueeze_multiple() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.unsqueeze(&[0])?.unsqueeze(&[2])?;
    assert_eq!(result.shape(), vec![1, 2, 1]);
    Ok(())
}

#[test]
fn test_permute() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let permuted = a.permute(&[1, 0])?;
    assert_eq!(permuted.shape(), vec![3, 2]);
    Ok(())
}

#[test]
fn test_permute_3d() -> Result<()> {
    let a = create_sequential_tensor(vec![2, 3, 4], DataType::F32)?;

    let permuted = a.permute(&[2, 0, 1])?;
    assert_eq!(permuted.shape(), vec![4, 2, 3]);
    Ok(())
}

#[test]
fn test_permute_invalid() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // Invalid permutation (repeated index)
    assert!(a.permute(&[0, 0]).is_err());

    // Wrong number of dimensions
    assert!(a.permute(&[0, 1, 2]).is_err());
}

#[test]
fn test_expand() -> Result<()> {
    let a = Tensor::ones(vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;

    let expanded = a.expand(&[4, 3])?;
    assert_eq!(expanded.shape(), vec![4, 3]);
    Ok(())
}

#[test]
fn test_expand_invalid() {
    let a = Tensor::ones(vec![3, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    // Cannot expand 3 to 2
    assert!(a.expand(&[2, 2]).is_err());
}

#[test]
fn test_slice_basic() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let sliced = a.slice(0, 2, 5)?;
    assert_eq!(sliced.shape(), vec![3]);
    assert_tensor_eq(&sliced, &[3.0, 4.0, 5.0])?;
    Ok(())
}

#[test]
fn test_slice_2d() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let sliced = a.slice(1, 1, 3)?;
    assert_eq!(sliced.shape(), vec![2, 2]);
    Ok(())
}

#[test]
fn test_slice_invalid_range() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // End > length
    assert!(a.slice(0, 0, 10).is_err());

    // Start >= end
    assert!(a.slice(0, 3, 2).is_err());
}

#[test]
fn test_concat_same_size() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let concat_0 = Tensor::concat(&[&a, &b], 0)?;
    assert_eq!(concat_0.shape(), vec![4, 2]);

    let concat_1 = Tensor::concat(&[&a, &b], 1)?;
    assert_eq!(concat_1.shape(), vec![2, 4]);
    Ok(())
}

#[test]
fn test_concat_multiple_tensors() -> Result<()> {
    let a = create_ones_tensor(vec![2, 3], DataType::F32)?;
    let b = create_ones_tensor(vec![2, 3], DataType::F32)?;
    let c = create_ones_tensor(vec![2, 3], DataType::F32)?;

    let result = Tensor::concat(&[&a, &b, &c], 0)?;
    assert_eq!(result.shape(), vec![6, 3]);
    Ok(())
}

#[test]
fn test_concat_incompatible_shapes() {
    let a = Tensor::ones(vec![2, 3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let b = Tensor::ones(vec![2, 4], DataType::F32, TensorLayout::RowMajor).unwrap();

    // Cannot concat along dimension 0 when other dimensions differ
    assert!(Tensor::concat(&[&a, &b], 0).is_err());
}

#[test]
fn test_stack_basic() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let stacked_0 = Tensor::stack(&[&a, &b], 0)?;
    assert_eq!(stacked_0.shape(), vec![2, 2, 2]);

    let stacked_1 = Tensor::stack(&[&a, &b], 1)?;
    assert_eq!(stacked_1.shape(), vec![2, 2, 2]);
    Ok(())
}

#[test]
fn test_stack_multiple_tensors() -> Result<()> {
    let tensors: Vec<Tensor> = (0..5)
        .map(|_| create_ones_tensor(vec![3, 4], DataType::F32))
        .collect::<Result<Vec<_>>>()?;

    let refs: Vec<&Tensor> = tensors.iter().collect();
    let result = Tensor::stack(&refs, 0)?;
    assert_eq!(result.shape(), vec![5, 3, 4]);
    Ok(())
}

#[test]
fn test_stack_different_shapes() {
    let a = Tensor::ones(vec![2, 3], DataType::F32, TensorLayout::RowMajor).unwrap();
    let b = Tensor::ones(vec![2, 4], DataType::F32, TensorLayout::RowMajor).unwrap();

    // Cannot stack tensors with different shapes
    assert!(Tensor::stack(&[&a, &b], 0).is_err());
}

#[test]
fn test_chunk_basic() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let chunks = a.chunk(3, 0)?;
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].shape(), vec![2]);
    assert_eq!(chunks[1].shape(), vec![2]);
    assert_eq!(chunks[2].shape(), vec![2]);
    Ok(())
}

#[test]
fn test_chunk_uneven() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let chunks = a.chunk(2, 0)?;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].shape(), vec![3]); // Ceiling(5/2) = 3
    assert_eq!(chunks[1].shape(), vec![2]);
    Ok(())
}

#[test]
fn test_repeat() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let repeated = a.repeat(&[3])?;
    assert_eq!(repeated.shape(), vec![6]);
    assert_tensor_eq(&repeated, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0])?;
    Ok(())
}

#[test]
fn test_repeat_2d() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let repeated = a.repeat(&[2, 3])?;
    assert_eq!(repeated.shape(), vec![4, 6]);
    Ok(())
}

#[test]
fn test_view() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let viewed = a.view(&[3, 2])?;
    assert_eq!(viewed.shape(), vec![3, 2]);
    Ok(())
}

#[test]
fn test_squeeze_unsqueeze_roundtrip() -> Result<()> {
    let original = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let with_extra_dims = original.unsqueeze(&[0])?.unsqueeze(&[3])?;
    assert_eq!(with_extra_dims.shape(), vec![1, 2, 2, 1]);

    let back = with_extra_dims.squeeze(None)?;
    assert_eq!(back.shape(), vec![2, 2]);
    assert_tensor_approx_eq(&back, &original, 1e-6)?;
    Ok(())
}

#[test]
fn test_reshape_flatten_equivalence() -> Result<()> {
    let a = create_sequential_tensor(vec![2, 3, 4], DataType::F32)?;

    let flattened = a.flatten()?;
    let reshaped = a.reshape(&[24])?;

    assert_tensor_approx_eq(&flattened, &reshaped, 1e-6)?;
    Ok(())
}

#[test]
fn test_complex_shape_transformation() -> Result<()> {
    // Start with [2, 3, 4]
    let a = create_sequential_tensor(vec![2, 3, 4], DataType::F32)?;

    // Reshape to [6, 4]
    let step1 = a.reshape(&[6, 4])?;
    assert_eq!(step1.shape(), vec![6, 4]);

    // Add dimension [1, 6, 4]
    let step2 = step1.unsqueeze(&[0])?;
    assert_eq!(step2.shape(), vec![1, 6, 4]);

    // Permute to [6, 1, 4]
    let step3 = step2.permute(&[1, 0, 2])?;
    assert_eq!(step3.shape(), vec![6, 1, 4]);

    // Squeeze to [6, 4]
    let step4 = step3.squeeze(None)?;
    assert_eq!(step4.shape(), vec![6, 4]);

    Ok(())
}

#[test]
fn test_zero_size_tensor_shapes() -> Result<()> {
    let a = Tensor::zeros(vec![0], DataType::F32, TensorLayout::RowMajor)?;

    // Reshape zero-size tensor
    let reshaped = a.reshape(&[0])?;
    assert_eq!(reshaped.numel(), 0);

    // Flatten zero-size tensor
    let flattened = a.flatten()?;
    assert_eq!(flattened.numel(), 0);

    Ok(())
}
