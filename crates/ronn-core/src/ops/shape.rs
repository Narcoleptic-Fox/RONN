//! Shape manipulation operations for tensors.
//!
//! This module provides operations for manipulating tensor shapes including
//! reshape, flatten, squeeze, unsqueeze, and permute operations.

use crate::tensor::Tensor;
use anyhow::{Result, anyhow};

/// Trait for shape manipulation operations on tensors.
pub trait ShapeOps {
    /// Reshape the tensor to a new shape.
    fn reshape(&self, new_shape: &[usize]) -> Result<Tensor>;

    /// Flatten the tensor to 1D.
    fn flatten(&self) -> Result<Tensor>;

    /// Flatten starting from a specific dimension.
    fn flatten_from(&self, start_dim: usize) -> Result<Tensor>;

    /// Remove dimensions of size 1.
    fn squeeze(&self) -> Result<Tensor>;

    /// Remove a specific dimension of size 1.
    fn squeeze_dim(&self, dim: usize) -> Result<Tensor>;

    /// Add a dimension of size 1.
    fn unsqueeze(&self, dim: usize) -> Result<Tensor>;

    /// Permute the dimensions of the tensor.
    fn permute(&self, dims: &[usize]) -> Result<Tensor>;

    /// Expand the tensor to a new shape (broadcasting).
    fn expand(&self, new_shape: &[usize]) -> Result<Tensor>;

    /// View the tensor with a new shape (no data copy).
    fn view(&self, new_shape: &[usize]) -> Result<Tensor>;
}

impl ShapeOps for Tensor {
    fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        // Calculate total elements to verify shape compatibility
        let current_elements = self.numel();
        let new_elements: usize = new_shape.iter().product();

        if current_elements != new_elements {
            return Err(anyhow!(
                "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                current_elements,
                new_shape,
                new_elements
            ));
        }

        let result_candle = self.candle_tensor().reshape(new_shape)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn flatten(&self) -> Result<Tensor> {
        let total_elements = self.numel();
        self.reshape(&[total_elements])
    }

    fn flatten_from(&self, start_dim: usize) -> Result<Tensor> {
        let shape = self.shape();

        if start_dim >= shape.len() {
            return Err(anyhow!(
                "start_dim {} is out of bounds for tensor with {} dimensions",
                start_dim,
                shape.len()
            ));
        }

        if start_dim == 0 {
            return self.flatten();
        }

        // Keep dimensions before start_dim, flatten the rest
        let mut new_shape = shape[..start_dim].to_vec();
        let remaining_elements: usize = shape[start_dim..].iter().product();
        new_shape.push(remaining_elements);

        self.reshape(&new_shape)
    }

    fn squeeze(&self) -> Result<Tensor> {
        let shape = self.shape();
        let new_shape: Vec<usize> = shape.into_iter().filter(|&dim| dim != 1).collect();

        // If all dimensions were 1, result should be scalar (empty shape)
        if new_shape.is_empty() {
            return self.reshape(&[1]);
        }

        self.reshape(&new_shape)
    }

    fn squeeze_dim(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        if shape[dim] != 1 {
            return Err(anyhow!(
                "Cannot squeeze dimension {} with size {}",
                dim,
                shape[dim]
            ));
        }

        let mut new_shape = shape;
        new_shape.remove(dim);

        if new_shape.is_empty() {
            new_shape.push(1);
        }

        self.reshape(&new_shape)
    }

    fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim > shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for unsqueeze (max {})",
                dim,
                shape.len()
            ));
        }

        let mut new_shape = shape;
        new_shape.insert(dim, 1);

        self.reshape(&new_shape)
    }

    fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let shape = self.shape();

        if dims.len() != shape.len() {
            return Err(anyhow!(
                "Number of dimensions in permutation ({}) doesn't match tensor dimensions ({})",
                dims.len(),
                shape.len()
            ));
        }

        // Check that all dimensions are valid and unique
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_unstable();
        let expected_dims: Vec<usize> = (0..shape.len()).collect();

        if sorted_dims != expected_dims {
            return Err(anyhow!("Invalid permutation: {:?}", dims));
        }

        let result_candle = self.candle_tensor().permute(dims)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn expand(&self, new_shape: &[usize]) -> Result<Tensor> {
        let current_shape = self.shape();

        // Check if expansion is valid
        if new_shape.len() < current_shape.len() {
            return Err(anyhow!(
                "Cannot expand tensor with {} dimensions to {} dimensions",
                current_shape.len(),
                new_shape.len()
            ));
        }

        // Align dimensions from the right
        let offset = new_shape.len() - current_shape.len();
        for (i, &current_dim) in current_shape.iter().enumerate() {
            let new_dim = new_shape[offset + i];
            if current_dim != 1 && current_dim != new_dim {
                return Err(anyhow!(
                    "Cannot expand dimension {} from {} to {}",
                    offset + i,
                    current_dim,
                    new_dim
                ));
            }
        }

        let result_candle = self.candle_tensor().expand(new_shape)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn view(&self, new_shape: &[usize]) -> Result<Tensor> {
        // View is similar to reshape but requires compatible strides
        self.reshape(new_shape)
    }
}

/// Additional shape manipulation methods.
impl Tensor {
    /// Split the tensor into chunks along a specific dimension.
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Tensor>> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        let dim_size = shape[dim];
        let chunk_size = (dim_size + chunks - 1) / chunks; // Ceiling division

        let mut result = Vec::new();

        for i in 0..chunks {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, dim_size);

            if start >= dim_size {
                break;
            }

            let chunk_tensor = self.slice(dim, start, end)?;
            result.push(chunk_tensor);
        }

        Ok(result)
    }

    /// Slice the tensor along a specific dimension.
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        if start >= end || end > shape[dim] {
            return Err(anyhow!(
                "Invalid slice range: {}:{} for dimension of size {}",
                start,
                end,
                shape[dim]
            ));
        }

        let result_candle = self.candle_tensor().narrow(dim, start, end - start)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    /// Concatenate tensors along a specific dimension.
    pub fn concat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(anyhow!("Cannot concatenate empty list of tensors"));
        }

        let first_tensor = tensors[0];
        let first_shape = first_tensor.shape();

        if dim >= first_shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                first_shape.len()
            ));
        }

        // Check that all tensors have compatible shapes
        for (i, tensor) in tensors.iter().enumerate() {
            let tensor_shape = tensor.shape();
            if tensor_shape.len() != first_shape.len() {
                return Err(anyhow!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    tensor_shape.len(),
                    first_shape.len()
                ));
            }

            for (j, (&dim_size, &expected_size)) in
                tensor_shape.iter().zip(first_shape.iter()).enumerate()
            {
                if j != dim && dim_size != expected_size {
                    return Err(anyhow!(
                        "Tensor {} has size {} in dimension {}, expected {}",
                        i,
                        dim_size,
                        j,
                        expected_size
                    ));
                }
            }
        }

        let candle_tensors: Vec<&candle_core::Tensor> =
            tensors.iter().map(|t| t.candle_tensor()).collect();

        let result_candle = candle_core::Tensor::cat(&candle_tensors, dim)?;

        Ok(Tensor::from_candle(
            result_candle,
            first_tensor.dtype(),
            first_tensor.layout(),
        ))
    }

    /// Repeat the tensor along specified dimensions.
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        let shape = self.shape();

        if repeats.len() != shape.len() {
            return Err(anyhow!(
                "Number of repeats ({}) must match tensor dimensions ({})",
                repeats.len(),
                shape.len()
            ));
        }

        let result_candle = self.candle_tensor().repeat(repeats)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    /// Tile the tensor with the given multiples.
    pub fn tile(&self, multiples: &[usize]) -> Result<Tensor> {
        // Tile is similar to repeat but with different semantics
        self.repeat(multiples)
    }

    /// Pad the tensor with zeros.
    pub fn pad_zeros(&self, padding: &[(usize, usize)]) -> Result<Tensor> {
        let shape = self.shape();

        if padding.len() != shape.len() {
            return Err(anyhow!(
                "Padding length ({}) must match tensor dimensions ({})",
                padding.len(),
                shape.len()
            ));
        }

        // Calculate new shape after padding
        let new_shape: Vec<usize> = shape
            .iter()
            .zip(padding.iter())
            .map(|(&dim, &(pad_before, pad_after))| dim + pad_before + pad_after)
            .collect();

        // Create a zero tensor with the new shape
        let _padded = Tensor::zeros(new_shape, self.dtype(), self.layout())?;

        // Calculate slice ranges for placing the original tensor
        let _slice_ranges: Vec<(usize, usize)> = padding
            .iter()
            .zip(shape.iter())
            .map(|(&(pad_before, _), &dim)| (pad_before, pad_before + dim))
            .collect();

        // This is a simplified implementation - in practice, we'd need more complex indexing
        // For now, this is a placeholder that works for simple cases
        if padding
            .iter()
            .all(|&(before, after)| before == 0 && after == 0)
        {
            // No padding needed
            return Ok(self.clone());
        }

        // For non-zero padding, we'd need to implement tensor slicing assignment
        // This is a complex operation that would require more advanced indexing
        Err(anyhow!(
            "Complex padding operations not yet fully implemented"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, TensorLayout};

    #[test]
    fn test_reshape() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let reshaped = a.reshape(&[3, 2])?;
        assert_eq!(reshaped.shape(), vec![3, 2]);

        let reshaped_1d = a.reshape(&[6])?;
        assert_eq!(reshaped_1d.shape(), vec![6]);

        Ok(())
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

        let flat_from = a.flatten_from(1)?;
        assert_eq!(flat_from.shape(), vec![2, 4]);

        Ok(())
    }

    #[test]
    fn test_squeeze_unsqueeze() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 2, 2, 1],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let squeezed = a.squeeze()?;
        assert_eq!(squeezed.shape(), vec![2, 2]);

        let squeeze_dim = a.squeeze_dim(0)?;
        assert_eq!(squeeze_dim.shape(), vec![2, 2, 1]);

        let unsqueezed = squeezed.unsqueeze(0)?;
        assert_eq!(unsqueezed.shape(), vec![1, 2, 2]);

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
    fn test_slice() -> Result<()> {
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
    fn test_concat() -> Result<()> {
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
    fn test_stack() -> Result<()> {
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
    fn test_chunk() -> Result<()> {
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
    fn test_repeat() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0],
            vec![2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let repeated = a.repeat(&[3])?;
        assert_eq!(repeated.shape(), vec![6]);

        let repeated_data = repeated.to_vec()?;
        assert_eq!(repeated_data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();

        // Invalid reshape
        assert!(a.reshape(&[3, 2]).is_err());

        // Out of bounds dimension for squeeze
        assert!(a.squeeze_dim(5).is_err());

        // Cannot squeeze dimension that's not size 1
        assert!(a.squeeze_dim(0).is_err());

        // Out of bounds dimension for unsqueeze
        assert!(a.unsqueeze(5).is_err());

        // Invalid permutation
        assert!(a.permute(&[0, 0]).is_err());
        assert!(a.permute(&[0, 1, 2]).is_err());

        // Invalid slice
        assert!(a.slice(0, 5, 6).is_err());
        assert!(a.slice(0, 2, 1).is_err());

        // Empty concat
        let empty_tensors: Vec<&Tensor> = vec![];
        assert!(Tensor::concat(&empty_tensors, 0).is_err());

        // Incompatible shapes for concat
        let b = Tensor::from_data(
            vec![1.0, 2.0, 3.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();
        assert!(Tensor::concat(&[&a, &b], 0).is_err());
    }
}
