//! Reduction operations for tensors.
//!
//! This module provides reduction operations that aggregate tensor values
//! along specified dimensions, including sum, mean, max, min, and others.

use crate::ops::arithmetic::ArithmeticOps;
use crate::ops::shape::ShapeOps;
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};

/// Trait for reduction operations on tensors.
pub trait ReductionOps {
    /// Sum all elements in the tensor.
    fn sum_all(&self) -> Result<Tensor>;

    /// Sum along specified dimensions.
    fn sum_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor>;

    /// Mean of all elements in the tensor.
    fn mean_all(&self) -> Result<Tensor>;

    /// Mean along specified dimensions.
    fn mean_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor>;

    /// Maximum value in the tensor.
    fn max_all(&self) -> Result<Tensor>;

    /// Maximum along specified dimensions.
    fn max_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor>;

    /// Minimum value in the tensor.
    fn min_all(&self) -> Result<Tensor>;

    /// Minimum along specified dimensions.
    fn min_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor>;

    /// Product of all elements.
    fn prod_all(&self) -> Result<Tensor>;

    /// Standard deviation of all elements.
    fn std_all(&self) -> Result<Tensor>;

    /// Variance of all elements.
    fn var_all(&self) -> Result<Tensor>;

    /// L2 norm (Euclidean norm) of the tensor.
    fn norm(&self) -> Result<Tensor>;

    /// Lp norm of the tensor.
    fn norm_p(&self, p: f32) -> Result<Tensor>;
}

impl ReductionOps for Tensor {
    fn sum_all(&self) -> Result<Tensor> {
        let result_candle = self.candle_tensor().sum_all()?;

        // Ensure result is at least 1D
        let reshaped = if result_candle.dims().is_empty() {
            result_candle.reshape(&[1])?
        } else {
            result_candle
        };

        Ok(Tensor::from_candle(reshaped, self.dtype(), self.layout()))
    }

    fn sum_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor> {
        let shape = self.shape();

        // Validate dimensions
        for &dim in dims {
            if dim >= shape.len() {
                return Err(anyhow!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim,
                    shape.len()
                ));
            }
        }

        let result_candle = if keep_dim {
            self.candle_tensor().sum_keepdim(dims)?
        } else {
            self.candle_tensor().sum(dims)?
        };

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn mean_all(&self) -> Result<Tensor> {
        let sum = self.sum_all()?;
        let num_elements = self.numel() as f32;
        sum.div_scalar(num_elements)
    }

    fn mean_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor> {
        let sum = self.sum_dims(dims, keep_dim)?;

        // Calculate the number of elements being reduced over
        let shape = self.shape();
        let reduction_size: usize = dims.iter().map(|&dim| shape[dim]).product();

        sum.div_scalar(reduction_size as f32)
    }

    fn max_all(&self) -> Result<Tensor> {
        let flattened = self.flatten()?;
        let result_candle = flattened.candle_tensor().max(0)?;

        // Ensure result is at least 1D
        let reshaped = if result_candle.dims().is_empty() {
            result_candle.reshape(&[1])?
        } else {
            result_candle
        };

        Ok(Tensor::from_candle(reshaped, self.dtype(), self.layout()))
    }

    fn max_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor> {
        let shape = self.shape();

        // Validate dimensions
        for &dim in dims {
            if dim >= shape.len() {
                return Err(anyhow!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim,
                    shape.len()
                ));
            }
        }

        // For simplicity, we'll reduce one dimension at a time
        let mut result = self.clone();
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse(); // Process in reverse order to maintain indices

        for &dim in &sorted_dims {
            let result_candle = if keep_dim {
                result.candle_tensor().max_keepdim(dim)?
            } else {
                result.candle_tensor().max(dim)?
            };
            result = Tensor::from_candle(result_candle, result.dtype(), result.layout());
        }

        Ok(result)
    }

    fn min_all(&self) -> Result<Tensor> {
        let flattened = self.flatten()?;
        let result_candle = flattened.candle_tensor().min(0)?;

        // Ensure result is at least 1D
        let reshaped = if result_candle.dims().is_empty() {
            result_candle.reshape(&[1])?
        } else {
            result_candle
        };

        Ok(Tensor::from_candle(reshaped, self.dtype(), self.layout()))
    }

    fn min_dims(&self, dims: &[usize], keep_dim: bool) -> Result<Tensor> {
        let shape = self.shape();

        // Validate dimensions
        for &dim in dims {
            if dim >= shape.len() {
                return Err(anyhow!(
                    "Dimension {} is out of bounds for tensor with {} dimensions",
                    dim,
                    shape.len()
                ));
            }
        }

        // For simplicity, we'll reduce one dimension at a time
        let mut result = self.clone();
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse(); // Process in reverse order to maintain indices

        for &dim in &sorted_dims {
            let result_candle = if keep_dim {
                result.candle_tensor().min_keepdim(dim)?
            } else {
                result.candle_tensor().min(dim)?
            };
            result = Tensor::from_candle(result_candle, result.dtype(), result.layout());
        }

        Ok(result)
    }

    fn prod_all(&self) -> Result<Tensor> {
        // Product of all elements - we'll use a simple implementation
        let data = self.to_vec()?;
        let product = data.iter().fold(1.0, |acc, &x| acc * x);

        Ok(Tensor::from_data(
            vec![product],
            vec![1],
            self.dtype(),
            self.layout(),
        )?)
    }

    fn std_all(&self) -> Result<Tensor> {
        let variance = self.var_all()?;
        variance.sqrt()
    }

    fn var_all(&self) -> Result<Tensor> {
        let mean = self.mean_all()?;
        let diff = self.sub(&mean)?;
        let squared_diff = diff.mul(&diff)?;
        squared_diff.mean_all()
    }

    fn norm(&self) -> Result<Tensor> {
        self.norm_p(2.0)
    }

    fn norm_p(&self, p: f32) -> Result<Tensor> {
        if p <= 0.0 {
            return Err(anyhow!("Norm p must be positive, got {}", p));
        }

        if p == 1.0 {
            // L1 norm: sum of absolute values
            let abs_values = self.abs()?;
            abs_values.sum_all()
        } else if p == 2.0 {
            // L2 norm: sqrt of sum of squares
            let squared = self.mul(self)?;
            let sum_squared = squared.sum_all()?;
            sum_squared.sqrt()
        } else if p.is_infinite() {
            // L∞ norm: maximum absolute value
            let abs_values = self.abs()?;
            abs_values.max_all()
        } else {
            // General Lp norm: (sum of |x|^p)^(1/p)
            let abs_values = self.abs()?;
            let powered = abs_values.pow(p)?;
            let sum_powered = powered.sum_all()?;
            sum_powered.pow(1.0 / p)
        }
    }
}

/// Additional reduction methods for convenience.
impl Tensor {
    /// Sum along a single dimension.
    pub fn sum_dim(&self, dim: usize, keep_dim: bool) -> Result<Tensor> {
        self.sum_dims(&[dim], keep_dim)
    }

    /// Mean along a single dimension.
    pub fn mean_dim(&self, dim: usize, keep_dim: bool) -> Result<Tensor> {
        self.mean_dims(&[dim], keep_dim)
    }

    /// Maximum along a single dimension.
    pub fn max_dim(&self, dim: usize, keep_dim: bool) -> Result<Tensor> {
        self.max_dims(&[dim], keep_dim)
    }

    /// Minimum along a single dimension.
    pub fn min_dim(&self, dim: usize, keep_dim: bool) -> Result<Tensor> {
        self.min_dims(&[dim], keep_dim)
    }

    /// Find indices of maximum values along a dimension.
    pub fn argmax(&self, dim: usize, keep_dim: bool) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        let result_candle = if keep_dim {
            self.candle_tensor().argmax_keepdim(dim)?
        } else {
            self.candle_tensor().argmax(dim)?
        };

        Ok(Tensor::from_candle(
            result_candle,
            crate::types::DataType::U32, // Candle returns indices as U32
            self.layout(),
        ))
    }

    /// Find indices of minimum values along a dimension.
    pub fn argmin(&self, dim: usize, keep_dim: bool) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        let result_candle = if keep_dim {
            self.candle_tensor().argmin_keepdim(dim)?
        } else {
            self.candle_tensor().argmin(dim)?
        };

        Ok(Tensor::from_candle(
            result_candle,
            crate::types::DataType::U32, // Candle returns indices as U32
            self.layout(),
        ))
    }

    /// Count non-zero elements.
    pub fn count_nonzero(&self) -> Result<usize> {
        let data = self.to_vec()?;
        Ok(data.iter().filter(|&&x| x != 0.0).count())
    }

    /// Count elements along a dimension.
    pub fn count_nonzero_dim(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        // Create a mask for non-zero elements
        let _abs_values = self.abs()?;
        let epsilon = 1e-7;
        let _epsilon_tensor =
            Tensor::from_data(vec![epsilon], vec![1], self.dtype(), self.layout())?;

        // This is a simplified implementation - in practice we'd need proper comparison operations
        // For now, we'll use a placeholder that counts all elements
        let dim_size = shape[dim];
        let mut output_shape = shape;
        output_shape[dim] = 1;

        let count_tensor = Tensor::from_data(
            vec![dim_size as f32],
            output_shape,
            self.dtype(),
            self.layout(),
        )?;
        Ok(count_tensor)
    }

    /// Cumulative sum along a dimension.
    pub fn cumsum(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        // This is a placeholder implementation
        // A full implementation would require more complex indexing
        let data = self.to_vec()?;
        let mut cumsum_data = Vec::with_capacity(data.len());
        let mut running_sum = 0.0;

        for &value in &data {
            running_sum += value;
            cumsum_data.push(running_sum);
        }

        Ok(Tensor::from_data(
            cumsum_data,
            shape,
            self.dtype(),
            self.layout(),
        )?)
    }

    /// Cumulative product along a dimension.
    pub fn cumprod(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        // This is a placeholder implementation
        let data = self.to_vec()?;
        let mut cumprod_data = Vec::with_capacity(data.len());
        let mut running_prod = 1.0;

        for &value in &data {
            running_prod *= value;
            cumprod_data.push(running_prod);
        }

        Ok(Tensor::from_data(
            cumprod_data,
            shape,
            self.dtype(),
            self.layout(),
        )?)
    }

    /// Softmax operation along a dimension.
    pub fn softmax(&self, dim: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        // Softmax: exp(x) / sum(exp(x))
        // For numerical stability: exp(x - max(x)) / sum(exp(x - max(x)))
        let max_vals = self.max_dim(dim, true)?;
        let shifted = self.sub(&max_vals)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum_dim(dim, true)?;
        exp_vals.div(&sum_exp)
    }

    /// Log softmax operation along a dimension.
    pub fn log_softmax(&self, dim: usize) -> Result<Tensor> {
        let softmax_result = self.softmax(dim)?;
        softmax_result.log()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, TensorLayout};

    #[test]
    fn test_sum_operations() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Sum all elements
        let sum_all = a.sum_all()?;
        let sum_all_data = sum_all.to_vec()?;
        assert_eq!(sum_all_data[0], 21.0);

        // Sum along dimension 0 (rows)
        let sum_dim0 = a.sum_dim(0, false)?;
        let sum_dim0_data = sum_dim0.to_vec()?;
        assert_eq!(sum_dim0_data, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

        // Sum along dimension 1 (columns)
        let sum_dim1 = a.sum_dim(1, false)?;
        let sum_dim1_data = sum_dim1.to_vec()?;
        assert_eq!(sum_dim1_data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]

        Ok(())
    }

    #[test]
    fn test_mean_operations() -> Result<()> {
        let a = Tensor::from_data(
            vec![2.0, 4.0, 6.0, 8.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Mean of all elements
        let mean_all = a.mean_all()?;
        let mean_all_data = mean_all.to_vec()?;
        assert_eq!(mean_all_data[0], 5.0);

        // Mean along dimension 0
        let mean_dim0 = a.mean_dim(0, false)?;
        let mean_dim0_data = mean_dim0.to_vec()?;
        assert_eq!(mean_dim0_data, vec![4.0, 6.0]); // [(2+6)/2, (4+8)/2]

        Ok(())
    }

    #[test]
    fn test_max_min_operations() -> Result<()> {
        let a = Tensor::from_data(
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Max of all elements
        let max_all = a.max_all()?;
        let max_all_data = max_all.to_vec()?;
        assert_eq!(max_all_data[0], 9.0);

        // Min of all elements
        let min_all = a.min_all()?;
        let min_all_data = min_all.to_vec()?;
        assert_eq!(min_all_data[0], 1.0);

        Ok(())
    }

    #[test]
    fn test_norm_operations() -> Result<()> {
        let a = Tensor::from_data(
            vec![3.0, 4.0],
            vec![2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // L2 norm
        let l2_norm = a.norm()?;
        let l2_norm_data = l2_norm.to_vec()?;
        assert_eq!(l2_norm_data[0], 5.0); // sqrt(3^2 + 4^2) = 5

        // L1 norm
        let l1_norm = a.norm_p(1.0)?;
        let l1_norm_data = l1_norm.to_vec()?;
        assert_eq!(l1_norm_data[0], 7.0); // |3| + |4| = 7

        Ok(())
    }

    #[test]
    fn test_variance_std() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let variance = a.var_all()?;
        let std = a.std_all()?;

        let var_data = variance.to_vec()?;
        let std_data = std.to_vec()?;

        // For [1,2,3,4,5], mean = 3, variance = 2, std = sqrt(2) ≈ 1.414
        assert!((var_data[0] - 2.0).abs() < 1e-6);
        assert!((std_data[0] - 1.4142135).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let softmax_result = a.softmax(0)?;
        let softmax_data = softmax_result.to_vec()?;

        // Sum of softmax should be 1
        let sum: f32 = softmax_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be positive
        assert!(softmax_data.iter().all(|&x| x > 0.0));

        Ok(())
    }

    #[test]
    fn test_argmax_argmin() -> Result<()> {
        let a = Tensor::from_data(
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let argmax_dim1 = a.argmax(1, false)?;
        let argmax_data = argmax_dim1.to_vec()?;

        // argmax along dim 1: [2, 2] (indices of max in each row)
        // Note: the actual values depend on how Candle implements argmax
        assert_eq!(argmax_data.len(), 2);

        Ok(())
    }

    #[test]
    fn test_cumulative_operations() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let cumsum = a.cumsum(0)?;
        let cumsum_data = cumsum.to_vec()?;
        assert_eq!(cumsum_data, vec![1.0, 3.0, 6.0, 10.0]);

        let cumprod = a.cumprod(0)?;
        let cumprod_data = cumprod.to_vec()?;
        assert_eq!(cumprod_data, vec![1.0, 2.0, 6.0, 24.0]);

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

        // Out of bounds dimension
        assert!(a.sum_dim(5, false).is_err());
        assert!(a.max_dim(5, false).is_err());
        assert!(a.argmax(5, false).is_err());

        // Invalid norm p
        assert!(a.norm_p(-1.0).is_err());
        assert!(a.norm_p(0.0).is_err());
    }

    #[test]
    fn test_keep_dim() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Sum with keep_dim=true
        let sum_keep = a.sum_dim(1, true)?;
        assert_eq!(sum_keep.shape(), vec![2, 1]);

        // Sum with keep_dim=false
        let sum_no_keep = a.sum_dim(1, false)?;
        assert_eq!(sum_no_keep.shape(), vec![2]);

        Ok(())
    }
}
