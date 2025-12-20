//! Matrix operations for tensors.
//!
//! This module provides matrix operations including matrix multiplication,
//! transpose, and other linear algebra operations using the Candle backend.

use crate::ops::arithmetic::ArithmeticOps;
use crate::ops::reduction::ReductionOps;
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};

/// Trait for matrix operations on tensors.
pub trait MatrixOps {
    /// Matrix multiplication.
    fn matmul(&self, other: &Tensor) -> Result<Tensor>;

    /// Transpose the tensor (swap last two dimensions).
    fn transpose(&self) -> Result<Tensor>;

    /// Transpose with specific dimension indices.
    fn transpose_dims(&self, dim1: usize, dim2: usize) -> Result<Tensor>;

    /// Batch matrix multiplication.
    fn batch_matmul(&self, other: &Tensor) -> Result<Tensor>;
}

impl MatrixOps for Tensor {
    fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // Check dimension compatibility for matrix multiplication
        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(anyhow!(
                "Matrix multiplication requires at least 2D tensors, got shapes {:?} and {:?}",
                self_shape,
                other_shape
            ));
        }

        let _self_rows = self_shape[self_shape.len() - 2];
        let self_cols = self_shape[self_shape.len() - 1];
        let other_rows = other_shape[other_shape.len() - 2];
        let _other_cols = other_shape[other_shape.len() - 1];

        if self_cols != other_rows {
            return Err(anyhow!(
                "Incompatible dimensions for matrix multiplication: {} vs {}",
                self_cols,
                other_rows
            ));
        }

        let result_candle = self.candle_tensor().matmul(other.candle_tensor())?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn transpose(&self) -> Result<Tensor> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(anyhow!(
                "Transpose requires at least 2D tensor, got shape {:?}",
                shape
            ));
        }

        let dim1 = shape.len() - 2;
        let dim2 = shape.len() - 1;
        self.transpose_dims(dim1, dim2)
    }

    fn transpose_dims(&self, dim1: usize, dim2: usize) -> Result<Tensor> {
        let shape = self.shape();

        if dim1 >= shape.len() || dim2 >= shape.len() {
            return Err(anyhow!(
                "Transpose dimensions {} and {} out of bounds for tensor with {} dimensions",
                dim1,
                dim2,
                shape.len()
            ));
        }

        let result_candle = self.candle_tensor().transpose(dim1, dim2)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn batch_matmul(&self, other: &Tensor) -> Result<Tensor> {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // For batch matrix multiplication, we need at least 3D tensors
        if self_shape.len() < 3 || other_shape.len() < 3 {
            // If one tensor is 2D, we can broadcast it
            return self.matmul(other);
        }

        // Check batch dimensions are compatible
        let self_batch = &self_shape[..self_shape.len() - 2];
        let other_batch = &other_shape[..other_shape.len() - 2];

        if self_batch != other_batch {
            return Err(anyhow!(
                "Incompatible batch dimensions for batch matrix multiplication: {:?} vs {:?}",
                self_batch,
                other_batch
            ));
        }

        let result_candle = self.candle_tensor().matmul(other.candle_tensor())?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }
}

/// Additional matrix operations as methods on Tensor.
impl Tensor {
    /// Compute the trace (sum of diagonal elements) of a 2D tensor.
    pub fn trace(&self) -> Result<Tensor> {
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(anyhow!(
                "Trace requires a square 2D tensor, got shape {:?}",
                shape
            ));
        }

        let diag = self.diagonal()?;
        diag.sum_all()
    }

    /// Get the diagonal elements of a 2D tensor.
    pub fn diagonal(&self) -> Result<Tensor> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(anyhow!(
                "Diagonal requires at least 2D tensor, got shape {:?}",
                shape
            ));
        }

        // For 2D matrices, manually extract diagonal elements
        if shape.len() == 2 {
            let data = self.to_vec()?;
            let rows = shape[0];
            let cols = shape[1];
            let min_dim = rows.min(cols);

            let mut diag_data = Vec::with_capacity(min_dim);
            for i in 0..min_dim {
                diag_data.push(data[i * cols + i]);
            }

            return Ok(Tensor::from_data(
                diag_data,
                vec![min_dim],
                self.dtype(),
                self.layout(),
            )?);
        }

        // For higher dimensions, this is more complex - placeholder implementation
        Err(anyhow!(
            "Diagonal extraction for >2D tensors not yet implemented"
        ))
    }

    /// Create an identity matrix of given size.
    pub fn eye(
        size: usize,
        dtype: crate::types::DataType,
        layout: crate::types::TensorLayout,
    ) -> Result<Tensor> {
        use candle_core::Device;

        let device = Device::Cpu;
        let candle_tensor = candle_core::Tensor::eye(size, dtype_to_candle(&dtype)?, &device)?;

        Ok(Tensor::from_candle(candle_tensor, dtype, layout))
    }

    /// Compute the determinant of a 2D square tensor.
    pub fn det(&self) -> Result<Tensor> {
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(anyhow!(
                "Determinant requires a square 2D tensor, got shape {:?}",
                shape
            ));
        }

        // For small matrices, we can compute determinant directly
        match shape[0] {
            1 => {
                let data = self.to_vec()?;
                Ok(Tensor::from_data(
                    vec![data[0]],
                    vec![1],
                    self.dtype(),
                    self.layout(),
                )?)
            }
            2 => {
                let data = self.to_vec()?;
                let det = data[0] * data[3] - data[1] * data[2];
                Ok(Tensor::from_data(
                    vec![det],
                    vec![1],
                    self.dtype(),
                    self.layout(),
                )?)
            }
            _ => {
                // For larger matrices, we'd need more complex algorithms like LU decomposition
                // This is a placeholder implementation
                Err(anyhow!(
                    "Determinant calculation for {}x{} matrices not yet implemented",
                    shape[0],
                    shape[1]
                ))
            }
        }
    }

    /// Compute matrix inverse (for 2x2 matrices only in this implementation).
    pub fn inverse(&self) -> Result<Tensor> {
        let shape = self.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(anyhow!(
                "Inverse requires a square 2D tensor, got shape {:?}",
                shape
            ));
        }

        if shape[0] != 2 {
            return Err(anyhow!(
                "Matrix inverse only implemented for 2x2 matrices, got {}x{}",
                shape[0],
                shape[1]
            ));
        }

        let data = self.to_vec()?;
        let a = data[0];
        let b = data[1];
        let c = data[2];
        let d = data[3];

        let det = a * d - b * c;
        if det.abs() < 1e-10 {
            return Err(anyhow!("Matrix is singular (determinant ≈ 0)"));
        }

        let inv_det = 1.0 / det;
        let inv_data = vec![d * inv_det, -b * inv_det, -c * inv_det, a * inv_det];

        Ok(Tensor::from_data(
            inv_data,
            vec![2, 2],
            self.dtype(),
            self.layout(),
        )?)
    }

    /// Compute the Frobenius norm of the tensor.
    pub fn frobenius_norm(&self) -> Result<Tensor> {
        let squared = self.mul(self)?;
        let sum = squared.sum_all()?;
        let sqrt_result = sum.sqrt()?;

        // Ensure result is at least 1D
        let sqrt_candle = sqrt_result.candle_tensor();
        let reshaped = if sqrt_candle.dims().is_empty() {
            sqrt_candle.reshape(&[1])?
        } else {
            sqrt_candle.clone()
        };

        Ok(Tensor::from_candle(
            reshaped,
            sqrt_result.dtype(),
            sqrt_result.layout(),
        ))
    }

    /// Create a tensor with ones on the diagonal and zeros elsewhere.
    pub fn diag_embed(&self) -> Result<Tensor> {
        let shape = self.shape();
        if shape.len() != 1 {
            return Err(anyhow!(
                "diag_embed requires a 1D tensor, got shape {:?}",
                shape
            ));
        }

        let n = shape[0];
        let mut diag_data = vec![0.0; n * n];
        let data = self.to_vec()?;

        for i in 0..n {
            diag_data[i * n + i] = data[i];
        }

        Ok(Tensor::from_data(
            diag_data,
            vec![n, n],
            self.dtype(),
            self.layout(),
        )?)
    }
}

/// Convert RONN DataType to Candle DType (helper function).
fn dtype_to_candle(dtype: &crate::types::DataType) -> Result<candle_core::DType> {
    use crate::types::DataType;
    use candle_core::DType;

    match dtype {
        DataType::F32 => Ok(DType::F32),
        DataType::F16 => Ok(DType::F16),
        DataType::BF16 => Ok(DType::BF16),
        DataType::F64 => Ok(DType::F64),
        DataType::U8 => Ok(DType::U8),
        DataType::U32 => Ok(DType::U32),
        // For unsupported types, use F32
        DataType::I8 | DataType::I32 | DataType::I64 | DataType::Bool => Ok(DType::F32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, TensorLayout};

    #[test]
    fn test_matrix_multiplication() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let b = Tensor::from_data(
            vec![2.0, 0.0, 1.0, 1.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let result = a.matmul(&b)?;
        let result_data = result.to_vec()?;

        // Expected: [[1*2+2*1, 1*0+2*1], [3*2+4*1, 3*0+4*1]] = [[4, 2], [10, 4]]
        assert_eq!(result_data, vec![4.0, 2.0, 10.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let transposed = MatrixOps::transpose(&a)?;
        let transposed_data = transposed.to_vec()?;
        assert_eq!(transposed.shape(), vec![3, 2]);

        // Expected: [[1, 4], [2, 5], [3, 6]] (flattened: [1, 4, 2, 5, 3, 6])
        assert_eq!(transposed_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        Ok(())
    }

    #[test]
    fn test_identity_matrix() -> Result<()> {
        let identity = Tensor::eye(3, DataType::F32, TensorLayout::RowMajor)?;
        let identity_data = identity.to_vec()?;

        assert_eq!(identity.shape(), vec![3, 3]);
        assert_eq!(
            identity_data,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );

        Ok(())
    }

    #[test]
    fn test_diagonal() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let diag = a.diagonal()?;
        let diag_data = diag.to_vec()?;

        assert_eq!(diag_data, vec![1.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_determinant_2x2() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let det = a.det()?;
        let det_data = det.to_vec()?;

        // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
        assert!((det_data[0] + 2.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_matrix_inverse_2x2() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let inv = a.inverse()?;
        let inv_data = inv.to_vec()?;

        // inv([[1, 2], [3, 4]]) = (1/(-2)) * [[4, -2], [-3, 1]] = [[-2, 1], [1.5, -0.5]]
        assert!((inv_data[0] + 2.0).abs() < 1e-6);
        assert!((inv_data[1] - 1.0).abs() < 1e-6);
        assert!((inv_data[2] - 1.5).abs() < 1e-6);
        assert!((inv_data[3] + 0.5).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_trace() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let trace = a.trace()?;
        let trace_data = trace.to_vec()?;

        // trace([[1, 2], [3, 4]]) = 1 + 4 = 5
        assert_eq!(trace_data[0], 5.0);

        Ok(())
    }

    #[test]
    fn test_diag_embed() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let diag_matrix = a.diag_embed()?;
        let diag_data = diag_matrix.to_vec()?;

        assert_eq!(diag_matrix.shape(), vec![3, 3]);
        assert_eq!(diag_data, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_frobenius_norm() -> Result<()> {
        let a = Tensor::from_data(
            vec![3.0, 4.0],
            vec![2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let norm = a.frobenius_norm()?;
        let norm_data = norm.to_vec()?;

        // ||[3, 4]||_F = sqrt(3^2 + 4^2) = sqrt(25) = 5
        assert_eq!(norm_data[0], 5.0);

        Ok(())
    }

    #[test]
    fn test_batch_matmul() -> Result<()> {
        // Create 2 batch matrices of size 2x3x2
        let a = Tensor::from_data(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let b = Tensor::from_data(
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            vec![2, 2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let result = a.batch_matmul(&b)?;
        assert_eq!(result.shape(), vec![2, 3, 2]);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        // Test incompatible dimensions for matmul
        let a = Tensor::from_data(
            vec![1.0, 2.0],
            vec![2],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();
        let b = Tensor::from_data(
            vec![1.0, 2.0, 3.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();
        assert!(a.matmul(&b).is_err());

        // Test transpose on 1D tensor
        assert!(MatrixOps::transpose(&a).is_err());

        // Test invalid transpose dimensions
        let c = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();
        assert!(c.transpose_dims(5, 6).is_err());

        // Test inverse on non-square matrix
        let d = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();
        assert!(d.inverse().is_err());

        // Test singular matrix inverse
        let singular = Tensor::from_data(
            vec![1.0, 2.0, 2.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();
        assert!(singular.inverse().is_err());
    }
}
