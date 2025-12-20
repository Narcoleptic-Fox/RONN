//! Arithmetic tensor operations with broadcasting support.
//!
//! This module provides element-wise arithmetic operations (Add, Sub, Mul, Div)
//! with full broadcasting support using the Candle backend.

use crate::tensor::Tensor;
use anyhow::{Result, anyhow};

/// Trait for arithmetic operations on tensors.
pub trait ArithmeticOps {
    /// Element-wise addition with broadcasting.
    fn add(&self, other: &Tensor) -> Result<Tensor>;

    /// Element-wise subtraction with broadcasting.
    fn sub(&self, other: &Tensor) -> Result<Tensor>;

    /// Element-wise multiplication with broadcasting.
    fn mul(&self, other: &Tensor) -> Result<Tensor>;

    /// Element-wise division with broadcasting.
    fn div(&self, other: &Tensor) -> Result<Tensor>;

    /// Add a scalar to all elements.
    fn add_scalar(&self, scalar: f32) -> Result<Tensor>;

    /// Subtract a scalar from all elements.
    fn sub_scalar(&self, scalar: f32) -> Result<Tensor>;

    /// Multiply all elements by a scalar.
    fn mul_scalar(&self, scalar: f32) -> Result<Tensor>;

    /// Divide all elements by a scalar.
    fn div_scalar(&self, scalar: f32) -> Result<Tensor>;

    /// Element-wise negation.
    fn neg(&self) -> Result<Tensor>;

    /// Element-wise absolute value.
    fn abs(&self) -> Result<Tensor>;

    /// Element-wise power operation.
    fn pow(&self, exponent: f32) -> Result<Tensor>;

    /// Element-wise square root.
    fn sqrt(&self) -> Result<Tensor>;

    /// Element-wise exponential.
    fn exp(&self) -> Result<Tensor>;

    /// Element-wise natural logarithm.
    fn log(&self) -> Result<Tensor>;
}

impl ArithmeticOps for Tensor {
    fn add(&self, other: &Tensor) -> Result<Tensor> {
        // Check if tensors can be broadcast together
        if !self.is_broadcastable_with(other) {
            return Err(anyhow!(
                "Cannot broadcast tensors with shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_candle = self.candle_tensor().broadcast_add(other.candle_tensor())?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn sub(&self, other: &Tensor) -> Result<Tensor> {
        if !self.is_broadcastable_with(other) {
            return Err(anyhow!(
                "Cannot broadcast tensors with shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_candle = self.candle_tensor().broadcast_sub(other.candle_tensor())?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if !self.is_broadcastable_with(other) {
            return Err(anyhow!(
                "Cannot broadcast tensors with shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_candle = self.candle_tensor().broadcast_mul(other.candle_tensor())?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn div(&self, other: &Tensor) -> Result<Tensor> {
        if !self.is_broadcastable_with(other) {
            return Err(anyhow!(
                "Cannot broadcast tensors with shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_candle = self.candle_tensor().broadcast_div(other.candle_tensor())?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        let result_candle = (self.candle_tensor() + scalar as f64)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn sub_scalar(&self, scalar: f32) -> Result<Tensor> {
        let result_candle = (self.candle_tensor() - scalar as f64)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        let result_candle = (self.candle_tensor() * scalar as f64)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        if scalar == 0.0 {
            return Err(anyhow!("Division by zero"));
        }

        let result_candle = (self.candle_tensor() / scalar as f64)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn neg(&self) -> Result<Tensor> {
        let result_candle = self.candle_tensor().neg()?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn abs(&self) -> Result<Tensor> {
        let result_candle = self.candle_tensor().abs()?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn pow(&self, exponent: f32) -> Result<Tensor> {
        let result_candle = self.candle_tensor().powf(exponent as f64)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn sqrt(&self) -> Result<Tensor> {
        let result_candle = self.candle_tensor().sqrt()?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn exp(&self) -> Result<Tensor> {
        let result_candle = self.candle_tensor().exp()?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    fn log(&self) -> Result<Tensor> {
        let result_candle = self.candle_tensor().log()?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }
}

/// Convenience functions for arithmetic operations.
impl Tensor {
    /// Clamp tensor values between min and max.
    pub fn clamp(&self, min: f32, max: f32) -> Result<Tensor> {
        if min > max {
            return Err(anyhow!(
                "Min value {} is greater than max value {}",
                min,
                max
            ));
        }

        let result_candle = self.candle_tensor().clamp(min as f64, max as f64)?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    /// Apply ReLU activation function.
    pub fn relu(&self) -> Result<Tensor> {
        self.clamp(0.0, f32::INFINITY)
    }

    /// Apply Sigmoid activation function.
    pub fn sigmoid(&self) -> Result<Tensor> {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = self.neg()?;
        let exp_neg_x = neg_x.exp()?;
        let one = Tensor::ones(vec![1], self.dtype(), self.layout())?;
        let one_plus_exp = one.add(&exp_neg_x)?;
        one.div(&one_plus_exp)
    }

    /// Apply Tanh activation function.
    pub fn tanh(&self) -> Result<Tensor> {
        let result_candle = self.candle_tensor().tanh()?;

        Ok(Tensor::from_candle(
            result_candle,
            self.dtype(),
            self.layout(),
        ))
    }

    /// Apply GELU activation function.
    pub fn gelu(&self) -> Result<Tensor> {
        // GELU(x) = x * Φ(x) where Φ is the standard Gaussian CDF
        // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let x = self;
        let x_cubed = x.pow(3.0)?;
        let term1 = x_cubed.mul_scalar(0.044715)?;
        let term2 = x.add(&term1)?;
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        let term3 = term2.mul_scalar(sqrt_2_over_pi)?;
        let tanh_term = term3.tanh()?;
        let one = Tensor::ones(vec![1], self.dtype(), self.layout())?;
        let one_plus_tanh = one.add(&tanh_term)?;
        let half = Tensor::from_data(vec![0.5], vec![1], self.dtype(), self.layout())?;
        let result = x.mul(&half)?.mul(&one_plus_tanh)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, TensorLayout};

    #[test]
    fn test_arithmetic_operations() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let b = Tensor::from_data(
            vec![2.0, 1.0, 1.0, 2.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Test addition
        let sum = a.add(&b)?;
        let sum_data = sum.to_vec()?;
        assert_eq!(sum_data, vec![3.0, 3.0, 4.0, 6.0]);

        // Test subtraction
        let diff = a.sub(&b)?;
        let diff_data = diff.to_vec()?;
        assert_eq!(diff_data, vec![-1.0, 1.0, 2.0, 2.0]);

        // Test multiplication
        let product = a.mul(&b)?;
        let product_data = product.to_vec()?;
        assert_eq!(product_data, vec![2.0, 2.0, 3.0, 8.0]);

        // Test division
        let quotient = a.div(&b)?;
        let quotient_data = quotient.to_vec()?;
        assert_eq!(quotient_data, vec![0.5, 2.0, 3.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_scalar_operations() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Test scalar addition
        let sum = a.add_scalar(5.0)?;
        let sum_data = sum.to_vec()?;
        assert_eq!(sum_data, vec![6.0, 7.0, 8.0, 9.0]);

        // Test scalar multiplication
        let product = a.mul_scalar(2.0)?;
        let product_data = product.to_vec()?;
        assert_eq!(product_data, vec![2.0, 4.0, 6.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_broadcasting() -> Result<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let b = Tensor::from_data(
            vec![10.0, 20.0, 30.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let sum = a.add(&b)?;
        let sum_data = sum.to_vec()?;
        assert_eq!(sum_data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

        Ok(())
    }

    #[test]
    fn test_activation_functions() -> Result<()> {
        let a = Tensor::from_data(
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![4],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Test ReLU
        let relu_result = a.relu()?;
        let relu_data = relu_result.to_vec()?;
        assert_eq!(relu_data, vec![0.0, 0.0, 1.0, 2.0]);

        // Test absolute value
        let abs_result = a.abs()?;
        let abs_data = abs_result.to_vec()?;
        assert_eq!(abs_data, vec![1.0, 0.0, 1.0, 2.0]);

        // Test negation
        let neg_result = a.neg()?;
        let neg_data = neg_result.to_vec()?;
        assert_eq!(neg_data, vec![1.0, 0.0, -1.0, -2.0]);

        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<()> {
        let x = Tensor::from_data(vec![0.0], vec![1], DataType::F32, TensorLayout::RowMajor)?;
        let sigmoid_result = x.sigmoid()?;
        let sigmoid_data = sigmoid_result.to_vec()?;

        // sigmoid(0) should be 0.5
        assert!((sigmoid_data[0] - 0.5).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
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

        // Should fail because shapes are not broadcastable
        assert!(a.add(&b).is_err());

        // Division by zero should fail
        assert!(a.div_scalar(0.0).is_err());

        // Invalid clamp should fail
        assert!(a.clamp(5.0, 1.0).is_err());
    }
}
