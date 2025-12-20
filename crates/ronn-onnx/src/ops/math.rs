use super::{OnnxOperator, Result};
use crate::error::OnnxError;
use ronn_core::NodeAttribute;
use ronn_core::ops::{ArithmeticOps, MatrixOps};
use ronn_core::tensor::Tensor;
use std::collections::HashMap;

// Add: element-wise addition with broadcasting
pub struct AddOp;

impl OnnxOperator for AddOp {
    fn op_type(&self) -> &str {
        "Add"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Add expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].add(inputs[1])?;
        Ok(vec![result])
    }
}

// Sub: element-wise subtraction with broadcasting
pub struct SubOp;

impl OnnxOperator for SubOp {
    fn op_type(&self) -> &str {
        "Sub"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Sub expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].sub(inputs[1])?;
        Ok(vec![result])
    }
}

// Mul: element-wise multiplication with broadcasting
pub struct MulOp;

impl OnnxOperator for MulOp {
    fn op_type(&self) -> &str {
        "Mul"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Mul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].mul(inputs[1])?;
        Ok(vec![result])
    }
}

// Div: element-wise division with broadcasting
pub struct DivOp;

impl OnnxOperator for DivOp {
    fn op_type(&self) -> &str {
        "Div"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Div expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].div(inputs[1])?;
        Ok(vec![result])
    }
}

// MatMul: matrix multiplication
pub struct MatMulOp;

impl OnnxOperator for MatMulOp {
    fn op_type(&self) -> &str {
        "MatMul"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].matmul(inputs[1])?;
        Ok(vec![result])
    }
}

// Sqrt: element-wise square root
pub struct SqrtOp;

impl OnnxOperator for SqrtOp {
    fn op_type(&self) -> &str {
        "Sqrt"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Sqrt expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].sqrt()?;
        Ok(vec![result])
    }
}

// Pow: element-wise exponentiation
pub struct PowOp;

impl OnnxOperator for PowOp {
    fn op_type(&self) -> &str {
        "Pow"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Pow expects 2 inputs (base, exponent), got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].pow_tensor(inputs[1])?;
        Ok(vec![result])
    }
}

// Exp: element-wise exponential (e^x)
pub struct ExpOp;

impl OnnxOperator for ExpOp {
    fn op_type(&self) -> &str {
        "Exp"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Exp expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].exp()?;
        Ok(vec![result])
    }
}

// Log: element-wise natural logarithm
pub struct LogOp;

impl OnnxOperator for LogOp {
    fn op_type(&self) -> &str {
        "Log"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Log expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].log()?;
        Ok(vec![result])
    }
}

// Neg: element-wise negation
pub struct NegOp;

impl OnnxOperator for NegOp {
    fn op_type(&self) -> &str {
        "Neg"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Neg expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].neg()?;
        Ok(vec![result])
    }
}

// Abs: element-wise absolute value
pub struct AbsOp;

impl OnnxOperator for AbsOp {
    fn op_type(&self) -> &str {
        "Abs"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Abs expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].abs()?;
        Ok(vec![result])
    }
}

// Clip: clamp values to [min, max]
pub struct ClipOp;

impl OnnxOperator for ClipOp {
    fn op_type(&self) -> &str {
        "Clip"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(OnnxError::InvalidGraph(
                "Clip expects at least 1 input".to_string(),
            ));
        }

        let input = inputs[0];

        // Get min/max from attributes or additional inputs
        let min = if inputs.len() > 1 {
            // Min as tensor input
            inputs[1].to_scalar_f32()?
        } else if let Some(NodeAttribute::Float(v)) = attributes.get("min") {
            *v as f32
        } else {
            f32::NEG_INFINITY
        };

        let max = if inputs.len() > 2 {
            // Max as tensor input
            inputs[2].to_scalar_f32()?
        } else if let Some(NodeAttribute::Float(v)) = attributes.get("max") {
            *v as f32
        } else {
            f32::INFINITY
        };

        let result = input.clip(min, max)?;
        Ok(vec![result])
    }
}
