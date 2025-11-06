use super::{OnnxOperator, Result};
use crate::error::OnnxError;
use ronn_core::NodeAttribute;
use ronn_core::tensor::Tensor;
use std::collections::HashMap;

// ReLU: max(0, x)
pub struct ReluOp;

impl OnnxOperator for ReluOp {
    fn op_type(&self) -> &str {
        "Relu"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Relu expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].relu()?;
        Ok(vec![result])
    }
}

// Sigmoid: 1 / (1 + exp(-x))
pub struct SigmoidOp;

impl OnnxOperator for SigmoidOp {
    fn op_type(&self) -> &str {
        "Sigmoid"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Sigmoid expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].sigmoid()?;
        Ok(vec![result])
    }
}

// Tanh: hyperbolic tangent
pub struct TanhOp;

impl OnnxOperator for TanhOp {
    fn op_type(&self) -> &str {
        "Tanh"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Tanh expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].tanh()?;
        Ok(vec![result])
    }
}

// Softmax: exp(x_i) / sum(exp(x_j))
pub struct SoftmaxOp;

impl OnnxOperator for SoftmaxOp {
    fn op_type(&self) -> &str {
        "Softmax"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Softmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Get axis attribute (default: last dimension)
        let axis = if let Some(NodeAttribute::Int(a)) = attributes.get("axis") {
            (*a).max(0) as usize
        } else {
            inputs[0].shape().len() - 1
        };

        let result = inputs[0].softmax(axis)?;
        Ok(vec![result])
    }
}

// GELU: Gaussian Error Linear Unit
// GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub struct GeluOp;

impl OnnxOperator for GeluOp {
    fn op_type(&self) -> &str {
        "Gelu"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Gelu expects 1 input, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].gelu()?;
        Ok(vec![result])
    }
}
