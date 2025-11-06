// ONNX operator implementations using Candle backend
//
// This module provides implementations for the 20 most common ONNX operators
// needed for running models like ResNet-18 and MobileNetV2.

mod activations;
mod math;
mod neural_network;
mod tensor_ops;

pub use activations::*;
pub use math::*;
pub use neural_network::*;
pub use tensor_ops::*;

use crate::error::{OnnxError, Result};
use ronn_core::tensor::Tensor;
use ronn_core::NodeAttribute;
use std::collections::HashMap;

/// Trait for ONNX operator execution
pub trait OnnxOperator: Send + Sync {
    /// Get the operator type name
    fn op_type(&self) -> &str;

    /// Execute the operator
    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>>;

    /// Validate inputs and attributes
    fn validate(
        &self,
        _inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<()> {
        // Default validation - can be overridden
        Ok(())
    }
}

/// Registry of all supported ONNX operators
pub struct OperatorRegistry {
    operators: HashMap<String, Box<dyn OnnxOperator>>,
}

impl OperatorRegistry {
    /// Create a new operator registry with all built-in operators
    pub fn new() -> Self {
        let mut registry = Self {
            operators: HashMap::new(),
        };

        // Register all operators
        registry.register_activations();
        registry.register_math();
        registry.register_neural_network();
        registry.register_tensor_ops();

        registry
    }

    /// Register an operator
    pub fn register(&mut self, op: Box<dyn OnnxOperator>) {
        self.operators.insert(op.op_type().to_string(), op);
    }

    /// Get an operator by name
    pub fn get(&self, op_type: &str) -> Result<&dyn OnnxOperator> {
        self.operators
            .get(op_type)
            .map(|op| op.as_ref())
            .ok_or_else(|| OnnxError::UnsupportedOperator {
                op_type: op_type.to_string(),
            })
    }

    /// Check if an operator is supported
    pub fn is_supported(&self, op_type: &str) -> bool {
        self.operators.contains_key(op_type)
    }

    /// Get list of all supported operators
    pub fn supported_operators(&self) -> Vec<String> {
        self.operators.keys().cloned().collect()
    }

    fn register_activations(&mut self) {
        self.register(Box::new(ReluOp));
        self.register(Box::new(SigmoidOp));
        self.register(Box::new(TanhOp));
        self.register(Box::new(SoftmaxOp));
        self.register(Box::new(GeluOp));
    }

    fn register_math(&mut self) {
        self.register(Box::new(AddOp));
        self.register(Box::new(SubOp));
        self.register(Box::new(MulOp));
        self.register(Box::new(DivOp));
        self.register(Box::new(MatMulOp));
    }

    fn register_neural_network(&mut self) {
        self.register(Box::new(Conv2dOp));
        self.register(Box::new(MaxPoolOp));
        self.register(Box::new(AvgPoolOp));
        self.register(Box::new(BatchNormOp));
        self.register(Box::new(LayerNormOp));
        self.register(Box::new(AttentionOp));
        self.register(Box::new(MultiHeadAttentionOp));
    }

    fn register_tensor_ops(&mut self) {
        self.register(Box::new(ReshapeOp));
        self.register(Box::new(TransposeOp));
        self.register(Box::new(ConcatOp));
        self.register(Box::new(SplitOp));
        self.register(Box::new(GatherOp));
        self.register(Box::new(SliceOp));
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
