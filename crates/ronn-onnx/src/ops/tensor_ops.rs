use super::{OnnxOperator, Result};
use crate::error::OnnxError;
use ronn_core::ops::ShapeOps;
use ronn_core::tensor::Tensor;
use ronn_core::NodeAttribute;
use std::collections::HashMap;

// Reshape: change tensor shape
pub struct ReshapeOp;

impl OnnxOperator for ReshapeOp {
    fn op_type(&self) -> &str {
        "Reshape"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Reshape expects 2 inputs (data, shape), got {}",
                inputs.len()
            )));
        }

        // Second input is the target shape tensor
        // Try i64 first (ONNX spec), fall back to f32 if needed
        let shape_usize: Vec<usize> = if let Ok(shape) = inputs[1].to_vec1::<i64>() {
            shape.iter().map(|&x| x as usize).collect()
        } else {
            // Fall back to f32 and convert
            let shape_f32 = inputs[1].to_vec1::<f32>()?;
            shape_f32.iter().map(|&x| x as usize).collect()
        };

        let result = inputs[0].reshape(&shape_usize)?;
        Ok(vec![result])
    }
}

// Transpose: permute tensor dimensions
pub struct TransposeOp;

impl OnnxOperator for TransposeOp {
    fn op_type(&self) -> &str {
        "Transpose"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Transpose expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Get permutation from attributes
        let perm: Vec<usize> = if let Some(NodeAttribute::IntArray(p)) = attributes.get("perm") {
            p.iter().map(|&x| x as usize).collect()
        } else {
            // Default: reverse all dimensions
            let rank = inputs[0].rank();
            (0..rank).rev().collect()
        };

        let result = inputs[0].transpose(&perm)?;
        Ok(vec![result])
    }
}

// Concat: concatenate tensors along an axis
pub struct ConcatOp;

impl OnnxOperator for ConcatOp {
    fn op_type(&self) -> &str {
        "Concat"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(OnnxError::InvalidGraph(
                "Concat expects at least 1 input".to_string(),
            ));
        }

        // Get axis from attributes
        let axis = if let Some(NodeAttribute::Int(a)) = attributes.get("axis") {
            *a as usize
        } else {
            return Err(OnnxError::InvalidAttribute {
                name: "axis".to_string(),
                reason: "Concat requires axis attribute".to_string(),
            });
        };

        let result = Tensor::concat(inputs, axis)?;
        Ok(vec![result])
    }
}

// Split: split tensor along an axis
pub struct SplitOp;

impl OnnxOperator for SplitOp {
    fn op_type(&self) -> &str {
        "Split"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() < 1 {
            return Err(OnnxError::InvalidGraph(
                "Split expects at least 1 input".to_string(),
            ));
        }

        // Get axis from attributes (default: 0)
        let axis = if let Some(NodeAttribute::Int(a)) = attributes.get("axis") {
            *a as usize
        } else {
            0
        };

        // Get split sizes
        let splits = if let Some(NodeAttribute::IntArray(s)) = attributes.get("split") {
            s.iter().map(|&x| x as usize).collect()
        } else if inputs.len() > 1 {
            // Split sizes provided as second input
            // Try i64 first (ONNX spec), fall back to f32 if needed
            if let Ok(split_i64) = inputs[1].to_vec1::<i64>() {
                split_i64.iter().map(|&x| x as usize).collect()
            } else {
                let split_f32 = inputs[1].to_vec1::<f32>()?;
                split_f32.iter().map(|&x| x as usize).collect()
            }
        } else {
            // Equal splits
            let num_outputs = if let Some(NodeAttribute::Int(n)) = attributes.get("num_outputs") {
                *n as usize
            } else {
                2 // Default to 2 splits
            };
            vec![inputs[0].shape()[axis] / num_outputs; num_outputs]
        };

        let results = inputs[0].split(splits[0], axis)?;
        Ok(results)
    }
}

// Gather: gather elements along an axis
pub struct GatherOp;

impl OnnxOperator for GatherOp {
    fn op_type(&self) -> &str {
        "Gather"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Gather expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        // Get axis from attributes (default: 0)
        let axis = if let Some(NodeAttribute::Int(a)) = attributes.get("axis") {
            *a as usize
        } else {
            0
        };

        let result = inputs[0].gather(inputs[1], axis)?;
        Ok(vec![result])
    }
}

// Slice: extract a slice from a tensor
pub struct SliceOp;

impl OnnxOperator for SliceOp {
    fn op_type(&self) -> &str {
        "Slice"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(OnnxError::InvalidGraph(
                "Slice expects at least 1 input".to_string(),
            ));
        }

        // ONNX opset >= 10 provides starts, ends, axes, steps as inputs
        // For simplicity, we'll support attribute-based slicing for now
        let starts = Self::get_ints_attr(attributes, "starts", vec![]);
        let ends = Self::get_ints_attr(attributes, "ends", vec![]);
        let axes = Self::get_ints_attr(attributes, "axes", vec![]);
        let steps = Self::get_ints_attr(attributes, "steps", vec![]);

        if starts.is_empty() && inputs.len() >= 3 {
            // Get from input tensors
            let starts_tensor = inputs[1].to_vec1::<i64>()?;
            let ends_tensor = inputs[2].to_vec1::<i64>()?;
            let axes_tensor = if inputs.len() > 3 {
                inputs[3].to_vec1::<i64>()?
            } else {
                (0..starts_tensor.len() as i64).collect()
            };
            let steps_tensor = if inputs.len() > 4 {
                inputs[4].to_vec1::<i64>()?
            } else {
                vec![1; starts_tensor.len()]
            };

            // Simplified implementation - full version would use multi-dimensional slicing
            let _ = (starts_tensor, ends_tensor, axes_tensor, steps_tensor);
            Err(OnnxError::UnsupportedOperator {
                op_type: "Slice with input tensors (not yet implemented)".to_string(),
            })
        } else {
            // Simplified implementation - full version would use multi-dimensional slicing
            let _ = (starts, ends, axes, steps);
            Err(OnnxError::UnsupportedOperator {
                op_type: "Slice (not yet fully implemented)".to_string(),
            })
        }
    }
}

impl SliceOp {
    fn get_ints_attr(
        attrs: &HashMap<String, NodeAttribute>,
        name: &str,
        default: Vec<i64>,
    ) -> Vec<i64> {
        if let Some(NodeAttribute::IntArray(v)) = attrs.get(name) {
            v.clone()
        } else {
            default
        }
    }
}

// Squeeze: remove dimensions of size 1
pub struct SqueezeOp;

impl OnnxOperator for SqueezeOp {
    fn op_type(&self) -> &str {
        "Squeeze"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Squeeze expects 1 input, got {}",
                inputs.len()
            )));
        }

        let axes = if let Some(NodeAttribute::IntArray(v)) = attributes.get("axes") {
            Some(v.iter().map(|&x| x as usize).collect::<Vec<_>>())
        } else {
            None
        };

        let result = inputs[0].squeeze(axes)?;
        Ok(vec![result])
    }
}

// Unsqueeze: add dimensions of size 1
pub struct UnsqueezeOp;

impl OnnxOperator for UnsqueezeOp {
    fn op_type(&self) -> &str {
        "Unsqueeze"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Unsqueeze expects 1 input, got {}",
                inputs.len()
            )));
        }

        let axes = if let Some(NodeAttribute::IntArray(v)) = attributes.get("axes") {
            v.iter().map(|&x| x as usize).collect()
        } else {
            return Err(OnnxError::InvalidGraph(
                "Unsqueeze requires 'axes' attribute".to_string(),
            ));
        };

        let result = inputs[0].unsqueeze(&axes)?;
        Ok(vec![result])
    }
}

// ReduceMean: compute mean along axes
pub struct ReduceMeanOp;

impl OnnxOperator for ReduceMeanOp {
    fn op_type(&self) -> &str {
        "ReduceMean"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "ReduceMean expects 1 input, got {}",
                inputs.len()
            )));
        }

        let axes = if let Some(NodeAttribute::IntArray(v)) = attributes.get("axes") {
            v.iter().map(|&x| x as usize).collect()
        } else {
            // Default: reduce all dimensions
            (0..inputs[0].shape().len()).collect()
        };

        let keepdims = if let Some(NodeAttribute::Int(v)) = attributes.get("keepdims") {
            *v != 0
        } else {
            true
        };

        let result = inputs[0].reduce_mean(&axes, keepdims)?;
        Ok(vec![result])
    }
}

// ReduceSum: compute sum along axes
pub struct ReduceSumOp;

impl OnnxOperator for ReduceSumOp {
    fn op_type(&self) -> &str {
        "ReduceSum"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "ReduceSum expects 1 input, got {}",
                inputs.len()
            )));
        }

        let axes = if let Some(NodeAttribute::IntArray(v)) = attributes.get("axes") {
            v.iter().map(|&x| x as usize).collect()
        } else {
            // Default: reduce all dimensions
            (0..inputs[0].shape().len()).collect()
        };

        let keepdims = if let Some(NodeAttribute::Int(v)) = attributes.get("keepdims") {
            *v != 0
        } else {
            true
        };

        let result = inputs[0].reduce_sum(&axes, keepdims)?;
        Ok(vec![result])
    }
}

// Cast: convert tensor data type
pub struct CastOp;

impl OnnxOperator for CastOp {
    fn op_type(&self) -> &str {
        "Cast"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "Cast expects 1 input, got {}",
                inputs.len()
            )));
        }

        let to = if let Some(NodeAttribute::Int(v)) = attributes.get("to") {
            // ONNX TensorProto data types
            match v {
                1 => ronn_core::DataType::F32,  // FLOAT
                10 => ronn_core::DataType::F16, // FLOAT16
                11 => ronn_core::DataType::F64, // DOUBLE
                6 => ronn_core::DataType::I32,  // INT32
                7 => ronn_core::DataType::I64,  // INT64
                2 => ronn_core::DataType::U8,   // UINT8
                _ => return Err(OnnxError::InvalidGraph(format!(
                    "Unsupported Cast target type: {}",
                    v
                ))),
            }
        } else {
            return Err(OnnxError::InvalidGraph(
                "Cast requires 'to' attribute".to_string(),
            ));
        };

        let result = inputs[0].cast(to)?;
        Ok(vec![result])
    }
}

// Embedding: gather operation for embedding lookup
pub struct EmbeddingOp;

impl OnnxOperator for EmbeddingOp {
    fn op_type(&self) -> &str {
        "Embedding"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Embedding expects 2 inputs (weights, indices), got {}",
                inputs.len()
            )));
        }

        let weights = inputs[0];  // Shape: [vocab_size, embedding_dim]
        let indices = inputs[1];  // Shape: [batch_size, seq_len]

        // Embedding is essentially a gather operation on axis 0
        let result = weights.gather(indices, 0)?;
        Ok(vec![result])
    }
}
