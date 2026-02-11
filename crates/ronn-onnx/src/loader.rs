use crate::error::{OnnxError, Result};
use crate::generated;
use crate::types::DataTypeMapper;
use prost::Message;
use ronn_core::{GraphNode, ModelGraph, NodeAttribute};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, info, warn};

/// Minimum supported ONNX IR version
const MIN_IR_VERSION: i64 = 3;

/// Recommended ONNX opset version
const RECOMMENDED_OPSET_VERSION: i64 = 13;

/// Loads ONNX models and converts them to RONN internal representation
pub struct ModelLoader;

impl ModelLoader {
    /// Load an ONNX model from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<LoadedModel> {
        info!("Loading ONNX model from: {:?}", path.as_ref());
        let bytes = fs::read(path)?;
        Self::load_from_bytes(&bytes)
    }

    /// Load an ONNX model from bytes
    pub fn load_from_bytes(bytes: &[u8]) -> Result<LoadedModel> {
        // Decode ONNX protobuf
        let model_proto = generated::ModelProto::decode(bytes)?;
        Self::convert_model(model_proto)
    }

    /// Convert ONNX ModelProto to RONN LoadedModel
    fn convert_model(model_proto: generated::ModelProto) -> Result<LoadedModel> {
        // Validate IR version
        let ir_version = model_proto.ir_version;
        if ir_version < MIN_IR_VERSION {
            return Err(OnnxError::UnsupportedIrVersion {
                version: ir_version,
                min_version: MIN_IR_VERSION,
            });
        }

        info!("ONNX model IR version: {}", ir_version);

        // Check opset version
        if let Some(opset) = model_proto.opset_import.first() {
            let opset_version = opset.version;
            if opset_version < RECOMMENDED_OPSET_VERSION {
                warn!(
                    "ONNX opset version {} is older than recommended {}",
                    opset_version, RECOMMENDED_OPSET_VERSION
                );
            } else {
                info!("ONNX opset version: {}", opset_version);
            }
        }

        let graph_proto = model_proto
            .graph
            .ok_or_else(|| OnnxError::ParseError("Model has no graph".to_string()))?;

        info!("Converting ONNX graph: {}", graph_proto.name);

        // Parse initializers (weights/constants)
        let mut initializers = HashMap::new();
        for init in &graph_proto.initializer {
            let name = init.name.clone();
            debug!("Loading initializer: {}", name);
            let tensor = Self::tensor_from_proto(init)?;
            initializers.insert(name, tensor);
        }

        // Parse inputs
        let mut inputs = Vec::new();
        for input in &graph_proto.input {
            let name = input.name.clone();
            let (shape, data_type) = Self::shape_and_type_from_value_info(input)?;
            debug!(
                "Input: {} with shape {:?}, type {:?}",
                name, shape, data_type
            );

            // Skip if it's an initializer (weights)
            if !initializers.contains_key(&name) {
                inputs.push(TensorInfo {
                    name: name.clone(),
                    shape,
                    data_type,
                });
            }
        }

        // Parse outputs
        let mut outputs = Vec::new();
        for output in &graph_proto.output {
            let name = output.name.clone();
            let (shape, data_type) = Self::shape_and_type_from_value_info(output)?;
            debug!(
                "Output: {} with shape {:?}, type {:?}",
                name, shape, data_type
            );
            outputs.push(TensorInfo {
                name: name.clone(),
                shape,
                data_type,
            });
        }

        // Parse nodes and build graph
        let mut nodes = Vec::new();
        for (idx, node_proto) in graph_proto.node.iter().enumerate() {
            let op_type = node_proto.op_type.clone();
            let name = if node_proto.name.is_empty() {
                format!("node_{}", idx)
            } else {
                node_proto.name.clone()
            };

            debug!("Processing node: {} ({})", name, op_type);

            // Parse attributes
            let mut attributes = HashMap::new();
            for attr in &node_proto.attribute {
                let attr_name = attr.name.clone();
                let attr_value = Self::convert_attribute(attr)?;
                attributes.insert(attr_name, attr_value);
            }

            let node = GraphNode {
                id: 0, // Will be assigned by graph.add_node()
                op_type,
                inputs: node_proto.input.clone(),
                outputs: node_proto.output.clone(),
                attributes,
                name: Some(name),
            };
            nodes.push(node);
        }

        // Build the model graph
        let graph = ModelGraph::from_nodes(nodes);

        Ok(LoadedModel {
            graph,
            inputs,
            outputs,
            initializers,
            producer_name: Some(model_proto.producer_name),
            ir_version,
        })
    }

    /// Convert ONNX AttributeProto to RONN NodeAttribute
    fn convert_attribute(attr: &generated::AttributeProto) -> Result<NodeAttribute> {
        use generated::attribute_proto::AttributeType;

        let attr_type = AttributeType::try_from(attr.r#type).unwrap_or(AttributeType::Undefined);

        match attr_type {
            AttributeType::Float => Ok(NodeAttribute::Float(attr.f as f64)),
            AttributeType::Int => Ok(NodeAttribute::Int(attr.i)),
            AttributeType::String => {
                let s = String::from_utf8_lossy(&attr.s).to_string();
                Ok(NodeAttribute::String(s))
            }
            AttributeType::Tensor => {
                if let Some(ref t) = attr.t {
                    // For now, store tensor as empty bytes placeholder
                    // Full implementation would serialize the tensor data
                    let _tensor = Self::tensor_from_proto(t)?;
                    Ok(NodeAttribute::Tensor(Vec::new()))
                } else {
                    Err(OnnxError::InvalidAttribute {
                        name: attr.name.clone(),
                        reason: "Tensor attribute has no value".to_string(),
                    })
                }
            }
            AttributeType::Floats => {
                let floats: Vec<f64> = attr.floats.iter().map(|&f| f as f64).collect();
                Ok(NodeAttribute::FloatArray(floats))
            }
            AttributeType::Ints => Ok(NodeAttribute::IntArray(attr.ints.clone())),
            _ => {
                warn!("Unsupported attribute type: {:?}", attr_type);
                Ok(NodeAttribute::String(format!(
                    "unsupported_type_{:?}",
                    attr_type
                )))
            }
        }
    }

    /// Extract shape and data type from ValueInfoProto
    fn shape_and_type_from_value_info(
        value_info: &generated::ValueInfoProto,
    ) -> Result<(Vec<usize>, ronn_core::types::DataType)> {
        let type_proto = value_info
            .r#type
            .as_ref()
            .ok_or_else(|| OnnxError::ParseError("ValueInfo has no type".to_string()))?;

        let tensor_type = match &type_proto.value {
            Some(generated::type_proto::Value::TensorType(t)) => t,
            _ => {
                return Err(OnnxError::ParseError(
                    "Type is not a tensor type".to_string(),
                ));
            }
        };

        // Extract data type
        let data_type = DataTypeMapper::from_onnx(tensor_type.elem_type)?;

        // Extract shape
        let shape = if let Some(ref shape_proto) = tensor_type.shape {
            shape_proto
                .dim
                .iter()
                .map(|d| {
                    if let Some(ref value) = d.value {
                        match value {
                            generated::tensor_shape_proto::dimension::Value::DimValue(v) => {
                                *v as usize
                            }
                            generated::tensor_shape_proto::dimension::Value::DimParam(_) => 0, // Dynamic
                        }
                    } else {
                        0 // Unknown dimension
                    }
                })
                .collect()
        } else {
            Vec::new() // Scalar or unknown rank
        };

        Ok((shape, data_type))
    }

    /// Convert ONNX TensorProto to RONN Tensor
    fn tensor_from_proto(
        tensor_proto: &generated::TensorProto,
    ) -> Result<ronn_core::tensor::Tensor> {
        let data_type = DataTypeMapper::from_onnx(tensor_proto.data_type)?;
        let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();

        // Extract actual tensor data based on data type
        let tensor = if !tensor_proto.raw_data.is_empty() {
            // Data is in raw_data field (packed binary)
            Self::tensor_from_raw_data(&tensor_proto.raw_data, &shape, data_type)?
        } else {
            // Data is in typed fields (float_data, int32_data, etc.)
            Self::tensor_from_typed_data(tensor_proto, &shape, data_type)?
        };

        Ok(tensor)
    }

    /// Create tensor from raw_data field
    fn tensor_from_raw_data(
        raw_data: &[u8],
        shape: &[usize],
        data_type: ronn_core::types::DataType,
    ) -> Result<ronn_core::tensor::Tensor> {
        use ronn_core::types::TensorLayout;

        let tensor = match data_type {
            ronn_core::types::DataType::F32 => {
                if raw_data.len() % 4 != 0 {
                    return Err(OnnxError::ParseError(format!(
                        "Invalid raw_data length {} for F32 tensor",
                        raw_data.len()
                    )));
                }
                let values = raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>();
                ronn_core::tensor::Tensor::from_data(
                    values,
                    shape.to_vec(),
                    data_type,
                    TensorLayout::RowMajor,
                )?
            }
            ronn_core::types::DataType::I64 => {
                if raw_data.len() % 8 != 0 {
                    return Err(OnnxError::ParseError(format!(
                        "Invalid raw_data length {} for I64 tensor",
                        raw_data.len()
                    )));
                }
                let values = raw_data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect::<Vec<i64>>();
                ronn_core::tensor::Tensor::from_i64(values, shape.to_vec(), TensorLayout::RowMajor)?
            }
            ronn_core::types::DataType::I32 => {
                if raw_data.len() % 4 != 0 {
                    return Err(OnnxError::ParseError(format!(
                        "Invalid raw_data length {} for I32 tensor",
                        raw_data.len()
                    )));
                }
                let values = raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<i32>>();
                ronn_core::tensor::Tensor::from_i32(values, shape.to_vec(), TensorLayout::RowMajor)?
            }
            _ => {
                // Keep fallback behavior for less common dtypes until each is mapped.
                ronn_core::tensor::Tensor::zeros(
                    shape.to_vec(),
                    data_type,
                    TensorLayout::RowMajor,
                )?
            }
        };

        debug!(
            "Decoded tensor from raw_data: shape={:?}, type={:?}, bytes={}",
            shape,
            data_type,
            raw_data.len()
        );
        Ok(tensor)
    }

    /// Create tensor from typed data fields
    fn tensor_from_typed_data(
        tensor_proto: &generated::TensorProto,
        shape: &[usize],
        data_type: ronn_core::types::DataType,
    ) -> Result<ronn_core::tensor::Tensor> {
        use ronn_core::types::{DataType, TensorLayout};

        let tensor = match data_type {
            DataType::F32 => {
                if !tensor_proto.float_data.is_empty() {
                    ronn_core::tensor::Tensor::from_data(
                        tensor_proto.float_data.clone(),
                        shape.to_vec(),
                        data_type,
                        TensorLayout::RowMajor,
                    )?
                } else {
                    // Empty tensor
                    ronn_core::tensor::Tensor::zeros(
                        shape.to_vec(),
                        data_type,
                        TensorLayout::RowMajor,
                    )?
                }
            }
            DataType::I32 => {
                if !tensor_proto.int32_data.is_empty() {
                    ronn_core::tensor::Tensor::from_i32(
                        tensor_proto.int32_data.clone(),
                        shape.to_vec(),
                        TensorLayout::RowMajor,
                    )?
                } else {
                    ronn_core::tensor::Tensor::zeros(
                        shape.to_vec(),
                        data_type,
                        TensorLayout::RowMajor,
                    )?
                }
            }
            DataType::I64 => {
                if !tensor_proto.int64_data.is_empty() {
                    ronn_core::tensor::Tensor::from_i64(
                        tensor_proto.int64_data.clone(),
                        shape.to_vec(),
                        TensorLayout::RowMajor,
                    )?
                } else {
                    ronn_core::tensor::Tensor::zeros(
                        shape.to_vec(),
                        data_type,
                        TensorLayout::RowMajor,
                    )?
                }
            }
            _ => {
                // For other types, create zero tensor for now
                ronn_core::tensor::Tensor::zeros(shape.to_vec(), data_type, TensorLayout::RowMajor)?
            }
        };

        Ok(tensor)
    }
}

/// Information about a tensor (input/output)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: ronn_core::types::DataType,
}

/// A loaded ONNX model converted to RONN representation
pub struct LoadedModel {
    pub graph: ModelGraph,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    pub initializers: HashMap<String, ronn_core::tensor::Tensor>,
    pub producer_name: Option<String>,
    pub ir_version: i64,
}

impl LoadedModel {
    /// Get the model graph
    pub fn graph(&self) -> &ModelGraph {
        &self.graph
    }

    /// Get input tensor information
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// Get output tensor information
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// Get initializer tensors (weights, constants)
    pub fn initializers(&self) -> &HashMap<String, ronn_core::tensor::Tensor> {
        &self.initializers
    }
}
