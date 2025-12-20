use crate::error::{OnnxError, Result};
use crate::proto;
use ronn_core::types::DataType;

/// Maps ONNX data types to RONN internal data types
pub struct DataTypeMapper;

impl DataTypeMapper {
    pub fn from_onnx(onnx_type: i32) -> Result<DataType> {
        match onnx_type {
            1 => Ok(DataType::F32),   // FLOAT
            2 => Ok(DataType::U8),    // UINT8
            3 => Ok(DataType::I8),    // INT8
            6 => Ok(DataType::I32),   // INT32
            7 => Ok(DataType::I64),   // INT64
            9 => Ok(DataType::Bool),  // BOOL
            10 => Ok(DataType::F16),  // FLOAT16
            11 => Ok(DataType::F64),  // DOUBLE
            12 => Ok(DataType::U32),  // UINT32
            16 => Ok(DataType::BF16), // BFLOAT16
            _ => Err(OnnxError::TypeConversionError(format!(
                "Unsupported ONNX data type: {}",
                onnx_type
            ))),
        }
    }

    pub fn to_onnx(ronn_type: DataType) -> i32 {
        match ronn_type {
            DataType::F32 => 1,
            DataType::U8 => 2,
            DataType::I8 => 3,
            DataType::I32 => 6,
            DataType::I64 => 7,
            DataType::Bool => 9,
            DataType::F16 => 10,
            DataType::F64 => 11,
            DataType::U32 => 12,
            DataType::BF16 => 16,
        }
    }
}

/// Converts ONNX TensorProto to RONN Tensor
pub fn tensor_from_proto(tensor_proto: &proto::TensorProto) -> Result<ronn_core::tensor::Tensor> {
    let data_type = DataTypeMapper::from_onnx(tensor_proto.data_type.unwrap_or(0))?;
    let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();

    // For MVP: Create a placeholder tensor
    // TODO: Properly parse tensor data from proto
    use ronn_core::types::TensorLayout;
    let tensor = ronn_core::tensor::Tensor::zeros(shape, data_type, TensorLayout::RowMajor)?;

    Ok(tensor)
}

/// Extract shape from ValueInfoProto
pub fn shape_from_value_info(value_info: &proto::ValueInfoProto) -> Result<Vec<usize>> {
    let type_proto = value_info
        .r#type
        .as_ref()
        .ok_or_else(|| OnnxError::ParseError("ValueInfo has no type".to_string()))?;

    let tensor_type = match &type_proto.value {
        Some(proto::TypeProtoValue::TensorType(t)) => t,
        _ => {
            return Err(OnnxError::ParseError(
                "ValueInfo type is not tensor".to_string(),
            ));
        }
    };

    let shape_proto = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| OnnxError::ParseError("TensorType has no shape".to_string()))?;

    let shape: Vec<usize> = shape_proto
        .dim
        .iter()
        .map(|d| match &d.value {
            Some(proto::DimensionValue::DimValue(v)) => *v as usize,
            Some(proto::DimensionValue::DimParam(_)) => 0, // Dynamic dimension
            None => 0,
        })
        .collect();

    Ok(shape)
}
