//! GGUF file writing helpers.

use crate::metadata::GGUFMetadata;
use crate::types::{GGUFValue, GGUFValueType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION_3};
use nnx_quant::GGMLType;
use std::path::Path;

/// Tensor payload to serialize into a GGUF file.
#[derive(Debug, Clone)]
pub struct GGUFWriterTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GGMLType,
    pub data: Vec<u8>,
}

/// Serialize GGUF bytes for the provided metadata and tensors.
pub fn write_gguf_bytes(
    version: u32,
    metadata: &GGUFMetadata,
    tensors: &[GGUFWriterTensor],
) -> Result<Vec<u8>, String> {
    let alignment = metadata
        .get("general.alignment")
        .and_then(|value| value.as_u32())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u32) as usize;

    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

    let mut metadata_entries: Vec<_> = metadata.values.iter().collect();
    metadata_entries.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    buf.extend_from_slice(&(metadata_entries.len() as u64).to_le_bytes());

    for (key, value) in metadata_entries {
        write_string(&mut buf, key);
        write_value(&mut buf, value)?;
    }

    let mut data_offset = 0u64;
    for tensor in tensors {
        write_string(&mut buf, &tensor.name);
        buf.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&(tensor.dtype as u32).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        data_offset += tensor.data.len() as u64;
    }

    let aligned = buf.len().div_ceil(alignment) * alignment;
    buf.resize(aligned, 0u8);

    for tensor in tensors {
        buf.extend_from_slice(&tensor.data);
    }

    Ok(buf)
}

/// Write a GGUF file to disk using the default v3 format.
pub fn write_gguf_file(
    path: &Path,
    metadata: &GGUFMetadata,
    tensors: &[GGUFWriterTensor],
) -> Result<(), String> {
    let bytes = write_gguf_bytes(GGUF_VERSION_3, metadata, tensors)?;
    std::fs::write(path, bytes).map_err(|error| format!("failed to write {}: {}", path.display(), error))
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_value(buf: &mut Vec<u8>, value: &GGUFValue) -> Result<(), String> {
    match value {
        GGUFValue::Uint8(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Uint8 as u32).to_le_bytes());
            buf.push(*inner);
        }
        GGUFValue::Int8(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Int8 as u32).to_le_bytes());
            buf.push(*inner as u8);
        }
        GGUFValue::Uint16(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Uint16 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_le_bytes());
        }
        GGUFValue::Int16(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Int16 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_le_bytes());
        }
        GGUFValue::Uint32(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Uint32 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_le_bytes());
        }
        GGUFValue::Int32(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Int32 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_le_bytes());
        }
        GGUFValue::Float32(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Float32 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_bits().to_le_bytes());
        }
        GGUFValue::Bool(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Bool as u32).to_le_bytes());
            buf.push(u8::from(*inner));
        }
        GGUFValue::String(inner) => {
            buf.extend_from_slice(&(GGUFValueType::String as u32).to_le_bytes());
            write_string(buf, inner);
        }
        GGUFValue::Array(values) => {
            buf.extend_from_slice(&(GGUFValueType::Array as u32).to_le_bytes());
            let elem_type = infer_array_type(values)?;
            buf.extend_from_slice(&(elem_type as u32).to_le_bytes());
            buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                write_array_element(buf, value, elem_type)?;
            }
        }
        GGUFValue::Uint64(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Uint64 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_le_bytes());
        }
        GGUFValue::Int64(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Int64 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_le_bytes());
        }
        GGUFValue::Float64(inner) => {
            buf.extend_from_slice(&(GGUFValueType::Float64 as u32).to_le_bytes());
            buf.extend_from_slice(&inner.to_bits().to_le_bytes());
        }
    }

    Ok(())
}

fn infer_array_type(values: &[GGUFValue]) -> Result<GGUFValueType, String> {
    let first = values
        .first()
        .ok_or("GGUF arrays must have at least one element")?;
    let elem_type = match first {
        GGUFValue::Uint8(_) => GGUFValueType::Uint8,
        GGUFValue::Uint32(_) => GGUFValueType::Uint32,
        GGUFValue::Int32(_) => GGUFValueType::Int32,
        GGUFValue::Float32(_) => GGUFValueType::Float32,
        GGUFValue::String(_) => GGUFValueType::String,
        GGUFValue::Uint64(_) => GGUFValueType::Uint64,
        other => return Err(format!("unsupported GGUF array element {:?}", other)),
    };

    for value in values {
        let current = match value {
            GGUFValue::Uint8(_) => GGUFValueType::Uint8,
            GGUFValue::Uint32(_) => GGUFValueType::Uint32,
            GGUFValue::Int32(_) => GGUFValueType::Int32,
            GGUFValue::Float32(_) => GGUFValueType::Float32,
            GGUFValue::String(_) => GGUFValueType::String,
            GGUFValue::Uint64(_) => GGUFValueType::Uint64,
            other => return Err(format!("unsupported GGUF array element {:?}", other)),
        };
        if current != elem_type {
            return Err("GGUF arrays must be homogeneous".into());
        }
    }

    Ok(elem_type)
}

fn write_array_element(
    buf: &mut Vec<u8>,
    value: &GGUFValue,
    elem_type: GGUFValueType,
) -> Result<(), String> {
    match (elem_type, value) {
        (GGUFValueType::Uint8, GGUFValue::Uint8(inner)) => buf.push(*inner),
        (GGUFValueType::Uint32, GGUFValue::Uint32(inner)) => buf.extend_from_slice(&inner.to_le_bytes()),
        (GGUFValueType::Int32, GGUFValue::Int32(inner)) => buf.extend_from_slice(&inner.to_le_bytes()),
        (GGUFValueType::Float32, GGUFValue::Float32(inner)) => {
            buf.extend_from_slice(&inner.to_bits().to_le_bytes())
        }
        (GGUFValueType::String, GGUFValue::String(inner)) => write_string(buf, inner),
        (GGUFValueType::Uint64, GGUFValue::Uint64(inner)) => buf.extend_from_slice(&inner.to_le_bytes()),
        _ => return Err("GGUF array element type mismatch".into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::GGUFFile;

    #[test]
    fn test_write_and_read_roundtrip() {
        let mut metadata = GGUFMetadata::new();
        metadata.values.insert(
            "general.architecture".into(),
            GGUFValue::String("llama".into()),
        );
        metadata.values.insert(
            "tokenizer.ggml.tokens".into(),
            GGUFValue::Array(vec![GGUFValue::String("a".into()), GGUFValue::String("b".into())]),
        );

        let tensors = vec![GGUFWriterTensor {
            name: "token_embd.weight".into(),
            dims: vec![2, 2],
            dtype: GGMLType::F32,
            data: vec![
                0, 0, 128, 63, 0, 0, 0, 64,
                0, 0, 64, 64, 0, 0, 128, 64,
            ],
        }];

        let bytes = write_gguf_bytes(GGUF_VERSION_3, &metadata, &tensors).unwrap();
        let path = std::env::temp_dir().join("nnx_gguf_writer_roundtrip.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let gguf = GGUFFile::open(&path).unwrap();
        assert_eq!(gguf.metadata.architecture(), Some("llama"));
        assert_eq!(gguf.tensor_names(), vec!["token_embd.weight"]);
        assert_eq!(gguf.tensor_data("token_embd.weight").unwrap(), tensors[0].data.as_slice());

        std::fs::remove_file(path).ok();
    }
}