//! GGUF file parser — reads header, metadata, and tensor descriptors.
//!
//! Supports both GGUF v2 and v3 formats. Tensor data is accessed via
//! memory mapping for zero-copy loading.

use crate::metadata::GGUFMetadata;
use crate::types::*;
use nnx_core::error::{EngineError, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

/// A parsed GGUF file with memory-mapped tensor data.
pub struct GGUFFile {
    /// GGUF format version.
    pub version: u32,
    /// Parsed metadata.
    pub metadata: GGUFMetadata,
    /// Tensor descriptors, keyed by name.
    pub tensors: HashMap<String, GGUFTensorInfo>,
    /// Memory-mapped file data.
    mmap: memmap2::Mmap,
    /// Byte offset where tensor data begins.
    data_offset: usize,
}

impl GGUFFile {
    /// Open and parse a GGUF file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            EngineError::ModelLoad(format!("failed to open {}: {}", path.display(), e))
        })?;

        // SAFETY: memory mapping is unsafe but standard practice for model loading.
        // The file must not be modified while mapped.
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                EngineError::ModelLoad(format!("failed to mmap {}: {}", path.display(), e))
            })?
        };

        let data = &mmap[..];
        if data.len() < 16 {
            return Err(EngineError::ModelLoad("file too small for GGUF header".into()));
        }

        // Parse header
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != GGUF_MAGIC {
            return Err(EngineError::UnsupportedFormat(format!(
                "not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})",
                magic, GGUF_MAGIC
            )));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != GGUF_VERSION_2 && version != GGUF_VERSION_3 {
            return Err(EngineError::UnsupportedFormat(format!(
                "unsupported GGUF version: {} (expected 2 or 3)",
                version
            )));
        }

        let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let metadata_kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;

        info!(
            "GGUF v{}: {} tensors, {} metadata entries",
            version, tensor_count, metadata_kv_count
        );

        // Parse metadata and tensor info
        let mut cursor = 24usize;
        let metadata = Self::parse_metadata(data, &mut cursor, metadata_kv_count)?;
        let tensors = Self::parse_tensor_infos(data, &mut cursor, tensor_count)?;

        // Tensor data starts after alignment
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u32) as usize;

        let data_offset = (cursor + alignment - 1) / alignment * alignment;

        Ok(Self {
            version,
            metadata,
            tensors,
            mmap,
            data_offset,
        })
    }

    /// Get raw bytes for a tensor by name.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self.tensors.get(name).ok_or_else(|| {
            EngineError::ModelLoad(format!("tensor not found: {}", name))
        })?;
        let start = self.data_offset + info.offset as usize;
        let numel = info.numel() as usize;
        let block_numel = info.dtype.block_numel();
        let num_blocks = (numel + block_numel - 1) / block_numel;
        let size = num_blocks * info.dtype.block_size_bytes();
        let end = start + size;

        if end > self.mmap.len() {
            return Err(EngineError::ModelLoad(format!(
                "tensor {} data extends past end of file",
                name
            )));
        }

        Ok(&self.mmap[start..end])
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Total file size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    // -- Private parsing methods --

    fn parse_metadata(
        data: &[u8],
        cursor: &mut usize,
        count: usize,
    ) -> Result<GGUFMetadata> {
        let mut metadata = GGUFMetadata::new();
        for _ in 0..count {
            let key = Self::read_string(data, cursor)?;
            let value = Self::read_value(data, cursor)?;
            metadata.values.insert(key, value);
        }
        Ok(metadata)
    }

    fn parse_tensor_infos(
        data: &[u8],
        cursor: &mut usize,
        count: usize,
    ) -> Result<HashMap<String, GGUFTensorInfo>> {
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let name = Self::read_string(data, cursor)?;
            let n_dims = Self::read_u32(data, cursor)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(data, cursor)?);
            }
            let dtype_raw = Self::read_u32(data, cursor)?;
            let dtype = nnx_quant::GGMLType::from_u32(dtype_raw).ok_or_else(|| {
                EngineError::UnsupportedFormat(format!("unknown GGML type: {}", dtype_raw))
            })?;
            let offset = Self::read_u64(data, cursor)?;

            tensors.insert(
                name.clone(),
                GGUFTensorInfo {
                    name,
                    n_dims,
                    dims,
                    dtype,
                    offset,
                },
            );
        }
        Ok(tensors)
    }

    fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
        if *cursor + 4 > data.len() {
            return Err(EngineError::ModelLoad("unexpected EOF reading u32".into()));
        }
        let v = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        Ok(v)
    }

    fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64> {
        if *cursor + 8 > data.len() {
            return Err(EngineError::ModelLoad("unexpected EOF reading u64".into()));
        }
        let v = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
        *cursor += 8;
        Ok(v)
    }

    fn read_string(data: &[u8], cursor: &mut usize) -> Result<String> {
        let len = Self::read_u64(data, cursor)? as usize;
        if *cursor + len > data.len() {
            return Err(EngineError::ModelLoad("unexpected EOF reading string".into()));
        }
        let s = std::str::from_utf8(&data[*cursor..*cursor + len])
            .map_err(|e| EngineError::ModelLoad(format!("invalid UTF-8 in metadata: {}", e)))?
            .to_string();
        *cursor += len;
        Ok(s)
    }

    fn read_value(data: &[u8], cursor: &mut usize) -> Result<GGUFValue> {
        let vtype = Self::read_u32(data, cursor)?;
        let vtype = GGUFValueType::from_u32(vtype).ok_or_else(|| {
            EngineError::UnsupportedFormat(format!("unknown GGUF value type: {}", vtype))
        })?;

        match vtype {
            GGUFValueType::Uint8 => {
                let v = data.get(*cursor).copied().ok_or_else(|| {
                    EngineError::ModelLoad("EOF reading u8".into())
                })?;
                *cursor += 1;
                Ok(GGUFValue::Uint8(v))
            }
            GGUFValueType::Int8 => {
                let v = data.get(*cursor).copied().ok_or_else(|| {
                    EngineError::ModelLoad("EOF reading i8".into())
                })? as i8;
                *cursor += 1;
                Ok(GGUFValue::Int8(v))
            }
            GGUFValueType::Uint16 => {
                let v = u16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
                *cursor += 2;
                Ok(GGUFValue::Uint16(v))
            }
            GGUFValueType::Int16 => {
                let v = i16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
                *cursor += 2;
                Ok(GGUFValue::Int16(v))
            }
            GGUFValueType::Uint32 => Ok(GGUFValue::Uint32(Self::read_u32(data, cursor)?)),
            GGUFValueType::Int32 => {
                let v = Self::read_u32(data, cursor)? as i32;
                Ok(GGUFValue::Int32(v))
            }
            GGUFValueType::Float32 => {
                let bits = Self::read_u32(data, cursor)?;
                Ok(GGUFValue::Float32(f32::from_bits(bits)))
            }
            GGUFValueType::Bool => {
                let v = data.get(*cursor).copied().ok_or_else(|| {
                    EngineError::ModelLoad("EOF reading bool".into())
                })?;
                *cursor += 1;
                Ok(GGUFValue::Bool(v != 0))
            }
            GGUFValueType::String => Ok(GGUFValue::String(Self::read_string(data, cursor)?)),
            GGUFValueType::Uint64 => Ok(GGUFValue::Uint64(Self::read_u64(data, cursor)?)),
            GGUFValueType::Int64 => {
                let v = Self::read_u64(data, cursor)? as i64;
                Ok(GGUFValue::Int64(v))
            }
            GGUFValueType::Float64 => {
                let bits = Self::read_u64(data, cursor)?;
                Ok(GGUFValue::Float64(f64::from_bits(bits)))
            }
            GGUFValueType::Array => {
                let elem_type = Self::read_u32(data, cursor)?;
                let len = Self::read_u64(data, cursor)? as usize;
                let _ = elem_type; // element type used for homogeneous arrays
                let mut arr = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    // For arrays, we re-read based on the element type
                    // Simplified: store element type and re-parse
                    let val = Self::read_array_element(data, cursor, elem_type)?;
                    arr.push(val);
                }
                Ok(GGUFValue::Array(arr))
            }
        }
    }

    fn read_array_element(data: &[u8], cursor: &mut usize, elem_type: u32) -> Result<GGUFValue> {
        match elem_type {
            0 => { let v = data[*cursor]; *cursor += 1; Ok(GGUFValue::Uint8(v)) }
            4 => Ok(GGUFValue::Uint32(Self::read_u32(data, cursor)?)),
            5 => Ok(GGUFValue::Int32(Self::read_u32(data, cursor)? as i32)),
            6 => Ok(GGUFValue::Float32(f32::from_bits(Self::read_u32(data, cursor)?))),
            8 => Ok(GGUFValue::String(Self::read_string(data, cursor)?)),
            10 => Ok(GGUFValue::Uint64(Self::read_u64(data, cursor)?)),
            _ => Err(EngineError::UnsupportedFormat(format!(
                "unsupported array element type: {}", elem_type
            ))),
        }
    }
}
