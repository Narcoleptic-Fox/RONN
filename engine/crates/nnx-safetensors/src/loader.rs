//! SafeTensors file loading with memory-mapped access.

use nnx_core::dtype::DType;
use nnx_core::error::{EngineError, Result};
use nnx_core::shape::Shape;
use nnx_core::tensor::TensorView;
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

/// Tensor metadata from the SafeTensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Shape,
    pub data_offsets: (usize, usize), // (start, end) in data section
}

/// A parsed SafeTensors file with memory-mapped data.
pub struct SafeTensorsFile {
    tensors: HashMap<String, TensorInfo>,
    mmap: memmap2::Mmap,
    data_start: usize,
}

impl SafeTensorsFile {
    /// Open and parse a SafeTensors file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            EngineError::ModelLoad(format!("failed to open {}: {}", path.display(), e))
        })?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                EngineError::ModelLoad(format!("failed to mmap {}: {}", path.display(), e))
            })?
        };

        let data = &mmap[..];
        if data.len() < 8 {
            return Err(EngineError::ModelLoad("file too small for SafeTensors header".into()));
        }

        // First 8 bytes: header size as u64 LE
        let header_size =
            u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;

        if 8 + header_size > data.len() {
            return Err(EngineError::ModelLoad("header extends past EOF".into()));
        }

        // Header is JSON
        let header_json = std::str::from_utf8(&data[8..8 + header_size])
            .map_err(|e| EngineError::ModelLoad(format!("invalid header UTF-8: {}", e)))?;

        let header: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| EngineError::ModelLoad(format!("invalid header JSON: {}", e)))?;

        let data_start = 8 + header_size;
        let mut tensors = HashMap::new();

        if let serde_json::Value::Object(map) = header {
            for (name, value) in map {
                if name == "__metadata__" {
                    continue;
                }
                let info = Self::parse_tensor_info(&name, &value)?;
                tensors.insert(name, info);
            }
        }

        info!(
            "SafeTensors: {} tensors, {:.1} MB",
            tensors.len(),
            mmap.len() as f64 / 1_048_576.0
        );

        Ok(Self {
            tensors,
            mmap,
            data_start,
        })
    }

    /// Get a zero-copy view of a tensor's data.
    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            EngineError::ModelLoad(format!("tensor not found: {}", name))
        })?;
        let start = self.data_start + info.data_offsets.0;
        let end = self.data_start + info.data_offsets.1;
        TensorView::new(&self.mmap[start..end], info.shape.clone(), info.dtype)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor metadata by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    fn parse_tensor_info(name: &str, value: &serde_json::Value) -> Result<TensorInfo> {
        let dtype_str = value["dtype"]
            .as_str()
            .ok_or_else(|| EngineError::ModelLoad(format!("{}: missing dtype", name)))?;

        let dtype = match dtype_str {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F64" => DType::F64,
            "I8" => DType::I8,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U8" => DType::U8,
            "BOOL" => DType::Bool,
            other => {
                return Err(EngineError::UnsupportedFormat(format!(
                    "{}: unknown dtype '{}'",
                    name, other
                )))
            }
        };

        let shape_arr = value["shape"]
            .as_array()
            .ok_or_else(|| EngineError::ModelLoad(format!("{}: missing shape", name)))?;
        let dims: Vec<usize> = shape_arr
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let offsets = value["data_offsets"]
            .as_array()
            .ok_or_else(|| EngineError::ModelLoad(format!("{}: missing data_offsets", name)))?;
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;

        Ok(TensorInfo {
            name: name.to_string(),
            dtype,
            shape: Shape::from(dims),
            data_offsets: (start, end),
        })
    }
}
