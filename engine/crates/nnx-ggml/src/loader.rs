//! GGML binary file loader.
//!
//! GGML files have a simpler structure than GGUF:
//! - Magic number (0x67676d6c for "ggml" or versioned variants)
//! - Hyperparameters (architecture-specific, no standard layout)
//! - Vocabulary
//! - Tensor data
//!
//! Because GGML has no standardized metadata, the caller must provide
//! architecture information (number of layers, hidden dim, etc.).

use nnx_core::error::{EngineError, Result};
use nnx_core::shape::Shape;
use nnx_quant::GGMLType;
use std::collections::HashMap;
use std::path::Path;
use tracing::warn;

/// Known GGML magic numbers.
pub const GGML_MAGIC: u32 = 0x67676D6C; // "ggml"
pub const GGML_MAGIC_V1: u32 = 0x67676D66; // "ggmf" (versioned)
pub const GGML_MAGIC_V2: u32 = 0x67676A74; // "ggjt" (version 2)

/// Architecture hint — since GGML files don't self-describe, the caller
/// must tell us what architecture to expect.
#[derive(Debug, Clone)]
pub struct GGMLArchHint {
    pub architecture: String,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
}

/// A parsed GGML file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GGMLHeader {
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub f16: bool,
}

/// Tensor descriptor for one GGML tensor entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GGMLTensorInfo {
    pub name: String,
    pub dtype: GGMLType,
    pub shape: Shape,
    pub data_offsets: (usize, usize),
}

/// Zero-copy view into a GGML tensor payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GGMLTensorView<'a> {
    data: &'a [u8],
    shape: Shape,
    dtype: GGMLType,
}

impl<'a> GGMLTensorView<'a> {
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> GGMLType {
        self.dtype
    }
}

pub struct GGMLFile {
    pub magic: u32,
    pub arch: GGMLArchHint,
    pub header: GGMLHeader,
    pub tensors: HashMap<String, GGMLTensorInfo>,
    mmap: memmap2::Mmap,
}

impl GGMLFile {
    /// Open a GGML file with an architecture hint.
    pub fn open(path: &Path, arch: GGMLArchHint) -> Result<Self> {
        warn!(
            "Loading legacy GGML format from {}. Consider converting to GGUF.",
            path.display()
        );

        let file = std::fs::File::open(path).map_err(|e| {
            EngineError::ModelLoad(format!("failed to open {}: {}", path.display(), e))
        })?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                EngineError::ModelLoad(format!("failed to mmap {}: {}", path.display(), e))
            })?
        };

        let data = &mmap[..];
        if data.len() < 4 {
            return Err(EngineError::ModelLoad("file too small".into()));
        }

        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != GGML_MAGIC && magic != GGML_MAGIC_V1 && magic != GGML_MAGIC_V2 {
            return Err(EngineError::UnsupportedFormat(format!(
                "not a GGML file (magic: 0x{:08X})",
                magic
            )));
        }

        let mut cursor = 4usize;
        let header = GGMLHeader {
            n_vocab: Self::read_u32(data, &mut cursor)?,
            n_embd: Self::read_u32(data, &mut cursor)?,
            n_head: Self::read_u32(data, &mut cursor)?,
            n_layer: Self::read_u32(data, &mut cursor)?,
            f16: Self::read_u32(data, &mut cursor)? != 0,
        };

        let tensors = Self::parse_tensors(data, &mut cursor)?;

        Ok(Self {
            magic,
            arch,
            header,
            tensors,
            mmap,
        })
    }

    /// File size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|name| name.as_str()).collect()
    }

    /// Get tensor metadata by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GGMLTensorInfo> {
        self.tensors.get(name)
    }

    /// Get a zero-copy view of a tensor's raw bytes.
    pub fn tensor_view(&self, name: &str) -> Result<GGMLTensorView<'_>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| EngineError::ModelLoad(format!("tensor not found: {}", name)))?;
        Ok(GGMLTensorView {
            data: &self.mmap[info.data_offsets.0..info.data_offsets.1],
            shape: info.shape.clone(),
            dtype: info.dtype,
        })
    }

    fn parse_tensors(data: &[u8], cursor: &mut usize) -> Result<HashMap<String, GGMLTensorInfo>> {
        let mut tensors = HashMap::new();

        while *cursor < data.len() {
            let remaining = data.len() - *cursor;
            if remaining < 12 {
                break;
            }

            let n_dims = Self::read_u32(data, cursor)? as usize;
            let name_len = Self::read_u32(data, cursor)? as usize;
            let ftype_raw = Self::read_u32(data, cursor)?;
            let dtype = GGMLType::from_u32(ftype_raw).ok_or_else(|| {
                EngineError::UnsupportedFormat(format!("unknown GGML tensor type: {}", ftype_raw))
            })?;

            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(Self::read_u32(data, cursor)? as usize);
            }

            let name = Self::read_string(data, cursor, name_len)?;
            *cursor = Self::align_up(*cursor, 32);

            let shape = Shape::from(dims);
            let start = *cursor;
            let end = start + Self::tensor_size_bytes(dtype, shape.numel())?;
            if end > data.len() {
                return Err(EngineError::ModelLoad(format!(
                    "tensor {} extends past end of file",
                    name
                )));
            }

            tensors.insert(
                name.clone(),
                GGMLTensorInfo {
                    name,
                    dtype,
                    shape,
                    data_offsets: (start, end),
                },
            );

            *cursor = end;
        }

        Ok(tensors)
    }

    fn tensor_size_bytes(dtype: GGMLType, numel: usize) -> Result<usize> {
        let block_numel = dtype.block_numel();
        let block_size = dtype.block_size_bytes();
        if block_size == 0 {
            return Err(EngineError::UnsupportedFormat(format!(
                "GGML tensor type {} does not have a known block size",
                dtype
            )));
        }

        let blocks = numel.div_ceil(block_numel);
        Ok(blocks * block_size)
    }

    fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
        if *cursor + 4 > data.len() {
            return Err(EngineError::ModelLoad("unexpected EOF reading u32".into()));
        }
        let value = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        Ok(value)
    }

    fn read_string(data: &[u8], cursor: &mut usize, len: usize) -> Result<String> {
        if *cursor + len > data.len() {
            return Err(EngineError::ModelLoad(
                "unexpected EOF reading tensor name".into(),
            ));
        }

        let value = std::str::from_utf8(&data[*cursor..*cursor + len])
            .map_err(|e| EngineError::ModelLoad(format!("invalid tensor name UTF-8: {}", e)))?
            .to_string();
        *cursor += len;
        Ok(value)
    }

    fn align_up(value: usize, alignment: usize) -> usize {
        value.div_ceil(alignment) * alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("nnx_ggml_test_{}", name))
    }

    fn write_test_file(path: &Path, data: &[u8]) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(data).unwrap();
        file.sync_all().unwrap();
    }

    fn arch_hint() -> GGMLArchHint {
        GGMLArchHint {
            architecture: "llama".into(),
            num_layers: 1,
            hidden_dim: 2,
            num_heads: 1,
            vocab_size: 8,
        }
    }

    fn build_synthetic_ggml() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGML_MAGIC_V2.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(21u32).to_le_bytes());
        bytes.extend_from_slice(&(GGMLType::F32 as u32).to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(b"tok_embeddings.weight");

        let aligned_len = GGMLFile::align_up(bytes.len(), 32);
        bytes.resize(aligned_len, 0);
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn open_parses_header_and_tensor_directory() {
        let path = temp_path("roundtrip.bin");
        write_test_file(&path, &build_synthetic_ggml());

        let ggml = GGMLFile::open(&path, arch_hint()).unwrap();
        assert_eq!(ggml.magic, GGML_MAGIC_V2);
        assert_eq!(ggml.header.n_vocab, 8);
        assert_eq!(ggml.header.n_embd, 2);
        assert_eq!(ggml.header.n_head, 1);
        assert_eq!(ggml.header.n_layer, 1);
        assert!(!ggml.header.f16);

        let mut names = ggml.tensor_names();
        names.sort_unstable();
        assert_eq!(names, vec!["tok_embeddings.weight"]);

        let info = ggml.tensor_info("tok_embeddings.weight").unwrap();
        assert_eq!(info.dtype, GGMLType::F32);
        assert_eq!(info.shape.dims(), &[3, 2]);

        let view = ggml.tensor_view("tok_embeddings.weight").unwrap();
        assert_eq!(view.dtype(), GGMLType::F32);
        assert_eq!(view.shape().dims(), &[3, 2]);
        assert_eq!(view.as_bytes().len(), 24);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_invalid_magic() {
        let path = temp_path("bad_magic.bin");
        write_test_file(&path, &[0, 1, 2, 3, 4, 5, 6, 7]);
        let result = GGMLFile::open(&path, arch_hint());
        assert!(result.is_err());
        std::fs::remove_file(path).ok();
    }
}
