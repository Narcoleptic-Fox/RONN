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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use std::io::Write;

    // -- Helper: write a GGUF string (u64 length + utf-8 bytes) --
    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    // -- Helper: write a GGUF metadata KV pair --
    fn write_metadata_kv(buf: &mut Vec<u8>, key: &str, value_type: u32, value_bytes: &[u8]) {
        write_gguf_string(buf, key);
        buf.extend_from_slice(&value_type.to_le_bytes());
        buf.extend_from_slice(value_bytes);
    }

    // -- Helper: write a GGUF string value (type 8) --
    fn string_value_bytes(s: &str) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
        v
    }

    /// Build a synthetic GGUF file in memory.
    ///
    /// `metadata` is a list of (key, value_type_u32, raw_value_bytes).
    /// `tensors` is a list of (name, dims, dtype_u32, data_bytes).
    /// `alignment` overrides the default 32-byte alignment if provided.
    fn build_gguf_bytes(
        version: u32,
        metadata: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, &[u64], u32, &[u8])],
        alignment: Option<u32>,
    ) -> Vec<u8> {
        let effective_alignment = alignment.unwrap_or(GGUF_DEFAULT_ALIGNMENT as u32) as usize;

        let mut buf = Vec::new();

        // Header: magic, version, tensor_count, metadata_kv_count
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

        // Metadata count: include the alignment key if alignment override was
        // requested, because the parser reads it from metadata.
        let extra_meta = if alignment.is_some() { 1 } else { 0 };
        buf.extend_from_slice(&((metadata.len() + extra_meta) as u64).to_le_bytes());

        // Write metadata KV pairs
        for (key, vtype, vbytes) in metadata {
            write_metadata_kv(&mut buf, key, *vtype, vbytes);
        }

        // Write the alignment metadata key if overridden
        if let Some(align_val) = alignment {
            write_metadata_kv(
                &mut buf,
                "general.alignment",
                GGUFValueType::Uint32 as u32,
                &align_val.to_le_bytes(),
            );
        }

        // Write tensor info descriptors
        // Track the cumulative data offset for each tensor
        let mut data_offset: u64 = 0;
        for (name, dims, dtype, data) in tensors {
            write_gguf_string(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&dtype.to_le_bytes());
            buf.extend_from_slice(&data_offset.to_le_bytes());
            data_offset += data.len() as u64;
        }

        // Align to boundary for tensor data
        let current = buf.len();
        let aligned = (current + effective_alignment - 1) / effective_alignment * effective_alignment;
        buf.resize(aligned, 0u8);

        // Write raw tensor data
        for (_, _, _, data) in tensors {
            buf.extend_from_slice(data);
        }

        buf
    }

    /// Write bytes to a temporary file and return its path.
    fn write_temp_file(name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("nnx_gguf_test_{}", name));
        let mut f = std::fs::File::create(&path).expect("failed to create temp file");
        f.write_all(data).expect("failed to write temp file");
        f.sync_all().expect("failed to sync temp file");
        path
    }

    // ---------------------------------------------------------------
    // Test: parse an empty GGUF (0 tensors, 0 metadata)
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_minimal_gguf() {
        let bytes = build_gguf_bytes(3, &[], &[], None);
        let path = write_temp_file("minimal.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse minimal GGUF");
        assert_eq!(gguf.version, 3);
        assert!(gguf.tensors.is_empty());
        assert!(gguf.metadata.values.is_empty());
    }

    // ---------------------------------------------------------------
    // Test: parse all metadata value types
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_metadata_types() {
        let mut meta: Vec<(&str, u32, Vec<u8>)> = Vec::new();

        // Uint8 (type 0)
        meta.push(("key_u8", 0, vec![42]));

        // Int8 (type 1)
        meta.push(("key_i8", 1, vec![0xFE])); // -2 as i8

        // Uint16 (type 2)
        meta.push(("key_u16", 2, 1000u16.to_le_bytes().to_vec()));

        // Int16 (type 3)
        meta.push(("key_i16", 3, (-500i16).to_le_bytes().to_vec()));

        // Uint32 (type 4)
        meta.push(("key_u32", 4, 100_000u32.to_le_bytes().to_vec()));

        // Int32 (type 5)
        meta.push(("key_i32", 5, (-42i32 as u32).to_le_bytes().to_vec()));

        // Float32 (type 6)
        meta.push(("key_f32", 6, 3.14f32.to_bits().to_le_bytes().to_vec()));

        // Bool (type 7)
        meta.push(("key_bool_true", 7, vec![1]));
        meta.push(("key_bool_false", 7, vec![0]));

        // String (type 8)
        meta.push(("key_string", 8, string_value_bytes("hello world")));

        // Uint64 (type 10)
        meta.push(("key_u64", 10, 999_999_999u64.to_le_bytes().to_vec()));

        // Int64 (type 11)
        meta.push(("key_i64", 11, (-123456i64 as u64).to_le_bytes().to_vec()));

        // Float64 (type 12)
        meta.push(("key_f64", 12, 2.71828f64.to_bits().to_le_bytes().to_vec()));

        let bytes = build_gguf_bytes(3, &meta, &[], None);
        let path = write_temp_file("meta_types.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse metadata types");

        // Uint8
        match gguf.metadata.get("key_u8") {
            Some(GGUFValue::Uint8(v)) => assert_eq!(*v, 42),
            other => panic!("expected Uint8(42), got {:?}", other),
        }

        // Int8
        match gguf.metadata.get("key_i8") {
            Some(GGUFValue::Int8(v)) => assert_eq!(*v, -2),
            other => panic!("expected Int8(-2), got {:?}", other),
        }

        // Uint16
        match gguf.metadata.get("key_u16") {
            Some(GGUFValue::Uint16(v)) => assert_eq!(*v, 1000),
            other => panic!("expected Uint16(1000), got {:?}", other),
        }

        // Int16
        match gguf.metadata.get("key_i16") {
            Some(GGUFValue::Int16(v)) => assert_eq!(*v, -500),
            other => panic!("expected Int16(-500), got {:?}", other),
        }

        // Uint32
        match gguf.metadata.get("key_u32") {
            Some(GGUFValue::Uint32(v)) => assert_eq!(*v, 100_000),
            other => panic!("expected Uint32(100000), got {:?}", other),
        }

        // Int32 — note: stored as u32 bits, parsed via read_u32 then cast to i32
        match gguf.metadata.get("key_i32") {
            Some(GGUFValue::Int32(v)) => assert_eq!(*v, -42),
            other => panic!("expected Int32(-42), got {:?}", other),
        }

        // Float32
        match gguf.metadata.get("key_f32") {
            Some(GGUFValue::Float32(v)) => assert!((v - 3.14).abs() < 0.001),
            other => panic!("expected Float32(~3.14), got {:?}", other),
        }

        // Bool
        match gguf.metadata.get("key_bool_true") {
            Some(GGUFValue::Bool(v)) => assert!(*v),
            other => panic!("expected Bool(true), got {:?}", other),
        }
        match gguf.metadata.get("key_bool_false") {
            Some(GGUFValue::Bool(v)) => assert!(!*v),
            other => panic!("expected Bool(false), got {:?}", other),
        }

        // String
        match gguf.metadata.get("key_string") {
            Some(GGUFValue::String(v)) => assert_eq!(v, "hello world"),
            other => panic!("expected String(\"hello world\"), got {:?}", other),
        }

        // Uint64
        match gguf.metadata.get("key_u64") {
            Some(GGUFValue::Uint64(v)) => assert_eq!(*v, 999_999_999),
            other => panic!("expected Uint64(999999999), got {:?}", other),
        }

        // Int64
        match gguf.metadata.get("key_i64") {
            Some(GGUFValue::Int64(v)) => assert_eq!(*v, -123456),
            other => panic!("expected Int64(-123456), got {:?}", other),
        }

        // Float64
        match gguf.metadata.get("key_f64") {
            Some(GGUFValue::Float64(v)) => assert!((v - 2.71828).abs() < 0.00001),
            other => panic!("expected Float64(~2.71828), got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Test: parse a single tensor descriptor (F32 type, 2D shape)
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_tensor_info() {
        // F32 = GGMLType::F32 = 0, 2x3 tensor = 6 elements * 4 bytes = 24 bytes
        let tensor_data = vec![0u8; 24];
        let tensors = [("my_tensor", [2u64, 3].as_slice(), 0u32, tensor_data.as_slice())];
        let bytes = build_gguf_bytes(3, &[], &tensors, None);
        let path = write_temp_file("tensor_info.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse tensor info");
        assert_eq!(gguf.tensors.len(), 1);

        let info = gguf.tensors.get("my_tensor").expect("tensor not found");
        assert_eq!(info.name, "my_tensor");
        assert_eq!(info.n_dims, 2);
        assert_eq!(info.dims, vec![2, 3]);
        assert_eq!(info.dtype, nnx_quant::GGMLType::F32);
        assert_eq!(info.numel(), 6);
    }

    // ---------------------------------------------------------------
    // Test: verify tensor_data() returns correct bytes
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_tensor_data() {
        // Create a 1x4 F32 tensor with known values
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors = [("weights", [1u64, 4].as_slice(), 0u32, tensor_data.as_slice())];
        let bytes = build_gguf_bytes(3, &[], &tensors, None);
        let path = write_temp_file("tensor_data.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse tensor data");
        let data = gguf.tensor_data("weights").expect("should get tensor data");
        assert_eq!(data.len(), 16); // 4 * 4 bytes

        // Verify the actual float values
        for (i, expected) in values.iter().enumerate() {
            let offset = i * 4;
            let bits = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            let actual = f32::from_bits(bits);
            assert_eq!(actual, *expected, "mismatch at index {}", i);
        }
    }

    // ---------------------------------------------------------------
    // Test: both GGUF version 2 and 3 parse correctly
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_version_2_and_3() {
        for version in [2u32, 3] {
            let bytes = build_gguf_bytes(version, &[], &[], None);
            let path = write_temp_file(&format!("v{}.gguf", version), &bytes);
            let result = GGUFFile::open(&path);
            std::fs::remove_file(&path).ok();

            let gguf = result.unwrap_or_else(|e| panic!("v{} should parse: {}", version, e));
            assert_eq!(gguf.version, version);
        }
    }

    // ---------------------------------------------------------------
    // Test: wrong magic number produces an error
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_bad_magic() {
        let mut bytes = build_gguf_bytes(3, &[], &[], None);
        // Overwrite magic with garbage
        bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let path = write_temp_file("bad_magic.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        match result {
            Err(e) => {
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("not a GGUF file"),
                    "unexpected error: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("should have failed on bad magic"),
        }
    }

    // ---------------------------------------------------------------
    // Test: unsupported version produces an error
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_bad_version() {
        let mut bytes = build_gguf_bytes(3, &[], &[], None);
        // Overwrite version to 99
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());

        let path = write_temp_file("bad_version.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        match result {
            Err(e) => {
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("unsupported GGUF version"),
                    "unexpected error: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("should have failed on bad version"),
        }
    }

    // ---------------------------------------------------------------
    // Test: file too short for header
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_truncated_file() {
        // Only 10 bytes — less than the 16 bytes needed for the initial check
        let bytes = vec![0u8; 10];
        let path = write_temp_file("truncated.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        match result {
            Err(e) => {
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("too small"),
                    "unexpected error: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("should have failed on truncated file"),
        }
    }

    // ---------------------------------------------------------------
    // Test: metadata extends past EOF
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_truncated_metadata() {
        // Build a file that claims 1 metadata kv, but has no metadata bytes
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata kv (but none present)
        // No metadata bytes follow — parser will try to read a string key and fail

        let path = write_temp_file("truncated_meta.gguf", &buf);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        assert!(result.is_err(), "should fail on truncated metadata");
    }

    // ---------------------------------------------------------------
    // Test: tensor data extends past EOF
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_truncated_tensor_data() {
        // Build a GGUF with 1 tensor that expects 16 bytes of data, but only
        // provide 4 bytes in the data section.
        let short_data = vec![0u8; 4]; // Only 4 bytes instead of needed 16
        let tensors = [("short", [2u64, 2].as_slice(), 0u32, short_data.as_slice())];
        let bytes = build_gguf_bytes(3, &[], &tensors, None);

        // The builder wrote 4 bytes of data. The tensor descriptor says offset=0,
        // shape 2x2 F32 = 16 bytes needed, but only 4 exist.
        // tensor_data() should fail.
        let path = write_temp_file("truncated_tensor.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("header should parse fine");
        let data_result = gguf.tensor_data("short");
        assert!(
            data_result.is_err(),
            "should fail when tensor data extends past EOF"
        );
    }

    // ---------------------------------------------------------------
    // Test: Unicode strings in metadata
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_string_metadata() {
        let unicode_str = "hello \u{1F600} world \u{00E9}\u{00FC}\u{00F1}";
        let meta = [("unicode_key", 8u32, string_value_bytes(unicode_str))];
        let bytes = build_gguf_bytes(3, &meta, &[], None);
        let path = write_temp_file("unicode.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse unicode metadata");
        match gguf.metadata.get("unicode_key") {
            Some(GGUFValue::String(v)) => assert_eq!(v, unicode_str),
            other => panic!("expected unicode string, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Test: arrays of strings and u32s
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_array_metadata() {
        // Array of u32s: type=9 (array), element_type=4 (u32), length=3, values
        let mut u32_arr_bytes = Vec::new();
        u32_arr_bytes.extend_from_slice(&4u32.to_le_bytes()); // elem type = Uint32
        u32_arr_bytes.extend_from_slice(&3u64.to_le_bytes()); // 3 elements
        u32_arr_bytes.extend_from_slice(&10u32.to_le_bytes());
        u32_arr_bytes.extend_from_slice(&20u32.to_le_bytes());
        u32_arr_bytes.extend_from_slice(&30u32.to_le_bytes());

        // Array of strings: type=9 (array), element_type=8 (string), length=2
        let mut str_arr_bytes = Vec::new();
        str_arr_bytes.extend_from_slice(&8u32.to_le_bytes()); // elem type = String
        str_arr_bytes.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
        // String "foo"
        str_arr_bytes.extend_from_slice(&3u64.to_le_bytes());
        str_arr_bytes.extend_from_slice(b"foo");
        // String "bar"
        str_arr_bytes.extend_from_slice(&3u64.to_le_bytes());
        str_arr_bytes.extend_from_slice(b"bar");

        let meta = [
            ("u32_array", 9u32, u32_arr_bytes),
            ("str_array", 9u32, str_arr_bytes),
        ];
        let bytes = build_gguf_bytes(3, &meta, &[], None);
        let path = write_temp_file("arrays.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse array metadata");

        // Check u32 array
        match gguf.metadata.get("u32_array") {
            Some(GGUFValue::Array(arr)) => {
                assert_eq!(arr.len(), 3);
                match &arr[0] {
                    GGUFValue::Uint32(v) => assert_eq!(*v, 10),
                    other => panic!("expected Uint32, got {:?}", other),
                }
                match &arr[1] {
                    GGUFValue::Uint32(v) => assert_eq!(*v, 20),
                    other => panic!("expected Uint32, got {:?}", other),
                }
                match &arr[2] {
                    GGUFValue::Uint32(v) => assert_eq!(*v, 30),
                    other => panic!("expected Uint32, got {:?}", other),
                }
            }
            other => panic!("expected Array, got {:?}", other),
        }

        // Check string array
        match gguf.metadata.get("str_array") {
            Some(GGUFValue::Array(arr)) => {
                assert_eq!(arr.len(), 2);
                match &arr[0] {
                    GGUFValue::String(v) => assert_eq!(v, "foo"),
                    other => panic!("expected String, got {:?}", other),
                }
                match &arr[1] {
                    GGUFValue::String(v) => assert_eq!(v, "bar"),
                    other => panic!("expected String, got {:?}", other),
                }
            }
            other => panic!("expected Array, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Test: tensor_names() returns all names
    // ---------------------------------------------------------------
    #[test]
    fn test_tensor_names() {
        let d1 = vec![0u8; 4]; // 1-element F32
        let d2 = vec![0u8; 8]; // 2-element F32
        let tensors = [
            ("alpha", [1u64].as_slice(), 0u32, d1.as_slice()),
            ("beta", [2u64].as_slice(), 0u32, d2.as_slice()),
        ];
        let bytes = build_gguf_bytes(3, &[], &tensors, None);
        let path = write_temp_file("names.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse");
        let mut names = gguf.tensor_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    // ---------------------------------------------------------------
    // Test: file_size() returns correct value
    // ---------------------------------------------------------------
    #[test]
    fn test_file_size() {
        let bytes = build_gguf_bytes(3, &[], &[], None);
        let expected_size = bytes.len();
        let path = write_temp_file("size.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse");
        assert_eq!(gguf.file_size(), expected_size);
    }

    // ---------------------------------------------------------------
    // Test: custom alignment in metadata
    // ---------------------------------------------------------------
    #[test]
    fn test_alignment() {
        // Use a custom alignment of 64 bytes
        let tensor_data = vec![0u8; 4]; // 1-element F32
        let tensors = [("aligned", [1u64].as_slice(), 0u32, tensor_data.as_slice())];
        let bytes = build_gguf_bytes(3, &[], &tensors, Some(64));
        let path = write_temp_file("alignment.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse with custom alignment");
        // Verify we can still read the tensor data correctly
        let data = gguf.tensor_data("aligned").expect("should read tensor");
        assert_eq!(data.len(), 4);

        // Verify alignment metadata was parsed
        match gguf.metadata.get("general.alignment") {
            Some(GGUFValue::Uint32(v)) => assert_eq!(*v, 64),
            other => panic!("expected alignment=64, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Test: multiple tensors with different dtypes
    // ---------------------------------------------------------------
    #[test]
    fn test_multiple_tensors() {
        // F32 tensor (dtype=0): 2 elements = 8 bytes
        let f32_data = vec![0u8; 8];
        // F16 tensor (dtype=1): 4 elements = 8 bytes (2 bytes each)
        let f16_data = vec![0u8; 8];
        // Q8_0 tensor (dtype=8): 32 elements = 34 bytes (one block)
        let q8_data = vec![0u8; 34];

        let tensors = [
            ("f32_t", [2u64].as_slice(), 0u32, f32_data.as_slice()),
            ("f16_t", [4u64].as_slice(), 1u32, f16_data.as_slice()),
            ("q8_t", [32u64].as_slice(), 8u32, q8_data.as_slice()),
        ];
        let bytes = build_gguf_bytes(3, &[], &tensors, None);
        let path = write_temp_file("multi_tensor.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse multiple tensors");
        assert_eq!(gguf.tensors.len(), 3);

        let f32_info = gguf.tensors.get("f32_t").unwrap();
        assert_eq!(f32_info.dtype, nnx_quant::GGMLType::F32);
        assert_eq!(f32_info.dims, vec![2]);

        let f16_info = gguf.tensors.get("f16_t").unwrap();
        assert_eq!(f16_info.dtype, nnx_quant::GGMLType::F16);
        assert_eq!(f16_info.dims, vec![4]);

        let q8_info = gguf.tensors.get("q8_t").unwrap();
        assert_eq!(q8_info.dtype, nnx_quant::GGMLType::Q8_0);
        assert_eq!(q8_info.dims, vec![32]);

        // Verify data access works for all
        gguf.tensor_data("f32_t").expect("should read f32_t");
        gguf.tensor_data("f16_t").expect("should read f16_t");
        gguf.tensor_data("q8_t").expect("should read q8_t");
    }

    // ---------------------------------------------------------------
    // Test: tensor_data for a nonexistent tensor returns error
    // ---------------------------------------------------------------
    #[test]
    fn test_tensor_data_not_found() {
        let bytes = build_gguf_bytes(3, &[], &[], None);
        let path = write_temp_file("no_tensor.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse");
        let err = gguf.tensor_data("nonexistent");
        assert!(err.is_err());
    }

    // ---------------------------------------------------------------
    // Test: metadata combined with tensor info (realistic scenario)
    // ---------------------------------------------------------------
    #[test]
    fn test_metadata_with_tensors() {
        let meta = [
            ("general.architecture", 8u32, string_value_bytes("llama")),
            ("general.name", 8u32, string_value_bytes("test-model")),
            ("llama.block_count", 4u32, 32u32.to_le_bytes().to_vec()),
        ];
        let tensor_data = vec![0u8; 16]; // 4-element F32
        let tensors = [("blk.0.attn_q.weight", [4u64].as_slice(), 0u32, tensor_data.as_slice())];
        let bytes = build_gguf_bytes(3, &meta, &tensors, None);
        let path = write_temp_file("meta_tensors.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse");

        // Verify metadata accessors
        assert_eq!(gguf.metadata.architecture(), Some("llama"));
        assert_eq!(gguf.metadata.name(), Some("test-model"));
        assert_eq!(gguf.metadata.num_layers(), Some(32));

        // Verify tensor
        assert_eq!(gguf.tensors.len(), 1);
        let data = gguf.tensor_data("blk.0.attn_q.weight").expect("should read tensor");
        assert_eq!(data.len(), 16);
    }

    // ---------------------------------------------------------------
    // Test: 3D tensor shape
    // ---------------------------------------------------------------
    #[test]
    fn test_3d_tensor_shape() {
        // 2x3x4 F32 tensor = 24 elements = 96 bytes
        let tensor_data = vec![0u8; 96];
        let tensors = [("cube", [2u64, 3, 4].as_slice(), 0u32, tensor_data.as_slice())];
        let bytes = build_gguf_bytes(3, &[], &tensors, None);
        let path = write_temp_file("3d_tensor.gguf", &bytes);
        let result = GGUFFile::open(&path);
        std::fs::remove_file(&path).ok();

        let gguf = result.expect("should parse");
        let info = gguf.tensors.get("cube").unwrap();
        assert_eq!(info.n_dims, 3);
        assert_eq!(info.dims, vec![2, 3, 4]);
        assert_eq!(info.numel(), 24);
    }
}
