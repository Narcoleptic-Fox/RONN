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
            return Err(EngineError::ModelLoad(
                "file too small for SafeTensors header".into(),
            ));
        }

        // First 8 bytes: header size as u64 LE
        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;

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
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| EngineError::ModelLoad(format!("tensor not found: {}", name)))?;
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
                )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a synthetic SafeTensors file in memory.
    ///
    /// `tensors` is a list of (name, dtype_string, shape, raw_data_bytes).
    /// `include_metadata` controls whether a __metadata__ key is added to the header.
    fn build_safetensors_bytes(
        tensors: &[(&str, &str, &[usize], &[u8])],
        include_metadata: bool,
    ) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;

        for (name, dtype, shape, data) in tensors {
            let end = offset + data.len();
            let shape_json: Vec<serde_json::Value> =
                shape.iter().map(|&d| serde_json::json!(d)).collect();
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape_json,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }

        if include_metadata {
            header.insert(
                "__metadata__".to_string(),
                serde_json::json!({"format": "pt", "version": "1.0"}),
            );
        }

        let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let header_bytes = header_json.as_bytes();

        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        for (_, _, _, data) in tensors {
            buf.extend_from_slice(data);
        }
        buf
    }

    /// Write bytes to a temporary file and return its path.
    fn write_temp_file(name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("nnx_st_test_{}", name));
        let mut f = std::fs::File::create(&path).expect("failed to create temp file");
        f.write_all(data).expect("failed to write temp file");
        f.sync_all().expect("failed to sync temp file");
        path
    }

    // ---------------------------------------------------------------
    // Test: single F32 tensor
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_minimal_safetensors() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors = [("weight", "F32", [2usize, 3].as_slice(), data.as_slice())];
        let bytes = build_safetensors_bytes(&tensors, false);
        let path = write_temp_file("minimal.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse minimal safetensors");
        assert_eq!(st.tensor_names().len(), 1);

        let info = st.tensor_info("weight").expect("tensor should exist");
        assert_eq!(info.dtype, DType::F32);
        assert_eq!(info.shape.dims(), &[2, 3]);
    }

    // ---------------------------------------------------------------
    // Test: multiple tensors with different dtypes
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_multiple_tensors() {
        // F32: 2 elements = 8 bytes
        let f32_data = vec![0u8; 8];
        // I32: 3 elements = 12 bytes
        let i32_data = vec![0u8; 12];
        // U8: 4 elements = 4 bytes
        let u8_data = vec![0u8; 4];

        let tensors = [
            ("t_f32", "F32", [2usize].as_slice(), f32_data.as_slice()),
            ("t_i32", "I32", [3usize].as_slice(), i32_data.as_slice()),
            ("t_u8", "U8", [4usize].as_slice(), u8_data.as_slice()),
        ];
        let bytes = build_safetensors_bytes(&tensors, false);
        let path = write_temp_file("multiple.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse multiple tensors");
        assert_eq!(st.tensor_names().len(), 3);

        let f32_info = st.tensor_info("t_f32").unwrap();
        assert_eq!(f32_info.dtype, DType::F32);

        let i32_info = st.tensor_info("t_i32").unwrap();
        assert_eq!(i32_info.dtype, DType::I32);

        let u8_info = st.tensor_info("t_u8").unwrap();
        assert_eq!(u8_info.dtype, DType::U8);
    }

    // ---------------------------------------------------------------
    // Test: all supported dtype strings
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_all_dtypes() {
        let dtypes_and_sizes: Vec<(&str, &str, DType, usize)> = vec![
            ("t_f32", "F32", DType::F32, 4),
            ("t_f16", "F16", DType::F16, 2),
            ("t_bf16", "BF16", DType::BF16, 2),
            ("t_f64", "F64", DType::F64, 8),
            ("t_i8", "I8", DType::I8, 1),
            ("t_i16", "I16", DType::I16, 2),
            ("t_i32", "I32", DType::I32, 4),
            ("t_i64", "I64", DType::I64, 8),
            ("t_u8", "U8", DType::U8, 1),
            ("t_bool", "BOOL", DType::Bool, 1),
        ];

        // Each tensor has 1 element
        let tensors: Vec<(&str, &str, &[usize], Vec<u8>)> = dtypes_and_sizes
            .iter()
            .map(|(name, dtype_str, _, size)| {
                (*name, *dtype_str, [1usize].as_slice(), vec![0u8; *size])
            })
            .collect();

        let tensor_refs: Vec<(&str, &str, &[usize], &[u8])> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| (*name, *dtype, *shape, data.as_slice()))
            .collect();

        let bytes = build_safetensors_bytes(&tensor_refs, false);
        let path = write_temp_file("all_dtypes.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse all dtypes");
        assert_eq!(st.tensor_names().len(), dtypes_and_sizes.len());

        for (name, _, expected_dtype, _) in &dtypes_and_sizes {
            let info = st
                .tensor_info(name)
                .unwrap_or_else(|| panic!("tensor {} not found", name));
            assert_eq!(info.dtype, *expected_dtype, "dtype mismatch for {}", name);
        }
    }

    // ---------------------------------------------------------------
    // Test: __metadata__ key is ignored
    // ---------------------------------------------------------------
    #[test]
    fn test_parse_metadata_skipped() {
        let data = vec![0u8; 4]; // 1 F32 element
        let tensors = [("weight", "F32", [1usize].as_slice(), data.as_slice())];
        let bytes = build_safetensors_bytes(&tensors, true);
        let path = write_temp_file("with_metadata.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse with metadata");
        // Should have only 1 tensor, not 2 (metadata should be skipped)
        assert_eq!(st.tensor_names().len(), 1);
        assert!(st.tensor_info("__metadata__").is_none());
        assert!(st.tensor_info("weight").is_some());
    }

    // ---------------------------------------------------------------
    // Test: tensor_view() returns correct data
    // ---------------------------------------------------------------
    #[test]
    fn test_tensor_view() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors = [("view_test", "F32", [2usize, 2].as_slice(), data.as_slice())];
        let bytes = build_safetensors_bytes(&tensors, false);
        let path = write_temp_file("view.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse");
        let view = st.tensor_view("view_test").expect("should get tensor view");

        assert_eq!(view.shape().dims(), &[2, 2]);
        assert_eq!(view.dtype(), DType::F32);
        assert_eq!(view.as_bytes().len(), 16); // 4 * 4 bytes

        // Verify actual float values through the byte data
        let raw = view.as_bytes();
        for (i, expected) in values.iter().enumerate() {
            let offset = i * 4;
            let bits = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
            let actual = f32::from_bits(bits);
            assert_eq!(actual, *expected, "mismatch at index {}", i);
        }
    }

    // ---------------------------------------------------------------
    // Test: tensor_names() returns all names
    // ---------------------------------------------------------------
    #[test]
    fn test_tensor_names() {
        let d = vec![0u8; 4];
        let tensors = [
            ("alpha", "F32", [1usize].as_slice(), d.as_slice()),
            ("beta", "F32", [1usize].as_slice(), d.as_slice()),
            ("gamma", "F32", [1usize].as_slice(), d.as_slice()),
        ];
        // Offsets will overlap but that's fine for this test — we only check names
        // Actually, build_safetensors_bytes accumulates offsets so they won't overlap.
        let bytes = build_safetensors_bytes(&tensors, false);
        let path = write_temp_file("names.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse");
        let mut names = st.tensor_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    // ---------------------------------------------------------------
    // Test: tensor_info() returns correct metadata
    // ---------------------------------------------------------------
    #[test]
    fn test_tensor_info() {
        let data = vec![0u8; 48]; // 2x3 F32 = 6 elements * 4 bytes = 24... no, let's be correct
        // 2x3 F64 = 6 elements * 8 bytes = 48 bytes
        let tensors = [("info_test", "F64", [2usize, 3].as_slice(), data.as_slice())];
        let bytes = build_safetensors_bytes(&tensors, false);
        let path = write_temp_file("info.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse");
        let info = st.tensor_info("info_test").expect("tensor should exist");
        assert_eq!(info.name, "info_test");
        assert_eq!(info.dtype, DType::F64);
        assert_eq!(info.shape.dims(), &[2, 3]);
        assert_eq!(info.shape.numel(), 6);
        assert_eq!(info.data_offsets, (0, 48));

        // Nonexistent tensor returns None
        assert!(st.tensor_info("nonexistent").is_none());
    }

    // ---------------------------------------------------------------
    // Test: file too short for header size field
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_truncated_header() {
        // Only 4 bytes — less than the 8 needed for the header size
        let bytes = vec![0u8; 4];
        let path = write_temp_file("truncated.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
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
            Ok(_) => panic!("should have failed on truncated header"),
        }
    }

    // ---------------------------------------------------------------
    // Test: header claims more bytes than file contains
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_header_extends_past_eof() {
        // Header size says 1000 bytes, but file only has 16 bytes total
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1000u64.to_le_bytes());
        bytes.extend_from_slice(b"{}"); // 2 bytes of JSON, not 1000
        let path = write_temp_file("header_past_eof.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        match result {
            Err(e) => {
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("header extends past EOF"),
                    "unexpected error: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("should have failed when header extends past EOF"),
        }
    }

    // ---------------------------------------------------------------
    // Test: invalid JSON in header
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_invalid_json() {
        let bad_json = b"{ this is not valid json }}}";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(bad_json);
        let path = write_temp_file("bad_json.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        match result {
            Err(e) => {
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("invalid header JSON"),
                    "unexpected error: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("should have failed on invalid JSON"),
        }
    }

    // ---------------------------------------------------------------
    // Test: tensor without dtype field
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_missing_dtype() {
        let header = serde_json::json!({
            "bad_tensor": {
                "shape": [2, 3],
                "data_offsets": [0, 24]
            }
        });
        let header_json = serde_json::to_string(&header).unwrap();
        let header_bytes = header_json.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_bytes);
        bytes.extend_from_slice(&vec![0u8; 24]); // data section

        let path = write_temp_file("missing_dtype.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        match result {
            Err(e) => {
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("missing dtype"),
                    "unexpected error: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("should have failed on missing dtype"),
        }
    }

    // ---------------------------------------------------------------
    // Test: unknown dtype string
    // ---------------------------------------------------------------
    #[test]
    fn test_reject_unknown_dtype() {
        let header = serde_json::json!({
            "tensor": {
                "dtype": "COMPLEX128",
                "shape": [1],
                "data_offsets": [0, 16]
            }
        });
        let header_json = serde_json::to_string(&header).unwrap();
        let header_bytes = header_json.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_bytes);
        bytes.extend_from_slice(&vec![0u8; 16]);

        let path = write_temp_file("unknown_dtype.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        match result {
            Err(e) => {
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("unknown dtype"),
                    "unexpected error: {}",
                    err_msg
                );
            }
            Ok(_) => panic!("should have failed on unknown dtype"),
        }
    }

    // ---------------------------------------------------------------
    // Test: multi-dimensional shapes (0D, 1D, 2D, 3D, 4D)
    // ---------------------------------------------------------------
    #[test]
    fn test_multi_dimensional_shapes() {
        // 0D (scalar): 1 element
        let d0 = vec![0u8; 4]; // 1 F32
        // 1D: [5] = 5 elements
        let d1 = vec![0u8; 20]; // 5 F32
        // 2D: [2,3] = 6 elements
        let d2 = vec![0u8; 24]; // 6 F32
        // 3D: [2,3,4] = 24 elements
        let d3 = vec![0u8; 96]; // 24 F32
        // 4D: [2,3,4,5] = 120 elements
        let d4 = vec![0u8; 480]; // 120 F32

        let tensors = [
            ("scalar", "F32", [].as_slice(), d0.as_slice()),
            ("vec", "F32", [5usize].as_slice(), d1.as_slice()),
            ("matrix", "F32", [2usize, 3].as_slice(), d2.as_slice()),
            ("cube", "F32", [2usize, 3, 4].as_slice(), d3.as_slice()),
            (
                "hypercube",
                "F32",
                [2usize, 3, 4, 5].as_slice(),
                d4.as_slice(),
            ),
        ];

        let bytes = build_safetensors_bytes(&tensors, false);
        let path = write_temp_file("shapes.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse multi-dimensional shapes");
        assert_eq!(st.tensor_names().len(), 5);

        let scalar_info = st.tensor_info("scalar").unwrap();
        assert_eq!(scalar_info.shape.dims(), &[] as &[usize]);
        assert_eq!(scalar_info.shape.numel(), 1);

        let vec_info = st.tensor_info("vec").unwrap();
        assert_eq!(vec_info.shape.dims(), &[5]);

        let matrix_info = st.tensor_info("matrix").unwrap();
        assert_eq!(matrix_info.shape.dims(), &[2, 3]);

        let cube_info = st.tensor_info("cube").unwrap();
        assert_eq!(cube_info.shape.dims(), &[2, 3, 4]);

        let hyper_info = st.tensor_info("hypercube").unwrap();
        assert_eq!(hyper_info.shape.dims(), &[2, 3, 4, 5]);
    }

    // ---------------------------------------------------------------
    // Test: tensor_view for nonexistent tensor returns error
    // ---------------------------------------------------------------
    #[test]
    fn test_tensor_view_not_found() {
        let data = vec![0u8; 4];
        let tensors = [("exists", "F32", [1usize].as_slice(), data.as_slice())];
        let bytes = build_safetensors_bytes(&tensors, false);
        let path = write_temp_file("view_notfound.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse");
        let view_err = st.tensor_view("does_not_exist");
        assert!(view_err.is_err());
    }

    // ---------------------------------------------------------------
    // Test: empty header (no tensors)
    // ---------------------------------------------------------------
    #[test]
    fn test_empty_header() {
        let bytes = build_safetensors_bytes(&[], false);
        let path = write_temp_file("empty.safetensors", &bytes);
        let result = SafeTensorsFile::open(&path);
        std::fs::remove_file(&path).ok();

        let st = result.expect("should parse empty safetensors");
        assert!(st.tensor_names().is_empty());
    }
}
