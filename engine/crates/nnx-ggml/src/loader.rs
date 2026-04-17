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

/// Default values for config inference when metadata is unavailable.
const DEFAULT_MAX_CONTEXT: usize = 2048;
const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;

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

/// Load a GGML model into an `nnx_transformer::Model`.
///
/// Infers model configuration from the GGML header. Returns an error if
/// architecture cannot be determined or the file is malformed.
///
/// # Arguments
/// * `path` - Path to the GGML model file
///
/// # Returns
/// A loaded `nnx_transformer::Model` or an error describing the issue.
///
/// # Example
/// ```ignore
/// use nnx_ggml::load_ggml;
/// let model = load_ggml("model.ggml")?;
/// ```
pub fn load_ggml(
    path: impl AsRef<Path>,
) -> std::result::Result<nnx_transformer::model::Model, String> {
    use nnx_transformer::block::BlockWeights;
    use nnx_transformer::config::{
        Architecture, BlockStyle, FFNType, ModelConfig, NormType, PosEncoding,
    };
    use nnx_transformer::model::{Model, ModelWeights};
    use nnx_transformer::weights::Matrix;

    let path = path.as_ref();

    // We need hints to open GGML files. Try to infer from filename or use defaults.
    let arch_hint = infer_arch_hint_from_filename(path);

    let ggml_file = GGMLFile::open(path, arch_hint.clone()).map_err(|e| format!("Failed to open GGML file: {}", e))?;
    let header = &ggml_file.header;

    // Safety check: architecture inference
    if header.n_layer == 0 || header.n_embd == 0 || header.n_head == 0 || header.n_vocab == 0 {
        return Err(format!(
            "Invalid GGML header: layers={}, embd={}, heads={}, vocab={}",
            header.n_layer, header.n_embd, header.n_head, header.n_vocab
        ));
    }

    let config = infer_model_config(&arch_hint, header)?;

    // Map GGML tensor names to ModelWeights
    let token_embedding = load_matrix_or_quantized(
        &ggml_file,
        "tok_embeddings.weight",
        config.vocab_size,
        config.hidden_dim,
    )?;

    let final_norm = load_vector_f32(&ggml_file, "norm.weight", config.hidden_dim)?;

    // Check if lm_head exists, otherwise tie to embeddings
    let lm_head = if ggml_file.tensor_info("output.weight").is_some() {
        load_matrix_or_quantized(
            &ggml_file,
            "output.weight",
            config.vocab_size,
            config.hidden_dim,
        )?
    } else {
        token_embedding.clone()
    };

    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let mut layers = Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        layers.push(BlockWeights {
            attn_norm: load_vector_f32(
                &ggml_file,
                &format!("layers.{layer_idx}.attention_norm.weight"),
                config.hidden_dim,
            )?,
            ffn_norm: load_vector_f32(
                &ggml_file,
                &format!("layers.{layer_idx}.ffn_norm.weight"),
                config.hidden_dim,
            )?,
            wq: load_matrix_or_quantized(
                &ggml_file,
                &format!("layers.{layer_idx}.attention.wq.weight"),
                q_dim,
                config.hidden_dim,
            )?,
            wk: load_matrix_or_quantized(
                &ggml_file,
                &format!("layers.{layer_idx}.attention.wk.weight"),
                kv_dim,
                config.hidden_dim,
            )?,
            wv: load_matrix_or_quantized(
                &ggml_file,
                &format!("layers.{layer_idx}.attention.wv.weight"),
                kv_dim,
                config.hidden_dim,
            )?,
            wo: load_matrix_or_quantized(
                &ggml_file,
                &format!("layers.{layer_idx}.attention.wo.weight"),
                config.hidden_dim,
                q_dim,
            )?,
            w_gate: load_matrix_or_quantized(
                &ggml_file,
                &format!("layers.{layer_idx}.feed_forward.w1.weight"),
                config.intermediate_dim,
                config.hidden_dim,
            )?,
            w_up: load_matrix_or_quantized(
                &ggml_file,
                &format!("layers.{layer_idx}.feed_forward.w3.weight"),
                config.intermediate_dim,
                config.hidden_dim,
            )?,
            w_down: load_matrix_or_quantized(
                &ggml_file,
                &format!("layers.{layer_idx}.feed_forward.w2.weight"),
                config.hidden_dim,
                config.intermediate_dim,
            )?,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            attn_norm_bias: None,
            ffn_norm_bias: None,
        });
    }

    Ok(Model::new(
        config,
        ModelWeights {
            token_embedding,
            position_embedding: None,
            layers,
            final_norm,
            final_norm_bias: None,
            lm_head,
        },
    ))
}

/// Infer architecture hint from filename or return minimal defaults.
fn infer_arch_hint_from_filename(path: &Path) -> GGMLArchHint {
    GGMLArchHint {
        architecture: "llama".to_string(),
        num_layers: 32, // Will be corrected by header
        hidden_dim: 4096,
        num_heads: 32,
        vocab_size: 32000,
    }
}

/// Infer ModelConfig from GGML header.
fn infer_model_config(
    arch_hint: &GGMLArchHint,
    header: &GGMLHeader,
) -> std::result::Result<nnx_transformer::config::ModelConfig, String> {
    use nnx_transformer::config::{
        Architecture, BlockStyle, FFNType, ModelConfig, NormType, PosEncoding,
    };

    let head_dim = header.n_embd as usize / header.n_head as usize;
    let intermediate_dim = (header.n_embd as usize * 4).next_power_of_two().min(header.n_embd as usize * 8);

    let architecture = match arch_hint.architecture.as_str() {
        "llama" | "llama2" | "llama3" => Architecture::Llama,
        "mistral" => Architecture::Mistral,
        "phi" => Architecture::Phi,
        "gpt2" => Architecture::GPT2,
        _ => Architecture::Llama, // Default fallback
    };

    Ok(ModelConfig {
        architecture: arch_hint.architecture.clone(),
        arch: architecture,
        num_layers: header.n_layer as usize,
        hidden_dim: header.n_embd as usize,
        num_heads: header.n_head as usize,
        num_kv_heads: header.n_head as usize, // Assume MHA unless specified
        head_dim,
        intermediate_dim,
        vocab_size: header.n_vocab as usize,
        max_context_length: DEFAULT_MAX_CONTEXT,
        rope_freq_base: DEFAULT_ROPE_FREQ_BASE,
        rms_norm_eps: DEFAULT_RMS_NORM_EPS,
        norm_type: NormType::RMSNorm,
        ffn_type: FFNType::SwiGLU,
        pos_encoding: PosEncoding::RoPE {
            freq_base: DEFAULT_ROPE_FREQ_BASE,
        },
        block_style: BlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
        embedding_scale: None,
        activation_quantization: nnx_transformer::config::ActivationQuantization::None,
    })
}

/// Load a vector as f32 from a GGML tensor.
fn load_vector_f32(
    ggml_file: &GGMLFile,
    name: &str,
    expected_len: usize,
) -> std::result::Result<Vec<f32>, String> {
    let view = ggml_file
        .tensor_view(name)
        .map_err(|e| format!("Missing tensor '{}': {}", name, e))?;

    if view.shape().numel() != expected_len {
        return Err(format!(
            "Tensor '{}' has {} elements, expected {}",
            name,
            view.shape().numel(),
            expected_len
        ));
    }

    if view.dtype() != GGMLType::F32 {
        return Err(format!(
            "Tensor '{}' has dtype {:?}, expected F32",
            name,
            view.dtype()
        ));
    }

    let bytes = view.as_bytes();
    let mut result = Vec::with_capacity(expected_len);
    for i in 0..expected_len {
        let offset = i * 4;
        let value = f32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| format!("Failed to read f32 at byte {}", offset))?,
        );
        result.push(value);
    }

    Ok(result)
}

/// Load a matrix from a GGML tensor, supporting both dense and quantized storage.
fn load_matrix_or_quantized(
    ggml_file: &GGMLFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> std::result::Result<nnx_transformer::weights::Matrix, String> {
    use nnx_transformer::weights::Matrix;

    let view = ggml_file
        .tensor_view(name)
        .map_err(|e| format!("Missing tensor '{}': {}", name, e))?;

    let expected_numel = rows * cols;
    if view.shape().numel() != expected_numel {
        return Err(format!(
            "Tensor '{}' has shape {:?}, expected {}x{} = {} elements",
            name,
            view.shape().dims(),
            rows,
            cols,
            expected_numel
        ));
    }

    match view.dtype() {
        GGMLType::F32 => {
            let data = load_vector_f32(ggml_file, name, expected_numel)?;
            Ok(Matrix::dense(data, rows, cols))
        }
        _ => {
            // For quantized types, store as Matrix::Quantized with raw bytes
            Matrix::quantized(view.as_bytes().to_vec(), view.dtype(), rows, cols)
        }
    }
}

/// Map GGMLType to a string representation for Matrix::Quantized.
fn map_ggml_type_to_string(dtype: GGMLType) -> String {
    match dtype {
        GGMLType::F32 => "f32".to_string(),
        GGMLType::F16 => "f16".to_string(),
        GGMLType::Q4_0 => "q4_0".to_string(),
        GGMLType::Q4_1 => "q4_1".to_string(),
        GGMLType::Q5_0 => "q5_0".to_string(),
        GGMLType::Q5_1 => "q5_1".to_string(),
        GGMLType::Q8_0 => "q8_0".to_string(),
        GGMLType::Q8_1 => "q8_1".to_string(),
        GGMLType::Q2K => "q2_k".to_string(),
        GGMLType::Q3K => "q3_k".to_string(),
        GGMLType::Q4K => "q4_k".to_string(),
        GGMLType::Q5K => "q5_k".to_string(),
        GGMLType::Q6K => "q6_k".to_string(),
        GGMLType::Q8K => "q8_k".to_string(),
        GGMLType::IQ2XXS => "iq2_xxs".to_string(),
        GGMLType::IQ2XS => "iq2_xs".to_string(),
        GGMLType::IQ3XXS => "iq3_xxs".to_string(),
        GGMLType::IQ1S => "iq1_s".to_string(),
        GGMLType::IQ4NL => "iq4_nl".to_string(),
        GGMLType::IQ3S => "iq3_s".to_string(),
        GGMLType::IQ2S => "iq2_s".to_string(),
        GGMLType::IQ4XS => "iq4_xs".to_string(),
        GGMLType::I8 => "i8".to_string(),
        GGMLType::I16 => "i16".to_string(),
        GGMLType::I32 => "i32".to_string(),
        GGMLType::I64 => "i64".to_string(),
        GGMLType::F64 => "f64".to_string(),
        GGMLType::IQ1M => "iq1_m".to_string(),
        GGMLType::BF16 => "bf16".to_string(),
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

    #[test]
    fn test_load_ggml_minimal() {
        use crate::load_ggml;

        // Build a more complete synthetic GGML file with minimal valid tensors
        let mut bytes = Vec::new();

        // Header
        bytes.extend_from_slice(&GGML_MAGIC_V2.to_le_bytes());
        bytes.extend_from_slice(&100u32.to_le_bytes()); // vocab_size
        bytes.extend_from_slice(&128u32.to_le_bytes()); // hidden_dim
        bytes.extend_from_slice(&8u32.to_le_bytes());   // num_heads
        bytes.extend_from_slice(&1u32.to_le_bytes());   // num_layers
        bytes.extend_from_slice(&0u32.to_le_bytes());   // f16: false

        // Helper to add a tensor
        let mut add_tensor = |name: &str, dims: &[u32], data_len: usize| {
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&(GGMLType::F32 as u32).to_le_bytes());

            for &dim in dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }

            bytes.extend_from_slice(name.as_bytes());

            let aligned_len = GGMLFile::align_up(bytes.len(), 32);
            bytes.resize(aligned_len, 0);

            for _ in 0..data_len {
                bytes.extend_from_slice(&0.1f32.to_le_bytes());
            }
        };

        // Add required tensors for a valid model
        add_tensor("tok_embeddings.weight", &[100, 128], 100 * 128);
        add_tensor("norm.weight", &[128], 128);
        add_tensor("output.weight", &[100, 128], 100 * 128);

        // Layer 0 tensors
        add_tensor("layers.0.attention_norm.weight", &[128], 128);
        add_tensor("layers.0.ffn_norm.weight", &[128], 128);
        add_tensor("layers.0.attention.wq.weight", &[128, 128], 128 * 128);
        add_tensor("layers.0.attention.wk.weight", &[128, 128], 128 * 128);
        add_tensor("layers.0.attention.wv.weight", &[128, 128], 128 * 128);
        add_tensor("layers.0.attention.wo.weight", &[128, 128], 128 * 128);
        add_tensor("layers.0.feed_forward.w1.weight", &[512, 128], 512 * 128);
        add_tensor("layers.0.feed_forward.w2.weight", &[128, 512], 128 * 512);
        add_tensor("layers.0.feed_forward.w3.weight", &[512, 128], 512 * 128);

        let path = temp_path("load_ggml_test.bin");
        write_test_file(&path, &bytes);

        let model = load_ggml(&path).expect("Failed to load GGML model");

        // Verify config was inferred correctly
        assert_eq!(model.config.vocab_size, 100);
        assert_eq!(model.config.hidden_dim, 128);
        assert_eq!(model.config.num_layers, 1);
        assert_eq!(model.config.num_heads, 8);
        assert_eq!(model.config.head_dim, 128 / 8);

        // Verify architecture
        assert!(matches!(
            model.config.arch,
            nnx_transformer::config::Architecture::Llama
        ));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_ggml_invalid_header() {
        use crate::load_ggml;

        // Build GGML file with invalid (zero) dimensions
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGML_MAGIC_V2.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // vocab_size = 0 (invalid)
        bytes.extend_from_slice(&0u32.to_le_bytes()); // hidden_dim = 0 (invalid)
        bytes.extend_from_slice(&0u32.to_le_bytes()); // num_heads = 0 (invalid)
        bytes.extend_from_slice(&0u32.to_le_bytes()); // num_layers = 0 (invalid)
        bytes.extend_from_slice(&0u32.to_le_bytes()); // f16

        let path = temp_path("invalid_header.bin");
        write_test_file(&path, &bytes);

        let _result = load_ggml(&path);
        assert!(_result.is_err(), "Should fail with invalid header");

        std::fs::remove_file(path).ok();
    }
}
