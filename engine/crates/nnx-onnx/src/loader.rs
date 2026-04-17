//! ONNX parsing and narrow Llama-family loading for NNX.

use crate::error::{OnnxError, Result};
use crate::proto;
use half::{bf16, f16};
use nnx_transformer::block::BlockWeights;
use nnx_transformer::config::{
    Architecture, BlockStyle, FFNType, ModelConfig, NormType, PosEncoding, find_profile_by_hf,
};
use nnx_transformer::model::{Model, ModelWeights};
use nnx_transformer::weights::Matrix;
use prost::Message;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const DEFAULT_MAX_CONTEXT_LENGTH: usize = 4096;
const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;

/// Parses ONNX protobuf payloads into a small model-inspection object.
#[derive(Debug, Default, Clone, Copy)]
pub struct OnnxParser;

impl OnnxParser {
    /// Parse an ONNX model from bytes.
    pub fn parse_bytes(bytes: &[u8]) -> Result<ParsedModel> {
        let model = proto::ModelProto::decode(bytes)?;
        ParsedModel::from_proto(model)
    }

    /// Parse an ONNX model from a file path.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<ParsedModel> {
        let bytes = fs::read(path)?;
        Self::parse_bytes(&bytes)
    }
}

/// Dense tensor element types recognized by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorElementType {
    Float32,
    Float16,
    BFloat16,
    Float64,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Bool,
}

impl TensorElementType {
    fn from_onnx(value: i32) -> Result<Self> {
        use proto::tensor_proto::DataType;

        match DataType::try_from(value).ok() {
            Some(DataType::Float) => Ok(Self::Float32),
            Some(DataType::Float16) => Ok(Self::Float16),
            Some(DataType::Bfloat16) => Ok(Self::BFloat16),
            Some(DataType::Double) => Ok(Self::Float64),
            Some(DataType::Int8) => Ok(Self::Int8),
            Some(DataType::Uint8) => Ok(Self::Uint8),
            Some(DataType::Int16) => Ok(Self::Int16),
            Some(DataType::Uint16) => Ok(Self::Uint16),
            Some(DataType::Int32) => Ok(Self::Int32),
            Some(DataType::Uint32) => Ok(Self::Uint32),
            Some(DataType::Int64) => Ok(Self::Int64),
            Some(DataType::Uint64) => Ok(Self::Uint64),
            Some(DataType::Bool) => Ok(Self::Bool),
            Some(other) => Err(OnnxError::UnsupportedDType(other.as_str_name().to_string())),
            None => Err(OnnxError::UnsupportedDType(format!(
                "unknown dtype id {}",
                value
            ))),
        }
    }
}

/// How a tensor payload is stored in the ONNX protobuf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDataEncoding {
    RawData,
    TypedFields,
    ExternalData,
}

/// Dense tensor metadata extracted from an ONNX initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    pub name: String,
    pub data_type: TensorElementType,
    pub shape: Vec<usize>,
}

impl TensorMetadata {
    /// Total number of elements implied by the shape.
    pub fn element_count(&self) -> usize {
        element_count(&self.shape)
    }
}

/// A single ONNX initializer with metadata and lazy value decoding.
#[derive(Debug, Clone)]
pub struct Initializer {
    metadata: TensorMetadata,
    encoding: TensorDataEncoding,
    tensor: proto::TensorProto,
}

impl Initializer {
    /// Returns the tensor metadata.
    pub fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    /// Returns the ONNX storage encoding used by this initializer.
    pub fn encoding(&self) -> TensorDataEncoding {
        self.encoding
    }

    /// Decode this dense initializer into `f32` values when the dtype is supported.
    pub fn decode_f32(&self) -> Result<Vec<f32>> {
        decode_tensor_to_f32(&self.tensor, &self.metadata)
    }
}

/// Parsed ONNX model view used for inspection and narrow model loading.
#[derive(Debug, Clone)]
pub struct ParsedModel {
    ir_version: i64,
    producer_name: Option<String>,
    graph_name: String,
    metadata: BTreeMap<String, String>,
    initializers: BTreeMap<String, Initializer>,
}

impl ParsedModel {
    fn from_proto(model: proto::ModelProto) -> Result<Self> {
        let graph = model.graph.ok_or(OnnxError::MissingGraph)?;
        let metadata = model
            .metadata_props
            .into_iter()
            .map(|entry| (entry.key, entry.value))
            .collect::<BTreeMap<_, _>>();

        let mut initializers = BTreeMap::new();
        for tensor in graph.initializer {
            let initializer = Initializer::from_tensor(tensor)?;
            let name = initializer.metadata.name.clone();
            initializers.insert(name, initializer);
        }

        Ok(Self {
            ir_version: model.ir_version,
            producer_name: (!model.producer_name.is_empty()).then_some(model.producer_name),
            graph_name: graph.name,
            metadata,
            initializers,
        })
    }

    /// ONNX IR version stored in the model.
    pub fn ir_version(&self) -> i64 {
        self.ir_version
    }

    /// Producer name if present.
    pub fn producer_name(&self) -> Option<&str> {
        self.producer_name.as_deref()
    }

    /// Graph name if present.
    pub fn graph_name(&self) -> &str {
        &self.graph_name
    }

    /// String metadata value lookup from `ModelProto.metadata_props`.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Number of parsed initializers.
    pub fn tensor_count(&self) -> usize {
        self.initializers.len()
    }

    /// All parsed initializer names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.initializers.keys().map(String::as_str)
    }

    /// Lookup a parsed initializer by name.
    pub fn initializer(&self, name: &str) -> Option<&Initializer> {
        self.initializers.get(name)
    }

    /// Returns the shape for a named initializer.
    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.initializer(name)
            .map(|value| value.metadata.shape.as_slice())
    }

    /// Returns all parsed initializers.
    pub fn initializers(&self) -> &BTreeMap<String, Initializer> {
        &self.initializers
    }

    /// Build a dense Llama-family `nnx-transformer` model from ONNX initializers.
    pub fn load_dense_llama_model(&self) -> Result<Model> {
        let config = infer_llama_config(self)?;
        let token_embedding = self.decode_matrix(
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_dim,
        )?;
        let final_norm = self.decode_vector("model.norm.weight", config.hidden_dim)?;
        let lm_head = if self.initializers.contains_key("lm_head.weight") {
            self.decode_matrix("lm_head.weight", config.vocab_size, config.hidden_dim)?
        } else {
            token_embedding.clone()
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        for layer_idx in 0..config.num_layers {
            let prefix = format!("model.layers.{layer_idx}");
            layers.push(BlockWeights {
                attn_norm: self.decode_vector(
                    &format!("{prefix}.input_layernorm.weight"),
                    config.hidden_dim,
                )?,
                ffn_norm: self.decode_vector(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    config.hidden_dim,
                )?,
                wq: self.decode_matrix(
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    q_dim,
                    config.hidden_dim,
                )?,
                wk: self.decode_matrix(
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.hidden_dim,
                )?,
                wv: self.decode_matrix(
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    kv_dim,
                    config.hidden_dim,
                )?,
                wo: self.decode_matrix(
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    config.hidden_dim,
                    q_dim,
                )?,
                w_gate: self.decode_matrix(
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    config.intermediate_dim,
                    config.hidden_dim,
                )?,
                w_up: self.decode_matrix(
                    &format!("{prefix}.mlp.up_proj.weight"),
                    config.intermediate_dim,
                    config.hidden_dim,
                )?,
                w_down: self.decode_matrix(
                    &format!("{prefix}.mlp.down_proj.weight"),
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

    fn decode_vector(&self, name: &str, expected_len: usize) -> Result<Vec<f32>> {
        let initializer = self
            .initializer(name)
            .ok_or_else(|| OnnxError::MissingInitializer(name.to_string()))?;
        let data = initializer.decode_f32()?;
        if data.len() != expected_len {
            return Err(OnnxError::InvalidTensorData {
                name: name.to_string(),
                reason: format!("expected {} elements, found {}", expected_len, data.len()),
            });
        }
        Ok(data)
    }

    fn decode_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        let initializer = self
            .initializer(name)
            .ok_or_else(|| OnnxError::MissingInitializer(name.to_string()))?;
        validate_shape(name, initializer.metadata(), &[rows, cols])?;
        Ok(Matrix::dense(initializer.decode_f32()?, rows, cols))
    }
}

impl Initializer {
    fn from_tensor(tensor: proto::TensorProto) -> Result<Self> {
        let name = if tensor.name.is_empty() {
            return Err(OnnxError::UnnamedInitializer);
        } else {
            tensor.name.clone()
        };

        let data_type = TensorElementType::from_onnx(tensor.data_type)?;
        let shape = tensor
            .dims
            .iter()
            .map(|&dim| {
                usize::try_from(dim).map_err(|_| OnnxError::InvalidShape {
                    name: name.clone(),
                    shape: tensor.dims.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let encoding = if tensor.data_location == proto::tensor_proto::DataLocation::External as i32
        {
            TensorDataEncoding::ExternalData
        } else if !tensor.raw_data.is_empty() {
            TensorDataEncoding::RawData
        } else {
            TensorDataEncoding::TypedFields
        };

        Ok(Self {
            metadata: TensorMetadata {
                name,
                data_type,
                shape,
            },
            encoding,
            tensor,
        })
    }
}

fn decode_tensor_to_f32(
    tensor: &proto::TensorProto,
    metadata: &TensorMetadata,
) -> Result<Vec<f32>> {
    if tensor.data_location == proto::tensor_proto::DataLocation::External as i32 {
        return Err(OnnxError::ExternalDataUnsupported(metadata.name.clone()));
    }

    let expected_len = metadata.element_count();
    let values = if !tensor.raw_data.is_empty() {
        decode_raw_data(
            &tensor.raw_data,
            metadata.data_type,
            expected_len,
            &metadata.name,
        )?
    } else {
        decode_typed_data(tensor, metadata.data_type, expected_len, &metadata.name)?
    };

    if values.len() != expected_len {
        return Err(OnnxError::InvalidTensorData {
            name: metadata.name.clone(),
            reason: format!(
                "expected {} elements, decoded {}",
                expected_len,
                values.len()
            ),
        });
    }

    Ok(values)
}

fn decode_raw_data(
    raw_data: &[u8],
    data_type: TensorElementType,
    expected_len: usize,
    name: &str,
) -> Result<Vec<f32>> {
    let values = match data_type {
        TensorElementType::Float32 => decode_chunks(raw_data, 4, name, f32::from_le_bytes),
        TensorElementType::Float64 => decode_chunks(raw_data, 8, name, f64::from_le_bytes)
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
        TensorElementType::Float16 => {
            decode_chunks(raw_data, 2, name, u16::from_le_bytes).map(|values| {
                values
                    .into_iter()
                    .map(|value| f16::from_bits(value).to_f32())
                    .collect()
            })
        }
        TensorElementType::BFloat16 => {
            decode_chunks(raw_data, 2, name, u16::from_le_bytes).map(|values| {
                values
                    .into_iter()
                    .map(|value| bf16::from_bits(value).to_f32())
                    .collect()
            })
        }
        TensorElementType::Int8 => Ok(raw_data.iter().map(|value| (*value as i8) as f32).collect()),
        TensorElementType::Uint8 => Ok(raw_data.iter().map(|&value| value as f32).collect()),
        TensorElementType::Int16 => decode_chunks(raw_data, 2, name, i16::from_le_bytes)
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
        TensorElementType::Uint16 => decode_chunks(raw_data, 2, name, u16::from_le_bytes)
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
        TensorElementType::Int32 => decode_chunks(raw_data, 4, name, i32::from_le_bytes)
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
        TensorElementType::Uint32 => decode_chunks(raw_data, 4, name, u32::from_le_bytes)
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
        TensorElementType::Int64 => decode_chunks(raw_data, 8, name, i64::from_le_bytes)
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
        TensorElementType::Uint64 => decode_chunks(raw_data, 8, name, u64::from_le_bytes)
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
        TensorElementType::Bool => Ok(raw_data
            .iter()
            .map(|&value| if value == 0 { 0.0 } else { 1.0 })
            .collect()),
    }?;

    if values.len() != expected_len {
        return Err(OnnxError::InvalidTensorData {
            name: name.to_string(),
            reason: format!(
                "expected {} elements, found {} in raw_data",
                expected_len,
                values.len()
            ),
        });
    }

    Ok(values)
}

fn decode_typed_data(
    tensor: &proto::TensorProto,
    data_type: TensorElementType,
    expected_len: usize,
    name: &str,
) -> Result<Vec<f32>> {
    let values = match data_type {
        TensorElementType::Float32 => tensor.float_data.iter().copied().collect(),
        TensorElementType::Float64 => tensor
            .double_data
            .iter()
            .map(|&value| value as f32)
            .collect(),
        TensorElementType::Float16 => tensor
            .int32_data
            .iter()
            .map(|&value| {
                let bits = u16::try_from(value).map_err(|_| OnnxError::InvalidTensorData {
                    name: name.to_string(),
                    reason: format!("FLOAT16 int32_data value {} does not fit in u16", value),
                })?;
                Ok(f16::from_bits(bits).to_f32())
            })
            .collect::<Result<Vec<_>>>()?,
        TensorElementType::BFloat16 => tensor
            .int32_data
            .iter()
            .map(|&value| {
                let bits = u16::try_from(value).map_err(|_| OnnxError::InvalidTensorData {
                    name: name.to_string(),
                    reason: format!("BFLOAT16 int32_data value {} does not fit in u16", value),
                })?;
                Ok(bf16::from_bits(bits).to_f32())
            })
            .collect::<Result<Vec<_>>>()?,
        TensorElementType::Int8
        | TensorElementType::Uint8
        | TensorElementType::Int16
        | TensorElementType::Uint16
        | TensorElementType::Int32 => tensor
            .int32_data
            .iter()
            .map(|&value| value as f32)
            .collect(),
        TensorElementType::Bool => tensor
            .int32_data
            .iter()
            .map(|&value| if value == 0 { 0.0 } else { 1.0 })
            .collect(),
        TensorElementType::Uint32 | TensorElementType::Uint64 => tensor
            .uint64_data
            .iter()
            .map(|&value| value as f32)
            .collect(),
        TensorElementType::Int64 => tensor
            .int64_data
            .iter()
            .map(|&value| value as f32)
            .collect(),
    };

    if values.len() != expected_len {
        return Err(OnnxError::InvalidTensorData {
            name: name.to_string(),
            reason: format!(
                "expected {} elements, found {} in typed fields",
                expected_len,
                values.len()
            ),
        });
    }

    Ok(values)
}

fn decode_chunks<const N: usize, T>(
    bytes: &[u8],
    width: usize,
    name: &str,
    convert: fn([u8; N]) -> T,
) -> Result<Vec<T>> {
    if bytes.len() % width != 0 {
        return Err(OnnxError::InvalidTensorData {
            name: name.to_string(),
            reason: format!(
                "raw_data byte length {} is not divisible by {}",
                bytes.len(),
                width
            ),
        });
    }

    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| {
            let mut array = [0u8; N];
            array.copy_from_slice(chunk);
            convert(array)
        })
        .collect())
}

fn validate_shape(name: &str, metadata: &TensorMetadata, expected: &[usize]) -> Result<()> {
    if metadata.shape.as_slice() != expected {
        return Err(OnnxError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: metadata.shape.clone(),
        });
    }
    Ok(())
}

fn infer_llama_config(model: &ParsedModel) -> Result<ModelConfig> {
    let embed = model
        .initializer("model.embed_tokens.weight")
        .ok_or_else(|| OnnxError::MissingInitializer("model.embed_tokens.weight".to_string()))?;
    ensure_rank(embed.metadata(), 2)?;

    let hidden_dim = model
        .metadata("hidden_size")
        .and_then(parse_usize)
        .unwrap_or(embed.metadata().shape[1]);
    let vocab_size = model
        .metadata("vocab_size")
        .and_then(parse_usize)
        .unwrap_or(embed.metadata().shape[0]);
    let num_layers = model
        .metadata("num_hidden_layers")
        .and_then(parse_usize)
        .unwrap_or_else(|| infer_num_layers(model));

    if num_layers == 0 {
        return Err(OnnxError::UnsupportedModel(
            "could not infer any Llama layers from ONNX initializer names".to_string(),
        ));
    }

    let q_meta = model
        .initializer("model.layers.0.self_attn.q_proj.weight")
        .ok_or_else(|| {
            OnnxError::MissingInitializer("model.layers.0.self_attn.q_proj.weight".to_string())
        })?
        .metadata();
    let k_meta = model
        .initializer("model.layers.0.self_attn.k_proj.weight")
        .ok_or_else(|| {
            OnnxError::MissingInitializer("model.layers.0.self_attn.k_proj.weight".to_string())
        })?
        .metadata();
    let gate_meta = model
        .initializer("model.layers.0.mlp.gate_proj.weight")
        .ok_or_else(|| {
            OnnxError::MissingInitializer("model.layers.0.mlp.gate_proj.weight".to_string())
        })?
        .metadata();

    ensure_rank(q_meta, 2)?;
    ensure_rank(k_meta, 2)?;
    ensure_rank(gate_meta, 2)?;

    let q_rows = q_meta.shape[0];
    let k_rows = k_meta.shape[0];

    let explicit_head_dim = model.metadata("head_dim").and_then(parse_usize);
    let explicit_num_heads = model.metadata("num_attention_heads").and_then(parse_usize);
    let explicit_num_kv_heads = model.metadata("num_key_value_heads").and_then(parse_usize);

    let (num_heads, num_kv_heads, head_dim) = infer_attention_geometry(
        hidden_dim,
        q_rows,
        k_rows,
        explicit_head_dim,
        explicit_num_heads,
        explicit_num_kv_heads,
    )?;

    let intermediate_dim = model
        .metadata("intermediate_size")
        .and_then(parse_usize)
        .unwrap_or(gate_meta.shape[0]);
    let max_context_length = model
        .metadata("max_position_embeddings")
        .and_then(parse_usize)
        .unwrap_or(DEFAULT_MAX_CONTEXT_LENGTH);
    let rope_freq_base = model
        .metadata("rope_theta")
        .and_then(parse_f32)
        .or_else(|| model.metadata("rope_freq_base").and_then(parse_f32))
        .unwrap_or(DEFAULT_ROPE_FREQ_BASE);
    let rms_norm_eps = model
        .metadata("rms_norm_eps")
        .and_then(parse_f32)
        .unwrap_or(DEFAULT_RMS_NORM_EPS);

    let architecture_name = model
        .metadata("model_type")
        .or_else(|| model.metadata("architecture"))
        .unwrap_or("llama");
    let profile = find_profile_by_hf(architecture_name).ok_or_else(|| {
        OnnxError::UnsupportedModel(format!(
            "unsupported HuggingFace architecture '{}' for ONNX dense loader",
            architecture_name
        ))
    })?;

    let arch = match profile.name {
        "llama" => Architecture::Llama,
        "mistral" => Architecture::Mistral,
        "codellama" => Architecture::CodeLlama,
        other => {
            return Err(OnnxError::UnsupportedModel(format!(
                "architecture '{}' is not in the supported Llama-family subset",
                other
            )));
        }
    };

    Ok(ModelConfig {
        architecture: profile.name.to_string(),
        arch,
        num_layers,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_dim,
        vocab_size,
        max_context_length,
        rope_freq_base,
        rms_norm_eps,
        norm_type: NormType::RMSNorm,
        ffn_type: FFNType::SwiGLU,
        pos_encoding: PosEncoding::RoPE {
            freq_base: rope_freq_base,
        },
        block_style: BlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
        embedding_scale: None,
    })
}

fn infer_num_layers(model: &ParsedModel) -> usize {
    let mut seen = BTreeSet::new();
    for name in model.tensor_names() {
        if let Some(index) = parse_layer_index(name) {
            seen.insert(index);
        }
    }

    seen.last().copied().map(|index| index + 1).unwrap_or(0)
}

fn parse_layer_index(name: &str) -> Option<usize> {
    let suffix = name.strip_prefix("model.layers.")?;
    suffix.split('.').next()?.parse().ok()
}

fn infer_attention_geometry(
    hidden_dim: usize,
    q_rows: usize,
    k_rows: usize,
    explicit_head_dim: Option<usize>,
    explicit_num_heads: Option<usize>,
    explicit_num_kv_heads: Option<usize>,
) -> Result<(usize, usize, usize)> {
    if let Some(head_dim) = explicit_head_dim {
        if head_dim == 0 || q_rows % head_dim != 0 || k_rows % head_dim != 0 {
            return Err(OnnxError::UnsupportedModel(format!(
                "invalid head_dim {} for q_rows={} and k_rows={}",
                head_dim, q_rows, k_rows
            )));
        }
        let num_heads = explicit_num_heads.unwrap_or(q_rows / head_dim);
        let num_kv_heads = explicit_num_kv_heads.unwrap_or(k_rows / head_dim);
        return Ok((num_heads, num_kv_heads, head_dim));
    }

    if let Some(num_heads) = explicit_num_heads {
        if num_heads == 0 || q_rows % num_heads != 0 {
            return Err(OnnxError::UnsupportedModel(format!(
                "invalid num_attention_heads {} for q_rows={}",
                num_heads, q_rows
            )));
        }
        let head_dim = q_rows / num_heads;
        if head_dim == 0 || k_rows % head_dim != 0 {
            return Err(OnnxError::UnsupportedModel(format!(
                "inferred head_dim {} is incompatible with k_rows={}",
                head_dim, k_rows
            )));
        }
        let num_kv_heads = explicit_num_kv_heads.unwrap_or(k_rows / head_dim);
        return Ok((num_heads, num_kv_heads, head_dim));
    }

    let preferred_head_dims = [
        256usize, 192, 160, 128, 96, 80, 64, 48, 40, 32, 24, 16, 8, 4, 2, 1,
    ];
    let head_dim = preferred_head_dims
        .into_iter()
        .find(|candidate| {
            *candidate <= hidden_dim
                && *candidate > 0
                && q_rows % *candidate == 0
                && k_rows % *candidate == 0
        })
        .ok_or_else(|| {
            OnnxError::UnsupportedModel(format!(
                "could not infer head geometry from hidden_dim={}, q_rows={}, k_rows={}",
                hidden_dim, q_rows, k_rows
            ))
        })?;

    Ok((q_rows / head_dim, k_rows / head_dim, head_dim))
}

fn ensure_rank(metadata: &TensorMetadata, expected_rank: usize) -> Result<()> {
    if metadata.shape.len() != expected_rank {
        return Err(OnnxError::ShapeMismatch {
            name: metadata.name.clone(),
            expected: vec![0; expected_rank],
            actual: metadata.shape.clone(),
        });
    }
    Ok(())
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>().max(1)
}

fn parse_usize(value: &str) -> Option<usize> {
    value.parse().ok()
}

fn parse_f32(value: &str) -> Option<f32> {
    value.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{GraphProto, ModelProto, StringStringEntryProto, TensorProto};

    fn f32_raw_tensor(name: &str, shape: &[i64], values: &[f32]) -> TensorProto {
        TensorProto {
            dims: shape.to_vec(),
            data_type: proto::tensor_proto::DataType::Float as i32,
            name: name.to_string(),
            raw_data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            ..Default::default()
        }
    }

    fn f32_typed_tensor(name: &str, shape: &[i64], values: &[f32]) -> TensorProto {
        TensorProto {
            dims: shape.to_vec(),
            data_type: proto::tensor_proto::DataType::Float as i32,
            name: name.to_string(),
            float_data: values.to_vec(),
            ..Default::default()
        }
    }

    fn metadata_entry(key: &str, value: &str) -> StringStringEntryProto {
        StringStringEntryProto {
            key: key.to_string(),
            value: value.to_string(),
        }
    }

    fn synthetic_model(initializers: Vec<TensorProto>) -> Vec<u8> {
        ModelProto {
            ir_version: 8,
            producer_name: "nnx-onnx-test".into(),
            graph: Some(GraphProto {
                name: "synthetic".into(),
                initializer: initializers,
                ..Default::default()
            }),
            ..Default::default()
        }
        .encode_to_vec()
    }

    #[test]
    fn parser_decodes_real_protobuf_payload() {
        let bytes = synthetic_model(vec![f32_typed_tensor(
            "weight",
            &[2, 2],
            &[1.0, 2.0, 3.0, 4.0],
        )]);
        let parsed = OnnxParser::parse_bytes(&bytes).expect("parser should decode prost payload");

        assert_eq!(parsed.ir_version(), 8);
        assert_eq!(parsed.producer_name(), Some("nnx-onnx-test"));
        assert_eq!(parsed.tensor_count(), 1);
        assert_eq!(parsed.tensor_shape("weight"), Some(&[2, 2][..]));
        assert_eq!(
            parsed.initializer("weight").unwrap().decode_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn initializer_metadata_and_decoding_are_available() {
        let bytes = synthetic_model(vec![
            f32_raw_tensor("raw.weight", &[2, 2], &[0.5, 1.5, 2.5, 3.5]),
            TensorProto {
                dims: vec![3],
                data_type: proto::tensor_proto::DataType::Int32 as i32,
                name: "typed.bias".into(),
                int32_data: vec![1, 2, 3],
                ..Default::default()
            },
        ]);

        let parsed = OnnxParser::parse_bytes(&bytes).unwrap();
        let raw = parsed.initializer("raw.weight").unwrap();
        let typed = parsed.initializer("typed.bias").unwrap();

        assert_eq!(raw.metadata().data_type, TensorElementType::Float32);
        assert_eq!(raw.metadata().shape, vec![2, 2]);
        assert_eq!(raw.encoding(), TensorDataEncoding::RawData);
        assert_eq!(raw.decode_f32().unwrap(), vec![0.5, 1.5, 2.5, 3.5]);

        assert_eq!(typed.metadata().shape, vec![3]);
        assert_eq!(typed.encoding(), TensorDataEncoding::TypedFields);
        assert_eq!(typed.decode_f32().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn llama_like_initializers_build_dense_model() {
        let hidden = 8usize;
        let vocab = 16usize;
        let intermediate = 12usize;
        let num_layers = 1usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;

        let mut initializers = vec![
            f32_raw_tensor(
                "model.embed_tokens.weight",
                &[vocab as i64, hidden as i64],
                &vec![0.01; vocab * hidden],
            ),
            f32_typed_tensor("model.norm.weight", &[hidden as i64], &vec![1.0; hidden]),
            f32_raw_tensor(
                "lm_head.weight",
                &[vocab as i64, hidden as i64],
                &vec![0.02; vocab * hidden],
            ),
            f32_typed_tensor(
                "model.layers.0.input_layernorm.weight",
                &[hidden as i64],
                &vec![1.0; hidden],
            ),
            f32_typed_tensor(
                "model.layers.0.post_attention_layernorm.weight",
                &[hidden as i64],
                &vec![1.0; hidden],
            ),
            f32_raw_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                &[(num_heads * head_dim) as i64, hidden as i64],
                &vec![0.1; num_heads * head_dim * hidden],
            ),
            f32_raw_tensor(
                "model.layers.0.self_attn.k_proj.weight",
                &[(num_kv_heads * head_dim) as i64, hidden as i64],
                &vec![0.1; num_kv_heads * head_dim * hidden],
            ),
            f32_raw_tensor(
                "model.layers.0.self_attn.v_proj.weight",
                &[(num_kv_heads * head_dim) as i64, hidden as i64],
                &vec![0.1; num_kv_heads * head_dim * hidden],
            ),
            f32_raw_tensor(
                "model.layers.0.self_attn.o_proj.weight",
                &[hidden as i64, (num_heads * head_dim) as i64],
                &vec![0.1; hidden * num_heads * head_dim],
            ),
            f32_raw_tensor(
                "model.layers.0.mlp.gate_proj.weight",
                &[intermediate as i64, hidden as i64],
                &vec![0.1; intermediate * hidden],
            ),
            f32_raw_tensor(
                "model.layers.0.mlp.up_proj.weight",
                &[intermediate as i64, hidden as i64],
                &vec![0.1; intermediate * hidden],
            ),
            f32_raw_tensor(
                "model.layers.0.mlp.down_proj.weight",
                &[hidden as i64, intermediate as i64],
                &vec![0.1; hidden * intermediate],
            ),
        ];

        let bytes = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                name: "llama".into(),
                initializer: std::mem::take(&mut initializers),
                ..Default::default()
            }),
            metadata_props: vec![
                metadata_entry("model_type", "LlamaForCausalLM"),
                metadata_entry("hidden_size", &hidden.to_string()),
                metadata_entry("num_hidden_layers", &num_layers.to_string()),
                metadata_entry("num_attention_heads", &num_heads.to_string()),
                metadata_entry("num_key_value_heads", &num_kv_heads.to_string()),
                metadata_entry("head_dim", &head_dim.to_string()),
                metadata_entry("intermediate_size", &intermediate.to_string()),
                metadata_entry("vocab_size", &vocab.to_string()),
                metadata_entry("max_position_embeddings", "32"),
                metadata_entry("rms_norm_eps", "1e-5"),
            ],
            ..Default::default()
        }
        .encode_to_vec();

        let parsed = OnnxParser::parse_bytes(&bytes).unwrap();
        let model = parsed.load_dense_llama_model().unwrap();

        assert_eq!(model.config.architecture, "llama");
        assert_eq!(model.config.arch, Architecture::Llama);
        assert_eq!(model.config.num_layers, 1);
        assert_eq!(model.config.hidden_dim, hidden);
        assert_eq!(model.config.num_heads, num_heads);
        assert_eq!(model.config.num_kv_heads, num_kv_heads);
        assert_eq!(model.config.head_dim, head_dim);

        match &model.weights.layers[0].wq {
            Matrix::Dense { rows, cols, .. } => {
                assert_eq!((*rows, *cols), (num_heads * head_dim, hidden));
            }
            other => panic!("expected dense q projection, got {other:?}"),
        }

        match &model.weights.layers[0].w_down {
            Matrix::Dense { rows, cols, .. } => {
                assert_eq!((*rows, *cols), (hidden, intermediate));
            }
            other => panic!("expected dense down projection, got {other:?}"),
        }
    }
}
