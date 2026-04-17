use clap::{Parser, ValueEnum};
use nnx_gguf::{GGUFMetadata, GGUFValue, GGUFWriterTensor, write_gguf_file};
use nnx_quant::GGMLType;
use nnx_quant::encode::quantize_matrix;
use nnx_safetensors::SafeTensorsFile;
use nnx_transformer::config::{FFNType, PosEncodingKind, find_profile_by_hf};
use nnx_transformer::loader::load_gguf;
use nnx_transformer::weight_names::{WeightNameMap, load_shard_index, map_hf_architecture, split_fused_qkv_bias, split_fused_qkv_weight};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliDType {
    Q4_0,
    Q8_0,
}

impl From<CliDType> for GGMLType {
    fn from(value: CliDType) -> Self {
        match value {
            CliDType::Q4_0 => GGMLType::Q4_0,
            CliDType::Q8_0 => GGMLType::Q8_0,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "nnx-quantize-model")]
#[command(about = "Convert a supported SafeTensors checkpoint into a quantized GGUF model")]
struct Args {
    #[arg(long, help = "Path to a .safetensors file or a model directory with shards/index")]
    model: PathBuf,
    #[arg(long, help = "Optional path to config.json; defaults next to the model")]
    config: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    dtype: CliDType,
    #[arg(long, help = "Optional model name to store in GGUF metadata")]
    name: Option<String>,
    #[arg(long, help = "Load the produced GGUF with nnx-transformer as a verification step")]
    verify: bool,
}

#[derive(Debug, Clone)]
struct ConversionConfig {
    gguf_architecture: String,
    architecture_name: String,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
    vocab_size: usize,
    max_context_length: usize,
    rope_freq_base: f32,
    rms_norm_eps: f32,
    pos_encoding_kind: PosEncodingKind,
    ffn_type: FFNType,
    has_qkv_bias: bool,
    has_output_bias: bool,
}

struct SourceCheckpoint {
    single: Option<SafeTensorsFile>,
    shard_index: Option<nnx_transformer::weight_names::ShardIndex>,
    shard_cache: HashMap<PathBuf, SafeTensorsFile>,
}

impl SourceCheckpoint {
    fn open(model_path: &Path) -> Result<Self, String> {
        if let Some(shard_index) = load_shard_index(model_path)? {
            let mut shard_cache = HashMap::new();
            for path in &shard_index.shard_files {
                let file = SafeTensorsFile::open(path)
                    .map_err(|error| format!("failed to open shard {}: {}", path.display(), error))?;
                shard_cache.insert(path.clone(), file);
            }
            return Ok(Self {
                single: None,
                shard_index: Some(shard_index),
                shard_cache,
            });
        }

        let single = SafeTensorsFile::open(model_path)
            .map_err(|error| format!("failed to open {}: {}", model_path.display(), error))?;
        Ok(Self {
            single: Some(single),
            shard_index: None,
            shard_cache: HashMap::new(),
        })
    }

    fn file_for(&self, name: &str) -> Option<&SafeTensorsFile> {
        if let Some(single) = &self.single {
            return single.tensor_info(name).map(|_| single);
        }

        let shard_index = self.shard_index.as_ref()?;
        let path = shard_index.tensor_to_file.get(name)?;
        self.shard_cache.get(path)
    }

    fn tensor_exists(&self, name: &str) -> bool {
        self.file_for(name).and_then(|file| file.tensor_info(name)).is_some()
    }

    fn load_tensor_f32(&self, name: &str, expected_numel: usize) -> Result<Vec<f32>, String> {
        let file = self
            .file_for(name)
            .ok_or_else(|| format!("tensor not found: {}", name))?;
        let view = file
            .tensor_view(name)
            .map_err(|error| format!("failed to load tensor {}: {}", name, error))?;
        if view.shape().numel() != expected_numel {
            return Err(format!(
                "tensor {} has {} elements, expected {}",
                name,
                view.shape().numel(),
                expected_numel
            ));
        }

        let bytes = view.as_bytes();
        Ok(match view.dtype() {
            nnx_core::dtype::DType::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            nnx_core::dtype::DType::F16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            nnx_core::dtype::DType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                    f32::from_bits((bits as u32) << 16)
                })
                .collect(),
            other => return Err(format!("unsupported dtype {:?} for tensor {}", other, name)),
        })
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let output_dtype: GGMLType = args.dtype.into();
    convert_safetensors_to_gguf(
        &args.model,
        args.config.as_deref(),
        &args.output,
        output_dtype,
        args.name.as_deref(),
        args.verify,
    )
}

fn convert_safetensors_to_gguf(
    model_path: &Path,
    config_path: Option<&Path>,
    output_path: &Path,
    output_dtype: GGMLType,
    model_name: Option<&str>,
    verify: bool,
) -> Result<(), String> {
    let config_path = resolve_config_path(model_path, config_path)?;
    let config = load_conversion_config(&config_path)?;
    let arch = map_hf_architecture(&config.architecture_name)
        .ok_or_else(|| format!("unsupported HuggingFace architecture: {}", config.architecture_name))?;
    let name_map = WeightNameMap::from_architecture(arch);
    let source = SourceCheckpoint::open(model_path)?;
    let mut tensors = Vec::new();

    append_global_tensors(&source, &config, &name_map, output_dtype, &mut tensors)?;
    append_layer_tensors(&source, &config, &name_map, output_dtype, &mut tensors)?;
    append_output_tensors(&source, &config, &name_map, output_dtype, &mut tensors)?;

    let metadata = build_metadata(&config, model_name.or_else(|| infer_model_name(&config_path)));
    write_gguf_file(output_path, &metadata, &tensors)?;

    println!(
        "wrote {} tensors to {} as {}",
        tensors.len(),
        output_path.display(),
        output_dtype
    );

    if verify {
        let model = load_gguf(output_path)?;
        println!(
            "verification: loaded {} ({} layers, hidden={}, vocab={})",
            model.config.architecture,
            model.config.num_layers,
            model.config.hidden_dim,
            model.config.vocab_size
        );
    }

    Ok(())
}

fn resolve_config_path(model_path: &Path, explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }

    let dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let path = dir.join("config.json");
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("config.json not found next to {}", model_path.display()))
    }
}

fn load_conversion_config(config_path: &Path) -> Result<ConversionConfig, String> {
    let text = std::fs::read_to_string(config_path)
        .map_err(|error| format!("failed to read {}: {}", config_path.display(), error))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("invalid JSON in {}: {}", config_path.display(), error))?;

    let architecture_name = value
        .get("architectures")
        .and_then(|architectures| architectures.as_array())
        .and_then(|architectures| architectures.first())
        .and_then(|architecture| architecture.as_str())
        .or_else(|| value.get("model_type").and_then(|model_type| model_type.as_str()))
        .ok_or("config.json is missing architectures/model_type")?
        .to_string();

    let profile = find_profile_by_hf(&architecture_name)
        .ok_or_else(|| format!("unsupported HuggingFace architecture: {}", architecture_name))?;

    let hidden_dim = json_usize(&value, &["hidden_size", "n_embd"])?.ok_or("missing hidden_size")?;
    let num_layers = json_usize(&value, &["num_hidden_layers", "n_layer"])?.ok_or("missing num_hidden_layers")?;
    let num_heads = json_usize(&value, &["num_attention_heads", "n_head"])?.ok_or("missing num_attention_heads")?;
    let num_kv_heads = json_usize(&value, &["num_key_value_heads", "n_head_kv"])?.unwrap_or(num_heads);
    let intermediate_dim = json_usize(&value, &["intermediate_size", "n_inner"])?.unwrap_or(hidden_dim * 4);
    let vocab_size = json_usize(&value, &["vocab_size"])?.ok_or("missing vocab_size")?;
    let max_context_length = json_usize(&value, &["max_position_embeddings", "n_positions", "n_ctx"])?.unwrap_or(4096);
    let rope_freq_base = json_f32(&value, &["rope_theta", "rotary_emb_base"])?.unwrap_or(10000.0);
    let rms_norm_eps = json_f32(&value, &["rms_norm_eps", "layer_norm_epsilon"])?.unwrap_or(1e-5);

    Ok(ConversionConfig {
        gguf_architecture: profile.name.to_string(),
        architecture_name,
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_dim,
        vocab_size,
        max_context_length,
        rope_freq_base,
        rms_norm_eps,
        pos_encoding_kind: profile.pos_encoding_kind,
        ffn_type: profile.ffn_type,
        has_qkv_bias: profile.has_qkv_bias,
        has_output_bias: profile.has_output_bias,
    })
}

fn append_global_tensors(
    source: &SourceCheckpoint,
    config: &ConversionConfig,
    name_map: &WeightNameMap,
    dtype: GGMLType,
    tensors: &mut Vec<GGUFWriterTensor>,
) -> Result<(), String> {
    let embed_name = name_map.resolve_global("token_embd.weight").ok_or("missing token embedding mapping")?;
    append_matrix_tensor(source, tensors, embed_name, "token_embd.weight", config.vocab_size, config.hidden_dim, dtype)?;

    if matches!(config.pos_encoding_kind, PosEncodingKind::Learned) {
        let pos_name = name_map.resolve_global("pos_embd.weight").ok_or("missing position embedding mapping")?;
        append_matrix_tensor(source, tensors, pos_name, "pos_embd.weight", config.max_context_length, config.hidden_dim, dtype)?;
    }

    Ok(())
}

fn append_layer_tensors(
    source: &SourceCheckpoint,
    config: &ConversionConfig,
    name_map: &WeightNameMap,
    dtype: GGMLType,
    tensors: &mut Vec<GGUFWriterTensor>,
) -> Result<(), String> {
    let head_dim = config.hidden_dim / config.num_heads;
    let q_dim = config.num_heads * head_dim;
    let kv_dim = config.num_kv_heads * head_dim;

    for layer in 0..config.num_layers {
        append_vector_tensor(
            source,
            tensors,
            &name_map.resolve_layer(&format!("blk.{layer}.attn_norm.weight"), layer).ok_or("missing attn_norm mapping")?,
            &format!("blk.{layer}.attn_norm.weight"),
            config.hidden_dim,
        )?;
        append_optional_vector_tensor(
            source,
            tensors,
            name_map.resolve_layer(&format!("blk.{layer}.attn_norm.bias"), layer),
            &format!("blk.{layer}.attn_norm.bias"),
            config.hidden_dim,
        )?;
        append_vector_tensor(
            source,
            tensors,
            &name_map.resolve_layer(&format!("blk.{layer}.ffn_norm.weight"), layer).ok_or("missing ffn_norm mapping")?,
            &format!("blk.{layer}.ffn_norm.weight"),
            config.hidden_dim,
        )?;
        append_optional_vector_tensor(
            source,
            tensors,
            name_map.resolve_layer(&format!("blk.{layer}.ffn_norm.bias"), layer),
            &format!("blk.{layer}.ffn_norm.bias"),
            config.hidden_dim,
        )?;

        if name_map.has_fused_qkv() {
            let fused_name = name_map.fused_qkv_weight_name(layer).ok_or("missing fused qkv mapping")?;
            let fused = source.load_tensor_f32(&fused_name, (q_dim + kv_dim + kv_dim) * config.hidden_dim)?;
            let (q, k, v) = split_fused_qkv_weight(&fused, q_dim, kv_dim, config.hidden_dim)?;
            tensors.push(quantized_matrix(&format!("blk.{layer}.attn_q.weight"), q_dim, config.hidden_dim, &q, dtype)?);
            tensors.push(quantized_matrix(&format!("blk.{layer}.attn_k.weight"), kv_dim, config.hidden_dim, &k, dtype)?);
            tensors.push(quantized_matrix(&format!("blk.{layer}.attn_v.weight"), kv_dim, config.hidden_dim, &v, dtype)?);

            if config.has_qkv_bias {
                if let Some(fused_bias_name) = name_map.fused_qkv_bias_name(layer) {
                    if source.tensor_exists(&fused_bias_name) {
                        let fused_bias = source.load_tensor_f32(&fused_bias_name, q_dim + kv_dim + kv_dim)?;
                        let (bq, bk, bv) = split_fused_qkv_bias(&fused_bias, q_dim, kv_dim)?;
                        tensors.push(f32_vector(&format!("blk.{layer}.attn_q.bias"), bq));
                        tensors.push(f32_vector(&format!("blk.{layer}.attn_k.bias"), bk));
                        tensors.push(f32_vector(&format!("blk.{layer}.attn_v.bias"), bv));
                    }
                }
            }
        } else {
            append_matrix_tensor(
                source,
                tensors,
                &name_map.resolve_layer(&format!("blk.{layer}.attn_q.weight"), layer).ok_or("missing q weight mapping")?,
                &format!("blk.{layer}.attn_q.weight"),
                q_dim,
                config.hidden_dim,
                dtype,
            )?;
            append_matrix_tensor(
                source,
                tensors,
                &name_map.resolve_layer(&format!("blk.{layer}.attn_k.weight"), layer).ok_or("missing k weight mapping")?,
                &format!("blk.{layer}.attn_k.weight"),
                kv_dim,
                config.hidden_dim,
                dtype,
            )?;
            append_matrix_tensor(
                source,
                tensors,
                &name_map.resolve_layer(&format!("blk.{layer}.attn_v.weight"), layer).ok_or("missing v weight mapping")?,
                &format!("blk.{layer}.attn_v.weight"),
                kv_dim,
                config.hidden_dim,
                dtype,
            )?;
            if config.has_qkv_bias {
                append_optional_vector_tensor(
                    source,
                    tensors,
                    name_map.resolve_layer(&format!("blk.{layer}.attn_q.bias"), layer),
                    &format!("blk.{layer}.attn_q.bias"),
                    q_dim,
                )?;
                append_optional_vector_tensor(
                    source,
                    tensors,
                    name_map.resolve_layer(&format!("blk.{layer}.attn_k.bias"), layer),
                    &format!("blk.{layer}.attn_k.bias"),
                    kv_dim,
                )?;
                append_optional_vector_tensor(
                    source,
                    tensors,
                    name_map.resolve_layer(&format!("blk.{layer}.attn_v.bias"), layer),
                    &format!("blk.{layer}.attn_v.bias"),
                    kv_dim,
                )?;
            }
        }

        append_matrix_tensor(
            source,
            tensors,
            &name_map.resolve_layer(&format!("blk.{layer}.attn_output.weight"), layer).ok_or("missing output projection mapping")?,
            &format!("blk.{layer}.attn_output.weight"),
            config.hidden_dim,
            q_dim,
            dtype,
        )?;
        if config.has_output_bias {
            append_optional_vector_tensor(
                source,
                tensors,
                name_map.resolve_layer(&format!("blk.{layer}.attn_output.bias"), layer),
                &format!("blk.{layer}.attn_output.bias"),
                config.hidden_dim,
            )?;
        }

        match config.ffn_type {
            FFNType::SwiGLU | FFNType::GeGLU => {
                append_matrix_tensor(
                    source,
                    tensors,
                    &name_map.resolve_layer(&format!("blk.{layer}.ffn_gate.weight"), layer).ok_or("missing gate mapping")?,
                    &format!("blk.{layer}.ffn_gate.weight"),
                    config.intermediate_dim,
                    config.hidden_dim,
                    dtype,
                )?;
                append_matrix_tensor(
                    source,
                    tensors,
                    &name_map.resolve_layer(&format!("blk.{layer}.ffn_up.weight"), layer).ok_or("missing up mapping")?,
                    &format!("blk.{layer}.ffn_up.weight"),
                    config.intermediate_dim,
                    config.hidden_dim,
                    dtype,
                )?;
                append_matrix_tensor(
                    source,
                    tensors,
                    &name_map.resolve_layer(&format!("blk.{layer}.ffn_down.weight"), layer).ok_or("missing down mapping")?,
                    &format!("blk.{layer}.ffn_down.weight"),
                    config.hidden_dim,
                    config.intermediate_dim,
                    dtype,
                )?;
            }
            FFNType::GELU => {
                append_matrix_tensor(
                    source,
                    tensors,
                    &name_map.resolve_layer(&format!("blk.{layer}.ffn_gate.weight"), layer).ok_or("missing fc1 mapping")?,
                    &format!("blk.{layer}.ffn_gate.weight"),
                    config.intermediate_dim,
                    config.hidden_dim,
                    dtype,
                )?;
                append_matrix_tensor(
                    source,
                    tensors,
                    &name_map.resolve_layer(&format!("blk.{layer}.ffn_down.weight"), layer).ok_or("missing fc2 mapping")?,
                    &format!("blk.{layer}.ffn_down.weight"),
                    config.hidden_dim,
                    config.intermediate_dim,
                    dtype,
                )?;
            }
        }
    }

    Ok(())
}

fn append_output_tensors(
    source: &SourceCheckpoint,
    config: &ConversionConfig,
    name_map: &WeightNameMap,
    dtype: GGMLType,
    tensors: &mut Vec<GGUFWriterTensor>,
) -> Result<(), String> {
    let norm_name = name_map.resolve_global("output_norm.weight").ok_or("missing output norm mapping")?;
    append_vector_tensor(source, tensors, norm_name, "output_norm.weight", config.hidden_dim)?;
    append_optional_vector_tensor(
        source,
        tensors,
        name_map.resolve_global("output_norm.bias").map(str::to_string),
        "output_norm.bias",
        config.hidden_dim,
    )?;

    if let Some(output_name) = name_map.resolve_global("output.weight") {
        if source.tensor_exists(output_name) {
            append_matrix_tensor(source, tensors, output_name, "output.weight", config.vocab_size, config.hidden_dim, dtype)?;
        }
    }

    Ok(())
}

fn append_matrix_tensor(
    source: &SourceCheckpoint,
    tensors: &mut Vec<GGUFWriterTensor>,
    source_name: &str,
    target_name: &str,
    rows: usize,
    cols: usize,
    dtype: GGMLType,
) -> Result<(), String> {
    let data = source.load_tensor_f32(source_name, rows * cols)?;
    tensors.push(quantized_matrix(target_name, rows, cols, &data, dtype)?);
    Ok(())
}

fn append_vector_tensor(
    source: &SourceCheckpoint,
    tensors: &mut Vec<GGUFWriterTensor>,
    source_name: &str,
    target_name: &str,
    len: usize,
) -> Result<(), String> {
    let data = source.load_tensor_f32(source_name, len)?;
    tensors.push(f32_vector(target_name, data));
    Ok(())
}

fn append_optional_vector_tensor(
    source: &SourceCheckpoint,
    tensors: &mut Vec<GGUFWriterTensor>,
    source_name: Option<String>,
    target_name: &str,
    len: usize,
) -> Result<(), String> {
    if let Some(source_name) = source_name {
        if source.tensor_exists(&source_name) {
            append_vector_tensor(source, tensors, &source_name, target_name, len)?;
        }
    }
    Ok(())
}

fn quantized_matrix(
    name: &str,
    rows: usize,
    cols: usize,
    data: &[f32],
    dtype: GGMLType,
) -> Result<GGUFWriterTensor, String> {
    Ok(GGUFWriterTensor {
        name: name.to_string(),
        dims: vec![rows as u64, cols as u64],
        dtype,
        data: quantize_matrix(data, rows, cols, dtype)?,
    })
}

fn f32_vector(name: &str, data: Vec<f32>) -> GGUFWriterTensor {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<f32>());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    GGUFWriterTensor {
        name: name.to_string(),
        dims: vec![bytes.len() as u64 / 4],
        dtype: GGMLType::F32,
        data: bytes,
    }
}

fn build_metadata(config: &ConversionConfig, model_name: Option<&str>) -> GGUFMetadata {
    let mut metadata = GGUFMetadata::new();
    metadata.values.insert(
        "general.architecture".into(),
        GGUFValue::String(config.gguf_architecture.clone()),
    );
    metadata.values.insert(
        "general.name".into(),
        GGUFValue::String(model_name.unwrap_or("nnx-quantized").to_string()),
    );

    let arch = config.gguf_architecture.clone();
    metadata.values.insert(format!("{arch}.block_count"), GGUFValue::Uint32(config.num_layers as u32));
    metadata.values.insert(format!("{arch}.embedding_length"), GGUFValue::Uint32(config.hidden_dim as u32));
    metadata.values.insert(format!("{arch}.attention.head_count"), GGUFValue::Uint32(config.num_heads as u32));
    metadata.values.insert(format!("{arch}.attention.head_count_kv"), GGUFValue::Uint32(config.num_kv_heads as u32));
    metadata.values.insert(format!("{arch}.context_length"), GGUFValue::Uint32(config.max_context_length as u32));
    metadata.values.insert(format!("{arch}.feed_forward_length"), GGUFValue::Uint32(config.intermediate_dim as u32));
    metadata.values.insert(
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        GGUFValue::Float32(config.rms_norm_eps),
    );
    if matches!(config.pos_encoding_kind, PosEncodingKind::RoPE | PosEncodingKind::PartialRoPE) {
        metadata.values.insert(format!("{arch}.rope.freq_base"), GGUFValue::Float32(config.rope_freq_base));
    }
    metadata.values.insert(
        "tokenizer.ggml.tokens".into(),
        GGUFValue::Array((0..config.vocab_size).map(|index| GGUFValue::String(index.to_string())).collect()),
    );
    metadata
}

fn infer_model_name(config_path: &Path) -> Option<&str> {
    config_path.file_stem().and_then(|stem| stem.to_str())
}

fn json_usize(value: &serde_json::Value, keys: &[&str]) -> Result<Option<usize>, String> {
    for key in keys {
        if let Some(number) = value.get(*key).and_then(|candidate| candidate.as_u64()) {
            return Ok(Some(number as usize));
        }
    }
    Ok(None)
}

fn json_f32(value: &serde_json::Value, keys: &[&str]) -> Result<Option<f32>, String> {
    for key in keys {
        if let Some(number) = value.get(*key).and_then(|candidate| candidate.as_f64()) {
            return Ok(Some(number as f32));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnx_gguf::GGUFFile;
    use nnx_transformer::Matrix;
    use std::io::Write;

    fn dtype_size(dtype: &str) -> usize {
        match dtype {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => panic!("unsupported test dtype {}", other),
        }
    }

    fn build_safetensors_bytes(tensors: &[(&str, &str, &[usize])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut tensor_data = Vec::new();

        for (name, dtype, shape) in tensors {
            let numel: usize = shape.iter().product();
            let size = numel * dtype_size(dtype);
            let end = offset + size;
            let shape_json: Vec<serde_json::Value> = shape.iter().map(|&d| serde_json::json!(d)).collect();
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape_json,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
            tensor_data.extend(std::iter::repeat_n(0u8, size));
        }

        let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_json.as_bytes());
        bytes.extend_from_slice(&tensor_data);
        bytes
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
    }

    #[test]
    fn test_convert_gpt2_safetensors_to_quantized_gguf() {
        let hidden = 32usize;
        let vocab = 16usize;
        let ctx = 32usize;
        let inter = 64usize;
        let qkv = hidden * 3;
        let temp_dir = std::env::temp_dir().join(format!(
            "nnx_quantize_model_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let st_path = temp_dir.join("model.safetensors");
        let config_path = temp_dir.join("config.json");
        let gguf_path = temp_dir.join("model-q8.gguf");

        let embed_shape = [vocab, hidden];
        let pos_shape = [ctx, hidden];
        let hidden_shape = [hidden];
        let fused_attn_shape = [qkv, hidden];
        let fused_attn_bias_shape = [qkv];
        let attn_proj_shape = [hidden, hidden];
        let ffn_up_shape = [inter, hidden];
        let ffn_down_shape = [hidden, inter];

        let tensors = [
            ("transformer.wte.weight", "F32", embed_shape.as_slice()),
            ("transformer.wpe.weight", "F32", pos_shape.as_slice()),
            ("transformer.h.0.ln_1.weight", "F32", hidden_shape.as_slice()),
            ("transformer.h.0.ln_1.bias", "F32", hidden_shape.as_slice()),
            ("transformer.h.0.ln_2.weight", "F32", hidden_shape.as_slice()),
            ("transformer.h.0.ln_2.bias", "F32", hidden_shape.as_slice()),
            ("transformer.h.0.attn.c_attn.weight", "F32", fused_attn_shape.as_slice()),
            ("transformer.h.0.attn.c_attn.bias", "F32", fused_attn_bias_shape.as_slice()),
            ("transformer.h.0.attn.c_proj.weight", "F32", attn_proj_shape.as_slice()),
            ("transformer.h.0.attn.c_proj.bias", "F32", hidden_shape.as_slice()),
            ("transformer.h.0.mlp.c_fc.weight", "F32", ffn_up_shape.as_slice()),
            ("transformer.h.0.mlp.c_proj.weight", "F32", ffn_down_shape.as_slice()),
            ("transformer.ln_f.weight", "F32", hidden_shape.as_slice()),
            ("transformer.ln_f.bias", "F32", hidden_shape.as_slice()),
            ("lm_head.weight", "F32", embed_shape.as_slice()),
        ];
        write_file(&st_path, &build_safetensors_bytes(&tensors));

        let config_json = serde_json::json!({
            "architectures": ["GPT2LMHeadModel"],
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": inter,
            "vocab_size": vocab,
            "max_position_embeddings": ctx,
            "layer_norm_epsilon": 1e-5
        });
        write_file(&config_path, serde_json::to_string(&config_json).unwrap().as_bytes());

        convert_safetensors_to_gguf(&st_path, Some(&config_path), &gguf_path, GGMLType::Q8_0, Some("tiny-gpt2"), true).unwrap();

        let gguf = GGUFFile::open(&gguf_path).unwrap();
        assert_eq!(gguf.metadata.architecture(), Some("gpt2"));
        assert!(gguf.tensors.contains_key("blk.0.attn_q.weight"));
        assert_eq!(gguf.tensors["blk.0.attn_q.weight"].dtype, GGMLType::Q8_0);

        let model = load_gguf(&gguf_path).unwrap();
        match &model.weights.layers[0].wq {
            Matrix::Quantized { dtype, .. } => assert_eq!(*dtype, GGMLType::Q8_0),
            other => panic!("expected quantized q weight, got {:?}", other),
        }

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}