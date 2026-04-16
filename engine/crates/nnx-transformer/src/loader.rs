//! Load a complete model from a GGUF file.
//!
//! Parses the GGUF file, extracts config from metadata, dequantizes all
//! weight tensors, and assembles a ready-to-run Model.
//!
//! Supports architecture-specific tensor names and optional bias loading.

use crate::block::BlockWeights;
use crate::config::{
    Architecture, FFNType, ModelConfig, NormType, build_pos_encoding, find_profile_by_hf,
    known_architecture_names,
};
use crate::model::{Model, ModelWeights};
use crate::weight_names::{
    WeightNameMap, load_shard_index, split_fused_qkv_bias, split_fused_qkv_weight,
};
use crate::weights::Matrix;
use nnx_gguf::GGUFFile;
use nnx_quant::GGMLType;
use nnx_quant::dequant::dequantize_alloc;
use std::path::Path;
use tracing::{info, warn};

/// Load a model from a GGUF file path.
///
/// This dequantizes all weights to f32. For large models this uses
/// significant memory -- quantized-in-memory compute is a future optimization.
pub fn load_gguf(path: &Path) -> Result<Model, String> {
    info!("Loading GGUF model from {}", path.display());

    let gguf = GGUFFile::open(path).map_err(|e| format!("failed to open GGUF: {}", e))?;
    let config = ModelConfig::from_gguf(&gguf.metadata)?;

    info!(
        "Model: {} ({:?}) -- {} layers, hidden={}, heads={}/{}, vocab={}, ctx={}",
        config.architecture,
        config.arch,
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.vocab_size,
        config.max_context_length,
    );
    info!(
        "Config: norm={:?}, ffn={:?}, pos={:?}, block={:?}, qkv_bias={}, out_bias={}",
        config.norm_type,
        config.ffn_type,
        config.pos_encoding,
        config.block_style,
        config.has_qkv_bias,
        config.has_output_bias,
    );
    info!(
        "Estimated params: {:.1}B",
        config.estimated_params() as f64 / 1e9
    );

    let weights = load_weights(&gguf, &config)?;
    Ok(Model::new(config, weights))
}

fn load_weights(gguf: &GGUFFile, config: &ModelConfig) -> Result<ModelWeights, String> {
    // Token embedding
    let token_embedding = load_matrix_tensor(
        gguf,
        "token_embd.weight",
        config.vocab_size,
        config.hidden_dim,
    )?;
    let position_embedding = match config.pos_encoding {
        crate::config::PosEncoding::Learned => Some(load_matrix_tensor(
            gguf,
            "pos_embd.weight",
            config.max_context_length,
            config.hidden_dim,
        )?),
        _ => None,
    };

    // Per-layer weights
    let mut layers = Vec::with_capacity(config.num_layers);
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");
        let layer = load_block_weights(gguf, config, &prefix, q_dim, kv_dim)?;
        layers.push(layer);

        if (i + 1) % 10 == 0 || i == config.num_layers - 1 {
            info!("Loaded layer {}/{}", i + 1, config.num_layers);
        }
    }

    // Final norm
    let final_norm = load_tensor(gguf, "output_norm.weight", config.hidden_dim)?;

    // Final norm bias (for LayerNorm architectures)
    let final_norm_bias = match config.norm_type {
        NormType::LayerNorm => try_load_tensor(gguf, "output_norm.bias", config.hidden_dim),
        NormType::RMSNorm => None,
    };

    // LM head -- sometimes tied with token_embd.weight
    let lm_head = if gguf.tensors.contains_key("output.weight") {
        load_matrix_tensor(gguf, "output.weight", config.vocab_size, config.hidden_dim)?
    } else {
        warn!("output.weight not found, using token_embd.weight as LM head (tied embeddings)");
        token_embedding.clone()
    };

    Ok(ModelWeights {
        token_embedding,
        position_embedding,
        layers,
        final_norm,
        final_norm_bias,
        lm_head,
    })
}

/// Load weights for a single transformer block, including optional bias terms.
fn load_block_weights(
    gguf: &GGUFFile,
    config: &ModelConfig,
    prefix: &str,
    q_dim: usize,
    kv_dim: usize,
) -> Result<BlockWeights, String> {
    // Attention and FFN norm weights
    let attn_norm = load_tensor(
        gguf,
        &format!("{prefix}.attn_norm.weight"),
        config.hidden_dim,
    )?;
    let ffn_norm = load_tensor_or_default(
        gguf,
        &format!("{prefix}.ffn_norm.weight"),
        config.hidden_dim,
        1.0,
    );

    // Q, K, V, O projections
    let wq = load_matrix_tensor(
        gguf,
        &format!("{prefix}.attn_q.weight"),
        q_dim,
        config.hidden_dim,
    )?;
    let wk = load_matrix_tensor(
        gguf,
        &format!("{prefix}.attn_k.weight"),
        kv_dim,
        config.hidden_dim,
    )?;
    let wv = load_matrix_tensor(
        gguf,
        &format!("{prefix}.attn_v.weight"),
        kv_dim,
        config.hidden_dim,
    )?;
    let wo = load_matrix_tensor(
        gguf,
        &format!("{prefix}.attn_output.weight"),
        config.hidden_dim,
        q_dim,
    )?;

    // FFN weights -- architecture-specific
    let (w_gate, w_up, w_down) = load_ffn_weights(gguf, config, prefix)?;

    // Optional bias terms
    let bq = if config.has_qkv_bias {
        try_load_tensor(gguf, &format!("{prefix}.attn_q.bias"), q_dim)
    } else {
        None
    };
    let bk = if config.has_qkv_bias {
        try_load_tensor(gguf, &format!("{prefix}.attn_k.bias"), kv_dim)
    } else {
        None
    };
    let bv = if config.has_qkv_bias {
        try_load_tensor(gguf, &format!("{prefix}.attn_v.bias"), kv_dim)
    } else {
        None
    };
    let bo = if config.has_output_bias {
        try_load_tensor(
            gguf,
            &format!("{prefix}.attn_output.bias"),
            config.hidden_dim,
        )
    } else {
        None
    };

    // Norm bias (for LayerNorm architectures)
    let attn_norm_bias = match config.norm_type {
        NormType::LayerNorm => {
            try_load_tensor(gguf, &format!("{prefix}.attn_norm.bias"), config.hidden_dim)
        }
        NormType::RMSNorm => None,
    };
    let ffn_norm_bias = match config.norm_type {
        NormType::LayerNorm => {
            try_load_tensor(gguf, &format!("{prefix}.ffn_norm.bias"), config.hidden_dim)
        }
        NormType::RMSNorm => None,
    };

    Ok(BlockWeights {
        attn_norm,
        ffn_norm,
        wq,
        wk,
        wv,
        wo,
        w_gate,
        w_up,
        w_down,
        bq,
        bk,
        bv,
        bo,
        attn_norm_bias,
        ffn_norm_bias,
    })
}

/// Load FFN weights, handling different naming conventions per architecture.
///
/// For SwiGLU/GeGLU: gate, up, down (3 matrices)
/// For GELU (GPT-2): fc1 -> w_gate, fc2 -> w_down, w_up is empty
fn load_ffn_weights(
    gguf: &GGUFFile,
    config: &ModelConfig,
    prefix: &str,
) -> Result<(Matrix, Matrix, Matrix), String> {
    let inter = config.intermediate_dim;
    let hidden = config.hidden_dim;

    match config.ffn_type {
        FFNType::SwiGLU | FFNType::GeGLU => {
            let w_gate =
                load_matrix_tensor(gguf, &format!("{prefix}.ffn_gate.weight"), inter, hidden)?;
            let w_up = load_matrix_tensor(gguf, &format!("{prefix}.ffn_up.weight"), inter, hidden)?;
            let w_down =
                load_matrix_tensor(gguf, &format!("{prefix}.ffn_down.weight"), hidden, inter)?;
            Ok((w_gate, w_up, w_down))
        }
        FFNType::GELU => {
            // GPT-2 uses ffn_up/ffn_down or fc1/fc2 naming
            // Try multiple name patterns
            let w_fc1 = load_matrix_tensor_multi_name(
                gguf,
                &[
                    format!("{prefix}.ffn_up.weight"),
                    format!("{prefix}.ffn_gate.weight"),
                ],
                inter,
                hidden,
                "fc1/ffn_up",
            )?;
            let w_fc2 = load_matrix_tensor_multi_name(
                gguf,
                &[format!("{prefix}.ffn_down.weight")],
                hidden,
                inter,
                "fc2/ffn_down",
            )?;
            // w_up is unused for GELU FFN, but we keep it for struct consistency
            let w_up = Matrix::dense(Vec::new(), 0, hidden);
            Ok((w_fc1, w_up, w_fc2))
        }
    }
}

/// Load and dequantize a single tensor from GGUF.
fn load_tensor(gguf: &GGUFFile, name: &str, expected_numel: usize) -> Result<Vec<f32>, String> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor not found in GGUF: {}", name))?;

    let numel = info.numel() as usize;
    if numel != expected_numel {
        return Err(format!(
            "tensor {} has {} elements, expected {}",
            name, numel, expected_numel
        ));
    }

    let data = gguf
        .tensor_data(name)
        .map_err(|e| format!("failed to read tensor {}: {}", name, e))?;

    let output = dequantize_alloc(data, info.dtype, numel);
    if output.iter().take(numel).all(|&v| v == 0.0) && numel > 0 {
        warn!(
            "tensor {} dequantized to all zeros (dtype: {})",
            name, info.dtype
        );
    }

    Ok(output)
}

/// Load a matrix tensor, keeping compact GGUF storage for non-f32 dtypes.
fn load_matrix_tensor(
    gguf: &GGUFFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix, String> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor not found in GGUF: {}", name))?;

    let expected_numel = rows * cols;
    let numel = info.numel() as usize;
    if numel != expected_numel {
        return Err(format!(
            "tensor {} has {} elements, expected {}",
            name, numel, expected_numel
        ));
    }

    let data = gguf
        .tensor_data(name)
        .map_err(|e| format!("failed to read tensor {}: {}", name, e))?;

    match info.dtype {
        GGMLType::F32 => Ok(Matrix::dense(
            dequantize_alloc(data, info.dtype, numel),
            rows,
            cols,
        )),
        dtype => Matrix::quantized(data.to_vec(), dtype, rows, cols),
    }
}

/// Try to load a tensor; return None if it doesn't exist in the GGUF file.
fn try_load_tensor(gguf: &GGUFFile, name: &str, expected_numel: usize) -> Option<Vec<f32>> {
    load_tensor(gguf, name, expected_numel).ok()
}

/// Load a tensor or return a default-filled vector if not found.
fn load_tensor_or_default(
    gguf: &GGUFFile,
    name: &str,
    expected_numel: usize,
    default_value: f32,
) -> Vec<f32> {
    match load_tensor(gguf, name, expected_numel) {
        Ok(data) => data,
        Err(_) => vec![default_value; expected_numel],
    }
}

// ============================================================
// SafeTensors loading
// ============================================================

/// Load a model from a SafeTensors file.
///
/// Infers model configuration from tensor shapes since SafeTensors
/// don't have a metadata section like GGUF.
pub fn load_safetensors(path: &Path) -> Result<Model, String> {
    load_safetensors_with_budget(path, 0)
}

/// Load a SafeTensors model with memory budget enforcement.
///
/// Automatically detects multi-file (sharded) models via model.safetensors.index.json
/// and routes tensor loads to the correct shard file.
pub fn load_safetensors_with_budget(path: &Path, memory_budget: usize) -> Result<Model, String> {
    use nnx_safetensors::SafeTensorsFile;

    info!("Loading SafeTensors model from {}", path.display());

    // Check for a multi-file shard index before opening the path as a single file.
    match load_shard_index(path) {
        Ok(Some(shard_idx)) => {
            info!(
                "Detected sharded model: {} tensors across {} files",
                shard_idx.tensor_to_file.len(),
                shard_idx.shard_files.len()
            );
            return load_safetensors_sharded(path, &shard_idx, memory_budget);
        }
        Ok(None) => {} // single-file model, continue below
        Err(e) => {
            warn!(
                "could not read shard index: {} — proceeding as single file",
                e
            );
        }
    }

    let st =
        SafeTensorsFile::open(path).map_err(|e| format!("failed to open SafeTensors: {}", e))?;

    let config = infer_config_from_safetensors_path(path, &st)?;
    config.validate()?;

    if memory_budget > 0 {
        let estimated = estimate_safetensors_memory(&st, &config);
        if estimated > memory_budget {
            return Err(format!(
                "model requires ~{:.1} GB but memory budget is {:.1} GB",
                estimated as f64 / 1e9,
                memory_budget as f64 / 1e9
            ));
        }
    }

    info!(
        "Model: {} ({:?}) — {} layers, hidden={}, heads={}/{}, vocab={}",
        config.architecture,
        config.arch,
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.vocab_size,
    );

    let name_map = WeightNameMap::from_architecture(config.arch.clone());
    let weights = load_safetensors_weights(&st, &config, &name_map)?;
    Ok(Model::new(config, weights))
}

/// Load a sharded (multi-file) SafeTensors model.
fn load_safetensors_sharded(
    model_path: &Path,
    shard_idx: &crate::weight_names::ShardIndex,
    memory_budget: usize,
) -> Result<Model, String> {
    use nnx_safetensors::SafeTensorsFile;
    use std::collections::HashMap;

    // Resolve the model directory.
    let dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf()
    };

    // Use the first shard to probe tensor shapes for config inference.
    let first_shard = shard_idx
        .shard_files
        .first()
        .ok_or("shard index lists no shard files")?;
    let first_st = SafeTensorsFile::open(first_shard)
        .map_err(|e| format!("failed to open shard {}: {}", first_shard.display(), e))?;

    // The index file sits in the same directory as config.json.
    let index_path = dir.join("model.safetensors.index.json");
    let config = infer_config_from_safetensors_path(&index_path, &first_st)?;
    config.validate()?;

    if memory_budget > 0 {
        let estimated: usize = shard_idx
            .tensor_to_file
            .keys()
            .filter_map(|name| {
                let shard = shard_idx.tensor_to_file.get(name)?;
                let st = SafeTensorsFile::open(shard).ok()?;
                let info = st.tensor_info(name)?;
                Some(info.shape.numel() * std::mem::size_of::<f32>())
            })
            .sum();
        if estimated > memory_budget {
            return Err(format!(
                "model requires ~{:.1} GB but memory budget is {:.1} GB",
                estimated as f64 / 1e9,
                memory_budget as f64 / 1e9
            ));
        }
    }

    info!(
        "Sharded model: {} ({:?}) — {} layers, hidden={}, heads={}/{}, vocab={}",
        config.architecture,
        config.arch,
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.vocab_size,
    );

    // Open each shard file once and cache the handle.
    let mut shard_cache: HashMap<std::path::PathBuf, SafeTensorsFile> = HashMap::new();
    for shard_path in &shard_idx.shard_files {
        let st = SafeTensorsFile::open(shard_path)
            .map_err(|e| format!("failed to open shard {}: {}", shard_path.display(), e))?;
        shard_cache.insert(shard_path.clone(), st);
    }

    let name_map = WeightNameMap::from_architecture(config.arch.clone());
    let loader = ShardedTensorLoader::new(&shard_cache, shard_idx);
    let weights = load_safetensors_weights_sharded(&loader, &config, &name_map)?;
    Ok(Model::new(config, weights))
}

fn infer_config_from_safetensors_path(
    path: &Path,
    st: &nnx_safetensors::SafeTensorsFile,
) -> Result<ModelConfig, String> {
    if let Some(config) = infer_config_from_adjacent_config(path, st)? {
        return Ok(config);
    }
    infer_config_from_safetensors(st)
}

fn infer_config_from_safetensors(
    st: &nnx_safetensors::SafeTensorsFile,
) -> Result<ModelConfig, String> {
    let embed_info = st
        .tensor_info("model.embed_tokens.weight")
        .ok_or("missing model.embed_tokens.weight")?;
    let vocab_size = embed_info.shape.dims()[0];
    let hidden_dim = embed_info.shape.dims()[1];

    let mut num_layers = 0;
    loop {
        let name = format!("model.layers.{}.self_attn.q_proj.weight", num_layers);
        if st.tensor_info(&name).is_none() {
            break;
        }
        num_layers += 1;
    }
    if num_layers == 0 {
        return Err("no transformer layers found".into());
    }

    let q_info = st
        .tensor_info("model.layers.0.self_attn.q_proj.weight")
        .ok_or("missing q_proj")?;
    let k_info = st
        .tensor_info("model.layers.0.self_attn.k_proj.weight")
        .ok_or("missing k_proj")?;
    let q_dim = q_info.shape.dims()[0];
    let kv_dim = k_info.shape.dims()[0];

    // Infer head_dim from standard values
    let head_dim = [128, 64, 32, 16, 8, 4]
        .into_iter()
        .find(|&hd| q_dim % hd == 0 && kv_dim % hd == 0 && hidden_dim % hd == 0)
        .unwrap_or(64);
    let num_heads = q_dim / head_dim;
    let num_kv_heads = kv_dim / head_dim;

    let gate_info = st
        .tensor_info("model.layers.0.mlp.gate_proj.weight")
        .ok_or("missing gate_proj")?;
    let intermediate_dim = gate_info.shape.dims()[0];

    Ok(ModelConfig {
        architecture: "llama".into(),
        arch: Architecture::Llama,
        num_layers,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_dim,
        vocab_size,
        max_context_length: 4096,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        norm_type: NormType::RMSNorm,
        ffn_type: FFNType::SwiGLU,
        pos_encoding: crate::config::PosEncoding::RoPE { freq_base: 10000.0 },
        block_style: crate::config::BlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
        embedding_scale: None,
    })
}

fn infer_config_from_adjacent_config(
    path: &Path,
    st: &nnx_safetensors::SafeTensorsFile,
) -> Result<Option<ModelConfig>, String> {
    let Some(parent) = path.parent() else {
        return Ok(None);
    };
    let config_path = parent.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }

    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read {}: {}", config_path.display(), e))?;
    let value: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|e| format!("failed to parse {}: {}", config_path.display(), e))?;

    let arch_name = value
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .or_else(|| value.get("model_type").and_then(|v| v.as_str()))
        .ok_or("config.json missing architecture/model_type")?;

    // Use the profile registry instead of a hardcoded match statement.
    // Adding support for a new architecture only requires adding a profile to
    // KNOWN_ARCHITECTURES in config.rs — this function never needs to change.
    let profile = find_profile_by_hf(arch_name).ok_or_else(|| {
        format!(
            "unsupported SafeTensors architecture hint: '{}'. Known architectures: {:?}",
            arch_name,
            known_architecture_names()
        )
    })?;

    let hidden_dim = json_usize(&value, "hidden_size")
        .or_else(|| {
            st.tensor_info("model.embed_tokens.weight")
                .map(|info| info.shape.dims()[1])
        })
        .ok_or("missing hidden_size")?;
    // Layer count: prefer explicit field; fall back to probing via architecture-
    // specific tensor name so GPT-2 and Llama-family both count correctly.
    let arch_enum_for_probe =
        crate::weight_names::map_hf_architecture(arch_name).unwrap_or(Architecture::Llama);
    let probe_map = WeightNameMap::from_architecture(arch_enum_for_probe);
    let num_layers = json_usize(&value, "num_hidden_layers")
        .or_else(|| {
            let n = probe_map.count_layers(&|name| st.tensor_info(name).is_some());
            if n > 0 { Some(n) } else { None }
        })
        .ok_or("missing num_hidden_layers")?;
    let num_heads =
        json_usize(&value, "num_attention_heads").ok_or("missing num_attention_heads")?;
    let num_kv_heads = json_usize(&value, "num_key_value_heads").unwrap_or(num_heads);
    let head_dim = hidden_dim / num_heads;
    let intermediate_dim = json_usize(&value, "intermediate_size")
        .or_else(|| {
            st.tensor_info("model.layers.0.mlp.gate_proj.weight")
                .map(|info| info.shape.dims()[0])
        })
        .ok_or("missing intermediate_size")?;
    let vocab_size = json_usize(&value, "vocab_size")
        .or_else(|| {
            st.tensor_info("model.embed_tokens.weight")
                .map(|info| info.shape.dims()[0])
        })
        .ok_or("missing vocab_size")?;
    let max_context_length = json_usize(&value, "max_position_embeddings").unwrap_or(4096);
    let rope_freq_base = json_f32(&value, "rope_theta").unwrap_or(10000.0);
    let rms_norm_eps = json_f32(&value, "rms_norm_eps")
        .or_else(|| json_f32(&value, "layer_norm_epsilon"))
        .unwrap_or(1e-5);

    let pos_encoding = build_pos_encoding(profile.pos_encoding_kind, rope_freq_base, head_dim);
    let embedding_scale = if profile.has_embedding_scale {
        Some((hidden_dim as f32).sqrt())
    } else {
        None
    };
    let arch = crate::config::profile_to_arch_enum(profile, arch_name);

    // Probe the actual checkpoint for bias tensors rather than relying solely
    // on profile defaults.  Some checkpoints omit bias even when the
    // architecture spec says it should be present, and probing avoids
    // errors later when we try to load a bias that isn't there.
    //
    // We build the name map *after* we know arch so the probe uses
    // architecture-correct HF names (e.g., Phi uses qkv_proj not q_proj).
    let bias_probe_map = WeightNameMap::from_architecture(arch.clone());
    let has_qkv_bias = if profile.has_qkv_bias {
        probe_qkv_bias(st, &bias_probe_map)
    } else {
        false
    };
    let has_output_bias = if profile.has_output_bias {
        probe_output_bias(st, &bias_probe_map)
    } else {
        false
    };

    Ok(Some(ModelConfig {
        architecture: arch_name.to_string(),
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
        norm_type: profile.norm_type,
        ffn_type: profile.ffn_type,
        pos_encoding,
        block_style: profile.block_style,
        has_qkv_bias,
        has_output_bias,
        embedding_scale,
    }))
}

/// Architecture-aware single-file SafeTensors weight loader.
///
/// Uses `WeightNameMap` to translate internal weight keys to architecture-specific
/// HuggingFace tensor names. Handles split QKV (Llama/Mistral/Gemma/Qwen2) and
/// fused QKV (GPT-2/Phi) transparently.
fn load_safetensors_weights(
    st: &nnx_safetensors::SafeTensorsFile,
    config: &ModelConfig,
    name_map: &WeightNameMap,
) -> Result<ModelWeights, String> {
    let embed_name = name_map
        .resolve_global("token_embd.weight")
        .unwrap_or("model.embed_tokens.weight");
    let token_embedding = Matrix::dense(
        load_st_tensor(st, embed_name, config.vocab_size * config.hidden_dim)?,
        config.vocab_size,
        config.hidden_dim,
    );
    let position_embedding = match config.pos_encoding {
        crate::config::PosEncoding::Learned => {
            let pos_name = name_map
                .resolve_global("pos_embd.weight")
                .unwrap_or("transformer.wpe.weight");
            Some(Matrix::dense(
                load_st_tensor(st, pos_name, config.max_context_length * config.hidden_dim)?,
                config.max_context_length,
                config.hidden_dim,
            ))
        }
        _ => None,
    };

    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_block_weights_hf(
            &SingleFileTensorSource { st },
            config,
            name_map,
            i,
            q_dim,
            kv_dim,
        )?;
        layers.push(layer);
        if (i + 1) % 10 == 0 || i == config.num_layers - 1 {
            info!("Loaded layer {}/{}", i + 1, config.num_layers);
        }
    }

    let norm_name = name_map
        .resolve_global("output_norm.weight")
        .unwrap_or("model.norm.weight");
    let final_norm = load_st_tensor(st, norm_name, config.hidden_dim)?;

    let final_norm_bias = if matches!(config.norm_type, NormType::LayerNorm) {
        let bias_name = name_map
            .resolve_global("output_norm.bias")
            .unwrap_or("model.norm.bias");
        try_load_st_tensor(st, bias_name, config.hidden_dim)
    } else {
        None
    };

    let lm_head_name = name_map
        .resolve_global("output.weight")
        .unwrap_or("lm_head.weight");
    let lm_head = if st.tensor_info(lm_head_name).is_some() {
        Matrix::dense(
            load_st_tensor(st, lm_head_name, config.vocab_size * config.hidden_dim)?,
            config.vocab_size,
            config.hidden_dim,
        )
    } else {
        warn!(
            "{} not found, using embed_tokens (tied embeddings)",
            lm_head_name
        );
        token_embedding.clone()
    };

    Ok(ModelWeights {
        token_embedding,
        position_embedding,
        layers,
        final_norm,
        final_norm_bias,
        lm_head,
    })
}

// ============================================================
// Sharded SafeTensors loading
// ============================================================

/// Holds open shard file handles indexed by path.
///
/// This struct exists to avoid HRTB closure-returning-reference issues.
/// Instead of a closure `|name: &str| -> Option<&SafeTensorsFile>`, callers
/// use the `load_tensor` and `try_load_tensor` methods which open the correct
/// shard for each tensor directly.
struct ShardedTensorLoader<'a> {
    shard_cache:
        &'a std::collections::HashMap<std::path::PathBuf, nnx_safetensors::SafeTensorsFile>,
    shard_idx: &'a crate::weight_names::ShardIndex,
}

impl<'a> ShardedTensorLoader<'a> {
    fn new(
        shard_cache: &'a std::collections::HashMap<
            std::path::PathBuf,
            nnx_safetensors::SafeTensorsFile,
        >,
        shard_idx: &'a crate::weight_names::ShardIndex,
    ) -> Self {
        Self {
            shard_cache,
            shard_idx,
        }
    }

    /// Return the shard file that contains `tensor_name`, or `None`.
    fn shard_for(&self, tensor_name: &str) -> Option<&nnx_safetensors::SafeTensorsFile> {
        let path = self.shard_idx.tensor_to_file.get(tensor_name)?;
        self.shard_cache.get(path)
    }

    fn load_tensor(&self, name: &str, expected_numel: usize) -> Result<Vec<f32>, String> {
        let st = self
            .shard_for(name)
            .ok_or_else(|| format!("tensor '{}' not found in any shard", name))?;
        load_st_tensor(st, name, expected_numel)
    }

    fn try_load_tensor(&self, name: &str, expected_numel: usize) -> Option<Vec<f32>> {
        let st = self.shard_for(name)?;
        load_st_tensor(st, name, expected_numel).ok()
    }

    fn tensor_exists(&self, name: &str) -> bool {
        self.shard_for(name)
            .and_then(|st| st.tensor_info(name))
            .is_some()
    }
}

/// Architecture-aware multi-shard SafeTensors weight loader.
fn load_safetensors_weights_sharded(
    loader: &ShardedTensorLoader<'_>,
    config: &ModelConfig,
    name_map: &WeightNameMap,
) -> Result<ModelWeights, String> {
    let embed_name = name_map
        .resolve_global("token_embd.weight")
        .unwrap_or("model.embed_tokens.weight");
    let token_embedding = Matrix::dense(
        loader.load_tensor(embed_name, config.vocab_size * config.hidden_dim)?,
        config.vocab_size,
        config.hidden_dim,
    );
    let position_embedding = match config.pos_encoding {
        crate::config::PosEncoding::Learned => {
            let pos_name = name_map
                .resolve_global("pos_embd.weight")
                .unwrap_or("transformer.wpe.weight");
            Some(Matrix::dense(
                loader.load_tensor(pos_name, config.max_context_length * config.hidden_dim)?,
                config.max_context_length,
                config.hidden_dim,
            ))
        }
        _ => None,
    };

    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = load_block_weights_hf(loader, config, name_map, i, q_dim, kv_dim)?;
        layers.push(layer);
        if (i + 1) % 10 == 0 || i == config.num_layers - 1 {
            info!("Loaded layer {}/{}", i + 1, config.num_layers);
        }
    }

    let norm_name = name_map
        .resolve_global("output_norm.weight")
        .unwrap_or("model.norm.weight");
    let final_norm = loader.load_tensor(norm_name, config.hidden_dim)?;

    let final_norm_bias = if matches!(config.norm_type, NormType::LayerNorm) {
        let bias_name = name_map
            .resolve_global("output_norm.bias")
            .unwrap_or("model.norm.bias");
        loader.try_load_tensor(bias_name, config.hidden_dim)
    } else {
        None
    };

    let lm_head_name = name_map
        .resolve_global("output.weight")
        .unwrap_or("lm_head.weight");
    let lm_head = if loader.tensor_exists(lm_head_name) {
        Matrix::dense(
            loader.load_tensor(lm_head_name, config.vocab_size * config.hidden_dim)?,
            config.vocab_size,
            config.hidden_dim,
        )
    } else {
        warn!(
            "{} not found in shards, using embed_tokens (tied embeddings)",
            lm_head_name
        );
        token_embedding.clone()
    };

    Ok(ModelWeights {
        token_embedding,
        position_embedding,
        layers,
        final_norm,
        final_norm_bias,
        lm_head,
    })
}

// ============================================================
// Unified tensor source trait
// ============================================================

/// Abstraction over single-file and sharded tensor sources.
///
/// This trait lets `load_block_weights_hf` work identically for both the
/// single-file and multi-shard cases without duplicating logic.
trait TensorSource {
    fn load_tensor(&self, name: &str, expected_numel: usize) -> Result<Vec<f32>, String>;
    fn try_load_tensor(&self, name: &str, expected_numel: usize) -> Option<Vec<f32>>;
    fn tensor_exists(&self, name: &str) -> bool;
}

/// Single-file tensor source backed by a `SafeTensorsFile`.
struct SingleFileTensorSource<'a> {
    st: &'a nnx_safetensors::SafeTensorsFile,
}

impl<'a> TensorSource for SingleFileTensorSource<'a> {
    fn load_tensor(&self, name: &str, expected_numel: usize) -> Result<Vec<f32>, String> {
        load_st_tensor(self.st, name, expected_numel)
    }

    fn try_load_tensor(&self, name: &str, expected_numel: usize) -> Option<Vec<f32>> {
        try_load_st_tensor(self.st, name, expected_numel)
    }

    fn tensor_exists(&self, name: &str) -> bool {
        self.st.tensor_info(name).is_some()
    }
}

impl<'a> TensorSource for ShardedTensorLoader<'a> {
    fn load_tensor(&self, name: &str, expected_numel: usize) -> Result<Vec<f32>, String> {
        self.load_tensor(name, expected_numel)
    }

    fn try_load_tensor(&self, name: &str, expected_numel: usize) -> Option<Vec<f32>> {
        self.try_load_tensor(name, expected_numel)
    }

    fn tensor_exists(&self, name: &str) -> bool {
        self.tensor_exists(name)
    }
}

// ============================================================
// Per-layer HF weight loading
// ============================================================

/// Load a single transformer block's weights from any tensor source.
///
/// Handles fused QKV (GPT-2, Phi) by splitting after loading, and split QKV
/// (Llama, Mistral, Gemma, Qwen2) directly.  FFN naming varies per architecture
/// but the internal keys are stable.
fn load_block_weights_hf(
    src: &dyn TensorSource,
    config: &ModelConfig,
    name_map: &WeightNameMap,
    layer: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<BlockWeights, String> {
    // Helper: resolve an internal layer key to the HF name for this layer.
    let hf_name = |internal: &str| -> String {
        name_map
            .resolve_layer(internal, layer)
            .unwrap_or_else(|| format!("model.layers.{layer}.{internal}"))
    };

    // Attention norm
    let attn_norm = src
        .load_tensor(
            &hf_name(&format!("blk.{layer}.attn_norm.weight")),
            config.hidden_dim,
        )
        .or_else(|_| {
            // Some architectures use `input_layernorm` at the global naming level
            // which resolve_layer already handles; this fallback is just defensive.
            src.load_tensor(
                &format!("model.layers.{layer}.input_layernorm.weight"),
                config.hidden_dim,
            )
        })?;

    // FFN norm (may default to ones for RMSNorm-only arches that don't store it)
    let ffn_norm = src
        .try_load_tensor(
            &hf_name(&format!("blk.{layer}.ffn_norm.weight")),
            config.hidden_dim,
        )
        .or_else(|| {
            src.try_load_tensor(
                &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                config.hidden_dim,
            )
        })
        .unwrap_or_else(|| vec![1.0f32; config.hidden_dim]);

    // Norm biases (LayerNorm only)
    let attn_norm_bias = if matches!(config.norm_type, NormType::LayerNorm) {
        src.try_load_tensor(
            &hf_name(&format!("blk.{layer}.attn_norm.bias")),
            config.hidden_dim,
        )
    } else {
        None
    };
    let ffn_norm_bias = if matches!(config.norm_type, NormType::LayerNorm) {
        src.try_load_tensor(
            &hf_name(&format!("blk.{layer}.ffn_norm.bias")),
            config.hidden_dim,
        )
    } else {
        None
    };

    // Q, K, V projections — fused or split depending on architecture
    let (wq, wk, wv, bq, bk, bv) = if name_map.has_fused_qkv() {
        load_fused_qkv(src, name_map, config, layer, q_dim, kv_dim)?
    } else {
        load_split_qkv(src, &hf_name, config, layer, q_dim, kv_dim)?
    };

    // Output projection
    let wo_name = hf_name(&format!("blk.{layer}.attn_output.weight"));
    let wo = Matrix::dense(
        src.load_tensor(&wo_name, config.hidden_dim * q_dim)?,
        config.hidden_dim,
        q_dim,
    );
    let bo = if config.has_output_bias {
        let bo_name = hf_name(&format!("blk.{layer}.attn_output.bias"));
        src.try_load_tensor(&bo_name, config.hidden_dim)
    } else {
        None
    };

    // FFN weights
    let (w_gate, w_up, w_down) = load_ffn_weights_hf(src, &hf_name, config, layer)?;

    Ok(BlockWeights {
        attn_norm,
        ffn_norm,
        wq,
        wk,
        wv,
        wo,
        w_gate,
        w_up,
        w_down,
        bq,
        bk,
        bv,
        bo,
        attn_norm_bias,
        ffn_norm_bias,
    })
}

/// Load split Q, K, V projections (standard Llama/Mistral/Gemma/Qwen2 layout).
fn load_split_qkv(
    src: &dyn TensorSource,
    hf_name: &impl Fn(&str) -> String,
    config: &ModelConfig,
    layer: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<
    (
        Matrix,
        Matrix,
        Matrix,
        Option<Vec<f32>>,
        Option<Vec<f32>>,
        Option<Vec<f32>>,
    ),
    String,
> {
    let wq_name = hf_name(&format!("blk.{layer}.attn_q.weight"));
    let wq = Matrix::dense(
        src.load_tensor(&wq_name, q_dim * config.hidden_dim)?,
        q_dim,
        config.hidden_dim,
    );

    let wk_name = hf_name(&format!("blk.{layer}.attn_k.weight"));
    let wk = Matrix::dense(
        src.load_tensor(&wk_name, kv_dim * config.hidden_dim)?,
        kv_dim,
        config.hidden_dim,
    );

    let wv_name = hf_name(&format!("blk.{layer}.attn_v.weight"));
    let wv = Matrix::dense(
        src.load_tensor(&wv_name, kv_dim * config.hidden_dim)?,
        kv_dim,
        config.hidden_dim,
    );

    let (bq, bk, bv) = if config.has_qkv_bias {
        let bq = src.try_load_tensor(&hf_name(&format!("blk.{layer}.attn_q.bias")), q_dim);
        let bk = src.try_load_tensor(&hf_name(&format!("blk.{layer}.attn_k.bias")), kv_dim);
        let bv = src.try_load_tensor(&hf_name(&format!("blk.{layer}.attn_v.bias")), kv_dim);
        (bq, bk, bv)
    } else {
        (None, None, None)
    };

    Ok((wq, wk, wv, bq, bk, bv))
}

/// Load and split a fused QKV projection (GPT-2 `c_attn`, Phi `qkv_proj`).
fn load_fused_qkv(
    src: &dyn TensorSource,
    name_map: &WeightNameMap,
    config: &ModelConfig,
    layer: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<
    (
        Matrix,
        Matrix,
        Matrix,
        Option<Vec<f32>>,
        Option<Vec<f32>>,
        Option<Vec<f32>>,
    ),
    String,
> {
    let fused_weight_name = name_map.fused_qkv_weight_name(layer).ok_or_else(|| {
        format!("architecture claims fused QKV but no weight pattern at layer {layer}")
    })?;

    let fused_numel = (q_dim + kv_dim + kv_dim) * config.hidden_dim;
    let fused = src.load_tensor(&fused_weight_name, fused_numel)?;
    let (q_data, k_data, v_data) =
        split_fused_qkv_weight(&fused, q_dim, kv_dim, config.hidden_dim)?;

    let wq = Matrix::dense(q_data, q_dim, config.hidden_dim);
    let wk = Matrix::dense(k_data, kv_dim, config.hidden_dim);
    let wv = Matrix::dense(v_data, kv_dim, config.hidden_dim);

    let (bq, bk, bv) = if config.has_qkv_bias {
        if let Some(bias_name) = name_map.fused_qkv_bias_name(layer) {
            let fused_bias_numel = q_dim + kv_dim + kv_dim;
            match src.try_load_tensor(&bias_name, fused_bias_numel) {
                Some(fused_bias) => {
                    let (bq, bk, bv) = split_fused_qkv_bias(&fused_bias, q_dim, kv_dim)?;
                    (Some(bq), Some(bk), Some(bv))
                }
                None => (None, None, None),
            }
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };

    Ok((wq, wk, wv, bq, bk, bv))
}

/// Load FFN weights using architecture-specific HF names.
///
/// For SwiGLU/GeGLU (gate + up + down), for GELU (fc1 → gate, fc2 → down, up empty).
fn load_ffn_weights_hf(
    src: &dyn TensorSource,
    hf_name: &impl Fn(&str) -> String,
    config: &ModelConfig,
    layer: usize,
) -> Result<(Matrix, Matrix, Matrix), String> {
    let inter = config.intermediate_dim;
    let hidden = config.hidden_dim;

    match config.ffn_type {
        FFNType::SwiGLU | FFNType::GeGLU => {
            let gate_name = hf_name(&format!("blk.{layer}.ffn_gate.weight"));
            let up_name = hf_name(&format!("blk.{layer}.ffn_up.weight"));
            let down_name = hf_name(&format!("blk.{layer}.ffn_down.weight"));

            let w_gate = Matrix::dense(src.load_tensor(&gate_name, inter * hidden)?, inter, hidden);
            let w_up = Matrix::dense(src.load_tensor(&up_name, inter * hidden)?, inter, hidden);
            let w_down = Matrix::dense(src.load_tensor(&down_name, hidden * inter)?, hidden, inter);
            Ok((w_gate, w_up, w_down))
        }
        FFNType::GELU => {
            // GELU architectures (GPT-2, Phi) use fc1/fc2 naming, mapped to
            // ffn_gate/ffn_down in the internal name table.
            let fc1_name = hf_name(&format!("blk.{layer}.ffn_gate.weight"));
            let fc2_name = hf_name(&format!("blk.{layer}.ffn_down.weight"));

            let w_gate = Matrix::dense(src.load_tensor(&fc1_name, inter * hidden)?, inter, hidden);
            // w_up is unused for GELU FFN; keep as empty placeholder for struct consistency.
            let w_up = Matrix::dense(Vec::new(), 0, hidden);
            let w_down = Matrix::dense(src.load_tensor(&fc2_name, hidden * inter)?, hidden, inter);
            Ok((w_gate, w_up, w_down))
        }
    }
}

/// Check whether the checkpoint actually contains Q projection bias tensors.
///
/// Some architectures specify QKV bias in their profile but individual checkpoints
/// omit them. Probing layer 0 avoids hard errors when loading bias-less models.
fn probe_qkv_bias(st: &nnx_safetensors::SafeTensorsFile, name_map: &WeightNameMap) -> bool {
    // Try architecture-specific layer-0 Q bias name; fall back to Llama HF convention.
    let layer0_q_bias = name_map
        .resolve_layer("attn_q.bias", 0)
        .unwrap_or_else(|| "model.layers.0.self_attn.q_proj.bias".to_string());
    st.tensor_info(&layer0_q_bias).is_some()
}

/// Check whether the checkpoint actually contains output projection bias tensors.
fn probe_output_bias(st: &nnx_safetensors::SafeTensorsFile, name_map: &WeightNameMap) -> bool {
    let layer0_o_bias = name_map
        .resolve_layer("attn_output.bias", 0)
        .unwrap_or_else(|| "model.layers.0.self_attn.o_proj.bias".to_string());
    st.tensor_info(&layer0_o_bias).is_some()
}

/// Load a tensor from SafeTensors, converting to f32.
fn load_st_tensor(
    st: &nnx_safetensors::SafeTensorsFile,
    name: &str,
    expected_numel: usize,
) -> Result<Vec<f32>, String> {
    let view = st
        .tensor_view(name)
        .map_err(|e| format!("failed to load tensor {}: {}", name, e))?;

    let numel = view.shape().numel();
    if numel != expected_numel {
        return Err(format!(
            "tensor {} has {} elements, expected {}",
            name, numel, expected_numel
        ));
    }

    let data = view.as_bytes();
    use nnx_core::dtype::DType;
    match view.dtype() {
        DType::F32 => {
            let mut out = vec![0.0f32; numel];
            for i in 0..numel {
                out[i] = f32::from_le_bytes([
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ]);
            }
            Ok(out)
        }
        DType::F16 => {
            let mut out = vec![0.0f32; numel];
            for i in 0..numel {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                out[i] = half::f16::from_bits(bits).to_f32();
            }
            Ok(out)
        }
        DType::BF16 => {
            let mut out = vec![0.0f32; numel];
            for i in 0..numel {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                out[i] = f32::from_bits((bits as u32) << 16);
            }
            Ok(out)
        }
        other => Err(format!("unsupported dtype {:?} for tensor {}", other, name)),
    }
}

fn try_load_st_tensor(
    st: &nnx_safetensors::SafeTensorsFile,
    name: &str,
    expected_numel: usize,
) -> Option<Vec<f32>> {
    load_st_tensor(st, name, expected_numel).ok()
}

// ============================================================
// Memory budget estimation
// ============================================================

/// Estimate total memory usage for a model (weights + KV cache).
pub fn estimate_memory(gguf: &GGUFFile, config: &ModelConfig) -> usize {
    let mut total = estimate_loaded_weights_memory(gguf, config);
    // KV cache: 2 * num_layers * max_ctx * num_kv_heads * head_dim * 4 bytes
    total += 2
        * config.num_layers
        * config.max_context_length
        * config.num_kv_heads
        * config.head_dim
        * 4;
    total
}

fn estimate_safetensors_memory(
    st: &nnx_safetensors::SafeTensorsFile,
    config: &ModelConfig,
) -> usize {
    let mut total = 0usize;
    for name in st.tensor_names() {
        if let Some(info) = st.tensor_info(name) {
            total += info.shape.numel() * std::mem::size_of::<f32>();
        }
    }
    total += 2
        * config.num_layers
        * config.max_context_length
        * config.num_kv_heads
        * config.head_dim
        * 4;
    total
}

/// Load a GGUF model with memory budget enforcement.
pub fn load_gguf_with_budget(path: &Path, memory_budget: usize) -> Result<Model, String> {
    let gguf = GGUFFile::open(path).map_err(|e| format!("failed to open GGUF: {}", e))?;
    let config = ModelConfig::from_gguf(&gguf.metadata)?;

    if memory_budget > 0 {
        let estimated = estimate_memory(&gguf, &config);
        if estimated > memory_budget {
            return Err(format!(
                "model requires ~{:.1} GB but memory budget is {:.1} GB",
                estimated as f64 / 1e9,
                memory_budget as f64 / 1e9
            ));
        }
    }

    info!(
        "Model: {} ({:?}) — {} layers, hidden={}, heads={}/{}, vocab={}, ctx={}",
        config.architecture,
        config.arch,
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.vocab_size,
        config.max_context_length,
    );
    info!(
        "Config: norm={:?}, ffn={:?}, pos={:?}, block={:?}, qkv_bias={}, out_bias={}",
        config.norm_type,
        config.ffn_type,
        config.pos_encoding,
        config.block_style,
        config.has_qkv_bias,
        config.has_output_bias,
    );
    info!(
        "Estimated params: {:.1}B",
        config.estimated_params() as f64 / 1e9
    );

    let weights = load_weights(&gguf, &config)?;
    Ok(Model::new(config, weights))
}

/// Try multiple tensor names, returning the first one found.
fn load_matrix_tensor_multi_name(
    gguf: &GGUFFile,
    names: &[String],
    rows: usize,
    cols: usize,
    description: &str,
) -> Result<Matrix, String> {
    for name in names {
        if let Ok(data) = load_matrix_tensor(gguf, name, rows, cols) {
            return Ok(data);
        }
    }
    Err(format!(
        "none of the expected tensor names found for {}: {:?}",
        description, names
    ))
}

fn estimate_loaded_weights_memory(gguf: &GGUFFile, config: &ModelConfig) -> usize {
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let mut total = 0usize;

    total += estimate_dense_tensor_bytes(config.vocab_size * config.hidden_dim);
    if matches!(config.pos_encoding, crate::config::PosEncoding::Learned) {
        total += estimate_matrix_storage_bytes(
            gguf,
            "pos_embd.weight",
            config.max_context_length * config.hidden_dim,
        );
    }

    for i in 0..config.num_layers {
        let prefix = format!("blk.{i}");
        total += estimate_dense_tensor_bytes(config.hidden_dim); // attn_norm
        total += estimate_dense_tensor_bytes(config.hidden_dim); // ffn_norm or default ones
        total += estimate_matrix_storage_bytes(
            gguf,
            &format!("{prefix}.attn_q.weight"),
            q_dim * config.hidden_dim,
        );
        total += estimate_matrix_storage_bytes(
            gguf,
            &format!("{prefix}.attn_k.weight"),
            kv_dim * config.hidden_dim,
        );
        total += estimate_matrix_storage_bytes(
            gguf,
            &format!("{prefix}.attn_v.weight"),
            kv_dim * config.hidden_dim,
        );
        total += estimate_matrix_storage_bytes(
            gguf,
            &format!("{prefix}.attn_output.weight"),
            config.hidden_dim * q_dim,
        );

        match config.ffn_type {
            FFNType::SwiGLU | FFNType::GeGLU => {
                total += estimate_matrix_storage_bytes(
                    gguf,
                    &format!("{prefix}.ffn_gate.weight"),
                    config.intermediate_dim * config.hidden_dim,
                );
                total += estimate_matrix_storage_bytes(
                    gguf,
                    &format!("{prefix}.ffn_up.weight"),
                    config.intermediate_dim * config.hidden_dim,
                );
                total += estimate_matrix_storage_bytes(
                    gguf,
                    &format!("{prefix}.ffn_down.weight"),
                    config.hidden_dim * config.intermediate_dim,
                );
            }
            FFNType::GELU => {
                total += estimate_matrix_storage_multi_name_bytes(
                    gguf,
                    &[
                        format!("{prefix}.ffn_up.weight"),
                        format!("{prefix}.ffn_gate.weight"),
                    ],
                    config.intermediate_dim * config.hidden_dim,
                );
                total += estimate_matrix_storage_multi_name_bytes(
                    gguf,
                    &[format!("{prefix}.ffn_down.weight")],
                    config.hidden_dim * config.intermediate_dim,
                );
            }
        }

        if config.has_qkv_bias {
            total +=
                estimate_optional_dense_tensor_bytes(gguf, &format!("{prefix}.attn_q.bias"), q_dim);
            total += estimate_optional_dense_tensor_bytes(
                gguf,
                &format!("{prefix}.attn_k.bias"),
                kv_dim,
            );
            total += estimate_optional_dense_tensor_bytes(
                gguf,
                &format!("{prefix}.attn_v.bias"),
                kv_dim,
            );
        }
        if config.has_output_bias {
            total += estimate_optional_dense_tensor_bytes(
                gguf,
                &format!("{prefix}.attn_output.bias"),
                config.hidden_dim,
            );
        }
        if matches!(config.norm_type, NormType::LayerNorm) {
            total += estimate_optional_dense_tensor_bytes(
                gguf,
                &format!("{prefix}.attn_norm.bias"),
                config.hidden_dim,
            );
            total += estimate_optional_dense_tensor_bytes(
                gguf,
                &format!("{prefix}.ffn_norm.bias"),
                config.hidden_dim,
            );
        }
    }

    total += estimate_dense_tensor_bytes(config.hidden_dim); // output_norm.weight
    if matches!(config.norm_type, NormType::LayerNorm) {
        total += estimate_optional_dense_tensor_bytes(gguf, "output_norm.bias", config.hidden_dim);
    }

    if gguf.tensors.contains_key("output.weight") {
        total += estimate_matrix_storage_bytes(
            gguf,
            "output.weight",
            config.vocab_size * config.hidden_dim,
        );
    } else {
        total += estimate_dense_tensor_bytes(config.vocab_size * config.hidden_dim);
    }

    total
}

fn estimate_dense_tensor_bytes(numel: usize) -> usize {
    numel * std::mem::size_of::<f32>()
}

fn estimate_optional_dense_tensor_bytes(gguf: &GGUFFile, name: &str, numel: usize) -> usize {
    if gguf.tensors.contains_key(name) {
        estimate_dense_tensor_bytes(numel)
    } else {
        0
    }
}

fn estimate_matrix_storage_bytes(gguf: &GGUFFile, name: &str, numel: usize) -> usize {
    let Some(info) = gguf.tensors.get(name) else {
        return 0;
    };

    match info.dtype {
        GGMLType::F32 => estimate_dense_tensor_bytes(numel),
        _ => gguf
            .tensor_data(name)
            .map(|data| data.len())
            .unwrap_or_else(|_| estimate_dense_tensor_bytes(numel)),
    }
}

fn estimate_matrix_storage_multi_name_bytes(
    gguf: &GGUFFile,
    names: &[String],
    numel: usize,
) -> usize {
    for name in names {
        if gguf.tensors.contains_key(name) {
            return estimate_matrix_storage_bytes(gguf, name, numel);
        }
    }
    0
}

fn json_usize(value: &serde_json::Value, key: &str) -> Option<usize> {
    value.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

fn json_f32(value: &serde_json::Value, key: &str) -> Option<f32> {
    value.get(key).and_then(|v| v.as_f64()).map(|v| v as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
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
            tensor_data.extend(std::iter::repeat_n(0u8, size));
        }

        let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let header_bytes = header_json.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_bytes);
        bytes.extend_from_slice(&tensor_data);
        bytes
    }

    fn write_test_file(path: &Path, data: &[u8]) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(data).unwrap();
        file.sync_all().unwrap();
    }

    #[test]
    fn test_infer_config_from_adjacent_config() {
        let temp_dir = std::env::temp_dir().join(format!(
            "nnx_loader_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let st_path = temp_dir.join("model.safetensors");
        let config_path = temp_dir.join("config.json");

        let tensors = [
            ("model.embed_tokens.weight", "F32", [16usize, 8].as_slice()),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "F32",
                [8usize, 8].as_slice(),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "F32",
                [8usize, 8].as_slice(),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                [16usize, 8].as_slice(),
            ),
            (
                "model.layers.0.self_attn.q_proj.bias",
                "F32",
                [8usize].as_slice(),
            ),
        ];
        write_test_file(&st_path, &build_safetensors_bytes(&tensors));

        let config_json = serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5
        });
        write_test_file(
            &config_path,
            serde_json::to_string(&config_json).unwrap().as_bytes(),
        );

        let st = nnx_safetensors::SafeTensorsFile::open(&st_path).unwrap();
        let config = infer_config_from_adjacent_config(&st_path, &st)
            .unwrap()
            .unwrap();

        assert_eq!(config.arch, Architecture::Qwen);
        assert!(config.has_qkv_bias);
        assert_eq!(config.num_layers, 1);
        assert_eq!(config.vocab_size, 16);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_load_safetensors_with_budget_rejects_small_budget() {
        let temp_dir = std::env::temp_dir().join(format!(
            "nnx_loader_budget_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let st_path = temp_dir.join("model.safetensors");
        let config_path = temp_dir.join("config.json");

        let tensors = [
            ("model.embed_tokens.weight", "F32", [16usize, 8].as_slice()),
            (
                "model.layers.0.input_layernorm.weight",
                "F32",
                [8usize].as_slice(),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "F32",
                [8usize].as_slice(),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "F32",
                [8usize, 8].as_slice(),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "F32",
                [8usize, 8].as_slice(),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "F32",
                [8usize, 8].as_slice(),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "F32",
                [8usize, 8].as_slice(),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                [16usize, 8].as_slice(),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "F32",
                [16usize, 8].as_slice(),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "F32",
                [8usize, 16].as_slice(),
            ),
            ("model.norm.weight", "F32", [8usize].as_slice()),
            ("lm_head.weight", "F32", [16usize, 8].as_slice()),
        ];
        write_test_file(&st_path, &build_safetensors_bytes(&tensors));

        let config_json = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5
        });
        write_test_file(
            &config_path,
            serde_json::to_string(&config_json).unwrap().as_bytes(),
        );

        let err = load_safetensors_with_budget(&st_path, 1)
            .err()
            .expect("small budget should reject the model");
        assert!(err.contains("memory budget"));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_load_gpt2_safetensors_includes_position_embeddings() {
        let temp_dir = std::env::temp_dir().join(format!(
            "nnx_loader_gpt2_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let st_path = temp_dir.join("model.safetensors");
        let config_path = temp_dir.join("config.json");

        let tensors = [
            ("transformer.wte.weight", "F32", [16usize, 8].as_slice()),
            ("transformer.wpe.weight", "F32", [32usize, 8].as_slice()),
            ("transformer.h.0.ln_1.weight", "F32", [8usize].as_slice()),
            ("transformer.h.0.ln_1.bias", "F32", [8usize].as_slice()),
            ("transformer.h.0.ln_2.weight", "F32", [8usize].as_slice()),
            ("transformer.h.0.ln_2.bias", "F32", [8usize].as_slice()),
            (
                "transformer.h.0.attn.c_attn.weight",
                "F32",
                [24usize, 8].as_slice(),
            ),
            (
                "transformer.h.0.attn.c_attn.bias",
                "F32",
                [24usize].as_slice(),
            ),
            (
                "transformer.h.0.attn.c_proj.weight",
                "F32",
                [8usize, 8].as_slice(),
            ),
            (
                "transformer.h.0.attn.c_proj.bias",
                "F32",
                [8usize].as_slice(),
            ),
            (
                "transformer.h.0.mlp.c_fc.weight",
                "F32",
                [16usize, 8].as_slice(),
            ),
            (
                "transformer.h.0.mlp.c_proj.weight",
                "F32",
                [8usize, 16].as_slice(),
            ),
            ("transformer.ln_f.weight", "F32", [8usize].as_slice()),
            ("transformer.ln_f.bias", "F32", [8usize].as_slice()),
            ("lm_head.weight", "F32", [16usize, 8].as_slice()),
        ];
        write_test_file(&st_path, &build_safetensors_bytes(&tensors));

        let config_json = serde_json::json!({
            "architectures": ["GPT2LMHeadModel"],
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "n_head": 2,
            "n_embd": 8,
            "n_layer": 1,
            "n_positions": 32,
            "n_ctx": 32,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "layer_norm_epsilon": 1e-5
        });
        write_test_file(
            &config_path,
            serde_json::to_string(&config_json).unwrap().as_bytes(),
        );

        let model = load_safetensors_with_budget(&st_path, 0).expect("GPT-2 load should succeed");
        assert!(
            model.weights.position_embedding.is_some(),
            "GPT-2 checkpoints should load learned position embeddings"
        );

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
