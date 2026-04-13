//! Load a complete model from a GGUF file.
//!
//! Parses the GGUF file, extracts config from metadata, dequantizes all
//! weight tensors, and assembles a ready-to-run Model.
//!
//! Supports architecture-specific tensor names and optional bias loading.

use crate::block::BlockWeights;
use crate::config::{Architecture, FFNType, ModelConfig, NormType};
use crate::model::{Model, ModelWeights};
use nnx_gguf::GGUFFile;
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
        config.architecture, config.arch, config.num_layers, config.hidden_dim,
        config.num_heads, config.num_kv_heads, config.vocab_size,
        config.max_context_length,
    );
    info!(
        "Config: norm={:?}, ffn={:?}, pos={:?}, block={:?}, qkv_bias={}, out_bias={}",
        config.norm_type, config.ffn_type, config.pos_encoding,
        config.block_style, config.has_qkv_bias, config.has_output_bias,
    );
    info!("Estimated params: {:.1}B", config.estimated_params() as f64 / 1e9);

    let weights = load_weights(&gguf, &config)?;
    Ok(Model::new(config, weights))
}

fn load_weights(gguf: &GGUFFile, config: &ModelConfig) -> Result<ModelWeights, String> {
    // Token embedding
    let token_embedding = load_tensor(gguf, "token_embd.weight",
        config.vocab_size * config.hidden_dim)?;

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
        NormType::LayerNorm => {
            try_load_tensor(gguf, "output_norm.bias", config.hidden_dim)
        }
        NormType::RMSNorm => None,
    };

    // LM head -- sometimes tied with token_embd.weight
    let lm_head = if gguf.tensors.contains_key("output.weight") {
        load_tensor(gguf, "output.weight", config.vocab_size * config.hidden_dim)?
    } else {
        warn!("output.weight not found, using token_embd.weight as LM head (tied embeddings)");
        token_embedding.clone()
    };

    Ok(ModelWeights {
        token_embedding,
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
    let attn_norm = load_tensor(gguf, &format!("{prefix}.attn_norm.weight"),
        config.hidden_dim)?;
    let ffn_norm = load_tensor_or_default(gguf, &format!("{prefix}.ffn_norm.weight"),
        config.hidden_dim, 1.0);

    // Q, K, V, O projections
    let wq = load_tensor(gguf, &format!("{prefix}.attn_q.weight"),
        q_dim * config.hidden_dim)?;
    let wk = load_tensor(gguf, &format!("{prefix}.attn_k.weight"),
        kv_dim * config.hidden_dim)?;
    let wv = load_tensor(gguf, &format!("{prefix}.attn_v.weight"),
        kv_dim * config.hidden_dim)?;
    let wo = load_tensor(gguf, &format!("{prefix}.attn_output.weight"),
        config.hidden_dim * q_dim)?;

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
        try_load_tensor(gguf, &format!("{prefix}.attn_output.bias"), config.hidden_dim)
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
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let inter = config.intermediate_dim;
    let hidden = config.hidden_dim;

    match config.ffn_type {
        FFNType::SwiGLU | FFNType::GeGLU => {
            let w_gate = load_tensor(gguf, &format!("{prefix}.ffn_gate.weight"),
                inter * hidden)?;
            let w_up = load_tensor(gguf, &format!("{prefix}.ffn_up.weight"),
                inter * hidden)?;
            let w_down = load_tensor(gguf, &format!("{prefix}.ffn_down.weight"),
                hidden * inter)?;
            Ok((w_gate, w_up, w_down))
        }
        FFNType::GELU => {
            // GPT-2 uses ffn_up/ffn_down or fc1/fc2 naming
            // Try multiple name patterns
            let w_fc1 = load_tensor_multi_name(
                gguf,
                &[
                    format!("{prefix}.ffn_up.weight"),
                    format!("{prefix}.ffn_gate.weight"),
                ],
                inter * hidden,
                "fc1/ffn_up",
            )?;
            let w_fc2 = load_tensor_multi_name(
                gguf,
                &[
                    format!("{prefix}.ffn_down.weight"),
                ],
                hidden * inter,
                "fc2/ffn_down",
            )?;
            // w_up is unused for GELU FFN, but we keep it for struct consistency
            let w_up = Vec::new();
            Ok((w_fc1, w_up, w_fc2))
        }
    }
}

/// Load and dequantize a single tensor from GGUF.
fn load_tensor(gguf: &GGUFFile, name: &str, expected_numel: usize) -> Result<Vec<f32>, String> {
    let info = gguf.tensors.get(name)
        .ok_or_else(|| format!("tensor not found in GGUF: {}", name))?;

    let numel = info.numel() as usize;
    if numel != expected_numel {
        return Err(format!(
            "tensor {} has {} elements, expected {}",
            name, numel, expected_numel
        ));
    }

    let data = gguf.tensor_data(name)
        .map_err(|e| format!("failed to read tensor {}: {}", name, e))?;

    let output = dequantize_alloc(data, info.dtype, numel);
    if output.iter().take(numel).all(|&v| v == 0.0) && numel > 0 {
        warn!("tensor {} dequantized to all zeros (dtype: {})", name, info.dtype);
    }

    Ok(output)
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
    use nnx_safetensors::SafeTensorsFile;

    info!("Loading SafeTensors model from {}", path.display());

    let st = SafeTensorsFile::open(path)
        .map_err(|e| format!("failed to open SafeTensors: {}", e))?;

    let config = infer_config_from_safetensors(&st)?;

    info!(
        "Model (inferred): {} — {} layers, hidden={}, heads={}/{}, vocab={}",
        config.architecture, config.num_layers, config.hidden_dim,
        config.num_heads, config.num_kv_heads, config.vocab_size,
    );

    let weights = load_safetensors_weights(&st, &config)?;
    Ok(Model::new(config, weights))
}

fn infer_config_from_safetensors(st: &nnx_safetensors::SafeTensorsFile) -> Result<ModelConfig, String> {
    let embed_info = st.tensor_info("model.embed_tokens.weight")
        .ok_or("missing model.embed_tokens.weight")?;
    let vocab_size = embed_info.shape.dims()[0];
    let hidden_dim = embed_info.shape.dims()[1];

    let mut num_layers = 0;
    loop {
        let name = format!("model.layers.{}.self_attn.q_proj.weight", num_layers);
        if st.tensor_info(&name).is_none() { break; }
        num_layers += 1;
    }
    if num_layers == 0 {
        return Err("no transformer layers found".into());
    }

    let q_info = st.tensor_info("model.layers.0.self_attn.q_proj.weight")
        .ok_or("missing q_proj")?;
    let k_info = st.tensor_info("model.layers.0.self_attn.k_proj.weight")
        .ok_or("missing k_proj")?;
    let q_dim = q_info.shape.dims()[0];
    let kv_dim = k_info.shape.dims()[0];

    // Infer head_dim from standard values
    let head_dim = [128, 64, 32, 16, 8, 4].into_iter()
        .find(|&hd| q_dim % hd == 0 && kv_dim % hd == 0 && hidden_dim % hd == 0)
        .unwrap_or(64);
    let num_heads = q_dim / head_dim;
    let num_kv_heads = kv_dim / head_dim;

    let gate_info = st.tensor_info("model.layers.0.mlp.gate_proj.weight")
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

fn load_safetensors_weights(
    st: &nnx_safetensors::SafeTensorsFile,
    config: &ModelConfig,
) -> Result<ModelWeights, String> {
    let token_embedding = load_st_tensor(st, "model.embed_tokens.weight",
        config.vocab_size * config.hidden_dim)?;

    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("model.layers.{i}");
        let layer = BlockWeights {
            attn_norm: load_st_tensor(st, &format!("{prefix}.input_layernorm.weight"),
                config.hidden_dim)?,
            ffn_norm: load_st_tensor(st, &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_dim)?,
            wq: load_st_tensor(st, &format!("{prefix}.self_attn.q_proj.weight"),
                q_dim * config.hidden_dim)?,
            wk: load_st_tensor(st, &format!("{prefix}.self_attn.k_proj.weight"),
                kv_dim * config.hidden_dim)?,
            wv: load_st_tensor(st, &format!("{prefix}.self_attn.v_proj.weight"),
                kv_dim * config.hidden_dim)?,
            wo: load_st_tensor(st, &format!("{prefix}.self_attn.o_proj.weight"),
                config.hidden_dim * q_dim)?,
            w_gate: load_st_tensor(st, &format!("{prefix}.mlp.gate_proj.weight"),
                config.intermediate_dim * config.hidden_dim)?,
            w_up: load_st_tensor(st, &format!("{prefix}.mlp.up_proj.weight"),
                config.intermediate_dim * config.hidden_dim)?,
            w_down: load_st_tensor(st, &format!("{prefix}.mlp.down_proj.weight"),
                config.hidden_dim * config.intermediate_dim)?,
            bq: None, bk: None, bv: None, bo: None,
            attn_norm_bias: None, ffn_norm_bias: None,
        };
        layers.push(layer);
        if (i + 1) % 10 == 0 || i == config.num_layers - 1 {
            info!("Loaded layer {}/{}", i + 1, config.num_layers);
        }
    }

    let final_norm = load_st_tensor(st, "model.norm.weight", config.hidden_dim)?;
    let lm_head = if st.tensor_info("lm_head.weight").is_some() {
        load_st_tensor(st, "lm_head.weight", config.vocab_size * config.hidden_dim)?
    } else {
        warn!("lm_head.weight not found, using embed_tokens (tied embeddings)");
        token_embedding.clone()
    };

    Ok(ModelWeights {
        token_embedding,
        layers,
        final_norm,
        final_norm_bias: None,
        lm_head,
    })
}

/// Load a tensor from SafeTensors, converting to f32.
fn load_st_tensor(
    st: &nnx_safetensors::SafeTensorsFile,
    name: &str,
    expected_numel: usize,
) -> Result<Vec<f32>, String> {
    let view = st.tensor_view(name)
        .map_err(|e| format!("failed to load tensor {}: {}", name, e))?;

    let numel = view.shape().numel();
    if numel != expected_numel {
        return Err(format!("tensor {} has {} elements, expected {}", name, numel, expected_numel));
    }

    let data = view.as_bytes();
    use nnx_core::dtype::DType;
    match view.dtype() {
        DType::F32 => {
            let mut out = vec![0.0f32; numel];
            for i in 0..numel {
                out[i] = f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]);
            }
            Ok(out)
        }
        DType::F16 => {
            let mut out = vec![0.0f32; numel];
            for i in 0..numel {
                let bits = u16::from_le_bytes([data[i*2], data[i*2+1]]);
                out[i] = half::f16::from_bits(bits).to_f32();
            }
            Ok(out)
        }
        DType::BF16 => {
            let mut out = vec![0.0f32; numel];
            for i in 0..numel {
                let bits = u16::from_le_bytes([data[i*2], data[i*2+1]]);
                out[i] = f32::from_bits((bits as u32) << 16);
            }
            Ok(out)
        }
        other => Err(format!("unsupported dtype {:?} for tensor {}", other, name)),
    }
}

// ============================================================
// Memory budget estimation
// ============================================================

/// Estimate total memory usage for a model (weights + KV cache).
pub fn estimate_memory(gguf: &GGUFFile, config: &ModelConfig) -> usize {
    let mut total = 0usize;
    // All tensors dequantized to f32
    for (_name, info) in &gguf.tensors {
        total += info.numel() as usize * 4;
    }
    // KV cache: 2 * num_layers * max_ctx * num_kv_heads * head_dim * 4 bytes
    total += 2 * config.num_layers * config.max_context_length
        * config.num_kv_heads * config.head_dim * 4;
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
                estimated as f64 / 1e9, memory_budget as f64 / 1e9
            ));
        }
    }

    info!(
        "Model: {} ({:?}) — {} layers, hidden={}, heads={}/{}, vocab={}, ctx={}",
        config.architecture, config.arch, config.num_layers, config.hidden_dim,
        config.num_heads, config.num_kv_heads, config.vocab_size,
        config.max_context_length,
    );
    info!(
        "Config: norm={:?}, ffn={:?}, pos={:?}, block={:?}, qkv_bias={}, out_bias={}",
        config.norm_type, config.ffn_type, config.pos_encoding,
        config.block_style, config.has_qkv_bias, config.has_output_bias,
    );
    info!("Estimated params: {:.1}B", config.estimated_params() as f64 / 1e9);

    let weights = load_weights(&gguf, &config)?;
    Ok(Model::new(config, weights))
}

/// Try multiple tensor names, returning the first one found.
fn load_tensor_multi_name(
    gguf: &GGUFFile,
    names: &[String],
    expected_numel: usize,
    description: &str,
) -> Result<Vec<f32>, String> {
    for name in names {
        if let Ok(data) = load_tensor(gguf, name, expected_numel) {
            return Ok(data);
        }
    }
    Err(format!(
        "none of the expected tensor names found for {}: {:?}",
        description, names
    ))
}
