//! Load a complete model from a GGUF file.
//!
//! Parses the GGUF file, extracts config from metadata, dequantizes all
//! weight tensors, and assembles a ready-to-run Model.

use crate::block::BlockWeights;
use crate::config::ModelConfig;
use crate::model::{Model, ModelWeights};
use nnx_gguf::GGUFFile;
use nnx_quant::dequant::dequantize_alloc;
use std::path::Path;
use tracing::{info, warn};

/// Load a model from a GGUF file path.
///
/// This dequantizes all weights to f32. For large models this uses
/// significant memory — quantized-in-memory compute is a future optimization.
pub fn load_gguf(path: &Path) -> Result<Model, String> {
    info!("Loading GGUF model from {}", path.display());

    let gguf = GGUFFile::open(path).map_err(|e| format!("failed to open GGUF: {}", e))?;
    let config = ModelConfig::from_gguf(&gguf.metadata)?;

    info!(
        "Model: {} — {} layers, hidden={}, heads={}/{}, vocab={}, ctx={}",
        config.architecture, config.num_layers, config.hidden_dim,
        config.num_heads, config.num_kv_heads, config.vocab_size,
        config.max_context_length,
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
        let layer = BlockWeights {
            attn_norm: load_tensor(gguf, &format!("{prefix}.attn_norm.weight"),
                config.hidden_dim)?,
            ffn_norm: load_tensor(gguf, &format!("{prefix}.ffn_norm.weight"),
                config.hidden_dim)?,
            wq: load_tensor(gguf, &format!("{prefix}.attn_q.weight"),
                q_dim * config.hidden_dim)?,
            wk: load_tensor(gguf, &format!("{prefix}.attn_k.weight"),
                kv_dim * config.hidden_dim)?,
            wv: load_tensor(gguf, &format!("{prefix}.attn_v.weight"),
                kv_dim * config.hidden_dim)?,
            wo: load_tensor(gguf, &format!("{prefix}.attn_output.weight"),
                config.hidden_dim * q_dim)?,
            w_gate: load_tensor(gguf, &format!("{prefix}.ffn_gate.weight"),
                config.intermediate_dim * config.hidden_dim)?,
            w_up: load_tensor(gguf, &format!("{prefix}.ffn_up.weight"),
                config.intermediate_dim * config.hidden_dim)?,
            w_down: load_tensor(gguf, &format!("{prefix}.ffn_down.weight"),
                config.hidden_dim * config.intermediate_dim)?,
        };
        layers.push(layer);

        if (i + 1) % 10 == 0 || i == config.num_layers - 1 {
            info!("Loaded layer {}/{}", i + 1, config.num_layers);
        }
    }

    // Final norm
    let final_norm = load_tensor(gguf, "output_norm.weight", config.hidden_dim)?;

    // LM head — sometimes tied with token_embd.weight
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
        lm_head,
    })
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
