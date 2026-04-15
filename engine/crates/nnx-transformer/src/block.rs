//! Transformer block: architecture-aware pre-norm, attention, FFN, and residual.
//!
//! Supports sequential (Llama, GPT-2, Gemma, Qwen) and parallel (Phi) block styles.

use crate::attention;
use crate::config::{BlockStyle, ModelConfig, NormType};
use nnx_core::engine::KVStore;
use crate::ffn;
use crate::weights::Matrix;
use nnx_core::error::Result;

/// Weights for one transformer block.
pub struct BlockWeights {
    /// Attention norm weights [hidden_dim]
    pub attn_norm: Vec<f32>,
    /// FFN norm weights [hidden_dim] (unused in Parallel block style)
    pub ffn_norm: Vec<f32>,
    /// Query projection [num_heads * head_dim, hidden_dim]
    pub wq: Matrix,
    /// Key projection [num_kv_heads * head_dim, hidden_dim]
    pub wk: Matrix,
    /// Value projection [num_kv_heads * head_dim, hidden_dim]
    pub wv: Matrix,
    /// Output projection [hidden_dim, num_heads * head_dim]
    pub wo: Matrix,
    /// Gate projection [intermediate_dim, hidden_dim] (SwiGLU/GeGLU), or fc1 for GELU
    pub w_gate: Matrix,
    /// Up projection [intermediate_dim, hidden_dim] (SwiGLU/GeGLU; empty for GELU)
    pub w_up: Matrix,
    /// Down projection [hidden_dim, intermediate_dim] (SwiGLU/GeGLU), or fc2 for GELU
    pub w_down: Matrix,
    // --- Optional bias terms (architecture-dependent) ---
    /// Q bias [num_heads * head_dim] (GPT-2, Phi, Qwen)
    pub bq: Option<Vec<f32>>,
    /// K bias [num_kv_heads * head_dim] (GPT-2, Phi, Qwen)
    pub bk: Option<Vec<f32>>,
    /// V bias [num_kv_heads * head_dim] (GPT-2, Phi, Qwen)
    pub bv: Option<Vec<f32>>,
    /// Output projection bias [hidden_dim] (GPT-2, Phi)
    pub bo: Option<Vec<f32>>,
    /// Attention LayerNorm bias [hidden_dim] (GPT-2, Phi; None for RMSNorm)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// FFN LayerNorm bias [hidden_dim] (GPT-2; None for RMSNorm or parallel block style)
    pub ffn_norm_bias: Option<Vec<f32>>,
}

impl BlockWeights {
    /// Create weights with no bias terms (Llama-compatible).
    #[cfg(test)]
    pub(crate) fn test_no_bias(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        Self {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
            wq: Matrix::dense(
                vec![0.01; num_heads * head_dim * hidden_dim],
                num_heads * head_dim,
                hidden_dim,
            ),
            wk: Matrix::dense(
                vec![0.01; num_kv_heads * head_dim * hidden_dim],
                num_kv_heads * head_dim,
                hidden_dim,
            ),
            wv: Matrix::dense(
                vec![0.01; num_kv_heads * head_dim * hidden_dim],
                num_kv_heads * head_dim,
                hidden_dim,
            ),
            wo: Matrix::dense(
                vec![0.01; hidden_dim * num_heads * head_dim],
                hidden_dim,
                num_heads * head_dim,
            ),
            w_gate: Matrix::dense(
                vec![0.01; intermediate_dim * hidden_dim],
                intermediate_dim,
                hidden_dim,
            ),
            w_up: Matrix::dense(
                vec![0.01; intermediate_dim * hidden_dim],
                intermediate_dim,
                hidden_dim,
            ),
            w_down: Matrix::dense(
                vec![0.01; hidden_dim * intermediate_dim],
                hidden_dim,
                intermediate_dim,
            ),
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            attn_norm_bias: None,
            ffn_norm_bias: None,
        }
    }

    /// Create weights with bias terms for all Q/K/V/O projections and norms.
    #[cfg(test)]
    pub(crate) fn test_with_bias(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Self {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
            wq: Matrix::dense(vec![0.01; q_dim * hidden_dim], q_dim, hidden_dim),
            wk: Matrix::dense(vec![0.01; kv_dim * hidden_dim], kv_dim, hidden_dim),
            wv: Matrix::dense(vec![0.01; kv_dim * hidden_dim], kv_dim, hidden_dim),
            wo: Matrix::dense(vec![0.01; hidden_dim * q_dim], hidden_dim, q_dim),
            w_gate: Matrix::dense(
                vec![0.01; intermediate_dim * hidden_dim],
                intermediate_dim,
                hidden_dim,
            ),
            w_up: Matrix::dense(
                vec![0.01; intermediate_dim * hidden_dim],
                intermediate_dim,
                hidden_dim,
            ),
            w_down: Matrix::dense(
                vec![0.01; hidden_dim * intermediate_dim],
                hidden_dim,
                intermediate_dim,
            ),
            bq: Some(vec![0.001; q_dim]),
            bk: Some(vec![0.001; kv_dim]),
            bv: Some(vec![0.001; kv_dim]),
            bo: Some(vec![0.001; hidden_dim]),
            attn_norm_bias: Some(vec![0.0; hidden_dim]),
            ffn_norm_bias: Some(vec![0.0; hidden_dim]),
        }
    }
}

/// Apply pre-normalization to hidden state, writing result to `normed`.
fn apply_norm(
    hidden: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    normed: &mut [f32],
    norm_type: &NormType,
    eps: f32,
) -> Result<()> {
    let hidden_dim = hidden.len();
    match norm_type {
        NormType::RMSNorm => {
            nnx_kernels::rms_norm::rms_norm_f32_checked(hidden, weight, normed, eps)?;
        }
        NormType::LayerNorm => {
            nnx_kernels::normalization::layer_norm_f32_checked(
                hidden, weight, bias, normed, hidden_dim, eps,
            )?;
        }
    }
    Ok(())
}

fn apply_norm_batch(
    hidden_batch: &[f32],
    batch_size: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    normed_batch: &mut [f32],
    norm_type: &NormType,
    eps: f32,
) -> Result<()> {
    let hidden_dim = weight.len();
    debug_assert_eq!(hidden_batch.len(), batch_size * hidden_dim);
    debug_assert_eq!(normed_batch.len(), batch_size * hidden_dim);

    for batch_idx in 0..batch_size {
        let hidden = &hidden_batch[batch_idx * hidden_dim..(batch_idx + 1) * hidden_dim];
        let normed = &mut normed_batch[batch_idx * hidden_dim..(batch_idx + 1) * hidden_dim];
        apply_norm(hidden, weight, bias, normed, norm_type, eps)?;
    }
    Ok(())
}

/// Run a single transformer block on one token during decoding.
///
/// Modifies `hidden` in-place (residual connections).
/// Dispatches to the correct block style based on `config.block_style`.
pub fn forward_block(
    hidden: &mut [f32],
    weights: &BlockWeights,
    cache: &mut dyn KVStore,
    position: usize,
    config: &ModelConfig,
) -> Result<()> {
    let hidden_dim = hidden.len();
    let mut normed = vec![0.0f32; hidden_dim];

    // Pre-norm for attention
    apply_norm(
        hidden,
        &weights.attn_norm,
        weights.attn_norm_bias.as_deref(),
        &mut normed,
        &config.norm_type,
        config.rms_norm_eps,
    )?;

    match config.block_style {
        BlockStyle::Sequential => {
            // Standard: attn -> residual -> norm -> ffn -> residual
            let attn_out = attention::attention_decode_configurable(
                &normed, weights, cache, position, config,
            )?;
            nnx_kernels::activations::add_f32_inplace(hidden, &attn_out);

            // FFN norm
            apply_norm(
                hidden,
                &weights.ffn_norm,
                weights.ffn_norm_bias.as_deref(),
                &mut normed,
                &config.norm_type,
                config.rms_norm_eps,
            )?;

            let ffn_out = ffn::ffn_forward(&normed, weights, config);
            nnx_kernels::activations::add_f32_inplace(hidden, &ffn_out);
        }
        BlockStyle::Parallel => {
            // Phi-style: attn and ffn both use the same normed input
            let attn_out = attention::attention_decode_configurable(
                &normed, weights, cache, position, config,
            )?;
            let ffn_out = ffn::ffn_forward(&normed, weights, config);

            nnx_kernels::activations::add_f32_inplace(hidden, &attn_out);
            nnx_kernels::activations::add_f32_inplace(hidden, &ffn_out);
        }
    }
    Ok(())
}

/// Run a transformer block across a prompt batch during prefill.
pub fn forward_block_batch(
    hidden_batch: &mut [f32],
    batch_size: usize,
    weights: &BlockWeights,
    cache: &mut dyn KVStore,
    start_position: usize,
    config: &ModelConfig,
) -> Result<()> {
    let hidden_dim = config.hidden_dim;
    let mut normed = vec![0.0f32; batch_size * hidden_dim];

    apply_norm_batch(
        hidden_batch,
        batch_size,
        &weights.attn_norm,
        weights.attn_norm_bias.as_deref(),
        &mut normed,
        &config.norm_type,
        config.rms_norm_eps,
    )?;

    match config.block_style {
        BlockStyle::Sequential => {
            let attn_out = attention::attention_prefill_batch_configurable(
                &normed,
                batch_size,
                weights,
                cache,
                start_position,
                config,
            )?;
            nnx_kernels::activations::add_f32_inplace(hidden_batch, &attn_out);

            apply_norm_batch(
                hidden_batch,
                batch_size,
                &weights.ffn_norm,
                weights.ffn_norm_bias.as_deref(),
                &mut normed,
                &config.norm_type,
                config.rms_norm_eps,
            )?;

            let ffn_out = ffn::ffn_forward_batch(&normed, batch_size, weights, config);
            nnx_kernels::activations::add_f32_inplace(hidden_batch, &ffn_out);
        }
        BlockStyle::Parallel => {
            let attn_out = attention::attention_prefill_batch_configurable(
                &normed,
                batch_size,
                weights,
                cache,
                start_position,
                config,
            )?;
            let ffn_out = ffn::ffn_forward_batch(&normed, batch_size, weights, config);

            nnx_kernels::activations::add_f32_inplace(hidden_batch, &attn_out);
            nnx_kernels::activations::add_f32_inplace(hidden_batch, &ffn_out);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LayerCache;
    use crate::config::*;

    #[test]
    fn test_block_smoke_llama() {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig::test_llama(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            32,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config).unwrap();

        assert!(hidden.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_gpt2_block_forward() {
        // GPT-2: LayerNorm + GELU FFN + full bias
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig {
            architecture: "gpt2".into(),
            arch: Architecture::GPT2,
            num_layers: 1,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: 32,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: NormType::LayerNorm,
            ffn_type: FFNType::GELU,
            pos_encoding: PosEncoding::Learned,
            block_style: BlockStyle::Sequential,
            has_qkv_bias: true,
            has_output_bias: true,
            embedding_scale: None,
        };

        let weights = BlockWeights::test_with_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config).unwrap();

        assert!(hidden.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_phi_parallel_block() {
        // Phi: parallel attn+FFN, LayerNorm, partial RoPE, bias
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig {
            architecture: "phi".into(),
            arch: Architecture::Phi,
            num_layers: 1,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: 32,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: NormType::LayerNorm,
            ffn_type: FFNType::SwiGLU,
            pos_encoding: PosEncoding::PartialRoPE {
                freq_base: 10000.0,
                rotary_dim: 2,
            },
            block_style: BlockStyle::Parallel,
            has_qkv_bias: true,
            has_output_bias: true,
            embedding_scale: None,
        };

        let weights = BlockWeights::test_with_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config).unwrap();

        assert!(hidden.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_gemma_block_forward() {
        // Gemma: RMSNorm + GeGLU
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig {
            architecture: "gemma".into(),
            arch: Architecture::Gemma,
            num_layers: 1,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: 32,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: NormType::RMSNorm,
            ffn_type: FFNType::GeGLU,
            pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
            block_style: BlockStyle::Sequential,
            has_qkv_bias: false,
            has_output_bias: false,
            embedding_scale: Some((8.0f32).sqrt()),
        };

        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config).unwrap();

        assert!(hidden.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_qwen_block_forward() {
        // Qwen: RMSNorm + SwiGLU + QKV bias
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig {
            architecture: "qwen".into(),
            arch: Architecture::Qwen,
            num_layers: 1,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: 32,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: NormType::RMSNorm,
            ffn_type: FFNType::SwiGLU,
            pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
            block_style: BlockStyle::Sequential,
            has_qkv_bias: true,
            has_output_bias: false,
            embedding_scale: None,
        };

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        weights.bq = Some(vec![0.001; q_dim]);
        weights.bk = Some(vec![0.001; kv_dim]);
        weights.bv = Some(vec![0.001; kv_dim]);

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config).unwrap();

        assert!(hidden.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 1);
    }
}
