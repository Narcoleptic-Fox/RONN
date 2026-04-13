//! Transformer block: architecture-aware pre-norm, attention, FFN, and residual.
//!
//! Supports sequential (Llama, GPT-2, Gemma, Qwen) and parallel (Phi) block styles.

use crate::attention;
use crate::cache::LayerCache;
use crate::config::{BlockStyle, ModelConfig, NormType};
use crate::ffn;

/// Weights for one transformer block.
pub struct BlockWeights {
    /// Attention norm weights [hidden_dim]
    pub attn_norm: Vec<f32>,
    /// FFN norm weights [hidden_dim] (unused in Parallel block style)
    pub ffn_norm: Vec<f32>,
    /// Query projection [num_heads * head_dim, hidden_dim]
    pub wq: Vec<f32>,
    /// Key projection [num_kv_heads * head_dim, hidden_dim]
    pub wk: Vec<f32>,
    /// Value projection [num_kv_heads * head_dim, hidden_dim]
    pub wv: Vec<f32>,
    /// Output projection [hidden_dim, num_heads * head_dim]
    pub wo: Vec<f32>,
    /// Gate projection [intermediate_dim, hidden_dim] (SwiGLU/GeGLU), or fc1 for GELU
    pub w_gate: Vec<f32>,
    /// Up projection [intermediate_dim, hidden_dim] (SwiGLU/GeGLU; empty for GELU)
    pub w_up: Vec<f32>,
    /// Down projection [hidden_dim, intermediate_dim] (SwiGLU/GeGLU), or fc2 for GELU
    pub w_down: Vec<f32>,
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
            wq: vec![0.01; num_heads * head_dim * hidden_dim],
            wk: vec![0.01; num_kv_heads * head_dim * hidden_dim],
            wv: vec![0.01; num_kv_heads * head_dim * hidden_dim],
            wo: vec![0.01; hidden_dim * num_heads * head_dim],
            w_gate: vec![0.01; intermediate_dim * hidden_dim],
            w_up: vec![0.01; intermediate_dim * hidden_dim],
            w_down: vec![0.01; hidden_dim * intermediate_dim],
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
            wq: vec![0.01; q_dim * hidden_dim],
            wk: vec![0.01; kv_dim * hidden_dim],
            wv: vec![0.01; kv_dim * hidden_dim],
            wo: vec![0.01; hidden_dim * q_dim],
            w_gate: vec![0.01; intermediate_dim * hidden_dim],
            w_up: vec![0.01; intermediate_dim * hidden_dim],
            w_down: vec![0.01; hidden_dim * intermediate_dim],
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
) {
    let hidden_dim = hidden.len();
    match norm_type {
        NormType::RMSNorm => {
            nnx_kernels::rms_norm::rms_norm_f32(hidden, weight, normed, eps);
        }
        NormType::LayerNorm => {
            nnx_kernels::normalization::layer_norm_f32(
                hidden, weight, bias, normed, hidden_dim, eps,
            );
        }
    }
}

/// Run a single transformer block on one token during decoding.
///
/// Modifies `hidden` in-place (residual connections).
/// Dispatches to the correct block style based on `config.block_style`.
pub fn forward_block(
    hidden: &mut [f32],
    weights: &BlockWeights,
    cache: &mut LayerCache,
    position: usize,
    config: &ModelConfig,
) {
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
    );

    match config.block_style {
        BlockStyle::Sequential => {
            // Standard: attn -> residual -> norm -> ffn -> residual
            let attn_out = attention::attention_decode_configurable(
                &normed, weights, cache, position, config,
            );
            nnx_kernels::activations::add_f32_inplace(hidden, &attn_out);

            // FFN norm
            apply_norm(
                hidden,
                &weights.ffn_norm,
                weights.ffn_norm_bias.as_deref(),
                &mut normed,
                &config.norm_type,
                config.rms_norm_eps,
            );

            let ffn_out = ffn::ffn_forward(&normed, weights, config);
            nnx_kernels::activations::add_f32_inplace(hidden, &ffn_out);
        }
        BlockStyle::Parallel => {
            // Phi-style: attn and ffn both use the same normed input
            let attn_out = attention::attention_decode_configurable(
                &normed, weights, cache, position, config,
            );
            let ffn_out = ffn::ffn_forward(&normed, weights, config);

            nnx_kernels::activations::add_f32_inplace(hidden, &attn_out);
            nnx_kernels::activations::add_f32_inplace(hidden, &ffn_out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    #[test]
    fn test_block_smoke_llama() {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig::test_llama(
            hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim, 32,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config);

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
            hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config);

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
            pos_encoding: PosEncoding::PartialRoPE { freq_base: 10000.0, rotary_dim: 2 },
            block_style: BlockStyle::Parallel,
            has_qkv_bias: true,
            has_output_bias: true,
            embedding_scale: None,
        };

        let weights = BlockWeights::test_with_bias(
            hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config);

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
            hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
        );

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config);

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
            hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
        );
        weights.bq = Some(vec![0.001; q_dim]);
        weights.bk = Some(vec![0.001; kv_dim]);
        weights.bv = Some(vec![0.001; kv_dim]);

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(&mut hidden, &weights, &mut cache, 0, &config);

        assert!(hidden.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 1);
    }
}
