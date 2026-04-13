//! Transformer block: RMSNorm → Attention → Residual → RMSNorm → FFN → Residual.

use crate::attention;
use crate::cache::LayerCache;
use crate::ffn;

/// Weights for one transformer block.
pub struct BlockWeights {
    /// Attention norm weights [hidden_dim]
    pub attn_norm: Vec<f32>,
    /// FFN norm weights [hidden_dim]
    pub ffn_norm: Vec<f32>,
    /// Query projection [num_heads * head_dim, hidden_dim]
    pub wq: Vec<f32>,
    /// Key projection [num_kv_heads * head_dim, hidden_dim]
    pub wk: Vec<f32>,
    /// Value projection [num_kv_heads * head_dim, hidden_dim]
    pub wv: Vec<f32>,
    /// Output projection [hidden_dim, num_heads * head_dim]
    pub wo: Vec<f32>,
    /// Gate projection [intermediate_dim, hidden_dim]
    pub w_gate: Vec<f32>,
    /// Up projection [intermediate_dim, hidden_dim]
    pub w_up: Vec<f32>,
    /// Down projection [hidden_dim, intermediate_dim]
    pub w_down: Vec<f32>,
}

/// Run a single transformer block on one token during decoding.
///
/// Modifies `hidden` in-place (residual connections).
pub fn forward_block(
    hidden: &mut [f32],
    weights: &BlockWeights,
    cache: &mut LayerCache,
    position: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    rope_freq_base: f32,
    rms_norm_eps: f32,
) {
    let hidden_dim = hidden.len();

    // --- Attention sub-layer ---
    // 1. RMSNorm
    let mut normed = vec![0.0f32; hidden_dim];
    nnx_kernels::rms_norm::rms_norm_f32(hidden, &weights.attn_norm, &mut normed, rms_norm_eps);

    // 2. Attention
    let attn_out = attention::attention_decode(
        &normed,
        &weights.wq, &weights.wk, &weights.wv, &weights.wo,
        cache, position,
        num_heads, num_kv_heads, head_dim, rope_freq_base,
    );

    // 3. Residual
    nnx_kernels::activations::add_f32_inplace(hidden, &attn_out);

    // --- FFN sub-layer ---
    // 4. RMSNorm
    nnx_kernels::rms_norm::rms_norm_f32(hidden, &weights.ffn_norm, &mut normed, rms_norm_eps);

    // 5. SwiGLU FFN
    let ffn_out = ffn::swiglu_ffn(
        &normed,
        &weights.w_gate, &weights.w_up, &weights.w_down,
        hidden_dim, intermediate_dim,
    );

    // 6. Residual
    nnx_kernels::activations::add_f32_inplace(hidden, &ffn_out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_smoke() {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let weights = BlockWeights {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
            wq: vec![0.01; num_heads * head_dim * hidden_dim],
            wk: vec![0.01; num_kv_heads * head_dim * hidden_dim],
            wv: vec![0.01; num_kv_heads * head_dim * hidden_dim],
            wo: vec![0.01; hidden_dim * num_heads * head_dim],
            w_gate: vec![0.01; intermediate_dim * hidden_dim],
            w_up: vec![0.01; intermediate_dim * hidden_dim],
            w_down: vec![0.01; hidden_dim * intermediate_dim],
        };

        let mut hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        forward_block(
            &mut hidden, &weights, &mut cache, 0,
            num_heads, num_kv_heads, head_dim, intermediate_dim,
            10000.0, 1e-5,
        );

        assert!(hidden.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 1);
    }
}
