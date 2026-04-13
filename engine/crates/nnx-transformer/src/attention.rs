//! Multi-head attention with Grouped-Query Attention (GQA) support.
//!
//! Llama uses GQA where num_kv_heads < num_heads. Multiple query heads
//! share the same key/value head, reducing KV cache size.

use crate::cache::LayerCache;
use nnx_kernels::rope;
use nnx_kernels::softmax;

/// Compute single-token attention for one layer during decoding.
///
/// This is the decode-step attention: one new query token attends to all
/// cached keys, produces one output vector.
///
/// # Arguments
/// - `hidden`: input hidden state [hidden_dim]
/// - `wq`, `wk`, `wv`, `wo`: weight matrices, row-major, dequantized to f32
///   - wq: [num_heads * head_dim, hidden_dim]
///   - wk: [num_kv_heads * head_dim, hidden_dim]
///   - wv: [num_kv_heads * head_dim, hidden_dim]
///   - wo: [hidden_dim, num_heads * head_dim]
/// - `cache`: KV cache for this layer (keys/values from previous positions)
/// - `position`: current token position in the sequence
/// - Config params: num_heads, num_kv_heads, head_dim, rope_freq_base
///
/// # Returns
/// Output vector [hidden_dim]
pub fn attention_decode(
    hidden: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    cache: &mut LayerCache,
    position: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_freq_base: f32,
) -> Vec<f32> {
    let hidden_dim = hidden.len();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_kv = num_heads / num_kv_heads;

    // Project Q, K, V
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    nnx_kernels::matmul::matvec_f32(wq, hidden, &mut q, q_dim, hidden_dim);
    nnx_kernels::matmul::matvec_f32(wk, hidden, &mut k, kv_dim, hidden_dim);
    nnx_kernels::matmul::matvec_f32(wv, hidden, &mut v, kv_dim, hidden_dim);

    // Apply RoPE to Q and K (per-head)
    for h in 0..num_heads {
        rope::rope_f32(&mut q[h * head_dim..(h + 1) * head_dim], position, rope_freq_base);
    }
    for h in 0..num_kv_heads {
        rope::rope_f32(&mut k[h * head_dim..(h + 1) * head_dim], position, rope_freq_base);
    }

    // Store K, V in cache
    cache.store(&k, &v);
    let seq_len = cache.len(); // includes the just-stored token

    // Compute attention per head
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![0.0f32; q_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv; // which KV head this query head uses
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        // Compute attention scores: score[pos] = q · k[pos] * scale
        let mut scores = vec![0.0f32; seq_len];
        for pos in 0..seq_len {
            let k_pos = cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_pos[d];
            }
            scores[pos] = dot * scale;
        }

        // Softmax over scores
        softmax::softmax_f32(&mut scores);

        // Weighted sum of values
        let out_head = &mut attn_output[h * head_dim..(h + 1) * head_dim];
        for pos in 0..seq_len {
            let v_pos = cache.value_at(pos, kv_h);
            let s = scores[pos];
            for d in 0..head_dim {
                out_head[d] += s * v_pos[d];
            }
        }
    }

    // Output projection: hidden_dim = wo @ attn_output
    let mut output = vec![0.0f32; hidden_dim];
    nnx_kernels::matmul::matvec_f32(wo, &attn_output, &mut output, hidden_dim, q_dim);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_smoke() {
        // Tiny model: hidden=8, heads=2, kv_heads=2, head_dim=4
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let hidden = vec![1.0f32; hidden_dim];
        let wq = vec![0.1f32; q_dim * hidden_dim];
        let wk = vec![0.1f32; kv_dim * hidden_dim];
        let wv = vec![0.1f32; kv_dim * hidden_dim];
        let wo = vec![0.1f32; hidden_dim * q_dim];

        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        let output = attention_decode(
            &hidden, &wq, &wk, &wv, &wo,
            &mut cache, 0,
            num_heads, num_kv_heads, head_dim, 10000.0,
        );

        assert_eq!(output.len(), hidden_dim);
        assert_eq!(cache.len(), 1);
        // Values should be finite
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
