//! Multi-head attention with GQA support and parallel head computation.

use crate::cache::LayerCache;
use nnx_kernels::matmul;
use nnx_kernels::rope;
use nnx_kernels::softmax;
use rayon::prelude::*;

/// Minimum number of heads to use parallel computation.
const PARALLEL_HEAD_THRESHOLD: usize = 4;

/// Compute single-token attention for one layer during decoding.
///
/// Parallelizes across attention heads when there are enough heads.
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

    // Project Q, K, V (these matvecs are already parallelized internally)
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    matmul::matvec_f32(wq, hidden, &mut q, q_dim, hidden_dim);
    matmul::matvec_f32(wk, hidden, &mut k, kv_dim, hidden_dim);
    matmul::matvec_f32(wv, hidden, &mut v, kv_dim, hidden_dim);

    // Apply RoPE to Q and K (per-head)
    for h in 0..num_heads {
        rope::rope_f32(&mut q[h * head_dim..(h + 1) * head_dim], position, rope_freq_base);
    }
    for h in 0..num_kv_heads {
        rope::rope_f32(&mut k[h * head_dim..(h + 1) * head_dim], position, rope_freq_base);
    }

    // Store K, V in cache
    cache.store(&k, &v);
    let seq_len = cache.len();

    // Compute attention per head — parallel when enough heads
    let scale = 1.0 / (head_dim as f32).sqrt();

    let attn_output = if num_heads >= PARALLEL_HEAD_THRESHOLD {
        // Parallel: each head computed independently, results collected
        let head_outputs: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| {
                compute_head(h, heads_per_kv, &q, cache, head_dim, seq_len, scale)
            })
            .collect();

        // Flatten head outputs into [q_dim]
        let mut out = vec![0.0f32; q_dim];
        for (h, head_out) in head_outputs.iter().enumerate() {
            out[h * head_dim..(h + 1) * head_dim].copy_from_slice(head_out);
        }
        out
    } else {
        // Sequential for small head counts
        let mut out = vec![0.0f32; q_dim];
        for h in 0..num_heads {
            let head_out = compute_head(h, heads_per_kv, &q, cache, head_dim, seq_len, scale);
            out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&head_out);
        }
        out
    };

    // Output projection
    let mut output = vec![0.0f32; hidden_dim];
    matmul::matvec_f32(wo, &attn_output, &mut output, hidden_dim, q_dim);

    output
}

/// Compute attention for a single head. Pure function, safe to parallelize.
fn compute_head(
    h: usize,
    heads_per_kv: usize,
    q: &[f32],
    cache: &LayerCache,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let kv_h = h / heads_per_kv;
    let q_head = &q[h * head_dim..(h + 1) * head_dim];

    // Attention scores: score[pos] = q · k[pos] * scale
    let mut scores = vec![0.0f32; seq_len];
    for pos in 0..seq_len {
        let k_pos = cache.key_at(pos, kv_h);
        scores[pos] = matmul::dot_f32(q_head, k_pos) * scale;
    }

    softmax::softmax_f32(&mut scores);

    // Weighted sum of values
    let mut out = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        let v_pos = cache.value_at(pos, kv_h);
        let s = scores[pos];
        for d in 0..head_dim {
            out[d] += s * v_pos[d];
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_smoke() {
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
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_multi_heads_parallel() {
        // Enough heads to trigger parallel path
        let hidden_dim = 64;
        let num_heads = 8;
        let num_kv_heads = 4; // GQA: 2 query heads per KV head
        let head_dim = 8;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let hidden = vec![0.1f32; hidden_dim];
        let wq = vec![0.01f32; q_dim * hidden_dim];
        let wk = vec![0.01f32; kv_dim * hidden_dim];
        let wv = vec![0.01f32; kv_dim * hidden_dim];
        let wo = vec![0.01f32; hidden_dim * q_dim];

        let mut cache = LayerCache::new(32, num_kv_heads, head_dim);

        // Process several tokens
        for pos in 0..5 {
            let output = attention_decode(
                &hidden, &wq, &wk, &wv, &wo,
                &mut cache, pos,
                num_heads, num_kv_heads, head_dim, 10000.0,
            );
            assert_eq!(output.len(), hidden_dim);
            assert!(output.iter().all(|v| v.is_finite()));
        }
        assert_eq!(cache.len(), 5);
    }
}
