//! Multi-head attention with GQA support and parallel head computation.
//!
//! Supports architecture-specific features: optional bias, RoPE/partial RoPE/learned
//! position encoding, and GQA/MHA.

use crate::block::BlockWeights;
use crate::config::{ModelConfig, PosEncoding};
use nnx_core::engine::KVStore;
use nnx_core::error::Result;
use nnx_kernels::matmul;
use nnx_kernels::rope;
use nnx_kernels::softmax;
use rayon::prelude::*;

/// Minimum number of heads to use parallel computation.
const PARALLEL_HEAD_THRESHOLD: usize = 4;

/// Compute single-token attention for one layer during decoding (Llama-only legacy).
///
/// Kept for backward compatibility. New code should use `attention_decode_configurable`.
pub fn attention_decode(
    hidden: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    cache: &mut dyn KVStore,
    position: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_freq_base: f32,
) -> Result<Vec<f32>> {
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
        rope::rope_f32_checked(
            &mut q[h * head_dim..(h + 1) * head_dim],
            position,
            rope_freq_base,
        )?;
    }
    for h in 0..num_kv_heads {
        rope::rope_f32_checked(
            &mut k[h * head_dim..(h + 1) * head_dim],
            position,
            rope_freq_base,
        )?;
    }

    // Store K, V in cache
    cache.store(&k, &v)?;
    let seq_len = cache.len();

    // Compute attention per head
    let scale = 1.0 / (head_dim as f32).sqrt();

    let attn_output = if num_heads >= PARALLEL_HEAD_THRESHOLD {
        let head_outputs: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| compute_head(h, heads_per_kv, &q, cache, head_dim, seq_len, scale))
            .collect();

        let mut out = vec![0.0f32; q_dim];
        for (h, head_out) in head_outputs.iter().enumerate() {
            out[h * head_dim..(h + 1) * head_dim].copy_from_slice(head_out);
        }
        out
    } else {
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

    Ok(output)
}

/// Architecture-aware attention decode.
///
/// Handles bias terms, different position encodings (RoPE, partial RoPE, learned,
/// none), and GQA/MHA head configurations.
pub fn attention_decode_configurable(
    hidden: &[f32],
    weights: &BlockWeights,
    cache: &mut dyn KVStore,
    position: usize,
    config: &ModelConfig,
) -> Result<Vec<f32>> {
    let hidden_dim = hidden.len();
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let heads_per_kv = config.num_heads / config.num_kv_heads;

    // Project Q, K, V
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    weights.wq.matvec(hidden, &mut q);
    weights.wk.matvec(hidden, &mut k);
    weights.wv.matvec(hidden, &mut v);

    // Add bias if present
    if let Some(bq) = &weights.bq {
        for i in 0..q_dim {
            q[i] += bq[i];
        }
    }
    if let Some(bk) = &weights.bk {
        for i in 0..kv_dim {
            k[i] += bk[i];
        }
    }
    if let Some(bv) = &weights.bv {
        for i in 0..kv_dim {
            v[i] += bv[i];
        }
    }

    // Position encoding
    match &config.pos_encoding {
        PosEncoding::RoPE { freq_base } => {
            for h in 0..config.num_heads {
                rope::rope_f32_checked(
                    &mut q[h * config.head_dim..(h + 1) * config.head_dim],
                    position,
                    *freq_base,
                )?;
            }
            for h in 0..config.num_kv_heads {
                rope::rope_f32_checked(
                    &mut k[h * config.head_dim..(h + 1) * config.head_dim],
                    position,
                    *freq_base,
                )?;
            }
        }
        PosEncoding::PartialRoPE {
            freq_base,
            rotary_dim,
        } => {
            // Only rotate the first rotary_dim dimensions of each head.
            // The remaining dimensions pass through without rotation.
            for h in 0..config.num_heads {
                let start = h * config.head_dim;
                rope::rope_f32_checked(&mut q[start..start + rotary_dim], position, *freq_base)?;
            }
            for h in 0..config.num_kv_heads {
                let start = h * config.head_dim;
                rope::rope_f32_checked(&mut k[start..start + rotary_dim], position, *freq_base)?;
            }
        }
        PosEncoding::Learned | PosEncoding::None => {
            // No rotation: position handled elsewhere (learned embeddings added
            // at the embedding layer level) or not used.
        }
    }

    // Store K, V in cache
    cache.store(&k, &v)?;
    let seq_len = cache.len();

    // Compute attention per head — parallel when enough heads
    let scale = 1.0 / (config.head_dim as f32).sqrt();

    let attn_output = if config.num_heads >= PARALLEL_HEAD_THRESHOLD {
        let head_outputs: Vec<Vec<f32>> = (0..config.num_heads)
            .into_par_iter()
            .map(|h| compute_head(h, heads_per_kv, &q, cache, config.head_dim, seq_len, scale))
            .collect();

        let mut out = vec![0.0f32; q_dim];
        for (h, head_out) in head_outputs.iter().enumerate() {
            out[h * config.head_dim..(h + 1) * config.head_dim].copy_from_slice(head_out);
        }
        out
    } else {
        let mut out = vec![0.0f32; q_dim];
        for h in 0..config.num_heads {
            let head_out =
                compute_head(h, heads_per_kv, &q, cache, config.head_dim, seq_len, scale);
            out[h * config.head_dim..(h + 1) * config.head_dim].copy_from_slice(&head_out);
        }
        out
    };

    // Output projection with optional bias
    let mut output = vec![0.0f32; hidden_dim];
    weights.wo.matvec(&attn_output, &mut output);
    if let Some(bo) = &weights.bo {
        for i in 0..hidden_dim {
            output[i] += bo[i];
        }
    }

    Ok(output)
}

/// Batched prefill attention with true causal masking.
///
/// Computes the full causal attention matrix for all prompt tokens at once,
/// replacing the previous sequential position-by-position loop. The algorithm:
///
/// 1. Project all tokens' Q, K, V in one batched matmul.
/// 2. Apply RoPE to each token's Q/K at its absolute position.
/// 3. Store all K/V into the KV cache for future decode calls.
/// 4. Per head: compute scores[seq_len, total_kv_len] = Q * K^T / sqrt(d_head),
///    apply causal mask (future positions → -inf), softmax each row, contract
///    with V to get the output for each token.
/// 5. Apply the output projection in batch.
///
/// GQA is handled naturally: each Q head indexes the KV cache via
/// `kv_head = q_head / heads_per_kv`.
///
/// Memory: the scores matrix is `[batch_size, total_kv_len]` per head.
/// For a 512-token prompt with 8 KV heads and 128-dim heads, that's
/// ~512 × (cache_pos + 512) × 8 × 4 bytes ≈ a few MB — acceptable.
pub fn attention_prefill_batch_configurable(
    hidden_batch: &[f32],
    batch_size: usize,
    weights: &BlockWeights,
    cache: &mut dyn KVStore,
    start_position: usize,
    config: &ModelConfig,
) -> Result<Vec<f32>> {
    let hidden_dim = config.hidden_dim;
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let heads_per_kv = config.num_heads / config.num_kv_heads;

    // Step 1: Batch-project all tokens through Q, K, V weight matrices.
    // q: [batch_size, q_dim], k/v: [batch_size, kv_dim]
    let mut q_all = vec![0.0f32; batch_size * q_dim];
    let mut k_all = vec![0.0f32; batch_size * kv_dim];
    let mut v_all = vec![0.0f32; batch_size * kv_dim];

    weights
        .wq
        .matmul_input_rows(hidden_batch, batch_size, &mut q_all);
    weights
        .wk
        .matmul_input_rows(hidden_batch, batch_size, &mut k_all);
    weights
        .wv
        .matmul_input_rows(hidden_batch, batch_size, &mut v_all);

    // Step 2: Add bias terms if present.
    if let Some(bq) = &weights.bq {
        for chunk in q_all.chunks_mut(q_dim) {
            for i in 0..q_dim {
                chunk[i] += bq[i];
            }
        }
    }
    if let Some(bk) = &weights.bk {
        for chunk in k_all.chunks_mut(kv_dim) {
            for i in 0..kv_dim {
                chunk[i] += bk[i];
            }
        }
    }
    if let Some(bv) = &weights.bv {
        for chunk in v_all.chunks_mut(kv_dim) {
            for i in 0..kv_dim {
                chunk[i] += bv[i];
            }
        }
    }

    // Step 3: Apply position encoding to each token's Q and K at their
    // absolute positions (start_position + token_idx).
    for token_idx in 0..batch_size {
        let absolute_position = start_position + token_idx;
        let q_chunk = &mut q_all[token_idx * q_dim..(token_idx + 1) * q_dim];
        let k_chunk = &mut k_all[token_idx * kv_dim..(token_idx + 1) * kv_dim];

        match &config.pos_encoding {
            PosEncoding::RoPE { freq_base } => {
                for h in 0..config.num_heads {
                    rope::rope_f32_checked(
                        &mut q_chunk[h * config.head_dim..(h + 1) * config.head_dim],
                        absolute_position,
                        *freq_base,
                    )?;
                }
                for h in 0..config.num_kv_heads {
                    rope::rope_f32_checked(
                        &mut k_chunk[h * config.head_dim..(h + 1) * config.head_dim],
                        absolute_position,
                        *freq_base,
                    )?;
                }
            }
            PosEncoding::PartialRoPE {
                freq_base,
                rotary_dim,
            } => {
                for h in 0..config.num_heads {
                    let start = h * config.head_dim;
                    rope::rope_f32_checked(
                        &mut q_chunk[start..start + rotary_dim],
                        absolute_position,
                        *freq_base,
                    )?;
                }
                for h in 0..config.num_kv_heads {
                    let start = h * config.head_dim;
                    rope::rope_f32_checked(
                        &mut k_chunk[start..start + rotary_dim],
                        absolute_position,
                        *freq_base,
                    )?;
                }
            }
            PosEncoding::Learned | PosEncoding::None => {}
        }
    }

    // Step 4: Populate the KV cache with all new keys and values.
    // We do this before computing attention so the cache holds the complete
    // context (prior cached tokens + all new tokens). Causal masking is
    // applied per token during the attention computation below.
    let cache_pos_before_batch = cache.len();
    for token_idx in 0..batch_size {
        let k_chunk = &k_all[token_idx * kv_dim..(token_idx + 1) * kv_dim];
        let v_chunk = &v_all[token_idx * kv_dim..(token_idx + 1) * kv_dim];
        cache.store(k_chunk, v_chunk)?;
    }

    // Total KV positions available after the batch is stored.
    let total_kv_len = cache.len();
    let scale = 1.0 / (config.head_dim as f32).sqrt();

    // Step 5: Batched causal attention across all heads.
    //
    // For each head we compute an [batch_size x total_kv_len] score matrix,
    // apply the causal mask, softmax each row, then contract with V to get
    // [batch_size x head_dim] output. Heads are parallelized with rayon.
    //
    // Causal mask rule: token at batch index `i` (absolute position
    // `cache_pos_before_batch + i`) may attend to KV positions 0..=(cache_pos_before_batch + i).
    let attn_output = compute_prefill_attention_all_heads(
        &q_all,
        cache,
        batch_size,
        config.num_heads,
        heads_per_kv,
        config.head_dim,
        cache_pos_before_batch,
        total_kv_len,
        scale,
    );

    // Step 6: Output projection in batch, plus optional bias.
    let mut output = vec![0.0f32; batch_size * hidden_dim];
    weights
        .wo
        .matmul_input_rows(&attn_output, batch_size, &mut output);
    if let Some(bo) = &weights.bo {
        for chunk in output.chunks_mut(hidden_dim) {
            for i in 0..hidden_dim {
                chunk[i] += bo[i];
            }
        }
    }

    Ok(output)
}

/// Compute causal attention output for all heads over a prefill batch.
///
/// Returns `[batch_size * q_dim]` attention output (all heads concatenated).
///
/// For each Q head `h`:
/// - Gather the corresponding KV head via `kv_h = h / heads_per_kv`.
/// - Build a score row for each query token by dotting against all cached
///   keys up to and including that token's causal horizon.
/// - Mask future positions to -inf, softmax, then contract with values.
///
/// Parallelized across heads when `num_heads >= PARALLEL_HEAD_THRESHOLD`.
fn compute_prefill_attention_all_heads(
    q_all: &[f32],
    cache: &dyn KVStore,
    batch_size: usize,
    num_heads: usize,
    heads_per_kv: usize,
    head_dim: usize,
    cache_pos_before_batch: usize,
    total_kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    let q_dim = num_heads * head_dim;
    let mut attn_output = vec![0.0f32; batch_size * q_dim];

    // Each head produces a [batch_size x head_dim] block starting at
    // h * head_dim within each token's output slice.
    if num_heads >= PARALLEL_HEAD_THRESHOLD {
        // Compute all heads in parallel; collect into per-head vecs, then scatter.
        let head_outputs: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| {
                compute_prefill_head(
                    h,
                    heads_per_kv,
                    q_all,
                    cache,
                    batch_size,
                    head_dim,
                    cache_pos_before_batch,
                    total_kv_len,
                    scale,
                )
            })
            .collect();

        // Scatter each head's output into the interleaved attn_output layout:
        // attn_output[token * q_dim + h * head_dim .. + head_dim]
        for (h, head_out) in head_outputs.iter().enumerate() {
            for token_idx in 0..batch_size {
                let src = &head_out[token_idx * head_dim..(token_idx + 1) * head_dim];
                let dst_start = token_idx * q_dim + h * head_dim;
                attn_output[dst_start..dst_start + head_dim].copy_from_slice(src);
            }
        }
    } else {
        for h in 0..num_heads {
            let head_out = compute_prefill_head(
                h,
                heads_per_kv,
                q_all,
                cache,
                batch_size,
                head_dim,
                cache_pos_before_batch,
                total_kv_len,
                scale,
            );
            for token_idx in 0..batch_size {
                let src = &head_out[token_idx * head_dim..(token_idx + 1) * head_dim];
                let dst_start = token_idx * q_dim + h * head_dim;
                attn_output[dst_start..dst_start + head_dim].copy_from_slice(src);
            }
        }
    }

    attn_output
}

/// Compute batched causal attention for a single head over all prompt tokens.
///
/// Returns `[batch_size * head_dim]` (tokens × head values, packed row-major).
///
/// For token `i` (0-indexed within the batch):
///   - Its absolute KV position is `cache_pos_before_batch + i`.
///   - It can attend to KV cache positions 0..=`cache_pos_before_batch + i`.
///   - Positions beyond that limit are set to -inf before softmax.
fn compute_prefill_head(
    h: usize,
    heads_per_kv: usize,
    q_all: &[f32],
    cache: &dyn KVStore,
    batch_size: usize,
    head_dim: usize,
    cache_pos_before_batch: usize,
    total_kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    let kv_h = h / heads_per_kv;

    // q_all layout: [batch_size, num_heads * head_dim] (row-major).
    // The per-token stride is q_all.len() / batch_size = num_heads * head_dim.
    let q_stride = q_all.len() / batch_size;

    // scores: [batch_size x total_kv_len] — one row per query token.
    let mut scores = vec![0.0f32; batch_size * total_kv_len];

    // Build the score matrix with causal masking applied inline.
    //
    // scores[i, j] = Q[i] · K[j] * scale  if j <= cache_pos_before_batch + i
    //              = -inf                    otherwise (causal mask)
    for token_idx in 0..batch_size {
        // Extract this token's Q vector for head `h`.
        let q_head_start = token_idx * q_stride + h * head_dim;
        let q_head = &q_all[q_head_start..q_head_start + head_dim];

        // Token `token_idx` occupies absolute KV position `cache_pos_before_batch + token_idx`.
        // It may attend to all KV positions up to and including its own absolute position.
        let causal_horizon = cache_pos_before_batch + token_idx;

        let score_row = &mut scores[token_idx * total_kv_len..(token_idx + 1) * total_kv_len];

        for kv_pos in 0..total_kv_len {
            if kv_pos <= causal_horizon {
                let k_vec = cache.key_at(kv_pos, kv_h);
                score_row[kv_pos] = matmul::dot_f32(q_head, k_vec) * scale;
            } else {
                // Future position: -inf collapses to 0 after softmax.
                score_row[kv_pos] = f32::NEG_INFINITY;
            }
        }

        softmax::softmax_f32(score_row);
    }

    // Contract softmax weights with values: out[i] = sum_j scores[i,j] * V[j]
    let mut out = vec![0.0f32; batch_size * head_dim];
    for token_idx in 0..batch_size {
        let score_row = &scores[token_idx * total_kv_len..(token_idx + 1) * total_kv_len];
        let out_slice = &mut out[token_idx * head_dim..(token_idx + 1) * head_dim];
        for kv_pos in 0..total_kv_len {
            let s = score_row[kv_pos];
            // After softmax, masked positions have score 0. Skip them to avoid
            // multiplying by zero, which also avoids reading those V entries.
            if s == 0.0 {
                continue;
            }
            let v_vec = cache.value_at(kv_pos, kv_h);
            for d in 0..head_dim {
                out_slice[d] += s * v_vec[d];
            }
        }
    }

    out
}

/// Compute attention for a single head. Pure function, safe to parallelize.
fn compute_head(
    h: usize,
    heads_per_kv: usize,
    q: &[f32],
    cache: &dyn KVStore,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let kv_h = h / heads_per_kv;
    let q_head = &q[h * head_dim..(h + 1) * head_dim];

    // Attention scores: score[pos] = q . k[pos] * scale
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
    use crate::cache::LayerCache;
    use crate::weights::Matrix;

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
            &hidden,
            &wq,
            &wk,
            &wv,
            &wo,
            &mut cache,
            0,
            num_heads,
            num_kv_heads,
            head_dim,
            10000.0,
        )
        .unwrap();

        assert_eq!(output.len(), hidden_dim);
        assert_eq!(cache.len(), 1);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_multi_heads_parallel() {
        let hidden_dim = 64;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 8;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let hidden = vec![0.1f32; hidden_dim];
        let wq = vec![0.01f32; q_dim * hidden_dim];
        let wk = vec![0.01f32; kv_dim * hidden_dim];
        let wv = vec![0.01f32; kv_dim * hidden_dim];
        let wo = vec![0.01f32; hidden_dim * q_dim];

        let mut cache = LayerCache::new(32, num_kv_heads, head_dim);

        for pos in 0..5 {
            let output = attention_decode(
                &hidden,
                &wq,
                &wk,
                &wv,
                &wo,
                &mut cache,
                pos,
                num_heads,
                num_kv_heads,
                head_dim,
                10000.0,
            )
            .unwrap();
            assert_eq!(output.len(), hidden_dim);
            assert!(output.iter().all(|v| v.is_finite()));
        }
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_attention_with_bias() {
        // Test the configurable attention with Q/K/V/O bias (GPT-2 / Qwen style)
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig {
            architecture: "qwen".into(),
            arch: crate::config::Architecture::Qwen,
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
            norm_type: crate::config::NormType::RMSNorm,
            ffn_type: crate::config::FFNType::SwiGLU,
            pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
            block_style: crate::config::BlockStyle::Sequential,
            has_qkv_bias: true,
            has_output_bias: false,
            embedding_scale: None,
        };

        let weights = BlockWeights::test_with_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        let output =
            attention_decode_configurable(&hidden, &weights, &mut cache, 0, &config).unwrap();

        assert_eq!(output.len(), hidden_dim);
        assert_eq!(cache.len(), 1);
        assert!(output.iter().all(|v| v.is_finite()));

        // Also verify it produces different output than without bias
        let weights_no_bias = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let mut cache2 = LayerCache::new(16, num_kv_heads, head_dim);
        let output_no_bias =
            attention_decode_configurable(&hidden, &weights_no_bias, &mut cache2, 0, &config)
                .unwrap();

        // Outputs should differ because of the bias
        let diff: f32 = output
            .iter()
            .zip(output_no_bias.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "bias should change the output");
    }

    #[test]
    fn test_partial_rope() {
        // Phi-style partial RoPE: only first rotary_dim dimensions get rotated.
        // We verify that partial RoPE produces valid output and that the rotation
        // is correctly applied to a subset of dimensions.
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let rotary_dim = 2;

        let config = ModelConfig {
            architecture: "phi".into(),
            arch: crate::config::Architecture::Phi,
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
            norm_type: crate::config::NormType::LayerNorm,
            ffn_type: crate::config::FFNType::SwiGLU,
            pos_encoding: PosEncoding::PartialRoPE {
                freq_base: 10000.0,
                rotary_dim,
            },
            block_style: crate::config::BlockStyle::Parallel,
            has_qkv_bias: true,
            has_output_bias: true,
            embedding_scale: None,
        };

        // Use varied weights so different head dims produce different Q/K values
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let weights = BlockWeights {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
            wq: Matrix::dense(
                (0..q_dim * hidden_dim)
                    .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
                    .collect(),
                q_dim,
                hidden_dim,
            ),
            wk: Matrix::dense(
                (0..kv_dim * hidden_dim)
                    .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
                    .collect(),
                kv_dim,
                hidden_dim,
            ),
            wv: Matrix::dense(
                (0..kv_dim * hidden_dim)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                    .collect(),
                kv_dim,
                hidden_dim,
            ),
            wo: Matrix::dense(
                (0..hidden_dim * q_dim)
                    .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
                    .collect(),
                hidden_dim,
                q_dim,
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
            bq: Some(vec![0.1; q_dim]),
            bk: Some(vec![0.1; kv_dim]),
            bv: Some(vec![0.1; kv_dim]),
            bo: Some(vec![0.01; hidden_dim]),
            attn_norm_bias: Some(vec![0.0; hidden_dim]),
            ffn_norm_bias: Some(vec![0.0; hidden_dim]),
        };

        let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let hidden2: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 + 2.0) * 0.3).collect();

        // Build a cache with multiple entries so softmax produces non-trivial scores.
        // With partial RoPE, the first `rotary_dim` dims of Q/K get rotated,
        // which changes the attention score distribution.
        let mut cache_partial = LayerCache::new(16, num_kv_heads, head_dim);
        // Fill cache with several tokens
        let _ = attention_decode_configurable(&hidden, &weights, &mut cache_partial, 0, &config)
            .unwrap();
        let _ = attention_decode_configurable(&hidden2, &weights, &mut cache_partial, 1, &config)
            .unwrap();
        let output_partial =
            attention_decode_configurable(&hidden, &weights, &mut cache_partial, 2, &config)
                .unwrap();
        assert!(
            output_partial.iter().all(|v| v.is_finite()),
            "partial RoPE output should be finite"
        );

        // Compare with no RoPE using the same input sequence
        let config_no_rope = ModelConfig {
            pos_encoding: PosEncoding::None,
            ..config.clone()
        };
        let mut cache_none = LayerCache::new(16, num_kv_heads, head_dim);
        let _ =
            attention_decode_configurable(&hidden, &weights, &mut cache_none, 0, &config_no_rope)
                .unwrap();
        let _ =
            attention_decode_configurable(&hidden2, &weights, &mut cache_none, 1, &config_no_rope)
                .unwrap();
        let output_no_rope =
            attention_decode_configurable(&hidden, &weights, &mut cache_none, 2, &config_no_rope)
                .unwrap();
        assert!(
            output_no_rope.iter().all(|v| v.is_finite()),
            "no-RoPE output should be finite"
        );

        // With multiple cache entries, partial RoPE changes Q/K dot products and
        // thus the softmax distribution, producing different final output.
        let diff: f32 = output_partial
            .iter()
            .zip(output_no_rope.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "partial RoPE should differ from no RoPE with multi-token cache, diff={}",
            diff
        );
    }

    #[test]
    fn test_attention_no_rope() {
        // GPT-2 style: no RoPE, learned positions
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = ModelConfig {
            architecture: "gpt2".into(),
            arch: crate::config::Architecture::GPT2,
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
            norm_type: crate::config::NormType::LayerNorm,
            ffn_type: crate::config::FFNType::GELU,
            pos_encoding: PosEncoding::Learned,
            block_style: crate::config::BlockStyle::Sequential,
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

        let hidden = vec![1.0f32; hidden_dim];
        let mut cache = LayerCache::new(16, num_kv_heads, head_dim);

        let output =
            attention_decode_configurable(&hidden, &weights, &mut cache, 0, &config).unwrap();

        assert_eq!(output.len(), hidden_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // -----------------------------------------------------------------------
    // Batched prefill tests
    // -----------------------------------------------------------------------

    /// Build a Llama-style config for prefill tests.
    fn make_llama_config(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> ModelConfig {
        ModelConfig {
            architecture: "llama".into(),
            arch: crate::config::Architecture::Llama,
            num_layers: 1,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: 32,
            max_context_length: 128,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: crate::config::NormType::RMSNorm,
            ffn_type: crate::config::FFNType::SwiGLU,
            pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
            block_style: crate::config::BlockStyle::Sequential,
            has_qkv_bias: false,
            has_output_bias: false,
            embedding_scale: None,
        }
    }

    /// Prefill a sequence token-by-token using `attention_decode_configurable`,
    /// returning the outputs for all tokens (used as the reference implementation).
    fn prefill_sequential(
        hidden_batch: &[f32],
        batch_size: usize,
        weights: &BlockWeights,
        config: &ModelConfig,
        start_position: usize,
    ) -> Vec<f32> {
        let hidden_dim = config.hidden_dim;
        let mut cache = LayerCache::new(128, config.num_kv_heads, config.head_dim);
        let mut outputs = vec![0.0f32; batch_size * hidden_dim];

        for token_idx in 0..batch_size {
            let hidden = &hidden_batch[token_idx * hidden_dim..(token_idx + 1) * hidden_dim];
            let position = start_position + token_idx;
            let out = attention_decode_configurable(hidden, weights, &mut cache, position, config)
                .unwrap();
            outputs[token_idx * hidden_dim..(token_idx + 1) * hidden_dim].copy_from_slice(&out);
        }

        outputs
    }

    #[test]
    fn test_prefill_batch_output_is_finite() {
        // Basic smoke test: batched prefill produces finite values and correct shape.
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let batch_size = 4;

        let config = make_llama_config(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        // Give each token a distinct hidden state.
        let hidden_batch: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();

        let mut cache = LayerCache::new(128, num_kv_heads, head_dim);
        let output = attention_prefill_batch_configurable(
            &hidden_batch,
            batch_size,
            &weights,
            &mut cache,
            0,
            &config,
        )
        .unwrap();

        assert_eq!(output.len(), batch_size * hidden_dim);
        assert!(
            output.iter().all(|v| v.is_finite()),
            "all outputs should be finite"
        );
        // All batch tokens should have been stored in the KV cache.
        assert_eq!(
            cache.len(),
            batch_size,
            "cache should hold all prompt tokens"
        );
    }

    #[test]
    fn test_prefill_batch_matches_sequential() {
        // The batched prefill must produce the same result as processing each token
        // one-at-a-time with `attention_decode_configurable`. This verifies that the
        // causal masking is correct and that RoPE is applied at the right positions.
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let batch_size = 5;

        let config = make_llama_config(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let hidden_batch: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect();

        // Reference: sequential token-by-token decode.
        let sequential_out = prefill_sequential(&hidden_batch, batch_size, &weights, &config, 0);

        // System under test: batched prefill.
        let mut cache = LayerCache::new(128, num_kv_heads, head_dim);
        let batch_out = attention_prefill_batch_configurable(
            &hidden_batch,
            batch_size,
            &weights,
            &mut cache,
            0,
            &config,
        )
        .unwrap();

        // Allow for small floating-point differences due to summation order.
        let max_diff: f32 = sequential_out
            .iter()
            .zip(batch_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "batched prefill differs from sequential by {max_diff} (expected < 1e-4)"
        );
    }

    #[test]
    fn test_prefill_batch_gqa_matches_sequential() {
        // Test with GQA: num_kv_heads < num_heads.
        // Each pair of Q heads shares one KV head.
        let hidden_dim = 16;
        let num_heads = 4;
        let num_kv_heads = 2; // GQA: 2 Q heads per KV head
        let head_dim = 4;
        let intermediate_dim = 32;
        let batch_size = 4;

        let config = make_llama_config(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let hidden_batch: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.08)
            .collect();

        let sequential_out = prefill_sequential(&hidden_batch, batch_size, &weights, &config, 0);

        let mut cache = LayerCache::new(128, num_kv_heads, head_dim);
        let batch_out = attention_prefill_batch_configurable(
            &hidden_batch,
            batch_size,
            &weights,
            &mut cache,
            0,
            &config,
        )
        .unwrap();

        let max_diff: f32 = sequential_out
            .iter()
            .zip(batch_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "GQA batched prefill differs from sequential by {max_diff}"
        );
    }

    #[test]
    fn test_prefill_batch_with_cache_offset() {
        // Verify prefill with cache_pos > 0 (continuing from prior context).
        // We fill the cache with 3 tokens first, then prefill 4 more.
        // The batched path must apply causal masking relative to the correct
        // absolute positions (3, 4, 5, 6).
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let prior_tokens = 3;
        let batch_size = 4;

        let config = make_llama_config(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        // Hidden states for all tokens (prior + batch).
        let all_hidden: Vec<f32> = (0..(prior_tokens + batch_size) * hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.12)
            .collect();

        // Reference: process all tokens sequentially from scratch.
        let full_sequential =
            prefill_sequential(&all_hidden, prior_tokens + batch_size, &weights, &config, 0);
        let sequential_batch_out = &full_sequential[prior_tokens * hidden_dim..];

        // System under test: prefill prior tokens individually, then batch the rest.
        let mut cache = LayerCache::new(128, num_kv_heads, head_dim);
        for tok in 0..prior_tokens {
            let hidden = &all_hidden[tok * hidden_dim..(tok + 1) * hidden_dim];
            attention_decode_configurable(hidden, &weights, &mut cache, tok, &config).unwrap();
        }
        assert_eq!(cache.len(), prior_tokens);

        let batch_hidden = &all_hidden[prior_tokens * hidden_dim..];
        let batch_out = attention_prefill_batch_configurable(
            batch_hidden,
            batch_size,
            &weights,
            &mut cache,
            prior_tokens, // start_position = cache_pos_before_batch
            &config,
        )
        .unwrap();

        let max_diff: f32 = sequential_batch_out
            .iter()
            .zip(batch_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "prefill with cache offset differs from sequential by {max_diff}"
        );
    }

    #[test]
    fn test_prefill_causal_mask_token_cannot_see_future() {
        // Verify the causal mask: the first token's output must not change when we
        // add more tokens to the batch. If token 0 could attend to tokens 1, 2, …,
        // its output would differ between a single-token and multi-token batch.
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let config = make_llama_config(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        // Vary hidden states so the test is sensitive to cross-token attention.
        let single_hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let multi_hidden: Vec<f32> = {
            let mut v = single_hidden.clone();
            // Append a second token with a very different hidden state.
            v.extend((0..hidden_dim).map(|i| -(i as f32 + 1.0) * 0.7));
            v
        };

        // Single-token prefill — only token 0 is processed.
        let mut cache_single = LayerCache::new(128, num_kv_heads, head_dim);
        let single_out = attention_prefill_batch_configurable(
            &single_hidden,
            1,
            &weights,
            &mut cache_single,
            0,
            &config,
        )
        .unwrap();

        // Two-token prefill — token 0 output should be identical to single_out
        // because token 0 cannot attend to token 1 (future).
        let mut cache_multi = LayerCache::new(128, num_kv_heads, head_dim);
        let multi_out = attention_prefill_batch_configurable(
            &multi_hidden,
            2,
            &weights,
            &mut cache_multi,
            0,
            &config,
        )
        .unwrap();

        // Extract the first token's output from the two-token batch.
        let multi_token0_out = &multi_out[..hidden_dim];

        let max_diff: f32 = single_out
            .iter()
            .zip(multi_token0_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "token 0 output changed when future tokens were added (diff={max_diff}); \
             causal mask is broken"
        );
    }

    #[test]
    fn test_prefill_parallel_heads_matches_sequential_heads() {
        // Verify that the parallel-head path (num_heads >= PARALLEL_HEAD_THRESHOLD = 4)
        // produces the same result as the sequential-head path. We use 4 heads to
        // exercise the parallel branch, and compare against 2-head sequential config.
        let hidden_dim = 16;
        let num_heads = 4; // >= PARALLEL_HEAD_THRESHOLD
        let num_kv_heads = 4;
        let head_dim = 4;
        let intermediate_dim = 32;
        let batch_size = 3;

        let config = make_llama_config(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        let hidden_batch: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.15)
            .collect();

        let sequential_out = prefill_sequential(&hidden_batch, batch_size, &weights, &config, 0);

        let mut cache = LayerCache::new(128, num_kv_heads, head_dim);
        let batch_out = attention_prefill_batch_configurable(
            &hidden_batch,
            batch_size,
            &weights,
            &mut cache,
            0,
            &config,
        )
        .unwrap();

        let max_diff: f32 = sequential_out
            .iter()
            .zip(batch_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "parallel-head prefill differs from sequential by {max_diff}"
        );
    }
}
