//! Multi-head attention with GQA support and parallel head computation.
//!
//! Supports architecture-specific features: optional bias, RoPE/partial RoPE/learned
//! position encoding, and GQA/MHA.

use crate::block::BlockWeights;
use crate::cache::LayerCache;
use crate::config::{ModelConfig, PosEncoding};
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
    cache: &mut LayerCache,
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
    cache: &mut LayerCache,
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

/// Batched prefill attention.
///
/// The heavy Q/K/V/O projections are batched across the prompt, but causal
/// attention still walks positions sequentially so the KV cache is populated
/// correctly.
pub fn attention_prefill_batch_configurable(
    hidden_batch: &[f32],
    batch_size: usize,
    weights: &BlockWeights,
    cache: &mut LayerCache,
    start_position: usize,
    config: &ModelConfig,
) -> Result<Vec<f32>> {
    let hidden_dim = config.hidden_dim;
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let heads_per_kv = config.num_heads / config.num_kv_heads;

    let mut q = vec![0.0f32; batch_size * q_dim];
    let mut k = vec![0.0f32; batch_size * kv_dim];
    let mut v = vec![0.0f32; batch_size * kv_dim];

    weights
        .wq
        .matmul_input_rows(hidden_batch, batch_size, &mut q);
    weights
        .wk
        .matmul_input_rows(hidden_batch, batch_size, &mut k);
    weights
        .wv
        .matmul_input_rows(hidden_batch, batch_size, &mut v);

    if let Some(bq) = &weights.bq {
        for chunk in q.chunks_mut(q_dim) {
            for i in 0..q_dim {
                chunk[i] += bq[i];
            }
        }
    }
    if let Some(bk) = &weights.bk {
        for chunk in k.chunks_mut(kv_dim) {
            for i in 0..kv_dim {
                chunk[i] += bk[i];
            }
        }
    }
    if let Some(bv) = &weights.bv {
        for chunk in v.chunks_mut(kv_dim) {
            for i in 0..kv_dim {
                chunk[i] += bv[i];
            }
        }
    }

    for token_idx in 0..batch_size {
        let position = start_position + token_idx;
        let q_chunk = &mut q[token_idx * q_dim..(token_idx + 1) * q_dim];
        let k_chunk = &mut k[token_idx * kv_dim..(token_idx + 1) * kv_dim];

        match &config.pos_encoding {
            PosEncoding::RoPE { freq_base } => {
                for h in 0..config.num_heads {
                    rope::rope_f32_checked(
                        &mut q_chunk[h * config.head_dim..(h + 1) * config.head_dim],
                        position,
                        *freq_base,
                    )?;
                }
                for h in 0..config.num_kv_heads {
                    rope::rope_f32_checked(
                        &mut k_chunk[h * config.head_dim..(h + 1) * config.head_dim],
                        position,
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
                        position,
                        *freq_base,
                    )?;
                }
                for h in 0..config.num_kv_heads {
                    let start = h * config.head_dim;
                    rope::rope_f32_checked(
                        &mut k_chunk[start..start + rotary_dim],
                        position,
                        *freq_base,
                    )?;
                }
            }
            PosEncoding::Learned | PosEncoding::None => {}
        }
    }

    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut attn_output = vec![0.0f32; batch_size * q_dim];

    for token_idx in 0..batch_size {
        let k_chunk = &k[token_idx * kv_dim..(token_idx + 1) * kv_dim];
        let v_chunk = &v[token_idx * kv_dim..(token_idx + 1) * kv_dim];
        cache.store(k_chunk, v_chunk)?;

        let seq_len = cache.len();
        let q_chunk = &q[token_idx * q_dim..(token_idx + 1) * q_dim];
        let out_chunk = &mut attn_output[token_idx * q_dim..(token_idx + 1) * q_dim];

        if config.num_heads >= PARALLEL_HEAD_THRESHOLD {
            let head_outputs: Vec<Vec<f32>> = (0..config.num_heads)
                .into_par_iter()
                .map(|h| {
                    compute_head(
                        h,
                        heads_per_kv,
                        q_chunk,
                        cache,
                        config.head_dim,
                        seq_len,
                        scale,
                    )
                })
                .collect();

            for (h, head_out) in head_outputs.iter().enumerate() {
                out_chunk[h * config.head_dim..(h + 1) * config.head_dim].copy_from_slice(head_out);
            }
        } else {
            for h in 0..config.num_heads {
                let head_out = compute_head(
                    h,
                    heads_per_kv,
                    q_chunk,
                    cache,
                    config.head_dim,
                    seq_len,
                    scale,
                );
                out_chunk[h * config.head_dim..(h + 1) * config.head_dim]
                    .copy_from_slice(&head_out);
            }
        }
    }

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
}
