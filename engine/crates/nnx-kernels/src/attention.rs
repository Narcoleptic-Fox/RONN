//! Attention operations: CrossAttention, MultiHeadAttention helpers.
//!
//! These are higher-level ops that compose matmul, softmax, and masking.
//! Used by Whisper, T5, Stable Diffusion, and encoder-decoder models.

use crate::matmul::matmul_f32;
use crate::softmax::softmax_f32;

use nnx_core::error::EngineError;

/// Cross-attention: Q attends to K, V from a different source.
///
/// Q: [seq_q, d_k]   -- queries (from decoder)
/// K: [seq_kv, d_k]  -- keys (from encoder)
/// V: [seq_kv, d_v]  -- values (from encoder)
/// output: [seq_q, d_v]
///
/// score = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// Optional mask: [seq_q, seq_kv] -- true means "mask this position" (set to -inf).
pub fn cross_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    d_k: usize,
    d_v: usize,
    mask: Option<&[bool]>,
) {
    assert_eq!(q.len(), seq_q * d_k);
    assert_eq!(k.len(), seq_kv * d_k);
    assert_eq!(v.len(), seq_kv * d_v);
    assert_eq!(output.len(), seq_q * d_v);

    let scale = 1.0 / (d_k as f32).sqrt();

    // Step 1: Compute attention scores = Q @ K^T -> [seq_q, seq_kv]
    // K^T: [d_k, seq_kv], so we transpose K
    let mut k_t = vec![0.0f32; d_k * seq_kv];
    for r in 0..seq_kv {
        for c in 0..d_k {
            k_t[c * seq_kv + r] = k[r * d_k + c];
        }
    }

    let mut scores = vec![0.0f32; seq_q * seq_kv];
    matmul_f32(q, &k_t, &mut scores, seq_q, d_k, seq_kv);

    // Scale
    for s in scores.iter_mut() {
        *s *= scale;
    }

    // Apply mask if provided
    if let Some(m) = mask {
        assert_eq!(m.len(), seq_q * seq_kv);
        for i in 0..scores.len() {
            if m[i] {
                scores[i] = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax per query row
    for r in 0..seq_q {
        let row = &mut scores[r * seq_kv..(r + 1) * seq_kv];
        softmax_f32(row);
    }

    // Step 2: output = scores @ V -> [seq_q, d_v]
    matmul_f32(&scores, v, output, seq_q, seq_kv, d_v);
}

/// Checked version of `cross_attention_f32` that validates dimensions before computing.
pub fn cross_attention_f32_checked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    d_k: usize,
    d_v: usize,
    mask: Option<&[bool]>,
) -> nnx_core::error::Result<()> {
    if q.len() != seq_q * d_k {
        return Err(EngineError::ShapeMismatch(
            format!("cross_attention: q.len()={} but seq_q*d_k={}", q.len(), seq_q * d_k)
        ));
    }
    if k.len() != seq_kv * d_k {
        return Err(EngineError::ShapeMismatch(
            format!("cross_attention: k.len()={} but seq_kv*d_k={}", k.len(), seq_kv * d_k)
        ));
    }
    if v.len() != seq_kv * d_v {
        return Err(EngineError::ShapeMismatch(
            format!("cross_attention: v.len()={} but seq_kv*d_v={}", v.len(), seq_kv * d_v)
        ));
    }
    if output.len() != seq_q * d_v {
        return Err(EngineError::ShapeMismatch(
            format!("cross_attention: output.len()={} but seq_q*d_v={}", output.len(), seq_q * d_v)
        ));
    }
    if let Some(m) = mask {
        if m.len() != seq_q * seq_kv {
            return Err(EngineError::ShapeMismatch(
                format!("cross_attention: mask.len()={} but seq_q*seq_kv={}", m.len(), seq_q * seq_kv)
            ));
        }
    }
    cross_attention_f32(q, k, v, output, seq_q, seq_kv, d_k, d_v, mask);
    Ok(())
}

/// Multi-head cross-attention: runs cross_attention per head, concatenates results.
///
/// Q: [num_heads, seq_q, head_dim]
/// K: [num_heads, seq_kv, head_dim]  (or fewer heads for GQA)
/// V: [num_heads, seq_kv, head_dim]
/// output: [seq_q, num_heads * head_dim]
pub fn multi_head_cross_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    mask: Option<&[bool]>,
) {
    assert_eq!(q.len(), num_heads * seq_q * head_dim);
    assert_eq!(k.len(), num_kv_heads * seq_kv * head_dim);
    assert_eq!(v.len(), num_kv_heads * seq_kv * head_dim);
    assert_eq!(output.len(), seq_q * num_heads * head_dim);

    let heads_per_kv = num_heads / num_kv_heads;

    let mut head_output = vec![0.0f32; seq_q * head_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;

        let q_head = &q[h * seq_q * head_dim..(h + 1) * seq_q * head_dim];
        let k_head = &k[kv_h * seq_kv * head_dim..(kv_h + 1) * seq_kv * head_dim];
        let v_head = &v[kv_h * seq_kv * head_dim..(kv_h + 1) * seq_kv * head_dim];

        cross_attention_f32(
            q_head,
            k_head,
            v_head,
            &mut head_output,
            seq_q,
            seq_kv,
            head_dim,
            head_dim,
            mask,
        );

        // Scatter into output: output[t, h*head_dim .. (h+1)*head_dim] = head_output[t, :]
        for t in 0..seq_q {
            let src = &head_output[t * head_dim..(t + 1) * head_dim];
            let dst_start = t * num_heads * head_dim + h * head_dim;
            output[dst_start..dst_start + head_dim].copy_from_slice(src);
        }
    }
}

/// Checked version of `multi_head_cross_attention_f32` that validates dimensions before computing.
pub fn multi_head_cross_attention_f32_checked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    mask: Option<&[bool]>,
) -> nnx_core::error::Result<()> {
    if q.len() != num_heads * seq_q * head_dim {
        return Err(EngineError::ShapeMismatch(
            format!("multi_head_cross_attention: q.len()={} but num_heads*seq_q*head_dim={}", q.len(), num_heads * seq_q * head_dim)
        ));
    }
    if k.len() != num_kv_heads * seq_kv * head_dim {
        return Err(EngineError::ShapeMismatch(
            format!("multi_head_cross_attention: k.len()={} but num_kv_heads*seq_kv*head_dim={}", k.len(), num_kv_heads * seq_kv * head_dim)
        ));
    }
    if v.len() != num_kv_heads * seq_kv * head_dim {
        return Err(EngineError::ShapeMismatch(
            format!("multi_head_cross_attention: v.len()={} but num_kv_heads*seq_kv*head_dim={}", v.len(), num_kv_heads * seq_kv * head_dim)
        ));
    }
    if output.len() != seq_q * num_heads * head_dim {
        return Err(EngineError::ShapeMismatch(
            format!("multi_head_cross_attention: output.len()={} but seq_q*num_heads*head_dim={}", output.len(), seq_q * num_heads * head_dim)
        ));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(EngineError::ShapeMismatch(
            format!("multi_head_cross_attention: num_heads={} must be divisible by num_kv_heads={}", num_heads, num_kv_heads)
        ));
    }
    if let Some(m) = mask {
        if m.len() != seq_q * seq_kv {
            return Err(EngineError::ShapeMismatch(
                format!("multi_head_cross_attention: mask.len()={} but seq_q*seq_kv={}", m.len(), seq_q * seq_kv)
            ));
        }
    }
    multi_head_cross_attention_f32(q, k, v, output, num_heads, num_kv_heads, seq_q, seq_kv, head_dim, mask);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_attention_identity() {
        let q = [1.0, 0.0, 0.0, 1.0f32];
        let k = [1.0, 0.0, 0.0, 1.0f32];
        let v = [10.0, 20.0, 30.0, 40.0f32];
        let mut output = [0.0f32; 4];
        cross_attention_f32(&q, &k, &v, &mut output, 2, 2, 2, 2, None);
        assert!(output[0] < output[2], "row 0 col 0 should be < row 1 col 0");
    }

    #[test]
    fn test_cross_attention_uniform() {
        let q = [0.0; 4];
        let k = [0.0; 4];
        let v = [10.0, 20.0, 30.0, 40.0f32];
        let mut output = [0.0f32; 4];
        cross_attention_f32(&q, &k, &v, &mut output, 2, 2, 2, 2, None);
        for r in 0..2 {
            assert!((output[r * 2] - 20.0).abs() < 1e-4, "got {}", output[r * 2]);
            assert!((output[r * 2 + 1] - 30.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cross_attention_with_mask() {
        let q = [0.0; 4];
        let k = [0.0; 4];
        let v = [10.0, 20.0, 30.0, 40.0f32];
        let mask = [false, true, false, true];
        let mut output = [0.0f32; 4];
        cross_attention_f32(&q, &k, &v, &mut output, 2, 2, 2, 2, Some(&mask));
        for r in 0..2 {
            assert!((output[r * 2] - 10.0).abs() < 1e-4);
            assert!((output[r * 2 + 1] - 20.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_multi_head_cross_attention() {
        let q = [0.0; 4];
        let k = [0.0; 4];
        let v = [1.0, 2.0, 3.0, 4.0f32];
        let mut output = [0.0f32; 4];
        multi_head_cross_attention_f32(&q, &k, &v, &mut output, 2, 2, 1, 1, 2, None);
        assert!((output[0] - 1.0).abs() < 1e-4);
        assert!((output[1] - 2.0).abs() < 1e-4);
        assert!((output[2] - 3.0).abs() < 1e-4);
        assert!((output[3] - 4.0).abs() < 1e-4);
    }

    // Checked wrapper tests
    #[test]
    fn test_cross_attention_checked_valid() {
        let q = [0.0; 4];
        let k = [0.0; 4];
        let v = [10.0, 20.0, 30.0, 40.0f32];
        let mut output = [0.0f32; 4];
        let result = cross_attention_f32_checked(&q, &k, &v, &mut output, 2, 2, 2, 2, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_attention_checked_bad_q() {
        let q = [0.0; 3]; // wrong size
        let k = [0.0; 4];
        let v = [10.0, 20.0, 30.0, 40.0f32];
        let mut output = [0.0f32; 4];
        let result = cross_attention_f32_checked(&q, &k, &v, &mut output, 2, 2, 2, 2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_attention_checked_bad_mask() {
        let q = [0.0; 4];
        let k = [0.0; 4];
        let v = [10.0, 20.0, 30.0, 40.0f32];
        let mask = [false, true]; // wrong size
        let mut output = [0.0f32; 4];
        let result = cross_attention_f32_checked(&q, &k, &v, &mut output, 2, 2, 2, 2, Some(&mask));
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_checked_bad_kv_heads() {
        let q = [0.0; 4];
        let k = [0.0; 4];
        let v = [1.0, 2.0, 3.0, 4.0f32];
        let mut output = [0.0f32; 4];
        // num_heads=3, num_kv_heads=2 => 3 % 2 != 0
        let result = multi_head_cross_attention_f32_checked(&q, &k, &v, &mut output, 3, 2, 1, 1, 2, None);
        assert!(result.is_err());
    }
}
