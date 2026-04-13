//! RMS Normalization — used by Llama-family models instead of LayerNorm.
//!
//! Uses SIMD dot product for the sum-of-squares computation.

use crate::matmul::dot_f32;

use nnx_core::error::EngineError;

/// RMS normalization: output = (x / rms(x)) * weight
///
/// rms(x) = sqrt(mean(x^2) + eps)
pub fn rms_norm_f32(x: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    debug_assert_eq!(output.len(), n);

    // sum(x^2) via SIMD dot product: dot(x, x)
    let sum_sq = dot_f32(x, x);
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        output[i] = x[i] * inv_rms * weight[i];
    }
}

/// Checked version of `rms_norm_f32` that validates dimensions before computing.
pub fn rms_norm_f32_checked(
    x: &[f32], weight: &[f32], output: &mut [f32], eps: f32,
) -> nnx_core::error::Result<()> {
    let n = x.len();
    if weight.len() != n {
        return Err(EngineError::ShapeMismatch(
            format!("rms_norm: weight.len()={} != x.len()={}", weight.len(), n)
        ));
    }
    if output.len() != n {
        return Err(EngineError::ShapeMismatch(
            format!("rms_norm: output.len()={} != x.len()={}", output.len(), n)
        ));
    }
    if n == 0 {
        return Err(EngineError::ShapeMismatch(
            "rms_norm: input must be non-empty".to_string()
        ));
    }
    rms_norm_f32(x, weight, output, eps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_unit() {
        let x = [1.0, 1.0, 1.0, 1.0f32];
        let w = [1.0, 1.0, 1.0, 1.0f32];
        let mut out = [0.0f32; 4];
        rms_norm_f32(&x, &w, &mut out, 1e-5);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_scaling() {
        let x = [2.0, 2.0, 2.0, 2.0f32];
        let w = [1.0, 1.0, 1.0, 1.0f32];
        let mut out = [0.0f32; 4];
        rms_norm_f32(&x, &w, &mut out, 1e-5);
        // rms([2,2,2,2]) = 2.0, so output ≈ [1,1,1,1]
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_weights() {
        let x = [1.0, 1.0, 1.0, 1.0f32];
        let w = [2.0, 3.0, 4.0, 5.0f32];
        let mut out = [0.0f32; 4];
        rms_norm_f32(&x, &w, &mut out, 1e-5);
        // rms = 1.0, so output ≈ weights
        for i in 0..4 {
            assert!((out[i] - w[i]).abs() < 1e-3);
        }
    }

    // Checked wrapper tests
    #[test]
    fn test_rms_norm_checked_valid() {
        let x = [1.0, 1.0, 1.0, 1.0f32];
        let w = [1.0, 1.0, 1.0, 1.0f32];
        let mut out = [0.0f32; 4];
        assert!(rms_norm_f32_checked(&x, &w, &mut out, 1e-5).is_ok());
    }

    #[test]
    fn test_rms_norm_checked_bad_weight() {
        let x = [1.0, 1.0, 1.0, 1.0f32];
        let w = [1.0, 1.0f32]; // wrong size
        let mut out = [0.0f32; 4];
        assert!(rms_norm_f32_checked(&x, &w, &mut out, 1e-5).is_err());
    }

    #[test]
    fn test_rms_norm_checked_bad_output() {
        let x = [1.0, 1.0f32];
        let w = [1.0, 1.0f32];
        let mut out = [0.0f32; 3]; // wrong size
        assert!(rms_norm_f32_checked(&x, &w, &mut out, 1e-5).is_err());
    }

    #[test]
    fn test_rms_norm_checked_empty() {
        let x: [f32; 0] = [];
        let w: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        assert!(rms_norm_f32_checked(&x, &w, &mut out, 1e-5).is_err());
    }
}
