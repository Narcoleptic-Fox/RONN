//! RMS Normalization — used by Llama-family models instead of LayerNorm.
//!
//! Uses SIMD dot product for the sum-of-squares computation.

use crate::matmul::dot_f32;

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
}
