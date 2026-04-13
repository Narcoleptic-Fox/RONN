//! RMS Normalization — used by Llama-family models instead of LayerNorm.

/// RMS normalization: output = (x / rms(x)) * weight
///
/// rms(x) = sqrt(mean(x^2) + eps)
pub fn rms_norm_f32(x: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = x.len();
    assert_eq!(weight.len(), n);
    assert_eq!(output.len(), n);

    // Compute mean of squares
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
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
        // rms([1,1,1,1]) = 1.0, so output ≈ [1,1,1,1]
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }
}
