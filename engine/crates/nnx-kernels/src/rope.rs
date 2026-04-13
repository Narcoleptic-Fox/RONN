//! Rotary Position Embedding (RoPE) — position encoding for Llama-family models.

/// Apply RoPE to a query or key vector in-place.
///
/// `x` has shape [head_dim] where head_dim is even.
/// `position` is the token position in the sequence.
/// `freq_base` is the base frequency (typically 10000.0 or 500000.0).
pub fn rope_f32(x: &mut [f32], position: usize, freq_base: f32) {
    let head_dim = x.len();
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = position as f32 * freq;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos_theta - x1 * sin_theta;
        x[i + half] = x0 * sin_theta + x1 * cos_theta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero() {
        // At position 0, all thetas are 0, so cos=1, sin=0 -> identity
        let mut x = [1.0, 2.0, 3.0, 4.0f32];
        let original = x;
        rope_f32(&mut x, 0, 10000.0);
        for i in 0..4 {
            assert!((x[i] - original[i]).abs() < 1e-6);
        }
    }
}
