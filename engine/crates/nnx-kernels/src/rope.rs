//! Rotary Position Embedding (RoPE) -- position encoding for Llama-family models.

use nnx_core::error::EngineError;

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

/// Checked version of `rope_f32` that validates head_dim is even.
pub fn rope_f32_checked(x: &mut [f32], position: usize, freq_base: f32) -> nnx_core::error::Result<()> {
    if x.len() % 2 != 0 {
        return Err(EngineError::ShapeMismatch(
            format!("rope: head_dim must be even, got {}", x.len())
        ));
    }
    if x.is_empty() {
        return Err(EngineError::ShapeMismatch(
            "rope: input must be non-empty".to_string()
        ));
    }
    rope_f32(x, position, freq_base);
    Ok(())
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

    // Checked wrapper tests
    #[test]
    fn test_rope_checked_valid() {
        let mut x = [1.0, 2.0, 3.0, 4.0f32];
        assert!(rope_f32_checked(&mut x, 0, 10000.0).is_ok());
    }

    #[test]
    fn test_rope_checked_odd_dim() {
        let mut x = [1.0, 2.0, 3.0f32]; // odd dimension
        assert!(rope_f32_checked(&mut x, 0, 10000.0).is_err());
    }

    #[test]
    fn test_rope_checked_empty() {
        let mut x: [f32; 0] = [];
        assert!(rope_f32_checked(&mut x, 0, 10000.0).is_err());
    }
}
