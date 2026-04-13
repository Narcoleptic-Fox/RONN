//! Upsample and resize operations.
//!
//! Nearest-neighbor and bilinear interpolation for spatial upsampling.
//! Used by Stable Diffusion, YOLO, and other vision models.

use nnx_core::error::EngineError;

/// Nearest-neighbor upsample 2D.
///
/// Input:  [N, C, H_in, W_in]
/// Output: [N, C, H_out, W_out]
///
/// Each output pixel copies from the nearest input pixel.
pub fn upsample_nearest_2d_f32(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) {
    assert_eq!(input.len(), batch_size * channels * h_in * w_in);
    assert_eq!(output.len(), batch_size * channels * h_out * w_out);

    let h_scale = h_in as f32 / h_out as f32;
    let w_scale = w_in as f32 / w_out as f32;

    for n in 0..batch_size {
        for c in 0..channels {
            let in_offset = (n * channels + c) * h_in * w_in;
            let out_offset = (n * channels + c) * h_out * w_out;

            for oh in 0..h_out {
                let ih = ((oh as f32 + 0.5) * h_scale).floor() as usize;
                let ih = ih.min(h_in - 1);

                for ow in 0..w_out {
                    let iw = ((ow as f32 + 0.5) * w_scale).floor() as usize;
                    let iw = iw.min(w_in - 1);
                    output[out_offset + oh * w_out + ow] = input[in_offset + ih * w_in + iw];
                }
            }
        }
    }
}

/// Checked version of `upsample_nearest_2d_f32`.
pub fn upsample_nearest_2d_f32_checked(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> nnx_core::error::Result<()> {
    let expected_in = batch_size * channels * h_in * w_in;
    let expected_out = batch_size * channels * h_out * w_out;
    if input.len() != expected_in {
        return Err(EngineError::ShapeMismatch(
            format!("upsample_nearest_2d: input.len()={} but expected {}", input.len(), expected_in)
        ));
    }
    if output.len() != expected_out {
        return Err(EngineError::ShapeMismatch(
            format!("upsample_nearest_2d: output.len()={} but expected {}", output.len(), expected_out)
        ));
    }
    if h_in == 0 || w_in == 0 {
        return Err(EngineError::ShapeMismatch(
            "upsample_nearest_2d: input spatial dims must be non-zero".to_string()
        ));
    }
    upsample_nearest_2d_f32(input, output, batch_size, channels, h_in, w_in, h_out, w_out);
    Ok(())
}

/// Bilinear upsample 2D.
///
/// Input:  [N, C, H_in, W_in]
/// Output: [N, C, H_out, W_out]
///
/// Each output pixel is a weighted average of the 4 nearest input pixels.
pub fn upsample_bilinear_2d_f32(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) {
    assert_eq!(input.len(), batch_size * channels * h_in * w_in);
    assert_eq!(output.len(), batch_size * channels * h_out * w_out);

    let h_scale = if h_out > 1 {
        (h_in - 1) as f32 / (h_out - 1) as f32
    } else {
        0.0
    };
    let w_scale = if w_out > 1 {
        (w_in - 1) as f32 / (w_out - 1) as f32
    } else {
        0.0
    };

    for n in 0..batch_size {
        for c in 0..channels {
            let in_offset = (n * channels + c) * h_in * w_in;
            let out_offset = (n * channels + c) * h_out * w_out;

            for oh in 0..h_out {
                let h_src = oh as f32 * h_scale;
                let h0 = h_src.floor() as usize;
                let h1 = (h0 + 1).min(h_in - 1);
                let h_frac = h_src - h0 as f32;

                for ow in 0..w_out {
                    let w_src = ow as f32 * w_scale;
                    let w0 = w_src.floor() as usize;
                    let w1 = (w0 + 1).min(w_in - 1);
                    let w_frac = w_src - w0 as f32;

                    let v00 = input[in_offset + h0 * w_in + w0];
                    let v01 = input[in_offset + h0 * w_in + w1];
                    let v10 = input[in_offset + h1 * w_in + w0];
                    let v11 = input[in_offset + h1 * w_in + w1];

                    let top = v00 * (1.0 - w_frac) + v01 * w_frac;
                    let bot = v10 * (1.0 - w_frac) + v11 * w_frac;
                    output[out_offset + oh * w_out + ow] = top * (1.0 - h_frac) + bot * h_frac;
                }
            }
        }
    }
}

/// Checked version of `upsample_bilinear_2d_f32`.
pub fn upsample_bilinear_2d_f32_checked(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> nnx_core::error::Result<()> {
    let expected_in = batch_size * channels * h_in * w_in;
    let expected_out = batch_size * channels * h_out * w_out;
    if input.len() != expected_in {
        return Err(EngineError::ShapeMismatch(
            format!("upsample_bilinear_2d: input.len()={} but expected {}", input.len(), expected_in)
        ));
    }
    if output.len() != expected_out {
        return Err(EngineError::ShapeMismatch(
            format!("upsample_bilinear_2d: output.len()={} but expected {}", output.len(), expected_out)
        ));
    }
    if h_in == 0 || w_in == 0 {
        return Err(EngineError::ShapeMismatch(
            "upsample_bilinear_2d: input spatial dims must be non-zero".to_string()
        ));
    }
    upsample_bilinear_2d_f32(input, output, batch_size, channels, h_in, w_in, h_out, w_out);
    Ok(())
}

/// Nearest-neighbor upsample 1D.
///
/// Input:  [N, C, L_in]
/// Output: [N, C, L_out]
pub fn upsample_nearest_1d_f32(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    l_in: usize,
    l_out: usize,
) {
    assert_eq!(input.len(), batch_size * channels * l_in);
    assert_eq!(output.len(), batch_size * channels * l_out);

    let scale = l_in as f32 / l_out as f32;

    for n in 0..batch_size {
        for c in 0..channels {
            let in_offset = (n * channels + c) * l_in;
            let out_offset = (n * channels + c) * l_out;

            for ol in 0..l_out {
                let il = ((ol as f32 + 0.5) * scale).floor() as usize;
                let il = il.min(l_in - 1);
                output[out_offset + ol] = input[in_offset + il];
            }
        }
    }
}

/// Checked version of `upsample_nearest_1d_f32`.
pub fn upsample_nearest_1d_f32_checked(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    l_in: usize,
    l_out: usize,
) -> nnx_core::error::Result<()> {
    let expected_in = batch_size * channels * l_in;
    let expected_out = batch_size * channels * l_out;
    if input.len() != expected_in {
        return Err(EngineError::ShapeMismatch(
            format!("upsample_nearest_1d: input.len()={} but expected {}", input.len(), expected_in)
        ));
    }
    if output.len() != expected_out {
        return Err(EngineError::ShapeMismatch(
            format!("upsample_nearest_1d: output.len()={} but expected {}", output.len(), expected_out)
        ));
    }
    if l_in == 0 {
        return Err(EngineError::ShapeMismatch(
            "upsample_nearest_1d: l_in must be non-zero".to_string()
        ));
    }
    upsample_nearest_1d_f32(input, output, batch_size, channels, l_in, l_out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_2x() {
        let input = [1.0, 2.0, 3.0, 4.0f32];
        let mut output = [0.0f32; 16];
        upsample_nearest_2d_f32(&input, &mut output, 1, 1, 2, 2, 4, 4);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[3], 2.0);
        assert_eq!(output[8], 3.0);
    }

    #[test]
    fn test_bilinear_identity() {
        let input = [1.0, 2.0, 3.0, 4.0f32];
        let mut output = [0.0f32; 4];
        upsample_bilinear_2d_f32(&input, &mut output, 1, 1, 2, 2, 2, 2);
        for i in 0..4 {
            assert!((output[i] - input[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bilinear_2x() {
        let input = [0.0, 1.0, 1.0, 0.0f32];
        let mut output = [0.0f32; 9];
        upsample_bilinear_2d_f32(&input, &mut output, 1, 1, 2, 2, 3, 3);
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
        assert!((output[6] - 1.0).abs() < 1e-5);
        assert!((output[8] - 0.0).abs() < 1e-5);
        assert!((output[4] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_nearest_1d() {
        let input = [1.0, 2.0f32];
        let mut output = [0.0f32; 4];
        upsample_nearest_1d_f32(&input, &mut output, 1, 1, 2, 4);
        assert_eq!(output, [1.0, 1.0, 2.0, 2.0]);
    }

    // Checked wrapper tests
    #[test]
    fn test_nearest_2d_checked_valid() {
        let input = [1.0, 2.0, 3.0, 4.0f32];
        let mut output = [0.0f32; 16];
        assert!(upsample_nearest_2d_f32_checked(&input, &mut output, 1, 1, 2, 2, 4, 4).is_ok());
    }

    #[test]
    fn test_nearest_2d_checked_bad_input() {
        let input = [1.0, 2.0, 3.0f32]; // wrong size
        let mut output = [0.0f32; 16];
        assert!(upsample_nearest_2d_f32_checked(&input, &mut output, 1, 1, 2, 2, 4, 4).is_err());
    }

    #[test]
    fn test_nearest_2d_checked_zero_dim() {
        let input: [f32; 0] = [];
        let mut output = [0.0f32; 4];
        assert!(upsample_nearest_2d_f32_checked(&input, &mut output, 1, 1, 0, 2, 2, 2).is_err());
    }

    #[test]
    fn test_nearest_1d_checked_bad_output() {
        let input = [1.0, 2.0f32];
        let mut output = [0.0f32; 3]; // wrong
        assert!(upsample_nearest_1d_f32_checked(&input, &mut output, 1, 1, 2, 4).is_err());
    }
}
