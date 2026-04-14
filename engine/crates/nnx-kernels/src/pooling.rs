//! Pooling operations: MaxPool2D, AvgPool2D, GlobalAvgPool.

use nnx_core::error::EngineError;

/// Max pooling 2D
/// Input: [N, C, H, W], Output: [N, C, H_out, W_out]
pub fn max_pool2d_f32(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;

                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = (oh * stride_h + fh) as isize - pad_h as isize;
                            let iw = (ow * stride_w + fw) as isize - pad_w as isize;

                            if ih >= 0 && (ih as usize) < h && iw >= 0 && (iw as usize) < w {
                                let idx = n * channels * h * w
                                    + c * h * w
                                    + ih as usize * w
                                    + iw as usize;
                                max_val = max_val.max(input[idx]);
                            }
                        }
                    }

                    let out_idx =
                        n * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

/// Average pooling 2D
pub fn avg_pool2d_f32(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;

    for n in 0..batch_size {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;
                    let mut count = 0u32;

                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = (oh * stride_h + fh) as isize - pad_h as isize;
                            let iw = (ow * stride_w + fw) as isize - pad_w as isize;

                            if ih >= 0 && (ih as usize) < h && iw >= 0 && (iw as usize) < w {
                                let idx = n * channels * h * w
                                    + c * h * w
                                    + ih as usize * w
                                    + iw as usize;
                                sum += input[idx];
                                count += 1;
                            }
                        }
                    }

                    let out_idx =
                        n * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow;
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }
}

/// Global average pooling: average over all spatial dimensions
/// Input: [N, C, H, W], Output: [N, C]
pub fn global_avg_pool_f32(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    spatial_size: usize, // H * W
) {
    for n in 0..batch_size {
        for c in 0..channels {
            let offset = n * channels * spatial_size + c * spatial_size;
            let sum: f32 = input[offset..offset + spatial_size].iter().sum();
            output[n * channels + c] = sum / spatial_size as f32;
        }
    }
}

/// Checked version of `global_avg_pool_f32`.
pub fn global_avg_pool_f32_checked(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    spatial_size: usize,
) -> nnx_core::error::Result<()> {
    let expected_in = batch_size * channels * spatial_size;
    let expected_out = batch_size * channels;
    if input.len() != expected_in {
        return Err(EngineError::ShapeMismatch(format!(
            "global_avg_pool: input.len()={} but expected {}",
            input.len(),
            expected_in
        )));
    }
    if output.len() != expected_out {
        return Err(EngineError::ShapeMismatch(format!(
            "global_avg_pool: output.len()={} but expected {}",
            output.len(),
            expected_out
        )));
    }
    if spatial_size == 0 {
        return Err(EngineError::ShapeMismatch(
            "global_avg_pool: spatial_size must be non-zero".to_string(),
        ));
    }
    global_avg_pool_f32(input, output, batch_size, channels, spatial_size);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool_2x2() {
        let input = [1.0, 2.0, 3.0, 4.0f32]; // [1, 1, 2, 2]
        let mut output = [0.0f32; 1]; // 2x2 pool on 2x2 = 1x1
        max_pool2d_f32(&input, &mut output, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0);
        assert_eq!(output[0], 4.0);
    }

    #[test]
    fn test_avg_pool_2x2() {
        let input = [1.0, 2.0, 3.0, 4.0f32];
        let mut output = [0.0f32; 1];
        avg_pool2d_f32(&input, &mut output, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0);
        assert!((output[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_global_avg_pool() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0f32]; // [1, 2, 2, 2]
        let mut output = [0.0f32; 2]; // [1, 2]
        global_avg_pool_f32(&input, &mut output, 1, 2, 4);
        assert!((output[0] - 2.5).abs() < 1e-5);
        assert!((output[1] - 25.0).abs() < 1e-5);
    }

    // Checked wrapper tests
    #[test]
    fn test_global_avg_pool_checked_valid() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0f32];
        let mut output = [0.0f32; 2];
        assert!(global_avg_pool_f32_checked(&input, &mut output, 1, 2, 4).is_ok());
    }

    #[test]
    fn test_global_avg_pool_checked_bad_input() {
        let input = [1.0, 2.0f32]; // wrong
        let mut output = [0.0f32; 2];
        assert!(global_avg_pool_f32_checked(&input, &mut output, 1, 2, 4).is_err());
    }

    #[test]
    fn test_global_avg_pool_checked_zero_spatial() {
        let input: [f32; 0] = [];
        let mut output = [0.0f32; 2];
        assert!(global_avg_pool_f32_checked(&input, &mut output, 1, 2, 0).is_err());
    }
}
