//! Convolution operations.

use nnx_core::error::EngineError;

/// 2D convolution: output = conv2d(input, weight) + bias
///
/// Input:  [N, C_in, H, W]
/// Weight: [C_out, C_in, KH, KW]
/// Bias:   [C_out] (optional)
/// Output: [N, C_out, H_out, W_out]
///
/// H_out = (H + 2*pad_h - KH) / stride_h + 1
pub fn conv2d_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch_size: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;

    assert_eq!(input.len(), batch_size * c_in * h * w);
    assert_eq!(weight.len(), c_out * c_in * kh * kw);
    assert_eq!(output.len(), batch_size * c_out * h_out * w_out);

    for n in 0..batch_size {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = bias.map_or(0.0, |b| b[oc]);

                    for ic in 0..c_in {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = oh * stride_h + fh;
                                let iw = ow * stride_w + fw;

                                // Handle padding: check bounds
                                let ih_actual = ih as isize - pad_h as isize;
                                let iw_actual = iw as isize - pad_w as isize;

                                if ih_actual >= 0
                                    && (ih_actual as usize) < h
                                    && iw_actual >= 0
                                    && (iw_actual as usize) < w
                                {
                                    let input_idx = n * c_in * h * w
                                        + ic * h * w
                                        + ih_actual as usize * w
                                        + iw_actual as usize;
                                    let weight_idx =
                                        oc * c_in * kh * kw + ic * kh * kw + fh * kw + fw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }

                    let output_idx =
                        n * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

/// Checked version of `conv2d_f32` that validates dimensions before computing.
pub fn conv2d_f32_checked(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch_size: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> nnx_core::error::Result<()> {
    if stride_h == 0 || stride_w == 0 {
        return Err(EngineError::Kernel(
            "conv2d: stride must be non-zero".to_string(),
        ));
    }
    if h + 2 * pad_h < kh || w + 2 * pad_w < kw {
        return Err(EngineError::ShapeMismatch(format!(
            "conv2d: kernel ({}x{}) too large for padded input ({}x{})",
            kh,
            kw,
            h + 2 * pad_h,
            w + 2 * pad_w
        )));
    }
    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;

    let expected_input = batch_size * c_in * h * w;
    let expected_weight = c_out * c_in * kh * kw;
    let expected_output = batch_size * c_out * h_out * w_out;

    if input.len() != expected_input {
        return Err(EngineError::ShapeMismatch(format!(
            "conv2d: input.len()={} but expected {}",
            input.len(),
            expected_input
        )));
    }
    if weight.len() != expected_weight {
        return Err(EngineError::ShapeMismatch(format!(
            "conv2d: weight.len()={} but expected {}",
            weight.len(),
            expected_weight
        )));
    }
    if output.len() != expected_output {
        return Err(EngineError::ShapeMismatch(format!(
            "conv2d: output.len()={} but expected {}",
            output.len(),
            expected_output
        )));
    }
    if let Some(b) = bias {
        if b.len() != c_out {
            return Err(EngineError::ShapeMismatch(format!(
                "conv2d: bias.len()={} but c_out={}",
                b.len(),
                c_out
            )));
        }
    }
    conv2d_f32(
        input, weight, bias, output, batch_size, c_in, h, w, c_out, kh, kw, stride_h, stride_w,
        pad_h, pad_w,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_identity_kernel() {
        // 1x1 conv with identity kernel = pass-through
        let input = [1.0, 2.0, 3.0, 4.0f32]; // [1, 1, 2, 2]
        let weight = [1.0f32]; // [1, 1, 1, 1]
        let mut output = [0.0f32; 4]; // [1, 1, 2, 2]
        conv2d_f32(
            &input,
            &weight,
            None,
            &mut output,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        );
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv2d_3x3() {
        // Simple 3x3 avg filter on 3x3 input, no padding
        let input = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0f32]; // [1, 1, 3, 3]
        let weight = [1.0 / 9.0; 9]; // [1, 1, 3, 3] averaging kernel
        let mut output = [0.0f32; 1]; // [1, 1, 1, 1]
        conv2d_f32(
            &input,
            &weight,
            None,
            &mut output,
            1,
            1,
            3,
            3,
            1,
            3,
            3,
            1,
            1,
            0,
            0,
        );
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let input = [0.0f32; 4]; // [1, 1, 2, 2]
        let weight = [1.0f32]; // [1, 1, 1, 1]
        let bias = [5.0f32];
        let mut output = [0.0f32; 4];
        conv2d_f32(
            &input,
            &weight,
            Some(&bias),
            &mut output,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        );
        assert_eq!(output, [5.0, 5.0, 5.0, 5.0]);
    }

    // Checked wrapper tests
    #[test]
    fn test_conv2d_checked_valid() {
        let input = [1.0, 2.0, 3.0, 4.0f32];
        let weight = [1.0f32];
        let mut output = [0.0f32; 4];
        let result = conv2d_f32_checked(
            &input,
            &weight,
            None,
            &mut output,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_conv2d_checked_bad_input() {
        let input = [1.0, 2.0f32]; // wrong
        let weight = [1.0f32];
        let mut output = [0.0f32; 4];
        let result = conv2d_f32_checked(
            &input,
            &weight,
            None,
            &mut output,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_checked_zero_stride() {
        let input = [1.0; 4];
        let weight = [1.0f32];
        let mut output = [0.0f32; 4];
        let result = conv2d_f32_checked(
            &input,
            &weight,
            None,
            &mut output,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
        );
        assert!(result.is_err());
    }
}
