//! Convolution operations.

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
                                    let weight_idx = oc * c_in * kh * kw
                                        + ic * kh * kw
                                        + fh * kw
                                        + fw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }

                    let output_idx = n * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
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
        conv2d_f32(&input, &weight, None, &mut output, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv2d_3x3() {
        // Simple 3x3 avg filter on 3x3 input, no padding
        let input = [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0f32,
        ]; // [1, 1, 3, 3]
        let weight = [1.0 / 9.0; 9]; // [1, 1, 3, 3] averaging kernel
        let mut output = [0.0f32; 1]; // [1, 1, 1, 1]
        conv2d_f32(&input, &weight, None, &mut output, 1, 1, 3, 3, 1, 3, 3, 1, 1, 0, 0);
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let input = [0.0f32; 4]; // [1, 1, 2, 2]
        let weight = [1.0f32]; // [1, 1, 1, 1]
        let bias = [5.0f32];
        let mut output = [0.0f32; 4];
        conv2d_f32(&input, &weight, Some(&bias), &mut output, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
        assert_eq!(output, [5.0, 5.0, 5.0, 5.0]);
    }
}
