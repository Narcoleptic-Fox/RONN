//! GPU normalization kernels: RMS norm and Layer norm.

use cubecl::prelude::*;

/// RMS normalization kernel.
///
/// For each element: `output[i] = (x[i] / rms) * weight[i]`
/// where `rms = sqrt(mean(x^2) + eps)`.
///
/// Launch one cube per vector, with `CubeDim.x >= hidden_dim`.
#[cube(launch)]
pub fn rms_norm_kernel(
    input: &Array<f32>,
    weight: &Array<f32>,
    output: &mut Array<f32>,
    hidden_dim: u32,
    eps: f32,
) {
    let vec_offset = CUBE_POS_X * hidden_dim;
    let tid = UNIT_POS_X;

    // Step 1: Compute sum of squares (thread 0 serial — correct first, optimize later).
    // Each thread computes its element's contribution.
    if tid == 0u32 {
        let mut sum_sq = 0.0f32;
        for i in 0..hidden_dim {
            let val = input[vec_offset + i];
            sum_sq += val * val;
        }
        let rms = f32::sqrt(sum_sq / f32::cast_from(hidden_dim) + eps);

        // Normalize and scale all elements.
        for i in 0..hidden_dim {
            let val = input[vec_offset + i];
            output[vec_offset + i] = (val / rms) * weight[i];
        }
    }
}

/// Layer normalization kernel (without bias).
///
/// `output[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i]`
#[cube(launch)]
pub fn layer_norm_kernel(
    input: &Array<f32>,
    weight: &Array<f32>,
    output: &mut Array<f32>,
    hidden_dim: u32,
    eps: f32,
) {
    let vec_offset = CUBE_POS_X * hidden_dim;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        // Compute mean.
        let mut mean = 0.0f32;
        for i in 0..hidden_dim {
            mean += input[vec_offset + i];
        }
        mean /= f32::cast_from(hidden_dim);

        // Compute variance.
        let mut var = 0.0f32;
        for i in 0..hidden_dim {
            let diff = input[vec_offset + i] - mean;
            var += diff * diff;
        }
        var /= f32::cast_from(hidden_dim);

        let inv_std = 1.0f32 / f32::sqrt(var + eps);

        // Normalize and scale.
        for i in 0..hidden_dim {
            let normed = (input[vec_offset + i] - mean) * inv_std;
            output[vec_offset + i] = normed * weight[i];
        }
    }
}

/// Layer normalization kernel with bias.
///
/// `output[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]`
#[cube(launch)]
pub fn layer_norm_bias_kernel(
    input: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    hidden_dim: u32,
    eps: f32,
) {
    let vec_offset = CUBE_POS_X * hidden_dim;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let mut mean = 0.0f32;
        for i in 0..hidden_dim {
            mean += input[vec_offset + i];
        }
        mean /= f32::cast_from(hidden_dim);

        let mut var = 0.0f32;
        for i in 0..hidden_dim {
            let diff = input[vec_offset + i] - mean;
            var += diff * diff;
        }
        var /= f32::cast_from(hidden_dim);

        let inv_std = 1.0f32 / f32::sqrt(var + eps);

        for i in 0..hidden_dim {
            let normed = (input[vec_offset + i] - mean) * inv_std;
            output[vec_offset + i] = normed * weight[i] + bias[i];
        }
    }
}
