//! Normalization layers: LayerNorm, BatchNorm.
//!
//! RMSNorm is in its own module (rms_norm.rs) since it's the hot path
//! for Llama inference and has SIMD optimization.

use crate::matmul::dot_f32;

use nnx_core::error::EngineError;

/// Layer Normalization: output = ((x - mean) / sqrt(var + eps)) * weight + bias
///
/// Normalizes across the last `norm_size` elements.
/// `x` can be any shape; we normalize the trailing dimension.
pub fn layer_norm_f32(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    norm_size: usize,
    eps: f32,
) {
    assert_eq!(x.len(), output.len());
    assert_eq!(weight.len(), norm_size);
    assert!(x.len() % norm_size == 0);

    let num_groups = x.len() / norm_size;

    for g in 0..num_groups {
        let start = g * norm_size;
        let group = &x[start..start + norm_size];

        // Compute mean
        let mean: f32 = group.iter().sum::<f32>() / norm_size as f32;

        // Compute variance
        let var: f32 = group.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / norm_size as f32;

        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..norm_size {
            let normalized = (group[i] - mean) * inv_std;
            output[start + i] = normalized * weight[i]
                + bias.map_or(0.0, |b| b[i]);
        }
    }
}

/// Batch Normalization (inference mode only):
/// output = ((x - running_mean) / sqrt(running_var + eps)) * scale + bias
///
/// Input x has shape [N, C, ...] where C is the channel dimension.
/// scale, bias, running_mean, running_var all have shape [C].
pub fn batch_norm_f32(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    output: &mut [f32],
    num_channels: usize,
    spatial_size: usize, // H*W for 2D, or 1 for 1D
    eps: f32,
) {
    assert_eq!(x.len(), output.len());
    assert_eq!(scale.len(), num_channels);
    assert_eq!(bias.len(), num_channels);

    let batch_size = x.len() / (num_channels * spatial_size);

    for n in 0..batch_size {
        for c in 0..num_channels {
            let inv_std = 1.0 / (running_var[c] + eps).sqrt();
            let offset = n * num_channels * spatial_size + c * spatial_size;

            for s in 0..spatial_size {
                let idx = offset + s;
                output[idx] = (x[idx] - running_mean[c]) * inv_std * scale[c] + bias[c];
            }
        }
    }
}

// -- Checked wrappers --

/// Checked version of `layer_norm_f32` that validates dimensions before computing.
pub fn layer_norm_f32_checked(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    norm_size: usize,
    eps: f32,
) -> nnx_core::error::Result<()> {
    if x.len() != output.len() {
        return Err(EngineError::ShapeMismatch(
            format!("layer_norm: x.len()={} != output.len()={}", x.len(), output.len())
        ));
    }
    if weight.len() != norm_size {
        return Err(EngineError::ShapeMismatch(
            format!("layer_norm: weight.len()={} != norm_size={}", weight.len(), norm_size)
        ));
    }
    if norm_size == 0 {
        return Err(EngineError::ShapeMismatch(
            "layer_norm: norm_size must be non-zero".to_string()
        ));
    }
    if x.len() % norm_size != 0 {
        return Err(EngineError::ShapeMismatch(
            format!("layer_norm: x.len()={} is not divisible by norm_size={}", x.len(), norm_size)
        ));
    }
    if let Some(b) = bias {
        if b.len() != norm_size {
            return Err(EngineError::ShapeMismatch(
                format!("layer_norm: bias.len()={} != norm_size={}", b.len(), norm_size)
            ));
        }
    }
    layer_norm_f32(x, weight, bias, output, norm_size, eps);
    Ok(())
}

/// Checked version of `batch_norm_f32` that validates dimensions before computing.
pub fn batch_norm_f32_checked(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    output: &mut [f32],
    num_channels: usize,
    spatial_size: usize,
    eps: f32,
) -> nnx_core::error::Result<()> {
    if x.len() != output.len() {
        return Err(EngineError::ShapeMismatch(
            format!("batch_norm: x.len()={} != output.len()={}", x.len(), output.len())
        ));
    }
    if scale.len() != num_channels {
        return Err(EngineError::ShapeMismatch(
            format!("batch_norm: scale.len()={} != num_channels={}", scale.len(), num_channels)
        ));
    }
    if bias.len() != num_channels {
        return Err(EngineError::ShapeMismatch(
            format!("batch_norm: bias.len()={} != num_channels={}", bias.len(), num_channels)
        ));
    }
    if running_mean.len() != num_channels {
        return Err(EngineError::ShapeMismatch(
            format!("batch_norm: running_mean.len()={} != num_channels={}", running_mean.len(), num_channels)
        ));
    }
    if running_var.len() != num_channels {
        return Err(EngineError::ShapeMismatch(
            format!("batch_norm: running_var.len()={} != num_channels={}", running_var.len(), num_channels)
        ));
    }
    if num_channels == 0 || spatial_size == 0 {
        return Err(EngineError::ShapeMismatch(
            "batch_norm: num_channels and spatial_size must be non-zero".to_string()
        ));
    }
    if x.len() % (num_channels * spatial_size) != 0 {
        return Err(EngineError::ShapeMismatch(
            format!("batch_norm: x.len()={} not divisible by num_channels*spatial_size={}", x.len(), num_channels * spatial_size)
        ));
    }
    batch_norm_f32(x, scale, bias, running_mean, running_var, output, num_channels, spatial_size, eps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let x = [1.0, 2.0, 3.0, 4.0f32]; // norm_size=4
        let w = [1.0, 1.0, 1.0, 1.0f32];
        let b = [0.0, 0.0, 0.0, 0.0f32];
        let mut out = [0.0f32; 4];
        layer_norm_f32(&x, &w, Some(&b), &mut out, 4, 1e-5);

        // Mean=2.5, Var=1.25, should be centered and scaled
        let sum: f32 = out.iter().sum();
        assert!(sum.abs() < 1e-4, "should be zero-mean: sum={}", sum);
    }

    #[test]
    fn test_batch_norm() {
        // 1 batch, 2 channels, spatial=2
        let x = [1.0, 2.0, 3.0, 4.0f32]; // [N=1, C=2, S=2]
        let scale = [1.0, 1.0f32];
        let bias = [0.0, 0.0f32];
        let mean = [1.5, 3.5f32]; // running means
        let var = [0.25, 0.25f32]; // running vars
        let mut out = [0.0f32; 4];
        batch_norm_f32(&x, &scale, &bias, &mean, &var, &mut out, 2, 2, 1e-5);

        // Channel 0: (1.0-1.5)/0.5=-1.0, (2.0-1.5)/0.5=1.0
        assert!((out[0] - (-1.0)).abs() < 1e-3);
        assert!((out[1] - 1.0).abs() < 1e-3);
    }

    // Checked wrapper tests
    #[test]
    fn test_layer_norm_checked_valid() {
        let x = [1.0, 2.0, 3.0, 4.0f32];
        let w = [1.0, 1.0, 1.0, 1.0f32];
        let b = [0.0, 0.0, 0.0, 0.0f32];
        let mut out = [0.0f32; 4];
        let result = layer_norm_f32_checked(&x, &w, Some(&b), &mut out, 4, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_layer_norm_checked_bad_norm_size() {
        let x = [1.0, 2.0, 3.0f32]; // not divisible by norm_size=2
        let w = [1.0, 1.0f32];
        let mut out = [0.0f32; 3];
        let result = layer_norm_f32_checked(&x, &w, None, &mut out, 2, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_checked_bad_weight() {
        let x = [1.0, 2.0, 3.0, 4.0f32];
        let w = [1.0, 1.0f32]; // wrong size for norm_size=4
        let mut out = [0.0f32; 4];
        let result = layer_norm_f32_checked(&x, &w, None, &mut out, 4, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_checked_output_mismatch() {
        let x = [1.0, 2.0, 3.0, 4.0f32];
        let w = [1.0, 1.0, 1.0, 1.0f32];
        let mut out = [0.0f32; 3]; // wrong size
        let result = layer_norm_f32_checked(&x, &w, None, &mut out, 4, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_norm_checked_valid() {
        let x = [1.0, 2.0, 3.0, 4.0f32];
        let scale = [1.0, 1.0f32];
        let bias = [0.0, 0.0f32];
        let mean = [1.5, 3.5f32];
        let var = [0.25, 0.25f32];
        let mut out = [0.0f32; 4];
        let result = batch_norm_f32_checked(&x, &scale, &bias, &mean, &var, &mut out, 2, 2, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_norm_checked_bad_scale() {
        let x = [1.0, 2.0, 3.0, 4.0f32];
        let scale = [1.0f32]; // wrong for num_channels=2
        let bias = [0.0, 0.0f32];
        let mean = [1.5, 3.5f32];
        let var = [0.25, 0.25f32];
        let mut out = [0.0f32; 4];
        let result = batch_norm_f32_checked(&x, &scale, &bias, &mean, &var, &mut out, 2, 2, 1e-5);
        assert!(result.is_err());
    }
}
