//! Activation functions.

use std::f32::consts::PI;

/// ReLU: max(0, x)
pub fn relu_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].max(0.0);
    }
}

pub fn relu_f32_inplace(x: &mut [f32]) {
    for v in x.iter_mut() { *v = v.max(0.0); }
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = 1.0 / (1.0 + (-x[i]).exp());
    }
}

/// Tanh
pub fn tanh_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].tanh();
    }
}

/// GELU (Gaussian Error Linear Unit) — approximate version
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    let sqrt_2_pi = (2.0 / PI).sqrt();
    for i in 0..x.len() {
        let v = x[i];
        output[i] = 0.5 * v * (1.0 + (sqrt_2_pi * (v + 0.044715 * v * v * v)).tanh());
    }
}

pub fn gelu_f32_inplace(x: &mut [f32]) {
    let sqrt_2_pi = (2.0 / PI).sqrt();
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + (sqrt_2_pi * (*v + 0.044715 * *v * *v * *v)).tanh());
    }
}

/// SiLU (Sigmoid Linear Unit) / Swish: x * sigmoid(x)
pub fn silu_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

pub fn silu_f32_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// LeakyReLU: x if x > 0, alpha * x otherwise
pub fn leaky_relu_f32(x: &[f32], output: &mut [f32], alpha: f32) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = if x[i] > 0.0 { x[i] } else { alpha * x[i] };
    }
}

/// ELU: x if x > 0, alpha * (exp(x) - 1) otherwise
pub fn elu_f32(x: &[f32], output: &mut [f32], alpha: f32) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = if x[i] > 0.0 { x[i] } else { alpha * (x[i].exp() - 1.0) };
    }
}

/// Element-wise multiply: output = a * b
pub fn mul_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() { output[i] = a[i] * b[i]; }
}

pub fn mul_f32_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() { a[i] *= b[i]; }
}

/// Element-wise add in-place: a += b
pub fn add_f32_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() { a[i] += b[i]; }
}

/// Clip values to [min, max] range
pub fn clip_f32(x: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].clamp(min_val, max_val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = [-1.0, 0.0, 1.0, -0.5, 2.0f32];
        let mut out = [0.0f32; 5];
        relu_f32(&x, &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let x = [0.0f32];
        let mut out = [0.0f32; 1];
        sigmoid_f32(&x, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gelu() {
        let x = [0.0, 1.0, -1.0f32];
        let mut out = [0.0f32; 3];
        gelu_f32(&x, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-4);
        assert!((out[1] - 0.8412).abs() < 1e-3);
    }

    #[test]
    fn test_silu() {
        let x = [0.0, 1.0, -1.0f32];
        let mut out = [0.0f32; 3];
        silu_f32(&x, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_leaky_relu() {
        let x = [-1.0, 0.0, 1.0f32];
        let mut out = [0.0f32; 3];
        leaky_relu_f32(&x, &mut out, 0.01);
        assert_eq!(out, [-0.01, 0.0, 1.0]);
    }

    #[test]
    fn test_clip() {
        let x = [-2.0, 0.5, 3.0f32];
        let mut out = [0.0f32; 3];
        clip_f32(&x, &mut out, 0.0, 1.0);
        assert_eq!(out, [0.0, 0.5, 1.0]);
    }
}
