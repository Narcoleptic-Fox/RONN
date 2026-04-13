//! Activation functions.

/// SiLU (Sigmoid Linear Unit) — used by Llama's SwiGLU FFN.
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

/// SiLU in-place.
pub fn silu_f32_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// Element-wise multiply: output = a * b
pub fn mul_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

/// Element-wise multiply in-place: a *= b
pub fn mul_f32_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// Element-wise add in-place: a += b
pub fn add_f32_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        let x = [0.0, 1.0, -1.0f32];
        let mut out = [0.0f32; 3];
        silu_f32(&x, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.7310586).abs() < 1e-4);
        assert!((out[2] - (-0.2689414)).abs() < 1e-4);
    }
}
