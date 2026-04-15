//! CPU implementation of [`KernelBackend`] backed by the SIMD-optimized
//! kernels in this crate.

use nnx_core::backend::KernelBackend;

/// CPU compute backend. Delegates every operation to the SIMD / rayon
/// kernels in `nnx-kernels`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend.
    pub fn new() -> Self {
        Self
    }
}

impl KernelBackend for CpuBackend {
    type Buffer = Vec<f32>;

    // --- Allocation ---

    fn from_f32(&self, data: &[f32]) -> Self::Buffer {
        data.to_vec()
    }

    fn to_f32(&self, buffer: &Self::Buffer) -> Vec<f32> {
        buffer.clone()
    }

    fn zeros(&self, len: usize) -> Self::Buffer {
        vec![0.0f32; len]
    }

    // --- Matrix operations ---

    fn matvec(
        &self,
        matrix: &Self::Buffer,
        x: &Self::Buffer,
        y: &mut Self::Buffer,
        m: usize,
        k: usize,
    ) {
        crate::matmul::matvec_f32(matrix, x, y, m, k);
    }

    fn matvec_bias(
        &self,
        matrix: &Self::Buffer,
        x: &Self::Buffer,
        bias: &Self::Buffer,
        y: &mut Self::Buffer,
        m: usize,
        k: usize,
    ) {
        crate::matmul::matvec_f32(matrix, x, y, m, k);
        crate::activations::add_f32_inplace(y, bias);
    }

    fn dot(&self, a: &Self::Buffer, b: &Self::Buffer) -> f32 {
        crate::matmul::dot_f32(a, b)
    }

    // --- Normalization ---

    fn rms_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        eps: f32,
    ) {
        crate::rms_norm::rms_norm_f32(x, weight, output, eps);
    }

    fn layer_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        hidden_dim: usize,
        eps: f32,
    ) {
        crate::normalization::layer_norm_f32(x, weight, None, output, hidden_dim, eps);
    }

    fn layer_norm_bias(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        bias: &Self::Buffer,
        output: &mut Self::Buffer,
        hidden_dim: usize,
        eps: f32,
    ) {
        crate::normalization::layer_norm_f32(x, weight, Some(bias), output, hidden_dim, eps);
    }

    // --- Activations ---

    fn silu_inplace(&self, x: &mut Self::Buffer) {
        crate::activations::silu_f32_inplace(x);
    }

    fn gelu_inplace(&self, x: &mut Self::Buffer) {
        crate::activations::gelu_f32_inplace(x);
    }

    fn mul_inplace(&self, a: &mut Self::Buffer, b: &Self::Buffer) {
        crate::activations::mul_f32_inplace(a, b);
    }

    fn add_inplace(&self, a: &mut Self::Buffer, b: &Self::Buffer) {
        crate::activations::add_f32_inplace(a, b);
    }

    fn fused_swiglu(&self, gate: &mut Self::Buffer, up: &Self::Buffer) {
        crate::activations::silu_f32_inplace(gate);
        crate::activations::mul_f32_inplace(gate, up);
    }

    fn fused_geglu(&self, gate: &mut Self::Buffer, up: &Self::Buffer) {
        crate::activations::gelu_f32_inplace(gate);
        crate::activations::mul_f32_inplace(gate, up);
    }

    // --- Position encoding ---

    fn rope_inplace(
        &self,
        data: &mut Self::Buffer,
        head_offset: usize,
        head_dim: usize,
        position: usize,
        freq_base: f32,
    ) {
        let slice = &mut data[head_offset..head_offset + head_dim];
        crate::rope::rope_f32(slice, position, freq_base);
    }

    fn partial_rope_inplace(
        &self,
        data: &mut Self::Buffer,
        head_offset: usize,
        _head_dim: usize,
        rotary_dim: usize,
        position: usize,
        freq_base: f32,
    ) {
        if rotary_dim == 0 {
            return;
        }

        let slice = &mut data[head_offset..head_offset + rotary_dim];
        crate::rope::rope_f32(slice, position, freq_base);
    }

    // --- Attention ---

    fn softmax_inplace(&self, data: &mut Self::Buffer, offset: usize, len: usize) {
        let slice = &mut data[offset..offset + len];
        crate::softmax::softmax_f32(slice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matvec() {
        let backend = CpuBackend::new();

        // 2x3 matrix:
        //   [1, 2, 3]
        //   [4, 5, 6]
        let matrix = backend.from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = backend.from_f32(&[1.0, 1.0, 1.0]);
        let mut y = backend.zeros(2);

        backend.matvec(&matrix, &x, &mut y, 2, 3);

        let result = backend.to_f32(&y);
        // Row 0: 1+2+3 = 6
        // Row 1: 4+5+6 = 15
        assert!((result[0] - 6.0).abs() < 1e-5);
        assert!((result[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_matvec_bias() {
        let backend = CpuBackend::new();

        let matrix = backend.from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = backend.from_f32(&[1.0, 1.0, 1.0]);
        let bias = backend.from_f32(&[0.5, -1.0]);
        let mut y = backend.zeros(2);

        backend.matvec_bias(&matrix, &x, &bias, &mut y, 2, 3);

        let result = backend.to_f32(&y);
        assert!((result[0] - 6.5).abs() < 1e-5);
        assert!((result[1] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_dot() {
        let backend = CpuBackend::new();
        let a = backend.from_f32(&[1.0, 2.0, 3.0]);
        let b = backend.from_f32(&[4.0, 5.0, 6.0]);

        let result = backend.dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_rms_norm() {
        let backend = CpuBackend::new();
        let x = backend.from_f32(&[1.0, 1.0, 1.0, 1.0]);
        let weight = backend.from_f32(&[1.0, 1.0, 1.0, 1.0]);
        let mut output = backend.zeros(4);

        backend.rms_norm(&x, &weight, &mut output, 1e-5);

        let result = backend.to_f32(&output);
        // rms([1,1,1,1]) = 1.0, so output ~ [1,1,1,1]
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_cpu_rms_norm_scaling() {
        let backend = CpuBackend::new();
        let x = backend.from_f32(&[2.0, 2.0, 2.0, 2.0]);
        let weight = backend.from_f32(&[1.0, 1.0, 1.0, 1.0]);
        let mut output = backend.zeros(4);

        backend.rms_norm(&x, &weight, &mut output, 1e-5);

        let result = backend.to_f32(&output);
        // rms([2,2,2,2]) = 2.0, so output ~ [1,1,1,1]
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_cpu_softmax() {
        let backend = CpuBackend::new();
        let mut data = backend.from_f32(&[1.0, 2.0, 3.0]);

        backend.softmax_inplace(&mut data, 0, 3);

        let result = backend.to_f32(&data);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1.0, got {sum}"
        );

        // Values should be monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_cpu_softmax_with_offset() {
        let backend = CpuBackend::new();
        // Softmax only the last 3 elements, leaving the first 2 untouched
        let mut data = backend.from_f32(&[10.0, 20.0, 1.0, 2.0, 3.0]);

        backend.softmax_inplace(&mut data, 2, 3);

        let result = backend.to_f32(&data);
        // First two elements unchanged
        assert!((result[0] - 10.0).abs() < 1e-6);
        assert!((result[1] - 20.0).abs() < 1e-6);
        // Last three sum to 1
        let sum: f32 = result[2..5].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_silu() {
        let backend = CpuBackend::new();
        let mut x = backend.from_f32(&[0.0, 1.0, -1.0]);

        backend.silu_inplace(&mut x);

        let result = backend.to_f32(&x);
        // silu(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-6);
        // silu(1) = 1 * sigmoid(1) ~ 0.7311
        assert!((result[1] - 0.7311).abs() < 1e-3);
        // silu(-1) = -1 * sigmoid(-1) ~ -0.2689
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_gelu() {
        let backend = CpuBackend::new();
        let mut x = backend.from_f32(&[0.0, 1.0, -1.0]);

        backend.gelu_inplace(&mut x);

        let result = backend.to_f32(&x);
        // gelu(0) ~ 0
        assert!((result[0] - 0.0).abs() < 1e-4);
        // gelu(1) ~ 0.8412
        assert!((result[1] - 0.8412).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_rope() {
        let backend = CpuBackend::new();

        // At position 0, RoPE is identity (cos(0)=1, sin(0)=0)
        let mut data = backend.from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let original = backend.to_f32(&data);

        backend.rope_inplace(&mut data, 0, 4, 0, 10000.0);

        let result = backend.to_f32(&data);
        for i in 0..4 {
            assert!(
                (result[i] - original[i]).abs() < 1e-6,
                "position 0 should be identity, but element {i} changed: {} -> {}",
                original[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_cpu_rope_nonzero_position() {
        let backend = CpuBackend::new();

        // At position > 0, values should change
        let mut data = backend.from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let original = backend.to_f32(&data);

        backend.rope_inplace(&mut data, 0, 4, 5, 10000.0);

        let result = backend.to_f32(&data);
        // At least some elements should have changed
        let any_changed = result
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_changed, "RoPE at position 5 should modify the values");

        // Norm should be preserved (rotation doesn't change magnitude)
        let orig_norm: f32 = original.iter().map(|v| v * v).sum::<f32>().sqrt();
        let new_norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (orig_norm - new_norm).abs() < 1e-4,
            "RoPE should preserve norm: {orig_norm} vs {new_norm}"
        );
    }

    #[test]
    fn test_cpu_rope_with_offset() {
        let backend = CpuBackend::new();

        // Buffer with multiple heads; only apply RoPE to the second head
        let mut data = backend.from_f32(&[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]);
        let original = backend.to_f32(&data);

        backend.rope_inplace(&mut data, 4, 4, 5, 10000.0);

        let result = backend.to_f32(&data);
        // First head untouched
        for i in 0..4 {
            assert!((result[i] - original[i]).abs() < 1e-6);
        }
        // Second head modified
        let any_changed = result[4..8]
            .iter()
            .zip(original[4..8].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_changed, "second head should be modified by RoPE");
    }

    #[test]
    fn test_cpu_partial_rope_rotates_prefix_only() {
        let backend = CpuBackend::new();

        let mut data = backend.from_f32(&[
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]);
        let original = backend.to_f32(&data);

        let mut expected = original.clone();
        crate::rope::rope_f32(&mut expected[6..10], 3, 10000.0);

        backend.partial_rope_inplace(&mut data, 6, 6, 4, 3, 10000.0);

        let result = backend.to_f32(&data);
        assert_eq!(
            &result[0..6],
            &original[0..6],
            "first head should be untouched"
        );
        assert_eq!(
            &result[10..12],
            &original[10..12],
            "non-rotary suffix should be untouched"
        );
        for i in 6..10 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "rotary prefix mismatch at {i}: {} vs {}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_cpu_mul_inplace() {
        let backend = CpuBackend::new();
        let mut a = backend.from_f32(&[1.0, 2.0, 3.0]);
        let b = backend.from_f32(&[4.0, 5.0, 6.0]);

        backend.mul_inplace(&mut a, &b);

        let result = backend.to_f32(&a);
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 10.0).abs() < 1e-6);
        assert!((result[2] - 18.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_fused_swiglu_matches_separate_path() {
        let backend = CpuBackend::new();
        let gate_data = [0.5, 1.0, -1.0, 2.0];
        let up = backend.from_f32(&[1.0, 2.0, 0.5, 3.0]);

        let mut expected = backend.from_f32(&gate_data);
        backend.silu_inplace(&mut expected);
        backend.mul_inplace(&mut expected, &up);

        let mut fused = backend.from_f32(&gate_data);
        backend.fused_swiglu(&mut fused, &up);

        let expected = backend.to_f32(&expected);
        let result = backend.to_f32(&fused);
        for i in 0..expected.len() {
            assert!((result[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cpu_fused_geglu_matches_separate_path() {
        let backend = CpuBackend::new();
        let gate_data = [0.5, 1.0, -1.0, 2.0];
        let up = backend.from_f32(&[1.0, 2.0, 0.5, 3.0]);

        let mut expected = backend.from_f32(&gate_data);
        backend.gelu_inplace(&mut expected);
        backend.mul_inplace(&mut expected, &up);

        let mut fused = backend.from_f32(&gate_data);
        backend.fused_geglu(&mut fused, &up);

        let expected = backend.to_f32(&expected);
        let result = backend.to_f32(&fused);
        for i in 0..expected.len() {
            assert!((result[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cpu_add_inplace() {
        let backend = CpuBackend::new();
        let mut a = backend.from_f32(&[1.0, 2.0, 3.0]);
        let b = backend.from_f32(&[10.0, 20.0, 30.0]);

        backend.add_inplace(&mut a, &b);

        let result = backend.to_f32(&a);
        assert!((result[0] - 11.0).abs() < 1e-6);
        assert!((result[1] - 22.0).abs() < 1e-6);
        assert!((result[2] - 33.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_layer_norm() {
        let backend = CpuBackend::new();
        let x = backend.from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let weight = backend.from_f32(&[1.0, 1.0, 1.0, 1.0]);
        let mut output = backend.zeros(4);

        backend.layer_norm(&x, &weight, &mut output, 4, 1e-5);

        let result = backend.to_f32(&output);
        // Should be zero-mean after normalization
        let sum: f32 = result.iter().sum();
        assert!(
            sum.abs() < 1e-4,
            "layer norm output should be zero-mean, got sum={sum}"
        );
    }

    #[test]
    fn test_cpu_layer_norm_bias() {
        let backend = CpuBackend::new();
        let x = backend.from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let weight = backend.from_f32(&[1.0, 1.0, 1.0, 1.0]);
        let bias = backend.from_f32(&[10.0, 10.0, 10.0, 10.0]);
        let mut output = backend.zeros(4);

        backend.layer_norm_bias(&x, &weight, &bias, &mut output, 4, 1e-5);

        let result = backend.to_f32(&output);
        // Mean should be shifted by 10
        let mean: f32 = result.iter().sum::<f32>() / 4.0;
        assert!(
            (mean - 10.0).abs() < 1e-3,
            "biased layer norm mean should be ~10, got {mean}"
        );
    }

    #[test]
    fn test_cpu_zeros() {
        let backend = CpuBackend::new();
        let buf = backend.zeros(5);
        let result = backend.to_f32(&buf);
        assert_eq!(result.len(), 5);
        for &v in &result {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_cpu_from_to_f32_roundtrip() {
        let backend = CpuBackend::new();
        let data = [1.5, -2.3, 0.0, 42.0, f32::MIN, f32::MAX];
        let buf = backend.from_f32(&data);
        let result = backend.to_f32(&buf);
        assert_eq!(result.as_slice(), &data);
    }
}
