//! Integration tests for `CubeclBackend` through the `KernelBackend` trait.
//!
//! These tests prove the GPU backend abstraction works end-to-end by
//! running every trait method on the GPU and comparing results against
//! CPU reference implementations.
//!
//! Run with: `cargo test -p nnx-cubecl --features wgpu`

#![cfg(feature = "wgpu")]

use cubecl::wgpu::WgpuRuntime;
use nnx_core::backend::KernelBackend;
use nnx_cubecl::CubeclBackend;

fn backend() -> CubeclBackend<WgpuRuntime> {
    CubeclBackend::<WgpuRuntime>::new()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{msg}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "{msg}: mismatch at [{i}]: gpu={av}, cpu={bv}, diff={}",
            (av - bv).abs(),
        );
    }
}

// -----------------------------------------------------------------------
// CPU reference implementations
// -----------------------------------------------------------------------

fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

fn cpu_gelu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| {
            let cdf = 0.5 * (1.0 + (0.7978845608 * (v + 0.044715 * v * v * v)).tanh());
            v * cdf
        })
        .collect()
}

fn cpu_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| (v / rms) * w)
        .collect()
}

fn cpu_layer_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| ((v - mean) * inv_std) * w)
        .collect()
}

fn cpu_layer_norm_bias(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((v, w), b)| ((v - mean) * inv_std) * w + b)
        .collect()
}

fn cpu_matvec(matrix: &[f32], vector: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; m];
    for row in 0..m {
        for col in 0..k {
            out[row] += matrix[row * k + col] * vector[col];
        }
    }
    out
}

fn cpu_matvec_bias(matrix: &[f32], vector: &[f32], bias: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = cpu_matvec(matrix, vector, m, k);
    for (value, bias_value) in out.iter_mut().zip(bias.iter()) {
        *value += bias_value;
    }
    out
}

fn cpu_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

fn cpu_rope(data: &mut [f32], head_dim: usize, position: usize, freq_base: f32) {
    let half = head_dim / 2;
    for pair in 0..half {
        let dim_frac = (2 * pair) as f32 / head_dim as f32;
        let freq = 1.0 / freq_base.powf(dim_frac);
        let angle = position as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let x0 = data[2 * pair];
        let x1 = data[2 * pair + 1];
        data[2 * pair] = x0 * cos - x1 * sin;
        data[2 * pair + 1] = x0 * sin + x1 * cos;
    }
}

fn cpu_partial_rope(
    data: &mut [f32],
    head_offset: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    freq_base: f32,
) {
    assert!(rotary_dim <= head_dim);
    let slice = &mut data[head_offset..head_offset + rotary_dim];
    let half = rotary_dim / 2;
    for pair in 0..half {
        let dim_frac = (2 * pair) as f32 / rotary_dim as f32;
        let freq = 1.0 / freq_base.powf(dim_frac);
        let angle = position as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let x0 = slice[pair];
        let x1 = slice[pair + half];
        slice[pair] = x0 * cos - x1 * sin;
        slice[pair + half] = x0 * sin + x1 * cos;
    }
}

// -----------------------------------------------------------------------
// Allocation tests
// -----------------------------------------------------------------------

#[test]
fn test_from_to_f32_roundtrip() {
    let b = backend();
    let data = [1.5, -2.3, 0.0, 42.0, -100.0, 0.001];
    let buf = b.from_f32(&data);
    let result = b.to_f32(&buf);
    assert_close(&result, &data, 1e-6, "from_to_f32_roundtrip");
}

#[test]
fn test_zeros() {
    let b = backend();
    let buf = b.zeros(8);
    let result = b.to_f32(&buf);
    assert_eq!(result.len(), 8);
    for &v in &result {
        assert!((v - 0.0).abs() < f32::EPSILON, "expected zero, got {v}");
    }
}

// -----------------------------------------------------------------------
// Matrix operation tests
// -----------------------------------------------------------------------

#[test]
fn test_matvec() {
    let b = backend();
    // 3x4 matrix
    let matrix_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let vector_data = vec![1.0, 0.5, -1.0, 2.0];
    let m = 3;
    let k = 4;

    let expected = cpu_matvec(&matrix_data, &vector_data, m, k);

    let matrix = b.from_f32(&matrix_data);
    let x = b.from_f32(&vector_data);
    let mut y = b.zeros(m);

    b.matvec(&matrix, &x, &mut y, m, k);

    let result = b.to_f32(&y);
    assert_close(&result, &expected, 1e-4, "matvec");
}

#[test]
fn test_matvec_bias() {
    let b = backend();
    let matrix_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let vector_data = vec![1.0, 0.5, -1.0, 2.0];
    let bias_data = vec![0.25, -1.5, 3.0];
    let m = 3;
    let k = 4;

    let expected = cpu_matvec_bias(&matrix_data, &vector_data, &bias_data, m, k);

    let matrix = b.from_f32(&matrix_data);
    let x = b.from_f32(&vector_data);
    let bias = b.from_f32(&bias_data);
    let mut y = b.zeros(m);

    b.matvec_bias(&matrix, &x, &bias, &mut y, m, k);

    let result = b.to_f32(&y);
    assert_close(&result, &expected, 1e-4, "matvec_bias");
}

#[test]
fn test_dot() {
    let b = backend();
    let a = b.from_f32(&[1.0, 2.0, 3.0, 4.0]);
    let bv = b.from_f32(&[4.0, 3.0, 2.0, 1.0]);

    let result = b.dot(&a, &bv);
    // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
    assert!(
        (result - 20.0).abs() < 1e-4,
        "dot: expected 20.0, got {result}"
    );
}

// -----------------------------------------------------------------------
// Normalization tests
// -----------------------------------------------------------------------

#[test]
fn test_rms_norm() {
    let b = backend();
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 2.5];
    let weight_data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let expected = cpu_rms_norm(&x_data, &weight_data, eps);

    let x = b.from_f32(&x_data);
    let weight = b.from_f32(&weight_data);
    let mut output = b.zeros(x_data.len());

    b.rms_norm(&x, &weight, &mut output, eps);

    let result = b.to_f32(&output);
    assert_close(&result, &expected, 1e-4, "rms_norm");
}

#[test]
fn test_rms_norm_with_weights() {
    let b = backend();
    let x_data = vec![2.0, 2.0, 2.0, 2.0];
    let weight_data = vec![0.5, 1.0, 2.0, 3.0];
    let eps = 1e-5;

    let expected = cpu_rms_norm(&x_data, &weight_data, eps);

    let x = b.from_f32(&x_data);
    let weight = b.from_f32(&weight_data);
    let mut output = b.zeros(4);

    b.rms_norm(&x, &weight, &mut output, eps);

    let result = b.to_f32(&output);
    assert_close(&result, &expected, 1e-4, "rms_norm_with_weights");
}

#[test]
fn test_layer_norm() {
    let b = backend();
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let weight_data = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let expected = cpu_layer_norm(&x_data, &weight_data, eps);

    let x = b.from_f32(&x_data);
    let weight = b.from_f32(&weight_data);
    let mut output = b.zeros(4);

    b.layer_norm(&x, &weight, &mut output, 4, eps);

    let result = b.to_f32(&output);
    assert_close(&result, &expected, 1e-4, "layer_norm");

    // Output should be zero-mean
    let sum: f32 = result.iter().sum();
    assert!(
        sum.abs() < 1e-4,
        "layer_norm should be zero-mean, got sum={sum}"
    );
}

#[test]
fn test_layer_norm_bias() {
    let b = backend();
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let weight_data = vec![1.0, 1.0, 1.0, 1.0];
    let bias_data = vec![10.0, 10.0, 10.0, 10.0];
    let eps = 1e-5;

    let expected = cpu_layer_norm_bias(&x_data, &weight_data, &bias_data, eps);

    let x = b.from_f32(&x_data);
    let weight = b.from_f32(&weight_data);
    let bias = b.from_f32(&bias_data);
    let mut output = b.zeros(4);

    b.layer_norm_bias(&x, &weight, &bias, &mut output, 4, eps);

    let result = b.to_f32(&output);
    assert_close(&result, &expected, 1e-4, "layer_norm_bias");

    // Mean should be shifted by 10
    let mean: f32 = result.iter().sum::<f32>() / 4.0;
    assert!(
        (mean - 10.0).abs() < 1e-3,
        "biased layer norm mean should be ~10, got {mean}"
    );
}

// -----------------------------------------------------------------------
// Activation tests
// -----------------------------------------------------------------------

#[test]
fn test_silu_inplace() {
    let b = backend();
    let data = vec![0.0, 1.0, -1.0, 2.0, -0.5, 0.5, 3.0, -3.0];
    let expected = cpu_silu(&data);

    let mut buf = b.from_f32(&data);
    b.silu_inplace(&mut buf);

    let result = b.to_f32(&buf);
    assert_close(&result, &expected, 1e-5, "silu_inplace");
}

#[test]
fn test_gelu_inplace() {
    let b = backend();
    let data = vec![0.0, 1.0, -1.0, 2.0, -0.5, 0.5, 3.0, -3.0];
    let expected = cpu_gelu(&data);

    let mut buf = b.from_f32(&data);
    b.gelu_inplace(&mut buf);

    let result = b.to_f32(&buf);
    assert_close(&result, &expected, 1e-5, "gelu_inplace");
}

#[test]
fn test_mul_inplace() {
    let b = backend();
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![4.0, 3.0, 2.0, 1.0];
    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x * y)
        .collect();

    let mut a = b.from_f32(&a_data);
    let bv = b.from_f32(&b_data);
    b.mul_inplace(&mut a, &bv);

    let result = b.to_f32(&a);
    assert_close(&result, &expected, 1e-6, "mul_inplace");
}

#[test]
fn test_add_inplace() {
    let b = backend();
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![10.0, 20.0, 30.0, 40.0];
    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();

    let mut a = b.from_f32(&a_data);
    let bv = b.from_f32(&b_data);
    b.add_inplace(&mut a, &bv);

    let result = b.to_f32(&a);
    assert_close(&result, &expected, 1e-6, "add_inplace");
}

#[test]
fn test_fused_swiglu_matches_separate_path() {
    let b = backend();
    let gate_data = vec![0.5, 1.0, -1.0, 2.0, -0.25, 0.75];
    let up_data = vec![1.0, 2.0, 0.5, 3.0, 4.0, -2.0];

    let up = b.from_f32(&up_data);

    let mut expected = b.from_f32(&gate_data);
    b.silu_inplace(&mut expected);
    b.mul_inplace(&mut expected, &up);

    let mut fused = b.from_f32(&gate_data);
    b.fused_swiglu(&mut fused, &up);

    let expected = b.to_f32(&expected);
    let result = b.to_f32(&fused);
    assert_close(&result, &expected, 1e-5, "fused_swiglu");
}

#[test]
fn test_fused_geglu_matches_separate_path() {
    let b = backend();
    let gate_data = vec![0.5, 1.0, -1.0, 2.0, -0.25, 0.75];
    let up_data = vec![1.0, 2.0, 0.5, 3.0, 4.0, -2.0];

    let up = b.from_f32(&up_data);

    let mut expected = b.from_f32(&gate_data);
    b.gelu_inplace(&mut expected);
    b.mul_inplace(&mut expected, &up);

    let mut fused = b.from_f32(&gate_data);
    b.fused_geglu(&mut fused, &up);

    let expected = b.to_f32(&expected);
    let result = b.to_f32(&fused);
    assert_close(&result, &expected, 1e-5, "fused_geglu");
}

// -----------------------------------------------------------------------
// Softmax test
// -----------------------------------------------------------------------

#[test]
fn test_softmax_inplace() {
    let b = backend();
    let data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.5, -1.0, 3.0];
    let expected = cpu_softmax(&data);

    let mut buf = b.from_f32(&data);
    b.softmax_inplace(&mut buf, 0, data.len());

    let result = b.to_f32(&buf);
    assert_close(&result, &expected, 1e-5, "softmax_inplace");

    // Should sum to 1
    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_softmax_with_offset() {
    let b = backend();
    // Apply softmax only to elements [2..5], leaving [0..2] untouched
    let data = vec![10.0, 20.0, 1.0, 2.0, 3.0];
    let expected_tail = cpu_softmax(&data[2..5]);

    let mut buf = b.from_f32(&data);
    b.softmax_inplace(&mut buf, 2, 3);

    let result = b.to_f32(&buf);
    // First two unchanged
    assert!(
        (result[0] - 10.0).abs() < 1e-6,
        "element 0 should be untouched"
    );
    assert!(
        (result[1] - 20.0).abs() < 1e-6,
        "element 1 should be untouched"
    );
    // Last three = softmax result
    assert_close(&result[2..5], &expected_tail, 1e-5, "softmax_with_offset");
}

// -----------------------------------------------------------------------
// RoPE test
// -----------------------------------------------------------------------

#[test]
fn test_rope_position_zero_is_identity() {
    let b = backend();
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let mut buf = b.from_f32(&data);
    b.rope_inplace(&mut buf, 0, 4, 0, 10000.0);

    let result = b.to_f32(&buf);
    assert_close(&result, &data, 1e-6, "rope_position_zero");
}

#[test]
fn test_rope_nonzero_position() {
    let b = backend();
    let head_dim = 8;
    let position = 5;
    let freq_base = 10000.0;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut expected = input.clone();
    cpu_rope(&mut expected, head_dim, position, freq_base);

    let mut buf = b.from_f32(&input);
    b.rope_inplace(&mut buf, 0, head_dim, position, freq_base);

    let result = b.to_f32(&buf);
    assert_close(&result, &expected, 1e-4, "rope_nonzero");
}

#[test]
fn test_rope_with_offset() {
    let b = backend();
    // Buffer with two heads; apply RoPE only to the second
    let input = vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0];
    let mut expected_tail = input[4..8].to_vec();
    cpu_rope(&mut expected_tail, 4, 5, 10000.0);

    let mut buf = b.from_f32(&input);
    b.rope_inplace(&mut buf, 4, 4, 5, 10000.0);

    let result = b.to_f32(&buf);
    // First head untouched
    assert_close(&result[0..4], &input[0..4], 1e-6, "rope_offset_head0");
    // Second head modified
    assert_close(&result[4..8], &expected_tail, 1e-4, "rope_offset_head1");
}

#[test]
fn test_partial_rope_rotates_prefix_only() {
    let b = backend();
    let head_offset = 6;
    let head_dim = 6;
    let rotary_dim = 4;
    let position = 3;
    let freq_base = 10000.0;

    let input = vec![
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ];
    let mut expected = input.clone();
    cpu_partial_rope(
        &mut expected,
        head_offset,
        head_dim,
        rotary_dim,
        position,
        freq_base,
    );

    let mut buf = b.from_f32(&input);
    b.partial_rope_inplace(
        &mut buf,
        head_offset,
        head_dim,
        rotary_dim,
        position,
        freq_base,
    );

    let result = b.to_f32(&buf);
    assert_close(&result[0..6], &input[0..6], 1e-6, "partial_rope_head0");
    assert_close(&result[10..12], &input[10..12], 1e-6, "partial_rope_suffix");
    assert_close(
        &result[head_offset..head_offset + rotary_dim],
        &expected[head_offset..head_offset + rotary_dim],
        1e-4,
        "partial_rope_prefix",
    );
}

#[test]
fn test_rope_preserves_norm() {
    let b = backend();
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let orig_norm: f32 = input.iter().map(|v| v * v).sum::<f32>().sqrt();

    let mut buf = b.from_f32(&input);
    b.rope_inplace(&mut buf, 0, 8, 7, 10000.0);

    let result = b.to_f32(&buf);
    let new_norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();

    assert!(
        (orig_norm - new_norm).abs() < 1e-3,
        "RoPE should preserve norm: {orig_norm} vs {new_norm}"
    );
}

// -----------------------------------------------------------------------
// Cross-backend consistency (GPU vs CPU reference for a multi-step flow)
// -----------------------------------------------------------------------

#[test]
fn test_multi_step_flow() {
    // Simulate a mini transformer step: rms_norm -> matvec -> silu -> add
    let b = backend();

    let hidden = vec![1.0, 2.0, 3.0, 4.0];
    let norm_weight = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5f32;

    // CPU reference path
    let cpu_normed = cpu_rms_norm(&hidden, &norm_weight, eps);
    let matrix_data = vec![
        1.0, 0.0, 0.0, 0.0, // identity-ish rows
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let cpu_projected = cpu_matvec(&matrix_data, &cpu_normed, 4, 4);
    let cpu_activated = cpu_silu(&cpu_projected);
    let cpu_final: Vec<f32> = cpu_activated
        .iter()
        .zip(hidden.iter())
        .map(|(a, h)| a + h)
        .collect();

    // GPU path through the trait
    let x = b.from_f32(&hidden);
    let w = b.from_f32(&norm_weight);
    let mat = b.from_f32(&matrix_data);
    let residual = b.from_f32(&hidden);

    let mut normed = b.zeros(4);
    b.rms_norm(&x, &w, &mut normed, eps);

    let mut projected = b.zeros(4);
    b.matvec(&mat, &normed, &mut projected, 4, 4);

    b.silu_inplace(&mut projected);
    b.add_inplace(&mut projected, &residual);

    let result = b.to_f32(&projected);
    assert_close(&result, &cpu_final, 1e-4, "multi_step_flow");
}
