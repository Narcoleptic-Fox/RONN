//! GPU kernel edge case tests.
//!
//! Targets boundary conditions: buffer sizes at BLOCK_SIZE boundaries,
//! single-element buffers, large buffers, and numerical stability.
//!
//! Run with: `cargo test -p nnx-cubecl --features wgpu --test gpu_edge_cases`

#![cfg(feature = "wgpu")]

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use nnx_core::backend::KernelBackend;
use nnx_cubecl::backend::CubeclBackend;

type R = WgpuRuntime;

fn gpu() -> CubeclBackend<R> {
    CubeclBackend::<R>::new()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length mismatch {} vs {}",
        msg,
        a.len(),
        b.len()
    );
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "{}: mismatch at [{}]: got={}, expected={}, diff={}",
            msg,
            i,
            av,
            bv,
            (av - bv).abs(),
        );
    }
}

// ===================================================================
// Buffer size boundary conditions
// ===================================================================

#[test]
fn silu_single_element() {
    let b = gpu();
    let mut x = b.from_f32(&[1.0]);
    b.silu_inplace(&mut x);
    let result = b.to_f32(&x);
    let expected = 1.0 / (1.0 + (-1.0f32).exp());
    assert!((result[0] - expected).abs() < 1e-5);
}

#[test]
fn silu_at_block_boundary() {
    // Exactly BLOCK_SIZE elements (256).
    let b = gpu();
    let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let expected: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let mut buf = b.from_f32(&data);
    b.silu_inplace(&mut buf);
    assert_close(&b.to_f32(&buf), &expected, 1e-5, "silu@256");
}

#[test]
fn silu_above_block_boundary() {
    // BLOCK_SIZE + 1 = 257 elements — tests tiled launch with partial last cube.
    let b = gpu();
    let data: Vec<f32> = (0..257).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let expected: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let mut buf = b.from_f32(&data);
    b.silu_inplace(&mut buf);
    assert_close(&b.to_f32(&buf), &expected, 1e-5, "silu@257");
}

#[test]
fn silu_below_block_boundary() {
    // BLOCK_SIZE - 1 = 255 elements.
    let b = gpu();
    let data: Vec<f32> = (0..255).map(|i| (i as f32 - 127.0) * 0.01).collect();
    let expected: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let mut buf = b.from_f32(&data);
    b.silu_inplace(&mut buf);
    assert_close(&b.to_f32(&buf), &expected, 1e-5, "silu@255");
}

#[test]
fn gelu_large_buffer() {
    // 4096 elements — typical hidden_dim for a 7B model.
    let b = gpu();
    let data: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.001).collect();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| {
            let cdf = 0.5 * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh());
            x * cdf
        })
        .collect();

    let mut buf = b.from_f32(&data);
    b.gelu_inplace(&mut buf);
    assert_close(&b.to_f32(&buf), &expected, 1e-4, "gelu@4096");
}

#[test]
fn add_inplace_large() {
    // 11008 elements — typical intermediate_dim for Llama-7B.
    let b = gpu();
    let a_data: Vec<f32> = (0..11008).map(|i| i as f32 * 0.001).collect();
    let b_data: Vec<f32> = (0..11008).map(|i| -(i as f32) * 0.001).collect();
    let expected: Vec<f32> = vec![0.0; 11008]; // a + (-a) = 0

    let mut a = b.from_f32(&a_data);
    let bb = b.from_f32(&b_data);
    b.add_inplace(&mut a, &bb);
    assert_close(&b.to_f32(&a), &expected, 1e-3, "add@11008");
}

#[test]
fn mul_inplace_large() {
    // 4096 elements.
    let b = gpu();
    let a_data: Vec<f32> = (0..4096).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let b_data: Vec<f32> = vec![2.0; 4096];
    let expected: Vec<f32> = a_data.iter().map(|x| x * 2.0).collect();

    let mut a = b.from_f32(&a_data);
    let bb = b.from_f32(&b_data);
    b.mul_inplace(&mut a, &bb);
    assert_close(&b.to_f32(&a), &expected, 1e-4, "mul@4096");
}

// ===================================================================
// Numerical stability
// ===================================================================

#[test]
fn softmax_all_zeros() {
    let b = gpu();
    let mut data = b.from_f32(&[0.0, 0.0, 0.0, 0.0]);
    b.softmax_inplace(&mut data, 0, 4);
    let result = b.to_f32(&data);
    // Uniform distribution.
    for &v in &result {
        assert!(
            (v - 0.25).abs() < 1e-5,
            "softmax of zeros should be uniform, got {}",
            v
        );
    }
}

#[test]
fn softmax_large_values() {
    // Large values that would overflow naive exp() — stable softmax subtracts max first.
    let b = gpu();
    let mut data = b.from_f32(&[1000.0, 1001.0, 1002.0]);
    b.softmax_inplace(&mut data, 0, 3);
    let result = b.to_f32(&data);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum={}", sum);
    assert!(
        result[2] > result[1] && result[1] > result[0],
        "monotonicity"
    );
}

#[test]
fn softmax_with_neg_inf() {
    // -inf entries should become 0 after softmax (attention masking pattern).
    let b = gpu();
    let mut data = b.from_f32(&[1.0, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY]);
    b.softmax_inplace(&mut data, 0, 4);
    let result = b.to_f32(&data);
    assert!(
        result[1].abs() < 1e-6,
        "masked position should be ~0, got {}",
        result[1]
    );
    assert!(
        result[3].abs() < 1e-6,
        "masked position should be ~0, got {}",
        result[3]
    );
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
}

#[test]
fn rms_norm_all_ones() {
    let b = gpu();
    let x = b.from_f32(&[1.0, 1.0, 1.0, 1.0]);
    let w = b.from_f32(&[1.0, 1.0, 1.0, 1.0]);
    let mut out = b.zeros(4);
    b.rms_norm(&x, &w, &mut out, 1e-5);
    let result = b.to_f32(&out);
    // rms([1,1,1,1]) = 1, so output = [1,1,1,1]
    for &v in &result {
        assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {}", v);
    }
}

#[test]
fn rms_norm_near_zero() {
    // Near-zero input — eps prevents division by zero.
    let b = gpu();
    let x = b.from_f32(&[1e-10, 1e-10, 1e-10, 1e-10]);
    let w = b.from_f32(&[1.0, 1.0, 1.0, 1.0]);
    let mut out = b.zeros(4);
    b.rms_norm(&x, &w, &mut out, 1e-5);
    let result = b.to_f32(&out);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "near-zero should not produce NaN/Inf"
    );
}

#[test]
fn rope_position_zero_is_identity() {
    // At position 0, cos(0)=1, sin(0)=0, so RoPE is identity.
    let b = gpu();
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut data = b.from_f32(&original);
    b.rope_inplace(&mut data, 0, 8, 0, 10000.0);
    let result = b.to_f32(&data);
    assert_close(&result, &original, 1e-5, "rope@pos0");
}

#[test]
fn rope_preserves_norm() {
    // RoPE is a rotation — it should preserve the L2 norm.
    let b = gpu();
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let orig_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut data = b.from_f32(&input);
    b.rope_inplace(&mut data, 0, 8, 42, 10000.0);
    let result = b.to_f32(&data);
    let new_norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        (orig_norm - new_norm).abs() < 1e-3,
        "RoPE should preserve norm: {} vs {}",
        orig_norm,
        new_norm,
    );
}

// ===================================================================
// Matvec edge cases
// ===================================================================

#[test]
fn matvec_1x1() {
    let b = gpu();
    let mat = b.from_f32(&[3.0]);
    let x = b.from_f32(&[4.0]);
    let mut y = b.zeros(1);
    b.matvec(&mat, &x, &mut y, 1, 1);
    let result = b.to_f32(&y);
    assert!((result[0] - 12.0).abs() < 1e-5);
}

#[test]
fn matvec_identity() {
    let b = gpu();
    // 4x4 identity matrix
    let mat = b.from_f32(&[
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let x = b.from_f32(&[1.0, 2.0, 3.0, 4.0]);
    let mut y = b.zeros(4);
    b.matvec(&mat, &x, &mut y, 4, 4);
    assert_close(
        &b.to_f32(&y),
        &[1.0, 2.0, 3.0, 4.0],
        1e-5,
        "identity matvec",
    );
}

#[test]
fn dot_orthogonal_vectors() {
    let b = gpu();
    let a = b.from_f32(&[1.0, 0.0, 0.0]);
    let bb_buf = b.from_f32(&[0.0, 1.0, 0.0]);
    let result = b.dot(&a, &bb_buf);
    assert!(
        result.abs() < 1e-6,
        "orthogonal dot should be 0, got {}",
        result
    );
}

// ===================================================================
// Round-trip correctness
// ===================================================================

#[test]
fn from_to_f32_preserves_special_values() {
    let b = gpu();
    let data = vec![0.0, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::MAX, f32::MIN];
    let buf = b.from_f32(&data);
    let result = b.to_f32(&buf);
    for (i, (&orig, &got)) in data.iter().zip(result.iter()).enumerate() {
        assert!(
            orig == got || (orig.is_nan() && got.is_nan()),
            "round-trip failed at [{}]: {} vs {}",
            i,
            orig,
            got,
        );
    }
}

#[test]
fn zeros_is_actually_zero() {
    let b = gpu();
    let buf = b.zeros(1024);
    let result = b.to_f32(&buf);
    assert_eq!(result.len(), 1024);
    for (i, &v) in result.iter().enumerate() {
        assert!(v == 0.0, "zeros[{}] = {} (expected 0.0)", i, v);
    }
}
