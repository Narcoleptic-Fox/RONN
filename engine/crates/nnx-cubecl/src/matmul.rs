//! GPU matrix multiplication kernels.
//!
//! For large matmuls during prefill, use CubeCL's built-in optimized matmul
//! module (Tensor Core + TMA + warp specialization). This module provides
//! the decode-path workhorse: matrix-vector multiply.

use cubecl::prelude::*;

/// Matrix-vector multiply: `y = A @ x` where A is [m, k] and x is [k].
///
/// Each cube computes one output row. Launch with `CubeCount.x` = m.
/// Thread 0 in each cube does the full dot product (correct first,
/// parallel reduction optimization later).
#[cube(launch)]
pub fn matvec_kernel(
    matrix: &Array<f32>,
    vector: &Array<f32>,
    output: &mut Array<f32>,
    k: u32,
) {
    let row = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let mut sum = 0.0f32;
        for col in 0..k {
            sum += matrix[row * k + col] * vector[col];
        }
        output[row] = sum;
    }
}

/// Matrix-vector multiply with bias: `y = A @ x + bias`.
#[cube(launch)]
pub fn matvec_bias_kernel(
    matrix: &Array<f32>,
    vector: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    k: u32,
) {
    let row = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let mut sum = 0.0f32;
        for col in 0..k {
            sum += matrix[row * k + col] * vector[col];
        }
        output[row] = sum + bias[row];
    }
}

/// Dot product: `output[0] = sum(a[i] * b[i])`.
#[cube(launch)]
pub fn dot_kernel(
    a: &Array<f32>,
    b: &Array<f32>,
    output: &mut Array<f32>,
    len: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        output[0] = sum;
    }
}
