//! GPU activation and elementwise kernels.
//!
//! All elementwise kernels accept a `len` parameter for bounds checking,
//! allowing tiled launches where total threads may exceed the buffer size.

use cubecl::prelude::*;

/// Maximum threads per workgroup (safe across CUDA, Metal, WebGPU).
pub const BLOCK_SIZE: u32 = 256;

/// SiLU (Swish) activation: `x * sigmoid(x)`.
#[cube(launch)]
pub fn silu_kernel(input: &Array<f32>, output: &mut Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        let x = input[idx];
        output[idx] = x / (1.0f32 + f32::exp(-x));
    }
}

/// SiLU in-place.
#[cube(launch)]
pub fn silu_inplace_kernel(data: &mut Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        let x = data[idx];
        data[idx] = x / (1.0f32 + f32::exp(-x));
    }
}

/// GELU activation (approximate).
#[cube(launch)]
pub fn gelu_kernel(input: &Array<f32>, output: &mut Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        let x = input[idx];
        let cdf = 0.5f32 * (1.0f32 + f32::tanh(0.7978845608f32 * (x + 0.044715f32 * x * x * x)));
        output[idx] = x * cdf;
    }
}

/// GELU in-place.
#[cube(launch)]
pub fn gelu_inplace_kernel(data: &mut Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        let x = data[idx];
        let cdf = 0.5f32 * (1.0f32 + f32::tanh(0.7978845608f32 * (x + 0.044715f32 * x * x * x)));
        data[idx] = x * cdf;
    }
}

/// Elementwise multiply: `output[i] = a[i] * b[i]`.
#[cube(launch)]
pub fn mul_kernel(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        output[idx] = a[idx] * b[idx];
    }
}

/// Elementwise multiply in-place: `a[i] *= b[i]`.
#[cube(launch)]
pub fn mul_inplace_kernel(a: &mut Array<f32>, b: &Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        a[idx] = a[idx] * b[idx];
    }
}

/// Elementwise add in-place: `a[i] += b[i]`.
#[cube(launch)]
pub fn add_inplace_kernel(a: &mut Array<f32>, b: &Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        a[idx] = a[idx] + b[idx];
    }
}

/// Fused SwiGLU: `output = silu(gate) * up`.
#[cube(launch)]
pub fn fused_swiglu_kernel(gate: &Array<f32>, up: &Array<f32>, output: &mut Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        let g = gate[idx];
        let silu_g = g / (1.0f32 + f32::exp(-g));
        output[idx] = silu_g * up[idx];
    }
}

/// Fused GeGLU: `output = gelu(gate) * up`.
#[cube(launch)]
pub fn fused_geglu_kernel(gate: &Array<f32>, up: &Array<f32>, output: &mut Array<f32>, len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < len {
        let g = gate[idx];
        let cdf = 0.5f32 * (1.0f32 + f32::tanh(0.7978845608f32 * (g + 0.044715f32 * g * g * g)));
        output[idx] = (g * cdf) * up[idx];
    }
}
