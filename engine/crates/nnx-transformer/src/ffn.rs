//! SwiGLU Feed-Forward Network with parallel gate/up projections.
//!
//! SwiGLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
//!
//! gate_proj and up_proj are independent — compute them in parallel.

use rayon::prelude::*;

/// Compute SwiGLU FFN for one token.
///
/// gate_proj and up_proj run concurrently via rayon when intermediate_dim
/// is large enough to justify the threading overhead.
pub fn swiglu_ffn(
    hidden: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    // gate and up projections are independent — compute in parallel
    let (mut gate, up) = rayon::join(
        || {
            let mut g = vec![0.0f32; intermediate_dim];
            nnx_kernels::matmul::matvec_f32(w_gate, hidden, &mut g, intermediate_dim, hidden_dim);
            g
        },
        || {
            let mut u = vec![0.0f32; intermediate_dim];
            nnx_kernels::matmul::matvec_f32(w_up, hidden, &mut u, intermediate_dim, hidden_dim);
            u
        },
    );

    // gate = silu(gate) * up
    nnx_kernels::activations::silu_f32_inplace(&mut gate);
    nnx_kernels::activations::mul_f32_inplace(&mut gate, &up);

    // down projection
    let mut output = vec![0.0f32; hidden_dim];
    nnx_kernels::matmul::matvec_f32(w_down, &gate, &mut output, hidden_dim, intermediate_dim);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_smoke() {
        let hidden_dim = 8;
        let intermediate_dim = 16;

        let hidden = vec![1.0f32; hidden_dim];
        let w_gate = vec![0.1f32; intermediate_dim * hidden_dim];
        let w_up = vec![0.1f32; intermediate_dim * hidden_dim];
        let w_down = vec![0.1f32; hidden_dim * intermediate_dim];

        let output = swiglu_ffn(&hidden, &w_gate, &w_up, &w_down, hidden_dim, intermediate_dim);
        assert_eq!(output.len(), hidden_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ffn_deterministic() {
        let hidden_dim = 32;
        let intermediate_dim = 64;

        let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
        let w_gate: Vec<f32> = (0..intermediate_dim * hidden_dim).map(|i| ((i % 13) as f32) * 0.001).collect();
        let w_up: Vec<f32> = (0..intermediate_dim * hidden_dim).map(|i| ((i % 11) as f32) * 0.001).collect();
        let w_down: Vec<f32> = (0..hidden_dim * intermediate_dim).map(|i| ((i % 7) as f32) * 0.001).collect();

        let r1 = swiglu_ffn(&hidden, &w_gate, &w_up, &w_down, hidden_dim, intermediate_dim);
        let r2 = swiglu_ffn(&hidden, &w_gate, &w_up, &w_down, hidden_dim, intermediate_dim);

        for i in 0..hidden_dim {
            assert!((r1[i] - r2[i]).abs() < 1e-6, "non-deterministic at {}", i);
        }
    }
}
