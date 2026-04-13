//! SwiGLU Feed-Forward Network — used by Llama-family models.
//!
//! SwiGLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
//! Three weight matrices: gate, up, down.

/// Compute SwiGLU FFN for one token.
///
/// # Arguments
/// - `hidden`: input [hidden_dim]
/// - `w_gate`: gate projection [intermediate_dim, hidden_dim]
/// - `w_up`: up projection [intermediate_dim, hidden_dim]
/// - `w_down`: down projection [hidden_dim, intermediate_dim]
///
/// # Returns
/// Output [hidden_dim]
pub fn swiglu_ffn(
    hidden: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    // gate = gate_proj(x) → [intermediate_dim]
    let mut gate = vec![0.0f32; intermediate_dim];
    nnx_kernels::matmul::matvec_f32(w_gate, hidden, &mut gate, intermediate_dim, hidden_dim);

    // up = up_proj(x) → [intermediate_dim]
    let mut up = vec![0.0f32; intermediate_dim];
    nnx_kernels::matmul::matvec_f32(w_up, hidden, &mut up, intermediate_dim, hidden_dim);

    // gate = silu(gate) * up
    nnx_kernels::activations::silu_f32_inplace(&mut gate);
    nnx_kernels::activations::mul_f32_inplace(&mut gate, &up);

    // output = down_proj(gate) → [hidden_dim]
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
}
