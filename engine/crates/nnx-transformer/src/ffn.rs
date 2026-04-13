//! Feed-Forward Networks: SwiGLU, GeGLU, and GELU variants.
//!
//! SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))  [Llama, Qwen]
//! GeGLU:  output = down_proj(gelu(gate_proj(x)) * up_proj(x))  [Gemma]
//! GELU:   output = fc2(gelu(fc1(x)))                           [GPT-2]

use crate::block::BlockWeights;
use crate::config::{FFNType, ModelConfig};
use rayon::prelude::*;

/// Architecture-aware FFN forward pass.
///
/// Dispatches to the correct FFN variant based on `config.ffn_type`.
pub fn ffn_forward(hidden: &[f32], weights: &BlockWeights, config: &ModelConfig) -> Vec<f32> {
    let hidden_dim = hidden.len();
    let intermediate_dim = config.intermediate_dim;

    match config.ffn_type {
        FFNType::SwiGLU => {
            swiglu_ffn(
                hidden,
                &weights.w_gate,
                &weights.w_up,
                &weights.w_down,
                hidden_dim,
                intermediate_dim,
            )
        }
        FFNType::GeGLU => {
            geglu_ffn(
                hidden,
                &weights.w_gate,
                &weights.w_up,
                &weights.w_down,
                hidden_dim,
                intermediate_dim,
            )
        }
        FFNType::GELU => {
            gelu_ffn(
                hidden,
                &weights.w_gate,
                &weights.w_down,
                hidden_dim,
                intermediate_dim,
            )
        }
    }
}

/// SwiGLU FFN: gate and up projections computed in parallel, then SiLU + element-wise mul + down.
pub fn swiglu_ffn(
    hidden: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
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

/// GeGLU FFN: like SwiGLU but with GELU activation instead of SiLU (used by Gemma).
fn geglu_ffn(
    hidden: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
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

    // gate = gelu(gate) * up
    nnx_kernels::activations::gelu_f32_inplace(&mut gate);
    nnx_kernels::activations::mul_f32_inplace(&mut gate, &up);

    // down projection
    let mut output = vec![0.0f32; hidden_dim];
    nnx_kernels::matmul::matvec_f32(w_down, &gate, &mut output, hidden_dim, intermediate_dim);

    output
}

/// GPT-2 style GELU FFN: fc1 -> GELU -> fc2 (only 2 weight matrices, no gating).
///
/// `w_fc1` corresponds to `w_gate` in BlockWeights, `w_fc2` corresponds to `w_down`.
/// `w_up` is unused for this variant.
fn gelu_ffn(
    hidden: &[f32],
    w_fc1: &[f32],
    w_fc2: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    let mut intermediate = vec![0.0f32; intermediate_dim];
    nnx_kernels::matmul::matvec_f32(w_fc1, hidden, &mut intermediate, intermediate_dim, hidden_dim);
    nnx_kernels::activations::gelu_f32_inplace(&mut intermediate);

    let mut output = vec![0.0f32; hidden_dim];
    nnx_kernels::matmul::matvec_f32(w_fc2, &intermediate, &mut output, hidden_dim, intermediate_dim);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_ffn_smoke() {
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
    fn test_swiglu_ffn_deterministic() {
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

    #[test]
    fn test_geglu_ffn() {
        let hidden_dim = 8;
        let intermediate_dim = 16;

        let hidden = vec![1.0f32; hidden_dim];
        let w_gate = vec![0.1f32; intermediate_dim * hidden_dim];
        let w_up = vec![0.1f32; intermediate_dim * hidden_dim];
        let w_down = vec![0.1f32; hidden_dim * intermediate_dim];

        let output = geglu_ffn(&hidden, &w_gate, &w_up, &w_down, hidden_dim, intermediate_dim);
        assert_eq!(output.len(), hidden_dim);
        assert!(output.iter().all(|v| v.is_finite()));

        // GeGLU should produce different results than SwiGLU due to different activation
        let swiglu_output = swiglu_ffn(&hidden, &w_gate, &w_up, &w_down, hidden_dim, intermediate_dim);
        let diff: f32 = output.iter().zip(swiglu_output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "GeGLU and SwiGLU should produce different outputs");
    }

    #[test]
    fn test_gelu_ffn() {
        let hidden_dim = 8;
        let intermediate_dim = 16;

        let hidden = vec![1.0f32; hidden_dim];
        let w_fc1 = vec![0.1f32; intermediate_dim * hidden_dim];
        let w_fc2 = vec![0.1f32; hidden_dim * intermediate_dim];

        let output = gelu_ffn(&hidden, &w_fc1, &w_fc2, hidden_dim, intermediate_dim);
        assert_eq!(output.len(), hidden_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ffn_forward_dispatch() {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let hidden = vec![1.0f32; hidden_dim];
        let weights = BlockWeights::test_no_bias(
            hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
        );

        // SwiGLU
        let config_swiglu = ModelConfig {
            architecture: "test".into(),
            arch: crate::config::Architecture::Llama,
            num_layers: 1,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: 32,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: crate::config::NormType::RMSNorm,
            ffn_type: FFNType::SwiGLU,
            pos_encoding: crate::config::PosEncoding::RoPE { freq_base: 10000.0 },
            block_style: crate::config::BlockStyle::Sequential,
            has_qkv_bias: false,
            has_output_bias: false,
            embedding_scale: None,
        };
        let out_swiglu = ffn_forward(&hidden, &weights, &config_swiglu);
        assert_eq!(out_swiglu.len(), hidden_dim);

        // GeGLU
        let config_geglu = ModelConfig {
            ffn_type: FFNType::GeGLU,
            arch: crate::config::Architecture::Gemma,
            ..config_swiglu.clone()
        };
        let out_geglu = ffn_forward(&hidden, &weights, &config_geglu);
        assert_eq!(out_geglu.len(), hidden_dim);

        // GELU
        let config_gelu = ModelConfig {
            ffn_type: FFNType::GELU,
            arch: crate::config::Architecture::GPT2,
            ..config_swiglu.clone()
        };
        let out_gelu = ffn_forward(&hidden, &weights, &config_gelu);
        assert_eq!(out_gelu.len(), hidden_dim);

        // All three should produce different outputs
        let diff_sg: f32 = out_swiglu.iter().zip(out_geglu.iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff_sg2: f32 = out_swiglu.iter().zip(out_gelu.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_sg > 1e-6, "SwiGLU and GeGLU should differ");
        assert!(diff_sg2 > 1e-6, "SwiGLU and GELU should differ");
    }
}
