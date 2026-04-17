//! Feed-Forward Networks: SwiGLU, GeGLU, and GELU variants.
//!
//! SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))  [Llama, Qwen]
//! GeGLU:  output = down_proj(gelu(gate_proj(x)) * up_proj(x))  [Gemma]
//! GELU:   output = fc2(gelu(fc1(x)))                           [GPT-2]

use crate::block::BlockWeights;
use crate::config::{ActivationQuantization, FFNType, ModelConfig};
use crate::quant_utils::maybe_quantize_activations;
use crate::weights::Matrix;
use nnx_core::error::Result;
use rayon::prelude::*;

/// Architecture-aware FFN forward pass.
///
/// Dispatches to the correct FFN variant based on `config.ffn_type`.
pub fn ffn_forward(
    hidden: &[f32],
    weights: &BlockWeights,
    config: &ModelConfig,
) -> Result<Vec<f32>> {
    let hidden_dim = hidden.len();
    let intermediate_dim = config.intermediate_dim;

    match config.ffn_type {
        FFNType::SwiGLU => swiglu_ffn(
            hidden,
            &weights.w_gate,
            &weights.w_up,
            &weights.w_down,
            hidden_dim,
            intermediate_dim,
            config.activation_quantization,
        ),
        FFNType::GeGLU => geglu_ffn(
            hidden,
            &weights.w_gate,
            &weights.w_up,
            &weights.w_down,
            hidden_dim,
            intermediate_dim,
            config.activation_quantization,
        ),
        FFNType::GELU => gelu_ffn(
            hidden,
            &weights.w_gate,
            &weights.w_down,
            hidden_dim,
            intermediate_dim,
            config.activation_quantization,
        ),
    }
}

/// Batched FFN forward for prompt prefill.
pub fn ffn_forward_batch(
    hidden_batch: &[f32],
    batch_size: usize,
    weights: &BlockWeights,
    config: &ModelConfig,
) -> Result<Vec<f32>> {
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;

    match config.ffn_type {
        FFNType::SwiGLU => swiglu_ffn_batch(
            hidden_batch,
            batch_size,
            &weights.w_gate,
            &weights.w_up,
            &weights.w_down,
            hidden_dim,
            intermediate_dim,
            config.activation_quantization,
        ),
        FFNType::GeGLU => geglu_ffn_batch(
            hidden_batch,
            batch_size,
            &weights.w_gate,
            &weights.w_up,
            &weights.w_down,
            hidden_dim,
            intermediate_dim,
            config.activation_quantization,
        ),
        FFNType::GELU => gelu_ffn_batch(
            hidden_batch,
            batch_size,
            &weights.w_gate,
            &weights.w_down,
            hidden_dim,
            intermediate_dim,
            config.activation_quantization,
        ),
    }
}

/// SwiGLU FFN: gate and up projections computed in parallel, then SiLU + element-wise mul + down.
pub fn swiglu_ffn(
    hidden: &[f32],
    w_gate: &Matrix,
    w_up: &Matrix,
    w_down: &Matrix,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation_quantization: ActivationQuantization,
) -> Result<Vec<f32>> {
    let (mut gate, up) = rayon::join(
        || {
            let mut g = vec![0.0f32; intermediate_dim];
            w_gate.matvec(hidden, &mut g);
            g
        },
        || {
            let mut u = vec![0.0f32; intermediate_dim];
            w_up.matvec(hidden, &mut u);
            u
        },
    );

    // gate = silu(gate) * up
    nnx_kernels::activations::silu_f32_inplace(&mut gate);
    nnx_kernels::activations::mul_f32_inplace(&mut gate, &up);
    maybe_quantize_activations(&mut gate, 1, intermediate_dim, activation_quantization)?;

    // down projection
    let mut output = vec![0.0f32; hidden_dim];
    w_down.matvec(&gate, &mut output);

    Ok(output)
}

fn swiglu_ffn_batch(
    hidden_batch: &[f32],
    batch_size: usize,
    w_gate: &Matrix,
    w_up: &Matrix,
    w_down: &Matrix,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation_quantization: ActivationQuantization,
) -> Result<Vec<f32>> {
    let (mut gate, up) = rayon::join(
        || {
            let mut g = vec![0.0f32; batch_size * intermediate_dim];
            w_gate.matmul_input_rows(hidden_batch, batch_size, &mut g);
            g
        },
        || {
            let mut u = vec![0.0f32; batch_size * intermediate_dim];
            w_up.matmul_input_rows(hidden_batch, batch_size, &mut u);
            u
        },
    );

    nnx_kernels::activations::silu_f32_inplace(&mut gate);
    nnx_kernels::activations::mul_f32_inplace(&mut gate, &up);
    maybe_quantize_activations(
        &mut gate,
        batch_size,
        intermediate_dim,
        activation_quantization,
    )?;

    let mut output = vec![0.0f32; batch_size * hidden_dim];
    w_down.matmul_input_rows(&gate, batch_size, &mut output);
    Ok(output)
}

/// GeGLU FFN: like SwiGLU but with GELU activation instead of SiLU (used by Gemma).
fn geglu_ffn(
    hidden: &[f32],
    w_gate: &Matrix,
    w_up: &Matrix,
    w_down: &Matrix,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation_quantization: ActivationQuantization,
) -> Result<Vec<f32>> {
    let (mut gate, up) = rayon::join(
        || {
            let mut g = vec![0.0f32; intermediate_dim];
            w_gate.matvec(hidden, &mut g);
            g
        },
        || {
            let mut u = vec![0.0f32; intermediate_dim];
            w_up.matvec(hidden, &mut u);
            u
        },
    );

    // gate = gelu(gate) * up
    nnx_kernels::activations::gelu_f32_inplace(&mut gate);
    nnx_kernels::activations::mul_f32_inplace(&mut gate, &up);
    maybe_quantize_activations(&mut gate, 1, intermediate_dim, activation_quantization)?;

    // down projection
    let mut output = vec![0.0f32; hidden_dim];
    w_down.matvec(&gate, &mut output);

    Ok(output)
}

fn geglu_ffn_batch(
    hidden_batch: &[f32],
    batch_size: usize,
    w_gate: &Matrix,
    w_up: &Matrix,
    w_down: &Matrix,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation_quantization: ActivationQuantization,
) -> Result<Vec<f32>> {
    let (mut gate, up) = rayon::join(
        || {
            let mut g = vec![0.0f32; batch_size * intermediate_dim];
            w_gate.matmul_input_rows(hidden_batch, batch_size, &mut g);
            g
        },
        || {
            let mut u = vec![0.0f32; batch_size * intermediate_dim];
            w_up.matmul_input_rows(hidden_batch, batch_size, &mut u);
            u
        },
    );

    nnx_kernels::activations::gelu_f32_inplace(&mut gate);
    nnx_kernels::activations::mul_f32_inplace(&mut gate, &up);
    maybe_quantize_activations(
        &mut gate,
        batch_size,
        intermediate_dim,
        activation_quantization,
    )?;

    let mut output = vec![0.0f32; batch_size * hidden_dim];
    w_down.matmul_input_rows(&gate, batch_size, &mut output);
    Ok(output)
}

/// GPT-2 style GELU FFN: fc1 -> GELU -> fc2 (only 2 weight matrices, no gating).
///
/// `w_fc1` corresponds to `w_gate` in BlockWeights, `w_fc2` corresponds to `w_down`.
/// `w_up` is unused for this variant.
fn gelu_ffn(
    hidden: &[f32],
    w_fc1: &Matrix,
    w_fc2: &Matrix,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation_quantization: ActivationQuantization,
) -> Result<Vec<f32>> {
    let mut intermediate = vec![0.0f32; intermediate_dim];
    w_fc1.matvec(hidden, &mut intermediate);
    nnx_kernels::activations::gelu_f32_inplace(&mut intermediate);
    maybe_quantize_activations(
        &mut intermediate,
        1,
        intermediate_dim,
        activation_quantization,
    )?;

    let mut output = vec![0.0f32; hidden_dim];
    w_fc2.matvec(&intermediate, &mut output);

    Ok(output)
}

fn gelu_ffn_batch(
    hidden_batch: &[f32],
    batch_size: usize,
    w_fc1: &Matrix,
    w_fc2: &Matrix,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation_quantization: ActivationQuantization,
) -> Result<Vec<f32>> {
    let mut intermediate = vec![0.0f32; batch_size * intermediate_dim];
    w_fc1.matmul_input_rows(hidden_batch, batch_size, &mut intermediate);
    nnx_kernels::activations::gelu_f32_inplace(&mut intermediate);
    maybe_quantize_activations(
        &mut intermediate,
        batch_size,
        intermediate_dim,
        activation_quantization,
    )?;

    let mut output = vec![0.0f32; batch_size * hidden_dim];
    w_fc2.matmul_input_rows(&intermediate, batch_size, &mut output);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_ffn_smoke() {
        let hidden_dim = 8;
        let intermediate_dim = 16;

        let hidden = vec![1.0f32; hidden_dim];
        let w_gate = Matrix::dense(
            vec![0.1f32; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        );
        let w_up = Matrix::dense(
            vec![0.1f32; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        );
        let w_down = Matrix::dense(
            vec![0.1f32; hidden_dim * intermediate_dim],
            hidden_dim,
            intermediate_dim,
        );

        let output = swiglu_ffn(
            &hidden,
            &w_gate,
            &w_up,
            &w_down,
            hidden_dim,
            intermediate_dim,
            ActivationQuantization::None,
        )
        .unwrap();
        assert_eq!(output.len(), hidden_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_ffn_deterministic() {
        let hidden_dim = 32;
        let intermediate_dim = 64;

        let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
        let w_gate = Matrix::dense(
            (0..intermediate_dim * hidden_dim)
                .map(|i| ((i % 13) as f32) * 0.001)
                .collect(),
            intermediate_dim,
            hidden_dim,
        );
        let w_up = Matrix::dense(
            (0..intermediate_dim * hidden_dim)
                .map(|i| ((i % 11) as f32) * 0.001)
                .collect(),
            intermediate_dim,
            hidden_dim,
        );
        let w_down = Matrix::dense(
            (0..hidden_dim * intermediate_dim)
                .map(|i| ((i % 7) as f32) * 0.001)
                .collect(),
            hidden_dim,
            intermediate_dim,
        );

        let r1 = swiglu_ffn(
            &hidden,
            &w_gate,
            &w_up,
            &w_down,
            hidden_dim,
            intermediate_dim,
            ActivationQuantization::None,
        )
        .unwrap();
        let r2 = swiglu_ffn(
            &hidden,
            &w_gate,
            &w_up,
            &w_down,
            hidden_dim,
            intermediate_dim,
            ActivationQuantization::None,
        )
        .unwrap();

        for i in 0..hidden_dim {
            assert!((r1[i] - r2[i]).abs() < 1e-6, "non-deterministic at {}", i);
        }
    }

    #[test]
    fn test_geglu_ffn() {
        let hidden_dim = 8;
        let intermediate_dim = 16;

        let hidden = vec![1.0f32; hidden_dim];
        let w_gate = Matrix::dense(
            vec![0.1f32; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        );
        let w_up = Matrix::dense(
            vec![0.1f32; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        );
        let w_down = Matrix::dense(
            vec![0.1f32; hidden_dim * intermediate_dim],
            hidden_dim,
            intermediate_dim,
        );

        let output = geglu_ffn(
            &hidden,
            &w_gate,
            &w_up,
            &w_down,
            hidden_dim,
            intermediate_dim,
            ActivationQuantization::None,
        )
        .unwrap();
        assert_eq!(output.len(), hidden_dim);
        assert!(output.iter().all(|v| v.is_finite()));

        // GeGLU should produce different results than SwiGLU due to different activation
        let swiglu_output = swiglu_ffn(
            &hidden,
            &w_gate,
            &w_up,
            &w_down,
            hidden_dim,
            intermediate_dim,
            ActivationQuantization::None,
        )
        .unwrap();
        let diff: f32 = output
            .iter()
            .zip(swiglu_output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "GeGLU and SwiGLU should produce different outputs"
        );
    }

    #[test]
    fn test_gelu_ffn() {
        let hidden_dim = 8;
        let intermediate_dim = 16;

        let hidden = vec![1.0f32; hidden_dim];
        let w_fc1 = Matrix::dense(
            vec![0.1f32; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        );
        let w_fc2 = Matrix::dense(
            vec![0.1f32; hidden_dim * intermediate_dim],
            hidden_dim,
            intermediate_dim,
        );

        let output = gelu_ffn(
            &hidden,
            &w_fc1,
            &w_fc2,
            hidden_dim,
            intermediate_dim,
            ActivationQuantization::None,
        )
        .unwrap();
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
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
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
            activation_quantization: ActivationQuantization::None,
        };
        let out_swiglu = ffn_forward(&hidden, &weights, &config_swiglu).unwrap();
        assert_eq!(out_swiglu.len(), hidden_dim);

        // GeGLU
        let config_geglu = ModelConfig {
            ffn_type: FFNType::GeGLU,
            arch: crate::config::Architecture::Gemma,
            ..config_swiglu.clone()
        };
        let out_geglu = ffn_forward(&hidden, &weights, &config_geglu).unwrap();
        assert_eq!(out_geglu.len(), hidden_dim);

        // GELU
        let config_gelu = ModelConfig {
            ffn_type: FFNType::GELU,
            arch: crate::config::Architecture::GPT2,
            ..config_swiglu.clone()
        };
        let out_gelu = ffn_forward(&hidden, &weights, &config_gelu).unwrap();
        assert_eq!(out_gelu.len(), hidden_dim);

        // All three should produce different outputs
        let diff_sg: f32 = out_swiglu
            .iter()
            .zip(out_geglu.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let diff_sg2: f32 = out_swiglu
            .iter()
            .zip(out_gelu.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_sg > 1e-6, "SwiGLU and GeGLU should differ");
        assert!(diff_sg2 > 1e-6, "SwiGLU and GELU should differ");
    }

    #[test]
    fn test_activation_quantized_ffn_stays_close_across_variants() {
        let hidden_dim = 32;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 8;
        let intermediate_dim = 64;

        let hidden: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.17)
            .collect();
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );

        for (arch, ffn_type) in [
            (crate::config::Architecture::Llama, FFNType::SwiGLU),
            (crate::config::Architecture::Gemma, FFNType::GeGLU),
            (crate::config::Architecture::GPT2, FFNType::GELU),
        ] {
            let dense_config = ModelConfig {
                architecture: "test".into(),
                arch: arch.clone(),
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
                ffn_type,
                pos_encoding: crate::config::PosEncoding::RoPE { freq_base: 10000.0 },
                block_style: crate::config::BlockStyle::Sequential,
                has_qkv_bias: false,
                has_output_bias: false,
                embedding_scale: None,
                activation_quantization: ActivationQuantization::None,
            };
            let quantized_config = ModelConfig {
                activation_quantization: ActivationQuantization::Q8_0,
                ..dense_config.clone()
            };

            let dense = ffn_forward(&hidden, &weights, &dense_config).unwrap();
            let quantized = ffn_forward(&hidden, &weights, &quantized_config).unwrap();
            let max_abs = dense
                .iter()
                .zip(quantized.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 0.05, "{ffn_type:?} drift too large: {max_abs}");
        }
    }

    #[test]
    fn test_activation_quantized_ffn_batch_stays_close() {
        let hidden_dim = 32;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 8;
        let intermediate_dim = 64;
        let batch_size = 3;
        let hidden_batch: Vec<f32> = (0..batch_size * hidden_dim)
            .map(|i| (((i * 7 + 3) % 23) as f32 - 11.0) * 0.09)
            .collect();
        let weights = BlockWeights::test_no_bias(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        );
        let mut config = ModelConfig::test_llama(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            32,
        );

        let dense = ffn_forward_batch(&hidden_batch, batch_size, &weights, &config).unwrap();
        config.activation_quantization = ActivationQuantization::Q8_0;
        let quantized = ffn_forward_batch(&hidden_batch, batch_size, &weights, &config).unwrap();

        let max_abs = dense
            .iter()
            .zip(quantized.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 0.05, "batched drift too large: {max_abs}");
    }
}
