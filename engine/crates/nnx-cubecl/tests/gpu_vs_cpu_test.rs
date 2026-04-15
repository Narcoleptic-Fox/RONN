//! Cross-validation: GPU forward pass must produce the same logits as
//! the CPU forward pass for a tiny synthetic model.
//!
//! Run with: `cargo test -p nnx-cubecl --features wgpu -- gpu_vs_cpu`

#![cfg(feature = "wgpu")]

use cubecl::wgpu::WgpuRuntime;
use nnx_cubecl::inference::GpuInference;
use nnx_transformer::block::BlockWeights;
use nnx_transformer::cache::KVCache;
use nnx_transformer::config::*;
use nnx_transformer::model::{Model, ModelWeights};
use nnx_transformer::weights::Matrix;

/// Build a tiny Llama-style model with deterministic, varied weights.
///
/// Parameters:
/// - hidden_dim=16, num_heads=4, num_kv_heads=2, head_dim=4
/// - intermediate_dim=32, vocab_size=64, num_layers=2
/// - RMSNorm, SwiGLU, RoPE, Sequential, no bias
///
/// Weight values use `(i % prime) * scale - offset` patterns so that
/// each matrix has non-trivial, non-uniform data. This is the same
/// approach used by `make_tiny_model()` in `nnx-serving/src/backend.rs`.
fn make_test_model() -> Model {
    let hidden_dim = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let intermediate_dim = 32;
    let vocab_size = 64;
    let num_layers = 2;

    let q_dim = num_heads * head_dim; // 16
    let kv_dim = num_kv_heads * head_dim; // 8

    let config = ModelConfig {
        architecture: "test".into(),
        arch: Architecture::Llama,
        num_layers,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_dim,
        vocab_size,
        max_context_length: 128,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        norm_type: NormType::RMSNorm,
        ffn_type: FFNType::SwiGLU,
        pos_encoding: PosEncoding::RoPE {
            freq_base: 10000.0,
        },
        block_style: BlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
        embedding_scale: None,
    };

    let make_layer = |seed: usize| BlockWeights {
        attn_norm: vec![1.0; hidden_dim],
        ffn_norm: vec![1.0; hidden_dim],
        wq: Matrix::dense(
            (0..q_dim * hidden_dim)
                .map(|i| ((i + seed) % 7) as f32 * 0.02 - 0.06)
                .collect(),
            q_dim,
            hidden_dim,
        ),
        wk: Matrix::dense(
            (0..kv_dim * hidden_dim)
                .map(|i| ((i + seed) % 11) as f32 * 0.02 - 0.1)
                .collect(),
            kv_dim,
            hidden_dim,
        ),
        wv: Matrix::dense(
            (0..kv_dim * hidden_dim)
                .map(|i| ((i + seed) % 13) as f32 * 0.02 - 0.12)
                .collect(),
            kv_dim,
            hidden_dim,
        ),
        wo: Matrix::dense(
            (0..hidden_dim * q_dim)
                .map(|i| ((i + seed) % 5) as f32 * 0.02 - 0.04)
                .collect(),
            hidden_dim,
            q_dim,
        ),
        w_gate: Matrix::dense(
            (0..intermediate_dim * hidden_dim)
                .map(|i| ((i + seed) % 17) as f32 * 0.01 - 0.08)
                .collect(),
            intermediate_dim,
            hidden_dim,
        ),
        w_up: Matrix::dense(
            (0..intermediate_dim * hidden_dim)
                .map(|i| ((i + seed) % 19) as f32 * 0.01 - 0.09)
                .collect(),
            intermediate_dim,
            hidden_dim,
        ),
        w_down: Matrix::dense(
            (0..hidden_dim * intermediate_dim)
                .map(|i| ((i + seed) % 23) as f32 * 0.01 - 0.11)
                .collect(),
            hidden_dim,
            intermediate_dim,
        ),
        bq: None,
        bk: None,
        bv: None,
        bo: None,
        attn_norm_bias: None,
        ffn_norm_bias: None,
    };

    let layers: Vec<_> = (0..num_layers).map(|i| make_layer(i * 37)).collect();

    let weights = ModelWeights {
        token_embedding: Matrix::dense(
            (0..vocab_size * hidden_dim)
                .map(|i| (i % 19) as f32 * 0.05 - 0.45)
                .collect(),
            vocab_size,
            hidden_dim,
        ),
        layers,
        final_norm: vec![1.0; hidden_dim],
        final_norm_bias: None,
        lm_head: Matrix::dense(
            (0..vocab_size * hidden_dim)
                .map(|i| (i % 23) as f32 * 0.03 - 0.33)
                .collect(),
            vocab_size,
            hidden_dim,
        ),
    };

    Model::new(config, weights)
}

#[test]
fn gpu_forward_matches_cpu_forward() {
    let model = make_test_model();
    let tokens: [u32; 5] = [1, 5, 10, 3, 7];

    // --- CPU path ---
    let mut cpu_cache = model.new_cache();
    let mut cpu_logits = Vec::new();
    for &token in &tokens {
        let logits = model
            .forward_token(&mut cpu_cache, token)
            .expect("CPU forward_token failed");
        cpu_logits.push(logits);
    }

    // --- GPU path ---
    let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
        .expect("GPU model upload failed");
    let mut gpu_cache = gpu.new_cache();
    let mut gpu_logits_all = Vec::new();
    for &token in &tokens {
        let logits = gpu.forward_token(&mut gpu_cache, token);
        gpu_logits_all.push(logits);
    }

    // --- Compare ---
    // GPU floating point order can differ from CPU, especially in reductions
    // (softmax, dot products, RMSNorm).  After 5 sequential tokens through
    // 2 layers, accumulated drift of ~1-2% is typical for f32 arithmetic.
    // We use 5e-2 to account for worst-case accumulation while still catching
    // any real logic bugs (which would produce differences > 1.0).
    let tolerance = 5e-2;
    for (step, (cpu, gpu_logits)) in cpu_logits.iter().zip(gpu_logits_all.iter()).enumerate() {
        assert_eq!(
            cpu.len(),
            gpu_logits.len(),
            "step {}: logit length mismatch (cpu={}, gpu={})",
            step,
            cpu.len(),
            gpu_logits.len(),
        );
        for (i, (cv, gv)) in cpu.iter().zip(gpu_logits.iter()).enumerate() {
            assert!(
                (cv - gv).abs() < tolerance,
                "step {}, logit {}: cpu={}, gpu={}, diff={}",
                step,
                i,
                cv,
                gv,
                (cv - gv).abs(),
            );
        }
    }
}

#[test]
fn gpu_logits_are_finite_and_varied() {
    let model = make_test_model();
    let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
        .expect("GPU model upload failed");
    let mut cache = gpu.new_cache();

    let logits = gpu.forward_token(&mut cache, 1);

    // All logits must be finite
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "GPU produced non-finite logits"
    );

    // Logits should not all be the same value (would indicate a bug)
    let first = logits[0];
    let all_same = logits.iter().all(|v| (v - first).abs() < 1e-10);
    assert!(
        !all_same,
        "GPU logits are all identical ({}) — likely a kernel bug",
        first
    );
}

#[test]
fn gpu_cache_advances_position() {
    let model = make_test_model();
    let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
        .expect("GPU model upload failed");
    let mut cache = gpu.new_cache();

    // Initially empty
    for layer_cache in &cache {
        assert_eq!(layer_cache.len, 0, "cache should start empty");
    }

    // Process 3 tokens
    for &token in &[1u32, 5, 10] {
        gpu.forward_token(&mut cache, token);
    }

    // Each layer cache should have 3 entries
    for (i, layer_cache) in cache.iter().enumerate() {
        assert_eq!(
            layer_cache.len, 3,
            "layer {} cache should have 3 entries after 3 tokens",
            i
        );
    }
}
