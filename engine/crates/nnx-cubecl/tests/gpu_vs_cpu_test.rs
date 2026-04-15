//! Integration tests for `GpuInference` using the `wgpu` backend.
//!
//! These tests exercise the full `from_raw_weights` → `forward_token` path
//! end-to-end against the WebGPU runtime.  They do **not** import
//! `nnx-transformer` types (that would create a circular dependency); instead
//! they construct all weights inline.
//!
//! GPU vs CPU numerical parity tests (same model, compare logits within
//! tolerance) live in `nnx-transformer/src/backend.rs` under the `gpu` feature
//! flag, where both the CPU model and `GpuModel` are available together.
//!
//! Run with: `cargo test -p nnx-cubecl --features wgpu`

#![cfg(feature = "wgpu")]

use cubecl::wgpu::WgpuRuntime;
use nnx_core::gpu_config::{GpuConfig, GpuPosEncoding};
use nnx_cubecl::inference::{GpuInference, RawLayerWeights};

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

/// Build a small but non-trivial model config for integration tests.
///
/// Parameters match the internal tests in `inference.rs` so that failures
/// can be compared easily:
/// - hidden_dim=16, num_heads=4, num_kv_heads=2, head_dim=4
/// - intermediate_dim=32, vocab_size=64, num_layers=2
fn test_config() -> GpuConfig {
    GpuConfig {
        num_layers: 2,
        hidden_dim: 16,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_dim: 32,
        vocab_size: 64,
        max_context_length: 128,
        pos_encoding: GpuPosEncoding::RoPE { freq_base: 10000.0 },
        rms_norm_eps: 1e-5,
        embedding_scale: None,
    }
}

/// Build varied (non-trivial, non-uniform) weights for a single layer.
///
/// Uses modular arithmetic with different primes per matrix so each weight
/// is distinct.  This matches the approach used by `make_tiny_model()` in
/// `nnx-serving` and the old `gpu_vs_cpu_test.rs`.
fn test_layer(seed: usize) -> RawLayerWeights {
    let hd = 16usize;
    let q_dim = 4 * 4;  // num_heads * head_dim
    let kv_dim = 2 * 4; // num_kv_heads * head_dim
    let ffn = 32usize;

    RawLayerWeights {
        attn_norm: vec![1.0f32; hd],
        ffn_norm: vec![1.0f32; hd],
        wq: (0..q_dim * hd)
            .map(|i| ((i + seed) % 7) as f32 * 0.02 - 0.06)
            .collect(),
        wk: (0..kv_dim * hd)
            .map(|i| ((i + seed) % 11) as f32 * 0.02 - 0.10)
            .collect(),
        wv: (0..kv_dim * hd)
            .map(|i| ((i + seed) % 13) as f32 * 0.02 - 0.12)
            .collect(),
        wo: (0..hd * q_dim)
            .map(|i| ((i + seed) % 5) as f32 * 0.02 - 0.04)
            .collect(),
        w_gate: (0..ffn * hd)
            .map(|i| ((i + seed) % 17) as f32 * 0.01 - 0.08)
            .collect(),
        w_up: (0..ffn * hd)
            .map(|i| ((i + seed) % 19) as f32 * 0.01 - 0.09)
            .collect(),
        w_down: (0..hd * ffn)
            .map(|i| ((i + seed) % 23) as f32 * 0.01 - 0.11)
            .collect(),
        bq: None,
        bk: None,
        bv: None,
        bo: None,
        attn_norm_bias: None,
        ffn_norm_bias: None,
    }
}

/// Upload test model weights and return a ready `GpuInference`.
fn make_test_engine() -> GpuInference<WgpuRuntime> {
    let config = test_config();
    let hd = config.hidden_dim;
    let vs = config.vocab_size;
    let num_layers = config.num_layers;

    let token_embedding: Vec<f32> = (0..vs * hd)
        .map(|i| (i % 19) as f32 * 0.05 - 0.45)
        .collect();
    let lm_head: Vec<f32> = (0..vs * hd)
        .map(|i| (i % 23) as f32 * 0.03 - 0.33)
        .collect();
    let final_norm = vec![1.0f32; hd];

    let layers: Vec<RawLayerWeights> = (0..num_layers).map(|i| test_layer(i * 37)).collect();

    GpuInference::<WgpuRuntime>::from_raw_weights(
        config,
        &token_embedding,
        &lm_head,
        &final_norm,
        None,
        layers,
    )
    .expect("from_raw_weights should succeed for valid test model")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn gpu_logits_are_finite_and_varied() {
    let gpu = make_test_engine();
    let mut cache = gpu.new_cache();

    let logits = gpu.forward_token(&mut cache, 1);

    assert_eq!(
        logits.len(),
        test_config().vocab_size,
        "logit length must equal vocab_size"
    );

    assert!(
        logits.iter().all(|v| v.is_finite()),
        "GPU produced non-finite logits: {:?}",
        &logits[..logits.len().min(8)]
    );

    // Logits should not all be identical — that would indicate a kernel bug
    // (e.g. all-zero attention output, broken matvec).
    let first = logits[0];
    let all_same = logits.iter().all(|v| (v - first).abs() < 1e-10);
    assert!(
        !all_same,
        "GPU logits are all identical ({first}) — likely a kernel bug"
    );
}

#[test]
fn gpu_cache_advances_position() {
    let gpu = make_test_engine();
    let mut cache = gpu.new_cache();

    // Initially every layer cache must be empty.
    for layer_cache in &cache {
        assert_eq!(layer_cache.len, 0, "cache must start empty");
    }

    // Process 3 tokens.
    for &token in &[1u32, 5, 10] {
        gpu.forward_token(&mut cache, token);
    }

    // Each layer cache must have advanced to position 3.
    for (i, layer_cache) in cache.iter().enumerate() {
        assert_eq!(
            layer_cache.len, 3,
            "layer {i} cache should have 3 entries after 3 tokens"
        );
    }
}

#[test]
fn gpu_sequential_logits_differ() {
    // Verify that the KV cache is actually being used: successive tokens at
    // different positions should produce different logits.
    let gpu = make_test_engine();
    let mut cache = gpu.new_cache();

    let logits_t0 = gpu.forward_token(&mut cache, 1u32);
    let logits_t1 = gpu.forward_token(&mut cache, 1u32);

    // Same token at different positions must produce different logits because
    // the KV cache context changes.
    let max_diff = logits_t0
        .iter()
        .zip(logits_t1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-6,
        "logits for token at position 0 and position 1 should differ (max_diff={max_diff})"
    );
}

#[test]
fn gpu_different_tokens_produce_different_logits() {
    // Logits should differ for different input tokens.
    let gpu = make_test_engine();

    let logits_a = {
        let mut cache = gpu.new_cache();
        gpu.forward_token(&mut cache, 3u32)
    };
    let logits_b = {
        let mut cache = gpu.new_cache();
        gpu.forward_token(&mut cache, 7u32)
    };

    let max_diff = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-6,
        "different input tokens should produce different logits (max_diff={max_diff})"
    );
}

#[test]
fn gpu_multi_layer_forward_is_consistent() {
    // Running the same 5-token sequence twice on fresh caches should produce
    // bit-identical logits (determinism guarantee from the wgpu backend).
    let gpu = make_test_engine();
    let tokens: [u32; 5] = [1, 5, 10, 3, 7];

    let run = || {
        let mut cache = gpu.new_cache();
        tokens.iter().map(|&t| gpu.forward_token(&mut cache, t)).collect::<Vec<_>>()
    };

    let run_a = run();
    let run_b = run();

    for (step, (a, b)) in run_a.iter().zip(run_b.iter()).enumerate() {
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                va, vb,
                "determinism check: step {step}, logit {i}: first run={va}, second run={vb}"
            );
        }
    }
}

#[test]
fn gpu_from_raw_weights_rejects_wrong_layer_count() {
    let config = test_config(); // num_layers = 2
    let hd = config.hidden_dim;
    let vs = config.vocab_size;

    // Pass 1 layer instead of the required 2.
    let result = GpuInference::<WgpuRuntime>::from_raw_weights(
        config,
        &vec![0.0f32; vs * hd],
        &vec![0.0f32; vs * hd],
        &vec![1.0f32; hd],
        None,
        vec![test_layer(0)], // only 1 layer
    );

    assert!(
        result.is_err(),
        "should reject mismatched layer count"
    );
}
