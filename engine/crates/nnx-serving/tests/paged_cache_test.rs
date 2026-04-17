//! Cross-validation tests: verify that paged KV cache produces identical
//! attention outputs compared to the contiguous LayerCache.
//!
//! These tests run the actual attention functions from nnx-transformer
//! with both cache backends and assert bit-for-bit identical results.

use nnx_core::engine::KVStore;
use nnx_serving::block_manager::BlockAllocator;
use nnx_serving::page::PageId;
use nnx_serving::paged_cache::PagedLayerView;
use nnx_transformer::cache::LayerCache;

/// Store the same sequence of key/value pairs into both a LayerCache and
/// a PagedLayerView, then compare every position and head for exact equality.
fn cross_validate_store_read(
    num_tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
) {
    let kv_dim = num_kv_heads * head_dim;

    // Generate deterministic test data.
    let keys: Vec<Vec<f32>> = (0..num_tokens)
        .map(|t| {
            (0..kv_dim)
                .map(|d| ((t * kv_dim + d) as f32) * 0.01 + 0.5)
                .collect()
        })
        .collect();
    let vals: Vec<Vec<f32>> = (0..num_tokens)
        .map(|t| {
            (0..kv_dim)
                .map(|d| ((t * kv_dim + d) as f32) * -0.01 + 0.3)
                .collect()
        })
        .collect();

    // --- Contiguous path ---
    let mut contiguous = LayerCache::new(num_tokens + 16, num_kv_heads, head_dim);
    for t in 0..num_tokens {
        contiguous.store(&keys[t], &vals[t]).unwrap();
    }

    // --- Paged path ---
    let max_pages = (num_tokens / page_size + 2) * 2; // plenty of headroom
    let mut allocator = BlockAllocator::new(max_pages, page_size, num_kv_heads, head_dim);
    let mut page_table: Vec<PageId> = Vec::new();
    {
        let mut paged = PagedLayerView::new(
            &mut allocator,
            &mut page_table,
            0, // initial_tokens
            0, // layer_idx
            page_size,
            num_kv_heads,
            head_dim,
        );

        for t in 0..num_tokens {
            paged.store(&keys[t], &vals[t]).unwrap();
        }

        assert_eq!(contiguous.len(), paged.len(), "token count mismatch");

        // Compare every position and head — must be bit-for-bit identical.
        for t in 0..num_tokens {
            for h in 0..num_kv_heads {
                assert_eq!(
                    contiguous.key_at(t, h),
                    paged.key_at(t, h),
                    "key mismatch at token={}, head={} (page_size={})",
                    t,
                    h,
                    page_size,
                );
                assert_eq!(
                    contiguous.value_at(t, h),
                    paged.value_at(t, h),
                    "value mismatch at token={}, head={} (page_size={})",
                    t,
                    h,
                    page_size,
                );
            }
        }
    }
}

// -------------------------------------------------------------------
// Store/read round-trip cross-validation across different geometries
// -------------------------------------------------------------------

#[test]
fn cross_validate_1head_4dim_page4() {
    cross_validate_store_read(10, 1, 4, 4);
}

#[test]
fn cross_validate_2heads_4dim_page4() {
    cross_validate_store_read(10, 2, 4, 4);
}

#[test]
fn cross_validate_8heads_128dim_page16() {
    // Llama-7B-class geometry.
    cross_validate_store_read(50, 8, 128, 16);
}

#[test]
fn cross_validate_2heads_64dim_page8() {
    // Smaller model geometry.
    cross_validate_store_read(20, 2, 64, 8);
}

#[test]
fn cross_validate_exact_page_boundary() {
    // Exactly fills pages (no partial last page).
    cross_validate_store_read(16, 4, 32, 4);
}

#[test]
fn cross_validate_single_token() {
    cross_validate_store_read(1, 2, 4, 16);
}

#[test]
fn cross_validate_page_size_1() {
    // Degenerate: every token gets its own page.
    cross_validate_store_read(8, 2, 4, 1);
}

#[test]
fn cross_validate_large_sequence() {
    // 512 tokens, crossing many page boundaries.
    cross_validate_store_read(512, 4, 32, 16);
}

// -------------------------------------------------------------------
// Attention-level cross-validation
// -------------------------------------------------------------------
// These tests run the full attention_decode_configurable through both
// cache backends and verify identical logits.

/// Run attention decode for `num_steps` tokens using the contiguous LayerCache,
/// return the output of the last step.
fn run_contiguous_decode(
    hidden_seq: &[Vec<f32>],
    weights: &nnx_transformer::block::BlockWeights,
    config: &nnx_transformer::config::ModelConfig,
) -> Vec<Vec<f32>> {
    let mut cache = LayerCache::new(hidden_seq.len() + 16, config.num_kv_heads, config.head_dim);
    let mut outputs = Vec::new();
    for (pos, hidden) in hidden_seq.iter().enumerate() {
        let out = nnx_transformer::attention::attention_decode_configurable(
            hidden, weights, &mut cache, pos, config,
        )
        .unwrap();
        outputs.push(out);
    }
    outputs
}

/// Run the same sequence through the paged path.
fn run_paged_decode(
    hidden_seq: &[Vec<f32>],
    weights: &nnx_transformer::block::BlockWeights,
    config: &nnx_transformer::config::ModelConfig,
    page_size: usize,
) -> Vec<Vec<f32>> {
    let max_pages = (hidden_seq.len() / page_size + 2) * 2;
    let mut allocator =
        BlockAllocator::new(max_pages, page_size, config.num_kv_heads, config.head_dim);
    let mut page_table: Vec<PageId> = Vec::new();
    let mut num_tokens: usize = 0;

    let mut outputs = Vec::new();
    for (pos, hidden) in hidden_seq.iter().enumerate() {
        let mut view = PagedLayerView::new(
            &mut allocator,
            &mut page_table,
            num_tokens, // owned copy of current count
            0,          // layer_idx
            page_size,
            config.num_kv_heads,
            config.head_dim,
        );
        let out = nnx_transformer::attention::attention_decode_configurable(
            hidden, weights, &mut view, pos, config,
        )
        .unwrap();
        num_tokens = view.token_count(); // read back updated count
        outputs.push(out);
    }
    outputs
}

fn make_test_config(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    pos_encoding: nnx_transformer::config::PosEncoding,
) -> nnx_transformer::config::ModelConfig {
    nnx_transformer::config::ModelConfig {
        architecture: "test".into(),
        arch: nnx_transformer::config::Architecture::Llama,
        num_layers: 1,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_dim: hidden_dim * 4,
        vocab_size: 32,
        max_context_length: 128,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        norm_type: nnx_transformer::config::NormType::RMSNorm,
        ffn_type: nnx_transformer::config::FFNType::SwiGLU,
        pos_encoding,
        block_style: nnx_transformer::config::BlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
        embedding_scale: None,
        activation_quantization: nnx_transformer::config::ActivationQuantization::None,
    }
}

fn make_test_weights(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> nnx_transformer::block::BlockWeights {
    use nnx_transformer::weights::Matrix;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let intermediate_dim = hidden_dim * 4;

    nnx_transformer::block::BlockWeights {
        attn_norm: vec![1.0; hidden_dim],
        ffn_norm: vec![1.0; hidden_dim],
        wq: Matrix::dense(
            (0..q_dim * hidden_dim)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
                .collect(),
            q_dim,
            hidden_dim,
        ),
        wk: Matrix::dense(
            (0..kv_dim * hidden_dim)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
                .collect(),
            kv_dim,
            hidden_dim,
        ),
        wv: Matrix::dense(
            (0..kv_dim * hidden_dim)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                .collect(),
            kv_dim,
            hidden_dim,
        ),
        wo: Matrix::dense(
            (0..hidden_dim * q_dim)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.02)
                .collect(),
            hidden_dim,
            q_dim,
        ),
        w_gate: Matrix::dense(
            vec![0.01; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        ),
        w_up: Matrix::dense(
            vec![0.01; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        ),
        w_down: Matrix::dense(
            vec![0.01; hidden_dim * intermediate_dim],
            hidden_dim,
            intermediate_dim,
        ),
        bq: None,
        bk: None,
        bv: None,
        bo: None,
        attn_norm_bias: None,
        ffn_norm_bias: None,
    }
}

fn make_hidden_sequence(num_tokens: usize, hidden_dim: usize) -> Vec<Vec<f32>> {
    (0..num_tokens)
        .map(|t| {
            (0..hidden_dim)
                .map(|d| ((t * hidden_dim + d) as f32 * 0.037 + 0.5).sin())
                .collect()
        })
        .collect()
}

#[test]
fn attention_decode_rope_paged_matches_contiguous() {
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim; // 32

    let config = make_test_config(
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        nnx_transformer::config::PosEncoding::RoPE { freq_base: 10000.0 },
    );
    let weights = make_test_weights(hidden_dim, num_heads, num_kv_heads, head_dim);
    let hidden_seq = make_hidden_sequence(8, hidden_dim);

    let contiguous_out = run_contiguous_decode(&hidden_seq, &weights, &config);
    let paged_out = run_paged_decode(&hidden_seq, &weights, &config, 4);

    for (step, (c, p)) in contiguous_out.iter().zip(paged_out.iter()).enumerate() {
        assert_eq!(c.len(), p.len(), "output length mismatch at step {}", step);
        for (i, (cv, pv)) in c.iter().zip(p.iter()).enumerate() {
            assert!(
                (cv - pv).abs() < 1e-6,
                "output mismatch at step={}, dim={}: contiguous={}, paged={}",
                step,
                i,
                cv,
                pv,
            );
        }
    }
}

#[test]
fn attention_decode_no_rope_paged_matches_contiguous() {
    let num_heads = 4;
    let num_kv_heads = 4; // MHA
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;

    let config = make_test_config(
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        nnx_transformer::config::PosEncoding::None,
    );
    let weights = make_test_weights(hidden_dim, num_heads, num_kv_heads, head_dim);
    let hidden_seq = make_hidden_sequence(6, hidden_dim);

    let contiguous_out = run_contiguous_decode(&hidden_seq, &weights, &config);
    let paged_out = run_paged_decode(&hidden_seq, &weights, &config, 2); // small pages

    for (step, (c, p)) in contiguous_out.iter().zip(paged_out.iter()).enumerate() {
        for (i, (cv, pv)) in c.iter().zip(p.iter()).enumerate() {
            assert!(
                (cv - pv).abs() < 1e-6,
                "no-rope mismatch at step={}, dim={}: contiguous={}, paged={}",
                step,
                i,
                cv,
                pv,
            );
        }
    }
}

#[test]
fn attention_decode_gqa_paged_matches_contiguous() {
    // Strong GQA: 8 Q heads, 2 KV heads (4:1 ratio).
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim; // 32

    let config = make_test_config(
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        nnx_transformer::config::PosEncoding::RoPE { freq_base: 10000.0 },
    );
    let weights = make_test_weights(hidden_dim, num_heads, num_kv_heads, head_dim);
    let hidden_seq = make_hidden_sequence(10, hidden_dim);

    let contiguous_out = run_contiguous_decode(&hidden_seq, &weights, &config);
    let paged_out = run_paged_decode(&hidden_seq, &weights, &config, 4);

    for (step, (c, p)) in contiguous_out.iter().zip(paged_out.iter()).enumerate() {
        for (i, (cv, pv)) in c.iter().zip(p.iter()).enumerate() {
            assert!(
                (cv - pv).abs() < 1e-6,
                "GQA mismatch at step={}, dim={}: contiguous={}, paged={}",
                step,
                i,
                cv,
                pv,
            );
        }
    }
}

#[test]
fn attention_decode_page_size_1_matches_contiguous() {
    // Degenerate page size: every token gets its own page.
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let hidden_dim = num_heads * head_dim;

    let config = make_test_config(
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        nnx_transformer::config::PosEncoding::RoPE { freq_base: 10000.0 },
    );
    let weights = make_test_weights(hidden_dim, num_heads, num_kv_heads, head_dim);
    let hidden_seq = make_hidden_sequence(5, hidden_dim);

    let contiguous_out = run_contiguous_decode(&hidden_seq, &weights, &config);
    let paged_out = run_paged_decode(&hidden_seq, &weights, &config, 1);

    for (step, (c, p)) in contiguous_out.iter().zip(paged_out.iter()).enumerate() {
        for (i, (cv, pv)) in c.iter().zip(p.iter()).enumerate() {
            assert!(
                (cv - pv).abs() < 1e-6,
                "page_size=1 mismatch at step={}, dim={}: contiguous={}, paged={}",
                step,
                i,
                cv,
                pv,
            );
        }
    }
}
