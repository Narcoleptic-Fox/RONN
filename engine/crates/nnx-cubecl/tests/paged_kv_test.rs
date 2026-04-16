//! Integration tests for GPU paged KV storage and paged attention kernels.
//!
//! Run with: `cargo test -p nnx-cubecl --features wgpu --test paged_kv_test`

#![cfg(feature = "wgpu")]

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use nnx_core::backend::KernelBackend;
use nnx_core::gpu_config::{GpuBlockStyle, GpuConfig, GpuFFNType, GpuNormType, GpuPosEncoding};
use nnx_core::PageId;
use nnx_cubecl::attention;
use nnx_cubecl::backend::CubeclBackend;
use nnx_cubecl::paged_kv::{
    paged_attention_contract_kernel, paged_attention_scores_kernel, paged_cache_append_kernel,
    GpuPagePool,
};
use nnx_cubecl::{GpuInference, GpuPagedKvQuantConfig, RawLayerWeights};

type R = WgpuRuntime;

fn backend() -> CubeclBackend<R> {
    CubeclBackend::<R>::new()
}

fn tiny_gpu_config(num_layers: usize) -> GpuConfig {
    GpuConfig {
        num_layers,
        hidden_dim: 8,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_dim: 16,
        vocab_size: 32,
        max_context_length: 64,
        pos_encoding: GpuPosEncoding::RoPE { freq_base: 10000.0 },
        rms_norm_eps: 1e-5,
        embedding_scale: None,
        norm_type: GpuNormType::RMSNorm,
        ffn_type: GpuFFNType::SwiGLU,
        block_style: GpuBlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
    }
}

fn tiny_raw_layer(seed: usize) -> RawLayerWeights {
    let hidden_dim = 8usize;
    let q_dim = 2 * 4;
    let kv_dim = 2 * 4;
    let intermediate_dim = 16usize;

    RawLayerWeights {
        attn_norm: vec![1.0f32 + seed as f32 * 0.001; hidden_dim],
        ffn_norm: vec![1.0f32 + seed as f32 * 0.0015; hidden_dim],
        wq: (0..q_dim * hidden_dim)
            .map(|idx| ((idx + seed) % 7) as f32 * 0.01 - 0.03)
            .collect(),
        wk: (0..kv_dim * hidden_dim)
            .map(|idx| ((idx + seed) % 11) as f32 * 0.01 - 0.05)
            .collect(),
        wv: (0..kv_dim * hidden_dim)
            .map(|idx| ((idx + seed) % 13) as f32 * 0.01 - 0.06)
            .collect(),
        wo: (0..hidden_dim * q_dim)
            .map(|idx| ((idx + seed) % 5) as f32 * 0.01 - 0.02)
            .collect(),
        w_gate: (0..intermediate_dim * hidden_dim)
            .map(|idx| ((idx + seed) % 9) as f32 * 0.005 - 0.02)
            .collect(),
        w_up: (0..intermediate_dim * hidden_dim)
            .map(|idx| ((idx + seed) % 10) as f32 * 0.005 - 0.025)
            .collect(),
        w_down: (0..hidden_dim * intermediate_dim)
            .map(|idx| ((idx + seed) % 8) as f32 * 0.005 - 0.015)
            .collect(),
        bq: None,
        bk: None,
        bv: None,
        bo: None,
        attn_norm_bias: None,
        ffn_norm_bias: None,
    }
}

fn tiny_inference(num_layers: usize) -> GpuInference<R> {
    let cfg = tiny_gpu_config(num_layers);
    GpuInference::<R>::from_raw_weights(
        cfg,
        &vec![0.02f32; 32 * 8],
        None,
        &vec![0.03f32; 32 * 8],
        &vec![1.0f32; 8],
        None,
        (0..num_layers).map(|layer_idx| tiny_raw_layer(layer_idx * 17)).collect(),
    )
    .unwrap()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{msg}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (idx, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < tol,
            "{msg}: mismatch at [{idx}]: lhs={lhs}, rhs={rhs}, diff={}",
            (lhs - rhs).abs(),
        );
    }
}

fn append_token(
    backend: &CubeclBackend<R>,
    pool: &GpuPagePool,
    page_id: PageId,
    offset_in_page: usize,
    key: &[f32],
    value: &[f32],
) {
    let client = backend.client();
    let page = pool.page(page_id);
    let page_base = page.base_offset();
    let kv_dim = page.kv_dim();

    let key_gpu = backend.from_f32(key);
    let value_gpu = backend.from_f32(value);

    unsafe {
        paged_cache_append_kernel::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&key_gpu.handle, kv_dim, 1),
            ArrayArg::from_raw_parts::<f32>(&pool.flat_buffer().handle, pool.flat_buffer().len, 1),
            ScalarArg::new(page_base as u32),
            ScalarArg::new(offset_in_page as u32),
            ScalarArg::new(kv_dim as u32),
            ScalarArg::new(0u32),
        );
        paged_cache_append_kernel::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&value_gpu.handle, kv_dim, 1),
            ArrayArg::from_raw_parts::<f32>(&pool.flat_buffer().handle, pool.flat_buffer().len, 1),
            ScalarArg::new(page_base as u32),
            ScalarArg::new(offset_in_page as u32),
            ScalarArg::new(kv_dim as u32),
            ScalarArg::new(kv_dim as u32),
        );
    }
}

fn flatten_cache(tokens: &[Vec<f32>]) -> Vec<f32> {
    tokens
        .iter()
        .flat_map(|token| token.iter().copied())
        .collect()
}

fn launch_contiguous_scores(
    backend: &CubeclBackend<R>,
    q_head: &[f32],
    k_cache: &[f32],
    head_dim: usize,
    kv_dim: usize,
    kv_head: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let client = backend.client();
    let q_gpu = backend.from_f32(q_head);
    let k_gpu = backend.from_f32(k_cache);
    let scores_gpu = backend.zeros(seq_len);

    unsafe {
        attention::attention_scores_kernel::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&q_gpu.handle, head_dim, 1),
            ArrayArg::from_raw_parts::<f32>(&k_gpu.handle, k_cache.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&scores_gpu.handle, seq_len, 1),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(kv_dim as u32),
            ScalarArg::new(kv_head as u32),
            ScalarArg::new(seq_len as u32),
            ScalarArg::new(scale),
        );
    }

    backend.to_f32(&scores_gpu)
}

fn launch_paged_scores(
    backend: &CubeclBackend<R>,
    q_head: &[f32],
    pool: &GpuPagePool,
    page_table: &[u32],
    kv_head: usize,
    num_tokens: usize,
    scale: f32,
) -> Vec<f32> {
    let client = backend.client();
    let q_gpu = backend.from_f32(q_head);
    let page_table_gpu = backend.from_u32(page_table);
    let scores_gpu = backend.zeros(num_tokens);
    let dummy_u32 = backend.from_u32(&[0u32]);
    let quant_payload = pool.quantized().map(|storage| storage.quant_payload());
    unsafe {
        paged_attention_scores_kernel::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&q_gpu.handle, pool.head_dim(), 1),
            ArrayArg::from_raw_parts::<f32>(&pool.flat_buffer().handle, pool.flat_buffer().len, 1),
            ArrayArg::from_raw_parts::<u32>(
                &quant_payload.unwrap_or(&dummy_u32).handle,
                quant_payload.map(|buffer| buffer.len).unwrap_or(dummy_u32.len),
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&page_table_gpu.handle, page_table.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&scores_gpu.handle, num_tokens, 1),
            ScalarArg::new(pool.page_stride() as u32),
            ScalarArg::new(pool.page_size() as u32),
            ScalarArg::new(num_tokens as u32),
            ScalarArg::new(kv_head as u32),
            ScalarArg::new(pool.head_dim() as u32),
            ScalarArg::new(scale),
        );
    }

    backend.to_f32(&scores_gpu)
}

fn launch_contiguous_contract(
    backend: &CubeclBackend<R>,
    scores: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    kv_dim: usize,
    kv_head: usize,
    seq_len: usize,
) -> Vec<f32> {
    let client = backend.client();
    let scores_gpu = backend.from_f32(scores);
    let v_gpu = backend.from_f32(v_cache);
    let output_gpu = backend.zeros(head_dim);

    unsafe {
        attention::attention_contract_kernel::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&scores_gpu.handle, seq_len, 1),
            ArrayArg::from_raw_parts::<f32>(&v_gpu.handle, v_cache.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_gpu.handle, head_dim, 1),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(kv_dim as u32),
            ScalarArg::new(kv_head as u32),
            ScalarArg::new(seq_len as u32),
        );
    }

    backend.to_f32(&output_gpu)
}

fn launch_paged_contract(
    backend: &CubeclBackend<R>,
    scores: &[f32],
    pool: &GpuPagePool,
    page_table: &[u32],
    kv_head: usize,
    num_tokens: usize,
) -> Vec<f32> {
    let client = backend.client();
    let scores_gpu = backend.from_f32(scores);
    let page_table_gpu = backend.from_u32(page_table);
    let output_gpu = backend.zeros(pool.head_dim());
    let dummy_u32 = backend.from_u32(&[0u32]);
    let quant_payload = pool.quantized().map(|storage| storage.quant_payload());
    unsafe {
        paged_attention_contract_kernel::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&scores_gpu.handle, num_tokens, 1),
            ArrayArg::from_raw_parts::<f32>(&pool.flat_buffer().handle, pool.flat_buffer().len, 1),
            ArrayArg::from_raw_parts::<u32>(
                &quant_payload.unwrap_or(&dummy_u32).handle,
                quant_payload.map(|buffer| buffer.len).unwrap_or(dummy_u32.len),
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&page_table_gpu.handle, page_table.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_gpu.handle, pool.head_dim(), 1),
            ScalarArg::new(pool.page_stride() as u32),
            ScalarArg::new(pool.page_size() as u32),
            ScalarArg::new(num_tokens as u32),
            ScalarArg::new(kv_head as u32),
            ScalarArg::new(pool.head_dim() as u32),
        );
    }

    backend.to_f32(&output_gpu)
}

#[test]
fn gpu_page_pool_store_round_trip() {
    let backend = backend();
    let pool = GpuPagePool::new(3, 2, 2, 3, &backend);

    let key0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let val0 = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let key1 = vec![7.0, 8.0, 9.0, 16.0, 17.0, 18.0];
    let val1 = vec![19.0, 20.0, 21.0, 22.0, 23.0, 24.0];
    let key2 = vec![25.0, 26.0, 27.0, 28.0, 29.0, 30.0];
    let val2 = vec![31.0, 32.0, 33.0, 34.0, 35.0, 36.0];

    append_token(&backend, &pool, PageId(0), 0, &key0, &val0);
    append_token(&backend, &pool, PageId(0), 1, &key1, &val1);
    append_token(&backend, &pool, PageId(1), 0, &key2, &val2);

    let flat = backend.to_f32(pool.flat_buffer());
    let page0 = pool.page(PageId(0));
    let page1 = pool.page(PageId(1));

    assert_eq!(
        page0.memory_bytes(),
        page0.len() * std::mem::size_of::<f32>()
    );
    assert_eq!(
        &flat[page0.base_offset()..page0.base_offset() + page0.slot_size()],
        [key0.clone(), val0.clone()].concat().as_slice(),
    );
    assert_eq!(
        &flat[page0.base_offset() + page0.slot_size()..page0.base_offset() + page0.len()],
        [key1.clone(), val1.clone()].concat().as_slice(),
    );
    assert_eq!(
        &flat[page1.base_offset()..page1.base_offset() + page1.slot_size()],
        [key2.clone(), val2.clone()].concat().as_slice(),
    );
}

#[test]
fn paged_attention_matches_contiguous_across_page_boundary() {
    let backend = backend();
    let page_size = 2usize;
    let num_kv_heads = 2usize;
    let head_dim = 4usize;
    let kv_dim = num_kv_heads * head_dim;
    let seq_len = 5usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let pool = GpuPagePool::new(3, page_size, num_kv_heads, head_dim, &backend);
    let page_table = vec![0u32, 1u32, 2u32];

    let keys: Vec<Vec<f32>> = (0..seq_len)
        .map(|token| {
            (0..kv_dim)
                .map(|dim| token as f32 * 0.5 + dim as f32 * 0.25)
                .collect()
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..seq_len)
        .map(|token| {
            (0..kv_dim)
                .map(|dim| -(token as f32) * 0.75 + dim as f32 * 0.1)
                .collect()
        })
        .collect();

    for token in 0..seq_len {
        append_token(
            &backend,
            &pool,
            PageId(page_table[token / page_size]),
            token % page_size,
            &keys[token],
            &values[token],
        );
    }

    let q_head = vec![0.3, -0.1, 0.7, 0.2];
    let kv_head = 1usize;

    let contiguous_scores = launch_contiguous_scores(
        &backend,
        &q_head,
        &flatten_cache(&keys),
        head_dim,
        kv_dim,
        kv_head,
        seq_len,
        scale,
    );
    let paged_scores = launch_paged_scores(
        &backend,
        &q_head,
        &pool,
        &page_table,
        kv_head,
        seq_len,
        scale,
    );
    assert_close(
        &paged_scores,
        &contiguous_scores,
        1e-5,
        "paged scores vs contiguous scores",
    );

    let mut contiguous_probs = backend.from_f32(&contiguous_scores);
    backend.softmax_inplace(&mut contiguous_probs, 0, seq_len);
    let contiguous_probs = backend.to_f32(&contiguous_probs);

    let mut paged_probs = backend.from_f32(&paged_scores);
    backend.softmax_inplace(&mut paged_probs, 0, seq_len);
    let paged_probs = backend.to_f32(&paged_probs);
    assert_close(
        &paged_probs,
        &contiguous_probs,
        1e-5,
        "paged softmax vs contiguous softmax",
    );

    let contiguous_out = launch_contiguous_contract(
        &backend,
        &contiguous_probs,
        &flatten_cache(&values),
        head_dim,
        kv_dim,
        kv_head,
        seq_len,
    );
    let paged_out =
        launch_paged_contract(&backend, &paged_probs, &pool, &page_table, kv_head, seq_len);
    assert_close(
        &paged_out,
        &contiguous_out,
        1e-5,
        "paged contract vs contiguous contract",
    );
}

#[test]
fn paged_attention_supports_gqa() {
    let backend = backend();
    let page_size = 2usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 4usize;
    let heads_per_kv = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let seq_len = 4usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let pool = GpuPagePool::new(2, page_size, num_kv_heads, head_dim, &backend);
    let page_table = vec![0u32, 1u32];

    let keys: Vec<Vec<f32>> = (0..seq_len)
        .map(|token| {
            (0..kv_dim)
                .map(|dim| (token * kv_dim + dim) as f32 * 0.05)
                .collect()
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..seq_len)
        .map(|token| {
            (0..kv_dim)
                .map(|dim| ((token * kv_dim + dim) as f32 * 0.03) - 0.2)
                .collect()
        })
        .collect();

    for token in 0..seq_len {
        append_token(
            &backend,
            &pool,
            PageId(page_table[token / page_size]),
            token % page_size,
            &keys[token],
            &values[token],
        );
    }

    let q_heads = [
        vec![0.4, 0.1, -0.2, 0.7],
        vec![-0.6, 0.3, 0.5, 0.2],
        vec![0.9, -0.4, 0.2, -0.1],
        vec![0.15, 0.45, -0.35, 0.8],
    ];

    let contiguous_keys = flatten_cache(&keys);
    let contiguous_values = flatten_cache(&values);

    for (head_idx, q_head) in q_heads.iter().enumerate() {
        let kv_head = head_idx / heads_per_kv;

        let contiguous_scores = launch_contiguous_scores(
            &backend,
            q_head,
            &contiguous_keys,
            head_dim,
            kv_dim,
            kv_head,
            seq_len,
            scale,
        );
        let paged_scores = launch_paged_scores(
            &backend,
            q_head,
            &pool,
            &page_table,
            kv_head,
            seq_len,
            scale,
        );
        assert_close(
            &paged_scores,
            &contiguous_scores,
            1e-5,
            &format!("gqa scores head {head_idx}"),
        );

        let mut contiguous_probs = backend.from_f32(&contiguous_scores);
        backend.softmax_inplace(&mut contiguous_probs, 0, seq_len);
        let contiguous_probs = backend.to_f32(&contiguous_probs);

        let mut paged_probs = backend.from_f32(&paged_scores);
        backend.softmax_inplace(&mut paged_probs, 0, seq_len);
        let paged_probs = backend.to_f32(&paged_probs);

        let contiguous_out = launch_contiguous_contract(
            &backend,
            &contiguous_probs,
            &contiguous_values,
            head_dim,
            kv_dim,
            kv_head,
            seq_len,
        );
        let paged_out =
            launch_paged_contract(&backend, &paged_probs, &pool, &page_table, kv_head, seq_len);
        assert_close(
            &paged_out,
            &contiguous_out,
            1e-5,
            &format!("gqa contract head {head_idx}"),
        );
    }
}

#[test]
fn paged_forward_matches_contiguous_gpu_cache() {
    let gpu = tiny_inference(2);
    let pool = gpu.new_page_pool(4, 2, None);
    let mut contiguous_cache = gpu.new_cache();
    let layer0_page_ids = [PageId(0), PageId(1)];
    let layer1_page_ids = [PageId(2), PageId(3)];
    let page_tables = [layer0_page_ids.as_slice(), layer1_page_ids.as_slice()];
    let prompt = [1u32, 2, 3];

    let contiguous_prefill = gpu.forward_batch(&mut contiguous_cache, &prompt);
    let paged_prefill = gpu.forward_batch_paged(&pool, &page_tables, 0, &prompt);
    assert_close(
        &paged_prefill,
        &contiguous_prefill,
        1e-5,
        "paged prefill vs contiguous prefill",
    );

    let contiguous_decode = gpu.forward_token(&mut contiguous_cache, 4);
    let paged_decode = gpu.forward_token_paged(&pool, &page_tables, prompt.len(), 4);
    assert_close(
        &paged_decode,
        &contiguous_decode,
        1e-5,
        "paged decode vs contiguous decode",
    );
}

#[test]
fn quantized_paged_forward_matches_contiguous_gpu_cache() {
    let gpu = tiny_inference(2);
    let mut pool = gpu.new_page_pool(
        4,
        4,
        Some(GpuPagedKvQuantConfig {
            residual_sketch_dim: 16,
        }),
    );
    let mut contiguous_cache = gpu.new_cache();
    let layer0_page_ids = [PageId(0), PageId(1)];
    let layer1_page_ids = [PageId(2), PageId(3)];
    let page_tables = [layer0_page_ids.as_slice(), layer1_page_ids.as_slice()];
    let prompt = [1u32, 2, 3, 4, 5];

    let contiguous_prefill = gpu.forward_batch(&mut contiguous_cache, &prompt);
    let paged_prefill = gpu.forward_batch_paged(&pool, &page_tables, 0, &prompt);
    assert_close(
        &paged_prefill,
        &contiguous_prefill,
        1e-5,
        "quantized paged prefill vs contiguous prefill",
    );

    gpu.quantize_completed_pages(&mut pool, &page_tables, prompt.len());
    assert!(pool.is_page_quantized(PageId(0)));
    assert!(pool.is_page_quantized(PageId(2)));

    let contiguous_decode = gpu.forward_token(&mut contiguous_cache, 4);
    let quantized_decode = gpu.forward_token_paged(&pool, &page_tables, prompt.len(), 4);
    assert_close(
        &quantized_decode,
        &contiguous_decode,
        1e-4,
        "quantized paged decode vs contiguous decode",
    );
}

#[test]
fn quantized_paged_forward_matches_contiguous_across_multiple_quantized_pages() {
    let gpu = tiny_inference(2);
    let mut pool = gpu.new_page_pool(
        6,
        4,
        Some(GpuPagedKvQuantConfig {
            residual_sketch_dim: 16,
        }),
    );
    let mut contiguous_cache = gpu.new_cache();
    let layer0_page_ids = [PageId(0), PageId(1), PageId(2)];
    let layer1_page_ids = [PageId(3), PageId(4), PageId(5)];
    let page_tables = [layer0_page_ids.as_slice(), layer1_page_ids.as_slice()];
    let prompt = [1u32, 2, 3, 4, 5, 6, 7, 8, 9];

    let contiguous_prefill = gpu.forward_batch(&mut contiguous_cache, &prompt);
    let paged_prefill = gpu.forward_batch_paged(&pool, &page_tables, 0, &prompt);
    assert_close(
        &paged_prefill,
        &contiguous_prefill,
        1e-5,
        "multi-page quantized paged prefill vs contiguous prefill",
    );

    gpu.quantize_completed_pages(&mut pool, &page_tables, prompt.len());
    assert!(pool.is_page_quantized(PageId(0)));
    assert!(pool.is_page_quantized(PageId(1)));
    assert!(pool.is_page_quantized(PageId(3)));
    assert!(pool.is_page_quantized(PageId(4)));
    assert!(!pool.is_page_quantized(PageId(2)));
    assert!(!pool.is_page_quantized(PageId(5)));

    let contiguous_decode = gpu.forward_token(&mut contiguous_cache, 10);
    let quantized_decode = gpu.forward_token_paged(&pool, &page_tables, prompt.len(), 10);
    assert_close(
        &quantized_decode,
        &contiguous_decode,
        1e-4,
        "multi-page quantized paged decode vs contiguous decode",
    );
}

#[test]
#[should_panic(expected = "unsupported")]
fn quantized_page_pool_rejects_unsupported_residual_sketch_dim() {
    let backend = backend();
    let _pool = GpuPagePool::new_with_quantization(
        1,
        4,
        2,
        4,
        Some(GpuPagedKvQuantConfig {
            residual_sketch_dim: 8,
        }),
        &backend,
    );
}
