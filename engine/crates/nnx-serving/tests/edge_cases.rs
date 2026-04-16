//! Edge case tests for the serving infrastructure.
//!
//! These tests target boundary conditions, unusual state transitions,
//! and resource lifecycle correctness that are hard to hit with
//! happy-path integration tests.

#[cfg(feature = "gpu")]
use nnx_core::device::Device;
use nnx_core::engine::KVStore;
use nnx_serving::backend::ServingEngine;
use nnx_serving::block_manager::BlockAllocator;
use nnx_serving::config::ServingConfig;
use nnx_serving::page::PageId;
use nnx_serving::paged_cache::{PagedLayerView, SequencePageTable};
use nnx_serving::prefix_cache::{compute_hash_chain, PageHash, PrefixCache};
use nnx_serving::scheduler::Scheduler;
use nnx_serving::sequence::{FinishReason, Sequence, SequenceId, SequenceState};
use nnx_transformer::block::BlockWeights;
use nnx_transformer::config::*;
use nnx_transformer::model::{Model, ModelWeights};
use nnx_transformer::weights::Matrix;

// ===================================================================
// Helpers
// ===================================================================

fn tiny_model(num_layers: usize) -> Model {
    let hd = 8;
    let nh = 2;
    let nkv = 2;
    let hdim = 4;
    let inter = 16;
    let vocab = 32;

    let cfg = ModelConfig {
        architecture: "test".into(),
        arch: Architecture::Llama,
        num_layers,
        hidden_dim: hd,
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hdim,
        intermediate_dim: inter,
        vocab_size: vocab,
        max_context_length: 128,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        norm_type: NormType::RMSNorm,
        ffn_type: FFNType::SwiGLU,
        pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
        block_style: BlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
        embedding_scale: None,
    };

    let q_dim = nh * hdim;
    let kv_dim = nkv * hdim;

    let layer = |s: usize| BlockWeights {
        attn_norm: vec![1.0; hd],
        ffn_norm: vec![1.0; hd],
        wq: Matrix::dense(
            (0..q_dim * hd)
                .map(|i| ((i + s) % 7) as f32 * 0.02 - 0.06)
                .collect(),
            q_dim,
            hd,
        ),
        wk: Matrix::dense(
            (0..kv_dim * hd)
                .map(|i| ((i + s) % 11) as f32 * 0.02 - 0.1)
                .collect(),
            kv_dim,
            hd,
        ),
        wv: Matrix::dense(
            (0..kv_dim * hd)
                .map(|i| ((i + s) % 13) as f32 * 0.02 - 0.12)
                .collect(),
            kv_dim,
            hd,
        ),
        wo: Matrix::dense(
            (0..hd * q_dim)
                .map(|i| ((i + s) % 5) as f32 * 0.02 - 0.04)
                .collect(),
            hd,
            q_dim,
        ),
        w_gate: Matrix::dense(vec![0.01; inter * hd], inter, hd),
        w_up: Matrix::dense(vec![0.01; inter * hd], inter, hd),
        w_down: Matrix::dense(vec![0.01; hd * inter], hd, inter),
        bq: None,
        bk: None,
        bv: None,
        bo: None,
        attn_norm_bias: None,
        ffn_norm_bias: None,
    };

    let weights = ModelWeights {
        token_embedding: Matrix::dense(
            (0..vocab * hd)
                .map(|i| (i % 19) as f32 * 0.05 - 0.45)
                .collect(),
            vocab,
            hd,
        ),
        position_embedding: None,
        layers: (0..num_layers).map(|i| layer(i * 17)).collect(),
        final_norm: vec![1.0; hd],
        final_norm_bias: None,
        lm_head: Matrix::dense(
            (0..vocab * hd)
                .map(|i| (i % 23) as f32 * 0.03 - 0.33)
                .collect(),
            vocab,
            hd,
        ),
    };

    Model::new(cfg, weights)
}

fn serving_config(
    page_size: usize,
    max_pages: usize,
    max_seq: usize,
    prefix_cache: bool,
) -> ServingConfig {
    ServingConfig {
        page_size,
        max_pages,
        max_sequences: max_seq,
        max_batch_size: max_seq,
        enable_prefix_caching: prefix_cache,
        max_prefix_cache_entries: 4,
        max_prefill_tokens: 0,
        gpu_kv_quantization: Default::default(),
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as u32
}

/// Drive engine to completion, returning all generated token IDs per sequence.
fn drive_to_completion(
    engine: &mut ServingEngine,
    max_iters: usize,
) -> Vec<(nnx_serving::sequence::SequenceId, FinishReason)> {
    let mut finished = Vec::new();
    for _ in 0..max_iters {
        if !engine.has_work() {
            break;
        }
        let out = engine.step().unwrap();
        for s in &out.outputs {
            let tok = argmax(&s.logits);
            engine.on_token_generated(s.seq_id, tok, false);
        }
        finished.extend(out.finished);
    }
    finished
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_serving_prefix_cache_matches_uncached_logits() {
    let prompt = vec![1, 2, 3, 4, 5];

    let mut uncached = ServingEngine::new_with_device(
        tiny_model(2),
        serving_config(4, 64, 4, false),
        Device::Gpu(0),
    )
    .unwrap();
    let uncached_id = uncached.add_request(prompt.clone(), 1);
    let uncached_out = uncached.step().unwrap();
    let uncached_logits = uncached_out
        .outputs
        .iter()
        .find(|out| out.seq_id == uncached_id)
        .unwrap()
        .logits
        .clone();

    let mut cached = ServingEngine::new_with_device(
        tiny_model(2),
        serving_config(4, 64, 4, true),
        Device::Gpu(0),
    )
    .unwrap();

    let warm_id = cached.add_request(prompt.clone(), 1);
    cached.step().unwrap();
    cached.on_token_generated(warm_id, 10, false);
    cached.step().unwrap();

    let pages_before_hit = cached.allocator_stats().used_pages;
    let cached_id = cached.add_request(prompt.clone(), 1);
    let pages_after_hit = cached.allocator_stats().used_pages;
    assert_eq!(
        pages_after_hit, pages_before_hit,
        "prefix cache hit should share existing GPU pages instead of allocating new ones"
    );

    let cached_out = cached.step().unwrap();
    let cached_logits = cached_out
        .outputs
        .iter()
        .find(|out| out.seq_id == cached_id)
        .unwrap()
        .logits
        .clone();

    assert_eq!(uncached_logits.len(), cached_logits.len());
    for (idx, (lhs, rhs)) in uncached_logits.iter().zip(cached_logits.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-5,
            "GPU prefix-cache logit mismatch at {idx}: uncached={lhs}, cached={rhs}"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_serving_page_aligned_full_prefix_hit_recomputes_last_page() {
    let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let mut uncached = ServingEngine::new_with_device(
        tiny_model(2),
        serving_config(4, 64, 4, false),
        Device::Gpu(0),
    )
    .unwrap();
    let uncached_id = uncached.add_request(prompt.clone(), 1);
    let uncached_logits = uncached
        .step()
        .unwrap()
        .outputs
        .into_iter()
        .find(|out| out.seq_id == uncached_id)
        .unwrap()
        .logits;

    let mut cached = ServingEngine::new_with_device(
        tiny_model(2),
        serving_config(4, 64, 4, true),
        Device::Gpu(0),
    )
    .unwrap();

    let warm_id = cached.add_request(prompt.clone(), 1);
    cached.step().unwrap();
    cached.on_token_generated(warm_id, 10, false);
    cached.step().unwrap();

    let cached_id = cached.add_request(prompt.clone(), 1);
    let cached_logits = cached
        .step()
        .unwrap()
        .outputs
        .into_iter()
        .find(|out| out.seq_id == cached_id)
        .unwrap()
        .logits;

    assert_eq!(uncached_logits.len(), cached_logits.len());
    for (idx, (lhs, rhs)) in uncached_logits.iter().zip(cached_logits.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-5,
            "GPU full-hit prefix cache mismatch at {idx}: uncached={lhs}, cached={rhs}"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_serving_quantized_kv_matches_dense_decode() {
    let prompt = vec![1, 2, 3, 4, 5];

    let mut dense = ServingEngine::new_with_device(
        tiny_model(2),
        serving_config(4, 64, 4, false),
        Device::Gpu(0),
    )
    .unwrap();

    let mut quantized_cfg = serving_config(4, 64, 4, false);
    quantized_cfg.gpu_kv_quantization.enabled = true;
    let mut quantized =
        ServingEngine::new_with_device(tiny_model(2), quantized_cfg, Device::Gpu(0)).unwrap();

    let dense_id = dense.add_request(prompt.clone(), 2);
    let quantized_id = quantized.add_request(prompt.clone(), 2);

    let dense_prefill = dense.step().unwrap();
    let quantized_prefill = quantized.step().unwrap();

    let dense_prefill_logits = dense_prefill
        .outputs
        .iter()
        .find(|out| out.seq_id == dense_id)
        .unwrap()
        .logits
        .clone();
    let quantized_prefill_logits = quantized_prefill
        .outputs
        .iter()
        .find(|out| out.seq_id == quantized_id)
        .unwrap()
        .logits
        .clone();

    for (idx, (lhs, rhs)) in dense_prefill_logits
        .iter()
        .zip(quantized_prefill_logits.iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() < 1e-5,
            "GPU quantized prefill mismatch at {idx}: dense={lhs}, quantized={rhs}"
        );
    }

    dense.on_token_generated(dense_id, 7, false);
    quantized.on_token_generated(quantized_id, 7, false);

    let dense_decode_logits = dense
        .step()
        .unwrap()
        .outputs
        .into_iter()
        .find(|out| out.seq_id == dense_id)
        .unwrap()
        .logits;
    let quantized_decode_logits = quantized
        .step()
        .unwrap()
        .outputs
        .into_iter()
        .find(|out| out.seq_id == quantized_id)
        .unwrap()
        .logits;

    assert_eq!(argmax(&dense_decode_logits), argmax(&quantized_decode_logits));

    for (idx, (lhs, rhs)) in dense_decode_logits
        .iter()
        .zip(quantized_decode_logits.iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() < 5e-2,
            "GPU quantized decode mismatch at {idx}: dense={lhs}, quantized={rhs}"
        );
    }
}

// ===================================================================
// Block Allocator Edge Cases
// ===================================================================

#[test]
fn allocator_full_cycle_reuse() {
    // Allocate all pages, free all, re-allocate all — pool must be fully recyclable.
    let mut alloc = BlockAllocator::new(4, 2, 1, 2);
    let ids: Vec<PageId> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
    assert_eq!(alloc.free_count(), 0);

    for id in &ids {
        alloc.dec_ref(*id).unwrap();
    }
    assert_eq!(alloc.free_count(), 4);

    // Re-allocate all 4 — should work.
    let ids2: Vec<PageId> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
    assert_eq!(alloc.free_count(), 0);

    // Clean up.
    for id in &ids2 {
        alloc.dec_ref(*id).unwrap();
    }
    assert_eq!(alloc.free_count(), 4);
}

#[test]
fn allocator_cow_on_last_free_page() {
    // Pool has 3 pages. Allocate 2, share one, CoW needs the last free page.
    let mut alloc = BlockAllocator::new(3, 2, 1, 2);
    let p0 = alloc.allocate().unwrap();
    let _p1 = alloc.allocate().unwrap();
    alloc.inc_ref(p0).unwrap(); // p0 ref=2

    // 1 free page left — CoW should succeed using it.
    let p2 = alloc.cow_if_shared(p0).unwrap();
    assert_ne!(p0, p2);
    assert_eq!(alloc.free_count(), 0);
}

#[test]
fn allocator_high_ref_count() {
    let mut alloc = BlockAllocator::new(1, 2, 1, 2);
    let p = alloc.allocate().unwrap();

    // Inc ref 100 times.
    for _ in 0..100 {
        alloc.inc_ref(p).unwrap();
    }
    assert_eq!(alloc.ref_count(p), 101);

    // Dec ref 100 times — page should still be in use.
    for _ in 0..100 {
        alloc.dec_ref(p).unwrap();
    }
    assert_eq!(alloc.ref_count(p), 1);
    assert_eq!(alloc.free_count(), 0);

    // Final dec_ref frees it.
    alloc.dec_ref(p).unwrap();
    assert_eq!(alloc.free_count(), 1);
}

// ===================================================================
// Paged Cache Edge Cases
// ===================================================================

#[test]
fn paged_cache_exactly_fills_page() {
    // Store exactly page_size tokens — should fill one page exactly.
    let page_size = 4;
    let mut alloc = BlockAllocator::new(4, page_size, 1, 2);
    let mut pt: Vec<PageId> = Vec::new();

    let final_count = {
        let mut view = PagedLayerView::new(&mut alloc, &mut pt, 0, 0, page_size, 1, 2);
        for _ in 0..page_size {
            view.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        }
        view.token_count()
    };

    assert_eq!(final_count, page_size);
    assert_eq!(pt.len(), 1); // exactly one page
    assert_eq!(alloc.used_count(), 1);

    // One more token crosses to a second page.
    let final_count = {
        let mut view = PagedLayerView::new(&mut alloc, &mut pt, final_count, 0, page_size, 1, 2);
        view.store(&[5.0, 6.0], &[7.0, 8.0]).unwrap();
        view.token_count()
    };

    assert_eq!(final_count, page_size + 1);
    assert_eq!(pt.len(), 2);
    assert_eq!(alloc.used_count(), 2);
}

#[test]
fn paged_cache_continued_generation() {
    // Simulate continued generation: create view with initial_tokens > 0.
    let page_size = 4;
    let mut alloc = BlockAllocator::new(8, page_size, 1, 2);
    let mut pt: Vec<PageId> = Vec::new();

    // Phase 1: store 3 tokens.
    let count = {
        let mut view = PagedLayerView::new(&mut alloc, &mut pt, 0, 0, page_size, 1, 2);
        for i in 0..3 {
            view.store(&[i as f32, (i + 1) as f32], &[0.0; 2]).unwrap();
        }
        view.token_count()
    };
    assert_eq!(count, 3);

    // Phase 2: resume from count=3, store 2 more.
    let count = {
        let mut view = PagedLayerView::new(&mut alloc, &mut pt, count, 0, page_size, 1, 2);
        for i in 3..5 {
            view.store(&[i as f32, (i + 1) as f32], &[0.0; 2]).unwrap();
        }
        // Read back all 5 tokens.
        for i in 0..5 {
            assert_eq!(view.key_at(i, 0)[0], i as f32, "token {} key mismatch", i);
        }
        view.token_count()
    };
    assert_eq!(count, 5);
    assert_eq!(pt.len(), 2); // page 0: tokens 0-3, page 1: token 4
}

#[test]
fn paged_cache_page_size_1() {
    // Degenerate: every token gets its own page.
    let mut alloc = BlockAllocator::new(16, 1, 1, 2);
    let mut pt: Vec<PageId> = Vec::new();

    let count = {
        let mut view = PagedLayerView::new(&mut alloc, &mut pt, 0, 0, 1, 1, 2);
        for i in 0..5 {
            view.store(&[i as f32, 0.0], &[0.0, i as f32]).unwrap();
        }
        for i in 0..5 {
            assert_eq!(view.key_at(i, 0)[0], i as f32);
            assert_eq!(view.value_at(i, 0)[1], i as f32);
        }
        view.token_count()
    };

    assert_eq!(count, 5);
    assert_eq!(pt.len(), 5); // one page per token
}

// ===================================================================
// Sequence State Machine Edge Cases
// ===================================================================

#[test]
fn sequence_zero_length_prompt() {
    let mut seq = Sequence::new(SequenceId(1), vec![], 5, 2, 4);
    seq.start_prefill();
    // Empty prompt → should go straight to Decoding.
    assert_eq!(seq.state, SequenceState::Decoding);
}

#[test]
fn sequence_max_new_tokens_zero() {
    let mut seq = Sequence::new(SequenceId(1), vec![1, 2], 0, 2, 4);
    seq.start_prefill();
    seq.advance_prefill(2);
    assert_eq!(
        seq.state,
        SequenceState::Finished {
            reason: FinishReason::MaxTokens
        }
    );
    assert!(seq.should_stop()); // max_new_tokens=0 → stop immediately
}

// ===================================================================
// Scheduler Edge Cases
// ===================================================================

#[test]
fn scheduler_cancel_before_any_step() {
    let config = serving_config(4, 32, 4, false);
    let mut sched = Scheduler::new(config, 2);
    let alloc = BlockAllocator::new(32, 4, 1, 2);

    let id = sched.add_request(vec![1, 2, 3], 5);
    // Cancel while still in waiting queue — no pages allocated.
    let pages = sched.cancel_request(id).unwrap();
    assert!(pages.is_empty());
    assert!(!sched.has_work());
}

#[test]
fn scheduler_cancel_during_prefill() {
    let config = serving_config(4, 32, 4, false);
    let mut sched = Scheduler::new(config, 2);
    let alloc = BlockAllocator::new(32, 4, 1, 2);

    let id = sched.add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 5);
    sched.step(&alloc); // admit, start prefill
                        // Partially prefill.
    sched.on_prefill_advanced(id, 4);

    // Cancel during prefill.
    let pages = sched.cancel_request(id).unwrap();
    // Pages were returned (even if empty in this unit test — the engine populates them).
    assert!(!sched.has_work());
}

#[test]
fn scheduler_fill_cancel_refill() {
    // Fill to max_sequences, cancel one, add new — the slot should be reusable.
    let config = serving_config(4, 64, 2, false);
    let mut sched = Scheduler::new(config, 2);
    let alloc = BlockAllocator::new(64, 4, 1, 2);

    let id1 = sched.add_request(vec![1], 100);
    let id2 = sched.add_request(vec![2], 100);
    sched.step(&alloc); // admit both

    assert_eq!(sched.stats().running, 2);

    // Third request should wait (max_sequences=2).
    sched.add_request(vec![3], 100);
    let output = sched.step(&alloc);
    assert_eq!(sched.stats().running, 2);
    assert_eq!(sched.stats().waiting, 1);

    // Cancel one running request.
    sched.cancel_request(id1).unwrap();
    assert_eq!(sched.stats().running, 1);

    // Next step should admit the waiting request.
    let output = sched.step(&alloc);
    assert_eq!(sched.stats().running, 2);
    assert_eq!(sched.stats().waiting, 0);
}

#[test]
fn scheduler_all_finish_same_step() {
    let config = serving_config(4, 64, 4, false);
    let mut sched = Scheduler::new(config, 2);
    let alloc = BlockAllocator::new(64, 4, 1, 2);

    let id1 = sched.add_request(vec![1], 1);
    let id2 = sched.add_request(vec![2], 1);
    let id3 = sched.add_request(vec![3], 1);

    sched.step(&alloc); // prefill all
    for &id in &[id1, id2, id3] {
        sched.on_prefill_advanced(id, 1);
    }

    sched.step(&alloc); // decode all
    for &id in &[id1, id2, id3] {
        sched.on_token_generated(id, 99, false);
    }

    // All three should finish in the same step.
    let output = sched.step(&alloc);
    assert_eq!(output.finished.len(), 3);
    assert!(!sched.has_work());
}

#[test]
fn scheduler_step_with_no_work() {
    let config = serving_config(4, 32, 4, false);
    let mut sched = Scheduler::new(config, 2);
    let alloc = BlockAllocator::new(32, 4, 1, 2);

    // Step with nothing to do — should be a no-op.
    let output = sched.step(&alloc);
    assert!(output.prefill.is_empty());
    assert!(output.decode.is_empty());
    assert!(output.finished.is_empty());
}

// ===================================================================
// Prefix Cache Edge Cases
// ===================================================================

#[test]
fn prefix_cache_prompt_shorter_than_page() {
    // 3 tokens with page_size=4 — no complete page, nothing cached.
    let mut cache = PrefixCache::new(true);
    let page_size = 4;

    let chain = compute_hash_chain(&[1, 2, 3], page_size);
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].1.len(), 3); // partial page

    // Insert should be rejected (partial page).
    let inserted = cache.insert(chain[0].0, vec![PageId(0)], 3, page_size);
    assert!(!inserted);
    assert_eq!(cache.len(), 0);
}

#[test]
fn prefix_cache_prompt_exactly_one_page() {
    let mut cache = PrefixCache::new(true);
    let page_size = 4;

    let chain = compute_hash_chain(&[1, 2, 3, 4], page_size);
    assert_eq!(chain.len(), 1);

    let inserted = cache.insert(chain[0].0, vec![PageId(0)], 4, page_size);
    assert!(inserted);
    assert_eq!(cache.len(), 1);

    // Lookup should find it.
    let result = cache.lookup(&[1, 2, 3, 4], page_size);
    assert_eq!(result.cached_tokens, 4);
}

#[test]
fn prefix_cache_different_prompts_same_length_no_collision() {
    let mut cache = PrefixCache::new(true);
    let page_size = 4;

    let chain_a = compute_hash_chain(&[1, 2, 3, 4], page_size);
    let chain_b = compute_hash_chain(&[5, 6, 7, 8], page_size);

    // Different tokens → different hashes.
    assert_ne!(chain_a[0].0, chain_b[0].0);

    cache.insert(chain_a[0].0, vec![PageId(0)], 4, page_size);
    cache.insert(chain_b[0].0, vec![PageId(1)], 4, page_size);
    assert_eq!(cache.len(), 2);

    // Lookup A finds A, not B.
    let result = cache.lookup(&[1, 2, 3, 4], page_size);
    assert_eq!(result.matched_pages[0], vec![PageId(0)]);

    let result = cache.lookup(&[5, 6, 7, 8], page_size);
    assert_eq!(result.matched_pages[0], vec![PageId(1)]);
}

#[test]
fn prefix_cache_chain_dependency_prevents_false_sharing() {
    // Two prompts share the same second page tokens but different first page.
    // Chain hashing should prevent sharing the second page.
    let mut cache = PrefixCache::new(true);
    let page_size = 4;

    // Prompt A: [1,2,3,4, 9,10,11,12]
    let chain_a = compute_hash_chain(&[1, 2, 3, 4, 9, 10, 11, 12], page_size);
    cache.insert(chain_a[0].0, vec![PageId(0)], 4, page_size);
    cache.insert(chain_a[1].0, vec![PageId(1)], 4, page_size);

    // Prompt B: [5,6,7,8, 9,10,11,12] — same second page tokens, different first.
    let result = cache.lookup(&[5, 6, 7, 8, 9, 10, 11, 12], page_size);
    assert_eq!(
        result.cached_tokens, 0,
        "different first page should break the chain"
    );
}

#[test]
fn prefix_cache_evict_respects_lru_order() {
    let mut cache = PrefixCache::new(true);
    let page_size = 4;

    // Insert A, B, C in order.
    let ha = compute_hash_chain(&[1, 2, 3, 4], page_size)[0].0;
    let hb = compute_hash_chain(&[5, 6, 7, 8], page_size)[0].0;
    let hc = compute_hash_chain(&[9, 10, 11, 12], page_size)[0].0;

    cache.insert(ha, vec![PageId(0)], 4, page_size);
    cache.insert(hb, vec![PageId(1)], 4, page_size);
    cache.insert(hc, vec![PageId(2)], 4, page_size);

    // Touch B via lookup (moves to back of LRU).
    cache.lookup(&[5, 6, 7, 8], page_size);

    // Evict → should remove A (oldest untouched).
    let evicted = cache.evict_lru().unwrap();
    assert_eq!(evicted, vec![PageId(0)]);

    // Evict → should remove C (B was touched).
    let evicted = cache.evict_lru().unwrap();
    assert_eq!(evicted, vec![PageId(2)]);

    // Evict → should remove B.
    let evicted = cache.evict_lru().unwrap();
    assert_eq!(evicted, vec![PageId(1)]);

    assert!(cache.is_empty());
}

// ===================================================================
// ServingEngine Edge Cases
// ===================================================================

#[test]
fn engine_step_returns_empty_when_idle() {
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    let output = engine.step().unwrap();
    assert!(output.outputs.is_empty());
    assert!(output.finished.is_empty());
    assert_eq!(output.pages_freed, 0);
}

#[test]
fn engine_single_token_prompt() {
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    let id = engine.add_request(vec![1], 1);
    let out = engine.step().unwrap();
    assert_eq!(out.outputs.len(), 1);
    assert!(out.outputs[0].logits.iter().all(|v| v.is_finite()));

    engine.on_token_generated(id, 5, false);
    let out = engine.step().unwrap();
    assert_eq!(out.finished.len(), 1);
}

#[test]
fn engine_long_prompt_many_pages() {
    // 20 tokens, page_size=4 → 5 pages per layer, 2 layers → 10 pages.
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    let prompt: Vec<u32> = (1..=20).collect();
    let id = engine.add_request(prompt, 2);

    let out = engine.step().unwrap();
    assert_eq!(out.outputs.len(), 1);
    assert!(out.outputs[0].logits.iter().all(|v| v.is_finite()));

    engine.on_token_generated(id, 10, false);
    let out = engine.step().unwrap();
    assert_eq!(out.outputs.len(), 1);

    engine.on_token_generated(id, 11, false);
    let out = engine.step().unwrap();
    assert_eq!(out.finished.len(), 1);
    assert!(out.pages_freed > 0);
}

#[test]
fn engine_pages_fully_reclaimed_after_traffic() {
    let model = tiny_model(2);
    let config = serving_config(4, 64, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    let free_before = engine.allocator_stats().free_pages;

    // Run 5 sequential requests to completion.
    for prompt_start in 0..5u32 {
        let prompt: Vec<u32> = (prompt_start * 4 + 1..=prompt_start * 4 + 4).collect();
        let id = engine.add_request(prompt, 1);
        let out = engine.step().unwrap();
        engine.on_token_generated(id, 99, false);
        engine.step().unwrap(); // retire
    }

    let free_after = engine.allocator_stats().free_pages;
    assert_eq!(
        free_before, free_after,
        "all pages should be reclaimed after traffic: before={}, after={}",
        free_before, free_after,
    );
}

#[test]
fn engine_eos_on_first_decode_token() {
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    let id = engine.add_request(vec![1, 2], 100);
    engine.step().unwrap(); // prefill
    engine.on_token_generated(id, 0, true); // EOS immediately

    let out = engine.step().unwrap();
    assert_eq!(out.finished.len(), 1);
    assert_eq!(out.finished[0].1, FinishReason::EndOfSequence);
}

#[test]
fn engine_cancel_during_prefill_frees_pages() {
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    let free_before = engine.allocator_stats().free_pages;

    let id = engine.add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 10);
    engine.step().unwrap(); // prefill — allocates pages

    let free_during = engine.allocator_stats().free_pages;
    assert!(free_during < free_before);

    engine.cancel_request(id).unwrap();

    let free_after = engine.allocator_stats().free_pages;
    assert_eq!(
        free_before, free_after,
        "cancel during prefill should free pages"
    );
}

#[test]
fn engine_rapid_add_cancel_cycle() {
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    // Rapidly add and cancel 20 requests.
    for i in 0..20u32 {
        let id = engine.add_request(vec![i + 1], 1);
        // Cancel before stepping — still in waiting queue.
        engine.cancel_request(id).unwrap();
    }

    assert!(!engine.has_work());
    // No pages should have been leaked.
    let stats = engine.allocator_stats();
    assert_eq!(stats.used_pages, 0);
}

#[test]
fn engine_prefix_cache_fully_cached_prompt_no_panic() {
    // Regression test: fully-cached prompt should not panic.
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, true);

    let mut engine = ServingEngine::new(model, config).unwrap();

    // First request: 4 tokens (exactly 1 page).
    let prompt = vec![1, 2, 3, 4];
    let id1 = engine.add_request(prompt.clone(), 1);
    drive_to_completion(&mut engine, 10);

    // Second request: same 4 tokens — fully cached.
    let id2 = engine.add_request(prompt, 1);
    // This should NOT panic.
    let out = engine.step().unwrap();
    assert_eq!(
        out.outputs.len(),
        1,
        "fully-cached prompt should produce output"
    );
    assert!(out.outputs[0].logits.iter().all(|v| v.is_finite()));

    // Drive to completion.
    engine.on_token_generated(id2, 10, false);
    let out = engine.step().unwrap();
    assert_eq!(out.finished.len(), 1);
}

#[test]
fn engine_interleaved_prefill_and_decode() {
    // One request starts prefilling while another is already decoding.
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    // Start first request — goes through prefill → decode.
    let id1 = engine.add_request(vec![1, 2], 3);
    engine.step().unwrap(); // prefill id1
    engine.on_token_generated(id1, 10, false);

    // Now add second request while id1 is decoding.
    let id2 = engine.add_request(vec![3, 4, 5], 2);

    // Step should have id1 decoding and id2 prefilling.
    let out = engine.step().unwrap();
    assert_eq!(out.outputs.len(), 2, "both sequences should produce output");

    // Both outputs should be finite.
    for s in &out.outputs {
        assert!(
            s.logits.iter().all(|v| v.is_finite()),
            "seq {:?} produced non-finite logits",
            s.seq_id,
        );
    }
}

#[test]
fn engine_zero_generation_budget_finishes_after_prefill() {
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, false);
    let mut engine = ServingEngine::new(model, config).unwrap();

    let id = engine.add_request(vec![1, 2, 3], 0);

    let first = engine.step().unwrap();
    assert_eq!(first.outputs.len(), 1);
    assert_eq!(first.outputs[0].seq_id, id);

    let second = engine.step().unwrap();
    assert!(second.outputs.is_empty(), "zero-budget request must not decode");
    assert_eq!(second.finished.len(), 1);
    assert_eq!(second.finished[0].0, id);
    assert_eq!(second.finished[0].1, FinishReason::MaxTokens);
}

#[test]
fn engine_deterministic_across_runs() {
    // Same model + same tokens → same logits, twice.
    let run = || {
        let model = tiny_model(2);
        let config = serving_config(4, 256, 4, false);
        let mut engine = ServingEngine::new(model, config).unwrap();

        let id = engine.add_request(vec![1, 2, 3], 2);
        let out = engine.step().unwrap();
        out.outputs[0].logits.clone()
    };

    let logits_a = run();
    let logits_b = run();

    assert_eq!(logits_a.len(), logits_b.len());
    for (i, (a, b)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "non-determinism at logit {}: {} vs {}",
            i,
            a,
            b,
        );
    }
}

// ===================================================================
// Prefix Cache ↔ Scheduler Interaction Edge Cases
// ===================================================================

#[test]
fn cancel_waiting_prefix_hit_frees_shared_pages() {
    // Regression: add_request with a prefix cache hit pre-populates shared
    // pages in the waiting queue. Cancelling before step() must dec_ref
    // those shared pages, otherwise repeated add/cancel ratchets refs up.
    let model = tiny_model(2);
    let config = serving_config(4, 256, 4, true);
    let mut engine = ServingEngine::new(model, config).unwrap();

    // First request establishes the cached prefix [1,2,3,4].
    let prompt = vec![1, 2, 3, 4, 5, 6];
    let id1 = engine.add_request(prompt.clone(), 1);
    drive_to_completion(&mut engine, 10);

    let free_after_first = engine.allocator_stats().free_pages;

    // Second request hits the prefix cache — shared pages are inc_ref'd
    // and pushed into the waiting queue's page table.
    let id2 = engine.add_request(prompt.clone(), 1);

    let free_after_add = engine.allocator_stats().free_pages;
    // No new pages allocated (shared), but refs incremented.

    // Cancel BEFORE step() — the waiting-queue cancel path must free them.
    engine.cancel_request(id2).unwrap();

    let free_after_cancel = engine.allocator_stats().free_pages;
    assert_eq!(
        free_after_first, free_after_cancel,
        "cancelling a waiting prefix-hit request must restore ref counts: \
         after_first={}, after_cancel={}",
        free_after_first, free_after_cancel,
    );

    // Repeat 10 times — ref counts must not ratchet.
    for _ in 0..10 {
        let id = engine.add_request(prompt.clone(), 1);
        engine.cancel_request(id).unwrap();
    }

    let free_after_cycle = engine.allocator_stats().free_pages;
    assert_eq!(
        free_after_first, free_after_cycle,
        "repeated add/cancel must not leak refs: after_first={}, after_cycle={}",
        free_after_first, free_after_cycle,
    );
}

#[test]
fn prefix_hit_request_admitted_under_memory_pressure() {
    // Regression: estimate_pages_needed charged the full prompt length even
    // when prefix caching already covered most of it. A mostly-cached prompt
    // was blocked from admission under memory pressure.
    let model = tiny_model(2);
    // Tiny page pool: page_size=4, max_pages=8, 2 layers → 4 pages per layer.
    // A fresh 8-token prompt needs 2 pages/layer * 2 layers = 4 pages.
    let config = serving_config(4, 8, 4, true);
    let mut engine = ServingEngine::new(model, config).unwrap();

    // First request: 8 tokens → uses 4 pages (2 per layer).
    let id1 = engine.add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
    drive_to_completion(&mut engine, 10);
    // After completion, sequence pages freed, but prefix cache holds refs
    // to the first page-group (tokens 1-4). That's 2 pages pinned by cache.

    // Fill most of the remaining pool with another request.
    let id2 = engine.add_request(vec![10, 11, 12, 13], 1);
    drive_to_completion(&mut engine, 10);

    // Now pool is mostly free again. Add a request that hits the cached prefix.
    // The prompt is [1,2,3,4, 20,21,22,23] — first 4 tokens cached.
    // Only the suffix [20,21,22,23] needs new pages → 1 page/layer = 2 pages.
    let id3 = engine.add_request(vec![1, 2, 3, 4, 20, 21, 22, 23], 1);

    // Step should admit id3 (only needs 2 new pages, not 4).
    let out = engine.step().unwrap();
    assert!(
        !out.outputs.is_empty(),
        "prefix-hit request should be admitted — only suffix needs new pages"
    );
}
