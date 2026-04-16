//! GPU attention kernels for the decode path.
//!
//! These three small kernels handle KV cache management and per-head
//! attention score computation on the GPU, complementing the existing
//! matmul, softmax, and RoPE kernels.

use cubecl::prelude::*;

/// Copy a new K or V vector into the cache at the given position.
///
/// `cache` layout: `[max_seq_len, kv_dim]` flattened row-major.
/// `new_kv` is `[kv_dim]` — the single vector to store.
#[cube(launch)]
pub fn cache_append_kernel(
    new_kv: &Array<f32>,
    cache: &mut Array<f32>,
    position: u32,
    kv_dim: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        for i in 0..kv_dim {
            cache[position * kv_dim + i] = new_kv[i];
        }
    }
}

/// Compute attention scores for one query head against the key cache.
///
/// For each position `p` in `0..seq_len`:
///   `scores[p] = dot(q_head, k_cache[p, kv_head, :]) * scale`
///
/// `q_head` is `[head_dim]`.
/// `k_cache` is `[seq_len, kv_dim]` flattened where `kv_dim = num_kv_heads * head_dim`.
/// We read from the `kv_head`-th head within each position's K vector.
#[cube(launch)]
pub fn attention_scores_kernel(
    q_head: &Array<f32>,
    k_cache: &Array<f32>,
    scores: &mut Array<f32>,
    head_dim: u32,
    kv_dim: u32,
    kv_head: u32,
    seq_len: u32,
    scale: f32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        for p in 0..seq_len {
            let mut dot = 0.0f32;
            let k_base = p * kv_dim + kv_head * head_dim;
            for d in 0..head_dim {
                dot += q_head[d] * k_cache[k_base + d];
            }
            scores[p] = dot * scale;
        }
    }
}

/// Contract attention: weighted sum of cached V vectors.
///
/// For each output dimension `d` in `0..head_dim`:
///   `output[d] = sum_p(scores[p] * v_cache[p, kv_head, d])`
///
/// `scores` is `[seq_len]` (after softmax).
/// `v_cache` is `[seq_len, kv_dim]` flattened.
#[cube(launch)]
pub fn attention_contract_kernel(
    scores: &Array<f32>,
    v_cache: &Array<f32>,
    output: &mut Array<f32>,
    head_dim: u32,
    kv_dim: u32,
    kv_head: u32,
    seq_len: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for p in 0..seq_len {
                let v_idx = p * kv_dim + kv_head * head_dim + d;
                sum += scores[p] * v_cache[v_idx];
            }
            output[d] = sum;
        }
    }
}

/// Embedding lookup: copy one row from the embedding table.
///
/// `embedding` is `[vocab_size, hidden_dim]` flattened.
/// Copies row `token_id` into `output[0..hidden_dim]`.
#[cube(launch)]
pub fn embedding_lookup_kernel(
    embedding: &Array<f32>,
    output: &mut Array<f32>,
    token_id: u32,
    hidden_dim: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let base = token_id * hidden_dim;
        for i in 0..hidden_dim {
            output[i] = embedding[base + i];
        }
    }
}

/// Scale a vector in-place: `data[i] *= scale` for `i` in `0..len`.
///
/// Used for Gemma-style embedding scaling by `sqrt(hidden_dim)`.
#[cube(launch)]
pub fn scale_inplace_kernel(data: &mut Array<f32>, scale: f32, len: u32) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        for i in 0..len {
            data[i] = data[i] * scale;
        }
    }
}

/// Copy a slice from `src[src_offset..src_offset+len]` to `dst[dst_offset..dst_offset+len]`.
///
/// Used to assemble per-head attention outputs into the full attention output buffer.
#[cube(launch)]
pub fn copy_slice_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_offset: u32,
    dst_offset: u32,
    len: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        for i in 0..len {
            dst[dst_offset + i] = src[src_offset + i];
        }
    }
}

/// Batched embedding lookup: copy one row per token into a dense batch buffer.
///
/// `embedding` is `[vocab_size, hidden_dim]` flattened.
/// `token_ids` is `[batch_size]`.
/// `output` is `[batch_size, hidden_dim]` flattened row-major.
#[cube(launch)]
pub fn embedding_lookup_batch_kernel(
    embedding: &Array<f32>,
    token_ids: &Array<u32>,
    output: &mut Array<f32>,
    hidden_dim: u32,
) {
    let idx = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let token_idx = idx / hidden_dim;
        let dim = idx % hidden_dim;
        let token_id = token_ids[token_idx];
        output[idx] = embedding[token_id * hidden_dim + dim];
    }
}

/// Batched embedding add: `output[token, dim] += embedding[token_ids[token], dim]`.
#[cube(launch)]
pub fn embedding_add_batch_kernel(
    embedding: &Array<f32>,
    token_ids: &Array<u32>,
    output: &mut Array<f32>,
    hidden_dim: u32,
) {
    let idx = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let token_idx = idx / hidden_dim;
        let dim = idx % hidden_dim;
        let token_id = token_ids[token_idx];
        output[idx] += embedding[token_id * hidden_dim + dim];
    }
}

/// Append a batch of KV rows into the cache starting at `start_position`.
///
/// `source` is `[batch_size, kv_dim]` flattened and `cache` is
/// `[max_seq_len, kv_dim]` flattened row-major.
#[cube(launch)]
pub fn cache_append_batch_kernel(
    source: &Array<f32>,
    cache: &mut Array<f32>,
    start_position: u32,
    kv_dim: u32,
) {
    let idx = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let token_idx = idx / kv_dim;
        let dim = idx % kv_dim;
        let cache_pos = start_position + token_idx;
        cache[cache_pos * kv_dim + dim] = source[idx];
    }
}

/// Compute batched attention scores for a single head.
///
/// `q_batch` is `[batch_size, q_stride]` flattened row-major.
/// `k_cache` is `[seq_len, kv_dim]` flattened row-major.
/// `scores` is `[batch_size, seq_len]` flattened row-major.
#[cube(launch)]
pub fn attention_scores_batch_kernel(
    q_batch: &Array<f32>,
    k_cache: &Array<f32>,
    scores: &mut Array<f32>,
    q_stride: u32,
    head_dim: u32,
    kv_dim: u32,
    head_offset: u32,
    kv_head: u32,
    seq_len: u32,
    cache_pos_before_batch: u32,
    scale: f32,
) {
    let idx = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let token_idx = idx / seq_len;
        let kv_pos = idx % seq_len;
        let causal_horizon = cache_pos_before_batch + token_idx;

        if kv_pos > causal_horizon {
            scores[idx] = -1.0e30f32;
        } else {
            let q_base = token_idx * q_stride + head_offset;
            let k_base = kv_pos * kv_dim + kv_head * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_batch[q_base + d] * k_cache[k_base + d];
            }
            scores[idx] = dot * scale;
        }
    }
}

/// Contract batched attention scores with cached V rows for one head.
///
/// `scores` is `[batch_size, seq_len]`, `v_cache` is `[seq_len, kv_dim]`,
/// and `output` is `[batch_size, q_stride]` flattened row-major.
#[cube(launch)]
pub fn attention_contract_batch_kernel(
    scores: &Array<f32>,
    v_cache: &Array<f32>,
    output: &mut Array<f32>,
    q_stride: u32,
    head_dim: u32,
    kv_dim: u32,
    head_offset: u32,
    kv_head: u32,
    seq_len: u32,
) {
    let idx = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let token_idx = idx / head_dim;
        let dim = idx % head_dim;
        let mut sum = 0.0f32;

        for kv_pos in 0..seq_len {
            let score = scores[token_idx * seq_len + kv_pos];
            if score != 0.0 {
                let v_idx = kv_pos * kv_dim + kv_head * head_dim + dim;
                sum += score * v_cache[v_idx];
            }
        }

        output[token_idx * q_stride + head_offset + dim] = sum;
    }
}
