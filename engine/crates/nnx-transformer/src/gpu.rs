//! GPU model wrapper — bridges GpuInference into NnxBackend's dispatch.
//!
//! This module is compiled only when the `gpu` feature is enabled.  It wraps
//! `GpuInference<WgpuRuntime>` (from `nnx-cubecl`) into the types that
//! `NnxBackend` stores in its enum variants, providing a clean interface
//! boundary that hides the CubeCL generics from the rest of the backend.
//!
//! # Why not call `GpuInference::from_model` directly?
//!
//! `from_model` is gated behind `nnx-cubecl`'s `transformer` feature, which
//! itself depends on `nnx-transformer` — creating a circular dependency.
//! Instead, this module dequantizes the weights locally and calls
//! `GpuInference::from_raw_weights`, which has no dependency on
//! `nnx-transformer`.

use nnx_core::gpu_config::{GpuConfig, GpuPosEncoding};
use nnx_cubecl::{GpuInference, GpuLayerCache, RawLayerWeights, WgpuRuntime};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::config::{ModelConfig, PosEncoding};
use crate::model::Model;
use crate::weights::Matrix;

/// GPU-accelerated model wrapping `GpuInference<WgpuRuntime>`.
///
/// All weights are uploaded to the GPU at construction time.  The inner
/// `GpuInference` is generic over runtime; we fix it to `WgpuRuntime` here
/// so that `NnxBackend` doesn't need to be generic.
pub struct GpuModel {
    pub(crate) inner: GpuInference<WgpuRuntime>,
    #[cfg(test)]
    forward_token_calls: AtomicUsize,
    #[cfg(test)]
    forward_batch_calls: AtomicUsize,
}

/// GPU-resident KV cache for a single request/session.
///
/// Mirrors the logical structure of `KVCache` but with all buffers on GPU.
/// The `position` field tracks how many tokens have been processed; it
/// exactly matches the `len` field on each `GpuLayerCache` entry (they
/// are always kept in sync by `forward_token` and `forward_batch`).
pub struct GpuRequestCache {
    pub(crate) cache: Vec<GpuLayerCache>,
    /// Current number of tokens in the cache (= seq_len for the next decode).
    position: usize,
    /// Max context length — cached here so capacity() doesn't need the model.
    capacity: usize,
    /// kv_dim = num_kv_heads * head_dim, used for memory estimation.
    kv_dim: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dequantize a `Matrix` to a flat `Vec<f32>`.
///
/// Duplicated here from `nnx-cubecl::inference::matrix_to_f32` to avoid
/// needing the `transformer` feature on `nnx-cubecl`.
fn dequantize_matrix(matrix: &Matrix) -> Vec<f32> {
    match matrix {
        Matrix::Dense { data, .. } => data.clone(),
        Matrix::Quantized { rows, cols, .. } => {
            let mut out = vec![0.0f32; rows * cols];
            for row in 0..*rows {
                let start = row * cols;
                let end = start + cols;
                matrix.copy_row_to(row, &mut out[start..end]);
            }
            out
        }
    }
}

/// Convert a transformer `PosEncoding` to the stripped-down `GpuPosEncoding`
/// stored inside `GpuConfig`.
fn to_gpu_pos_encoding(enc: &PosEncoding) -> GpuPosEncoding {
    match enc {
        PosEncoding::RoPE { freq_base } => GpuPosEncoding::RoPE {
            freq_base: *freq_base,
        },
        PosEncoding::Learned => GpuPosEncoding::Learned,
        PosEncoding::PartialRoPE {
            freq_base,
            rotary_dim,
        } => GpuPosEncoding::PartialRoPE {
            freq_base: *freq_base,
            rotary_dim: *rotary_dim,
        },
        PosEncoding::None => GpuPosEncoding::None,
    }
}

/// Build the `GpuConfig` from a transformer `ModelConfig`.
fn build_gpu_config(cfg: &ModelConfig) -> GpuConfig {
    cfg.to_gpu_config()
}

// ---------------------------------------------------------------------------
// GpuModel
// ---------------------------------------------------------------------------

impl GpuModel {
    /// Dequantize CPU model weights and upload them to the GPU.
    ///
    /// Calls `GpuInference::from_raw_weights` (no circular dep) rather than
    /// the `transformer`-gated `GpuInference::from_model`.
    pub fn from_cpu_model(model: &Model) -> Result<Self, String> {
        let cfg = &model.config;
        let gpu_config = build_gpu_config(cfg);

        let token_embedding = dequantize_matrix(&model.weights.token_embedding);
        let position_embedding = model
            .weights
            .position_embedding
            .as_ref()
            .map(dequantize_matrix);
        let lm_head = dequantize_matrix(&model.weights.lm_head);
        let final_norm = model.weights.final_norm.clone();
        let final_norm_bias = model.weights.final_norm_bias.clone();

        let layers: Vec<RawLayerWeights> = model
            .weights
            .layers
            .iter()
            .map(|l| RawLayerWeights {
                attn_norm: l.attn_norm.clone(),
                ffn_norm: l.ffn_norm.clone(),
                wq: dequantize_matrix(&l.wq),
                wk: dequantize_matrix(&l.wk),
                wv: dequantize_matrix(&l.wv),
                wo: dequantize_matrix(&l.wo),
                w_gate: dequantize_matrix(&l.w_gate),
                w_up: dequantize_matrix(&l.w_up),
                w_down: dequantize_matrix(&l.w_down),
                bq: l.bq.clone(),
                bk: l.bk.clone(),
                bv: l.bv.clone(),
                bo: l.bo.clone(),
                attn_norm_bias: l.attn_norm_bias.clone(),
                ffn_norm_bias: l.ffn_norm_bias.clone(),
            })
            .collect();

        let inner = GpuInference::<WgpuRuntime>::from_raw_weights(
            gpu_config,
            &token_embedding,
            position_embedding.as_deref(),
            &lm_head,
            &final_norm,
            final_norm_bias.as_deref(),
            layers,
        )?;

        Ok(Self {
            inner,
            #[cfg(test)]
            forward_token_calls: AtomicUsize::new(0),
            #[cfg(test)]
            forward_batch_calls: AtomicUsize::new(0),
        })
    }

    /// Allocate an empty GPU KV cache for this model.
    pub fn new_cache(&self) -> GpuRequestCache {
        let cache = self.inner.new_cache();
        let cfg = self.inner.config();
        GpuRequestCache {
            position: 0,
            capacity: cfg.max_context_length,
            kv_dim: cfg.num_kv_heads * cfg.head_dim,
            cache,
        }
    }

    /// Run a single decode step, returning logits as `Vec<f32>` on CPU.
    ///
    /// Increments `cache.position` by 1.
    pub fn forward_token(&self, cache: &mut GpuRequestCache, token_id: u32) -> Vec<f32> {
        #[cfg(test)]
        self.forward_token_calls.fetch_add(1, Ordering::Relaxed);
        let logits = self.inner.forward_token(&mut cache.cache, token_id);
        cache.position += 1;
        logits
    }

    /// Run prompt prefill for a batch of tokens, returning logits for the last token.
    pub fn forward_batch(&self, cache: &mut GpuRequestCache, token_ids: &[u32]) -> Vec<f32> {
        #[cfg(test)]
        self.forward_batch_calls.fetch_add(1, Ordering::Relaxed);

        let logits = self.inner.forward_batch(&mut cache.cache, token_ids);
        cache.position += token_ids.len();
        logits
    }

    /// Access the GPU model configuration.
    pub fn config(&self) -> &nnx_core::gpu_config::GpuConfig {
        self.inner.config()
    }

    /// Vocabulary size (convenience accessor for NnxBackend dispatch).
    pub fn vocab_size(&self) -> usize {
        self.inner.config().vocab_size
    }

    #[cfg(test)]
    pub(crate) fn reset_call_counts(&self) {
        self.forward_token_calls.store(0, Ordering::Relaxed);
        self.forward_batch_calls.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn call_counts(&self) -> (usize, usize) {
        (
            self.forward_token_calls.load(Ordering::Relaxed),
            self.forward_batch_calls.load(Ordering::Relaxed),
        )
    }
}

// ---------------------------------------------------------------------------
// GpuRequestCache
// ---------------------------------------------------------------------------

impl GpuRequestCache {
    /// Current number of cached tokens (identical to seq_len for the next decode step).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Maximum number of tokens this cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset the cache to an empty state.
    ///
    /// We reset the `len` field on every layer entry.  The GPU buffers are
    /// NOT zeroed — stale entries beyond `position` are never read by
    /// attention because the attention kernel is given `seq_len = position`.
    pub fn clear(&mut self) {
        for layer in &mut self.cache {
            layer.len = 0;
        }
        self.position = 0;
    }

    /// Discard all cached state beyond position `n`.
    ///
    /// We only update the logical length.  Attention uses `layer_cache.len` as
    /// the effective sequence length, so stale entries at positions >= n are
    /// never accessed.
    pub fn truncate(&mut self, n: usize) {
        let new_pos = n.min(self.position);
        for layer in &mut self.cache {
            layer.len = new_pos;
        }
        self.position = new_pos;
    }

    /// Estimate memory usage from the buffer dimensions.
    ///
    /// Each layer holds two buffers of `max_context_length * kv_dim` f32
    /// values, giving `2 * capacity * kv_dim * 4` bytes per layer.
    pub fn memory_bytes(&self) -> usize {
        let per_layer = 2 * self.capacity * self.kv_dim * std::mem::size_of::<f32>();
        per_layer * self.cache.len()
    }
}
