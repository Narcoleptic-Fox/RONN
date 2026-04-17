//! GPU-accelerated transformer inference via CubeCL.
//!
//! `GpuInference<R>` uploads all model weights to the GPU once and runs the
//! full forward pass entirely on the device.  Weights and the KV cache remain
//! GPU-resident — the only CPU round-trip is downloading the final logits.
//!
//! # Dependency structure
//!
//! The core `GpuInference<R>` struct depends only on `nnx-core` (for
//! [`GpuConfig`]) and on the CubeCL runtime crates.  It does **not** depend
//! on `nnx-transformer` at compile time, which avoids a circular dependency
//! (`nnx-transformer` optionally depends on this crate for its `gpu` feature).
//!
//! To upload an `nnx-transformer::model::Model` to GPU, use
//! `nnx-transformer::gpu::GpuModel::from_cpu_model`, which dequantizes weights
//! and calls `GpuInference::from_raw_weights` directly.
//!
//! # Current Coverage
//!
//! - Weight upload supports all 10 architecture profiles defined by
//!   `nnx-transformer`.
//! - The forward pass executes the v1 path for both decode and prompt prefill:
//!   RMSNorm + SwiGLU + full RoPE + sequential blocks.

use cubecl::prelude::*;

use crate::backend::{CubeclBackend, GpuBuffer};
use crate::paged_kv::{
    GpuPagePool, GpuPagedKvQuantConfig, contiguous_to_paged_page_kernel,
    paged_attention_contract_kernel, paged_attention_scores_kernel, paged_cache_append_kernel,
    paged_quantize_page_kernel, paged_to_contiguous_page_kernel,
};
use nnx_core::PageId;
use nnx_core::backend::KernelBackend;
use nnx_core::gpu_config::{
    GpuActivationQuant, GpuBlockStyle, GpuConfig, GpuFFNType, GpuNormType, GpuPosEncoding,
};

// ---------------------------------------------------------------------------
// GPU weight storage
// ---------------------------------------------------------------------------

/// One transformer layer's weights stored on GPU.
pub struct GpuLayerWeights {
    pub attn_norm: GpuBuffer,
    pub ffn_norm: GpuBuffer,
    pub wq: GpuBuffer,
    pub wk: GpuBuffer,
    pub wv: GpuBuffer,
    pub wo: GpuBuffer,
    pub w_gate: GpuBuffer,
    pub w_up: GpuBuffer,
    pub w_down: GpuBuffer,
    pub bq: Option<GpuBuffer>,
    pub bk: Option<GpuBuffer>,
    pub bv: Option<GpuBuffer>,
    pub bo: Option<GpuBuffer>,
    pub attn_norm_bias: Option<GpuBuffer>,
    pub ffn_norm_bias: Option<GpuBuffer>,
}

/// All model weights stored on GPU.
pub struct GpuModelWeights {
    pub token_embedding: GpuBuffer,
    pub layers: Vec<GpuLayerWeights>,
    pub final_norm: GpuBuffer,
    pub final_norm_bias: Option<GpuBuffer>,
    pub position_embedding: Option<GpuBuffer>,
    pub lm_head: GpuBuffer,
}

// ---------------------------------------------------------------------------
// GPU KV cache
// ---------------------------------------------------------------------------

/// KV cache for one layer stored on GPU.
pub struct GpuLayerCache {
    /// Cached keys `[max_seq_len * kv_dim]` on GPU.
    pub keys: GpuBuffer,
    /// Cached values `[max_seq_len * kv_dim]` on GPU.
    pub values: GpuBuffer,
    /// Current number of positions stored.
    pub len: usize,
}

// ---------------------------------------------------------------------------
// GPU inference engine
// ---------------------------------------------------------------------------

/// GPU-accelerated inference engine.
///
/// Stores all model weights on GPU and runs the full decode step without
/// CPU round-trips (except downloading the final logits). Generic over
/// CubeCL `Runtime` — the caller picks CUDA, ROCm, Metal, Vulkan, or WebGPU.
pub struct GpuInference<R: Runtime> {
    backend: CubeclBackend<R>,
    weights: GpuModelWeights,
    /// Numeric model configuration.  Uses [`GpuConfig`] (from `nnx-core`)
    /// rather than `ModelConfig` (from `nnx-transformer`) so that this crate
    /// does not create a circular dependency.
    config: GpuConfig,
}

// Safety: CubeCL handles are internally synchronized by the compute server.
unsafe impl<R: Runtime> Send for GpuInference<R> {}
unsafe impl<R: Runtime> Sync for GpuInference<R> {}

/// Raw per-layer weight data passed to [`GpuInference::from_raw_weights`].
///
/// All matrices are pre-dequantized to `Vec<f32>` by the caller.
pub struct RawLayerWeights {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub bo: Option<Vec<f32>>,
    pub attn_norm_bias: Option<Vec<f32>>,
    pub ffn_norm_bias: Option<Vec<f32>>,
}

impl<R: Runtime> GpuInference<R> {
    /// Validate whether a GPU config is usable for inference.
    pub fn validate_config(cfg: &GpuConfig) -> Result<(), String> {
        if cfg.num_heads == 0 {
            return Err("num_heads must be non-zero".into());
        }
        if cfg.num_kv_heads == 0 {
            return Err("num_kv_heads must be non-zero".into());
        }
        if cfg.hidden_dim == 0 {
            return Err("hidden_dim must be non-zero".into());
        }
        if cfg.max_context_length == 0 {
            return Err("max_context_length must be non-zero".into());
        }
        if cfg.num_heads % cfg.num_kv_heads != 0 {
            return Err(format!(
                "num_heads {} must be divisible by num_kv_heads {}",
                cfg.num_heads, cfg.num_kv_heads
            ));
        }
        Ok(())
    }

    /// Upload raw pre-dequantized weights to GPU and create an inference engine.
    ///
    /// This constructor does not require the `transformer` feature.  Callers
    /// (such as `nnx-transformer`'s own `gpu` module) dequantize weights
    /// themselves and pass the resulting `Vec<f32>` slices here.
    pub fn from_raw_weights(
        config: GpuConfig,
        token_embedding: &[f32],
        position_embedding: Option<&[f32]>,
        lm_head: &[f32],
        final_norm: &[f32],
        final_norm_bias: Option<&[f32]>,
        layers: Vec<RawLayerWeights>,
    ) -> Result<Self, String> {
        Self::validate_config(&config)?;

        if layers.len() != config.num_layers {
            return Err(format!(
                "got {} layers of weights but config declares {}",
                layers.len(),
                config.num_layers
            ));
        }

        let backend = CubeclBackend::<R>::new();

        let upload = |data: &[f32]| -> GpuBuffer { backend.from_f32(data) };
        let upload_optional = |data: Option<&[f32]>| -> Option<GpuBuffer> { data.map(upload) };

        let token_embedding_buf = upload(token_embedding);
        let lm_head_buf = upload(lm_head);
        let final_norm_buf = upload(final_norm);
        let final_norm_bias_buf = upload_optional(final_norm_bias);

        let position_embedding = match &config.pos_encoding {
            GpuPosEncoding::Learned => {
                let expected = config.max_context_length * config.hidden_dim;
                let data = position_embedding.ok_or_else(|| {
                    format!(
                        "GpuPosEncoding::Learned requires position embeddings with {} values",
                        expected
                    )
                })?;
                if data.len() != expected {
                    return Err(format!(
                        "position embedding has {} values, expected {} (max_context_length {} * hidden_dim {})",
                        data.len(),
                        expected,
                        config.max_context_length,
                        config.hidden_dim
                    ));
                }
                Some(upload(data))
            }
            _ => None,
        };

        let gpu_layers = layers
            .into_iter()
            .map(|l| GpuLayerWeights {
                attn_norm: upload(&l.attn_norm),
                ffn_norm: upload(&l.ffn_norm),
                wq: upload(&l.wq),
                wk: upload(&l.wk),
                wv: upload(&l.wv),
                wo: upload(&l.wo),
                w_gate: upload(&l.w_gate),
                w_up: upload(&l.w_up),
                w_down: upload(&l.w_down),
                bq: upload_optional(l.bq.as_deref()),
                bk: upload_optional(l.bk.as_deref()),
                bv: upload_optional(l.bv.as_deref()),
                bo: upload_optional(l.bo.as_deref()),
                attn_norm_bias: upload_optional(l.attn_norm_bias.as_deref()),
                ffn_norm_bias: upload_optional(l.ffn_norm_bias.as_deref()),
            })
            .collect();

        let gpu_weights = GpuModelWeights {
            token_embedding: token_embedding_buf,
            layers: gpu_layers,
            final_norm: final_norm_buf,
            final_norm_bias: final_norm_bias_buf,
            position_embedding,
            lm_head: lm_head_buf,
        };

        Ok(Self {
            backend,
            weights: gpu_weights,
            config,
        })
    }

    /// Allocate empty KV cache on GPU for all layers.
    pub fn new_cache(&self) -> Vec<GpuLayerCache> {
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let max_seq = self.config.max_context_length;
        let cache_size = max_seq * kv_dim;

        (0..self.config.num_layers)
            .map(|_| GpuLayerCache {
                keys: self.backend.zeros(cache_size),
                values: self.backend.zeros(cache_size),
                len: 0,
            })
            .collect()
    }

    /// Access the GPU model configuration.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Allocate a GPU page pool with the model's KV geometry.
    pub fn new_page_pool(
        &self,
        num_pages: usize,
        page_size: usize,
        quant_config: Option<GpuPagedKvQuantConfig>,
    ) -> GpuPagePool {
        GpuPagePool::new_with_quantization(
            num_pages,
            page_size,
            self.config.num_kv_heads,
            self.config.head_dim,
            quant_config,
            &self.backend,
        )
    }

    pub fn quantize_completed_pages(
        &self,
        pool: &mut GpuPagePool,
        page_tables: &[&[PageId]],
        total_tokens: usize,
    ) {
        let Some(quantized) = pool.quantized() else {
            return;
        };
        let quant_payload = quantized.quant_payload().clone();
        let full_pages = total_tokens / pool.page_size();
        let kv_dim = pool.kv_dim();
        let head_dim = pool.head_dim();
        let num_kv_heads = pool.num_kv_heads();
        let page_size = pool.page_size();
        let words_per_token = kv_dim.div_ceil(4);
        let residual_words_per_token = quantized.config().residual_words_per_token();
        let payload_words_per_page =
            2 * page_size * words_per_token + page_size * num_kv_heads * residual_words_per_token;

        for page_table in page_tables {
            for page_idx in 0..full_pages {
                let Some(&page_id) = page_table.get(page_idx) else {
                    break;
                };
                if pool.is_page_quantized(page_id) {
                    continue;
                }

                let page = pool.page(page_id);
                let payload_base = page_id.0 as usize * payload_words_per_page;

                unsafe {
                    paged_quantize_page_kernel::launch::<R>(
                        self.backend.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &pool.flat_buffer().handle,
                            pool.flat_buffer().len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<u32>(
                            &quant_payload.handle,
                            quant_payload.len,
                            1,
                        ),
                        ScalarArg::new(page.base_offset() as u32),
                        ScalarArg::new(payload_base as u32),
                        ScalarArg::new(page_size as u32),
                        ScalarArg::new(kv_dim as u32),
                        ScalarArg::new(head_dim as u32),
                    );
                }

                pool.set_page_quantized(page_id);
            }
        }
    }

    pub fn mark_pages_dense(&self, pool: &mut GpuPagePool, page_ids: &[PageId]) {
        for &page_id in page_ids {
            pool.set_page_dense(page_id);
        }
    }

    /// Mirror one full logical page from a contiguous cache into paged GPU storage.
    pub fn copy_cache_page_to_pool(
        &self,
        layer_cache: &GpuLayerCache,
        src_page_idx: usize,
        page_size: usize,
        pool: &GpuPagePool,
        dst_page_id: PageId,
    ) {
        assert_eq!(
            pool.kv_dim(),
            self.config.num_kv_heads * self.config.head_dim,
            "GpuPagePool geometry must match the model KV geometry"
        );
        assert_eq!(
            pool.page_size(),
            page_size,
            "GpuPagePool page size must match ServingEngine page size"
        );

        let src_token_offset = src_page_idx * page_size;
        assert!(
            src_token_offset + page_size <= layer_cache.len,
            "cannot mirror incomplete cache page {} (cache len {}, page_size {})",
            src_page_idx,
            layer_cache.len,
            page_size
        );

        let page = pool.page(dst_page_id);
        unsafe {
            contiguous_to_paged_page_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&layer_cache.keys.handle, layer_cache.keys.len, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &layer_cache.values.handle,
                    layer_cache.values.len,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &pool.flat_buffer().handle,
                    pool.flat_buffer().len,
                    1,
                ),
                ScalarArg::new(page.base_offset() as u32),
                ScalarArg::new(src_token_offset as u32),
                ScalarArg::new(page_size as u32),
                ScalarArg::new(pool.kv_dim() as u32),
            );
        }
    }

    /// Hydrate one full logical page from paged GPU storage into a contiguous cache.
    pub fn copy_pool_page_to_cache(
        &self,
        pool: &GpuPagePool,
        src_page_id: PageId,
        layer_cache: &mut GpuLayerCache,
        dst_page_idx: usize,
        page_size: usize,
    ) {
        assert_eq!(
            pool.kv_dim(),
            self.config.num_kv_heads * self.config.head_dim,
            "GpuPagePool geometry must match the model KV geometry"
        );
        assert_eq!(
            pool.page_size(),
            page_size,
            "GpuPagePool page size must match ServingEngine page size"
        );

        let dst_token_offset = dst_page_idx * page_size;
        let capacity = layer_cache.keys.len / pool.kv_dim();
        assert!(
            dst_token_offset + page_size <= capacity,
            "cannot hydrate page {} into cache capacity {} with page_size {}",
            dst_page_idx,
            capacity,
            page_size
        );

        let page = pool.page(src_page_id);
        unsafe {
            paged_to_contiguous_page_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &pool.flat_buffer().handle,
                    pool.flat_buffer().len,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&layer_cache.keys.handle, layer_cache.keys.len, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &layer_cache.values.handle,
                    layer_cache.values.len,
                    1,
                ),
                ScalarArg::new(page.base_offset() as u32),
                ScalarArg::new(dst_token_offset as u32),
                ScalarArg::new(page_size as u32),
                ScalarArg::new(pool.kv_dim() as u32),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Forward pass helpers — architecture-dispatched
    // -----------------------------------------------------------------------

    fn lookup_embedding_row(
        &self,
        embedding: &GpuBuffer,
        row_id: u32,
        row_len: usize,
        output: &mut GpuBuffer,
    ) {
        unsafe {
            crate::attention::embedding_lookup_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&embedding.handle, embedding.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, row_len, 1),
                ScalarArg::new(row_id),
                ScalarArg::new(row_len as u32),
            );
        }
    }

    fn scale_inplace(&self, buf: &mut GpuBuffer, scale: f32) {
        let len = buf.len;
        unsafe {
            crate::attention::scale_inplace_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&buf.handle, len, 1),
                ScalarArg::new(scale),
                ScalarArg::new(len as u32),
            );
        }
    }

    fn copy_slice(
        &self,
        src: &GpuBuffer,
        src_offset: usize,
        dst: &mut GpuBuffer,
        dst_offset: usize,
        count: usize,
    ) {
        unsafe {
            crate::attention::copy_slice_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&src.handle, src.len, 1),
                ArrayArg::from_raw_parts::<f32>(&dst.handle, dst.len, 1),
                ScalarArg::new(src_offset as u32),
                ScalarArg::new(dst_offset as u32),
                ScalarArg::new(count as u32),
            );
        }
    }

    fn lookup_embedding_batch(
        &self,
        embedding: &GpuBuffer,
        token_ids: &GpuBuffer,
        output: &mut GpuBuffer,
        hidden_dim: usize,
    ) {
        let batch_size = token_ids.len;
        unsafe {
            crate::attention::embedding_lookup_batch_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static((batch_size * hidden_dim) as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&embedding.handle, embedding.len, 1),
                ArrayArg::from_raw_parts::<u32>(&token_ids.handle, token_ids.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, output.len, 1),
                ScalarArg::new(hidden_dim as u32),
            );
        }
    }

    fn add_embedding_batch(
        &self,
        embedding: &GpuBuffer,
        token_ids: &GpuBuffer,
        output: &mut GpuBuffer,
        hidden_dim: usize,
    ) {
        let batch_size = token_ids.len;
        unsafe {
            crate::attention::embedding_add_batch_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static((batch_size * hidden_dim) as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&embedding.handle, embedding.len, 1),
                ArrayArg::from_raw_parts::<u32>(&token_ids.handle, token_ids.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, output.len, 1),
                ScalarArg::new(hidden_dim as u32),
            );
        }
    }

    fn append_cache_batch(
        &self,
        source: &GpuBuffer,
        cache_buf: &mut GpuBuffer,
        start_position: usize,
        kv_dim: usize,
        batch_size: usize,
    ) {
        unsafe {
            crate::attention::cache_append_batch_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static((batch_size * kv_dim) as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&source.handle, source.len, 1),
                ArrayArg::from_raw_parts::<f32>(&cache_buf.handle, cache_buf.len, 1),
                ScalarArg::new(start_position as u32),
                ScalarArg::new(kv_dim as u32),
            );
        }
    }

    fn matvec_batch(
        &self,
        matrix: &GpuBuffer,
        vectors: &GpuBuffer,
        output: &mut GpuBuffer,
        m: usize,
        k: usize,
    ) {
        let batch_size = vectors.len / k;
        unsafe {
            crate::matmul::matvec_batch_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static((batch_size * m) as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&matrix.handle, matrix.len, 1),
                ArrayArg::from_raw_parts::<f32>(&vectors.handle, vectors.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, output.len, 1),
                ScalarArg::new(m as u32),
                ScalarArg::new(k as u32),
            );
        }
    }

    fn matvec_batch_bias(
        &self,
        matrix: &GpuBuffer,
        vectors: &GpuBuffer,
        bias: &GpuBuffer,
        output: &mut GpuBuffer,
        m: usize,
        k: usize,
    ) {
        let batch_size = vectors.len / k;
        unsafe {
            crate::matmul::matvec_batch_bias_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static((batch_size * m) as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&matrix.handle, matrix.len, 1),
                ArrayArg::from_raw_parts::<f32>(&vectors.handle, vectors.len, 1),
                ArrayArg::from_raw_parts::<f32>(&bias.handle, bias.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, output.len, 1),
                ScalarArg::new(m as u32),
                ScalarArg::new(k as u32),
            );
        }
    }

    fn append_cache_value(
        &self,
        data: &GpuBuffer,
        cache_buf: &mut GpuBuffer,
        position: usize,
        kv_dim: usize,
    ) {
        unsafe {
            crate::attention::cache_append_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&data.handle, kv_dim, 1),
                ArrayArg::from_raw_parts::<f32>(&cache_buf.handle, cache_buf.len, 1),
                ScalarArg::new(position as u32),
                ScalarArg::new(kv_dim as u32),
            );
        }
    }

    fn append_paged_cache_value(
        &self,
        data: &GpuBuffer,
        pool: &GpuPagePool,
        page_id: PageId,
        offset_in_page: usize,
        value_offset: usize,
    ) {
        let page = pool.page(page_id);
        unsafe {
            paged_cache_append_kernel::launch::<R>(
                self.backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&data.handle, data.len, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &pool.flat_buffer().handle,
                    pool.flat_buffer().len,
                    1,
                ),
                ScalarArg::new(page.base_offset() as u32),
                ScalarArg::new(offset_in_page as u32),
                ScalarArg::new(pool.kv_dim() as u32),
                ScalarArg::new(value_offset as u32),
            );
        }
    }

    fn upload_page_table(&self, pool: &GpuPagePool, page_table: &[PageId]) -> GpuBuffer {
        let page_ids: Vec<u32> = page_table
            .iter()
            .map(|page_id| {
                let mut encoded = page_id.0;
                if pool.is_page_quantized(*page_id) {
                    encoded |= 0x8000_0000;
                }
                encoded
            })
            .collect();
        self.backend.from_u32(&page_ids)
    }

    /// Dispatch normalization based on config.norm_type.
    fn norm_hidden(
        &self,
        hidden: &GpuBuffer,
        weight: &GpuBuffer,
        bias: Option<&GpuBuffer>,
        output: &mut GpuBuffer,
    ) {
        match self.config.norm_type {
            GpuNormType::RMSNorm => {
                self.backend
                    .rms_norm(hidden, weight, output, self.config.rms_norm_eps);
            }
            GpuNormType::LayerNorm => {
                if let Some(bias) = bias {
                    self.backend.layer_norm_bias(
                        hidden,
                        weight,
                        bias,
                        output,
                        self.config.hidden_dim,
                        self.config.rms_norm_eps,
                    );
                } else {
                    self.backend.layer_norm(
                        hidden,
                        weight,
                        output,
                        self.config.hidden_dim,
                        self.config.rms_norm_eps,
                    );
                }
            }
        }
    }

    /// Project Q, K, V — with or without bias.
    fn project_qkv(
        &self,
        layer: &GpuLayerWeights,
        input: &GpuBuffer,
        q: &mut GpuBuffer,
        k: &mut GpuBuffer,
        v: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        if cfg.has_qkv_bias {
            self.backend.matvec_bias(
                &layer.wq,
                input,
                layer.bq.as_ref().expect("missing Q bias"),
                q,
                q_dim,
                hidden_dim,
            );
            self.backend.matvec_bias(
                &layer.wk,
                input,
                layer.bk.as_ref().expect("missing K bias"),
                k,
                kv_dim,
                hidden_dim,
            );
            self.backend.matvec_bias(
                &layer.wv,
                input,
                layer.bv.as_ref().expect("missing V bias"),
                v,
                kv_dim,
                hidden_dim,
            );
        } else {
            self.backend.matvec(&layer.wq, input, q, q_dim, hidden_dim);
            self.backend.matvec(&layer.wk, input, k, kv_dim, hidden_dim);
            self.backend.matvec(&layer.wv, input, v, kv_dim, hidden_dim);
        }
    }

    /// Apply position encoding to Q and K — full RoPE, partial RoPE, or skip.
    fn apply_rope(&self, q: &mut GpuBuffer, k: &mut GpuBuffer, position: usize) {
        let cfg = &self.config;
        match &cfg.pos_encoding {
            GpuPosEncoding::RoPE { freq_base } => {
                for h in 0..cfg.num_heads {
                    self.backend.rope_inplace(
                        q,
                        h * cfg.head_dim,
                        cfg.head_dim,
                        position,
                        *freq_base,
                    );
                }
                for h in 0..cfg.num_kv_heads {
                    self.backend.rope_inplace(
                        k,
                        h * cfg.head_dim,
                        cfg.head_dim,
                        position,
                        *freq_base,
                    );
                }
            }
            GpuPosEncoding::PartialRoPE {
                freq_base,
                rotary_dim,
            } => {
                for h in 0..cfg.num_heads {
                    self.backend.partial_rope_inplace(
                        q,
                        h * cfg.head_dim,
                        cfg.head_dim,
                        *rotary_dim,
                        position,
                        *freq_base,
                    );
                }
                for h in 0..cfg.num_kv_heads {
                    self.backend.partial_rope_inplace(
                        k,
                        h * cfg.head_dim,
                        cfg.head_dim,
                        *rotary_dim,
                        position,
                        *freq_base,
                    );
                }
            }
            GpuPosEncoding::Learned | GpuPosEncoding::None => {
                // Learned positions are added at embedding time; None = skip.
            }
        }
    }

    /// Output projection — with or without bias.
    fn project_attention_output(
        &self,
        layer: &GpuLayerWeights,
        attn_output: &GpuBuffer,
        proj_out: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        if cfg.has_output_bias {
            self.backend.matvec_bias(
                &layer.wo,
                attn_output,
                layer.bo.as_ref().expect("missing output bias"),
                proj_out,
                hidden_dim,
                q_dim,
            );
        } else {
            self.backend
                .matvec(&layer.wo, attn_output, proj_out, hidden_dim, q_dim);
        }
    }

    /// Run full attention for one layer: project QKV, RoPE, cache, score, contract, output proj.
    fn run_attention(
        &self,
        layer: &GpuLayerWeights,
        layer_cache: &mut GpuLayerCache,
        position: usize,
        attn_input: &GpuBuffer,
        q: &mut GpuBuffer,
        k: &mut GpuBuffer,
        v: &mut GpuBuffer,
        attn_output: &mut GpuBuffer,
        head_out: &mut GpuBuffer,
        proj_out: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();

        self.project_qkv(layer, attn_input, q, k, v);
        self.apply_rope(q, k, position);

        self.append_cache_value(k, &mut layer_cache.keys, position, kv_dim);
        self.append_cache_value(v, &mut layer_cache.values, position, kv_dim);

        let seq_len = position + 1;
        layer_cache.len = seq_len;

        let mut scores = self.backend.zeros(seq_len);
        for h in 0..cfg.num_heads {
            let kv_head = h / heads_per_kv;

            self.copy_slice(q, h * cfg.head_dim, head_out, 0, cfg.head_dim);

            // Optional Q8_0 activation quantization for Q
            if cfg.activation_quant == GpuActivationQuant::Q8_0 {
                let (mut q_scales, mut q_quants) = self.backend.alloc_q8_0_buffers(cfg.head_dim);
                self.backend.quantize_f32_to_q8_0(head_out, &mut q_scales, &mut q_quants);
                self.backend.dequantize_q8_0_to_f32(&q_scales, &q_quants, head_out);
            }

            unsafe {
                crate::attention::attention_scores_kernel::launch::<R>(
                    self.backend.client(),
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&head_out.handle, head_out.len, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &layer_cache.keys.handle,
                        layer_cache.keys.len,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                    ScalarArg::new(cfg.head_dim as u32),
                    ScalarArg::new(kv_dim as u32),
                    ScalarArg::new(kv_head as u32),
                    ScalarArg::new(seq_len as u32),
                    ScalarArg::new(scale),
                );
            }

            // Optional Q8_0 activation quantization for attention scores
            if cfg.activation_quant == GpuActivationQuant::Q8_0 {
                let (mut score_scales, mut score_quants) = self.backend.alloc_q8_0_buffers(seq_len);
                self.backend.quantize_f32_to_q8_0(&scores, &mut score_scales, &mut score_quants);
                self.backend.dequantize_q8_0_to_f32(&score_scales, &score_quants, &mut scores);
            }

            self.backend.softmax_inplace(&mut scores, 0, seq_len);

            unsafe {
                crate::attention::attention_contract_kernel::launch::<R>(
                    self.backend.client(),
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &layer_cache.values.handle,
                        layer_cache.values.len,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<f32>(&head_out.handle, head_out.len, 1),
                    ScalarArg::new(cfg.head_dim as u32),
                    ScalarArg::new(kv_dim as u32),
                    ScalarArg::new(kv_head as u32),
                    ScalarArg::new(seq_len as u32),
                );
            }

            self.copy_slice(head_out, 0, attn_output, h * cfg.head_dim, cfg.head_dim);
        }

        self.project_attention_output(layer, attn_output, proj_out);
    }

    fn run_attention_paged(
        &self,
        layer: &GpuLayerWeights,
        pool: &GpuPagePool,
        page_table: &[PageId],
        position: usize,
        attn_input: &GpuBuffer,
        q: &mut GpuBuffer,
        k: &mut GpuBuffer,
        v: &mut GpuBuffer,
        attn_output: &mut GpuBuffer,
        head_out: &mut GpuBuffer,
        proj_out: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let num_tokens = position + 1;
        let required_pages = num_tokens.div_ceil(pool.page_size());

        assert!(
            page_table.len() >= required_pages,
            "paged attention requires {} logical pages, got {}",
            required_pages,
            page_table.len()
        );

        self.project_qkv(layer, attn_input, q, k, v);
        self.apply_rope(q, k, position);

        let page_id = page_table[position / pool.page_size()];
        let offset_in_page = position % pool.page_size();
        self.append_paged_cache_value(k, pool, page_id, offset_in_page, 0);
        self.append_paged_cache_value(v, pool, page_id, offset_in_page, kv_dim);

        let page_table_gpu = self.upload_page_table(pool, page_table);
        let dummy_u32 = self.backend.from_u32(&[0u32]);
        let mut scores = self.backend.zeros(num_tokens);
        let quantized = pool.quantized();
        for h in 0..cfg.num_heads {
            let kv_head = h / heads_per_kv;

            self.copy_slice(q, h * cfg.head_dim, head_out, 0, cfg.head_dim);

            // Optional Q8_0 activation quantization for Q
            if cfg.activation_quant == GpuActivationQuant::Q8_0 {
                let (mut q_scales, mut q_quants) = self.backend.alloc_q8_0_buffers(cfg.head_dim);
                self.backend.quantize_f32_to_q8_0(head_out, &mut q_scales, &mut q_quants);
                self.backend.dequantize_q8_0_to_f32(&q_scales, &q_quants, head_out);
            }

            unsafe {
                paged_attention_scores_kernel::launch::<R>(
                    self.backend.client(),
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&head_out.handle, head_out.len, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &pool.flat_buffer().handle,
                        pool.flat_buffer().len,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(
                        &quantized
                            .map(|storage| storage.quant_payload())
                            .unwrap_or(&dummy_u32)
                            .handle,
                        quantized
                            .map(|storage| storage.quant_payload().len)
                            .unwrap_or(dummy_u32.len),
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&page_table_gpu.handle, page_table_gpu.len, 1),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                    ScalarArg::new(pool.page_stride() as u32),
                    ScalarArg::new(pool.page_size() as u32),
                    ScalarArg::new(num_tokens as u32),
                    ScalarArg::new(kv_head as u32),
                    ScalarArg::new(cfg.head_dim as u32),
                    ScalarArg::new(scale),
                );
            }

            // Optional Q8_0 activation quantization for attention scores
            if cfg.activation_quant == GpuActivationQuant::Q8_0 {
                let (mut score_scales, mut score_quants) = self.backend.alloc_q8_0_buffers(num_tokens);
                self.backend.quantize_f32_to_q8_0(&scores, &mut score_scales, &mut score_quants);
                self.backend.dequantize_q8_0_to_f32(&score_scales, &score_quants, &mut scores);
            }

            self.backend.softmax_inplace(&mut scores, 0, num_tokens);

            unsafe {
                paged_attention_contract_kernel::launch::<R>(
                    self.backend.client(),
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &pool.flat_buffer().handle,
                        pool.flat_buffer().len,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(
                        &quantized
                            .map(|storage| storage.quant_payload())
                            .unwrap_or(&dummy_u32)
                            .handle,
                        quantized
                            .map(|storage| storage.quant_payload().len)
                            .unwrap_or(dummy_u32.len),
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&page_table_gpu.handle, page_table_gpu.len, 1),
                    ArrayArg::from_raw_parts::<f32>(&head_out.handle, head_out.len, 1),
                    ScalarArg::new(pool.page_stride() as u32),
                    ScalarArg::new(pool.page_size() as u32),
                    ScalarArg::new(num_tokens as u32),
                    ScalarArg::new(kv_head as u32),
                    ScalarArg::new(cfg.head_dim as u32),
                );
            }

            self.copy_slice(head_out, 0, attn_output, h * cfg.head_dim, cfg.head_dim);
        }

        self.project_attention_output(layer, attn_output, proj_out);
    }

    fn project_qkv_batch(
        &self,
        layer: &GpuLayerWeights,
        input: &GpuBuffer,
        batch_size: usize,
        q: &mut GpuBuffer,
        k: &mut GpuBuffer,
        v: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        if cfg.has_qkv_bias {
            self.matvec_batch_bias(
                &layer.wq,
                input,
                layer.bq.as_ref().expect("missing Q bias"),
                q,
                q_dim,
                hidden_dim,
            );
            self.matvec_batch_bias(
                &layer.wk,
                input,
                layer.bk.as_ref().expect("missing K bias"),
                k,
                kv_dim,
                hidden_dim,
            );
            self.matvec_batch_bias(
                &layer.wv,
                input,
                layer.bv.as_ref().expect("missing V bias"),
                v,
                kv_dim,
                hidden_dim,
            );
        } else {
            self.matvec_batch(&layer.wq, input, q, q_dim, hidden_dim);
            self.matvec_batch(&layer.wk, input, k, kv_dim, hidden_dim);
            self.matvec_batch(&layer.wv, input, v, kv_dim, hidden_dim);
        }

        debug_assert_eq!(input.len / hidden_dim, batch_size);
    }

    fn apply_rope_batch(
        &self,
        q: &mut GpuBuffer,
        k: &mut GpuBuffer,
        start_position: usize,
        batch_size: usize,
    ) {
        let cfg = &self.config;
        match &cfg.pos_encoding {
            GpuPosEncoding::RoPE { freq_base } => {
                for token_idx in 0..batch_size {
                    let position = start_position + token_idx;
                    let q_base = token_idx * cfg.num_heads * cfg.head_dim;
                    let k_base = token_idx * cfg.num_kv_heads * cfg.head_dim;
                    for h in 0..cfg.num_heads {
                        self.backend.rope_inplace(
                            q,
                            q_base + h * cfg.head_dim,
                            cfg.head_dim,
                            position,
                            *freq_base,
                        );
                    }
                    for h in 0..cfg.num_kv_heads {
                        self.backend.rope_inplace(
                            k,
                            k_base + h * cfg.head_dim,
                            cfg.head_dim,
                            position,
                            *freq_base,
                        );
                    }
                }
            }
            GpuPosEncoding::PartialRoPE {
                freq_base,
                rotary_dim,
            } => {
                for token_idx in 0..batch_size {
                    let position = start_position + token_idx;
                    let q_base = token_idx * cfg.num_heads * cfg.head_dim;
                    let k_base = token_idx * cfg.num_kv_heads * cfg.head_dim;
                    for h in 0..cfg.num_heads {
                        self.backend.partial_rope_inplace(
                            q,
                            q_base + h * cfg.head_dim,
                            cfg.head_dim,
                            *rotary_dim,
                            position,
                            *freq_base,
                        );
                    }
                    for h in 0..cfg.num_kv_heads {
                        self.backend.partial_rope_inplace(
                            k,
                            k_base + h * cfg.head_dim,
                            cfg.head_dim,
                            *rotary_dim,
                            position,
                            *freq_base,
                        );
                    }
                }
            }
            GpuPosEncoding::Learned | GpuPosEncoding::None => {}
        }
    }

    fn project_attention_output_batch(
        &self,
        layer: &GpuLayerWeights,
        attn_output: &GpuBuffer,
        proj_out: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        if cfg.has_output_bias {
            self.matvec_batch_bias(
                &layer.wo,
                attn_output,
                layer.bo.as_ref().expect("missing output bias"),
                proj_out,
                hidden_dim,
                q_dim,
            );
        } else {
            self.matvec_batch(&layer.wo, attn_output, proj_out, hidden_dim, q_dim);
        }
    }

    fn run_attention_batch(
        &self,
        layer: &GpuLayerWeights,
        layer_cache: &mut GpuLayerCache,
        start_position: usize,
        batch_size: usize,
        q: &GpuBuffer,
        k: &mut GpuBuffer,
        v: &mut GpuBuffer,
        attn_output: &mut GpuBuffer,
        scores: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let seq_len = start_position + batch_size;
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();

        self.append_cache_batch(k, &mut layer_cache.keys, start_position, kv_dim, batch_size);
        self.append_cache_batch(
            v,
            &mut layer_cache.values,
            start_position,
            kv_dim,
            batch_size,
        );
        layer_cache.len = seq_len;

        for h in 0..cfg.num_heads {
            let kv_head = h / heads_per_kv;
            let head_offset = h * cfg.head_dim;

            unsafe {
                crate::attention::attention_scores_batch_kernel::launch::<R>(
                    self.backend.client(),
                    CubeCount::Static((batch_size * seq_len) as u32, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&q.handle, q.len, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &layer_cache.keys.handle,
                        layer_cache.keys.len,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                    ScalarArg::new(q_dim as u32),
                    ScalarArg::new(cfg.head_dim as u32),
                    ScalarArg::new(kv_dim as u32),
                    ScalarArg::new(head_offset as u32),
                    ScalarArg::new(kv_head as u32),
                    ScalarArg::new(seq_len as u32),
                    ScalarArg::new(start_position as u32),
                    ScalarArg::new(scale),
                );
            }

            for token_idx in 0..batch_size {
                self.backend
                    .softmax_inplace(scores, token_idx * seq_len, seq_len);
            }

            unsafe {
                crate::attention::attention_contract_batch_kernel::launch::<R>(
                    self.backend.client(),
                    CubeCount::Static((batch_size * cfg.head_dim) as u32, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &layer_cache.values.handle,
                        layer_cache.values.len,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<f32>(&attn_output.handle, attn_output.len, 1),
                    ScalarArg::new(q_dim as u32),
                    ScalarArg::new(cfg.head_dim as u32),
                    ScalarArg::new(kv_dim as u32),
                    ScalarArg::new(head_offset as u32),
                    ScalarArg::new(kv_head as u32),
                    ScalarArg::new(seq_len as u32),
                );
            }
        }

        debug_assert_eq!(layer_cache.len, seq_len);
    }

    fn run_attention_batch_paged(
        &self,
        layer: &GpuLayerWeights,
        pool: &GpuPagePool,
        page_table: &[PageId],
        start_position: usize,
        batch_size: usize,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        attn_output: &mut GpuBuffer,
        scores: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
        let required_pages = (start_position + batch_size).div_ceil(pool.page_size());
        let page_table_gpu = self.upload_page_table(pool, page_table);
        let dummy_u32 = self.backend.from_u32(&[0u32]);
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let quantized = pool.quantized();
        let mut token_q = self.backend.zeros(cfg.head_dim);
        let mut token_k = self.backend.zeros(kv_dim);
        let mut token_v = self.backend.zeros(kv_dim);
        let head_out = self.backend.zeros(cfg.head_dim);

        assert!(
            page_table.len() >= required_pages,
            "paged attention requires {} logical pages, got {}",
            required_pages,
            page_table.len()
        );
        assert!(
            scores.len >= start_position + batch_size,
            "scores scratch buffer too small for paged batch attention"
        );

        for token_idx in 0..batch_size {
            let position = start_position + token_idx;
            let page_id = page_table[position / pool.page_size()];
            let offset_in_page = position % pool.page_size();
            let num_tokens = position + 1;

            self.copy_slice(k, token_idx * kv_dim, &mut token_k, 0, kv_dim);
            self.copy_slice(v, token_idx * kv_dim, &mut token_v, 0, kv_dim);
            self.append_paged_cache_value(&token_k, pool, page_id, offset_in_page, 0);
            self.append_paged_cache_value(&token_v, pool, page_id, offset_in_page, kv_dim);

            for h in 0..cfg.num_heads {
                let kv_head = h / heads_per_kv;
                let head_offset = h * cfg.head_dim;
                let q_offset = token_idx * q_dim + head_offset;

                self.copy_slice(q, q_offset, &mut token_q, 0, cfg.head_dim);

                unsafe {
                    paged_attention_scores_kernel::launch::<R>(
                        self.backend.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&token_q.handle, token_q.len, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &pool.flat_buffer().handle,
                            pool.flat_buffer().len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<u32>(
                            &quantized
                                .map(|storage| storage.quant_payload())
                                .unwrap_or(&dummy_u32)
                                .handle,
                            quantized
                                .map(|storage| storage.quant_payload().len)
                                .unwrap_or(dummy_u32.len),
                            1,
                        ),
                        ArrayArg::from_raw_parts::<u32>(
                            &page_table_gpu.handle,
                            page_table_gpu.len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                        ScalarArg::new(pool.page_stride() as u32),
                        ScalarArg::new(pool.page_size() as u32),
                        ScalarArg::new(num_tokens as u32),
                        ScalarArg::new(kv_head as u32),
                        ScalarArg::new(cfg.head_dim as u32),
                        ScalarArg::new(scale),
                    );
                }

                self.backend.softmax_inplace(scores, 0, num_tokens);

                unsafe {
                    paged_attention_contract_kernel::launch::<R>(
                        self.backend.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&scores.handle, scores.len, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &pool.flat_buffer().handle,
                            pool.flat_buffer().len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<u32>(
                            &quantized
                                .map(|storage| storage.quant_payload())
                                .unwrap_or(&dummy_u32)
                                .handle,
                            quantized
                                .map(|storage| storage.quant_payload().len)
                                .unwrap_or(dummy_u32.len),
                            1,
                        ),
                        ArrayArg::from_raw_parts::<u32>(
                            &page_table_gpu.handle,
                            page_table_gpu.len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(&head_out.handle, head_out.len, 1),
                        ScalarArg::new(pool.page_stride() as u32),
                        ScalarArg::new(pool.page_size() as u32),
                        ScalarArg::new(num_tokens as u32),
                        ScalarArg::new(kv_head as u32),
                        ScalarArg::new(cfg.head_dim as u32),
                    );
                }

                self.copy_slice(&head_out, 0, attn_output, q_offset, cfg.head_dim);
            }
        }
    }

    fn compute_ffn_batch(
        &self,
        layer: &GpuLayerWeights,
        input: &GpuBuffer,
        batch_size: usize,
        gate: &mut GpuBuffer,
        up: &mut GpuBuffer,
        down: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        match cfg.ffn_type {
            GpuFFNType::SwiGLU => {
                self.matvec_batch(
                    &layer.w_gate,
                    input,
                    gate,
                    cfg.intermediate_dim,
                    cfg.hidden_dim,
                );
                self.matvec_batch(&layer.w_up, input, up, cfg.intermediate_dim, cfg.hidden_dim);
                self.backend.fused_swiglu(gate, up);
                self.matvec_batch(
                    &layer.w_down,
                    gate,
                    down,
                    cfg.hidden_dim,
                    cfg.intermediate_dim,
                );
            }
            GpuFFNType::GeGLU => {
                self.matvec_batch(
                    &layer.w_gate,
                    input,
                    gate,
                    cfg.intermediate_dim,
                    cfg.hidden_dim,
                );
                self.matvec_batch(&layer.w_up, input, up, cfg.intermediate_dim, cfg.hidden_dim);
                self.backend.fused_geglu(gate, up);
                self.matvec_batch(
                    &layer.w_down,
                    gate,
                    down,
                    cfg.hidden_dim,
                    cfg.intermediate_dim,
                );
            }
            GpuFFNType::GELU => {
                self.matvec_batch(
                    &layer.w_gate,
                    input,
                    gate,
                    cfg.intermediate_dim,
                    cfg.hidden_dim,
                );
                self.backend.gelu_inplace(gate);
                self.matvec_batch(
                    &layer.w_down,
                    gate,
                    down,
                    cfg.hidden_dim,
                    cfg.intermediate_dim,
                );
            }
        }

        debug_assert_eq!(input.len / cfg.hidden_dim, batch_size);
    }

    /// Dispatch FFN based on config.ffn_type.
    fn compute_ffn(
        &self,
        layer: &GpuLayerWeights,
        input: &GpuBuffer,
        gate: &mut GpuBuffer,
        up: &mut GpuBuffer,
        down: &mut GpuBuffer,
    ) {
        let cfg = &self.config;
        match cfg.ffn_type {
            GpuFFNType::SwiGLU => {
                self.backend.matvec(
                    &layer.w_gate,
                    input,
                    gate,
                    cfg.intermediate_dim,
                    cfg.hidden_dim,
                );
                self.backend
                    .matvec(&layer.w_up, input, up, cfg.intermediate_dim, cfg.hidden_dim);
                self.backend.fused_swiglu(gate, up);
                self.backend.matvec(
                    &layer.w_down,
                    gate,
                    down,
                    cfg.hidden_dim,
                    cfg.intermediate_dim,
                );
            }
            GpuFFNType::GeGLU => {
                self.backend.matvec(
                    &layer.w_gate,
                    input,
                    gate,
                    cfg.intermediate_dim,
                    cfg.hidden_dim,
                );
                self.backend
                    .matvec(&layer.w_up, input, up, cfg.intermediate_dim, cfg.hidden_dim);
                self.backend.fused_geglu(gate, up);
                self.backend.matvec(
                    &layer.w_down,
                    gate,
                    down,
                    cfg.hidden_dim,
                    cfg.intermediate_dim,
                );
            }
            GpuFFNType::GELU => {
                // Plain GELU FFN: only 2 matrices (gate=fc1, down=fc2), no up projection.
                self.backend.matvec(
                    &layer.w_gate,
                    input,
                    gate,
                    cfg.intermediate_dim,
                    cfg.hidden_dim,
                );
                self.backend.gelu_inplace(gate);
                self.backend.matvec(
                    &layer.w_down,
                    gate,
                    down,
                    cfg.hidden_dim,
                    cfg.intermediate_dim,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Main forward pass
    // -----------------------------------------------------------------------

    /// Run a prompt prefill directly from paged GPU KV storage.
    pub fn forward_batch_paged(
        &self,
        pool: &GpuPagePool,
        page_tables: &[&[PageId]],
        current_tokens: usize,
        token_ids: &[u32],
    ) -> Vec<f32> {
        let cfg = &self.config;
        let batch_size = token_ids.len();
        if batch_size == 0 {
            return vec![0.0f32; cfg.vocab_size];
        }
        if batch_size == 1 {
            return self.forward_token_paged(pool, page_tables, current_tokens, token_ids[0]);
        }

        assert_eq!(
            page_tables.len(),
            cfg.num_layers,
            "expected one page table per transformer layer"
        );

        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let start_position = current_tokens;
        let seq_len = start_position + batch_size;

        let token_ids_buf = self.backend.from_u32(token_ids);
        let mut hidden = self.backend.zeros(batch_size * hidden_dim);
        self.lookup_embedding_batch(
            &self.weights.token_embedding,
            &token_ids_buf,
            &mut hidden,
            hidden_dim,
        );

        if matches!(&cfg.pos_encoding, GpuPosEncoding::Learned) {
            if let Some(pos_emb) = &self.weights.position_embedding {
                let positions: Vec<u32> = (0..batch_size)
                    .map(|i| (start_position + i) as u32)
                    .collect();
                let positions_buf = self.backend.from_u32(&positions);
                self.add_embedding_batch(pos_emb, &positions_buf, &mut hidden, hidden_dim);
            }
        }

        if let Some(scale) = cfg.embedding_scale {
            self.scale_inplace(&mut hidden, scale);
        }

        let mut normed = self.backend.zeros(batch_size * hidden_dim);
        let mut saved_normed = matches!(cfg.block_style, GpuBlockStyle::Parallel)
            .then(|| self.backend.zeros(batch_size * hidden_dim));
        let mut q = self.backend.zeros(batch_size * q_dim);
        let mut k = self.backend.zeros(batch_size * kv_dim);
        let mut v = self.backend.zeros(batch_size * kv_dim);
        let mut attn_output = self.backend.zeros(batch_size * q_dim);
        let mut proj_out = self.backend.zeros(batch_size * hidden_dim);
        let mut gate = self.backend.zeros(batch_size * cfg.intermediate_dim);
        let mut up = self.backend.zeros(batch_size * cfg.intermediate_dim);
        let mut down = self.backend.zeros(batch_size * hidden_dim);
        let mut scores = self.backend.zeros(seq_len);

        for layer_idx in 0..cfg.num_layers {
            let layer = &self.weights.layers[layer_idx];

            self.norm_hidden(
                &hidden,
                &layer.attn_norm,
                layer.attn_norm_bias.as_ref(),
                &mut normed,
            );

            self.project_qkv_batch(layer, &normed, batch_size, &mut q, &mut k, &mut v);
            self.apply_rope_batch(&mut q, &mut k, start_position, batch_size);

            match cfg.block_style {
                GpuBlockStyle::Sequential => {
                    self.run_attention_batch_paged(
                        layer,
                        pool,
                        page_tables[layer_idx],
                        start_position,
                        batch_size,
                        &q,
                        &k,
                        &v,
                        &mut attn_output,
                        &mut scores,
                    );
                    self.project_attention_output_batch(layer, &attn_output, &mut proj_out);
                    self.backend.add_inplace(&mut hidden, &proj_out);

                    self.norm_hidden(
                        &hidden,
                        &layer.ffn_norm,
                        layer.ffn_norm_bias.as_ref(),
                        &mut normed,
                    );
                    self.compute_ffn_batch(
                        layer, &normed, batch_size, &mut gate, &mut up, &mut down,
                    );
                    self.backend.add_inplace(&mut hidden, &down);
                }
                GpuBlockStyle::Parallel => {
                    let saved = saved_normed
                        .as_mut()
                        .expect("parallel block requires saved normed buffer");
                    self.copy_slice(&normed, 0, saved, 0, normed.len);

                    self.run_attention_batch_paged(
                        layer,
                        pool,
                        page_tables[layer_idx],
                        start_position,
                        batch_size,
                        &q,
                        &k,
                        &v,
                        &mut attn_output,
                        &mut scores,
                    );
                    self.compute_ffn_batch(layer, saved, batch_size, &mut gate, &mut up, &mut down);

                    self.project_attention_output_batch(layer, &attn_output, &mut proj_out);
                    self.backend.add_inplace(&mut hidden, &proj_out);
                    self.backend.add_inplace(&mut hidden, &down);
                }
            }
        }

        self.norm_hidden(
            &hidden,
            &self.weights.final_norm,
            self.weights.final_norm_bias.as_ref(),
            &mut normed,
        );

        let mut last_hidden = self.backend.zeros(hidden_dim);
        self.copy_slice(
            &normed,
            normed.len - hidden_dim,
            &mut last_hidden,
            0,
            hidden_dim,
        );

        let mut logits = self.backend.zeros(cfg.vocab_size);
        self.backend.matvec(
            &self.weights.lm_head,
            &last_hidden,
            &mut logits,
            cfg.vocab_size,
            hidden_dim,
        );

        self.backend.to_f32(&logits)
    }

    /// Run a full prompt prefill on GPU.
    ///
    /// Processes all prompt tokens in one batched pass: embeddings, batched
    /// block execution, and causal prompt attention. Returns logits for the
    /// final token, matching the CPU `forward_batch` contract.
    pub fn forward_batch(&self, cache: &mut [GpuLayerCache], token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.config;
        let batch_size = token_ids.len();
        if batch_size == 0 {
            return vec![0.0f32; cfg.vocab_size];
        }
        if batch_size == 1 {
            return self.forward_token(cache, token_ids[0]);
        }

        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let start_position = cache.first().map(|c| c.len).unwrap_or(0);
        let seq_len = start_position + batch_size;

        let token_ids_buf = self.backend.from_u32(token_ids);
        let mut hidden = self.backend.zeros(batch_size * hidden_dim);
        self.lookup_embedding_batch(
            &self.weights.token_embedding,
            &token_ids_buf,
            &mut hidden,
            hidden_dim,
        );

        if matches!(&cfg.pos_encoding, GpuPosEncoding::Learned) {
            if let Some(pos_emb) = &self.weights.position_embedding {
                let positions: Vec<u32> = (0..batch_size)
                    .map(|i| (start_position + i) as u32)
                    .collect();
                let positions_buf = self.backend.from_u32(&positions);
                self.add_embedding_batch(pos_emb, &positions_buf, &mut hidden, hidden_dim);
            }
        }

        if let Some(scale) = cfg.embedding_scale {
            self.scale_inplace(&mut hidden, scale);
        }

        let mut normed = self.backend.zeros(batch_size * hidden_dim);
        let mut saved_normed = matches!(cfg.block_style, GpuBlockStyle::Parallel)
            .then(|| self.backend.zeros(batch_size * hidden_dim));
        let mut q = self.backend.zeros(batch_size * q_dim);
        let mut k = self.backend.zeros(batch_size * kv_dim);
        let mut v = self.backend.zeros(batch_size * kv_dim);
        let mut attn_output = self.backend.zeros(batch_size * q_dim);
        let mut proj_out = self.backend.zeros(batch_size * hidden_dim);
        let mut gate = self.backend.zeros(batch_size * cfg.intermediate_dim);
        let mut up = self.backend.zeros(batch_size * cfg.intermediate_dim);
        let mut down = self.backend.zeros(batch_size * hidden_dim);
        let mut scores = self.backend.zeros(batch_size * seq_len);

        for layer_idx in 0..cfg.num_layers {
            let layer = &self.weights.layers[layer_idx];
            let layer_cache = &mut cache[layer_idx];

            self.norm_hidden(
                &hidden,
                &layer.attn_norm,
                layer.attn_norm_bias.as_ref(),
                &mut normed,
            );

            self.project_qkv_batch(layer, &normed, batch_size, &mut q, &mut k, &mut v);
            self.apply_rope_batch(&mut q, &mut k, start_position, batch_size);

            match cfg.block_style {
                GpuBlockStyle::Sequential => {
                    self.run_attention_batch(
                        layer,
                        layer_cache,
                        start_position,
                        batch_size,
                        &q,
                        &mut k,
                        &mut v,
                        &mut attn_output,
                        &mut scores,
                    );
                    self.project_attention_output_batch(layer, &attn_output, &mut proj_out);
                    self.backend.add_inplace(&mut hidden, &proj_out);

                    self.norm_hidden(
                        &hidden,
                        &layer.ffn_norm,
                        layer.ffn_norm_bias.as_ref(),
                        &mut normed,
                    );
                    self.compute_ffn_batch(
                        layer, &normed, batch_size, &mut gate, &mut up, &mut down,
                    );
                    self.backend.add_inplace(&mut hidden, &down);
                }
                GpuBlockStyle::Parallel => {
                    let saved = saved_normed
                        .as_mut()
                        .expect("parallel block requires saved normed buffer");
                    self.copy_slice(&normed, 0, saved, 0, normed.len);

                    self.run_attention_batch(
                        layer,
                        layer_cache,
                        start_position,
                        batch_size,
                        &q,
                        &mut k,
                        &mut v,
                        &mut attn_output,
                        &mut scores,
                    );
                    self.compute_ffn_batch(layer, saved, batch_size, &mut gate, &mut up, &mut down);

                    self.project_attention_output_batch(layer, &attn_output, &mut proj_out);
                    self.backend.add_inplace(&mut hidden, &proj_out);
                    self.backend.add_inplace(&mut hidden, &down);
                }
            }
        }

        self.norm_hidden(
            &hidden,
            &self.weights.final_norm,
            self.weights.final_norm_bias.as_ref(),
            &mut normed,
        );

        let mut last_hidden = self.backend.zeros(hidden_dim);
        self.copy_slice(
            &normed,
            normed.len - hidden_dim,
            &mut last_hidden,
            0,
            hidden_dim,
        );

        let mut logits = self.backend.zeros(cfg.vocab_size);
        self.backend.matvec(
            &self.weights.lm_head,
            &last_hidden,
            &mut logits,
            cfg.vocab_size,
            hidden_dim,
        );

        // Keep the cache lengths in sync with the number of prefetched tokens.
        for layer_cache in cache.iter_mut() {
            layer_cache.len = seq_len;
        }

        self.backend.to_f32(&logits)
    }

    /// Run a single decode step directly from paged GPU KV storage.
    pub fn forward_token_paged(
        &self,
        pool: &GpuPagePool,
        page_tables: &[&[PageId]],
        current_tokens: usize,
        token_id: u32,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        assert_eq!(
            page_tables.len(),
            cfg.num_layers,
            "expected one page table per transformer layer"
        );

        let mut hidden = self.backend.zeros(hidden_dim);
        self.lookup_embedding_row(
            &self.weights.token_embedding,
            token_id,
            hidden_dim,
            &mut hidden,
        );

        if matches!(&cfg.pos_encoding, GpuPosEncoding::Learned) {
            if let Some(pos_emb) = &self.weights.position_embedding {
                let mut pos_buf = self.backend.zeros(hidden_dim);
                self.lookup_embedding_row(pos_emb, current_tokens as u32, hidden_dim, &mut pos_buf);
                self.backend.add_inplace(&mut hidden, &pos_buf);
            }
        }

        if let Some(scale) = cfg.embedding_scale {
            self.scale_inplace(&mut hidden, scale);
        }

        let mut normed = self.backend.zeros(hidden_dim);
        let mut parallel_normed = matches!(cfg.block_style, GpuBlockStyle::Parallel)
            .then(|| self.backend.zeros(hidden_dim));
        let mut q = self.backend.zeros(q_dim);
        let mut k = self.backend.zeros(kv_dim);
        let mut v = self.backend.zeros(kv_dim);
        let mut attn_output = self.backend.zeros(q_dim);
        let mut head_out = self.backend.zeros(cfg.head_dim);
        let mut proj_out = self.backend.zeros(hidden_dim);
        let mut gate = self.backend.zeros(cfg.intermediate_dim);
        let mut up = self.backend.zeros(cfg.intermediate_dim);
        let mut down = self.backend.zeros(hidden_dim);

        for layer_idx in 0..cfg.num_layers {
            let layer = &self.weights.layers[layer_idx];

            self.norm_hidden(
                &hidden,
                &layer.attn_norm,
                layer.attn_norm_bias.as_ref(),
                &mut normed,
            );

            match cfg.block_style {
                GpuBlockStyle::Sequential => {
                    self.run_attention_paged(
                        layer,
                        pool,
                        page_tables[layer_idx],
                        current_tokens,
                        &normed,
                        &mut q,
                        &mut k,
                        &mut v,
                        &mut attn_output,
                        &mut head_out,
                        &mut proj_out,
                    );
                    self.backend.add_inplace(&mut hidden, &proj_out);

                    self.norm_hidden(
                        &hidden,
                        &layer.ffn_norm,
                        layer.ffn_norm_bias.as_ref(),
                        &mut normed,
                    );
                    self.compute_ffn(layer, &normed, &mut gate, &mut up, &mut down);
                    self.backend.add_inplace(&mut hidden, &down);
                }
                GpuBlockStyle::Parallel => {
                    let saved = parallel_normed
                        .as_mut()
                        .expect("parallel block requires saved normed buffer");
                    self.copy_slice(&normed, 0, saved, 0, hidden_dim);

                    self.run_attention_paged(
                        layer,
                        pool,
                        page_tables[layer_idx],
                        current_tokens,
                        &normed,
                        &mut q,
                        &mut k,
                        &mut v,
                        &mut attn_output,
                        &mut head_out,
                        &mut proj_out,
                    );
                    self.compute_ffn(layer, saved, &mut gate, &mut up, &mut down);

                    self.backend.add_inplace(&mut hidden, &proj_out);
                    self.backend.add_inplace(&mut hidden, &down);
                }
            }
        }

        self.norm_hidden(
            &hidden,
            &self.weights.final_norm,
            self.weights.final_norm_bias.as_ref(),
            &mut normed,
        );

        let mut logits = self.backend.zeros(cfg.vocab_size);
        self.backend.matvec(
            &self.weights.lm_head,
            &normed,
            &mut logits,
            cfg.vocab_size,
            hidden_dim,
        );

        self.backend.to_f32(&logits)
    }

    /// Run a full decode step for a single token on GPU.
    ///
    /// Dispatches on `norm_type`, `ffn_type`, `pos_encoding`, `block_style`,
    /// and bias flags to support all 10 architecture profiles.
    ///
    /// Returns logits as a CPU `Vec<f32>` of length `vocab_size`.
    pub fn forward_token(&self, cache: &mut [GpuLayerCache], token_id: u32) -> Vec<f32> {
        let cfg = &self.config;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        // 1. Embedding lookup
        let mut hidden = self.backend.zeros(hidden_dim);
        self.lookup_embedding_row(
            &self.weights.token_embedding,
            token_id,
            hidden_dim,
            &mut hidden,
        );

        // 1b. Learned position embedding (GPT-2)
        if matches!(&cfg.pos_encoding, GpuPosEncoding::Learned) {
            let position = cache.first().map(|c| c.len).unwrap_or(0);
            if let Some(pos_emb) = &self.weights.position_embedding {
                let mut pos_buf = self.backend.zeros(hidden_dim);
                self.lookup_embedding_row(pos_emb, position as u32, hidden_dim, &mut pos_buf);
                self.backend.add_inplace(&mut hidden, &pos_buf);
            }
        }

        // 1c. Embedding scale (Gemma)
        if let Some(scale) = cfg.embedding_scale {
            self.scale_inplace(&mut hidden, scale);
        }

        // Scratch buffers
        let mut normed = self.backend.zeros(hidden_dim);
        let mut parallel_normed = matches!(cfg.block_style, GpuBlockStyle::Parallel)
            .then(|| self.backend.zeros(hidden_dim));
        let mut q = self.backend.zeros(q_dim);
        let mut k = self.backend.zeros(kv_dim);
        let mut v = self.backend.zeros(kv_dim);
        let mut attn_output = self.backend.zeros(q_dim);
        let mut head_out = self.backend.zeros(cfg.head_dim);
        let mut proj_out = self.backend.zeros(hidden_dim);
        let mut gate = self.backend.zeros(cfg.intermediate_dim);
        let mut up = self.backend.zeros(cfg.intermediate_dim);
        let mut down = self.backend.zeros(hidden_dim);

        // 2. Transformer blocks
        for layer_idx in 0..cfg.num_layers {
            let layer = &self.weights.layers[layer_idx];
            let layer_cache = &mut cache[layer_idx];
            let position = layer_cache.len;

            // Pre-attention norm
            self.norm_hidden(
                &hidden,
                &layer.attn_norm,
                layer.attn_norm_bias.as_ref(),
                &mut normed,
            );

            match cfg.block_style {
                GpuBlockStyle::Sequential => {
                    // Attention → residual → FFN norm → FFN → residual
                    self.run_attention(
                        layer,
                        layer_cache,
                        position,
                        &normed,
                        &mut q,
                        &mut k,
                        &mut v,
                        &mut attn_output,
                        &mut head_out,
                        &mut proj_out,
                    );
                    self.backend.add_inplace(&mut hidden, &proj_out);

                    self.norm_hidden(
                        &hidden,
                        &layer.ffn_norm,
                        layer.ffn_norm_bias.as_ref(),
                        &mut normed,
                    );
                    self.compute_ffn(layer, &normed, &mut gate, &mut up, &mut down);
                    self.backend.add_inplace(&mut hidden, &down);
                }
                GpuBlockStyle::Parallel => {
                    // Save normed for FFN, run attention and FFN from same input
                    let saved = parallel_normed
                        .as_mut()
                        .expect("parallel block requires saved normed buffer");
                    self.copy_slice(&normed, 0, saved, 0, hidden_dim);

                    self.run_attention(
                        layer,
                        layer_cache,
                        position,
                        &normed,
                        &mut q,
                        &mut k,
                        &mut v,
                        &mut attn_output,
                        &mut head_out,
                        &mut proj_out,
                    );
                    self.compute_ffn(layer, saved, &mut gate, &mut up, &mut down);

                    self.backend.add_inplace(&mut hidden, &proj_out);
                    self.backend.add_inplace(&mut hidden, &down);
                }
            }
        }

        // 3. Final norm + LM head
        self.norm_hidden(
            &hidden,
            &self.weights.final_norm,
            self.weights.final_norm_bias.as_ref(),
            &mut normed,
        );

        let mut logits = self.backend.zeros(cfg.vocab_size);
        self.backend.matvec(
            &self.weights.lm_head,
            &normed,
            &mut logits,
            cfg.vocab_size,
            hidden_dim,
        );

        self.backend.to_f32(&logits)
    }
}

// =========================================================================
// Tests
// =========================================================================
//
// These tests use only `from_raw_weights` and do not depend on
// `nnx-transformer`, keeping this crate free of circular dependencies.
//
// Tests that exercise GPU vs CPU numerical parity live in
// `nnx-cubecl/tests/gpu_vs_cpu_test.rs` and require the `wgpu` feature.

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    use cubecl::wgpu::WgpuRuntime;
    use nnx_core::gpu_config::{
        GpuActivationQuant, GpuBlockStyle, GpuConfig, GpuFFNType, GpuNormType, GpuPosEncoding,
    };

    fn tiny_gpu_config() -> GpuConfig {
        GpuConfig {
            num_layers: 1,
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
            activation_quant: GpuActivationQuant::None,
        }
    }

    fn tiny_raw_layer() -> RawLayerWeights {
        let hd = 8usize;
        let q_dim = 2 * 4; // num_heads * head_dim
        let kv_dim = 2 * 4; // num_kv_heads * head_dim
        let ffn = 16usize;
        RawLayerWeights {
            attn_norm: vec![1.0f32; hd],
            ffn_norm: vec![1.0f32; hd],
            wq: vec![0.01f32; q_dim * hd],
            wk: vec![0.01f32; kv_dim * hd],
            wv: vec![0.01f32; kv_dim * hd],
            wo: vec![0.01f32; hd * q_dim],
            w_gate: vec![0.01f32; ffn * hd],
            w_up: vec![0.01f32; ffn * hd],
            w_down: vec![0.01f32; hd * ffn],
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            attn_norm_bias: None,
            ffn_norm_bias: None,
        }
    }

    #[test]
    fn test_from_raw_weights_creates_valid_engine() {
        let config = tiny_gpu_config();
        let hd = config.hidden_dim;
        let vs = config.vocab_size;

        let result = GpuInference::<WgpuRuntime>::from_raw_weights(
            config,
            &vec![0.1f32; vs * hd],
            None,
            &vec![0.01f32; vs * hd],
            &vec![1.0f32; hd],
            None,
            vec![tiny_raw_layer()],
        );
        assert!(result.is_ok(), "from_raw_weights should succeed");
    }

    #[test]
    fn test_forward_token_returns_correct_logit_shape() {
        let config = tiny_gpu_config();
        let hd = config.hidden_dim;
        let vs = config.vocab_size;

        let gpu = GpuInference::<WgpuRuntime>::from_raw_weights(
            config,
            &vec![0.1f32; vs * hd],
            None,
            &vec![0.01f32; vs * hd],
            &vec![1.0f32; hd],
            None,
            vec![tiny_raw_layer()],
        )
        .expect("upload should succeed");

        let mut cache = gpu.new_cache();
        let logits = gpu.forward_token(&mut cache, 1u32);

        assert_eq!(logits.len(), vs, "logits length must equal vocab_size");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite"
        );
        assert_eq!(cache[0].len, 1, "cache position must advance to 1");
    }

    #[test]
    fn test_forward_batch_matches_sequential_prefill_with_learned_positions() {
        let mut config = tiny_gpu_config();
        config.pos_encoding = GpuPosEncoding::Learned;
        let hd = config.hidden_dim;
        let vs = config.vocab_size;

        let position_embedding: Vec<f32> = (0..config.max_context_length * hd)
            .map(|i| ((i % 29) as f32) * 0.001 - 0.014)
            .collect();

        let gpu = GpuInference::<WgpuRuntime>::from_raw_weights(
            config,
            &vec![0.1f32; vs * hd],
            Some(&position_embedding),
            &vec![0.01f32; vs * hd],
            &vec![1.0f32; hd],
            None,
            vec![tiny_raw_layer()],
        )
        .expect("upload should succeed");

        let tokens = [1u32, 5, 10];

        let mut batch_cache = gpu.new_cache();
        let batch_logits = gpu.forward_batch(&mut batch_cache, &tokens);

        let mut sequential_cache = gpu.new_cache();
        let mut sequential_logits = Vec::new();
        for &token in &tokens {
            sequential_logits = gpu.forward_token(&mut sequential_cache, token);
        }

        assert_eq!(batch_logits.len(), vs);
        assert!(batch_logits.iter().all(|v| v.is_finite()));
        assert_eq!(batch_cache[0].len, tokens.len());

        let max_diff = batch_logits
            .iter()
            .zip(sequential_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-1,
            "batched prefill differs from sequential decode by {max_diff}"
        );
    }

    #[test]
    fn test_new_cache_dimensions() {
        let config = tiny_gpu_config();
        let hd = config.hidden_dim;
        let vs = config.vocab_size;

        let gpu = GpuInference::<WgpuRuntime>::from_raw_weights(
            config.clone(),
            &vec![0.1f32; vs * hd],
            None,
            &vec![0.01f32; vs * hd],
            &vec![1.0f32; hd],
            None,
            vec![tiny_raw_layer()],
        )
        .expect("upload should succeed");

        let cache = gpu.new_cache();
        assert_eq!(cache.len(), config.num_layers, "one cache entry per layer");
        let kv_dim = config.num_kv_heads * config.head_dim;
        let expected_size = config.max_context_length * kv_dim;
        for layer_cache in &cache {
            assert_eq!(layer_cache.keys.len, expected_size);
            assert_eq!(layer_cache.values.len, expected_size);
            assert_eq!(layer_cache.len, 0, "cache starts empty");
        }
    }

    #[test]
    fn test_from_raw_weights_rejects_wrong_layer_count() {
        let config = tiny_gpu_config();
        let hd = config.hidden_dim;
        let vs = config.vocab_size;
        // config.num_layers = 1, but we pass 0 layers
        let result = GpuInference::<WgpuRuntime>::from_raw_weights(
            config,
            &vec![0.0f32; vs * hd],
            None,
            &vec![0.0f32; vs * hd],
            &vec![1.0f32; hd],
            None,
            vec![],
        );
        assert!(result.is_err(), "should fail with mismatched layer count");
    }

    #[test]
    fn test_from_raw_weights_rejects_missing_learned_position_embeddings() {
        let mut config = tiny_gpu_config();
        config.pos_encoding = GpuPosEncoding::Learned;
        let hd = config.hidden_dim;
        let vs = config.vocab_size;

        let result = GpuInference::<WgpuRuntime>::from_raw_weights(
            config,
            &vec![0.1f32; vs * hd],
            None,
            &vec![0.01f32; vs * hd],
            &vec![1.0f32; hd],
            None,
            vec![tiny_raw_layer()],
        );
        assert!(
            result.is_err(),
            "learned-position configs must reject missing position embeddings"
        );
    }
}
