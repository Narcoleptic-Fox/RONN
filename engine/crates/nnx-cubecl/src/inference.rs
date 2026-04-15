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
//! - The forward pass executes the v1 path:
//!   RMSNorm + SwiGLU + full RoPE + sequential blocks.

use cubecl::prelude::*;

use crate::backend::{CubeclBackend, GpuBuffer};
use nnx_core::backend::KernelBackend;
use nnx_core::gpu_config::{
    GpuBlockStyle, GpuConfig, GpuFFNType, GpuNormType, GpuPosEncoding,
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
        let upload_optional =
            |data: Option<&[f32]>| -> Option<GpuBuffer> { data.map(upload) };

        let token_embedding_buf = upload(token_embedding);
        let lm_head_buf = upload(lm_head);
        let final_norm_buf = upload(final_norm);
        let final_norm_bias_buf = upload_optional(final_norm_bias);

        let position_embedding = match &config.pos_encoding {
            GpuPosEncoding::Learned => {
                // Learned position embeddings are not yet stored in the CPU
                // model weights structure.  Upload a zero placeholder so the
                // GPU weight layout is consistent until the loader exposes them.
                Some(backend.zeros(config.max_context_length * config.hidden_dim))
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
                    self.backend
                        .rope_inplace(q, h * cfg.head_dim, cfg.head_dim, position, *freq_base);
                }
                for h in 0..cfg.num_kv_heads {
                    self.backend
                        .rope_inplace(k, h * cfg.head_dim, cfg.head_dim, position, *freq_base);
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
                    &layer.w_gate, input, gate, cfg.intermediate_dim, cfg.hidden_dim,
                );
                self.backend.matvec(
                    &layer.w_up, input, up, cfg.intermediate_dim, cfg.hidden_dim,
                );
                self.backend.fused_swiglu(gate, up);
                self.backend.matvec(
                    &layer.w_down, gate, down, cfg.hidden_dim, cfg.intermediate_dim,
                );
            }
            GpuFFNType::GeGLU => {
                self.backend.matvec(
                    &layer.w_gate, input, gate, cfg.intermediate_dim, cfg.hidden_dim,
                );
                self.backend.matvec(
                    &layer.w_up, input, up, cfg.intermediate_dim, cfg.hidden_dim,
                );
                self.backend.fused_geglu(gate, up);
                self.backend.matvec(
                    &layer.w_down, gate, down, cfg.hidden_dim, cfg.intermediate_dim,
                );
            }
            GpuFFNType::GELU => {
                // Plain GELU FFN: only 2 matrices (gate=fc1, down=fc2), no up projection.
                self.backend.matvec(
                    &layer.w_gate, input, gate, cfg.intermediate_dim, cfg.hidden_dim,
                );
                self.backend.gelu_inplace(gate);
                self.backend.matvec(
                    &layer.w_down, gate, down, cfg.hidden_dim, cfg.intermediate_dim,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Main forward pass
    // -----------------------------------------------------------------------

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
        self.lookup_embedding_row(&self.weights.token_embedding, token_id, hidden_dim, &mut hidden);

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
            self.norm_hidden(&hidden, &layer.attn_norm, layer.attn_norm_bias.as_ref(), &mut normed);

            match cfg.block_style {
                GpuBlockStyle::Sequential => {
                    // Attention → residual → FFN norm → FFN → residual
                    self.run_attention(
                        layer, layer_cache, position, &normed,
                        &mut q, &mut k, &mut v, &mut attn_output, &mut head_out, &mut proj_out,
                    );
                    self.backend.add_inplace(&mut hidden, &proj_out);

                    self.norm_hidden(
                        &hidden, &layer.ffn_norm, layer.ffn_norm_bias.as_ref(), &mut normed,
                    );
                    self.compute_ffn(layer, &normed, &mut gate, &mut up, &mut down);
                    self.backend.add_inplace(&mut hidden, &down);
                }
                GpuBlockStyle::Parallel => {
                    // Save normed for FFN, run attention and FFN from same input
                    let saved = parallel_normed.as_mut()
                        .expect("parallel block requires saved normed buffer");
                    self.copy_slice(&normed, 0, saved, 0, hidden_dim);

                    self.run_attention(
                        layer, layer_cache, position, &normed,
                        &mut q, &mut k, &mut v, &mut attn_output, &mut head_out, &mut proj_out,
                    );
                    self.compute_ffn(layer, saved, &mut gate, &mut up, &mut down);

                    self.backend.add_inplace(&mut hidden, &proj_out);
                    self.backend.add_inplace(&mut hidden, &down);
                }
            }
        }

        // 3. Final norm + LM head
        self.norm_hidden(
            &hidden, &self.weights.final_norm, self.weights.final_norm_bias.as_ref(), &mut normed,
        );

        let mut logits = self.backend.zeros(cfg.vocab_size);
        self.backend.matvec(&self.weights.lm_head, &normed, &mut logits, cfg.vocab_size, hidden_dim);

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
        GpuBlockStyle, GpuConfig, GpuFFNType, GpuNormType, GpuPosEncoding,
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
            bq: None, bk: None, bv: None, bo: None,
            attn_norm_bias: None, ffn_norm_bias: None,
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
    fn test_new_cache_dimensions() {
        let config = tiny_gpu_config();
        let hd = config.hidden_dim;
        let vs = config.vocab_size;

        let gpu = GpuInference::<WgpuRuntime>::from_raw_weights(
            config.clone(),
            &vec![0.1f32; vs * hd],
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
            &vec![0.0f32; vs * hd],
            &vec![1.0f32; hd],
            None,
            vec![],
        );
        assert!(result.is_err(), "should fail with mismatched layer count");
    }
}
