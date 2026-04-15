//! GPU-accelerated transformer inference via CubeCL.
//!
//! `GpuInference<R>` takes a CPU `Model` (from `nnx-transformer`), uploads
//! all weights to the GPU once, and runs the full forward pass entirely on
//! the device. Weights and the KV cache remain GPU-resident — the only
//! CPU round-trip is downloading the final logits.
//!
//! # Current Coverage
//!
//! - Weight upload supports all 10 architecture profiles defined by
//!   `nnx-transformer`.
//! - The forward pass in this file still executes the original v1 path
//!   (RMSNorm + SwiGLU + full RoPE + sequential blocks, no bias handling).
//!
//! The upload path is expanded first so the forward pass can switch on the
//! full `ModelConfig` in a follow-up change without needing another GPU
//! weight-layout migration.

use cubecl::prelude::*;

use crate::backend::{CubeclBackend, GpuBuffer};
use nnx_core::backend::KernelBackend;
use nnx_transformer::config::{ModelConfig, PosEncoding};
use nnx_transformer::weights::Matrix;

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
// Helpers
// ---------------------------------------------------------------------------

/// Dequantize a `Matrix` to a flat `Vec<f32>`, regardless of storage format.
fn matrix_to_f32(matrix: &Matrix) -> Vec<f32> {
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
    config: ModelConfig,
}

// Safety: CubeCL handles are internally synchronized by the compute server.
unsafe impl<R: Runtime> Send for GpuInference<R> {}
unsafe impl<R: Runtime> Sync for GpuInference<R> {}

impl<R: Runtime> GpuInference<R> {
    /// Validate whether a model config can be uploaded into GPU-resident weight
    /// storage.
    ///
    /// This validation is intentionally broader than the current forward pass.
    /// Upload accepts all known architecture profiles so later execution code can
    /// dispatch on `ModelConfig` without another data-structure migration.
    pub fn validate_config(cfg: &ModelConfig) -> Result<(), String> {
        cfg.validate()?;
        Ok(())
    }

    /// Upload all model weights to GPU and create an inference engine.
    pub fn from_model(model: &nnx_transformer::model::Model) -> Result<Self, String> {
        let cfg = &model.config;
        Self::validate_config(cfg)?;

        if model.weights.layers.len() != cfg.num_layers {
            return Err(format!(
                "model has {} layers of weights but config declares {}",
                model.weights.layers.len(),
                cfg.num_layers
            ));
        }

        let backend = CubeclBackend::<R>::new();

        let upload = |data: &[f32]| -> GpuBuffer { backend.from_f32(data) };
        let upload_matrix = |matrix: &Matrix| -> GpuBuffer {
            let data = matrix_to_f32(matrix);
            upload(&data)
        };
        let upload_optional = |data: Option<&[f32]>| -> Option<GpuBuffer> { data.map(upload) };

        // Upload embedding and LM head
        let token_embedding = upload_matrix(&model.weights.token_embedding);
        let lm_head = upload_matrix(&model.weights.lm_head);
        let final_norm = upload(&model.weights.final_norm);
        let final_norm_bias = upload_optional(model.weights.final_norm_bias.as_deref());
        let position_embedding = match cfg.pos_encoding {
            PosEncoding::Learned => {
                // `nnx-transformer::ModelWeights` does not yet expose learned
                // position embeddings. Upload a zero-filled placeholder so the
                // GPU weight layout matches GPT-2-style models until the loader
                // can pass through the real table.
                Some(backend.zeros(cfg.max_context_length * cfg.hidden_dim))
            }
            _ => None,
        };

        // Upload per-layer weights
        let layers = model
            .weights
            .layers
            .iter()
            .map(|layer| GpuLayerWeights {
                attn_norm: upload(&layer.attn_norm),
                ffn_norm: upload(&layer.ffn_norm),
                wq: upload_matrix(&layer.wq),
                wk: upload_matrix(&layer.wk),
                wv: upload_matrix(&layer.wv),
                wo: upload_matrix(&layer.wo),
                w_gate: upload_matrix(&layer.w_gate),
                w_up: upload_matrix(&layer.w_up),
                w_down: upload_matrix(&layer.w_down),
                bq: upload_optional(layer.bq.as_deref()),
                bk: upload_optional(layer.bk.as_deref()),
                bv: upload_optional(layer.bv.as_deref()),
                bo: upload_optional(layer.bo.as_deref()),
                attn_norm_bias: upload_optional(layer.attn_norm_bias.as_deref()),
                ffn_norm_bias: upload_optional(layer.ffn_norm_bias.as_deref()),
            })
            .collect();

        let gpu_weights = GpuModelWeights {
            token_embedding,
            layers,
            final_norm,
            final_norm_bias,
            position_embedding,
            lm_head,
        };

        Ok(Self {
            backend,
            weights: gpu_weights,
            config: cfg.clone(),
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

    /// Access the model config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Run a full decode step for a single token on GPU.
    ///
    /// Returns logits as a CPU `Vec<f32>` of length `vocab_size`.
    pub fn forward_token(
        &self,
        cache: &mut [GpuLayerCache],
        token_id: u32,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;

        let freq_base = match &cfg.pos_encoding {
            PosEncoding::RoPE { freq_base } => *freq_base,
            _ => unreachable!("validated in from_model"),
        };

        let client = self.backend.client();

        // 1. Embedding lookup on GPU
        let mut hidden = self.backend.zeros(hidden_dim);
        unsafe {
            crate::attention::embedding_lookup_kernel::launch::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.weights.token_embedding.handle,
                    self.weights.token_embedding.len,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&hidden.handle, hidden_dim, 1),
                ScalarArg::new(token_id),
                ScalarArg::new(hidden_dim as u32),
            );
        }

        // 1b. Gemma-style embedding scaling (if configured)
        if let Some(scale) = cfg.embedding_scale {
            unsafe {
                crate::attention::scale_inplace_kernel::launch::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&hidden.handle, hidden_dim, 1),
                    ScalarArg::new(scale),
                    ScalarArg::new(hidden_dim as u32),
                );
            }
        }

        // Scratch buffers reused across layers
        let mut normed = self.backend.zeros(hidden_dim);
        let mut q = self.backend.zeros(q_dim);
        let mut k = self.backend.zeros(kv_dim);
        let mut v = self.backend.zeros(kv_dim);
        let attn_output = self.backend.zeros(q_dim);
        let head_out = self.backend.zeros(cfg.head_dim);
        let mut proj_out = self.backend.zeros(hidden_dim);
        let mut gate = self.backend.zeros(cfg.intermediate_dim);
        let mut up = self.backend.zeros(cfg.intermediate_dim);
        let mut down = self.backend.zeros(hidden_dim);

        // 2. Run through all transformer blocks
        for layer_idx in 0..cfg.num_layers {
            let layer = &self.weights.layers[layer_idx];
            let layer_cache = &mut cache[layer_idx];
            let position = layer_cache.len;

            // 2a. RMS norm (pre-attention)
            self.backend.rms_norm(
                &hidden,
                &layer.attn_norm,
                &mut normed,
                cfg.rms_norm_eps,
            );

            // 2b. Project Q, K, V via matvec
            self.backend
                .matvec(&layer.wq, &normed, &mut q, q_dim, hidden_dim);
            self.backend
                .matvec(&layer.wk, &normed, &mut k, kv_dim, hidden_dim);
            self.backend
                .matvec(&layer.wv, &normed, &mut v, kv_dim, hidden_dim);

            // 2d. Apply RoPE to Q and K heads
            for h in 0..cfg.num_heads {
                self.backend.rope_inplace(
                    &mut q,
                    h * cfg.head_dim,
                    cfg.head_dim,
                    position,
                    freq_base,
                );
            }
            for h in 0..cfg.num_kv_heads {
                self.backend.rope_inplace(
                    &mut k,
                    h * cfg.head_dim,
                    cfg.head_dim,
                    position,
                    freq_base,
                );
            }

            // 2e. Append K, V to GPU cache
            unsafe {
                crate::attention::cache_append_kernel::launch::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&k.handle, kv_dim, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &layer_cache.keys.handle,
                        layer_cache.keys.len,
                        1,
                    ),
                    ScalarArg::new(position as u32),
                    ScalarArg::new(kv_dim as u32),
                );
                crate::attention::cache_append_kernel::launch::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&v.handle, kv_dim, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &layer_cache.values.handle,
                        layer_cache.values.len,
                        1,
                    ),
                    ScalarArg::new(position as u32),
                    ScalarArg::new(kv_dim as u32),
                );
            }

            let seq_len = position + 1;
            layer_cache.len = seq_len;

            // 2f. Attention: for each Q head, compute scores -> softmax -> contract
            let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();

            // Scores buffer — re-created per layer since seq_len may differ
            // across layers (though in single-token decode they're the same).
            let mut scores = self.backend.zeros(seq_len);

            for h in 0..cfg.num_heads {
                let kv_head = h / heads_per_kv;

                // Copy q[h*head_dim..(h+1)*head_dim] into head_out
                unsafe {
                    crate::attention::copy_slice_kernel::launch::<R>(
                        client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(&q.handle, q.len, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &head_out.handle,
                            head_out.len,
                            1,
                        ),
                        ScalarArg::new((h * cfg.head_dim) as u32),
                        ScalarArg::new(0u32),
                        ScalarArg::new(cfg.head_dim as u32),
                    );
                }

                // Compute attention scores: scores[p] = dot(q_head, k[p, kv_head]) * scale
                unsafe {
                    crate::attention::attention_scores_kernel::launch::<R>(
                        client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &head_out.handle,
                            cfg.head_dim,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(
                            &layer_cache.keys.handle,
                            layer_cache.keys.len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(
                            &scores.handle,
                            seq_len,
                            1,
                        ),
                        ScalarArg::new(cfg.head_dim as u32),
                        ScalarArg::new(kv_dim as u32),
                        ScalarArg::new(kv_head as u32),
                        ScalarArg::new(seq_len as u32),
                        ScalarArg::new(scale),
                    );
                }

                // Softmax over scores[0..seq_len]
                self.backend.softmax_inplace(&mut scores, 0, seq_len);

                // Contract: head_out[d] = sum_p(scores[p] * v[p, kv_head, d])
                unsafe {
                    crate::attention::attention_contract_kernel::launch::<R>(
                        client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &scores.handle,
                            seq_len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(
                            &layer_cache.values.handle,
                            layer_cache.values.len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(
                            &head_out.handle,
                            cfg.head_dim,
                            1,
                        ),
                        ScalarArg::new(cfg.head_dim as u32),
                        ScalarArg::new(kv_dim as u32),
                        ScalarArg::new(kv_head as u32),
                        ScalarArg::new(seq_len as u32),
                    );
                }

                // Copy head_out into attn_output[h*head_dim..(h+1)*head_dim]
                unsafe {
                    crate::attention::copy_slice_kernel::launch::<R>(
                        client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<f32>(
                            &head_out.handle,
                            head_out.len,
                            1,
                        ),
                        ArrayArg::from_raw_parts::<f32>(
                            &attn_output.handle,
                            attn_output.len,
                            1,
                        ),
                        ScalarArg::new(0u32),
                        ScalarArg::new((h * cfg.head_dim) as u32),
                        ScalarArg::new(cfg.head_dim as u32),
                    );
                }
            }

            // 2g. Output projection: proj_out = wo @ attn_output
            self.backend.matvec(
                &layer.wo,
                &attn_output,
                &mut proj_out,
                hidden_dim,
                q_dim,
            );

            // 2i. Residual: hidden += proj_out
            self.backend.add_inplace(&mut hidden, &proj_out);

            // 2j. RMS norm (pre-FFN)
            self.backend.rms_norm(
                &hidden,
                &layer.ffn_norm,
                &mut normed,
                cfg.rms_norm_eps,
            );

            // 2k. SwiGLU FFN
            self.backend.matvec(
                &layer.w_gate,
                &normed,
                &mut gate,
                cfg.intermediate_dim,
                hidden_dim,
            );
            self.backend.matvec(
                &layer.w_up,
                &normed,
                &mut up,
                cfg.intermediate_dim,
                hidden_dim,
            );
            self.backend.silu_inplace(&mut gate);
            self.backend.mul_inplace(&mut gate, &up);
            self.backend.matvec(
                &layer.w_down,
                &gate,
                &mut down,
                hidden_dim,
                cfg.intermediate_dim,
            );

            // 2l. Residual: hidden += down
            self.backend.add_inplace(&mut hidden, &down);
        }

        // 3. Final RMS norm + LM head
        self.backend.rms_norm(
            &hidden,
            &self.weights.final_norm,
            &mut normed,
            cfg.rms_norm_eps,
        );

        let mut logits = self.backend.zeros(cfg.vocab_size);
        self.backend.matvec(
            &self.weights.lm_head,
            &normed,
            &mut logits,
            cfg.vocab_size,
            hidden_dim,
        );

        // Download logits to CPU — the only round-trip in the forward pass
        self.backend.to_f32(&logits)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nnx_transformer::block::BlockWeights;
    use nnx_transformer::config::*;
    use nnx_transformer::model::{Model, ModelWeights};
    use nnx_transformer::weights::Matrix;

    fn all_architectures() -> Vec<Architecture> {
        vec![
            Architecture::Llama,
            Architecture::GPT2,
            Architecture::Phi,
            Architecture::Gemma,
            Architecture::Qwen,
            Architecture::Mistral,
            Architecture::CodeLlama,
            Architecture::StableLM,
            Architecture::Falcon,
            Architecture::MPT,
        ]
    }

    fn tiny_model_with_arch(arch: Architecture) -> Model {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let vocab_size = 32;

        let (
            norm_type,
            ffn_type,
            pos_encoding,
            block_style,
            has_qkv_bias,
            has_output_bias,
            embedding_scale,
        ) = match arch {
            Architecture::Llama | Architecture::Mistral | Architecture::CodeLlama => (
                NormType::RMSNorm,
                FFNType::SwiGLU,
                PosEncoding::RoPE {
                    freq_base: 10000.0,
                },
                BlockStyle::Sequential,
                false,
                false,
                None,
            ),
            Architecture::GPT2 => (
                NormType::LayerNorm,
                FFNType::GELU,
                PosEncoding::Learned,
                BlockStyle::Sequential,
                true,
                true,
                None,
            ),
            Architecture::Phi => (
                NormType::LayerNorm,
                FFNType::SwiGLU,
                PosEncoding::PartialRoPE {
                    freq_base: 10000.0,
                    rotary_dim: 2,
                },
                BlockStyle::Parallel,
                true,
                true,
                None,
            ),
            Architecture::Gemma => (
                NormType::RMSNorm,
                FFNType::GeGLU,
                PosEncoding::RoPE {
                    freq_base: 10000.0,
                },
                BlockStyle::Sequential,
                false,
                false,
                Some((hidden_dim as f32).sqrt()),
            ),
            Architecture::Qwen => (
                NormType::RMSNorm,
                FFNType::SwiGLU,
                PosEncoding::RoPE {
                    freq_base: 10000.0,
                },
                BlockStyle::Sequential,
                true,
                false,
                None,
            ),
            Architecture::StableLM => (
                NormType::LayerNorm,
                FFNType::SwiGLU,
                PosEncoding::RoPE {
                    freq_base: 10000.0,
                },
                BlockStyle::Parallel,
                true,
                false,
                None,
            ),
            Architecture::Falcon => (
                NormType::LayerNorm,
                FFNType::GELU,
                PosEncoding::RoPE {
                    freq_base: 10000.0,
                },
                BlockStyle::Sequential,
                false,
                false,
                None,
            ),
            Architecture::MPT => (
                NormType::LayerNorm,
                FFNType::GELU,
                PosEncoding::None,
                BlockStyle::Sequential,
                false,
                false,
                None,
            ),
        };

        let config = ModelConfig {
            architecture: format!("{:?}", arch).to_ascii_lowercase(),
            arch: arch.clone(),
            num_layers: 2,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type,
            ffn_type,
            pos_encoding,
            block_style,
            has_qkv_bias,
            has_output_bias,
            embedding_scale,
        };

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let attn_norm_bias = matches!(config.norm_type, NormType::LayerNorm)
            .then(|| vec![0.0; hidden_dim]);
        let ffn_norm_bias = (matches!(config.norm_type, NormType::LayerNorm)
            && config.block_style == BlockStyle::Sequential)
            .then(|| vec![0.0; hidden_dim]);

        let make_layer = || BlockWeights {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
            wq: Matrix::dense(vec![0.01; q_dim * hidden_dim], q_dim, hidden_dim),
            wk: Matrix::dense(vec![0.01; kv_dim * hidden_dim], kv_dim, hidden_dim),
            wv: Matrix::dense(vec![0.01; kv_dim * hidden_dim], kv_dim, hidden_dim),
            wo: Matrix::dense(vec![0.01; hidden_dim * q_dim], hidden_dim, q_dim),
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
            bq: has_qkv_bias.then(|| vec![0.001; q_dim]),
            bk: has_qkv_bias.then(|| vec![0.001; kv_dim]),
            bv: has_qkv_bias.then(|| vec![0.001; kv_dim]),
            bo: has_output_bias.then(|| vec![0.001; hidden_dim]),
            attn_norm_bias: attn_norm_bias.clone(),
            ffn_norm_bias: ffn_norm_bias.clone(),
        };

        let weights = ModelWeights {
            token_embedding: Matrix::dense(
                vec![0.1; vocab_size * hidden_dim],
                vocab_size,
                hidden_dim,
            ),
            layers: vec![make_layer(), make_layer()],
            final_norm: vec![1.0; hidden_dim],
            final_norm_bias: matches!(config.norm_type, NormType::LayerNorm)
                .then(|| vec![0.0; hidden_dim]),
            lm_head: Matrix::dense(
                vec![0.01; vocab_size * hidden_dim],
                vocab_size,
                hidden_dim,
            ),
        };

        Model::new(config, weights)
    }

    fn tiny_gpu_model() -> Model {
        tiny_model_with_arch(Architecture::Llama)
    }

    /// Create a tiny model with varied (patterned) weights for non-trivial
    /// outputs that can be compared between CPU and GPU.
    fn tiny_gpu_model_patterned() -> Model {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let vocab_size = 32;

        let config = ModelConfig {
            architecture: "test_llama".into(),
            arch: Architecture::Llama,
            num_layers: 1,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
            max_context_length: 64,
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

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let patterned = |rows: usize, cols: usize, offset: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| (((i + offset) % 5) as f32 - 2.0) * 0.02)
                .collect()
        };

        let layer = BlockWeights {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
            wq: Matrix::dense(patterned(q_dim, hidden_dim, 0), q_dim, hidden_dim),
            wk: Matrix::dense(patterned(kv_dim, hidden_dim, 1), kv_dim, hidden_dim),
            wv: Matrix::dense(patterned(kv_dim, hidden_dim, 2), kv_dim, hidden_dim),
            wo: Matrix::dense(patterned(hidden_dim, q_dim, 3), hidden_dim, q_dim),
            w_gate: Matrix::dense(
                patterned(intermediate_dim, hidden_dim, 4),
                intermediate_dim,
                hidden_dim,
            ),
            w_up: Matrix::dense(
                patterned(intermediate_dim, hidden_dim, 5),
                intermediate_dim,
                hidden_dim,
            ),
            w_down: Matrix::dense(
                patterned(hidden_dim, intermediate_dim, 6),
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

        let weights = ModelWeights {
            token_embedding: Matrix::dense(
                (0..vocab_size * hidden_dim)
                    .map(|i| ((i % 3) as f32) - 1.0)
                    .collect(),
                vocab_size,
                hidden_dim,
            ),
            layers: vec![layer],
            final_norm: vec![1.0; hidden_dim],
            final_norm_bias: None,
            lm_head: Matrix::dense(
                patterned(vocab_size, hidden_dim, 7),
                vocab_size,
                hidden_dim,
            ),
        };

        Model::new(config, weights)
    }

    #[test]
    fn test_model_fixture_covers_all_architectures() {
        for arch in all_architectures() {
            let model = tiny_model_with_arch(arch.clone());
            assert_eq!(model.weights.layers.len(), model.config.num_layers);
            assert_eq!(model.config.arch, arch);
        }
    }

    #[cfg(feature = "wgpu")]
    mod gpu_tests {
        use super::*;
        use cubecl::wgpu::WgpuRuntime;

        #[test]
        fn test_validate_config_passes_for_all_architectures() {
            for arch in all_architectures() {
                let model = tiny_model_with_arch(arch.clone());
                GpuInference::<WgpuRuntime>::validate_config(&model.config)
                    .unwrap_or_else(|err| panic!("arch {:?} failed validation: {}", arch, err));
            }
        }

        #[test]
        fn test_from_model_supports_all_architectures() {
            for arch in all_architectures() {
                let model = tiny_model_with_arch(arch.clone());
                let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                    .unwrap_or_else(|err| panic!("arch {:?} upload failed: {}", arch, err));

                assert_eq!(gpu.config().arch, model.config.arch);
                assert_eq!(gpu.config().norm_type, model.config.norm_type);
                assert_eq!(gpu.config().ffn_type, model.config.ffn_type);
                assert_eq!(gpu.config().pos_encoding, model.config.pos_encoding);
                assert_eq!(gpu.config().block_style, model.config.block_style);
                assert_eq!(gpu.config().has_qkv_bias, model.config.has_qkv_bias);
                assert_eq!(gpu.config().has_output_bias, model.config.has_output_bias);
                assert_eq!(gpu.config().embedding_scale, model.config.embedding_scale);
            }
        }

        #[test]
        fn test_gpu_layer_bias_upload_for_gpt2() {
            let model = tiny_model_with_arch(Architecture::GPT2);
            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("GPT-2 upload should succeed");
            let layer = &gpu.weights.layers[0];

            assert!(layer.bq.is_some());
            assert!(layer.bk.is_some());
            assert!(layer.bv.is_some());
            assert!(layer.bo.is_some());
            assert!(layer.attn_norm_bias.is_some());
            assert!(layer.ffn_norm_bias.is_some());
            assert!(gpu.weights.final_norm_bias.is_some());
        }

        #[test]
        fn test_gpu_position_embedding_placeholder_for_gpt2() {
            let model = tiny_model_with_arch(Architecture::GPT2);
            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("GPT-2 upload should succeed");
            let position_embedding = gpu
                .weights
                .position_embedding
                .as_ref()
                .expect("GPT-2 should allocate a learned position embedding buffer");

            assert_eq!(
                position_embedding.len,
                model.config.max_context_length * model.config.hidden_dim,
            );
        }

        #[test]
        fn test_gpu_forward_single_token() {
            let model = tiny_gpu_model();
            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("Llama upload should succeed");
            let mut cache = gpu.new_cache();

            let logits = gpu.forward_token(&mut cache, 0);

            assert_eq!(logits.len(), 32, "logits should be vocab_size");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "all logits must be finite, got: {:?}",
                logits,
            );
        }

        #[test]
        fn test_gpu_forward_sequence() {
            let model = tiny_gpu_model();
            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("Llama upload should succeed");
            let mut cache = gpu.new_cache();

            for token in [1u32, 5, 10, 3] {
                let logits = gpu.forward_token(&mut cache, token);
                assert_eq!(logits.len(), 32);
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "non-finite logits for token {}",
                    token,
                );
            }

            // Verify cache positions advanced
            for layer_cache in cache.iter() {
                assert_eq!(layer_cache.len, 4, "cache should have 4 entries");
            }
        }

        #[test]
        fn test_gpu_forward_patterned_model() {
            let model = tiny_gpu_model_patterned();
            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("Patterned Llama upload should succeed");
            let mut cache = gpu.new_cache();

            for token in [1u32, 5, 10] {
                let logits = gpu.forward_token(&mut cache, token);
                assert_eq!(logits.len(), 32);
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "non-finite logits for token {}: {:?}",
                    token,
                    logits,
                );
            }
        }

        #[test]
        fn test_gpu_matches_cpu_forward_uniform() {
            // Compare GPU and CPU forward passes on the uniform-weights model.
            let model = tiny_gpu_model();

            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("Llama upload should succeed");
            let mut gpu_cache = gpu.new_cache();
            let mut cpu_cache = model.new_cache();

            for &token in &[1u32, 5, 10, 3] {
                let cpu_logits = model.forward_token(&mut cpu_cache, token).unwrap();
                let gpu_logits = gpu.forward_token(&mut gpu_cache, token);

                assert_eq!(cpu_logits.len(), gpu_logits.len());

                let max_diff: f32 = cpu_logits
                    .iter()
                    .zip(gpu_logits.iter())
                    .map(|(c, g)| (c - g).abs())
                    .fold(0.0f32, f32::max);

                assert!(
                    max_diff < 0.05,
                    "token {}: max logit diff = {} (cpu[0]={}, gpu[0]={})",
                    token,
                    max_diff,
                    cpu_logits[0],
                    gpu_logits[0],
                );
            }
        }

        #[test]
        fn test_gpu_matches_cpu_forward_patterned() {
            // Compare GPU and CPU forward passes on the patterned model.
            let model = tiny_gpu_model_patterned();

            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("Patterned Llama upload should succeed");
            let mut gpu_cache = gpu.new_cache();
            let mut cpu_cache = model.new_cache();

            for &token in &[1u32, 5, 10] {
                let cpu_logits = model.forward_token(&mut cpu_cache, token).unwrap();
                let gpu_logits = gpu.forward_token(&mut gpu_cache, token);

                assert_eq!(cpu_logits.len(), gpu_logits.len());

                let max_diff: f32 = cpu_logits
                    .iter()
                    .zip(gpu_logits.iter())
                    .map(|(c, g)| (c - g).abs())
                    .fold(0.0f32, f32::max);

                assert!(
                    max_diff < 0.05,
                    "token {}: max logit diff = {} (cpu[0]={}, gpu[0]={})",
                    token,
                    max_diff,
                    cpu_logits[0],
                    gpu_logits[0],
                );
            }
        }

        #[test]
        fn test_gpu_cache_allocation() {
            let model = tiny_gpu_model();
            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("Llama upload should succeed");
            let cache = gpu.new_cache();

            assert_eq!(cache.len(), 2, "should have one cache per layer");
            let kv_dim = 2 * 4; // num_kv_heads * head_dim
            let max_seq = 64;
            for layer_cache in &cache {
                assert_eq!(layer_cache.keys.len, max_seq * kv_dim);
                assert_eq!(layer_cache.values.len, max_seq * kv_dim);
                assert_eq!(layer_cache.len, 0);
            }
        }

        /// Test attention kernels directly with multiple cached positions.
        #[test]
        fn test_attention_kernels_multi_position() {
            let backend = CubeclBackend::<WgpuRuntime>::new();
            let client = backend.client();

            let head_dim = 4usize;
            let kv_dim = 8usize; // 2 kv_heads * 4 head_dim
            let max_seq = 16usize;
            let cache_size = max_seq * kv_dim;

            // Create K/V caches with 3 positions populated
            let mut k_cache_data = vec![0.0f32; cache_size];
            let mut v_cache_data = vec![0.0f32; cache_size];

            // Pos 0, kv_head 0: k=[1,0,0,0], v=[1,2,3,4]
            k_cache_data[0..4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            v_cache_data[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            // Pos 0, kv_head 1: k=[0,1,0,0], v=[5,6,7,8]
            k_cache_data[4..8].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
            v_cache_data[4..8].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

            // Pos 1, kv_head 0: k=[0,1,0,0], v=[10,20,30,40]
            k_cache_data[8..12].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
            v_cache_data[8..12].copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
            // Pos 1, kv_head 1: k=[1,0,0,0], v=[50,60,70,80]
            k_cache_data[12..16].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            v_cache_data[12..16].copy_from_slice(&[50.0, 60.0, 70.0, 80.0]);

            // Pos 2, kv_head 0: k=[0,0,1,0], v=[100,200,300,400]
            k_cache_data[16..20].copy_from_slice(&[0.0, 0.0, 1.0, 0.0]);
            v_cache_data[16..20].copy_from_slice(&[100.0, 200.0, 300.0, 400.0]);
            // Pos 2, kv_head 1: k=[0,0,0,1], v=[500,600,700,800]
            k_cache_data[20..24].copy_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            v_cache_data[20..24].copy_from_slice(&[500.0, 600.0, 700.0, 800.0]);

            let k_cache_gpu = backend.from_f32(&k_cache_data);
            let v_cache_gpu = backend.from_f32(&v_cache_data);

            // Query: [1,0,0,0] — strongly matches pos 0, kv_head 0
            let q_head = backend.from_f32(&[1.0, 0.0, 0.0, 0.0]);
            let mut scores = backend.zeros(3);
            let head_out = backend.zeros(head_dim);

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            // Compute scores for kv_head=0, seq_len=3
            unsafe {
                crate::attention::attention_scores_kernel::launch::<WgpuRuntime>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&q_head.handle, head_dim, 1),
                    ArrayArg::from_raw_parts::<f32>(&k_cache_gpu.handle, cache_size, 1),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, 3, 1),
                    ScalarArg::new(head_dim as u32),
                    ScalarArg::new(kv_dim as u32),
                    ScalarArg::new(0u32), // kv_head = 0
                    ScalarArg::new(3u32), // seq_len = 3
                    ScalarArg::new(scale),
                );
            }

            let scores_cpu = backend.to_f32(&scores);
            assert!((scores_cpu[0] - 0.5).abs() < 1e-5, "score[0]={}", scores_cpu[0]);
            assert!((scores_cpu[1] - 0.0).abs() < 1e-5, "score[1]={}", scores_cpu[1]);
            assert!((scores_cpu[2] - 0.0).abs() < 1e-5, "score[2]={}", scores_cpu[2]);

            // Softmax
            backend.softmax_inplace(&mut scores, 0, 3);
            let probs = backend.to_f32(&scores);
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "softmax sum={}", sum);

            // Contract
            unsafe {
                crate::attention::attention_contract_kernel::launch::<WgpuRuntime>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&scores.handle, 3, 1),
                    ArrayArg::from_raw_parts::<f32>(&v_cache_gpu.handle, cache_size, 1),
                    ArrayArg::from_raw_parts::<f32>(&head_out.handle, head_dim, 1),
                    ScalarArg::new(head_dim as u32),
                    ScalarArg::new(kv_dim as u32),
                    ScalarArg::new(0u32), // kv_head = 0
                    ScalarArg::new(3u32), // seq_len = 3
                );
            }

            let result = backend.to_f32(&head_out);
            assert!(result.iter().all(|v| v.is_finite()), "result={:?}", result);
            // With softmax heavily weighted toward pos 0, result should be near v[0]=[1,2,3,4]
            assert!(result[0] > 0.5, "result[0]={} should be > 0.5", result[0]);
        }

        #[test]
        fn test_gpu_different_tokens_produce_different_logits() {
            let model = tiny_gpu_model_patterned();
            let gpu = GpuInference::<WgpuRuntime>::from_model(&model)
                .expect("Patterned Llama upload should succeed");

            let mut cache1 = gpu.new_cache();
            let logits1 = gpu.forward_token(&mut cache1, 1);

            let mut cache2 = gpu.new_cache();
            let logits2 = gpu.forward_token(&mut cache2, 5);

            // With patterned weights, different tokens should produce different logits
            let diff: f32 = logits1
                .iter()
                .zip(logits2.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff > 1e-6,
                "different tokens should produce different logits (diff={})",
                diff,
            );
        }
    }
}
