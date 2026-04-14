//! Full transformer model: embedding -> N blocks -> final norm -> LM head.
//!
//! Supports multiple architectures: Llama, GPT-2, Phi, Gemma, Qwen.

use crate::block::{self, BlockWeights};
use crate::cache::KVCache;
use crate::config::{ModelConfig, NormType};
use crate::weights::Matrix;
use nnx_core::error::Result;

/// All weights for a complete model.
pub struct ModelWeights {
    /// Token embedding table [vocab_size, hidden_dim]
    pub token_embedding: Matrix,
    /// Per-layer transformer block weights.
    pub layers: Vec<BlockWeights>,
    /// Final norm weights [hidden_dim]
    pub final_norm: Vec<f32>,
    /// Final norm bias [hidden_dim] (for LayerNorm architectures; None for RMSNorm)
    pub final_norm_bias: Option<Vec<f32>>,
    /// LM head (output projection) [vocab_size, hidden_dim]
    /// Often tied with token_embedding (same data).
    pub lm_head: Matrix,
}

/// A loaded model ready for inference.
pub struct Model {
    pub config: ModelConfig,
    pub weights: ModelWeights,
}

impl Model {
    /// Create a model from config and weights.
    pub fn new(config: ModelConfig, weights: ModelWeights) -> Self {
        Self { config, weights }
    }

    /// Create a fresh KV cache for a new request/session.
    pub fn new_cache(&self) -> KVCache {
        KVCache::new(
            self.config.num_layers,
            self.config.max_context_length,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
    }

    fn forward_token_impl(
        &self,
        cache: &mut KVCache,
        token_id: u32,
        requested_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>)>)> {
        let cfg = &self.config;
        let position = cache.position();

        // 1. Token embedding lookup
        let mut hidden = vec![0.0f32; cfg.hidden_dim];
        self.weights
            .token_embedding
            .copy_row_to(token_id as usize, &mut hidden);

        // 1b. Gemma scales embeddings by sqrt(hidden_dim)
        if let Some(scale) = cfg.embedding_scale {
            for v in &mut hidden {
                *v *= scale;
            }
        }

        let mut layer_features = Vec::new();

        // 2. Run through all transformer blocks
        for layer_idx in 0..cfg.num_layers {
            block::forward_block(
                &mut hidden,
                &self.weights.layers[layer_idx],
                cache.layer_mut(layer_idx),
                position,
                cfg,
            )?;

            if requested_layers.contains(&layer_idx) {
                layer_features.push((layer_idx, hidden.clone()));
            }
        }

        // 3. Final norm (architecture-aware)
        let mut normed = vec![0.0f32; cfg.hidden_dim];
        match cfg.norm_type {
            NormType::RMSNorm => {
                nnx_kernels::rms_norm::rms_norm_f32_checked(
                    &hidden,
                    &self.weights.final_norm,
                    &mut normed,
                    cfg.rms_norm_eps,
                )?;
            }
            NormType::LayerNorm => {
                nnx_kernels::normalization::layer_norm_f32_checked(
                    &hidden,
                    &self.weights.final_norm,
                    self.weights.final_norm_bias.as_deref(),
                    &mut normed,
                    cfg.hidden_dim,
                    cfg.rms_norm_eps,
                )?;
            }
        }

        // 4. LM head: project to vocab
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.weights.lm_head.matvec(&normed, &mut logits);

        Ok((logits, layer_features))
    }

    /// Run forward pass for a single token, returning logits [vocab_size].
    ///
    /// This is the decode step -- processes one token using the KV cache.
    pub fn forward_token(&self, cache: &mut KVCache, token_id: u32) -> Result<Vec<f32>> {
        self.forward_token_impl(cache, token_id, &[])
            .map(|(logits, _)| logits)
    }

    /// Run forward pass for a single token and extract requested layer features.
    pub fn forward_token_with_features(
        &self,
        cache: &mut KVCache,
        token_id: u32,
        requested_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>)>)> {
        self.forward_token_impl(cache, token_id, requested_layers)
    }

    /// Run forward pass for a batch of tokens (prefill).
    /// Returns logits only for the LAST token [vocab_size].
    /// All tokens are added to the KV cache.
    pub fn forward_batch(&self, cache: &mut KVCache, token_ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_batch_with_features(cache, token_ids, &[])
            .map(|(logits, _)| logits)
    }

    /// Run forward pass for a batch and extract requested layer features for the last token.
    pub fn forward_batch_with_features(
        &self,
        cache: &mut KVCache,
        token_ids: &[u32],
        requested_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>)>)> {
        if token_ids.is_empty() {
            return Ok((vec![0.0; self.config.vocab_size], Vec::new()));
        }
        if token_ids.len() == 1 {
            return self.forward_token_impl(cache, token_ids[0], requested_layers);
        }

        let cfg = &self.config;
        let batch_size = token_ids.len();
        let mut hidden_batch = vec![0.0f32; batch_size * cfg.hidden_dim];

        for (batch_idx, &token_id) in token_ids.iter().enumerate() {
            let dst =
                &mut hidden_batch[batch_idx * cfg.hidden_dim..(batch_idx + 1) * cfg.hidden_dim];
            self.weights
                .token_embedding
                .copy_row_to(token_id as usize, dst);
            if let Some(scale) = cfg.embedding_scale {
                for value in dst.iter_mut() {
                    *value *= scale;
                }
            }
        }

        let mut last_features = Vec::new();
        for layer_idx in 0..cfg.num_layers {
            let start_position = cache.layer(layer_idx).len();
            block::forward_block_batch(
                &mut hidden_batch,
                batch_size,
                &self.weights.layers[layer_idx],
                cache.layer_mut(layer_idx),
                start_position,
                cfg,
            )?;

            if requested_layers.contains(&layer_idx) {
                let start = (batch_size - 1) * cfg.hidden_dim;
                last_features.push((
                    layer_idx,
                    hidden_batch[start..start + cfg.hidden_dim].to_vec(),
                ));
            }
        }

        let last_hidden =
            &hidden_batch[(batch_size - 1) * cfg.hidden_dim..batch_size * cfg.hidden_dim];
        let mut normed = vec![0.0f32; cfg.hidden_dim];
        match cfg.norm_type {
            NormType::RMSNorm => {
                nnx_kernels::rms_norm::rms_norm_f32_checked(
                    last_hidden,
                    &self.weights.final_norm,
                    &mut normed,
                    cfg.rms_norm_eps,
                )?;
            }
            NormType::LayerNorm => {
                nnx_kernels::normalization::layer_norm_f32_checked(
                    last_hidden,
                    &self.weights.final_norm,
                    self.weights.final_norm_bias.as_deref(),
                    &mut normed,
                    cfg.hidden_dim,
                    cfg.rms_norm_eps,
                )?;
            }
        }

        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.weights.lm_head.matvec(&normed, &mut logits);
        Ok((logits, last_features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use nnx_quant::GGMLType;
    use nnx_quant::blocks::BlockQ8_0;

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
            Architecture::Llama => (
                NormType::RMSNorm,
                FFNType::SwiGLU,
                PosEncoding::RoPE { freq_base: 10000.0 },
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
                PosEncoding::RoPE { freq_base: 10000.0 },
                BlockStyle::Sequential,
                false,
                false,
                Some((hidden_dim as f32).sqrt()),
            ),
            Architecture::Qwen => (
                NormType::RMSNorm,
                FFNType::SwiGLU,
                PosEncoding::RoPE { freq_base: 10000.0 },
                BlockStyle::Sequential,
                true,
                false,
                None,
            ),
            // Mistral and CodeLlama share the Llama execution path.
            Architecture::Mistral | Architecture::CodeLlama => (
                NormType::RMSNorm,
                FFNType::SwiGLU,
                PosEncoding::RoPE { freq_base: 10000.0 },
                BlockStyle::Sequential,
                false,
                false,
                None,
            ),
            // StableLM: LayerNorm + RoPE + SwiGLU + parallel attention.
            Architecture::StableLM => (
                NormType::LayerNorm,
                FFNType::SwiGLU,
                PosEncoding::RoPE { freq_base: 10000.0 },
                BlockStyle::Parallel,
                true,
                true,
                None,
            ),
            // Falcon and MPT: LayerNorm + MHA + GELU + sequential.
            Architecture::Falcon | Architecture::MPT => (
                NormType::LayerNorm,
                FFNType::GELU,
                PosEncoding::RoPE { freq_base: 10000.0 },
                BlockStyle::Sequential,
                false,
                false,
                None,
            ),
        };

        let config = ModelConfig {
            architecture: format!("{:?}", arch).to_lowercase(),
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
            norm_type: norm_type.clone(),
            ffn_type,
            pos_encoding,
            block_style,
            has_qkv_bias,
            has_output_bias,
            embedding_scale,
        };

        let make_layer = || {
            if has_qkv_bias {
                BlockWeights::test_with_bias(
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate_dim,
                )
            } else {
                BlockWeights::test_no_bias(
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate_dim,
                )
            }
        };

        let final_norm_bias = match norm_type {
            NormType::LayerNorm => Some(vec![0.0; hidden_dim]),
            NormType::RMSNorm => None,
        };

        let weights = ModelWeights {
            token_embedding: Matrix::dense(
                vec![0.1; vocab_size * hidden_dim],
                vocab_size,
                hidden_dim,
            ),
            layers: vec![make_layer(), make_layer()],
            final_norm: vec![1.0; hidden_dim],
            final_norm_bias,
            lm_head: Matrix::dense(vec![0.01; vocab_size * hidden_dim], vocab_size, hidden_dim),
        };

        Model::new(config, weights)
    }

    fn tiny_model() -> Model {
        tiny_model_with_arch(Architecture::Llama)
    }

    fn q8_matrix_from_dense(data: &[f32], rows: usize, cols: usize) -> Matrix {
        let mut bytes = Vec::new();
        let num_blocks = (cols + 31) / 32;

        for row in 0..rows {
            for block in 0..num_blocks {
                let mut quants = [0i8; 32];
                for lane in 0..32 {
                    let col = block * 32 + lane;
                    if col < cols {
                        quants[lane] = data[row * cols + col] as i8;
                    }
                }

                let packed = BlockQ8_0 {
                    scale: half::f16::from_f32(1.0),
                    quants,
                };
                bytes.extend_from_slice(&packed.scale.to_bits().to_le_bytes());
                bytes.extend(packed.quants.iter().map(|v| *v as u8));
            }
        }

        Matrix::quantized(bytes, GGMLType::Q8_0, rows, cols).unwrap()
    }

    fn compact_test_model(quantized: bool) -> Model {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let vocab_size = 32;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut config = ModelConfig::test_llama(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
        );
        config.num_layers = 1;

        let matrix = |data: Vec<f32>, rows: usize, cols: usize| {
            if quantized {
                q8_matrix_from_dense(&data, rows, cols)
            } else {
                Matrix::dense(data, rows, cols)
            }
        };
        let patterned = |rows: usize, cols: usize, offset: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i + offset) % 5) as f32 - 2.0)
                .collect()
        };

        let weights = ModelWeights {
            token_embedding: matrix(
                (0..vocab_size * hidden_dim)
                    .map(|i| ((i % 3) as f32) - 1.0)
                    .collect(),
                vocab_size,
                hidden_dim,
            ),
            layers: vec![BlockWeights {
                attn_norm: vec![1.0; hidden_dim],
                ffn_norm: vec![1.0; hidden_dim],
                wq: matrix(patterned(q_dim, hidden_dim, 0), q_dim, hidden_dim),
                wk: matrix(patterned(kv_dim, hidden_dim, 1), kv_dim, hidden_dim),
                wv: matrix(patterned(kv_dim, hidden_dim, 2), kv_dim, hidden_dim),
                wo: matrix(patterned(hidden_dim, q_dim, 3), hidden_dim, q_dim),
                w_gate: matrix(
                    patterned(intermediate_dim, hidden_dim, 4),
                    intermediate_dim,
                    hidden_dim,
                ),
                w_up: matrix(
                    patterned(intermediate_dim, hidden_dim, 5),
                    intermediate_dim,
                    hidden_dim,
                ),
                w_down: matrix(
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
            }],
            final_norm: vec![1.0; hidden_dim],
            final_norm_bias: None,
            lm_head: matrix(patterned(vocab_size, hidden_dim, 7), vocab_size, hidden_dim),
        };

        Model::new(config, weights)
    }

    #[test]
    fn test_forward_single_token() {
        let model = tiny_model();
        let mut cache = model.new_cache();
        let logits = model.forward_token(&mut cache, 0).unwrap();
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(cache.position(), 1);
    }

    #[test]
    fn test_forward_sequence() {
        let model = tiny_model();
        let mut cache = model.new_cache();
        for token in [1, 5, 10, 3] {
            let logits = model.forward_token(&mut cache, token).unwrap();
            assert_eq!(logits.len(), 32);
        }
        assert_eq!(cache.position(), 4);
    }

    #[test]
    fn test_gpt2_model_forward() {
        let model = tiny_model_with_arch(Architecture::GPT2);
        let mut cache = model.new_cache();
        let logits = model.forward_token(&mut cache, 0).unwrap();
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_phi_model_forward() {
        let model = tiny_model_with_arch(Architecture::Phi);
        let mut cache = model.new_cache();
        let logits = model.forward_token(&mut cache, 0).unwrap();
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gemma_model_forward() {
        let model = tiny_model_with_arch(Architecture::Gemma);
        let mut cache = model.new_cache();
        let logits = model.forward_token(&mut cache, 0).unwrap();
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_qwen_model_forward() {
        let model = tiny_model_with_arch(Architecture::Qwen);
        let mut cache = model.new_cache();
        let logits = model.forward_token(&mut cache, 0).unwrap();
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_all_architectures_multi_token() {
        for arch in [
            Architecture::Llama,
            Architecture::GPT2,
            Architecture::Phi,
            Architecture::Gemma,
            Architecture::Qwen,
        ] {
            let model = tiny_model_with_arch(arch.clone());
            let mut cache = model.new_cache();
            for token in [1, 5, 10, 3] {
                let logits = model.forward_token(&mut cache, token).unwrap();
                assert_eq!(logits.len(), 32, "arch {:?}: wrong logits size", arch);
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "arch {:?}: non-finite logits",
                    arch,
                );
            }
            assert_eq!(cache.position(), 4, "arch {:?}: wrong position", arch);
        }
    }

    #[test]
    fn test_forward_token_quantized_matches_dense() {
        let dense = compact_test_model(false);
        let quantized = compact_test_model(true);
        let mut dense_cache = dense.new_cache();
        let mut quantized_cache = quantized.new_cache();

        for token in [1, 5, 10] {
            let dense_logits = dense.forward_token(&mut dense_cache, token).unwrap();
            let quantized_logits = quantized
                .forward_token(&mut quantized_cache, token)
                .unwrap();

            for (dense_logit, quantized_logit) in dense_logits.iter().zip(quantized_logits.iter()) {
                assert!(
                    (dense_logit - quantized_logit).abs() < 1e-4,
                    "quantized logits diverged: dense={} quantized={}",
                    dense_logit,
                    quantized_logit
                );
            }
        }

        assert_eq!(dense_cache.position(), quantized_cache.position());
    }

    #[test]
    fn test_forward_batch_matches_sequential_dense() {
        let model = compact_test_model(false);
        let tokens = [1, 5, 10];

        let mut batch_cache = model.new_cache();
        let batch_logits = model.forward_batch(&mut batch_cache, &tokens).unwrap();

        let mut sequential_cache = model.new_cache();
        let mut sequential_logits = Vec::new();
        for &token in &tokens {
            sequential_logits = model.forward_token(&mut sequential_cache, token).unwrap();
        }

        for (batch_logit, sequential_logit) in batch_logits.iter().zip(sequential_logits.iter()) {
            assert!((batch_logit - sequential_logit).abs() < 1e-4);
        }
        assert_eq!(batch_cache.position(), sequential_cache.position());
    }

    #[test]
    fn test_forward_batch_matches_sequential_quantized() {
        let model = compact_test_model(true);
        let tokens = [1, 5, 10];

        let mut batch_cache = model.new_cache();
        let batch_logits = model.forward_batch(&mut batch_cache, &tokens).unwrap();

        let mut sequential_cache = model.new_cache();
        let mut sequential_logits = Vec::new();
        for &token in &tokens {
            sequential_logits = model.forward_token(&mut sequential_cache, token).unwrap();
        }

        for (batch_logit, sequential_logit) in batch_logits.iter().zip(sequential_logits.iter()) {
            assert!((batch_logit - sequential_logit).abs() < 1e-4);
        }
        assert_eq!(batch_cache.position(), sequential_cache.position());
    }
}
