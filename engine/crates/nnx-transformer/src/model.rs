//! Full transformer model: embedding -> N blocks -> final norm -> LM head.
//!
//! Supports multiple architectures: Llama, GPT-2, Phi, Gemma, Qwen.

use crate::block::{self, BlockWeights};
use crate::cache::KVCache;
use crate::config::{ModelConfig, NormType};

/// All weights for a complete model.
pub struct ModelWeights {
    /// Token embedding table [vocab_size, hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Per-layer transformer block weights.
    pub layers: Vec<BlockWeights>,
    /// Final norm weights [hidden_dim]
    pub final_norm: Vec<f32>,
    /// Final norm bias [hidden_dim] (for LayerNorm architectures; None for RMSNorm)
    pub final_norm_bias: Option<Vec<f32>>,
    /// LM head (output projection) [vocab_size, hidden_dim]
    /// Often tied with token_embedding (same data).
    pub lm_head: Vec<f32>,
}

/// A loaded model ready for inference.
pub struct Model {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub cache: KVCache,
}

impl Model {
    /// Create a model from config and weights, allocating KV cache.
    pub fn new(config: ModelConfig, weights: ModelWeights) -> Self {
        let cache = KVCache::new(
            config.num_layers,
            config.max_context_length,
            config.num_kv_heads,
            config.head_dim,
        );
        Self { config, weights, cache }
    }

    /// Run forward pass for a single token, returning logits [vocab_size].
    ///
    /// This is the decode step -- processes one token using the KV cache.
    pub fn forward_token(&mut self, token_id: u32) -> Vec<f32> {
        let cfg = &self.config;
        let position = self.cache.position();

        // 1. Token embedding lookup
        let emb_offset = token_id as usize * cfg.hidden_dim;
        let mut hidden = self.weights.token_embedding
            [emb_offset..emb_offset + cfg.hidden_dim]
            .to_vec();

        // 1b. Gemma scales embeddings by sqrt(hidden_dim)
        if let Some(scale) = cfg.embedding_scale {
            for v in &mut hidden {
                *v *= scale;
            }
        }

        // 2. Run through all transformer blocks
        for layer_idx in 0..cfg.num_layers {
            block::forward_block(
                &mut hidden,
                &self.weights.layers[layer_idx],
                self.cache.layer_mut(layer_idx),
                position,
                cfg,
            );
        }

        // 3. Final norm (architecture-aware)
        let mut normed = vec![0.0f32; cfg.hidden_dim];
        match cfg.norm_type {
            NormType::RMSNorm => {
                nnx_kernels::rms_norm::rms_norm_f32(
                    &hidden, &self.weights.final_norm, &mut normed, cfg.rms_norm_eps,
                );
            }
            NormType::LayerNorm => {
                nnx_kernels::normalization::layer_norm_f32(
                    &hidden,
                    &self.weights.final_norm,
                    self.weights.final_norm_bias.as_deref(),
                    &mut normed,
                    cfg.hidden_dim,
                    cfg.rms_norm_eps,
                );
            }
        }

        // 4. LM head: project to vocab
        let mut logits = vec![0.0f32; cfg.vocab_size];
        nnx_kernels::matmul::matvec_f32(
            &self.weights.lm_head, &normed, &mut logits,
            cfg.vocab_size, cfg.hidden_dim,
        );

        logits
    }

    /// Run forward pass for a batch of tokens (prefill).
    /// Returns logits only for the LAST token [vocab_size].
    /// All tokens are added to the KV cache.
    pub fn forward_batch(&mut self, token_ids: &[u32]) -> Vec<f32> {
        if token_ids.is_empty() {
            return vec![0.0; self.config.vocab_size];
        }
        if token_ids.len() == 1 {
            return self.forward_token(token_ids[0]);
        }

        // Process each token sequentially through all layers.
        // This populates the KV cache correctly and returns the last token's logits.
        // A true batched matmul prefill is a future optimization.
        let mut last_logits = Vec::new();
        for &token in token_ids {
            last_logits = self.forward_token(token);
        }
        last_logits
    }

    /// Reset the KV cache (start a new conversation).
    pub fn reset(&mut self) {
        self.cache.clear();
    }

    /// Current sequence position.
    pub fn position(&self) -> usize {
        self.cache.position()
    }

    /// KV cache memory usage in bytes.
    pub fn cache_memory_bytes(&self) -> usize {
        self.cache.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn tiny_model_with_arch(arch: Architecture) -> Model {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let vocab_size = 32;

        let (norm_type, ffn_type, pos_encoding, block_style, has_qkv_bias, has_output_bias, embedding_scale) =
            match arch {
                Architecture::Llama => (
                    NormType::RMSNorm, FFNType::SwiGLU,
                    PosEncoding::RoPE { freq_base: 10000.0 },
                    BlockStyle::Sequential, false, false, None,
                ),
                Architecture::GPT2 => (
                    NormType::LayerNorm, FFNType::GELU,
                    PosEncoding::Learned,
                    BlockStyle::Sequential, true, true, None,
                ),
                Architecture::Phi => (
                    NormType::LayerNorm, FFNType::SwiGLU,
                    PosEncoding::PartialRoPE { freq_base: 10000.0, rotary_dim: 2 },
                    BlockStyle::Parallel, true, true, None,
                ),
                Architecture::Gemma => (
                    NormType::RMSNorm, FFNType::GeGLU,
                    PosEncoding::RoPE { freq_base: 10000.0 },
                    BlockStyle::Sequential, false, false,
                    Some((hidden_dim as f32).sqrt()),
                ),
                Architecture::Qwen => (
                    NormType::RMSNorm, FFNType::SwiGLU,
                    PosEncoding::RoPE { freq_base: 10000.0 },
                    BlockStyle::Sequential, true, false, None,
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
                    hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
                )
            } else {
                BlockWeights::test_no_bias(
                    hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
                )
            }
        };

        let final_norm_bias = match norm_type {
            NormType::LayerNorm => Some(vec![0.0; hidden_dim]),
            NormType::RMSNorm => None,
        };

        let weights = ModelWeights {
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            layers: vec![make_layer(), make_layer()],
            final_norm: vec![1.0; hidden_dim],
            final_norm_bias,
            lm_head: vec![0.01; vocab_size * hidden_dim],
        };

        Model::new(config, weights)
    }

    fn tiny_model() -> Model {
        tiny_model_with_arch(Architecture::Llama)
    }

    #[test]
    fn test_forward_single_token() {
        let mut model = tiny_model();
        let logits = model.forward_token(0);
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(model.position(), 1);
    }

    #[test]
    fn test_forward_sequence() {
        let mut model = tiny_model();
        for token in [1, 5, 10, 3] {
            let logits = model.forward_token(token);
            assert_eq!(logits.len(), 32);
        }
        assert_eq!(model.position(), 4);
    }

    #[test]
    fn test_reset() {
        let mut model = tiny_model();
        model.forward_token(0);
        model.forward_token(1);
        assert_eq!(model.position(), 2);
        model.reset();
        assert_eq!(model.position(), 0);
    }

    #[test]
    fn test_gpt2_model_forward() {
        let mut model = tiny_model_with_arch(Architecture::GPT2);
        let logits = model.forward_token(0);
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_phi_model_forward() {
        let mut model = tiny_model_with_arch(Architecture::Phi);
        let logits = model.forward_token(0);
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gemma_model_forward() {
        let mut model = tiny_model_with_arch(Architecture::Gemma);
        let logits = model.forward_token(0);
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_qwen_model_forward() {
        let mut model = tiny_model_with_arch(Architecture::Qwen);
        let logits = model.forward_token(0);
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
            let mut model = tiny_model_with_arch(arch.clone());
            for token in [1, 5, 10, 3] {
                let logits = model.forward_token(token);
                assert_eq!(logits.len(), 32, "arch {:?}: wrong logits size", arch);
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "arch {:?}: non-finite logits",
                    arch,
                );
            }
            assert_eq!(model.position(), 4, "arch {:?}: wrong position", arch);
        }
    }
}
