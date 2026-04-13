//! Full Llama-family model: embedding → N blocks → final norm → LM head.

use crate::block::{self, BlockWeights};
use crate::cache::KVCache;
use crate::config::ModelConfig;

/// All weights for a complete model.
pub struct ModelWeights {
    /// Token embedding table [vocab_size, hidden_dim]
    pub token_embedding: Vec<f32>,
    /// Per-layer transformer block weights.
    pub layers: Vec<BlockWeights>,
    /// Final RMSNorm weights [hidden_dim]
    pub final_norm: Vec<f32>,
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
    /// This is the decode step — processes one token using the KV cache.
    pub fn forward_token(&mut self, token_id: u32) -> Vec<f32> {
        let cfg = &self.config;
        let position = self.cache.position();

        // 1. Token embedding lookup
        let emb_offset = token_id as usize * cfg.hidden_dim;
        let mut hidden = self.weights.token_embedding
            [emb_offset..emb_offset + cfg.hidden_dim]
            .to_vec();

        // 2. Run through all transformer blocks
        for layer_idx in 0..cfg.num_layers {
            block::forward_block(
                &mut hidden,
                &self.weights.layers[layer_idx],
                self.cache.layer_mut(layer_idx),
                position,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.intermediate_dim,
                cfg.rope_freq_base,
                cfg.rms_norm_eps,
            );
        }

        // 3. Final RMSNorm
        let mut normed = vec![0.0f32; cfg.hidden_dim];
        nnx_kernels::rms_norm::rms_norm_f32(
            &hidden, &self.weights.final_norm, &mut normed, cfg.rms_norm_eps,
        );

        // 4. LM head: project to vocab
        let mut logits = vec![0.0f32; cfg.vocab_size];
        nnx_kernels::matmul::matvec_f32(
            &self.weights.lm_head, &normed, &mut logits,
            cfg.vocab_size, cfg.hidden_dim,
        );

        logits
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

    fn tiny_model() -> Model {
        let config = ModelConfig {
            architecture: "test".into(),
            num_layers: 2,
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_dim: 16,
            vocab_size: 32,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
        };

        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let nkv = config.num_kv_heads;
        let hdim = config.head_dim;
        let inter = config.intermediate_dim;
        let vocab = config.vocab_size;

        let make_layer = || BlockWeights {
            attn_norm: vec![1.0; hd],
            ffn_norm: vec![1.0; hd],
            wq: vec![0.01; nh * hdim * hd],
            wk: vec![0.01; nkv * hdim * hd],
            wv: vec![0.01; nkv * hdim * hd],
            wo: vec![0.01; hd * nh * hdim],
            w_gate: vec![0.01; inter * hd],
            w_up: vec![0.01; inter * hd],
            w_down: vec![0.01; hd * inter],
        };

        let weights = ModelWeights {
            token_embedding: vec![0.1; vocab * hd],
            layers: vec![make_layer(), make_layer()],
            final_norm: vec![1.0; hd],
            lm_head: vec![0.01; vocab * hd],
        };

        Model::new(config, weights)
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
}
