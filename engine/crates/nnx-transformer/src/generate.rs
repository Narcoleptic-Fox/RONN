//! Autoregressive text generation loop.

use crate::model::Model;
use crate::sampler::{self, SamplerConfig};
use nnx_core::error::Result;

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Stop generation when this token ID is produced.
    pub eos_token_id: u32,
    /// Sampling configuration.
    pub sampler: SamplerConfig,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            eos_token_id: 2, // common default
            sampler: SamplerConfig::default(),
            seed: 42,
        }
    }
}

/// Result of a generation run.
#[derive(Debug)]
pub struct GenerateOutput {
    /// Generated token IDs (not including prompt).
    pub tokens: Vec<u32>,
    /// Reason generation stopped.
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EosToken,
    MaxTokens,
    ContextFull,
}

/// Generate tokens autoregressively.
///
/// Processes the prompt first (prefill), then generates tokens one at a time.
pub fn generate(
    model: &Model,
    cache: &mut crate::cache::KVCache,
    prompt_tokens: &[u32],
    config: &GenerateConfig,
) -> Result<GenerateOutput> {
    let mut rng_state = config.seed;
    let mut output_tokens = Vec::new();

    // Prefill: process all prompt tokens (builds KV cache)
    let mut last_logits = Vec::new();
    for &token in prompt_tokens {
        last_logits = model.forward_token(cache, token)?;
    }

    // Decode: generate tokens one at a time
    for _ in 0..config.max_tokens {
        if last_logits.is_empty() {
            break;
        }

        // Check context limit
        if cache.position() >= model.config.max_context_length - 1 {
            return Ok(GenerateOutput {
                tokens: output_tokens,
                stop_reason: StopReason::ContextFull,
            });
        }

        // Build history for repetition penalty (prompt + generated)
        let history: Vec<u32> = prompt_tokens
            .iter()
            .copied()
            .chain(output_tokens.iter().copied())
            .collect();

        // Sample next token
        let next_token = sampler::sample(&last_logits, &config.sampler, &history, &mut rng_state);

        // Check for EOS
        if next_token == config.eos_token_id {
            return Ok(GenerateOutput {
                tokens: output_tokens,
                stop_reason: StopReason::EosToken,
            });
        }

        output_tokens.push(next_token);

        // Forward the generated token
        last_logits = model.forward_token(cache, next_token)?;
    }

    Ok(GenerateOutput {
        tokens: output_tokens,
        stop_reason: StopReason::MaxTokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockWeights;
    use crate::config::*;
    use crate::model::{Model, ModelWeights};
    use crate::weights::Matrix;

    fn tiny_model() -> Model {
        let mut config = ModelConfig::test_llama(8, 2, 2, 4, 16, 32);
        config.num_layers = 1;

        let hd = 8;
        let weights = ModelWeights {
            token_embedding: Matrix::dense(vec![0.1; 32 * hd], 32, hd),
            layers: vec![BlockWeights::test_no_bias(hd, 2, 2, 4, 16)],
            final_norm: vec![1.0; hd],
            final_norm_bias: None,
            lm_head: Matrix::dense(vec![0.01; 32 * hd], 32, hd),
        };

        Model::new(config, weights)
    }

    #[test]
    fn test_generate_max_tokens() {
        let model = tiny_model();
        let mut cache = model.new_cache();
        let config = GenerateConfig {
            max_tokens: 5,
            eos_token_id: 999, // won't trigger
            sampler: SamplerConfig::greedy(),
            ..Default::default()
        };
        let result = generate(&model, &mut cache, &[1, 2, 3], &config).unwrap();
        assert_eq!(result.tokens.len(), 5);
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_generate_deterministic() {
        let config = GenerateConfig {
            max_tokens: 10,
            eos_token_id: 999,
            sampler: SamplerConfig::greedy(),
            seed: 42,
            ..Default::default()
        };

        let m1 = tiny_model();
        let mut c1 = m1.new_cache();
        let r1 = generate(&m1, &mut c1, &[1, 2], &config).unwrap();

        let m2 = tiny_model();
        let mut c2 = m2.new_cache();
        let r2 = generate(&m2, &mut c2, &[1, 2], &config).unwrap();

        assert_eq!(
            r1.tokens, r2.tokens,
            "greedy generation should be deterministic"
        );
    }
}
