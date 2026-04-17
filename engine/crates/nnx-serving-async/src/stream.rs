use nnx_serving::sequence::{FinishReason, SequenceId};
use nnx_transformer::SamplerConfig;
use thiserror::Error;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Request submitted to [`crate::engine::AsyncServingEngine`].
#[derive(Debug, Clone)]
pub struct AsyncServingRequest {
    /// Prompt tokens to prefill before decode begins.
    pub prompt_tokens: Vec<u32>,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling configuration used for each decode step.
    pub sampler: SamplerConfig,
    /// Optional EOS token. When generated, the sequence is marked finished.
    pub eos_token_id: Option<u32>,
    /// RNG seed for deterministic sampling in tests or replay.
    pub seed: u64,
}

impl AsyncServingRequest {
    /// Build a request with explicit sampler configuration.
    pub fn new(prompt_tokens: Vec<u32>, max_new_tokens: usize, sampler: SamplerConfig) -> Self {
        Self {
            prompt_tokens,
            max_new_tokens,
            sampler,
            eos_token_id: None,
            seed: 0,
        }
    }
}

impl Default for AsyncServingRequest {
    fn default() -> Self {
        Self {
            prompt_tokens: Vec::new(),
            max_new_tokens: 16,
            sampler: SamplerConfig::default(),
            eos_token_id: None,
            seed: 0,
        }
    }
}

/// One streamed token emitted by the async wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenOutput {
    /// Sequence that produced this token.
    pub seq_id: SequenceId,
    /// Sampled token id.
    pub token_id: u32,
    /// Finish reason if this token completed the sequence.
    pub finish_reason: Option<FinishReason>,
}

/// Concrete stream type returned by [`crate::engine::AsyncServingEngine::stream`].
pub type TokenStream = UnboundedReceiverStream<TokenOutput>;

/// Errors produced by the async serving wrapper.
#[derive(Debug, Error)]
pub enum AsyncServingError {
    #[error("serving engine error: {0}")]
    Engine(String),
    #[error("background worker stopped")]
    WorkerStopped,
    #[error("stream for sequence {0:?} is not available")]
    StreamNotFound(SequenceId),
}

impl From<nnx_core::error::EngineError> for AsyncServingError {
    fn from(value: nnx_core::error::EngineError) -> Self {
        Self::Engine(value.to_string())
    }
}
