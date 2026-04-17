//! Async wrapper around `nnx-serving`.
//!
//! The synchronous [`nnx_serving::backend::ServingEngine`] remains the source
//! of truth. This crate owns a background worker that drives `step()` and
//! exposes per-sequence token streams for server-style integration.

pub mod engine;
pub mod stream;

pub use engine::{AsyncServingEngine, RequestHandle};
pub use nnx_serving::sequence::{FinishReason, SequenceId};
pub use nnx_transformer::SamplerConfig;
pub use stream::{AsyncServingError, AsyncServingRequest, TokenOutput, TokenStream};
