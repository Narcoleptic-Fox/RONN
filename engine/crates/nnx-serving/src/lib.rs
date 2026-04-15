//! NNX Serving — paged attention, continuous batching, and prefix caching.
//!
//! This crate provides the serving infrastructure that makes NNX competitive
//! as a multi-tenant inference runtime. It builds on the core execution layer
//! in `nnx-transformer` without modifying it.
//!
//! # Architecture
//!
//! ```text
//! Layer 3: Prefix Caching         (content-addressed page lookup, LRU eviction)
//! Layer 2: Continuous Batching     (scheduler, iteration-level batching)
//! Layer 1: Paged KV Memory         (block allocator, page tables, paged attention)
//! Layer 0: Existing NNX             (Model, KVCache, attention, InferenceEngine)
//! ```

pub mod backend;
pub mod block_manager;
pub mod config;
pub mod error;
pub mod page;
pub mod paged_cache;
pub mod prefix_cache;
pub mod scheduler;
pub mod sequence;
