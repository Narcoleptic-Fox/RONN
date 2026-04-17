//! Paged inference backend.
//!
//! [`ServingEngine`] wraps a loaded model and provides paged-memory
//! inference with continuous batching and prefix caching.

use crate::block_manager::BlockAllocator;
use crate::config::ServingConfig;
use crate::paged_cache::{PagedLayerView, SequencePageTable};
use crate::prefix_cache::{compute_hash_chain, PrefixCache};
use crate::scheduler::{Scheduler, SchedulerOutput, SchedulerStats};
use crate::sequence::{FinishReason, SequenceId};

use nnx_core::device::Device;
use nnx_core::error::{EngineError, Result};
#[cfg(feature = "gpu")]
use nnx_cubecl::{GpuPagePool, GpuPagedKvQuantConfig};
use nnx_transformer::block;
use nnx_transformer::config::{ModelConfig, NormType};
#[cfg(feature = "gpu")]
use nnx_transformer::gpu::GpuModel;
use nnx_transformer::model::{Model, ModelWeights};

/// Output from one step of serving: per-sequence logits.
#[derive(Debug)]
pub struct StepOutput {
    /// Sequence ID.
    pub seq_id: SequenceId,
    /// Logits for the last token [vocab_size].
    pub logits: Vec<f32>,
}

/// Output from one serving iteration.
#[derive(Debug)]
pub struct IterationOutput {
    /// Per-sequence outputs for this iteration.
    pub outputs: Vec<StepOutput>,
    /// Sequences that finished this iteration.
    pub finished: Vec<(SequenceId, FinishReason)>,
    /// Number of KV pages freed this iteration.
    pub pages_freed: usize,
    /// Scheduler state after this iteration.
    pub stats: SchedulerStats,
}

/// The paged serving engine.
///
/// Manages a loaded model with paged KV cache, continuous batching scheduler,
/// and prefix caching. Processes multiple concurrent sequences.
pub struct ServingEngine {
    /// The loaded model (weights + config).
    model: Model,
    /// Execution device for forward passes.
    device: Device,
    /// Page allocator for all sequences.
    allocator: BlockAllocator,
    /// Continuous batching scheduler.
    scheduler: Scheduler,
    /// Prefix cache.
    prefix_cache: PrefixCache,
    /// Serving configuration.
    config: ServingConfig,
    /// Optional GPU-resident model used when `device` is GPU.
    #[cfg(feature = "gpu")]
    gpu_model: Option<GpuModel>,
    /// Allocator-backed GPU page storage used as the serving-time source of truth.
    #[cfg(feature = "gpu")]
    gpu_page_pool: Option<GpuPagePool>,
}

impl ServingEngine {
    /// Create a new serving engine for a loaded model.
    pub fn new(model: Model, config: ServingConfig) -> Result<Self> {
        Self::new_with_device(model, config, Device::Cpu)
    }

    /// Create a new serving engine targeting a specific execution device.
    pub fn new_with_device(model: Model, config: ServingConfig, device: Device) -> Result<Self> {
        config
            .validate()
            .map_err(|e| EngineError::Serving(format!("invalid serving config: {}", e)))?;

        let num_layers = model.config.num_layers;
        let num_kv_heads = model.config.num_kv_heads;
        let head_dim = model.config.head_dim;

        // Calculate max pages if auto.
        let max_pages = if config.max_pages > 0 {
            config.max_pages
        } else {
            // Default: enough for max_sequences * 2048 tokens across all layers.
            let tokens_per_seq = 2048;
            let pages_per_seq_per_layer =
                (tokens_per_seq + config.page_size - 1) / config.page_size;
            pages_per_seq_per_layer * num_layers * config.max_sequences
        };

        let allocator = BlockAllocator::new(max_pages, config.page_size, num_kv_heads, head_dim);
        let scheduler = Scheduler::new(config.clone(), num_layers);
        let prefix_cache = PrefixCache::new(config.enable_prefix_caching);

        if matches!(device, Device::Unified(_)) {
            return Err(EngineError::Device(
                "Unified memory devices are not yet supported by ServingEngine".into(),
            ));
        }

        #[cfg(feature = "gpu")]
        let gpu_model = if device.is_gpu() {
            Some(
                GpuModel::from_cpu_model(&model)
                    .map_err(|e| EngineError::Device(format!("GPU upload failed: {}", e)))?,
            )
        } else {
            None
        };

        #[cfg(feature = "gpu")]
        let gpu_page_pool = gpu_model
            .as_ref()
            .map(|gpu_model| {
                let quant_config = config.gpu_kv_quantization.enabled.then_some(
                    GpuPagedKvQuantConfig {
                        residual_sketch_dim: config.gpu_kv_quantization.residual_sketch_dim,
                    },
                );
                gpu_model.new_page_pool(max_pages, config.page_size, quant_config)
            });

        #[cfg(not(feature = "gpu"))]
        if device.is_gpu() {
            return Err(EngineError::Device(
                "GPU serving requires the 'gpu' feature to be enabled".into(),
            ));
        }

        Ok(Self {
            model,
            device,
            allocator,
            scheduler,
            prefix_cache,
            config,
            #[cfg(feature = "gpu")]
            gpu_model,
            #[cfg(feature = "gpu")]
            gpu_page_pool,
        })
    }

    /// Add a new request to the serving engine.
    ///
    /// If prefix caching is enabled, the engine will look up the prompt
    /// in the prefix cache and share pages for any matching prefix,
    /// skipping prefill for that portion.
    ///
    /// Empty prompts are rejected because the serving engine does not yet
    /// have model-specific BOS bootstrapping.
    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_new_tokens: usize) -> SequenceId {
        self.try_add_request(prompt_tokens, max_new_tokens)
            .expect("invalid serving request")
    }

    /// Try to add a new request to the serving engine.
    pub fn try_add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<SequenceId> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Serving(
                "empty prompts are not supported; prepend a BOS token or provide at least one token"
                    .into(),
            ));
        }

        // Query prefix cache before admission.
        let mut lookup = self
            .prefix_cache
            .lookup(&prompt_tokens, self.config.page_size);

        // A page-aligned full cache hit cannot currently be served correctly
        // because decode expects a pending token to append. Recompute the last
        // full page so the engine produces next-token logits from a normal
        // prefill path instead of replaying the final cached token.
        if lookup.cached_tokens == prompt_tokens.len() && lookup.cached_pages > 0 {
            lookup.cached_pages -= 1;
            lookup.cached_tokens = lookup.cached_pages * self.config.page_size;
            lookup.matched_pages.truncate(lookup.cached_pages);
        }

        let seq_id = self
            .scheduler
            .add_request(prompt_tokens.clone(), max_new_tokens);

        if lookup.cached_pages > 0 {
            // Pre-populate the sequence's page table with shared pages.
            if let Some(seq) = self.scheduler.get_sequence_mut(seq_id) {
                for page_idx in 0..lookup.cached_pages {
                    let layer_page_ids = &lookup.matched_pages[page_idx];
                    for (layer_idx, &page_id) in layer_page_ids.iter().enumerate() {
                        // Increment ref count for the shared page.
                        let _ = self.allocator.inc_ref(page_id);
                        seq.page_table.layer_pages_mut(layer_idx).push(page_id);
                    }
                }
                // Advance the sequence's token count and prefill tracker.
                seq.page_table.set_num_tokens(lookup.cached_tokens);
                seq.prefilled_tokens = lookup.cached_tokens;
            }
        }

        Ok(seq_id)
    }

    /// Cancel a request. Frees any allocated KV pages.
    pub fn cancel_request(&mut self, id: SequenceId) -> Result<()> {
        let pages = self
            .scheduler
            .cancel_request(id)
            .map_err(|e| EngineError::Serving(format!("cancel failed: {}", e)))?;
        let _ = self.allocator.free_sequence_pages(&pages);
        Ok(())
    }

    /// Run one iteration of the serving loop.
    ///
    /// This is the core method. Each call:
    /// 1. Asks the scheduler what to do.
    /// 2. Runs prefill for newly admitted sequences.
    /// 3. Runs decode (one token) for active sequences.
    /// 4. Returns per-sequence logits for this iteration.
    ///
    /// The caller is responsible for:
    /// - Sampling tokens from the returned logits.
    /// - Calling [`on_token_generated`] with the sampled tokens.
    /// - Calling [`step`] again for the next iteration.
    pub fn step(&mut self) -> Result<IterationOutput> {
        let sched_output = self.scheduler.step(&self.allocator);
        let mut outputs = Vec::new();

        // Process prefill sequences.
        for (seq_id, tokens_to_prefill) in &sched_output.prefill {
            let logits = self.run_prefill(*seq_id, *tokens_to_prefill)?;
            outputs.push(StepOutput {
                seq_id: *seq_id,
                logits,
            });
            self.scheduler
                .on_prefill_advanced(*seq_id, *tokens_to_prefill);

            // If prefill is now complete, cache the computed pages.
            if let Some(seq) = self.scheduler.get_sequence(*seq_id) {
                if matches!(seq.state, crate::sequence::SequenceState::Decoding) {
                    self.cache_new_pages(*seq_id);
                }
            }
        }

        // Process decode sequences.
        for seq_id in &sched_output.decode {
            let logits = self.run_decode(*seq_id)?;
            outputs.push(StepOutput {
                seq_id: *seq_id,
                logits,
            });
        }

        // Free KV pages for finished sequences.
        let mut pages_freed = 0;
        let mut finished_clean = Vec::new();
        for (seq_id, reason, page_ids) in sched_output.finished {
            pages_freed += page_ids.len();
            let _ = self.allocator.free_sequence_pages(&page_ids);
            finished_clean.push((seq_id, reason));
        }

        Ok(IterationOutput {
            outputs,
            finished: finished_clean,
            pages_freed,
            stats: self.scheduler.stats(),
        })
    }

    /// Notify the engine that a token was generated for a sequence.
    ///
    /// Call this after sampling from the logits returned by [`step`].
    pub fn on_token_generated(&mut self, seq_id: SequenceId, token_id: u32, is_eos: bool) {
        self.scheduler.on_token_generated(seq_id, token_id, is_eos);
    }

    /// Whether there are any sequences still active or waiting.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        self.scheduler.stats()
    }

    /// Get allocator statistics.
    pub fn allocator_stats(&self) -> crate::block_manager::AllocatorStats {
        self.allocator.stats()
    }

    /// Access the model config.
    pub fn model_config(&self) -> &ModelConfig {
        &self.model.config
    }

    /// Current serving device.
    pub fn device(&self) -> Device {
        self.device
    }

    // ------------------------------------------------------------------
    // Internal forward pass implementations
    // ------------------------------------------------------------------

    /// Run prefill for a sequence: process `num_tokens` prompt tokens.
    fn run_prefill(&mut self, seq_id: SequenceId, num_tokens: usize) -> Result<Vec<f32>> {
        let seq = self
            .scheduler
            .get_sequence(seq_id)
            .ok_or_else(|| EngineError::Serving(format!("sequence {:?} not found", seq_id)))?;

        let start_idx = seq.prefilled_tokens;
        let token_ids = seq.prompt_tokens[start_idx..start_idx + num_tokens].to_vec();
        let cfg = &self.model.config;

        #[cfg(feature = "gpu")]
        if self.device.is_gpu() {
            return self.run_prefill_gpu(seq_id, &token_ids);
        }

        if num_tokens == 1 {
            // Single-token prefill: use decode path.
            return self.forward_single_token_cpu(seq_id, token_ids[0]);
        }

        // Batch prefill.
        let batch_size = num_tokens;
        let mut hidden_batch = vec![0.0f32; batch_size * cfg.hidden_dim];

        for (i, &token_id) in token_ids.iter().enumerate() {
            let dst = &mut hidden_batch[i * cfg.hidden_dim..(i + 1) * cfg.hidden_dim];
            self.model
                .weights
                .token_embedding
                .copy_row_to(token_id as usize, dst);
            if let Some(scale) = cfg.embedding_scale {
                for v in dst.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Get the sequence's page table for layer-by-layer processing.
        // We need to work around the borrow checker: extract page table info,
        // process each layer, then write back.
        let seq = self.scheduler.get_sequence(seq_id).unwrap();
        let initial_tokens = seq.page_table.num_tokens();
        let num_layers = cfg.num_layers;

        // Process each layer. We create a PagedLayerView per layer,
        // which borrows the allocator and one layer's page table.
        let mut final_token_count = initial_tokens;
        for layer_idx in 0..num_layers {
            let start_position = initial_tokens; // position of first new token
            let seq = self.scheduler.get_sequence_mut(seq_id).unwrap();
            let page_table = seq.page_table.layer_pages_mut(layer_idx);

            let mut view = PagedLayerView::new(
                &mut self.allocator,
                page_table,
                initial_tokens,
                layer_idx,
                self.config.page_size,
                cfg.num_kv_heads,
                cfg.head_dim,
            );

            block::forward_block_batch(
                &mut hidden_batch,
                batch_size,
                &self.model.weights.layers[layer_idx],
                &mut view,
                start_position,
                cfg,
            )?;

            final_token_count = view.token_count();
        }

        // Update the sequence's token count.
        let seq = self.scheduler.get_sequence_mut(seq_id).unwrap();
        seq.page_table.set_num_tokens(final_token_count);

        // Final norm + lm_head on the LAST token.
        let last_hidden =
            &hidden_batch[(batch_size - 1) * cfg.hidden_dim..batch_size * cfg.hidden_dim];
        self.final_projection(last_hidden)
    }

    /// Run decode for a sequence: generate logits for the next token.
    fn run_decode(&mut self, seq_id: SequenceId) -> Result<Vec<f32>> {
        let seq = self.scheduler.get_sequence(seq_id).ok_or_else(|| {
            EngineError::Serving(format!("sequence {:?} not found for decode", seq_id))
        })?;

        // The token to process is the last generated token, or if none yet,
        // the last prompt token (should not happen — decode follows prefill).
        let token_id = seq
            .generated_tokens
            .last()
            .copied()
            .or_else(|| seq.prompt_tokens.last().copied())
            .ok_or_else(|| EngineError::Serving("no token available for decode".into()))?;

        #[cfg(feature = "gpu")]
        if self.device.is_gpu() {
            return self.forward_single_token_gpu(seq_id, token_id);
        }

        self.forward_single_token_cpu(seq_id, token_id)
    }

    /// Forward pass for a single token through all layers using paged cache.
    fn forward_single_token_cpu(&mut self, seq_id: SequenceId, token_id: u32) -> Result<Vec<f32>> {
        let cfg = &self.model.config;

        // 1. Embedding.
        let mut hidden = vec![0.0f32; cfg.hidden_dim];
        self.model
            .weights
            .token_embedding
            .copy_row_to(token_id as usize, &mut hidden);
        if let Some(scale) = cfg.embedding_scale {
            for v in &mut hidden {
                *v *= scale;
            }
        }

        // 2. Layer-by-layer forward with paged cache.
        let seq = self.scheduler.get_sequence(seq_id).unwrap();
        let initial_tokens = seq.page_table.num_tokens();
        let position = initial_tokens;
        let num_layers = cfg.num_layers;

        let mut final_token_count = initial_tokens;
        for layer_idx in 0..num_layers {
            let seq = self.scheduler.get_sequence_mut(seq_id).unwrap();
            let page_table = seq.page_table.layer_pages_mut(layer_idx);

            let mut view = PagedLayerView::new(
                &mut self.allocator,
                page_table,
                initial_tokens,
                layer_idx,
                self.config.page_size,
                cfg.num_kv_heads,
                cfg.head_dim,
            );

            block::forward_block(
                &mut hidden,
                &self.model.weights.layers[layer_idx],
                &mut view,
                position,
                cfg,
            )?;

            final_token_count = view.token_count();
        }

        // 3. Update sequence token count.
        let seq = self.scheduler.get_sequence_mut(seq_id).unwrap();
        seq.page_table.set_num_tokens(final_token_count);

        // 4. Final norm + lm_head.
        self.final_projection(&hidden)
    }

    /// Insert newly computed complete pages into the prefix cache.
    ///
    /// Called after prefill completes. Only complete (full) pages are cached.
    fn cache_new_pages(&mut self, seq_id: SequenceId) {
        if !self.config.enable_prefix_caching {
            return;
        }

        let seq = match self.scheduler.get_sequence(seq_id) {
            Some(s) => s,
            None => return,
        };

        let page_size = self.config.page_size;
        let num_layers = self.model.config.num_layers;
        let chain = compute_hash_chain(&seq.prompt_tokens, page_size);

        // Insert each complete page that was just computed.
        for (page_idx, (hash, chunk)) in chain.iter().enumerate() {
            if chunk.len() < page_size {
                break; // partial page — not cacheable
            }

            // Gather the page IDs for this page index across all layers.
            let mut all_layer_ids = Vec::with_capacity(num_layers);
            let mut valid = true;
            for layer_idx in 0..num_layers {
                let layer_pages = seq.page_table.layer_pages(layer_idx);
                if page_idx < layer_pages.len() {
                    all_layer_ids.push(layer_pages[page_idx]);
                } else {
                    valid = false;
                    break;
                }
            }

            if valid {
                // Only increment ref counts if the page is actually new to the cache.
                // Duplicates are a no-op touch — incrementing refs for them would leak.
                let inserted =
                    self.prefix_cache
                        .insert(*hash, all_layer_ids.clone(), chunk.len(), page_size);
                if inserted {
                    for &pid in &all_layer_ids {
                        let _ = self.allocator.inc_ref(pid);
                    }
                }
            }
        }

        // Evict LRU entries if the cache exceeds the configured capacity.
        let max_entries = self.config.max_prefix_cache_entries;
        if max_entries > 0 {
            while self.prefix_cache.len() > max_entries {
                if let Some(evicted_pages) = self.prefix_cache.evict_lru() {
                    let _ = self.allocator.free_sequence_pages(&evicted_pages);
                } else {
                    break;
                }
            }
        }
    }

    /// Apply final normalization and LM head projection.
    fn final_projection(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.model.config;
        let mut normed = vec![0.0f32; cfg.hidden_dim];

        match cfg.norm_type {
            NormType::RMSNorm => {
                nnx_kernels::rms_norm::rms_norm_f32_checked(
                    hidden,
                    &self.model.weights.final_norm,
                    &mut normed,
                    cfg.rms_norm_eps,
                )?;
            }
            NormType::LayerNorm => {
                nnx_kernels::normalization::layer_norm_f32_checked(
                    hidden,
                    &self.model.weights.final_norm,
                    self.model.weights.final_norm_bias.as_deref(),
                    &mut normed,
                    cfg.hidden_dim,
                    cfg.rms_norm_eps,
                )?;
            }
        }

        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.model.weights.lm_head.matvec(&normed, &mut logits);
        Ok(logits)
    }

    /// Ensure the sequence has enough allocator-owned pages for an upcoming GPU write.
    fn ensure_sequence_pages(&mut self, seq_id: SequenceId, tokens_added: usize) -> Result<()> {
        if tokens_added == 0 {
            return Ok(());
        }

        let (existing_pages, final_pages) = {
            let seq = self
                .scheduler
                .get_sequence(seq_id)
                .ok_or_else(|| EngineError::Serving(format!("sequence {:?} not found", seq_id)))?;
            let final_tokens = seq.page_table.num_tokens() + tokens_added;
            (
                seq.page_table.pages_per_layer(),
                final_tokens.div_ceil(self.config.page_size),
            )
        };
        let new_pages_per_layer = final_pages.saturating_sub(existing_pages);

        if new_pages_per_layer > 0 {
            let mut reserved_pages: Vec<Vec<nnx_core::PageId>> = (0..self.model.config.num_layers)
                .map(|_| Vec::with_capacity(new_pages_per_layer))
                .collect();

            for layer_idx in 0..self.model.config.num_layers {
                for _ in 0..new_pages_per_layer {
                    match self.allocator.allocate() {
                        Ok(page_id) => reserved_pages[layer_idx].push(page_id),
                        Err(e) => {
                            let reserved: Vec<_> = reserved_pages
                                .iter()
                                .flat_map(|pages| pages.iter().copied())
                                .collect();
                            if let Err(rollback_err) = self.allocator.free_sequence_pages(&reserved) {
                                return Err(EngineError::Cache(format!(
                                    "failed to allocate page for sequence {:?}, layer {}: {}; rollback also failed: {}",
                                    seq_id, layer_idx, e, rollback_err
                                )));
                            }
                            return Err(EngineError::Cache(format!(
                                "failed to allocate page for sequence {:?}, layer {}: {}",
                                seq_id, layer_idx, e
                            )));
                        }
                    }
                }
            }

            let seq = self.scheduler.get_sequence_mut(seq_id).ok_or_else(|| {
                EngineError::Serving(format!("sequence {:?} not found", seq_id))
            });

            match seq {
                Ok(seq) => {
                    for (layer_idx, pages) in reserved_pages.iter_mut().enumerate() {
                        seq.page_table.layer_pages_mut(layer_idx).append(pages);
                    }

                    let new_pages: Vec<_> = reserved_pages
                        .iter()
                        .flat_map(|pages| pages.iter().copied())
                        .collect();

                    let _ = seq;

                    #[cfg(feature = "gpu")]
                    if let (Some(pool), Some(gpu_model)) =
                        (self.gpu_page_pool.as_mut(), self.gpu_model.as_ref())
                    {
                        gpu_model.mark_pages_dense(pool, &new_pages);
                    }
                }
                Err(err) => {
                    let reserved: Vec<_> = reserved_pages
                        .iter()
                        .flat_map(|pages| pages.iter().copied())
                        .collect();
                    if let Err(rollback_err) = self.allocator.free_sequence_pages(&reserved) {
                        return Err(EngineError::Cache(format!(
                            "{}; rollback also failed: {}",
                            err, rollback_err
                        )));
                    }
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    /// Advance the logical paged token count after a successful GPU forward pass.
    fn advance_sequence_tokens(&mut self, seq_id: SequenceId, tokens_added: usize) -> Result<()> {
        if tokens_added == 0 {
            return Ok(());
        }

        let seq = self
            .scheduler
            .get_sequence_mut(seq_id)
            .ok_or_else(|| EngineError::Serving(format!("sequence {:?} not found", seq_id)))?;
        let final_tokens = seq.page_table.num_tokens() + tokens_added;
        seq.page_table.set_num_tokens(final_tokens);
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn run_prefill_gpu(&mut self, seq_id: SequenceId, token_ids: &[u32]) -> Result<Vec<f32>> {
        self.ensure_sequence_pages(seq_id, token_ids.len())?;

        let logits = {
            let gpu_model = self
                .gpu_model
                .as_ref()
                .ok_or_else(|| EngineError::Device("GPU model is not initialized".into()))?;
            let pool = self
                .gpu_page_pool
                .as_ref()
                .ok_or_else(|| EngineError::Device("GPU page pool is not initialized".into()))?;
            let seq = self
                .scheduler
                .get_sequence(seq_id)
                .ok_or_else(|| EngineError::Serving(format!("sequence {:?} not found", seq_id)))?;
            let current_tokens = seq.page_table.num_tokens();
            let page_tables: Vec<_> = (0..self.model.config.num_layers)
                .map(|layer_idx| seq.page_table.layer_pages(layer_idx))
                .collect();

            if token_ids.len() == 1 {
                gpu_model.forward_token_paged(pool, &page_tables, current_tokens, token_ids[0])
            } else {
                gpu_model.forward_batch_paged(pool, &page_tables, current_tokens, token_ids)
            }
        };

        self.advance_sequence_tokens(seq_id, token_ids.len())?;

        if let (Some(gpu_model), Some(pool)) = (self.gpu_model.as_ref(), self.gpu_page_pool.as_mut()) {
            let seq = self
                .scheduler
                .get_sequence(seq_id)
                .ok_or_else(|| EngineError::Serving(format!("sequence {:?} not found", seq_id)))?;
            let page_tables: Vec<_> = (0..self.model.config.num_layers)
                .map(|layer_idx| seq.page_table.layer_pages(layer_idx))
                .collect();
            gpu_model.quantize_completed_pages(pool, &page_tables, seq.page_table.num_tokens());
        }

        Ok(logits)
    }

    #[cfg(feature = "gpu")]
    fn forward_single_token_gpu(&mut self, seq_id: SequenceId, token_id: u32) -> Result<Vec<f32>> {
        self.ensure_sequence_pages(seq_id, 1)?;

        let logits = {
            let gpu_model = self
                .gpu_model
                .as_ref()
                .ok_or_else(|| EngineError::Device("GPU model is not initialized".into()))?;
            let pool = self
                .gpu_page_pool
                .as_ref()
                .ok_or_else(|| EngineError::Device("GPU page pool is not initialized".into()))?;
            let seq = self
                .scheduler
                .get_sequence(seq_id)
                .ok_or_else(|| EngineError::Serving(format!("sequence {:?} not found", seq_id)))?;
            let current_tokens = seq.page_table.num_tokens();
            let page_tables: Vec<_> = (0..self.model.config.num_layers)
                .map(|layer_idx| seq.page_table.layer_pages(layer_idx))
                .collect();
            gpu_model.forward_token_paged(pool, &page_tables, current_tokens, token_id)
        };

        self.advance_sequence_tokens(seq_id, 1)?;

        if let (Some(gpu_model), Some(pool)) = (self.gpu_model.as_ref(), self.gpu_page_pool.as_mut()) {
            let seq = self
                .scheduler
                .get_sequence(seq_id)
                .ok_or_else(|| EngineError::Serving(format!("sequence {:?} not found", seq_id)))?;
            let page_tables: Vec<_> = (0..self.model.config.num_layers)
                .map(|layer_idx| seq.page_table.layer_pages(layer_idx))
                .collect();
            gpu_model.quantize_completed_pages(pool, &page_tables, seq.page_table.num_tokens());
        }

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnx_transformer::config::*;
    use nnx_transformer::weights::Matrix;

    /// Build a tiny synthetic model for testing.
    fn make_tiny_model(num_layers: usize) -> Model {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;
        let vocab_size = 32;

        let config = ModelConfig {
            architecture: "test".into(),
            arch: Architecture::Llama,
            num_layers,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
            max_context_length: 128,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            norm_type: NormType::RMSNorm,
            ffn_type: FFNType::SwiGLU,
            pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
            block_style: BlockStyle::Sequential,
            has_qkv_bias: false,
            has_output_bias: false,
            embedding_scale: None,
            activation_quantization: ActivationQuantization::None,
        };

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let make_layer = |seed: usize| block::BlockWeights {
            attn_norm: vec![1.0; hidden_dim],
            ffn_norm: vec![1.0; hidden_dim],
            wq: Matrix::dense(
                (0..q_dim * hidden_dim)
                    .map(|i| ((i + seed) % 7) as f32 * 0.02 - 0.06)
                    .collect(),
                q_dim,
                hidden_dim,
            ),
            wk: Matrix::dense(
                (0..kv_dim * hidden_dim)
                    .map(|i| ((i + seed) % 11) as f32 * 0.02 - 0.1)
                    .collect(),
                kv_dim,
                hidden_dim,
            ),
            wv: Matrix::dense(
                (0..kv_dim * hidden_dim)
                    .map(|i| ((i + seed) % 13) as f32 * 0.02 - 0.12)
                    .collect(),
                kv_dim,
                hidden_dim,
            ),
            wo: Matrix::dense(
                (0..hidden_dim * q_dim)
                    .map(|i| ((i + seed) % 5) as f32 * 0.02 - 0.04)
                    .collect(),
                hidden_dim,
                q_dim,
            ),
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
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            attn_norm_bias: None,
            ffn_norm_bias: None,
        };

        let layers: Vec<_> = (0..num_layers).map(|i| make_layer(i * 17)).collect();

        let weights = ModelWeights {
            token_embedding: Matrix::dense(
                (0..vocab_size * hidden_dim)
                    .map(|i| (i % 19) as f32 * 0.05 - 0.45)
                    .collect(),
                vocab_size,
                hidden_dim,
            ),
            position_embedding: None,
            layers,
            final_norm: vec![1.0; hidden_dim],
            final_norm_bias: None,
            lm_head: Matrix::dense(
                (0..vocab_size * hidden_dim)
                    .map(|i| (i % 23) as f32 * 0.03 - 0.33)
                    .collect(),
                vocab_size,
                hidden_dim,
            ),
        };

        Model::new(config, weights)
    }

    fn make_serving_config() -> ServingConfig {
        ServingConfig {
            page_size: 4,
            max_pages: 256,
            max_sequences: 8,
            max_batch_size: 4,
            enable_prefix_caching: false,
            max_prefix_cache_entries: 64,
            max_prefill_tokens: 0,
            gpu_kv_quantization: Default::default(),
        }
    }

    #[test]
    fn single_sequence_prefill_decode() {
        let model = make_tiny_model(2);
        let config = make_serving_config();
        let mut engine = ServingEngine::new(model, config).unwrap();

        // Add a request.
        let seq_id = engine.add_request(vec![1, 2, 3], 2);

        // Step 1: should prefill.
        let output = engine.step().unwrap();
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].seq_id, seq_id);
        assert_eq!(output.outputs[0].logits.len(), 32); // vocab_size
        assert!(output.outputs[0].logits.iter().all(|v| v.is_finite()));

        // Simulate token generation.
        engine.on_token_generated(seq_id, 10, false);

        // Step 2: should decode.
        let output = engine.step().unwrap();
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].logits.len(), 32);
        assert!(output.outputs[0].logits.iter().all(|v| v.is_finite()));

        engine.on_token_generated(seq_id, 11, false);

        // Step 3: should finish (max_new_tokens=2).
        let output = engine.step().unwrap();
        assert_eq!(output.finished.len(), 1);
        assert_eq!(output.finished[0].0, seq_id);
        assert_eq!(output.finished[0].1, FinishReason::MaxTokens);
    }

    #[test]
    fn multiple_sequences_concurrent() {
        let model = make_tiny_model(2);
        let config = make_serving_config();
        let mut engine = ServingEngine::new(model, config).unwrap();

        let id1 = engine.add_request(vec![1, 2], 1);
        let id2 = engine.add_request(vec![3, 4, 5], 1);

        // Step 1: both should prefill.
        let output = engine.step().unwrap();
        assert_eq!(output.outputs.len(), 2);

        // Generate tokens for both.
        engine.on_token_generated(id1, 10, false);
        engine.on_token_generated(id2, 20, false);

        // Step 2: both should finish (max_new_tokens=1).
        let output = engine.step().unwrap();
        assert_eq!(output.finished.len(), 2);
        assert!(!engine.has_work());
    }

    #[test]
    fn paged_produces_finite_logits_multilayer() {
        // Verify that a 4-layer model produces finite logits at every step.
        let model = make_tiny_model(4);
        let config = make_serving_config();
        let mut engine = ServingEngine::new(model, config).unwrap();

        engine.add_request(vec![1, 2, 3, 4, 5], 3);

        // Prefill.
        let output = engine.step().unwrap();
        assert_eq!(output.outputs.len(), 1);
        let logits = &output.outputs[0].logits;
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "prefill logits not finite"
        );

        // 3 decode steps.
        for i in 0..3 {
            let seq_id = output.outputs[0].seq_id;
            engine.on_token_generated(seq_id, (10 + i) as u32, false);
            let output = engine.step().unwrap();
            if !output.outputs.is_empty() {
                assert!(
                    output.outputs[0].logits.iter().all(|v| v.is_finite()),
                    "decode step {} logits not finite",
                    i
                );
            }
        }
    }

    #[test]
    fn ensure_sequence_pages_rolls_back_on_allocation_failure() {
        let model = make_tiny_model(2);
        let mut config = make_serving_config();
        config.page_size = 2;
        config.max_pages = 4;
        let mut engine = ServingEngine::new(model, config).unwrap();

        let seq_id = engine.add_request(vec![1, 2], 1);

        engine.ensure_sequence_pages(seq_id, 2).unwrap();
        engine.advance_sequence_tokens(seq_id, 2).unwrap();

        let pages_before = engine.allocator_stats().used_pages;
        let page_tables_before: Vec<Vec<_>> = {
            let seq = engine.scheduler.get_sequence(seq_id).unwrap();
            (0..engine.model.config.num_layers)
                .map(|layer_idx| seq.page_table.layer_pages(layer_idx).to_vec())
                .collect()
        };

        let err = engine.ensure_sequence_pages(seq_id, 7).unwrap_err();
        assert!(
            matches!(&err, EngineError::Cache(message) if message.contains("failed to allocate page")),
            "unexpected error: {err}"
        );

        let pages_after = engine.allocator_stats().used_pages;
        let seq = engine.scheduler.get_sequence(seq_id).unwrap();
        let page_tables_after: Vec<Vec<_>> = (0..engine.model.config.num_layers)
            .map(|layer_idx| seq.page_table.layer_pages(layer_idx).to_vec())
            .collect();

        assert_eq!(pages_after, pages_before, "allocator pages should roll back");
        assert_eq!(
            page_tables_after, page_tables_before,
            "page tables should remain unchanged after allocation failure"
        );
        assert_eq!(
            seq.page_table.num_tokens(),
            2,
            "token count should remain unchanged after allocation failure"
        );
    }

    #[test]
    fn eos_stops_generation() {
        let model = make_tiny_model(2);
        let config = make_serving_config();
        let mut engine = ServingEngine::new(model, config).unwrap();

        let id = engine.add_request(vec![1], 100);

        engine.step().unwrap(); // prefill
        engine.on_token_generated(id, 5, true); // EOS

        let output = engine.step().unwrap();
        assert_eq!(output.finished.len(), 1);
        assert_eq!(output.finished[0].1, FinishReason::EndOfSequence);
    }

    #[test]
    fn prefix_cache_shares_pages() {
        let model = make_tiny_model(2);
        let mut config = make_serving_config();
        config.enable_prefix_caching = true;
        config.page_size = 4; // 4 tokens per page

        let mut engine = ServingEngine::new(model, config).unwrap();

        // First request with prompt [1,2,3,4,5,6,7,8].
        // After prefill, pages for [1,2,3,4] (complete) should be cached.
        let id1 = engine.add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        let pages_before = engine.allocator_stats().used_pages;

        engine.step().unwrap(); // prefill id1
        engine.on_token_generated(id1, 10, false);
        engine.step().unwrap(); // decode id1 → finishes (max_new_tokens=1)

        let pages_after_first = engine.allocator_stats().used_pages;
        // First request allocated pages across 2 layers.
        assert!(pages_after_first > 0, "should have allocated pages");

        // Second request with same prefix [1,2,3,4] then different suffix.
        // The first page (tokens 1,2,3,4) should be shared via prefix cache.
        let id2 = engine.add_request(vec![1, 2, 3, 4, 9, 10, 11, 12], 1);

        // Check that the sequence was pre-populated with cached tokens.
        let seq2 = engine.scheduler.get_sequence(id2).unwrap();
        assert_eq!(
            seq2.page_table.num_tokens(),
            4,
            "prefix cache should have pre-populated 4 tokens (1 full page)"
        );
        assert_eq!(
            seq2.prefilled_tokens, 4,
            "prefilled_tokens should skip the cached portion"
        );

        // The sequence should still have pages shared from the cache.
        // Each layer should have 1 shared page.
        for layer_idx in 0..2 {
            let pages = seq2.page_table.layer_pages(layer_idx);
            assert_eq!(
                pages.len(),
                1,
                "layer {} should have 1 cached page",
                layer_idx
            );
        }
    }

    #[test]
    fn prefix_cache_produces_correct_output() {
        // Verify that prefix-cached sequences produce the same logits
        // as non-cached sequences with the same prompt.
        let model = make_tiny_model(2);
        let mut config = make_serving_config();
        config.enable_prefix_caching = true;
        config.page_size = 4;

        // --- Run without prefix caching first ---
        let mut engine_no_cache = ServingEngine::new(make_tiny_model(2), {
            let mut c = make_serving_config();
            c.enable_prefix_caching = false;
            c
        })
        .unwrap();

        let prompt = vec![1, 2, 3, 4, 5];
        let id_no = engine_no_cache.add_request(prompt.clone(), 1);
        let out_no = engine_no_cache.step().unwrap();
        let logits_no_cache = out_no.outputs[0].logits.clone();

        // --- Run with prefix caching ---
        let mut engine_cache = ServingEngine::new(make_tiny_model(2), config).unwrap();

        // First request populates the cache.
        let id1 = engine_cache.add_request(vec![1, 2, 3, 4, 5], 1);
        engine_cache.step().unwrap();
        engine_cache.on_token_generated(id1, 10, false);
        engine_cache.step().unwrap(); // finish

        // Second request with same prompt should use cached prefix.
        let id2 = engine_cache.add_request(prompt, 1);
        let out_cached = engine_cache.step().unwrap();
        let logits_cached = out_cached.outputs[0].logits.clone();

        // Both should produce the same logits (the 5th token is the same).
        assert_eq!(logits_no_cache.len(), logits_cached.len());
        for (i, (a, b)) in logits_no_cache.iter().zip(logits_cached.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "logit mismatch at {}: no_cache={}, cached={}",
                i,
                a,
                b,
            );
        }
    }

    #[test]
    fn try_add_request_rejects_empty_prompt() {
        let model = make_tiny_model(2);
        let config = make_serving_config();
        let mut engine = ServingEngine::new(model, config).unwrap();

        let err = engine.try_add_request(Vec::new(), 1).unwrap_err();
        assert!(matches!(err, EngineError::Serving(_)));
    }

    #[test]
    fn prefix_cache_full_hit_recomputes_last_page_for_correctness() {
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let mut uncached = ServingEngine::new(make_tiny_model(2), {
            let mut config = make_serving_config();
            config.enable_prefix_caching = false;
            config.page_size = 4;
            config
        })
        .unwrap();

        let uncached_id = uncached.add_request(prompt.clone(), 1);
        let uncached_logits = uncached
            .step()
            .unwrap()
            .outputs
            .into_iter()
            .find(|out| out.seq_id == uncached_id)
            .unwrap()
            .logits;

        let mut cached = ServingEngine::new(make_tiny_model(2), {
            let mut config = make_serving_config();
            config.enable_prefix_caching = true;
            config.page_size = 4;
            config
        })
        .unwrap();

        let warm_id = cached.add_request(prompt.clone(), 1);
        cached.step().unwrap();
        cached.on_token_generated(warm_id, 10, false);
        cached.step().unwrap();

        let cached_id = cached.add_request(prompt.clone(), 1);
        let cached_seq = cached.scheduler.get_sequence(cached_id).unwrap();
        assert_eq!(cached_seq.prefilled_tokens, 4);
        assert_eq!(cached_seq.page_table.num_tokens(), 4);

        let cached_logits = cached
            .step()
            .unwrap()
            .outputs
            .into_iter()
            .find(|out| out.seq_id == cached_id)
            .unwrap()
            .logits;

        assert_eq!(uncached_logits.len(), cached_logits.len());
        for (index, (lhs, rhs)) in uncached_logits.iter().zip(cached_logits.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-5,
                "page-aligned prefix cache mismatch at {}: uncached={}, cached={}",
                index,
                lhs,
                rhs,
            );
        }
    }

    #[test]
    fn cancel_frees_pages() {
        let model = make_tiny_model(2);
        let config = make_serving_config();
        let mut engine = ServingEngine::new(model, config).unwrap();

        let free_before = engine.allocator_stats().free_pages;

        let id = engine.add_request(vec![1, 2, 3, 4], 10);
        engine.step().unwrap(); // prefill — allocates pages
        engine.on_token_generated(id, 10, false);
        engine.step().unwrap(); // decode — may allocate more pages

        let free_during = engine.allocator_stats().free_pages;
        assert!(
            free_during < free_before,
            "should have allocated pages: before={}, during={}",
            free_before,
            free_during,
        );

        // Cancel the running request.
        engine.cancel_request(id).unwrap();

        let free_after = engine.allocator_stats().free_pages;
        assert_eq!(
            free_after, free_before,
            "cancel should free all pages: before={}, after={}",
            free_before, free_after,
        );
    }

    #[test]
    fn finished_request_frees_pages() {
        let model = make_tiny_model(2);
        let config = make_serving_config();
        let mut engine = ServingEngine::new(model, config).unwrap();

        let free_before = engine.allocator_stats().free_pages;

        let id = engine.add_request(vec![1, 2], 1);
        engine.step().unwrap(); // prefill
        engine.on_token_generated(id, 10, false);
        // max_new_tokens=1 → auto-finish

        let output = engine.step().unwrap(); // retires
        assert_eq!(output.finished.len(), 1);
        assert!(output.pages_freed > 0, "should report freed pages");

        let free_after = engine.allocator_stats().free_pages;
        assert_eq!(
            free_after, free_before,
            "finished request should free all pages: before={}, after={}",
            free_before, free_after,
        );
    }

    #[test]
    fn prefix_cache_evicts_under_capacity() {
        let model = make_tiny_model(2);
        let mut config = make_serving_config();
        config.enable_prefix_caching = true;
        config.page_size = 4;
        config.max_prefix_cache_entries = 2; // very small cache

        let mut engine = ServingEngine::new(model, config).unwrap();

        // Run 4 requests with different 4-token prefixes (each fills 1 cache entry).
        for prompt_start in 0..4u32 {
            let prompt: Vec<u32> = (prompt_start * 4 + 1..=prompt_start * 4 + 4).collect();
            let id = engine.add_request(prompt, 1);
            engine.step().unwrap(); // prefill
            engine.on_token_generated(id, 99, false);
            engine.step().unwrap(); // retire
        }

        // Cache should be at capacity (2), not 4.
        // If eviction didn't work, all 4 entries would be cached and
        // their pages would be pinned forever.
        assert!(
            engine.allocator_stats().free_pages > 0,
            "prefix cache eviction should keep pages from being pinned forever"
        );
    }
}
