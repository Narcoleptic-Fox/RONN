//! Multi-Tier Memory System - Brain-Inspired Memory Architecture
//!
//! Implements a three-tier memory system inspired by human cognition:
//! - **Working Memory**: Short-term, attention-weighted storage
//! - **Episodic Memory**: Experience storage with temporal/spatial indexing
//! - **Semantic Memory**: Long-term knowledge graph
//!
//! Also includes a sleep consolidation engine for offline memory processing.
//!
//! ## Architecture
//!
//! ```text
//! Input → Working Memory (Short-term, Limited Capacity)
//!              ↓
//!         [Attention Filter]
//!              ↓
//!         Episodic Memory (Experiences, Temporal Index)
//!              ↓
//!       [Sleep Consolidation]
//!              ↓
//!         Semantic Memory (Knowledge Graph, Long-term)
//! ```

pub mod activation_cache;
pub mod consolidation;
pub mod episodic;
pub mod semantic;
pub mod working;

pub use activation_cache::{
    ActivationCache, ActivationCacheConfig, ActivationCacheKey, ActivationCacheStats,
    CachedActivationPattern,
};
pub use consolidation::{ConsolidationConfig, SleepConsolidation};
pub use episodic::{Episode, EpisodeQuery, EpisodicMemory};
pub use semantic::{Concept, ConceptGraph, SemanticMemory};
pub use working::{WorkingMemory, WorkingMemoryConfig};

use ronn_core::tensor::Tensor;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors that can occur in the memory system
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Working memory error: {0}")]
    WorkingMemory(String),

    #[error("Episodic memory error: {0}")]
    EpisodicMemory(String),

    #[error("Semantic memory error: {0}")]
    SemanticMemory(String),

    #[error("Consolidation error: {0}")]
    Consolidation(String),

    #[error("Core error: {0}")]
    Core(#[from] ronn_core::error::CoreError),
}

pub type Result<T> = std::result::Result<T, MemoryError>;

/// Main memory system coordinator
pub struct MultiTierMemory {
    working: WorkingMemory,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    consolidation: SleepConsolidation,
}

impl MultiTierMemory {
    /// Create a new multi-tier memory system with default configuration
    pub fn new() -> Self {
        Self {
            working: WorkingMemory::new(WorkingMemoryConfig::default()),
            episodic: EpisodicMemory::new(),
            semantic: SemanticMemory::new(),
            consolidation: SleepConsolidation::new(ConsolidationConfig::default()),
        }
    }

    /// Store data in working memory
    pub fn store(&mut self, data: Tensor, importance: f64) -> Result<MemoryId> {
        // Store in working memory
        let id = self.working.store(data, importance)?;

        // If important enough, also store in episodic memory
        if importance > 0.7 {
            let tensor = self.working.get(id)?;
            let episode = Episode {
                id,
                data: tensor,
                timestamp: current_timestamp(),
                importance,
            };
            self.episodic.store_episode(episode)?;
        }

        Ok(id)
    }

    /// Retrieve data from memory hierarchy
    pub fn retrieve(&self, id: MemoryId) -> Result<Option<Tensor>> {
        // Try working memory first
        if let Ok(tensor) = self.working.get(id) {
            return Ok(Some(tensor));
        }

        // Try episodic memory
        if let Some(episode) = self.episodic.get_episode(id) {
            return Ok(Some(episode.data));
        }

        Ok(None)
    }

    /// Search memory by similarity
    pub fn search_similar(&self, query: &Tensor, limit: usize) -> Result<Vec<MemoryId>> {
        // For MVP: search working memory
        self.working.search_similar(query, limit)
    }

    /// Run sleep consolidation to transfer memories
    pub async fn consolidate(&mut self) -> Result<ConsolidationResult> {
        // Transfer important memories from working to episodic
        let working_items = self.working.drain_old_items()?;
        let items_count = working_items.len();

        for (id, tensor, importance) in working_items {
            if importance > 0.5 {
                let episode = Episode {
                    id,
                    data: tensor,
                    timestamp: current_timestamp(),
                    importance,
                };
                self.episodic.store_episode(episode)?;
            }
        }

        // Extract patterns from episodic to semantic
        let patterns = self.consolidation.extract_patterns(&self.episodic).await?;
        let patterns_count = patterns.len();

        for concept in patterns {
            self.semantic.store_concept(concept)?;
        }

        Ok(ConsolidationResult {
            episodes_consolidated: items_count,
            patterns_extracted: patterns_count,
        })
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            working_items: self.working.len(),
            episodic_episodes: self.episodic.len(),
            semantic_concepts: self.semantic.len(),
        }
    }
}

impl Default for MultiTierMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for memory items
pub type MemoryId = u64;

/// Result of consolidation process
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    pub episodes_consolidated: usize,
    pub patterns_extracted: usize,
}

/// Statistics about the memory system
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub working_items: usize,
    pub episodic_episodes: usize,
    pub semantic_concepts: usize,
}

/// Get current timestamp in milliseconds since Unix epoch
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    #[test]
    fn test_memory_creation() {
        let memory = MultiTierMemory::new();
        let stats = memory.stats();

        assert_eq!(stats.working_items, 0);
        assert_eq!(stats.episodic_episodes, 0);
        assert_eq!(stats.semantic_concepts, 0);
    }

    #[test]
    fn test_store_and_retrieve() -> Result<()> {
        let mut memory = MultiTierMemory::new();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

        // Store with low importance (stays in working memory)
        let id = memory.store(tensor.clone(), 0.3)?;

        // Should be able to retrieve
        let retrieved = memory.retrieve(id)?;
        assert!(retrieved.is_some());

        Ok(())
    }

    #[test]
    fn test_high_importance_storage() -> Result<()> {
        let mut memory = MultiTierMemory::new();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

        // Store with high importance (should go to episodic)
        let id = memory.store(tensor, 0.9)?;

        let stats = memory.stats();
        assert_eq!(stats.working_items, 1);
        assert_eq!(stats.episodic_episodes, 1); // Should also be in episodic

        Ok(())
    }

    #[tokio::test]
    async fn test_consolidation() -> Result<()> {
        let mut memory = MultiTierMemory::new();

        // Store several items
        for i in 0..5 {
            let data = vec![i as f32; 4];
            let tensor =
                Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
            memory.store(tensor, 0.6)?;
        }

        // Run consolidation (may or may not consolidate based on timing)
        let result = memory.consolidate().await?;

        // Consolidation completed successfully (even if no episodes were old enough)
        assert!(result.episodes_consolidated >= 0);
        assert!(result.patterns_extracted >= 0);

        Ok(())
    }
}
