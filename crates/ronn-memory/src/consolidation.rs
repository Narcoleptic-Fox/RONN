//! Sleep Consolidation - Offline memory processing and pattern extraction

use crate::episodic::{Episode, EpisodicMemory};
use crate::semantic::Concept;
use crate::Result;
use std::collections::HashSet;

/// Configuration for sleep consolidation
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Minimum importance threshold for consolidation
    pub importance_threshold: f64,

    /// Maximum number of patterns to extract per cycle
    pub max_patterns: usize,

    /// Whether to enable pattern discovery
    pub pattern_discovery_enabled: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            importance_threshold: 0.5,
            max_patterns: 10,
            pattern_discovery_enabled: true,
        }
    }
}

/// Sleep consolidation engine for offline memory processing
pub struct SleepConsolidation {
    config: ConsolidationConfig,
    cycles_run: u64,
}

impl SleepConsolidation {
    /// Create new sleep consolidation engine
    pub fn new(config: ConsolidationConfig) -> Self {
        Self {
            config,
            cycles_run: 0,
        }
    }

    /// Extract patterns from episodic memory
    pub async fn extract_patterns(&mut self, episodic: &EpisodicMemory) -> Result<Vec<Concept>> {
        self.cycles_run += 1;

        if !self.config.pattern_discovery_enabled {
            return Ok(Vec::new());
        }

        let episodes = episodic.all_episodes();

        // Filter important episodes
        let important: Vec<&Episode> = episodes
            .into_iter()
            .filter(|ep| ep.importance >= self.config.importance_threshold)
            .collect();

        // For MVP: Create simple concepts from important episodes
        let mut patterns = Vec::new();

        for (idx, episode) in important.iter().enumerate().take(self.config.max_patterns) {
            let concept = Concept {
                id: episode.id + 10000, // Offset to avoid collisions
                name: format!("Pattern_{}", idx),
                activation: episode.importance,
                related_concepts: HashSet::new(),
            };

            patterns.push(concept);
        }

        Ok(patterns)
    }

    /// Assess importance of an episode for consolidation
    pub fn assess_importance(&self, episode: &Episode) -> f64 {
        // Combine multiple factors:
        // 1. Explicit importance score
        // 2. Recency (more recent = more important)
        // 3. In production: frequency, novelty, emotional salience

        let base_importance = episode.importance;

        // For MVP: just use the base importance
        base_importance
    }

    /// Run a full consolidation cycle
    pub async fn consolidate_cycle(
        &mut self,
        episodic: &EpisodicMemory,
    ) -> Result<ConsolidationStats> {
        let patterns = self.extract_patterns(episodic).await?;
        let episodes_processed = episodic.len();

        Ok(ConsolidationStats {
            episodes_processed,
            patterns_extracted: patterns.len(),
            cycle_number: self.cycles_run,
        })
    }

    /// Get number of consolidation cycles run
    pub fn cycles_run(&self) -> u64 {
        self.cycles_run
    }
}

/// Statistics from a consolidation cycle
#[derive(Debug, Clone)]
pub struct ConsolidationStats {
    pub episodes_processed: usize,
    pub patterns_extracted: usize,
    pub cycle_number: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use crate::current_timestamp;
    use ronn_core::tensor::Tensor;
    use ronn_core::types::{DataType, TensorLayout};

    #[tokio::test]
    async fn test_pattern_extraction() -> Result<()> {
        let mut consolidation = SleepConsolidation::new(ConsolidationConfig::default());
        let mut episodic = EpisodicMemory::new();

        // Add some episodes
        for i in 0..5 {
            let data = vec![i as f32; 3];
            let tensor =
                Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;

            let episode = Episode {
                id: i as MemoryId,
                data: tensor,
                timestamp: current_timestamp(),
                importance: 0.6 + (i as f64 * 0.1),
            };

            episodic.store_episode(episode)?;
        }

        // Extract patterns
        let patterns = consolidation.extract_patterns(&episodic).await?;

        assert!(!patterns.is_empty());
        assert_eq!(consolidation.cycles_run(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_full_cycle() -> Result<()> {
        let mut consolidation = SleepConsolidation::new(ConsolidationConfig::default());
        let mut episodic = EpisodicMemory::new();

        // Add episodes
        for i in 0..3 {
            let data = vec![i as f32; 2];
            let tensor =
                Tensor::from_data(data, vec![1, 2], DataType::F32, TensorLayout::RowMajor)?;

            let episode = Episode {
                id: i as MemoryId,
                data: tensor,
                timestamp: current_timestamp(),
                importance: 0.7,
            };

            episodic.store_episode(episode)?;
        }

        // Run consolidation cycle
        let stats = consolidation.consolidate_cycle(&episodic).await?;

        assert_eq!(stats.episodes_processed, 3);
        assert!(stats.patterns_extracted > 0);
        assert_eq!(stats.cycle_number, 1);

        Ok(())
    }

    #[test]
    fn test_importance_assessment() -> Result<()> {
        let consolidation = SleepConsolidation::new(ConsolidationConfig::default());

        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;

        let episode = Episode {
            id: 1,
            data: tensor,
            timestamp: current_timestamp(),
            importance: 0.8,
        };

        let assessed = consolidation.assess_importance(&episode);
        assert_eq!(assessed, 0.8);

        Ok(())
    }
}
