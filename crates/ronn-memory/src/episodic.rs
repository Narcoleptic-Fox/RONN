//! Episodic Memory - Experience storage with temporal indexing

use crate::{MemoryId, Result};
use ronn_core::tensor::Tensor;
use std::collections::HashMap;

/// An episode (experience) in memory
#[derive(Clone)]
pub struct Episode {
    pub id: MemoryId,
    pub data: Tensor,
    pub timestamp: u64,
    pub importance: f64,
}

/// Query parameters for episodic memory
#[derive(Debug, Clone)]
pub struct EpisodeQuery {
    pub start_time: Option<u64>,
    pub end_time: Option<u64>,
    pub min_importance: Option<f64>,
    pub limit: usize,
}

impl Default for EpisodeQuery {
    fn default() -> Self {
        Self {
            start_time: None,
            end_time: None,
            min_importance: None,
            limit: 10,
        }
    }
}

/// Episodic memory storage
pub struct EpisodicMemory {
    episodes: HashMap<MemoryId, Episode>,
    temporal_index: Vec<(u64, MemoryId)>, // (timestamp, id) for temporal queries
}

impl EpisodicMemory {
    /// Create new episodic memory
    pub fn new() -> Self {
        Self {
            episodes: HashMap::new(),
            temporal_index: Vec::new(),
        }
    }

    /// Store an episode
    pub fn store_episode(&mut self, episode: Episode) -> Result<()> {
        let id = episode.id;
        let timestamp = episode.timestamp;

        self.episodes.insert(id, episode);

        // Add to temporal index
        self.temporal_index.push((timestamp, id));

        // Keep temporal index sorted
        self.temporal_index.sort_by_key(|(ts, _)| *ts);

        Ok(())
    }

    /// Get episode by ID
    pub fn get_episode(&self, id: MemoryId) -> Option<Episode> {
        self.episodes.get(&id).cloned()
    }

    /// Query episodes by criteria
    pub fn query(&self, query: &EpisodeQuery) -> Vec<Episode> {
        self.episodes
            .values()
            .filter(|ep| {
                // Filter by time range
                if let Some(start) = query.start_time {
                    if ep.timestamp < start {
                        return false;
                    }
                }
                if let Some(end) = query.end_time {
                    if ep.timestamp > end {
                        return false;
                    }
                }

                // Filter by importance
                if let Some(min_imp) = query.min_importance {
                    if ep.importance < min_imp {
                        return false;
                    }
                }

                true
            })
            .take(query.limit)
            .cloned()
            .collect()
    }

    /// Get number of episodes
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Get all episodes (for consolidation)
    pub fn all_episodes(&self) -> Vec<&Episode> {
        self.episodes.values().collect()
    }
}

impl Default for EpisodicMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};
    use crate::current_timestamp;

    #[test]
    fn test_store_and_retrieve() -> Result<()> {
        let mut em = EpisodicMemory::new();

        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;

        let episode = Episode {
            id: 1,
            data: tensor,
            timestamp: current_timestamp(),
            importance: 0.8,
        };

        em.store_episode(episode.clone())?;

        let retrieved = em.get_episode(1);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 1);

        Ok(())
    }

    #[test]
    fn test_temporal_query() -> Result<()> {
        let mut em = EpisodicMemory::new();
        let base_time = current_timestamp();

        // Store episodes at different times
        for i in 0..5 {
            let data = vec![i as f32; 2];
            let tensor = Tensor::from_data(data, vec![1, 2], DataType::F32, TensorLayout::RowMajor)?;

            let episode = Episode {
                id: i as MemoryId,
                data: tensor,
                timestamp: base_time + (i * 1000),
                importance: 0.5 + (i as f64 * 0.1),
            };

            em.store_episode(episode)?;
        }

        // Query for episodes
        let query = EpisodeQuery {
            min_importance: Some(0.7),
            ..Default::default()
        };

        let results = em.query(&query);
        assert!(results.len() >= 2);

        Ok(())
    }
}
