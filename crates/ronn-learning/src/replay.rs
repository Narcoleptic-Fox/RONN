//! Experience Replay - Rehearsing important experiences

use crate::Result;
use ronn_core::tensor::Tensor;
use std::collections::VecDeque;

/// Replay sampling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayStrategy {
    /// Random sampling
    Random,
    /// Sample based on importance
    Importance,
    /// Sample recent experiences
    Recent,
}

/// An experience for replay
#[derive(Clone)]
pub struct Experience {
    pub input: Tensor,
    pub target: Tensor,
    pub importance: f64,
    pub timestamp: u64,
}

/// Experience replay buffer
pub struct ExperienceReplay {
    buffer: VecDeque<Experience>,
    capacity: usize,
    strategy: ReplayStrategy,
    stored_count: u64,
}

impl ExperienceReplay {
    /// Create new experience replay buffer
    pub fn new(capacity: usize, strategy: ReplayStrategy) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            strategy,
            stored_count: 0,
        }
    }

    /// Store an experience
    pub fn store(&mut self, experience: Experience) -> Result<()> {
        // If at capacity, remove oldest
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }

        self.buffer.push_back(experience);
        self.stored_count += 1;

        Ok(())
    }

    /// Sample experiences for replay
    pub fn sample(&self, batch_size: usize) -> Result<Vec<Experience>> {
        let actual_size = batch_size.min(self.buffer.len());

        match self.strategy {
            ReplayStrategy::Random => self.sample_random(actual_size),
            ReplayStrategy::Importance => self.sample_importance(actual_size),
            ReplayStrategy::Recent => self.sample_recent(actual_size),
        }
    }

    /// Sample randomly
    fn sample_random(&self, size: usize) -> Result<Vec<Experience>> {
        // For MVP: Return first N items
        // In production: Use actual random sampling
        Ok(self.buffer.iter().take(size).cloned().collect())
    }

    /// Sample by importance
    fn sample_importance(&self, size: usize) -> Result<Vec<Experience>> {
        // Sort by importance and take top N
        let mut sorted: Vec<_> = self.buffer.iter().collect();
        sorted.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

        Ok(sorted.into_iter().take(size).cloned().collect())
    }

    /// Sample recent experiences
    fn sample_recent(&self, size: usize) -> Result<Vec<Experience>> {
        // Take most recent N
        Ok(self.buffer.iter().rev().take(size).cloned().collect())
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get total experiences stored (including evicted)
    pub fn total_stored(&self) -> u64 {
        self.stored_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

    #[test]
    fn test_replay_creation() {
        let replay = ExperienceReplay::new(100, ReplayStrategy::Random);
        assert_eq!(replay.capacity, 100);
        assert_eq!(replay.len(), 0);
    }

    #[test]
    fn test_store_experience() -> Result<()> {
        let mut replay = ExperienceReplay::new(10, ReplayStrategy::Random);

        let input = Tensor::from_data(
            vec![1.0f32],
            vec![1, 1],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let target = Tensor::from_data(
            vec![0.5f32],
            vec![1, 1],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let exp = Experience {
            input,
            target,
            importance: 0.8,
            timestamp: 1000,
        };

        replay.store(exp)?;

        assert_eq!(replay.len(), 1);
        assert_eq!(replay.total_stored(), 1);

        Ok(())
    }

    #[test]
    fn test_capacity_eviction() -> Result<()> {
        let mut replay = ExperienceReplay::new(3, ReplayStrategy::Random);

        // Add 5 experiences (should evict oldest 2)
        for i in 0..5 {
            let input = Tensor::from_data(
                vec![i as f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;
            let target = Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;

            let exp = Experience {
                input,
                target,
                importance: 0.5,
                timestamp: i as u64,
            };

            replay.store(exp)?;
        }

        assert_eq!(replay.len(), 3);
        assert_eq!(replay.total_stored(), 5);

        Ok(())
    }

    #[test]
    fn test_importance_sampling() -> Result<()> {
        let mut replay = ExperienceReplay::new(10, ReplayStrategy::Importance);

        // Add experiences with varying importance
        for i in 0..5 {
            let input = Tensor::from_data(
                vec![i as f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;
            let target = Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;

            let exp = Experience {
                input,
                target,
                importance: 0.1 * (i as f64 + 1.0), // Increasing importance
                timestamp: i as u64,
            };

            replay.store(exp)?;
        }

        // Sample should prefer high importance
        let sampled = replay.sample(2)?;

        assert_eq!(sampled.len(), 2);
        // Most important should be sampled first
        assert!(sampled[0].importance >= sampled[1].importance);

        Ok(())
    }

    #[test]
    fn test_recent_sampling() -> Result<()> {
        let mut replay = ExperienceReplay::new(10, ReplayStrategy::Recent);

        // Add experiences
        for i in 0..5 {
            let input = Tensor::from_data(
                vec![i as f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;
            let target = Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;

            let exp = Experience {
                input,
                target,
                importance: 0.5,
                timestamp: i as u64,
            };

            replay.store(exp)?;
        }

        // Sample recent
        let sampled = replay.sample(2)?;

        assert_eq!(sampled.len(), 2);
        // Most recent should be sampled first
        assert!(sampled[0].timestamp >= sampled[1].timestamp);

        Ok(())
    }
}
