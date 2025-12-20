//! Working Memory - Short-term storage with attention weighting

use crate::{MemoryError, MemoryId, Result, current_timestamp};
use ronn_core::tensor::Tensor;
use std::collections::{HashMap, VecDeque};

/// Configuration for working memory
#[derive(Debug, Clone)]
pub struct WorkingMemoryConfig {
    /// Maximum number of items in working memory
    pub capacity: usize,

    /// Time-to-live for items in milliseconds
    pub ttl_ms: u64,

    /// Whether to use attention weighting
    pub attention_enabled: bool,
}

impl Default for WorkingMemoryConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            ttl_ms: 60_000, // 1 minute
            attention_enabled: true,
        }
    }
}

/// An item in working memory
#[derive(Clone)]
pub struct WorkingMemoryItem {
    pub id: MemoryId,
    pub data: Tensor,
    pub importance: f64,
    pub timestamp: u64,
    pub access_count: u64,
}

/// Working memory with attention-weighted storage
pub struct WorkingMemory {
    config: WorkingMemoryConfig,
    items: HashMap<MemoryId, WorkingMemoryItem>,
    lru_order: VecDeque<MemoryId>,
    next_id: MemoryId,
}

impl WorkingMemory {
    /// Create new working memory with configuration
    pub fn new(config: WorkingMemoryConfig) -> Self {
        Self {
            config,
            items: HashMap::new(),
            lru_order: VecDeque::new(),
            next_id: 1,
        }
    }

    /// Store data with importance score
    pub fn store(&mut self, data: Tensor, importance: f64) -> Result<MemoryId> {
        // Evict old items if at capacity
        while self.items.len() >= self.config.capacity {
            self.evict_lru()?;
        }

        let id = self.next_id;
        self.next_id += 1;

        let item = WorkingMemoryItem {
            id,
            data,
            importance,
            timestamp: current_timestamp(),
            access_count: 0,
        };

        self.items.insert(id, item);
        self.lru_order.push_back(id);

        Ok(id)
    }

    /// Retrieve data by ID
    pub fn get(&self, id: MemoryId) -> Result<Tensor> {
        self.items
            .get(&id)
            .map(|item| item.data.clone())
            .ok_or_else(|| MemoryError::WorkingMemory(format!("Item {} not found", id)))
    }

    /// Search for similar items
    pub fn search_similar(&self, _query: &Tensor, limit: usize) -> Result<Vec<MemoryId>> {
        // For MVP: return most recent items
        Ok(self.lru_order.iter().rev().take(limit).copied().collect())
    }

    /// Drain old items for consolidation
    pub fn drain_old_items(&mut self) -> Result<Vec<(MemoryId, Tensor, f64)>> {
        let current_time = current_timestamp();
        let mut drained = Vec::new();

        let expired_ids: Vec<MemoryId> = self
            .items
            .iter()
            .filter(|(_, item)| current_time - item.timestamp > self.config.ttl_ms)
            .map(|(id, _)| *id)
            .collect();

        for id in expired_ids {
            if let Some(item) = self.items.remove(&id) {
                drained.push((item.id, item.data, item.importance));
                self.lru_order.retain(|&x| x != id);
            }
        }

        Ok(drained)
    }

    /// Evict least recently used item
    fn evict_lru(&mut self) -> Result<()> {
        if let Some(id) = self.lru_order.pop_front() {
            self.items.remove(&id);
        }
        Ok(())
    }

    /// Get number of items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

    #[test]
    fn test_store_and_retrieve() -> Result<()> {
        let mut wm = WorkingMemory::new(WorkingMemoryConfig::default());

        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;

        let id = wm.store(tensor.clone(), 0.5)?;
        let retrieved = wm.get(id)?;

        assert_eq!(retrieved.shape(), tensor.shape());
        assert_eq!(wm.len(), 1);

        Ok(())
    }

    #[test]
    fn test_capacity_eviction() -> Result<()> {
        let config = WorkingMemoryConfig {
            capacity: 3,
            ..Default::default()
        };
        let mut wm = WorkingMemory::new(config);

        // Add 4 items (should evict oldest)
        for i in 0..4 {
            let data = vec![i as f32; 2];
            let tensor =
                Tensor::from_data(data, vec![1, 2], DataType::F32, TensorLayout::RowMajor)?;
            wm.store(tensor, 0.5)?;
        }

        assert_eq!(wm.len(), 3);

        Ok(())
    }
}
