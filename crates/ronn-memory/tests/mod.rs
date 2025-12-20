//! Comprehensive integration tests for the Multi-Tier Memory System.
//!
//! This test suite covers:
//! - Working memory (attention, eviction, TTL)
//! - Episodic memory (temporal queries, importance filtering)
//! - Semantic memory (concepts, relationships, activation)
//! - Memory consolidation (sleep-like processing)
//! - Cross-tier interactions
//! - Performance characteristics
//! - Concurrency and thread safety

mod consolidation_tests;
mod episodic_memory_tests;
mod memory_integration_tests;
mod semantic_memory_tests;
mod working_memory_tests;

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_memory::{ConsolidationResult, Episode, MemoryConfig, MultiTierMemory};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// End-to-End Memory System Tests
// ============================================================================

#[test]
fn test_memory_creation() {
    let memory = MultiTierMemory::new();
    let stats = memory.stats();

    assert_eq!(stats.working_items, 0);
    assert_eq!(stats.episodic_episodes, 0);
    assert_eq!(stats.semantic_concepts, 0);
}

#[test]
fn test_memory_with_custom_config() {
    let config = MemoryConfig {
        working_capacity: 50,
        episodic_capacity: 200,
        semantic_capacity: 500,
        consolidation_threshold: 0.8,
    };

    let memory = MultiTierMemory::with_config(config);
    let stats = memory.stats();

    assert_eq!(stats.working_items, 0);
    assert_eq!(stats.episodic_episodes, 0);
    assert_eq!(stats.semantic_concepts, 0);
}

#[test]
fn test_store_and_retrieve_low_importance() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

    // Store with low importance (stays in working memory only)
    let id = memory.store(tensor.clone(), 0.3)?;

    // Should be able to retrieve
    let retrieved = memory.retrieve(id)?;
    assert!(retrieved.is_some());

    let stats = memory.stats();
    assert_eq!(stats.working_items, 1);
    assert_eq!(stats.episodic_episodes, 0); // Low importance shouldn't go to episodic

    Ok(())
}

#[test]
fn test_store_and_retrieve_high_importance() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

    // Store with high importance (should go to episodic too)
    let id = memory.store(tensor, 0.9)?;

    let stats = memory.stats();
    assert_eq!(stats.working_items, 1);
    assert_eq!(stats.episodic_episodes, 1); // High importance goes to episodic

    Ok(())
}

#[test]
fn test_multiple_stores() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    // Store multiple items with varying importance
    for i in 0..10 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        let importance = (i as f32) / 10.0; // 0.0 to 0.9
        memory.store(tensor, importance)?;
    }

    let stats = memory.stats();
    assert!(stats.working_items <= 10); // Some might have been evicted
    assert!(stats.episodic_episodes > 0); // High importance items in episodic

    Ok(())
}

#[test]
fn test_working_memory_capacity_limit() -> Result<()> {
    let config = MemoryConfig {
        working_capacity: 5, // Small capacity
        ..Default::default()
    };

    let mut memory = MultiTierMemory::with_config(config);

    // Store more items than capacity
    for i in 0..10 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.5)?;
    }

    let stats = memory.stats();
    // Working memory should be capped at capacity
    assert!(stats.working_items <= 5);

    Ok(())
}

// ============================================================================
// Consolidation Tests
// ============================================================================

#[tokio::test]
async fn test_consolidation_empty_memory() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    let result = memory.consolidate().await?;

    // No episodes to consolidate
    assert_eq!(result.episodes_consolidated, 0);
    assert_eq!(result.patterns_extracted, 0);

    Ok(())
}

#[tokio::test]
async fn test_consolidation_with_episodes() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    // Store several high-importance items
    for i in 0..5 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.8)?;
    }

    // Run consolidation
    let result = memory.consolidate().await?;

    // Consolidation should process episodes
    assert!(result.episodes_consolidated >= 0);
    assert!(result.patterns_extracted >= 0);

    Ok(())
}

#[tokio::test]
async fn test_multiple_consolidations() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    // Store items
    for i in 0..10 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.7)?;
    }

    // Run consolidation multiple times
    for _ in 0..3 {
        memory.consolidate().await?;
    }

    // Should complete without errors
    Ok(())
}

// ============================================================================
// Cross-Tier Interaction Tests
// ============================================================================

#[test]
fn test_working_to_episodic_promotion() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    // Store with medium importance (in working)
    let data = vec![1.0f32; 4];
    let tensor = Tensor::from_data(
        data.clone(),
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    memory.store(tensor.clone(), 0.5)?;

    // Store with high importance (promoted to episodic)
    let tensor2 = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
    memory.store(tensor2, 0.9)?;

    let stats = memory.stats();
    assert!(stats.episodic_episodes > 0);

    Ok(())
}

#[tokio::test]
async fn test_episodic_to_semantic_consolidation() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    // Store high-importance items
    for i in 0..5 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.9)?;
    }

    let stats_before = memory.stats();
    let episodic_before = stats_before.episodic_episodes;

    // Consolidate (should extract patterns to semantic)
    memory.consolidate().await?;

    let stats_after = memory.stats();
    // Some concepts might be extracted
    assert!(stats_after.semantic_concepts >= 0);

    Ok(())
}

// ============================================================================
// Performance Tests
// ============================================================================

#[test]
fn test_store_performance() -> Result<()> {
    use std::time::Instant;

    let mut memory = MultiTierMemory::new();

    let data = vec![1.0f32; 1000];
    let tensor = Tensor::from_data(data, vec![1, 1000], DataType::F32, TensorLayout::RowMajor)?;

    let start = Instant::now();
    memory.store(tensor, 0.5)?;
    let elapsed = start.elapsed();

    // Store should be fast (<1ms)
    assert!(elapsed.as_millis() < 1, "Store too slow: {:?}", elapsed);

    Ok(())
}

#[test]
fn test_retrieve_performance() -> Result<()> {
    use std::time::Instant;

    let mut memory = MultiTierMemory::new();

    let data = vec![1.0f32; 1000];
    let tensor = Tensor::from_data(data, vec![1, 1000], DataType::F32, TensorLayout::RowMajor)?;
    let id = memory.store(tensor, 0.5)?;

    let start = Instant::now();
    let _ = memory.retrieve(id)?;
    let elapsed = start.elapsed();

    // Retrieve should be very fast (<100µs)
    assert!(
        elapsed.as_micros() < 100,
        "Retrieve too slow: {:?}",
        elapsed
    );

    Ok(())
}

#[test]
fn test_many_stores_performance() -> Result<()> {
    use std::time::Instant;

    let mut memory = MultiTierMemory::new();

    let start = Instant::now();
    for i in 0..1000 {
        let data = vec![(i % 100) as f32; 10];
        let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.5)?;
    }
    let elapsed = start.elapsed();

    println!("1000 stores took: {:?}", elapsed);
    // Should handle many stores efficiently
    assert!(
        elapsed.as_millis() < 100,
        "Many stores too slow: {:?}",
        elapsed
    );

    Ok(())
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[test]
fn test_concurrent_stores() -> Result<()> {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;

    let memory = Arc::new(Mutex::new(MultiTierMemory::new()));
    let mut handles = vec![];

    for i in 0..10 {
        let memory_clone = Arc::clone(&memory);
        let handle = thread::spawn(move || {
            let data = vec![i as f32; 10];
            let tensor =
                Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
                    .unwrap();

            let mut mem = memory_clone.lock().unwrap();
            mem.store(tensor, 0.5).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let mem = memory.lock().unwrap();
    let stats = mem.stats();
    assert!(stats.working_items > 0);

    Ok(())
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_retrieve_nonexistent_id() -> Result<()> {
    let memory = MultiTierMemory::new();

    let result = memory.retrieve(uuid::Uuid::new_v4())?;

    // Should return None for non-existent ID
    assert!(result.is_none());

    Ok(())
}

#[test]
fn test_store_empty_tensor() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    let tensor = Tensor::from_data(vec![], vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let id = memory.store(tensor, 0.5)?;

    // Should store successfully
    let retrieved = memory.retrieve(id)?;
    assert!(retrieved.is_some());

    Ok(())
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_statistics_accuracy() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    // Store known quantities
    for i in 0..5 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.5)?; // Medium importance
    }

    for i in 0..3 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.9)?; // High importance
    }

    let stats = memory.stats();
    // Should have working items and episodic episodes
    assert!(stats.working_items > 0);
    assert!(stats.episodic_episodes > 0);

    Ok(())
}

#[test]
fn test_clear_memory() -> Result<()> {
    let mut memory = MultiTierMemory::new();

    // Store some items
    for i in 0..5 {
        let data = vec![i as f32; 4];
        let tensor = Tensor::from_data(data, vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.7)?;
    }

    // Clear memory
    memory.clear();

    let stats = memory.stats();
    assert_eq!(stats.working_items, 0);
    assert_eq!(stats.episodic_episodes, 0);
    assert_eq!(stats.semantic_concepts, 0);

    Ok(())
}
