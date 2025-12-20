//! Comprehensive tests for working memory functionality.

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_memory::working::WorkingMemory;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn test_working_memory_creation() {
    let memory = WorkingMemory::new(10);
    assert_eq!(memory.len(), 0);
    assert!(memory.is_empty());
}

#[test]
fn test_working_memory_store_and_retrieve() -> Result<()> {
    let mut memory = WorkingMemory::new(10);

    let data = vec![1.0f32, 2.0, 3.0];
    let tensor = Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;

    let id = memory.store(tensor.clone(), 0.8)?;

    let retrieved = memory.get(id);
    assert!(retrieved.is_some());

    Ok(())
}

#[test]
fn test_working_memory_capacity_eviction() -> Result<()> {
    let mut memory = WorkingMemory::new(5); // Small capacity

    // Store more than capacity
    for i in 0..10 {
        let data = vec![i as f32; 3];
        let tensor = Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;
        memory.store(tensor, 0.5)?;
    }

    // Should not exceed capacity
    assert!(memory.len() <= 5);

    Ok(())
}

#[test]
fn test_working_memory_importance_ordering() -> Result<()> {
    let mut memory = WorkingMemory::new(5);

    // Store items with different importance
    let low_id = memory.store(
        Tensor::from_data(
            vec![1.0f32; 3],
            vec![1, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?,
        0.2,
    )?;

    let high_id = memory.store(
        Tensor::from_data(
            vec![2.0f32; 3],
            vec![1, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?,
        0.9,
    )?;

    // Fill to capacity
    for i in 0..4 {
        memory.store(
            Tensor::from_data(
                vec![i as f32; 3],
                vec![1, 3],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
            0.5,
        )?;
    }

    // High importance should still be present
    assert!(memory.get(high_id).is_some());

    Ok(())
}

#[test]
fn test_working_memory_recency() -> Result<()> {
    let mut memory = WorkingMemory::new(10);

    let mut ids = Vec::new();
    for i in 0..5 {
        let data = vec![i as f32; 3];
        let tensor = Tensor::from_data(data, vec![1, 3], DataType::F32, TensorLayout::RowMajor)?;
        let id = memory.store(tensor, 0.5)?;
        ids.push(id);
    }

    // Access first item to update recency
    memory.get(ids[0]);

    // First item should now be more recent
    Ok(())
}
