//! Integration tests for the Continual Learning Engine.

mod ewc_tests;
mod replay_tests;
mod timescales_tests;

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_learning::{ContinualLearningEngine, LearningConfig};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn test_engine_creation() {
    let engine = ContinualLearningEngine::new(LearningConfig::default());
    let stats = engine.stats();

    assert_eq!(stats.total_updates, 0);
    assert_eq!(stats.tasks_learned, 0);
    assert_eq!(stats.replay_count, 0);
}

#[test]
fn test_custom_config() {
    let config = LearningConfig {
        ewc_lambda: 100.0,
        replay_frequency: 10,
        replay_batch_size: 16,
        ..Default::default()
    };

    let engine = ContinualLearningEngine::new(config);
    let stats = engine.stats();

    assert_eq!(stats.total_updates, 0);
}

#[test]
fn test_single_learning_update() -> Result<()> {
    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0],
        vec![1, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let target = Tensor::from_data(
        vec![0.5f32, 0.5],
        vec![1, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = engine.learn(input, target, 0.8)?;

    assert_eq!(engine.stats().total_updates, 1);
    assert!(result.fast_weight_change >= 0.0);
    assert!(result.slow_weight_change >= 0.0);

    Ok(())
}

#[test]
fn test_multiple_learning_updates() -> Result<()> {
    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    for i in 0..10 {
        let input = Tensor::from_data(
            vec![(i as f32), (i + 1) as f32],
            vec![1, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let target = Tensor::from_data(
            vec![0.5f32],
            vec![1, 1],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        engine.learn(input, target, 0.5)?;
    }

    let stats = engine.stats();
    assert_eq!(stats.total_updates, 10);

    Ok(())
}

#[test]
fn test_task_consolidation() -> Result<()> {
    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    // Create task data
    let task_data = vec![
        (
            Tensor::from_data(
                vec![1.0f32, 2.0],
                vec![1, 2],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
            Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
        ),
        (
            Tensor::from_data(
                vec![2.0f32, 3.0],
                vec![1, 2],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
            Tensor::from_data(
                vec![0.7f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
        ),
    ];

    engine.consolidate_task(&task_data)?;

    assert_eq!(engine.stats().tasks_learned, 1);

    Ok(())
}

#[test]
fn test_replay_triggering() -> Result<()> {
    let config = LearningConfig {
        replay_frequency: 5, // Replay every 5 updates
        ..Default::default()
    };

    let mut engine = ContinualLearningEngine::new(config);

    // Perform updates to trigger replay
    for i in 0..10 {
        let input = Tensor::from_data(
            vec![i as f32, (i + 1) as f32],
            vec![1, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let target = Tensor::from_data(
            vec![0.5f32],
            vec![1, 1],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        engine.learn(input, target, 0.5)?;
    }

    let stats = engine.stats();
    // Should have triggered replay at least once
    assert!(stats.replay_count > 0);

    Ok(())
}

#[test]
fn test_learning_performance() -> Result<()> {
    use std::time::Instant;

    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    let input = Tensor::from_data(
        vec![1.0f32; 100],
        vec![1, 100],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let target = Tensor::from_data(
        vec![0.5f32; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let start = Instant::now();
    engine.learn(input, target, 0.5)?;
    let elapsed = start.elapsed();

    // Learning should be reasonably fast
    assert!(elapsed.as_millis() < 50, "Learning too slow: {:?}", elapsed);

    Ok(())
}

#[test]
fn test_reset_stats() -> Result<()> {
    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    let input = Tensor::from_data(
        vec![1.0f32, 2.0],
        vec![1, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let target = Tensor::from_data(
        vec![0.5f32],
        vec![1, 1],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    engine.learn(input, target, 0.5)?;
    assert_eq!(engine.stats().total_updates, 1);

    engine.reset_stats();
    assert_eq!(engine.stats().total_updates, 0);

    Ok(())
}
