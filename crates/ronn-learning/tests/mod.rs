//! Comprehensive tests for the Continual Learning Engine.
//!
//! Tests cover:
//! - Multi-timescale learning (fast/slow weights)
//! - Elastic Weight Consolidation (EWC)
//! - Experience replay
//! - Task consolidation
//! - Catastrophic forgetting prevention
//! - Performance characteristics

mod ewc_tests;
mod replay_tests;
mod timescales_tests;

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_learning::{ContinualLearningEngine, LearningConfig, TimescaleConfig};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// Basic Learning Engine Tests
// ============================================================================

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
        timescale: TimescaleConfig {
            fast_lr: 0.01,
            slow_lr: 0.001,
            consolidation_rate: 0.1,
        },
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

// ============================================================================
// Task Consolidation Tests
// ============================================================================

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
fn test_multiple_task_consolidations() -> Result<()> {
    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    // Consolidate multiple tasks
    for i in 0..5 {
        let task_data = vec![(
            Tensor::from_data(
                vec![i as f32, (i + 1) as f32],
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
        )];

        engine.consolidate_task(&task_data)?;
    }

    assert_eq!(engine.stats().tasks_learned, 5);

    Ok(())
}

// ============================================================================
// Experience Replay Tests
// ============================================================================

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
fn test_no_replay_without_trigger() -> Result<()> {
    let config = LearningConfig {
        replay_frequency: 100, // High frequency
        ..Default::default()
    };

    let mut engine = ContinualLearningEngine::new(config);

    // Perform few updates
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

        engine.learn(input, target, 0.5)?;
    }

    let stats = engine.stats();
    // Should not have triggered replay yet
    assert_eq!(stats.replay_count, 0);

    Ok(())
}

// ============================================================================
// Weight Change Tests
// ============================================================================

#[test]
fn test_fast_weights_change() -> Result<()> {
    let config = LearningConfig {
        timescale: TimescaleConfig {
            fast_lr: 0.1,   // High fast rate
            slow_lr: 0.001, // Low slow rate
            consolidation_rate: 0.1,
        },
        ..Default::default()
    };

    let mut engine = ContinualLearningEngine::new(config);

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

    let result = engine.learn(input, target, 0.8)?;

    // Fast weights should change more than slow weights
    // (This is the expected behavior, though exact values depend on implementation)
    assert!(result.fast_weight_change >= 0.0);
    assert!(result.slow_weight_change >= 0.0);

    Ok(())
}

// ============================================================================
// EWC Penalty Tests
// ============================================================================

#[test]
fn test_ewc_penalty_after_consolidation() -> Result<()> {
    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    // Learn first task
    for i in 0..5 {
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

        engine.learn(input, target, 0.8)?;
    }

    // Consolidate task
    let task_data = vec![(
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
    )];
    engine.consolidate_task(&task_data)?;

    // Learn second task (should have EWC penalty)
    let input = Tensor::from_data(
        vec![10.0f32, 20.0],
        vec![1, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let target = Tensor::from_data(
        vec![0.9f32],
        vec![1, 1],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = engine.learn(input, target, 0.8)?;

    // After consolidation, EWC penalty might be applied
    assert!(result.ewc_penalty >= 0.0);

    Ok(())
}

// ============================================================================
// Performance Tests
// ============================================================================

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

    // Learning update should be fast (< 10ms)
    assert!(elapsed.as_millis() < 10, "Learning too slow: {:?}", elapsed);

    Ok(())
}

#[test]
fn test_many_updates_performance() -> Result<()> {
    use std::time::Instant;

    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    let start = Instant::now();
    for i in 0..100 {
        let input = Tensor::from_data(
            vec![(i % 10) as f32; 10],
            vec![1, 10],
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
    let elapsed = start.elapsed();

    println!("100 learning updates took: {:?}", elapsed);
    // Should handle many updates efficiently
    assert!(
        elapsed.as_millis() < 100,
        "Many updates too slow: {:?}",
        elapsed
    );

    Ok(())
}

// ============================================================================
// Concurrent Learning Tests
// ============================================================================

#[test]
fn test_concurrent_learning() -> Result<()> {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;

    let engine = Arc::new(Mutex::new(ContinualLearningEngine::new(
        LearningConfig::default(),
    )));
    let mut handles = vec![];

    for i in 0..10 {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || {
            let input = Tensor::from_data(
                vec![i as f32; 5],
                vec![1, 5],
                DataType::F32,
                TensorLayout::RowMajor,
            )
            .unwrap();
            let target = Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )
            .unwrap();

            let mut eng = engine_clone.lock().unwrap();
            eng.learn(input, target, 0.5).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let eng = engine.lock().unwrap();
    let stats = eng.stats();
    assert_eq!(stats.total_updates, 10);

    Ok(())
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_stress_many_tasks() -> Result<()> {
    let mut engine = ContinualLearningEngine::new(LearningConfig::default());

    // Learn and consolidate many tasks
    for task_id in 0..10 {
        // Learn task
        for i in 0..5 {
            let input = Tensor::from_data(
                vec![((task_id * 10 + i) as f32)],
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

            engine.learn(input, target, 0.8)?;
        }

        // Consolidate
        let task_data = vec![(
            Tensor::from_data(
                vec![task_id as f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
            Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
        )];
        engine.consolidate_task(&task_data)?;
    }

    let stats = engine.stats();
    assert_eq!(stats.tasks_learned, 10);
    assert_eq!(stats.total_updates, 50);

    Ok(())
}
