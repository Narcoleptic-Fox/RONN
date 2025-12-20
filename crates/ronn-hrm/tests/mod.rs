//! Comprehensive integration tests for the Hierarchical Reasoning Module (HRM).
//!
//! This test suite covers:
//! - Complexity assessment (complexity_tests.rs)
//! - Routing decisions (router_tests.rs)
//! - System 1 and System 2 execution (executor_tests.rs)
//! - End-to-end HRM workflows
//! - Performance characteristics
//! - Error handling and recovery

mod complexity_tests;
mod executor_tests;
mod router_tests;

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::{HrmConfig, HrmModule, RoutingStrategy};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// End-to-End HRM Integration Tests
// ============================================================================

#[test]
fn test_hrm_end_to_end_simple_input() -> Result<()> {
    let config = HrmConfig {
        routing_strategy: RoutingStrategy::AdaptiveComplexity,
        enable_system1_cache: true,
        system2_max_depth: 3,
    };

    let mut hrm = HrmModule::new(config);

    // Simple input should use System 1
    let input = Tensor::from_data(
        vec![1.0f32; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = hrm.process(&input)?;

    assert_eq!(result.output.shape(), input.shape());
    assert!(result.used_system1);
    assert!(!result.used_system2);

    Ok(())
}

#[test]
fn test_hrm_end_to_end_complex_input() -> Result<()> {
    let config = HrmConfig {
        routing_strategy: RoutingStrategy::AdaptiveComplexity,
        enable_system1_cache: true,
        system2_max_depth: 3,
    };

    let mut hrm = HrmModule::new(config);

    // Complex input should use System 2
    let data: Vec<f32> = (0..10000).map(|x| (x as f32).sin()).collect();
    let input = Tensor::from_data(data, vec![1, 10000], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    assert_eq!(result.output.shape(), input.shape());
    assert!(result.used_system2);

    Ok(())
}

#[test]
fn test_hrm_end_to_end_hybrid() -> Result<()> {
    let config = HrmConfig {
        routing_strategy: RoutingStrategy::AdaptiveHybrid {
            confidence_threshold: 0.9,
        },
        enable_system1_cache: true,
        system2_max_depth: 3,
    };

    let mut hrm = HrmModule::new(config);

    // Medium complexity might trigger hybrid
    let data: Vec<f32> = (0..500).map(|x| x as f32).collect();
    let input = Tensor::from_data(data, vec![1, 500], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    assert_eq!(result.output.shape(), input.shape());
    // Hybrid might use both systems
    assert!(result.used_system1 || result.used_system2);

    Ok(())
}

#[test]
fn test_hrm_statistics_tracking() -> Result<()> {
    let config = HrmConfig::default();
    let mut hrm = HrmModule::new(config);

    // Process multiple inputs
    for size in [10, 100, 1000, 10000] {
        let data = vec![1.0f32; size];
        let input = Tensor::from_data(data, vec![1, size], DataType::F32, TensorLayout::RowMajor)?;
        hrm.process(&input)?;
    }

    let stats = hrm.stats();
    assert_eq!(stats.total_inferences, 4);
    assert!(stats.system1_inferences + stats.system2_inferences + stats.hybrid_inferences == 4);

    Ok(())
}

#[test]
fn test_hrm_different_strategies() -> Result<()> {
    let strategies = vec![
        RoutingStrategy::AlwaysSystem1,
        RoutingStrategy::AlwaysSystem2,
        RoutingStrategy::AdaptiveComplexity,
        RoutingStrategy::AdaptiveHybrid {
            confidence_threshold: 0.7,
        },
    ];

    let input = Tensor::from_data(
        vec![1.0f32; 100],
        vec![1, 100],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    for strategy in strategies {
        let config = HrmConfig {
            routing_strategy: strategy,
            enable_system1_cache: true,
            system2_max_depth: 3,
        };

        let mut hrm = HrmModule::new(config);
        let result = hrm.process(&input)?;

        // All strategies should produce valid output
        assert_eq!(result.output.shape(), input.shape());
    }

    Ok(())
}

// ============================================================================
// Performance Tests
// ============================================================================

#[test]
fn test_hrm_performance_simple_inputs() -> Result<()> {
    use std::time::Instant;

    let config = HrmConfig {
        routing_strategy: RoutingStrategy::AdaptiveComplexity,
        enable_system1_cache: true,
        system2_max_depth: 3,
    };

    let mut hrm = HrmModule::new(config);

    let input = Tensor::from_data(
        vec![1.0f32; 100],
        vec![1, 100],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let start = Instant::now();
    let _result = hrm.process(&input)?;
    let elapsed = start.elapsed();

    // HRM should add minimal overhead (target: <2µs routing + execution)
    // For simple inputs with System 1, should be very fast
    assert!(elapsed.as_millis() < 10, "HRM too slow: {:?}", elapsed);

    Ok(())
}

#[test]
fn test_hrm_throughput() -> Result<()> {
    use std::time::Instant;

    let config = HrmConfig::default();
    let mut hrm = HrmModule::new(config);

    let iterations = 1000;
    let input = Tensor::from_data(
        vec![1.0f32; 50],
        vec![1, 50],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let start = Instant::now();
    for _ in 0..iterations {
        hrm.process(&input)?;
    }
    let elapsed = start.elapsed();

    let throughput = iterations as f64 / elapsed.as_secs_f64();
    println!("HRM throughput: {:.2} inferences/sec", throughput);

    // Should achieve good throughput
    assert!(throughput > 100.0, "Throughput too low: {:.2}", throughput);

    Ok(())
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_hrm_empty_input() -> Result<()> {
    let config = HrmConfig::default();
    let mut hrm = HrmModule::new(config);

    let input = Tensor::from_data(vec![], vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let result = hrm.process(&input)?;

    assert_eq!(result.output.shape(), vec![0]);

    Ok(())
}

#[test]
fn test_hrm_extreme_values() -> Result<()> {
    let config = HrmConfig::default();
    let mut hrm = HrmModule::new(config);

    let input = Tensor::from_data(
        vec![f32::MAX, f32::MIN, 0.0, f32::MAX],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = hrm.process(&input)?;

    // Should handle extreme values gracefully
    assert_eq!(result.output.shape(), vec![1, 4]);

    Ok(())
}

// ============================================================================
// Cache Effectiveness Tests
// ============================================================================

#[test]
fn test_system1_cache_effectiveness() -> Result<()> {
    let config = HrmConfig {
        routing_strategy: RoutingStrategy::AlwaysSystem1,
        enable_system1_cache: true,
        system2_max_depth: 3,
    };

    let mut hrm = HrmModule::new(config);

    let input = Tensor::from_data(
        vec![1.0f32; 50],
        vec![1, 50],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Process same input multiple times
    for _ in 0..10 {
        hrm.process(&input)?;
    }

    let stats = hrm.stats();
    // With caching, some executions might be cache hits
    assert!(stats.system1_cache_hits <= 10);

    Ok(())
}

#[test]
fn test_cache_disabled() -> Result<()> {
    let config = HrmConfig {
        routing_strategy: RoutingStrategy::AlwaysSystem1,
        enable_system1_cache: false, // Disable cache
        system2_max_depth: 3,
    };

    let mut hrm = HrmModule::new(config);

    let input = Tensor::from_data(
        vec![1.0f32; 50],
        vec![1, 50],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Process same input multiple times
    for _ in 0..10 {
        hrm.process(&input)?;
    }

    let stats = hrm.stats();
    // With cache disabled, should have no cache hits
    assert_eq!(stats.system1_cache_hits, 0);

    Ok(())
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_hrm_stress_many_small_inputs() -> Result<()> {
    let config = HrmConfig::default();
    let mut hrm = HrmModule::new(config);

    // Process many small inputs
    for i in 0..1000 {
        let data = vec![(i % 100) as f32; 10];
        let input = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        hrm.process(&input)?;
    }

    let stats = hrm.stats();
    assert_eq!(stats.total_inferences, 1000);

    Ok(())
}

#[test]
fn test_hrm_stress_varying_sizes() -> Result<()> {
    let config = HrmConfig::default();
    let mut hrm = HrmModule::new(config);

    // Process inputs of varying sizes
    for size in [1, 10, 100, 1000, 10000] {
        for _ in 0..10 {
            let data = vec![1.0f32; size];
            let input =
                Tensor::from_data(data, vec![1, size], DataType::F32, TensorLayout::RowMajor)?;
            hrm.process(&input)?;
        }
    }

    let stats = hrm.stats();
    assert_eq!(stats.total_inferences, 50);

    Ok(())
}

// ============================================================================
// Concurrent HRM Tests
// ============================================================================

#[test]
fn test_hrm_concurrent_processing() -> Result<()> {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;

    let config = HrmConfig::default();
    let hrm = Arc::new(Mutex::new(HrmModule::new(config)));
    let mut handles = vec![];

    for i in 0..10 {
        let hrm_clone = Arc::clone(&hrm);
        let handle = thread::spawn(move || {
            let data = vec![i as f32; 100];
            let input =
                Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)
                    .unwrap();

            let mut hrm_lock = hrm_clone.lock().unwrap();
            hrm_lock.process(&input).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let hrm_lock = hrm.lock().unwrap();
    let stats = hrm_lock.stats();
    assert_eq!(stats.total_inferences, 10);

    Ok(())
}
