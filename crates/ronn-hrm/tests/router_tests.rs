//! Comprehensive tests for HRM routing decisions.
//!
//! Tests cover:
//! - All routing strategies (AlwaysSystem1, AlwaysSystem2, AdaptiveComplexity, AdaptiveHybrid)
//! - Routing decision correctness
//! - Confidence thresholds
//! - Hybrid execution logic
//! - Performance characteristics
//! - Statistics tracking

use ronn_core::types::{DataType, TensorLayout};
use ronn_core::Tensor;
use ronn_hrm::router::{HrmRouter, RoutingDecision, RoutingStrategy};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// Basic Routing Strategy Tests
// ============================================================================

#[test]
fn test_always_system1_strategy() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AlwaysSystem1);

    // Any input should route to System1
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    assert!(decision.use_system1);
    assert!(!decision.use_system2);
    assert!(decision.confidence > 0.0);

    Ok(())
}

#[test]
fn test_always_system2_strategy() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AlwaysSystem2);

    // Any input should route to System2
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    assert!(!decision.use_system1);
    assert!(decision.use_system2);
    assert!(decision.confidence > 0.0);

    Ok(())
}

#[test]
fn test_adaptive_complexity_low() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    // Small, simple tensor should route to System1
    let data = vec![1.0f32; 10];
    let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    assert!(decision.use_system1);
    assert!(!decision.use_system2);
    assert!(decision.confidence > 0.5);

    Ok(())
}

#[test]
fn test_adaptive_complexity_high() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    // Large tensor with high variance should route to System2
    let data: Vec<f32> = (0..10000)
        .map(|x| (x as f32).sin() * (x as f32).cos())
        .collect();
    let tensor = Tensor::from_data(data, vec![1, 10000], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // Should prefer System2 for complex input
    assert!(decision.use_system2);

    Ok(())
}

#[test]
fn test_adaptive_hybrid_medium() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveHybrid {
        confidence_threshold: 0.7,
    });

    // Medium complexity might trigger hybrid
    let data: Vec<f32> = (0..500).map(|x| x as f32).collect();
    let tensor = Tensor::from_data(data, vec![1, 500], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // Should make a decision (might be hybrid)
    assert!(decision.use_system1 || decision.use_system2);

    Ok(())
}

// ============================================================================
// Hybrid Execution Tests
// ============================================================================

#[test]
fn test_hybrid_execution_both_systems() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveHybrid {
        confidence_threshold: 0.9, // Very high threshold
    });

    // Medium complexity with moderate variance
    let data: Vec<f32> = (0..200).map(|x| (x as f32).sin()).collect();
    let tensor = Tensor::from_data(data, vec![1, 200], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // With high confidence threshold, might use both systems
    if decision.confidence < 0.9 {
        assert!(decision.use_system1 && decision.use_system2);
    }

    Ok(())
}

#[test]
fn test_hybrid_low_confidence() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveHybrid {
        confidence_threshold: 0.5,
    });

    // Create borderline case
    let data = vec![1.0f32; 100]; // Exactly at boundary
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // Should make a valid decision
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);

    Ok(())
}

// ============================================================================
// Confidence Threshold Tests
// ============================================================================

#[test]
fn test_confidence_threshold_low() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveHybrid {
        confidence_threshold: 0.1, // Very low threshold
    });

    let data = vec![1.0f32; 50];
    let tensor = Tensor::from_data(data, vec![1, 50], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // With low threshold, should rarely use hybrid
    // Either system1 OR system2, not both
    assert!(!(decision.use_system1 && decision.use_system2));

    Ok(())
}

#[test]
fn test_confidence_threshold_high() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveHybrid {
        confidence_threshold: 0.99, // Very high threshold
    });

    let data = vec![1.0f32; 50];
    let tensor = Tensor::from_data(data, vec![1, 50], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // With very high threshold, might use hybrid more often
    // But should still be a valid decision
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);

    Ok(())
}

// ============================================================================
// Routing Decision Properties
// ============================================================================

#[test]
fn test_decision_mutual_exclusion_simple() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![1.0f32; 50];
    let tensor = Tensor::from_data(data, vec![1, 50], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // AdaptiveComplexity should never use both systems
    assert!(!(decision.use_system1 && decision.use_system2));
    // But should use at least one
    assert!(decision.use_system1 || decision.use_system2);

    Ok(())
}

#[test]
fn test_decision_at_least_one_system() -> Result<()> {
    let strategies = vec![
        RoutingStrategy::AlwaysSystem1,
        RoutingStrategy::AlwaysSystem2,
        RoutingStrategy::AdaptiveComplexity,
        RoutingStrategy::AdaptiveHybrid {
            confidence_threshold: 0.7,
        },
    ];

    for strategy in strategies {
        let router = HrmRouter::new(strategy);
        let data = vec![1.0f32; 100];
        let tensor =
            Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

        let decision = router.route(&tensor)?;

        // Must use at least one system
        assert!(
            decision.use_system1 || decision.use_system2,
            "Must use at least one system"
        );
    }

    Ok(())
}

#[test]
fn test_decision_confidence_bounds() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // Confidence must be between 0 and 1
    assert!(decision.confidence >= 0.0);
    assert!(decision.confidence <= 1.0);

    Ok(())
}

// ============================================================================
// Statistics Tracking Tests
// ============================================================================

#[test]
fn test_statistics_tracking() -> Result<()> {
    let mut router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    // Initial stats should be zero
    let stats = router.stats();
    assert_eq!(stats.total_routes, 0);
    assert_eq!(stats.system1_routes, 0);
    assert_eq!(stats.system2_routes, 0);
    assert_eq!(stats.hybrid_routes, 0);

    // Route a simple tensor
    let data = vec![1.0f32; 10];
    let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
    router.route(&tensor)?;

    // Stats should increment
    let stats = router.stats();
    assert_eq!(stats.total_routes, 1);
    assert!(stats.system1_routes > 0 || stats.system2_routes > 0);

    Ok(())
}

#[test]
fn test_statistics_multiple_routes() -> Result<()> {
    let mut router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    // Route multiple tensors
    for size in [10, 100, 1000, 10000] {
        let data = vec![1.0f32; size];
        let tensor =
            Tensor::from_data(data, vec![1, size], DataType::F32, TensorLayout::RowMajor)?;
        router.route(&tensor)?;
    }

    let stats = router.stats();
    assert_eq!(stats.total_routes, 4);
    assert!(stats.system1_routes + stats.system2_routes + stats.hybrid_routes == 4);

    Ok(())
}

#[test]
fn test_statistics_hybrid_routes() -> Result<()> {
    let mut router = HrmRouter::new(RoutingStrategy::AdaptiveHybrid {
        confidence_threshold: 0.9,
    });

    // Route multiple medium-complexity tensors
    for i in 0..10 {
        let data: Vec<f32> = (0..100).map(|x| (x + i) as f32).collect();
        let tensor =
            Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;
        router.route(&tensor)?;
    }

    let stats = router.stats();
    assert_eq!(stats.total_routes, 10);
    // With hybrid strategy, some might be hybrid
    assert!(stats.system1_routes + stats.system2_routes + stats.hybrid_routes == 10);

    Ok(())
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_tensor_routing() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![];
    let tensor = Tensor::from_data(data, vec![0], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // Empty tensor should be low complexity -> System1
    assert!(decision.use_system1);
    assert!(!decision.use_system2);

    Ok(())
}

#[test]
fn test_single_element_routing() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![42.0f32];
    let tensor = Tensor::from_data(data, vec![1], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // Single element should be low complexity -> System1
    assert!(decision.use_system1);
    assert!(!decision.use_system2);

    Ok(())
}

#[test]
fn test_very_large_tensor_routing() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    // Very large tensor
    let data = vec![1.0f32; 1_000_000];
    let tensor = Tensor::from_data(
        data,
        vec![1, 1_000_000],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let decision = router.route(&tensor)?;

    // Very large should be high complexity -> System2
    assert!(decision.use_system2);

    Ok(())
}

#[test]
fn test_extreme_values_routing() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![f32::MAX, f32::MIN, 0.0, f32::MAX, f32::MIN];
    let tensor = Tensor::from_data(data, vec![1, 5], DataType::F32, TensorLayout::RowMajor)?;

    let decision = router.route(&tensor)?;

    // Should handle extreme values without panic
    assert!(decision.confidence.is_finite());

    Ok(())
}

// ============================================================================
// Performance Tests
// ============================================================================

#[test]
fn test_routing_performance() -> Result<()> {
    use std::time::Instant;

    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![1.0f32; 10000];
    let tensor = Tensor::from_data(data, vec![1, 10000], DataType::F32, TensorLayout::RowMajor)?;

    let start = Instant::now();
    let _decision = router.route(&tensor)?;
    let elapsed = start.elapsed();

    // Routing should be very fast (< 5ms target, as per TASKS.md: <2µs for decisions)
    // But complexity assessment might take longer for large tensors
    assert!(
        elapsed.as_millis() < 5,
        "Routing too slow: {:?}",
        elapsed
    );

    Ok(())
}

#[test]
fn test_repeated_routing_consistency() -> Result<()> {
    let router = HrmRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    // Route the same tensor multiple times
    let decision1 = router.route(&tensor)?;
    let decision2 = router.route(&tensor)?;
    let decision3 = router.route(&tensor)?;

    // Should give consistent results
    assert_eq!(decision1.use_system1, decision2.use_system1);
    assert_eq!(decision1.use_system2, decision2.use_system2);
    assert_eq!(decision2.use_system1, decision3.use_system1);
    assert_eq!(decision2.use_system2, decision3.use_system2);

    Ok(())
}

// ============================================================================
// Strategy Comparison Tests
// ============================================================================

#[test]
fn test_strategy_differences() -> Result<()> {
    let strategies = vec![
        RoutingStrategy::AlwaysSystem1,
        RoutingStrategy::AlwaysSystem2,
        RoutingStrategy::AdaptiveComplexity,
    ];

    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let mut decisions = Vec::new();
    for strategy in strategies {
        let router = HrmRouter::new(strategy);
        let decision = router.route(&tensor)?;
        decisions.push(decision);
    }

    // AlwaysSystem1 should differ from AlwaysSystem2
    assert_ne!(
        decisions[0].use_system1,
        decisions[1].use_system1,
        "AlwaysSystem1 and AlwaysSystem2 should differ"
    );

    Ok(())
}

// ============================================================================
// Concurrent Routing Tests
// ============================================================================

#[test]
fn test_concurrent_routing() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let router = Arc::new(HrmRouter::new(RoutingStrategy::AdaptiveComplexity));
    let mut handles = vec![];

    // Spawn multiple threads routing concurrently
    for i in 0..10 {
        let router_clone = Arc::clone(&router);
        let handle = thread::spawn(move || {
            let data = vec![i as f32; 100];
            let tensor =
                Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)
                    .unwrap();
            router_clone.route(&tensor).unwrap();
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}
