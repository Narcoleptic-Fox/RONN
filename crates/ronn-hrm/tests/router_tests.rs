//! Tests for HRM routing decisions.

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::router::{HRMRouter, RoutingStrategy};
use ronn_hrm::executor::ExecutionPath;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn test_always_system1_strategy() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem1);

    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = router.assess_complexity(&tensor)?;
    let decision = router.route(&metrics)?;

    assert_eq!(decision.path, ExecutionPath::System1);
    assert!(decision.confidence > 0.0);

    Ok(())
}

#[test]
fn test_always_system2_strategy() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem2);

    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = router.assess_complexity(&tensor)?;
    let decision = router.route(&metrics)?;

    assert_eq!(decision.path, ExecutionPath::System2);
    assert!(decision.confidence > 0.0);

    Ok(())
}

#[test]
fn test_adaptive_complexity_low() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);

    // Small, simple tensor should route to System1
    let data = vec![1.0f32; 10];
    let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = router.assess_complexity(&tensor)?;
    let decision = router.route(&metrics)?;

    assert_eq!(decision.path, ExecutionPath::System1);
    assert!(decision.confidence > 0.5);

    Ok(())
}

#[test]
fn test_adaptive_hybrid_strategy() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AdaptiveHybrid);

    // Medium complexity might trigger hybrid
    let data: Vec<f32> = (0..500).map(|x| x as f32).collect();
    let tensor = Tensor::from_data(data, vec![1, 500], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = router.assess_complexity(&tensor)?;
    let decision = router.route(&metrics)?;

    // Should make a valid decision
    assert!(matches!(
        decision.path,
        ExecutionPath::System1 | ExecutionPath::System2 | ExecutionPath::Hybrid
    ));

    Ok(())
}

#[test]
fn test_statistics_tracking() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);

    // Initial stats should be zero
    let stats = router.stats();
    assert_eq!(stats.total_decisions, 0);
    assert_eq!(stats.system1_routes, 0);
    assert_eq!(stats.system2_routes, 0);

    // Route a simple tensor
    let data = vec![1.0f32; 10];
    let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
    let metrics = router.assess_complexity(&tensor)?;
    router.route(&metrics)?;

    // Stats should increment
    let stats = router.stats();
    assert_eq!(stats.total_decisions, 1);
    assert!(stats.system1_routes > 0 || stats.system2_routes > 0);

    Ok(())
}

#[test]
fn test_decision_confidence_bounds() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;

    let metrics = router.assess_complexity(&tensor)?;
    let decision = router.route(&metrics)?;

    // Confidence must be between 0 and 1
    assert!(decision.confidence >= 0.0);
    assert!(decision.confidence <= 1.0);

    Ok(())
}

#[test]
fn test_routing_performance() -> Result<()> {
    use std::time::Instant;

    let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);

    let data = vec![1.0f32; 10000];
    let tensor = Tensor::from_data(data, vec![1, 10000], DataType::F32, TensorLayout::RowMajor)?;

    let start = Instant::now();
    let metrics = router.assess_complexity(&tensor)?;
    let _decision = router.route(&metrics)?;
    let elapsed = start.elapsed();

    // Routing should be fast (< 50ms)
    assert!(elapsed.as_millis() < 50, "Routing too slow: {:?}", elapsed);

    Ok(())
}
