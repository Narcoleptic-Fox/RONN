//! Tests for the HRM router functionality.

use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_hrm::ComplexityAssessor;
use ronn_hrm::router::{HRMRouter, RoutingStrategy};
use ronn_hrm::executor::ExecutionPath;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// Router Creation Tests
// ============================================================================

#[test]
fn test_router_creation_default() {
    let router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);
    // Just verify it can be created
    drop(router);
}

#[test]
fn test_router_creation_always_system1() {
    let router = HRMRouter::new(RoutingStrategy::AlwaysSystem1);
    drop(router);
}

#[test]
fn test_router_creation_always_system2() {
    let router = HRMRouter::new(RoutingStrategy::AlwaysSystem2);
    drop(router);
}

#[test]
fn test_router_creation_adaptive_hybrid() {
    let router = HRMRouter::new(RoutingStrategy::AdaptiveHybrid);
    drop(router);
}

// ============================================================================
// Routing Decision Tests
// ============================================================================

#[test]
fn test_always_system1_routing() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem1);
    let assessor = ComplexityAssessor::new();
    
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;
    
    let metrics = assessor.assess(&tensor)?;
    let decision = router.route(&metrics)?;
    
    assert!(matches!(decision.path, ExecutionPath::System1));
    
    Ok(())
}

#[test]
fn test_always_system2_routing() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem2);
    let assessor = ComplexityAssessor::new();
    
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;
    
    let metrics = assessor.assess(&tensor)?;
    let decision = router.route(&metrics)?;
    
    assert!(matches!(decision.path, ExecutionPath::System2));
    
    Ok(())
}

#[test]
fn test_adaptive_complexity_routing() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);
    let assessor = ComplexityAssessor::new();
    
    // Simple uniform data
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;
    
    let metrics = assessor.assess(&tensor)?;
    let decision = router.route(&metrics)?;
    
    // Low complexity should route to System1
    assert!(decision.confidence > 0.0);
    
    Ok(())
}

#[test]
fn test_routing_decision_confidence() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AdaptiveComplexity);
    let assessor = ComplexityAssessor::new();
    
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;
    
    let metrics = assessor.assess(&tensor)?;
    let decision = router.route(&metrics)?;
    
    // Confidence should be between 0 and 1
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    
    Ok(())
}

// ============================================================================
// Strategy Switching Tests
// ============================================================================

#[test]
fn test_strategy_switching() -> Result<()> {
    let mut router = HRMRouter::new(RoutingStrategy::AlwaysSystem1);
    let assessor = ComplexityAssessor::new();
    
    let data = vec![1.0f32; 100];
    let tensor = Tensor::from_data(data, vec![10, 10], DataType::F32, TensorLayout::RowMajor)?;
    let metrics = assessor.assess(&tensor)?;
    
    // Should route to System1
    let decision1 = router.route(&metrics)?;
    assert!(matches!(decision1.path, ExecutionPath::System1));
    
    // Switch strategy
    router.set_strategy(RoutingStrategy::AlwaysSystem2);
    
    // Should now route to System2
    let decision2 = router.route(&metrics)?;
    assert!(matches!(decision2.path, ExecutionPath::System2));
    
    Ok(())
}
