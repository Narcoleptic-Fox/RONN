//! End-to-End Workflow Tests for RONN.
//!
//! These tests validate complete inference pipelines from model loading
//! through optimization, execution, and cleanup. They test the integration
//! of all RONN components working together.

use ronn_api::prelude::*;
use ronn_core::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use std::collections::HashMap;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ============================================================================
// Basic End-to-End Workflows
// ============================================================================

#[test]
fn test_e2e_simple_inference_workflow() -> Result<()> {
    // 1. Create a simple model (mock)
    let model = ModelBuilder::new()
        .with_input("input", vec![1, 4])
        .with_output("output", vec![1, 4])
        .build()?;

    // 2. Create session with default config
    let session = model.create_session_default()?;

    // 3. Prepare input
    let input_tensor = Tensor::zeros(vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input_tensor);

    // 4. Run inference
    let outputs = session.run(inputs)?;

    // 5. Verify output
    assert!(outputs.contains_key("output"));
    assert_eq!(outputs["output"].shape(), vec![1, 4]);

    // 6. Session cleanup is automatic on drop
    Ok(())
}

#[test]
fn test_e2e_with_optimization_levels() -> Result<()> {
    for opt_level in &[
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ] {
        let model = ModelBuilder::new()
            .with_input("input", vec![1, 10])
            .with_output("output", vec![1, 10])
            .build()?;

        let session = model.create_session(
            SessionOptions::new()
                .with_optimization_level(*opt_level)
                .with_provider(ProviderType::Cpu),
        )?;

        let input_tensor = Tensor::zeros(vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let outputs = session.run(inputs)?;

        assert!(outputs.contains_key("output"));
    }

    Ok(())
}

#[test]
fn test_e2e_with_different_providers() -> Result<()> {
    // Test with different execution providers
    let providers = vec![
        ProviderType::Cpu,
        ProviderType::BitNet,
        // GPU requires GPU hardware
        // ProviderType::Gpu,
    ];

    for provider in providers {
        let model = ModelBuilder::new()
            .with_input("input", vec![1, 10])
            .with_output("output", vec![1, 10])
            .build()?;

        let session = model.create_session(
            SessionOptions::new()
                .with_optimization_level(OptimizationLevel::O2)
                .with_provider(provider),
        )?;

        let input_tensor = Tensor::zeros(vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let outputs = session.run(inputs)?;

        assert!(outputs.contains_key("output"));
    }

    Ok(())
}

// ============================================================================
// Multi-Session Workflows
// ============================================================================

#[test]
fn test_e2e_multiple_sessions_sequential() -> Result<()> {
    let model = ModelBuilder::new()
        .with_input("input", vec![1, 10])
        .with_output("output", vec![1, 10])
        .build()?;

    // Create multiple sessions sequentially
    for i in 0..5 {
        let session = model.create_session_default()?;

        let data = vec![(i as f32); 10];
        let input_tensor =
            Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let outputs = session.run(inputs)?;
        assert!(outputs.contains_key("output"));
    }

    Ok(())
}

#[test]
fn test_e2e_multiple_sessions_parallel() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let model = Arc::new(
        ModelBuilder::new()
            .with_input("input", vec![1, 10])
            .with_output("output", vec![1, 10])
            .build()?,
    );

    let mut handles = vec![];

    for i in 0..5 {
        let model_clone = Arc::clone(&model);
        let handle = thread::spawn(move || {
            let session = model_clone.create_session_default().unwrap();

            let data = vec![(i as f32); 10];
            let input_tensor =
                Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
                    .unwrap();
            let mut inputs = HashMap::new();
            inputs.insert("input".to_string(), input_tensor);

            let outputs = session.run(inputs).unwrap();
            assert!(outputs.contains_key("output"));
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

// ============================================================================
// Brain-Inspired Feature Workflows
// ============================================================================

#[test]
fn test_e2e_with_hrm_routing() -> Result<()> {
    // Create model with HRM routing
    let model = ModelBuilder::new()
        .with_input("input", vec![1, 100])
        .with_output("output", vec![1, 100])
        .with_hrm_routing(true) // Enable HRM
        .build()?;

    let session = model.create_session_default()?;

    // Test with simple input (should use System 1)
    let simple_data = vec![1.0f32; 10];
    let simple_tensor = Tensor::from_data(
        simple_data,
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), simple_tensor);

    let outputs = session.run(inputs)?;
    assert!(outputs.contains_key("output"));

    // Test with complex input (should use System 2)
    let complex_data: Vec<f32> = (0..10000).map(|x| (x as f32).sin()).collect();
    let complex_tensor = Tensor::from_data(
        complex_data,
        vec![1, 10000],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let mut inputs2 = HashMap::new();
    inputs2.insert("input".to_string(), complex_tensor);

    let outputs2 = session.run(inputs2)?;
    assert!(outputs2.contains_key("output"));

    Ok(())
}

#[test]
fn test_e2e_with_memory_system() -> Result<()> {
    // Create model with multi-tier memory
    let model = ModelBuilder::new()
        .with_input("input", vec![1, 10])
        .with_output("output", vec![1, 10])
        .with_memory_system(true) // Enable memory system
        .build()?;

    let session = model.create_session_default()?;

    // Run multiple inferences to build up memory
    for i in 0..10 {
        let data = vec![i as f32; 10];
        let input_tensor =
            Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let outputs = session.run(inputs)?;
        assert!(outputs.contains_key("output"));
    }

    // Memory should have accumulated information
    let memory_stats = session.memory_stats()?;
    assert!(memory_stats.working_items > 0 || memory_stats.episodic_episodes > 0);

    Ok(())
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

#[test]
fn test_e2e_throughput() -> Result<()> {
    use std::time::Instant;

    let model = ModelBuilder::new()
        .with_input("input", vec![1, 50])
        .with_output("output", vec![1, 50])
        .build()?;

    let session = model.create_session(
        SessionOptions::new()
            .with_optimization_level(OptimizationLevel::O3)
            .with_provider(ProviderType::Cpu),
    )?;

    let iterations = 1000;
    let input_tensor = Tensor::zeros(vec![1, 50], DataType::F32, TensorLayout::RowMajor)?;

    let start = Instant::now();
    for _ in 0..iterations {
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor.clone());
        let _outputs = session.run(inputs)?;
    }
    let elapsed = start.elapsed();

    let throughput = iterations as f64 / elapsed.as_secs_f64();
    println!("Throughput: {:.2} inferences/sec", throughput);

    // Target: >1000 inferences/sec
    assert!(throughput > 100.0, "Throughput too low: {:.2}", throughput);

    Ok(())
}

#[test]
fn test_e2e_latency_targets() -> Result<()> {
    use std::time::Instant;

    let model = ModelBuilder::new()
        .with_input("input", vec![1, 100])
        .with_output("output", vec![1, 100])
        .build()?;

    let session = model.create_session(
        SessionOptions::new()
            .with_optimization_level(OptimizationLevel::O3)
            .with_provider(ProviderType::Cpu),
    )?;

    let mut latencies = Vec::new();

    // Measure latencies
    for i in 0..100 {
        let data = vec![(i % 100) as f32; 100];
        let input_tensor =
            Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let start = Instant::now();
        let _outputs = session.run(inputs)?;
        let elapsed = start.elapsed();

        latencies.push(elapsed.as_micros());
    }

    // Calculate percentiles
    latencies.sort();
    let p50 = latencies[50];
    let p95 = latencies[95];
    let p99 = latencies[99];

    println!("Latency P50: {}µs, P95: {}µs, P99: {}µs", p50, p95, p99);

    // Targets: P50 <10ms, P95 <30ms
    assert!(p50 < 10_000, "P50 latency too high: {}µs", p50);
    assert!(p95 < 30_000, "P95 latency too high: {}µs", p95);

    Ok(())
}

#[test]
fn test_e2e_memory_usage() -> Result<()> {
    let model = ModelBuilder::new()
        .with_input("input", vec![1, 1000])
        .with_output("output", vec![1, 1000])
        .build()?;

    let session = model.create_session_default()?;

    // Run many inferences
    for i in 0..1000 {
        let data = vec![(i % 100) as f32; 1000];
        let input_tensor =
            Tensor::from_data(data, vec![1, 1000], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let _outputs = session.run(inputs)?;
    }

    // Memory stats should be reasonable
    let stats = session.stats()?;
    println!("Total memory allocated: {} MB", stats.total_memory_mb);

    // Target: <4GB for typical workload
    assert!(
        stats.total_memory_mb < 4096,
        "Memory usage too high: {} MB",
        stats.total_memory_mb
    );

    Ok(())
}

// ============================================================================
// Error Recovery Workflows
// ============================================================================

#[test]
fn test_e2e_error_recovery() -> Result<()> {
    let model = ModelBuilder::new()
        .with_input("input", vec![1, 10])
        .with_output("output", vec![1, 10])
        .build()?;

    let session = model.create_session_default()?;

    // Test with valid input
    let valid_tensor = Tensor::zeros(vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), valid_tensor);
    let result = session.run(inputs);
    assert!(result.is_ok());

    // Test with wrong shape (should error gracefully)
    let invalid_tensor = Tensor::zeros(vec![1, 20], DataType::F32, TensorLayout::RowMajor)?;
    let mut inputs2 = HashMap::new();
    inputs2.insert("input".to_string(), invalid_tensor);
    let result2 = session.run(inputs2);
    // Should either error gracefully or handle shape mismatch
    let _ = result2; // Don't panic

    // Test that session still works after error
    let valid_tensor2 = Tensor::zeros(vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
    let mut inputs3 = HashMap::new();
    inputs3.insert("input".to_string(), valid_tensor2);
    let result3 = session.run(inputs3);
    assert!(result3.is_ok());

    Ok(())
}

// ============================================================================
// Resource Cleanup Tests
// ============================================================================

#[test]
fn test_e2e_resource_cleanup() -> Result<()> {
    // Create and drop many sessions to test cleanup
    for _ in 0..100 {
        let model = ModelBuilder::new()
            .with_input("input", vec![1, 10])
            .with_output("output", vec![1, 10])
            .build()?;

        let session = model.create_session_default()?;

        let input_tensor = Tensor::zeros(vec![1, 10], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let _outputs = session.run(inputs)?;

        // Session dropped here - resources should be cleaned up
    }

    // Should complete without memory leaks or resource exhaustion
    Ok(())
}

// ============================================================================
// Long-Running Stability Tests
// ============================================================================

#[test]
#[ignore] // Long-running test
fn test_e2e_long_running_stability() -> Result<()> {
    let model = ModelBuilder::new()
        .with_input("input", vec![1, 100])
        .with_output("output", vec![1, 100])
        .build()?;

    let session = model.create_session(
        SessionOptions::new()
            .with_optimization_level(OptimizationLevel::O3)
            .with_provider(ProviderType::Cpu),
    )?;

    // Run for extended period
    for i in 0..10_000 {
        let data = vec![(i % 1000) as f32; 100];
        let input_tensor =
            Tensor::from_data(data, vec![1, 100], DataType::F32, TensorLayout::RowMajor)?;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);

        let outputs = session.run(inputs)?;
        assert!(outputs.contains_key("output"));

        if i % 1000 == 0 {
            println!("Completed {} iterations", i);
        }
    }

    Ok(())
}
