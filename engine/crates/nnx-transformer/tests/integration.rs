//! Integration tests that run against real model files.
//!
//! These tests are skipped if no model file is available.
//! Set the environment variable `NNX_TEST_MODEL` to the path of a GGUF model file.
//!
//! Example:
//!   NNX_TEST_MODEL=path/to/model.gguf cargo test -p nnx-transformer --test integration
//!
//! Recommended test models (small, freely available):
//!   - SmolLM-135M (Q8_0, ~145MB)
//!   - TinyLlama-1.1B-Chat (Q4_K_M, ~700MB)
//!   - Phi-2 (Q4_0, ~1.6GB)

use std::path::PathBuf;

use nnx_core::engine::{InferenceEngine, LoadConfig, TokenBatch};
use nnx_transformer::backend::NnxBackend;
use nnx_transformer::{GenerateConfig, SamplerConfig};

/// Returns the model path from `NNX_TEST_MODEL` if set and the file exists.
fn test_model_path() -> Option<PathBuf> {
    std::env::var("NNX_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Skip the test if no model is available. Call at the start of each test.
macro_rules! require_model {
    () => {
        match test_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping: NNX_TEST_MODEL not set or file not found");
                return;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

#[test]
fn test_load_real_model() {
    let path = require_model!();
    let backend = NnxBackend::new();
    let config = LoadConfig::default();

    let handle = backend.load_model(&path, &config);
    assert!(handle.is_ok(), "Failed to load model: {:?}", handle.err());

    let model = handle.unwrap();
    let info = backend.model_info(model).unwrap();

    // Basic sanity checks — every model must have these.
    assert!(info.num_layers > 0, "Model should have layers");
    assert!(info.hidden_dim > 0, "Model should have hidden dim");
    assert!(info.vocab_size > 0, "Model should have vocab");
    assert!(info.num_heads > 0, "Model should have heads");
    assert!(info.head_dim > 0, "Model should have head dim");

    println!(
        "Loaded: {} -- {}L/{}H/{} -- {:.1}B params",
        info.architecture,
        info.num_layers,
        info.num_heads,
        info.hidden_dim,
        info.num_parameters as f64 / 1e9
    );

    backend.unload_model(model).unwrap();
}

#[test]
fn test_model_info_complete() {
    let path = require_model!();
    let backend = NnxBackend::new();
    let model = backend.load_model(&path, &LoadConfig::default()).unwrap();
    let info = backend.model_info(model).unwrap();

    // All fields should be populated with reasonable values.
    assert!(
        !info.architecture.is_empty(),
        "architecture must not be empty"
    );
    assert!(info.num_layers >= 1, "num_layers={}", info.num_layers);
    assert!(info.hidden_dim >= 64, "hidden_dim={}", info.hidden_dim);
    assert!(info.num_heads >= 1, "num_heads={}", info.num_heads);
    assert!(info.num_kv_heads >= 1, "num_kv_heads={}", info.num_kv_heads);
    assert!(info.head_dim >= 1, "head_dim={}", info.head_dim);
    assert!(
        info.intermediate_dim >= 1,
        "intermediate_dim={}",
        info.intermediate_dim
    );
    assert!(info.vocab_size >= 100, "vocab_size={}", info.vocab_size);
    assert!(
        info.max_context_length >= 128,
        "max_context_length={}",
        info.max_context_length
    );
    assert!(
        info.num_parameters > 0,
        "num_parameters={}",
        info.num_parameters
    );
    assert!(
        info.file_size_bytes > 0,
        "file_size_bytes={}",
        info.file_size_bytes
    );

    // Dimensional consistency: hidden_dim == num_heads * head_dim
    assert_eq!(
        info.hidden_dim,
        info.num_heads * info.head_dim,
        "hidden_dim ({}) != num_heads ({}) * head_dim ({})",
        info.hidden_dim,
        info.num_heads,
        info.head_dim
    );

    backend.unload_model(model).unwrap();
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_forward_produces_valid_logits() {
    let path = require_model!();
    let backend = NnxBackend::new();
    let model = backend.load_model(&path, &LoadConfig::default()).unwrap();
    let request = backend.create_request(model).unwrap();
    let info = backend.model_info(model).unwrap();

    // Forward pass with a simple token sequence.
    let input = TokenBatch::single(vec![1, 2, 3], 0);
    let output = backend.forward(request, &input).unwrap();

    // Logits shape: [1, vocab_size]
    assert_eq!(
        output.logits.shape().dims(),
        &[1, info.vocab_size],
        "logits shape mismatch"
    );

    // Logits should be finite (no NaN or Inf).
    let logits = output.logits.as_f32();
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "Logits contain NaN or Inf"
    );

    // Logits should not all be zero (forward pass is actually doing work).
    assert!(
        logits.iter().any(|v| *v != 0.0),
        "Logits are all zero -- forward pass likely broken"
    );

    // Logits should have reasonable range (not exploded).
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("Logits range: [{:.2}, {:.2}]", min, max);
    assert!(max < 1000.0, "Logits exploded: max={}", max);
    assert!(min > -1000.0, "Logits exploded: min={}", min);

    backend.drop_request(request).unwrap();
    backend.unload_model(model).unwrap();
}

#[test]
fn test_forward_deterministic() {
    let path = require_model!();
    let backend = NnxBackend::new();

    // Run 1
    let model1 = backend.load_model(&path, &LoadConfig::default()).unwrap();
    let request1 = backend.create_request(model1).unwrap();
    let input = TokenBatch::single(vec![1, 2, 3], 0);
    let out1 = backend.forward(request1, &input).unwrap();
    backend.drop_request(request1).unwrap();
    backend.unload_model(model1).unwrap();

    // Run 2 (fresh load — new KV cache)
    let model2 = backend.load_model(&path, &LoadConfig::default()).unwrap();
    let request2 = backend.create_request(model2).unwrap();
    let out2 = backend.forward(request2, &input).unwrap();
    backend.drop_request(request2).unwrap();
    backend.unload_model(model2).unwrap();

    let d1 = out1.logits.as_f32();
    let d2 = out2.logits.as_f32();
    assert_eq!(d1.len(), d2.len(), "logit vector lengths differ");
    for i in 0..d1.len() {
        assert!(
            (d1[i] - d2[i]).abs() < 1e-4,
            "Non-deterministic at logit {}: {} vs {}",
            i,
            d1[i],
            d2[i]
        );
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_tokens_update_cache() {
    let path = require_model!();
    let backend = NnxBackend::new();
    let model = backend.load_model(&path, &LoadConfig::default()).unwrap();
    let request = backend.create_request(model).unwrap();

    // First token
    let input1 = TokenBatch::single(vec![1], 0);
    let out1 = backend.forward(request, &input1).unwrap();
    assert_eq!(
        backend.cache_tokens(request).unwrap(),
        1,
        "cache should hold 1 token after first forward"
    );

    // Second token (incremental decode)
    let input2 = TokenBatch::single(vec![2], 1);
    let out2 = backend.forward(request, &input2).unwrap();
    assert_eq!(
        backend.cache_tokens(request).unwrap(),
        2,
        "cache should hold 2 tokens after second forward"
    );

    // Logits should differ (different context).
    let d1 = out1.logits.as_f32();
    let d2 = out2.logits.as_f32();
    assert!(d1 != d2, "Different tokens should produce different logits");

    backend.drop_request(request).unwrap();
    backend.unload_model(model).unwrap();
}

#[test]
fn test_cache_clear() {
    let path = require_model!();
    let backend = NnxBackend::new();
    let model = backend.load_model(&path, &LoadConfig::default()).unwrap();
    let request = backend.create_request(model).unwrap();

    // Process some tokens.
    backend
        .forward(request, &TokenBatch::single(vec![1, 2, 3], 0))
        .unwrap();
    assert!(
        backend.cache_tokens(request).unwrap() > 0,
        "cache should be non-empty after forward"
    );

    // Clear and verify.
    backend.cache_clear(request).unwrap();
    assert_eq!(
        backend.cache_tokens(request).unwrap(),
        0,
        "cache should be empty after clear"
    );

    backend.drop_request(request).unwrap();
    backend.unload_model(model).unwrap();
}

// ---------------------------------------------------------------------------
// Text generation
// ---------------------------------------------------------------------------

#[test]
fn test_generate_text() {
    let path = require_model!();
    let backend = NnxBackend::new();
    let model = backend.load_model(&path, &LoadConfig::default()).unwrap();

    let config = GenerateConfig {
        max_tokens: 20,
        eos_token_id: 2,
        sampler: SamplerConfig::greedy(),
        seed: 42,
    };

    // generate_text requires an embedded tokenizer — GGUF models usually have one.
    let result = backend.generate_text(model, "Hello", &config);
    match result {
        Ok(text) => {
            println!("Generated: {}", text);
            assert!(!text.is_empty(), "Generated text should not be empty");
        }
        Err(e) => {
            // Missing tokenizer is acceptable — the model may not embed one.
            let msg = format!("{}", e);
            if !msg.contains("tokenizer") {
                panic!("Unexpected generation error: {}", e);
            }
            eprintln!("Skipping text generation: no embedded tokenizer");
        }
    }

    backend.unload_model(model).unwrap();
}

// ---------------------------------------------------------------------------
// Memory budget enforcement
// ---------------------------------------------------------------------------

#[test]
fn test_memory_budget_rejection() {
    let path = require_model!();
    let backend = NnxBackend::new();

    // A budget of 1 byte should always be rejected.
    let config = LoadConfig {
        memory_budget: 1,
        ..LoadConfig::default()
    };
    let result = backend.load_model(&path, &config);
    assert!(result.is_err(), "Loading with 1-byte budget should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("memory budget"),
        "Error should mention memory budget, got: {}",
        err_msg
    );
}

#[test]
fn test_memory_budget_zero_unlimited() {
    let path = require_model!();
    let backend = NnxBackend::new();

    // memory_budget=0 means unlimited — should always succeed.
    let config = LoadConfig {
        memory_budget: 0,
        ..LoadConfig::default()
    };
    let result = backend.load_model(&path, &config);
    assert!(
        result.is_ok(),
        "Loading with budget=0 (unlimited) should succeed: {:?}",
        result.err()
    );
    backend.unload_model(result.unwrap()).unwrap();
}

#[test]
fn test_generation_speed() {
    let path = require_model!();
    let backend = NnxBackend::new();
    let model = backend.load_model(&path, &LoadConfig::default()).unwrap();
    let request = backend.create_request(model).unwrap();

    let config = GenerateConfig {
        max_tokens: 50,
        eos_token_id: 2,
        sampler: SamplerConfig::greedy(),
        seed: 42,
    };

    let start = std::time::Instant::now();
    let result = backend.generate_text(model, "The quick brown fox", &config);
    let elapsed = start.elapsed();

    match result {
        Ok(text) => {
            let chars = text.len();
            println!(
                "Generated {} chars in {:.2}s ({:.1} chars/s)",
                chars,
                elapsed.as_secs_f64(),
                chars as f64 / elapsed.as_secs_f64()
            );
        }
        Err(_) => {
            // No tokenizer — measure raw forward speed instead.
            let info = backend.model_info(model).unwrap();
            backend.cache_clear(request).unwrap();

            let token_count = 50usize;
            let start = std::time::Instant::now();
            for i in 0..token_count {
                let input = TokenBatch::single(vec![(i + 1) as u32], i);
                backend.forward(request, &input).unwrap();
            }
            let elapsed = start.elapsed();
            println!(
                "{} ({}) -- {} tokens in {:.2}s ({:.1} tok/s)",
                info.architecture,
                path.file_name().unwrap_or_default().to_string_lossy(),
                token_count,
                elapsed.as_secs_f64(),
                token_count as f64 / elapsed.as_secs_f64()
            );
        }
    }

    backend.drop_request(request).unwrap();
    backend.unload_model(model).unwrap();
}
